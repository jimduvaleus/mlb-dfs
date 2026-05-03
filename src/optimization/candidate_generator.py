"""Fast ownership-weighted stacked lineup generator for GPP candidate pool.

Generates M valid DraftKings Classic lineups via rejection sampling with a
heavy bias toward correlated multi-team stacks required for GPP portfolio
construction.
"""
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.contest import (
    _build_pos_pools,
    _is_valid_field_lineup,
    _player_meta_from_df,
)
from src.optimization.lineup import Lineup, ROSTER_REQUIREMENTS, SALARY_CAP

logger = logging.getLogger(__name__)

_BATTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")
_MAX_HITTERS_PER_TEAM = 5


class CandidateGenerator:
    """Generate M valid DK Classic stacked lineups via rejection sampling.

    Each returned lineup satisfies the GPP stacking requirement:
      - top team ≥ 5 hitters, OR
      - top two teams combined ≥ 6 hitters with second team contributing ≥ 2

    This covers patterns: 5, 5-2, 5-3, 4-4, 4-3, 4-2, 4-2-2, 3-3, 3-3-2
    (where numbers are hitter counts per team, fill may extend the pattern).
    """

    # (primary_stack_size, secondary_stack_size); 0 = no secondary stack.
    # Patterns are sampled uniformly; success rate naturally weights each.
    STACK_PATTERNS: list[tuple[int, int]] = [
        (5, 0),  # 5
        (5, 2),  # 5-2
        (5, 3),  # 5-3
        (4, 4),  # 4-4
        (4, 3),  # 4-3
        (4, 2),  # 4-2 (fill may produce 4-2-2)
        (3, 3),  # 3-3 (fill may produce 3-3-2)
    ]

    def __init__(
        self,
        players_df: pd.DataFrame,
        ownership_vec: np.ndarray,
        rng_seed: Optional[int] = None,
        salary_floor: Optional[float] = None,
    ) -> None:
        self._players_df = players_df
        self._ownership_vec = ownership_vec
        self._rng_seed = rng_seed
        self._salary_floor = salary_floor

        self._pmeta = _player_meta_from_df(players_df)
        self._pos_pools = _build_pos_pools(players_df, ownership_vec)

        # Per-team batter lists and ownership-sum weights for stack team selection.
        self._team_batters: dict[str, list[int]] = {}
        self._team_weights: dict[str, float] = {}
        for pos_name in _BATTER_POSITIONS:
            p_ids, p_w = self._pos_pools.get(pos_name, ([], np.array([])))
            for pid, w in zip(p_ids, p_w):
                t = self._pmeta[pid]["team"]
                self._team_batters.setdefault(t, []).append(pid)
                self._team_weights[t] = self._team_weights.get(t, 0.0) + float(w)

        # Flat ownership lookup across all batter positions.
        self._pid_to_ow: dict[int, float] = {}
        for pos_name in _BATTER_POSITIONS:
            p_ids, p_w = self._pos_pools.get(pos_name, ([], np.array([])))
            for pid, w in zip(p_ids, p_w):
                self._pid_to_ow[pid] = w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_candidates: int = 10_000,
        max_attempts_multiplier: int = 50,
        progress_cb: Optional[Callable[[int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[Lineup]:
        """Generate up to n_candidates valid stacked lineups.

        Parameters
        ----------
        n_candidates : target pool size
        max_attempts_multiplier : total attempts = n_candidates * this value
        progress_cb : optional callable(n_generated_so_far), called every 500 lineups

        Returns fewer than n_candidates (with a warning) if attempts are exhausted.
        """
        games = {
            self._pmeta[pid]["game"]
            for pid in self._pmeta
            if self._pmeta[pid]["game"]
        }
        if len(games) < 2:
            raise RuntimeError(
                f"CandidateGenerator requires at least 2 games; found {len(games)}. "
                "Cannot satisfy the DraftKings min_games=2 constraint."
            )

        rng = np.random.default_rng(self._rng_seed)
        results: list[Lineup] = []
        max_attempts = n_candidates * max_attempts_multiplier
        total_attempts = 0
        n_patterns = len(self.STACK_PATTERNS)

        while len(results) < n_candidates and total_attempts < max_attempts:
            total_attempts += 1
            idx = int(rng.integers(n_patterns))
            primary_size, secondary_size = self.STACK_PATTERNS[idx]
            ids = self._sample_one(rng, primary_size, secondary_size)
            if ids is None:
                continue
            if not self._check_stack(ids):
                continue
            if self._salary_floor is not None:
                total_sal = sum(self._pmeta[pid]["salary"] for pid in ids)
                if total_sal < self._salary_floor:
                    continue
            results.append(Lineup(player_ids=ids))
            if len(results) % 500 == 0:
                if progress_cb is not None:
                    progress_cb(len(results))
                if stop_check is not None and stop_check():
                    logger.info("CandidateGenerator: stop requested after %d candidates.", len(results))
                    break

        if len(results) < n_candidates * 0.5:
            logger.warning(
                "CandidateGenerator: only produced %d / %d candidates after %d attempts. "
                "Consider a larger max_attempts_multiplier or a less restrictive salary_floor.",
                len(results), n_candidates, total_attempts,
            )
        elif len(results) < n_candidates:
            logger.info(
                "CandidateGenerator: produced %d / %d candidates after %d attempts",
                len(results), n_candidates, total_attempts,
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_stack(self, ids: list[int]) -> bool:
        """Return True iff lineup satisfies the GPP stacking requirement."""
        team_counts: dict[str, int] = {}
        for pid in ids:
            if self._pmeta[pid]["position"] != "P":
                t = self._pmeta[pid]["team"]
                team_counts[t] = team_counts.get(t, 0) + 1
        if not team_counts:
            return False
        sorted_counts = sorted(team_counts.values(), reverse=True)
        top = sorted_counts[0]
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0
        # top ≥ 5  OR  (combined top-2 ≥ 6 with second team ≥ 2)
        return top >= 5 or (top + second >= 6 and second >= 2)

    def _sample_one(
        self,
        rng: np.random.Generator,
        primary_size: int,
        secondary_size: int,
    ) -> Optional[list[int]]:
        """Attempt to build one stacked lineup. Returns None on any failure."""
        pmeta = self._pmeta
        pos_pools = self._pos_pools
        pid_to_ow = self._pid_to_ow

        # --- Primary team (ownership-weighted) ---
        teams = list(self._team_weights.keys())
        if not teams:
            return None
        tw = np.array([self._team_weights[t] for t in teams], dtype=np.float64)
        tw /= tw.sum()
        primary_team = teams[int(rng.choice(len(teams), p=tw))]

        primary_pool = [
            pid for pid in self._team_batters.get(primary_team, [])
            if pid in pid_to_ow
        ]
        if len(primary_pool) < primary_size:
            return None

        sw = np.array([pid_to_ow[pid] for pid in primary_pool], dtype=np.float64)
        sw /= sw.sum()
        chosen_idx = rng.choice(len(primary_pool), size=primary_size, replace=False, p=sw)
        primary_ids = {int(primary_pool[i]) for i in chosen_idx}
        used: set[int] = set(primary_ids)

        # --- Secondary team (prefer different game for min_games safety) ---
        secondary_ids: set[int] = set()
        if secondary_size > 0:
            primary_games = {pmeta[pid]["game"] for pid in primary_ids}
            other_teams = [
                t for t in teams
                if t != primary_team and t in self._team_batters
            ]
            if not other_teams:
                return None

            sec_weights: list[float] = []
            for t in other_teams:
                # Determine this team's game from its first known batter.
                team_game = next(
                    (pmeta[pid]["game"] for pid in self._team_batters[t] if pid in pmeta),
                    "",
                )
                base = self._team_weights.get(t, 0.0)
                # 2× boost for teams from a different game (helps min_games).
                sec_weights.append(base * 2.0 if team_game not in primary_games else base)

            sec_w_arr = np.array(sec_weights, dtype=np.float64)
            if sec_w_arr.sum() == 0:
                return None
            sec_w_arr /= sec_w_arr.sum()
            secondary_team = other_teams[int(rng.choice(len(other_teams), p=sec_w_arr))]

            sec_pool = [
                pid for pid in self._team_batters.get(secondary_team, [])
                if pid in pid_to_ow and pid not in used
            ]
            if len(sec_pool) < secondary_size:
                return None

            sec_sw = np.array([pid_to_ow[pid] for pid in sec_pool], dtype=np.float64)
            sec_sw /= sec_sw.sum()
            sec_idx = rng.choice(len(sec_pool), size=secondary_size, replace=False, p=sec_sw)
            secondary_ids = {int(sec_pool[i]) for i in sec_idx}
            used |= secondary_ids

        stack_ids = primary_ids | secondary_ids

        # --- Pitchers (avoid opposing any stacked batter's opponent) ---
        excluded_opps = {pmeta[pid]["opponent"] for pid in stack_ids}
        p_ids, p_w = pos_pools.get("P", ([], np.array([])))
        pitcher_pool = [
            pid for pid in p_ids
            if pmeta[pid]["team"] not in excluded_opps and pid not in used
        ]
        if len(pitcher_pool) < 2:
            return None
        p_w_dict = dict(zip(p_ids, p_w))
        pit_w = np.array(
            [p_w_dict.get(pid, 1.0 / len(pitcher_pool)) for pid in pitcher_pool],
            dtype=np.float64,
        )
        pit_w /= pit_w.sum()
        pit_idx = rng.choice(len(pitcher_pool), size=2, replace=False, p=pit_w)
        pitcher_ids = [int(pitcher_pool[i]) for i in pit_idx]
        # DK requires two pitchers from different teams.
        if pmeta[pitcher_ids[0]]["team"] == pmeta[pitcher_ids[1]]["team"]:
            return None
        used |= set(pitcher_ids)

        # Teams that the chosen pitchers oppose — fill batters must not come from these.
        pitcher_opp_teams = {
            pmeta[pid]["opponent"]
            for pid in pitcher_ids
            if pmeta[pid]["opponent"]
        }

        # --- Fill remaining hitter positions ---
        # Track how many hitters each team has so we can enforce the cap during fill.
        hitter_team_count: dict[str, int] = {}
        for pid in stack_ids:
            t = pmeta[pid]["team"]
            hitter_team_count[t] = hitter_team_count.get(t, 0) + 1

        remaining: list[int] = []
        for pos_name in _BATTER_POSITIONS:
            already_covered = sum(
                1 for pid in stack_ids if pmeta[pid]["position"] == pos_name
            )
            need = ROSTER_REQUIREMENTS.get(pos_name, 0) - already_covered
            if need <= 0:
                continue
            p_ids_pos, p_w_pos = pos_pools.get(pos_name, ([], np.array([])))
            cands = [
                pid for pid in p_ids_pos
                if pid not in used
                and hitter_team_count.get(pmeta[pid]["team"], 0) < _MAX_HITTERS_PER_TEAM
                and pmeta[pid]["team"] not in pitcher_opp_teams
            ]
            if len(cands) < need:
                return None
            cw_dict = dict(zip(p_ids_pos, p_w_pos))
            cw = np.array(
                [cw_dict.get(pid, 1.0 / len(cands)) for pid in cands],
                dtype=np.float64,
            )
            cw /= cw.sum()
            chosen = [
                int(cands[i])
                for i in rng.choice(len(cands), size=need, replace=False, p=cw)
            ]
            for pid in chosen:
                t = pmeta[pid]["team"]
                hitter_team_count[t] = hitter_team_count.get(t, 0) + 1
            remaining.extend(chosen)
            used |= set(chosen)

        all_ids = pitcher_ids + list(stack_ids) + remaining
        if len(set(all_ids)) != 10 or len(all_ids) != 10:
            return None
        if not _is_valid_field_lineup(all_ids, pmeta):
            return None
        return all_ids
