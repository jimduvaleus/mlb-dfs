"""Fast ownership-weighted stacked lineup generator for GPP candidate pool.

Generates M valid DraftKings Classic lineups via a two-phase approach:
  Phase 1 — round-robin: each team gets an equal quota of primary-stack
    lineups; within each team's quota, secondary teams are cycled in
    order to ensure pair coverage without explicit pair-cap tracking.
  Phase 2 — backfill: any remaining slots are filled with weighted
    sampling, skipping teams that already hit quota.

Stack-size distribution targets (configurable via GROUP_FRACTIONS):
  5-hitter primary (5, 5-2, 5-3): 50 %
  4-hitter primary (4-4, 4-3, 4-2): 40 %
  3-hitter primary (3-3):           10 %

Diversity parameters:
  team_weight_power : float, default 0.5
      Exponent applied to team ownership-sum before sampling (backfill
      phase and secondary team selection). 0.5 = sqrt; 0.0 = uniform.
  fill_weight_power : float, default 0.0
      Exponent applied to ownership weights for fill-batter slot sampling.
      0.0 = uniform within salary-feasible pool.
"""
import logging
import math
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


def _gumbel_choice(rng: np.random.Generator, n: int, size: int, weights: np.ndarray) -> np.ndarray:
    """Weighted sampling without replacement via Efraimidis-Spirakis (O(n))."""
    log_u = np.log(rng.random(n))
    keys = np.where(weights > 0, log_u / weights, -np.inf)
    return np.argpartition(keys, -size)[-size:]


class CandidateGenerator:
    """Generate M valid DK Classic stacked lineups.

    Stack patterns by primary hitter count:
      5-group: (5,0), (5,2), (5,3)
      4-group: (4,4), (4,3), (4,2)
      3-group: (3,3)

    Each lineup satisfies the GPP stacking requirement:
      top team ≥ 5 hitters  OR  top-2 combined ≥ 6 with second ≥ 2.
    """

    STACK_GROUPS: dict[int, list[tuple[int, int]]] = {
        5: [(5, 0), (5, 2), (5, 3)],
        4: [(4, 4), (4, 3), (4, 2)],
        3: [(3, 3)],
    }
    # Target fraction of generated lineups per stack-size group.
    GROUP_FRACTIONS: dict[int, float] = {5: 0.50, 4: 0.40, 3: 0.10}

    @property
    def STACK_PATTERNS(self) -> list[tuple[int, int]]:
        """Flat list of all patterns (backward-compat property)."""
        return [p for pats in self.STACK_GROUPS.values() for p in pats]

    def __init__(
        self,
        players_df: pd.DataFrame,
        ownership_vec: np.ndarray,
        rng_seed: Optional[int] = None,
        salary_floor: Optional[float] = None,
        team_weight_power: float = 0.5,
        fill_weight_power: float = 0.0,
    ) -> None:
        self._players_df = players_df
        self._ownership_vec = ownership_vec
        self._rng_seed = rng_seed
        self._salary_floor = salary_floor
        self._team_weight_power = float(team_weight_power)
        self._fill_weight_power = float(fill_weight_power)

        self._pmeta = _player_meta_from_df(players_df)
        self._pos_pools = _build_pos_pools(players_df, ownership_vec)

        # Per-team batter lists and raw ownership-sum weights.
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

        # Flat per-player lookups.
        self._pid_team: dict[int, str] = {
            pid: meta["team"] for pid, meta in self._pmeta.items()
        }
        self._pid_salary: dict[int, int] = {
            pid: int(meta["salary"]) for pid, meta in self._pmeta.items()
        }
        self._pid_opponent: dict[int, str] = {
            pid: meta.get("opponent", "") for pid, meta in self._pmeta.items()
        }

        # Salary look-ahead: sorted batter salaries descending for fill estimation.
        batter_salaries = [self._pid_salary[pid] for pid in self._pid_to_ow]
        self._min_batter_salary: float = float(min(batter_salaries)) if batter_salaries else 0.0
        self._top_batter_salaries: list[int] = sorted(batter_salaries, reverse=True)

        # Per-team top-k salary sum for secondary feasibility filter (k = 1..5).
        self._team_top_k_salary: dict[str, dict[int, int]] = {}
        for t, pids in self._team_batters.items():
            sals = sorted(
                (self._pid_salary[p] for p in pids if p in self._pid_to_ow),
                reverse=True,
            )
            self._team_top_k_salary[t] = {k: sum(sals[:k]) for k in range(1, len(sals) + 1)}

        # Precomputed team→game.
        self._team_game: dict[str, str] = {}
        for t, pids in self._team_batters.items():
            for pid in pids:
                g = self._pmeta[pid].get("game", "")
                if g:
                    self._team_game[t] = g
                    break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_candidates: int = 10_000,
        max_attempts_multiplier: int = 50,
        team_pair_cap: Optional[int] = None,  # kept for API compat, ignored
        progress_cb: Optional[Callable[[int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        floor_relief: int = 2500,
    ) -> list[Lineup]:
        """Generate up to n_candidates valid stacked lineups.

        Phase 1: round-robin through teams, each getting quota = n_candidates//n_teams.
          Within each team's turn, secondary teams are cycled in shuffled order.
          Dynamic floor relief: if a team's success rate drops below 2% over a 200-attempt
          window, the effective salary floor is lowered by $500 per window (down to a
          minimum of salary_floor - floor_relief).  Each team's floor resets to the
          configured value at the start of its slot.
        Phase 2: backfill remaining slots via weighted sampling, skipping teams at quota.
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

        n_teams = len(self._team_weights)
        quota = max(1, n_candidates // n_teams)
        # Hard cap per team for distribution enforcement: quota +33%.
        # Applied in phase 2 so no single team monopolises the backfill.
        hard_cap = math.ceil(quota * 1.33)

        # Phase 1 uses only 4/5-group patterns (the ones that count toward quota).
        # 3-group lineups are generated in phase 2 only, preventing small-roster teams
        # from flooding results with 3-group during their phase 1 slot.
        _main_keys = [5, 4]
        _main_sum = sum(self.GROUP_FRACTIONS[k] for k in _main_keys)
        _main_probs = np.array(
            [self.GROUP_FRACTIONS[k] / _main_sum for k in _main_keys], dtype=np.float64
        )

        # Phase 2 uses all groups (including 3-group).
        _all_keys = [5, 4, 3]
        _all_probs = np.array(
            [self.GROUP_FRACTIONS[k] for k in _all_keys], dtype=np.float64
        )

        rng = np.random.default_rng(self._rng_seed)
        results: list[Lineup] = []
        # team_counts tracks 4-group and 5-group lineups only; 3-group lineups
        # are added to results but do not count toward any team's quota.
        team_counts: dict[str, int] = {}
        total_attempts = 0
        max_attempts = n_candidates * max_attempts_multiplier

        def _credit(pt: str, ps: int, ss: int, st: str) -> None:
            """Credit quota for 4/5-group lineups; both teams for (4,4)."""
            if ps < 4:
                return
            team_counts[pt] = team_counts.get(pt, 0) + 1
            if ps == 4 and ss == 4 and st:
                team_counts[st] = team_counts.get(st, 0) + 1

        def _maybe_report(n: int) -> bool:
            if n % 500 == 0:
                if progress_cb is not None:
                    progress_cb(n)
                if stop_check is not None and stop_check():
                    logger.info(
                        "CandidateGenerator: stop requested after %d candidates.", n
                    )
                    return True
            return False

        # Dynamic floor relief constants.
        _floor_check_interval = 200   # attempts per window
        _floor_min_rate = 0.02        # trigger reduction below this success rate
        _floor_step = 500             # dollars per reduction step

        # --- Phase 1: round-robin ---
        teams_shuffled = list(self._team_weights)
        rng.shuffle(teams_shuffled)

        for primary_team in teams_shuffled:
            if len(results) >= n_candidates or total_attempts >= max_attempts:
                break

            other_teams = [t for t in teams_shuffled if t != primary_team]
            rng.shuffle(other_teams)
            sec_cursor = 0
            team_attempt = 0
            team_max_attempts = quota * max_attempts_multiplier

            # Per-team dynamic floor: starts at configured floor, can decrease.
            base_floor = self._salary_floor or 0.0
            effective_floor = base_floor
            floor_min_val = max(0.0, base_floor - float(floor_relief))
            team_success_count = 0
            last_check_success = 0

            while (
                team_counts.get(primary_team, 0) < quota
                and team_attempt < team_max_attempts
                and total_attempts < max_attempts
                and len(results) < n_candidates
            ):
                team_attempt += 1
                total_attempts += 1

                # Dynamic floor: check every window; reduce if success rate is low.
                if (
                    team_attempt % _floor_check_interval == 0
                    and self._salary_floor is not None
                    and effective_floor > floor_min_val
                ):
                    window_rate = (team_success_count - last_check_success) / _floor_check_interval
                    if window_rate < _floor_min_rate:
                        effective_floor = max(floor_min_val, effective_floor - _floor_step)
                        logger.debug(
                            "CandidateGenerator: %s floor reduced to %.0f "
                            "(window success rate %.1f%%)",
                            primary_team, effective_floor, window_rate * 100,
                        )
                    last_check_success = team_success_count

                g = _main_keys[int(rng.choice(2, p=_main_probs))]
                pats = self.STACK_GROUPS[g]
                primary_size, secondary_size = pats[int(rng.integers(len(pats)))]

                hint = other_teams[sec_cursor % len(other_teams)] if other_teams else ''
                ids, pt, st = self._sample_one(
                    rng, primary_team, primary_size, secondary_size,
                    hint_secondary=hint,
                    effective_floor=effective_floor,
                )
                sec_cursor += 1

                if ids is None or not self._check_stack(ids):
                    continue

                team_success_count += 1
                _credit(primary_team, primary_size, secondary_size, st)
                results.append(Lineup(player_ids=ids))
                if _maybe_report(len(results)):
                    return results

        # --- Phase 2: backfill ---
        # Teams below hard_cap remain eligible; those at hard_cap are skipped.
        while len(results) < n_candidates and total_attempts < max_attempts:
            total_attempts += 1

            teams = list(self._team_weights)
            raw_tw = np.array(
                [
                    (self._team_weights[t] ** self._team_weight_power
                     if team_counts.get(t, 0) < hard_cap else 0.0)
                    for t in teams
                ],
                dtype=np.float64,
            )
            tw_sum = raw_tw.sum()
            if tw_sum == 0:
                break
            tw = raw_tw / tw_sum
            primary_team = teams[int(np.searchsorted(np.cumsum(tw), rng.random()))]

            g = _all_keys[int(rng.choice(3, p=_all_probs))]
            pats = self.STACK_GROUPS[g]
            primary_size, secondary_size = pats[int(rng.integers(len(pats)))]

            ids, pt, st = self._sample_one(rng, primary_team, primary_size, secondary_size)
            if ids is None or not self._check_stack(ids):
                continue

            _credit(primary_team, primary_size, secondary_size, st)
            results.append(Lineup(player_ids=ids))
            if _maybe_report(len(results)):
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
        pid_team = self._pid_team
        pmeta = self._pmeta
        for pid in ids:
            if pmeta[pid]["position"] != "P":
                t = pid_team[pid]
                team_counts[t] = team_counts.get(t, 0) + 1
        if not team_counts:
            return False
        sorted_counts = sorted(team_counts.values(), reverse=True)
        top = sorted_counts[0]
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0
        return top >= 5 or (top + second >= 6 and second >= 2)

    def _sample_one(
        self,
        rng: np.random.Generator,
        primary_team: str,
        primary_size: int,
        secondary_size: int,
        hint_secondary: str = '',
        effective_floor: Optional[float] = None,
    ) -> tuple[Optional[list[int]], str, str]:
        """Attempt to build one stacked lineup.

        Parameters
        ----------
        primary_team : pre-selected primary team (from generate's round-robin)
        hint_secondary : preferred secondary team from round-robin cursor;
            used if it has enough eligible batters, otherwise falls back to
            weighted sampling.

        Returns
        -------
        (ids, primary_team, secondary_team) on success, or (None, '', '') on failure.
        """
        pmeta = self._pmeta
        pos_pools = self._pos_pools
        pid_to_ow = self._pid_to_ow
        pid_team = self._pid_team
        pid_salary = self._pid_salary
        team_game = self._team_game
        salary_floor_val = (
            effective_floor if effective_floor is not None else (self._salary_floor or 0.0)
        )

        # --- Primary stack ---
        primary_pool = [
            pid for pid in self._team_batters.get(primary_team, [])
            if pid in pid_to_ow
        ]
        if len(primary_pool) < primary_size:
            return None, '', ''

        sw = np.array([pid_to_ow[pid] for pid in primary_pool], dtype=np.float64)
        sw /= sw.sum()
        chosen_idx = _gumbel_choice(rng, len(primary_pool), primary_size, sw)
        primary_ids = {int(primary_pool[i]) for i in chosen_idx}
        used: set[int] = set(primary_ids)
        current_salary: int = sum(pid_salary[pid] for pid in primary_ids)

        # --- Secondary team selection ---
        secondary_team = ''
        secondary_ids: set[int] = set()
        if secondary_size > 0:
            primary_games = {pmeta[pid]["game"] for pid in primary_ids}

            # All eligible secondary teams (excluding primary).
            teams = list(self._team_weights.keys())
            other_teams = [t for t in teams if t != primary_team and t in self._team_batters]
            if not other_teams:
                return None, '', ''

            # Salary feasibility pre-filter: can the secondary team's top-K batters
            # contribute enough, given the max possible from remaining slots after secondary?
            n_after_sec = 10 - primary_size - secondary_size
            max_from_rest = sum(self._top_batter_salaries[:n_after_sec]) if n_after_sec > 0 else 0
            min_sec_sum_needed = max(0, salary_floor_val - current_salary - max_from_rest)
            if min_sec_sum_needed > 0:
                other_teams = [
                    t for t in other_teams
                    if self._team_top_k_salary.get(t, {}).get(secondary_size, 0) >= min_sec_sum_needed
                ]
            if not other_teams:
                return None, '', ''

            # Try hint_secondary first (round-robin cursor from generate()).
            hint_pool = []
            if hint_secondary and hint_secondary in other_teams:
                hint_pool = [
                    pid for pid in self._team_batters.get(hint_secondary, [])
                    if pid in pid_to_ow and pid not in used
                ]

            if hint_pool and len(hint_pool) >= secondary_size:
                secondary_team = hint_secondary
                sec_pool = hint_pool
            else:
                # Weighted fallback among feasible other teams.
                sec_raw = np.array(
                    [
                        (self._team_weights.get(t, 0.0) ** self._team_weight_power
                         if self._team_weights.get(t, 0.0) > 0 else 0.0)
                        * (2.0 if team_game.get(t, "") not in primary_games else 1.0)
                        for t in other_teams
                    ],
                    dtype=np.float64,
                )
                sec_sum = sec_raw.sum()
                if sec_sum == 0:
                    return None, '', ''
                sec_w = sec_raw / sec_sum
                secondary_team = other_teams[int(np.searchsorted(np.cumsum(sec_w), rng.random()))]
                sec_pool = [
                    pid for pid in self._team_batters.get(secondary_team, [])
                    if pid in pid_to_ow and pid not in used
                ]

            if len(sec_pool) < secondary_size:
                return None, '', ''

            sec_sw = np.array([pid_to_ow[pid] for pid in sec_pool], dtype=np.float64)
            sec_sw /= sec_sw.sum()
            sec_idx = _gumbel_choice(rng, len(sec_pool), secondary_size, sec_sw)
            secondary_ids = {int(sec_pool[i]) for i in sec_idx}
            used |= secondary_ids
            current_salary += sum(pid_salary[pid] for pid in secondary_ids)

        stack_ids = primary_ids | secondary_ids

        # --- Build fill_slots early (needed for pitcher look-ahead) ---
        hitter_team_count: dict[str, int] = {}
        for pid in stack_ids:
            if pmeta[pid]["position"] != "P":
                t = pid_team[pid]
                hitter_team_count[t] = hitter_team_count.get(t, 0) + 1

        fill_slots: list[str] = []
        for pos_name in _BATTER_POSITIONS:
            already_covered = sum(1 for pid in stack_ids if pmeta[pid]["position"] == pos_name)
            need = ROSTER_REQUIREMENTS.get(pos_name, 0) - already_covered
            if need > 0:
                fill_slots.extend([pos_name] * need)

        n_fill = len(fill_slots)

        # --- Pitchers with salary look-ahead ---
        excluded_opps = {pmeta[pid]["opponent"] for pid in stack_ids}
        p_ids_all, p_w_all = pos_pools.get("P", ([], np.array([])))
        pitcher_pool = [
            pid for pid in p_ids_all
            if pid_team[pid] not in excluded_opps and pid not in used
        ]
        if len(pitcher_pool) < 2:
            return None, '', ''

        # Max fill salary: conservative upper bound using global top batter salaries.
        max_fill_sum = sum(self._top_batter_salaries[:n_fill]) if n_fill > 0 else 0

        # Early exit: even best pitchers + best fill can't reach floor.
        pit_sals_sorted = sorted((pid_salary[pid] for pid in pitcher_pool), reverse=True)
        max_pit_sum = sum(pit_sals_sorted[:2])
        if current_salary + max_pit_sum + max_fill_sum < salary_floor_val:
            return None, '', ''

        # Per-pitcher floor: filter out pitchers that make a feasible pair impossible.
        min_pit_sum = max(0, salary_floor_val - current_salary - max_fill_sum)
        max_single_pit = pit_sals_sorted[0] if pit_sals_sorted else 0
        min_per_pit = max(0, min_pit_sum - max_single_pit)
        if min_per_pit > 0:
            pitcher_pool = [pid for pid in pitcher_pool if pid_salary[pid] >= min_per_pit]
        if len(pitcher_pool) < 2:
            return None, '', ''

        p_w_dict = dict(zip(p_ids_all, p_w_all))
        pit_w = np.array(
            [p_w_dict.get(pid, 1.0 / len(pitcher_pool)) for pid in pitcher_pool],
            dtype=np.float64,
        )
        pit_w /= pit_w.sum()
        pit_idx = _gumbel_choice(rng, len(pitcher_pool), 2, pit_w)
        pitcher_ids = [int(pitcher_pool[i]) for i in pit_idx]
        if pid_team[pitcher_ids[0]] == pid_team[pitcher_ids[1]]:
            return None, '', ''
        used |= set(pitcher_ids)
        current_salary += pid_salary[pitcher_ids[0]] + pid_salary[pitcher_ids[1]]

        pitcher_opp_teams: set[str] = {
            self._pid_opponent[pid] for pid in pitcher_ids if self._pid_opponent.get(pid)
        }

        # --- Fill remaining hitter positions (salary-window-aware) ---
        remaining_fill = n_fill
        min_batter_sal = self._min_batter_salary

        remaining: list[int] = []
        for pos_name in fill_slots:
            p_ids_pos, p_w_pos = pos_pools.get(pos_name, ([], np.array([])))

            min_sal = max(
                0.0,
                (salary_floor_val - current_salary) - min_batter_sal * (remaining_fill - 1),
            )
            max_sal = (SALARY_CAP - current_salary) - min_batter_sal * (remaining_fill - 1)

            cands = [
                pid for pid in p_ids_pos
                if pid not in used
                and hitter_team_count.get(pid_team[pid], 0) < _MAX_HITTERS_PER_TEAM
                and pid_team[pid] not in pitcher_opp_teams
                and pid_salary[pid] >= min_sal
                and pid_salary[pid] <= max_sal
            ]
            if not cands:
                return None, '', ''

            if self._fill_weight_power == 0.0:
                chosen_pid = int(cands[int(rng.integers(len(cands)))])
            else:
                cw_dict = dict(zip(p_ids_pos, p_w_pos))
                raw_cw = np.array(
                    [cw_dict.get(pid, 1.0 / len(cands)) for pid in cands],
                    dtype=np.float64,
                )
                if self._fill_weight_power != 1.0:
                    raw_cw = np.power(raw_cw, self._fill_weight_power)
                raw_cw /= raw_cw.sum()
                chosen_pid = int(cands[int(rng.choice(len(cands), p=raw_cw))])

            t = pid_team[chosen_pid]
            hitter_team_count[t] = hitter_team_count.get(t, 0) + 1
            current_salary += pid_salary[chosen_pid]
            remaining_fill -= 1
            used.add(chosen_pid)
            remaining.append(chosen_pid)

        all_ids = pitcher_ids + list(stack_ids) + remaining
        if len(set(all_ids)) != 10 or len(all_ids) != 10:
            return None, '', ''
        # Full-stack floor guard (fill_slots empty when primary+secondary cover all 8 slots).
        if self._salary_floor is not None and current_salary < self._salary_floor:
            return None, '', ''
        if not _is_valid_field_lineup(all_ids, pmeta):
            return None, '', ''
        return all_ids, primary_team, secondary_team
