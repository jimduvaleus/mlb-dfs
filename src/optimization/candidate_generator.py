"""Fast ownership-weighted stacked lineup generator for GPP candidate pool.

Generates M valid DraftKings Classic lineups via a two-phase approach:
  Phase 1 — round-robin: each team gets an equal quota of primary-stack
    lineups; within each team's quota, secondary teams are cycled in
    order to ensure pair coverage without explicit pair-cap tracking.
  Phase 2 — backfill: any remaining slots are filled with weighted
    sampling, skipping teams that already hit quota.

Stack-size distribution targets (configurable via GROUP_FRACTIONS):
  5-hitter primary (5, 5-2, 5-3): 62 %
  4-hitter primary (4-4, 4-3, 4-2): 31 %
  3-hitter primary (3-3):            7 %

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

# Secondary stacks from the opposite side of the primary stack's game share
# its run environment (park, weather, umpire — the copula's latent env factor),
# so they correlate positively with the primary and get a slight selection
# boost. Different-game secondaries carry no such coupling.
# _PROB overrides the round-robin secondary hint with the primary's opponent
# (the path that assigns most secondaries); _BOOST applies the same preference
# in the weighted fallback used when the hint is infeasible.
_SAME_GAME_SECONDARY_PROB = 0.10
_SAME_GAME_SECONDARY_BOOST = 1.25


def _gumbel_choice(rng: np.random.Generator, n: int, size: int, weights: np.ndarray) -> np.ndarray:
    """Weighted sampling without replacement via Efraimidis-Spirakis (O(n))."""
    log_u = np.log(rng.random(n))
    keys = np.where(weights > 0, log_u / weights, -np.inf)
    return np.argpartition(keys, -size)[-size:]


def _capped_gumbel_choice(
    rng: np.random.Generator,
    pool: list,
    size: int,
    weights: np.ndarray,
    pos_of: dict,
    room: dict,
) -> Optional[list[int]]:
    """Weighted sampling without replacement (Efraimidis-Spirakis) that skips
    players whose primary position has no remaining roster room.

    Position-blind stack draws are the dominant candidate-attrition source:
    an 8-batter (5,3)/(4,4) two-team stack sampled without position caps fits
    the C/1B/2B/3B/SS/OF×3 slots only ~4% of the time, so big-secondary
    patterns die at the fill/roster stage ~99% of the time and the realized
    pool skews to (5,0)/(5,2) regardless of pattern draw weights. Capping by
    position keeps every draw roster-feasible.

    Mutates `room` (position → remaining slots) for the chosen players.
    Returns the chosen player ids, or None when the pool cannot supply
    `size` players within the caps (room is left unchanged in that case).
    """
    log_u = np.log(rng.random(len(pool)))
    keys = np.where(weights > 0, log_u / weights, -np.inf)
    chosen: list[int] = []
    for i in np.argsort(keys)[::-1]:
        if keys[int(i)] == -np.inf:
            break
        pid = int(pool[int(i)])
        p = pos_of[pid]["position"]
        if room.get(p, 0) > 0:
            room[p] -= 1
            chosen.append(pid)
            if len(chosen) == size:
                return chosen
    for pid in chosen:
        room[pos_of[pid]["position"]] += 1
    return None


class CandidateGenerator:
    """Generate M valid DK Classic stacked lineups.

    Stack patterns by primary hitter count (repeated entries = higher draw
    weight; patterns are drawn uniformly from each group's list):
      5-group: (5,3)×2, (5,2), (5,0)
      4-group: (4,3)×2, (4,4), (4,2)
      3-group: (3,3)
    The x-3 patterns are overweighted because real top-1% GPP lineups carry a
    2-3 batter secondary stack far more often than uniform pattern draws
    produce after sampling attrition (measured 83% vs 54%; see
    scripts/measure_pool_ceiling.py composition tables).

    Each lineup satisfies the GPP stacking requirement:
      top team ≥ 5 hitters  OR  top-2 combined ≥ 6 with second ≥ 2.
    """

    STACK_GROUPS: dict[int, list[tuple[int, int]]] = {
        5: [(5, 3), (5, 3), (5, 2), (5, 0)],
        4: [(4, 3), (4, 3), (4, 4), (4, 2)],
        3: [(3, 3)],
    }
    # Target fraction of generated lineups per stack-size group.
    # Tuned 2026-07-06 against real top-1% composition (prim5 ~55-68% across
    # slates; realized pool prim5 ≈ 60% at these fractions).
    GROUP_FRACTIONS: dict[int, float] = {5: 0.62, 4: 0.31, 3: 0.07}

    @property
    def STACK_PATTERNS(self) -> list[tuple[int, int]]:
        """Flat list of distinct patterns (backward-compat property)."""
        return list(dict.fromkeys(
            p for pats in self.STACK_GROUPS.values() for p in pats
        ))

    def __init__(
        self,
        players_df: pd.DataFrame,
        ownership_vec: np.ndarray,
        rng_seed: Optional[int] = None,
        salary_floor: Optional[float] = None,
        team_weight_power: float = 0.5,
        fill_weight_power: float = 0.0,
        fill_salary_tilt: float = 2.0,
        stack_salary_tilt: float = 1.0,
        spend_up_prob: float = 0.30,
    ) -> None:
        self._players_df = players_df
        self._ownership_vec = ownership_vec
        self._rng_seed = rng_seed
        self._salary_floor = salary_floor
        self._team_weight_power = float(team_weight_power)
        self._fill_weight_power = float(fill_weight_power)
        # Tilt fill-slot sampling toward the expensive end of each slot's
        # feasible salary window: weight ∝ (salary/window_max)^tilt. Uniform
        # fills systematically underspend relative to real top-1% lineups
        # (pool salary mean ~49.0k / 14% at-cap vs 49.7k / 28%). 0 = uniform.
        self._fill_salary_tilt = float(fill_salary_tilt)
        # Same idea for the stack batter draws (weight ∝ ownership ×
        # (salary/team_max)^tilt). With big secondaries most lineups have 0-1
        # fill slots, so total spend is set by the stack picks — fill tilt
        # alone cannot close the at-cap gap (pool 10-17% vs real top-1% ~29%).
        # Expensive studs are also the high-owned ones, so a mild tilt keeps
        # ownership realism. 0 = pure ownership weights.
        self._stack_salary_tilt = float(stack_salary_tilt)
        # Probability that a completed lineup gets a greedy spend-up pass
        # (same-team same-position upgrades until the leftover budget is
        # spent). Salary tilts alone cannot reproduce the real field's at-cap
        # rate (~24-30% of entries land on exactly $50,000): hitting the cap
        # is a budget-exhausting *construction* behavior, not a marginal
        # preference. 0 = disabled.
        self._spend_up_prob = float(spend_up_prob)

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

        # (team, primary position) -> player ids, for the spend-up pass.
        # Same-team same-position swaps preserve every structural property of
        # a lineup (stack counts, roster multiset, pitcher-opponent rules), so
        # spend-up only ever changes total salary.
        self._team_pos_players: dict[tuple[str, str], list[int]] = {}
        for pid, meta in self._pmeta.items():
            self._team_pos_players.setdefault(
                (meta["team"], meta["position"]), []
            ).append(pid)

        # Salary look-ahead: sorted batter salaries descending for fill estimation.
        batter_salaries = [self._pid_salary[pid] for pid in self._pid_to_ow]
        self._min_batter_salary: float = float(min(batter_salaries)) if batter_salaries else 0.0
        self._top_batter_salaries: list[int] = sorted(batter_salaries, reverse=True)

        # Per-team minimum fill batter salary, excluding that team's own batters.
        # Used in _sample_one fill window so that a cheap primary team's own salary
        # doesn't artificially inflate the min_sal for fill positions (fill batters
        # must come from other teams anyway, since the primary batters are already used).
        self._min_fill_salary_excl_team: dict[str, float] = {
            team: float(min(
                (self._pid_salary[pid] for pid in self._pid_to_ow
                 if self._pid_team[pid] != team),
                default=self._min_batter_salary,
            ))
            for team in self._team_batters
        }

        # Top-2 pitcher salaries for secondary feasibility estimate (pitchers are NOT batters).
        p_ids_init, _ = self._pos_pools.get("P", ([], np.array([])))
        self._top_pitcher_salary_sum: int = sum(
            sorted((self._pid_salary[pid] for pid in p_ids_init), reverse=True)[:2]
        )

        # Per-team top-k salary sum for secondary feasibility filter (k = 1..5).
        self._team_top_k_salary: dict[str, dict[int, int]] = {}
        for t, pids in self._team_batters.items():
            sals = sorted(
                (self._pid_salary[p] for p in pids if p in self._pid_to_ow),
                reverse=True,
            )
            self._team_top_k_salary[t] = {k: sum(sals[:k]) for k in range(1, len(sals) + 1)}

        # Precomputed team→game and team→opponent.
        self._team_game: dict[str, str] = {}
        for t, pids in self._team_batters.items():
            for pid in pids:
                g = self._pmeta[pid].get("game", "")
                if g:
                    self._team_game[t] = g
                    break
        self._team_opponent: dict[str, str] = {}
        for t, pids in self._team_batters.items():
            for pid in pids:
                opp = self._pmeta[pid].get("opponent", "")
                if opp:
                    self._team_opponent[t] = opp
                    break

        # Per-team effective salary floor.
        # For each primary team, walk all stack patterns and compute the best salary
        # achievable: top-k primary batters + best secondary k' from any other team +
        # top-2 pitchers available (excluding opponent's game) + top fill batters.
        # The team floor is min(configured_floor, max_achievable_across_patterns).
        # Teams with cheap rosters automatically get a lower floor so the generator
        # can produce a proportional share of lineups for them.
        self._team_salary_floor: dict[str, float] = {}
        if self._salary_floor is not None:
            _team_to_opp = self._team_opponent

            # Top-2 pitcher salary sum available for each primary team
            # (pitchers from the opponent's team are excluded).
            # Also compute the "typical" pair (median-ranked pitchers) to derive
            # a realistic floor for cheap-roster teams.
            _p_ids_all, _ = self._pos_pools.get("P", ([], np.array([])))
            _team_to_pit_sum: dict[str, int] = {}
            _team_to_typical_pit_sum: dict[str, int] = {}
            for _team in self._team_batters:
                _opp = _team_to_opp.get(_team, "")
                _avail = sorted(
                    (self._pid_salary[pid] for pid in _p_ids_all
                     if self._pid_team[pid] != _opp),
                    reverse=True,
                )
                _n_p = len(_avail)
                _team_to_pit_sum[_team] = sum(_avail[:2]) if _n_p >= 2 else sum(_avail)
                # Typical pair: two pitchers centered on the median rank.
                if _n_p >= 2:
                    _mid = _n_p // 2
                    _i1 = min(_mid, _n_p - 2)
                    _team_to_typical_pit_sum[_team] = _avail[_i1] + _avail[_i1 + 1]
                else:
                    _team_to_typical_pit_sum[_team] = sum(_avail)

            # Global median batter salary for typical fill estimation.
            _all_batter_sals_sorted = sorted(
                self._pid_salary[p] for p in self._pid_to_ow
            )
            _typical_batter_sal = float(
                np.median(_all_batter_sals_sorted)
            ) if _all_batter_sals_sorted else 0.0

            # Sorted batter salaries per team (descending).
            _team_batter_sals: dict[str, list[int]] = {
                t: sorted(
                    (self._pid_salary[p] for p in pids if p in self._pid_to_ow),
                    reverse=True,
                )
                for t, pids in self._team_batters.items()
            }

            for _team, _team_sals in _team_batter_sals.items():
                _pit_sum = _team_to_pit_sum.get(_team, self._top_pitcher_salary_sum)
                _typical_pit_sum = _team_to_typical_pit_sum.get(_team, _pit_sum)
                _max_achievable = 0.0
                # Typical achievable: track the MINIMUM across patterns so that
                # even the hardest pattern (e.g. 5-0, no secondary) can succeed
                # at the derived floor with typical pitcher+fill combinations.
                # Patterns with secondary always add salary, so the minimum
                # comes from the no-secondary (or small-secondary) patterns.
                _typical_achievable_min = float("inf")
                for _prim_sz, _sec_sz in self.STACK_PATTERNS:
                    if len(_team_sals) < _prim_sz:
                        continue
                    _prim_sum = sum(_team_sals[:_prim_sz])
                    _sec_sum = (
                        max(
                            (self._team_top_k_salary.get(t, {}).get(_sec_sz, 0)
                             for t in self._team_batters if t != _team),
                            default=0,
                        )
                        if _sec_sz > 0 else 0
                    )
                    _n_fill = 8 - _prim_sz - _sec_sz
                    # Max achievable (best-case pitchers + best-case fill).
                    _fill_sum = sum(self._top_batter_salaries[:_n_fill]) if _n_fill > 0 else 0
                    _pat_max = min(
                        float(SALARY_CAP),
                        float(_prim_sum + _sec_sum + _pit_sum + _fill_sum),
                    )
                    if _pat_max > _max_achievable:
                        _max_achievable = _pat_max
                    # Typical achievable (median-ranked pitchers + median fill salary).
                    _fill_sum_typ = _typical_batter_sal * _n_fill if _n_fill > 0 else 0.0
                    _pat_typ = min(
                        float(SALARY_CAP),
                        float(_prim_sum + _sec_sum + _typical_pit_sum + _fill_sum_typ),
                    )
                    if _pat_typ < _typical_achievable_min:
                        _typical_achievable_min = _pat_typ

                _typical_achievable = (
                    _typical_achievable_min if _typical_achievable_min < float("inf") else 0.0
                )

                # Floor = min(configured, max_achievable). max_achievable is the
                # best-case salary (top pitchers + top fill) a team can build with
                # the *hardest* stack pattern — i.e. whether the configured floor
                # is even reachable at all for this team. Using typical_achievable
                # (median-case pitchers/fill) here instead would relax the floor
                # for any team whose *typical* construction undershoots, which in
                # practice is nearly every team (median-case is a pessimistic bar,
                # not an infeasibility signal) — verified against a live 16-team
                # slate: max_achievable was $50,000 (cap) for every team, yet the
                # old typical-based floor still got lowered for 15/16 of them,
                # silently exempting ~42% of the candidate pool from the
                # configured floor even though every team could clear it.
                # Teams that are genuinely hard (but not infeasible) to hit at the
                # configured floor are still covered by the bounded, runtime
                # dynamic relief below (capped at floor_relief per generate()).
                _floor = min(self._salary_floor, _max_achievable)
                self._team_salary_floor[_team] = _floor
                if _floor < self._salary_floor:
                    logger.info(
                        "CandidateGenerator: %s salary floor lowered to %.0f "
                        "(max achievable %.0f, typical achievable %.0f, configured %.0f)",
                        _team, _floor, _max_achievable, _typical_achievable,
                        self._salary_floor,
                    )

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

            # Per-team dynamic floor: starts at the team's effective floor, can decrease.
            base_floor = float(self._team_salary_floor.get(primary_team, self._salary_floor or 0.0))
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

            _p2_floor = float(self._team_salary_floor.get(primary_team, self._salary_floor or 0.0))
            ids, pt, st = self._sample_one(
                rng, primary_team, primary_size, secondary_size,
                effective_floor=_p2_floor,
            )
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

    def generate_sim_winners(
        self,
        sim_matrix: np.ndarray,
        sim_player_ids: list[int],
        sim_indices: list[int],
        per_world: int = 1,
        temp: float = 0.15,
        own_blend: float = 0.25,
        attempts_per_lineup: int = 6,
        progress_cb: Optional[Callable[[int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[Lineup]:
        """Sample "sim winner" lineups: for each simulated world, draw stacked
        lineups through the normal _sample_one machinery but with sampling
        weights derived from that world's realized player scores instead of
        ownership.

        This is the scaled, diversity-preserving replacement for per-sim exact
        ILP optima (generate_sim_optimal_lineups): sampling near the top of
        each world rather than taking its argmax avoids the seed redundancy
        and structural extremity of exact per-world optima while still
        concentrating the pool on lineups that win *some* simulated world.

        Parameters
        ----------
        sim_matrix : (n_sims, n_players) from SimulationResults
        sim_player_ids : column order of sim_matrix
        sim_indices : simulated worlds to draw from (typically
            stratified_sim_sample so quiet and explosive run environments are
            both represented)
        per_world : lineups drawn per world
        temp : rank-softmax temperature over each world's realized scores
            (weight = exp((rank_pct - 1) / temp)); lower = greedier toward the
            world's top scorers
        own_blend : exponent on normalized ownership multiplied into the
            world weights — 0 = pure sim-score sampling, 1 = full ownership
            damping (keeps picks from drifting to unowned punts)
        """
        col_map = {int(p): i for i, p in enumerate(sim_player_ids)}
        batter_pids = [pid for pid in self._pid_to_ow]
        bat_cols = np.array([col_map.get(pid, -1) for pid in batter_pids], dtype=np.int64)
        bat_own = np.array([self._pid_to_ow[pid] for pid in batter_pids], dtype=np.float64)
        pit_ids, pit_own = self._pos_pools.get("P", ([], np.array([])))
        pit_cols = np.array([col_map.get(pid, -1) for pid in pit_ids], dtype=np.int64)
        pit_own = np.asarray(pit_own, dtype=np.float64)

        def _norm(w: np.ndarray) -> np.ndarray:
            mx = w.max() if len(w) and w.max() > 0 else 1.0
            return w / mx

        bat_own_n = np.power(np.clip(_norm(bat_own), 1e-6, None), own_blend)
        pit_own_n = np.power(np.clip(_norm(pit_own), 1e-6, None), own_blend)

        def _world_weights(scores: np.ndarray, own_n: np.ndarray, cols: np.ndarray) -> np.ndarray:
            # rank_pct in (0, 1]; missing-from-sim players get the world floor.
            n = len(scores)
            rank_pct = np.empty(n, dtype=np.float64)
            rank_pct[np.argsort(scores, kind="stable")] = (np.arange(n) + 1) / n
            w = np.exp((rank_pct - 1.0) / max(temp, 1e-3)) * own_n
            w[cols == -1] = w.min() if len(w) else 0.0
            return w

        # Per-team index into batter_pids for fast team-weight sums.
        team_bat_idx: dict[str, np.ndarray] = {}
        for i, pid in enumerate(batter_pids):
            team_bat_idx.setdefault(self._pid_team[pid], []).append(i)
        team_bat_idx = {t: np.array(ix, dtype=np.int64) for t, ix in team_bat_idx.items()}

        _saved = (self._pid_to_ow, self._pos_pools, self._team_weights)
        _all_keys = [5, 4, 3]
        _all_probs = np.array(
            [self.GROUP_FRACTIONS[k] for k in _all_keys], dtype=np.float64
        )
        rng = np.random.default_rng(self._rng_seed)
        results: list[Lineup] = []
        seen: set[frozenset] = set()
        n_failed_worlds = 0
        try:
            for w_i, sim_idx in enumerate(sim_indices):
                if stop_check is not None and w_i % 200 == 0 and stop_check():
                    break
                bat_scores = np.where(
                    bat_cols >= 0, sim_matrix[sim_idx, np.maximum(bat_cols, 0)], -np.inf
                )
                pit_scores = np.where(
                    pit_cols >= 0, sim_matrix[sim_idx, np.maximum(pit_cols, 0)], -np.inf
                )
                bw = _world_weights(bat_scores, bat_own_n, bat_cols)
                pw = _world_weights(pit_scores, pit_own_n, pit_cols)

                self._pid_to_ow = dict(zip(batter_pids, bw))
                pools = dict(self._pos_pools)
                for pos_name in _BATTER_POSITIONS:
                    ids, _ = pools.get(pos_name, ([], np.array([])))
                    pools[pos_name] = (ids, np.array(
                        [self._pid_to_ow.get(pid, 0.0) for pid in ids], dtype=np.float64
                    ))
                pools["P"] = (pit_ids, pw)
                self._pos_pools = pools
                self._team_weights = {
                    t: float(bw[ix].sum()) for t, ix in team_bat_idx.items()
                }

                # Primary team ∝ the world's team weight mass — the "hot team"
                # of this world, softened by the same rank-softmax that formed
                # the player weights.
                teams = list(self._team_weights)
                tw = np.array([self._team_weights[t] for t in teams], dtype=np.float64)
                if tw.sum() <= 0:
                    n_failed_worlds += 1
                    continue
                tw /= tw.sum()

                made = 0
                for _ in range(attempts_per_lineup * per_world):
                    if made >= per_world:
                        break
                    primary_team = teams[int(np.searchsorted(np.cumsum(tw), rng.random()))]
                    g = _all_keys[int(rng.choice(3, p=_all_probs))]
                    pats = self.STACK_GROUPS[g]
                    primary_size, secondary_size = pats[int(rng.integers(len(pats)))]
                    floor = float(self._team_salary_floor.get(
                        primary_team, self._salary_floor or 0.0
                    ))
                    ids, _pt, _st = self._sample_one(
                        rng, primary_team, primary_size, secondary_size,
                        effective_floor=floor,
                    )
                    if ids is None or not self._check_stack(ids):
                        continue
                    key = frozenset(int(p) for p in ids)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(Lineup(player_ids=ids))
                    made += 1
                if made == 0:
                    n_failed_worlds += 1
                if progress_cb is not None and (w_i + 1) % 500 == 0:
                    progress_cb(len(results))
        finally:
            self._pid_to_ow, self._pos_pools, self._team_weights = _saved

        logger.info(
            "generate_sim_winners: %d lineups from %d worlds "
            "(%d worlds produced nothing, %d duplicates avoided).",
            len(results), len(sim_indices), n_failed_worlds,
            len(sim_indices) * per_world - len(results) - n_failed_worlds * per_world,
        )
        return results

    def generate_mutants(
        self,
        parents: list[Lineup],
        n_per_parent: int,
        seen: set,
        rng_seed: Optional[int] = None,
        max_attempts_per_mutant: int = 25,
        salary_locality: float = 2000.0,
        pitcher_swap_weight: float = 0.15,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[Lineup]:
        """Generate neighborhood mutants of high-EV parent lineups.

        Each mutant swaps 1-2 players for same-primary-position replacements,
        biased toward similar salary so the cap/floor window stays feasible.
        Mutants must pass the GPP stack requirement and full lineup validity
        (bipartite slot matching covers parents that used multi-position
        players at secondary slots, e.g. ILP-seeded lineups).

        Parameters
        ----------
        parents : lineups to mutate (typically the pool's top $EV rows)
        n_per_parent : mutants to produce per parent
        seen : set of frozenset(player_ids) covering the whole pool; mutants
            are deduped against it and added to it
        salary_locality : exponential decay scale (dollars) for replacement
            sampling — smaller values keep swaps closer in salary
        pitcher_swap_weight : relative probability of mutating a pitcher slot
            vs a batter slot
        """
        rng = np.random.default_rng(
            rng_seed if rng_seed is not None else self._rng_seed
        )

        # Validation metadata: _pmeta plus eligible_positions when available,
        # so Lineup.is_valid()'s slot matching honors multi-position players.
        if not hasattr(self, "_mutant_meta"):
            self._mutant_meta = {pid: dict(m) for pid, m in self._pmeta.items()}
            if "eligible_positions" in self._players_df.columns:
                for r in self._players_df.itertuples(index=False):
                    ep = r.eligible_positions
                    if ep is not None:
                        self._mutant_meta[int(r.player_id)]["eligible_positions"] = list(ep)

        out: list[Lineup] = []
        for parent in parents:
            if stop_check is not None and stop_check():
                break
            p_ids = [int(pid) for pid in parent.player_ids]
            if any(pid not in self._pmeta for pid in p_ids):
                continue
            parent_salary = sum(self._pid_salary[pid] for pid in p_ids)
            roster_size = len(p_ids)

            slot_w = np.array(
                [
                    pitcher_swap_weight if self._pmeta[pid]["position"] == "P" else 1.0
                    for pid in p_ids
                ],
                dtype=np.float64,
            )
            slot_w /= slot_w.sum()

            produced = 0
            attempts = 0
            max_attempts = n_per_parent * max_attempts_per_mutant
            while produced < n_per_parent and attempts < max_attempts:
                attempts += 1
                ids = list(p_ids)
                n_swaps = 1 if rng.random() < 0.6 else 2
                swap_slots = rng.choice(roster_size, size=n_swaps, replace=False, p=slot_w)

                feasible_swap = True
                for j in swap_slots:
                    old_pid = ids[j]
                    pos = self._pmeta[old_pid]["position"]
                    pool_ids, _ = self._pos_pools.get(pos, ([], np.array([])))
                    in_lineup = set(ids)
                    cand_pids = [pid for pid in pool_ids if pid not in in_lineup]
                    if not cand_pids:
                        feasible_swap = False
                        break
                    old_sal = self._pid_salary[old_pid]
                    w = np.array(
                        [
                            math.exp(-abs(self._pid_salary[pid] - old_sal) / salary_locality)
                            for pid in cand_pids
                        ],
                        dtype=np.float64,
                    )
                    w_sum = w.sum()
                    if w_sum <= 0:
                        feasible_swap = False
                        break
                    pick = int(np.searchsorted(np.cumsum(w / w_sum), rng.random()))
                    ids[j] = cand_pids[min(pick, len(cand_pids) - 1)]
                if not feasible_swap:
                    continue

                key = frozenset(ids)
                if key in seen:
                    continue

                salary = sum(self._pid_salary[pid] for pid in ids)
                # Parents may sit below the configured floor (floor relief),
                # so a mutant is acceptable if it meets the floor OR doesn't
                # fall below its parent's salary.
                if (
                    self._salary_floor is not None
                    and salary < self._salary_floor
                    and salary < parent_salary
                ):
                    continue

                if not self._check_stack(ids):
                    continue
                mutant = Lineup(player_ids=ids)
                if not mutant.is_valid(self._mutant_meta):
                    continue

                seen.add(key)
                out.append(mutant)
                produced += 1

        return out

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

        # Batter-slot room shared by primary and secondary draws so the
        # stack is always roster-feasible (see _capped_gumbel_choice).
        slot_room = {p: n for p, n in ROSTER_REQUIREMENTS.items() if p != "P"}

        sw = np.array([pid_to_ow[pid] for pid in primary_pool], dtype=np.float64)
        if self._stack_salary_tilt != 0.0:
            _sal = np.array([pid_salary[pid] for pid in primary_pool], dtype=np.float64)
            sw = sw * np.power(_sal / _sal.max(), self._stack_salary_tilt)
        sw /= sw.sum()
        picked = _capped_gumbel_choice(rng, primary_pool, primary_size, sw, pmeta, slot_room)
        if picked is None:
            return None, '', ''
        primary_ids = set(picked)
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
            # Fill batter slots = 8 batter slots - primary - secondary (pitchers are separate).
            n_fill_batters = 8 - primary_size - secondary_size
            max_from_rest = (
                (sum(self._top_batter_salaries[:n_fill_batters]) if n_fill_batters > 0 else 0)
                + self._top_pitcher_salary_sum
            )
            min_sec_sum_needed = max(0, salary_floor_val - current_salary - max_from_rest)
            if min_sec_sum_needed > 0:
                other_teams = [
                    t for t in other_teams
                    if self._team_top_k_salary.get(t, {}).get(secondary_size, 0) >= min_sec_sum_needed
                ]
            if not other_teams:
                return None, '', ''

            # Slight same-game preference: with small probability override the
            # round-robin hint with the primary's opponent when it is feasible
            # (see _SAME_GAME_SECONDARY_PROB). The round-robin cursor assigns
            # most secondaries, so the preference must act here, not just in
            # the weighted fallback below.
            opp_team = self._team_opponent.get(primary_team, "")
            if (
                opp_team
                and opp_team in other_teams
                and rng.random() < _SAME_GAME_SECONDARY_PROB
            ):
                hint_secondary = opp_team

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
                        * (_SAME_GAME_SECONDARY_BOOST if team_game.get(t, "") in primary_games else 1.0)
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
            if self._stack_salary_tilt != 0.0:
                _ssal = np.array([pid_salary[pid] for pid in sec_pool], dtype=np.float64)
                sec_sw = sec_sw * np.power(_ssal / _ssal.max(), self._stack_salary_tilt)
            sec_sw /= sec_sw.sum()
            picked = _capped_gumbel_choice(
                rng, sec_pool, secondary_size, sec_sw, pmeta, slot_room,
            )
            if picked is None:
                return None, '', ''
            secondary_ids = set(picked)
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

        # Per-pitcher floor: filter to pitchers where any sampled pair is guaranteed
        # to sum to >= min_pit_sum. ceil(min_pit_sum/2) is the tightest per-pitcher
        # lower bound that ensures this regardless of which two are drawn.
        min_pit_sum = max(0, salary_floor_val - current_salary - max_fill_sum)
        min_per_pit = max(0, math.ceil(min_pit_sum / 2))
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
        # Use the min fill salary excluding the primary team's batters: those are
        # already used in the stack, so fill slots must come from other teams.
        # Using the global min (which may be a cheap primary team's own salary)
        # would make min_sal artificially high for cheap-roster primaries.
        min_batter_sal = self._min_fill_salary_excl_team.get(
            primary_team, self._min_batter_salary
        )

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

            if self._fill_weight_power == 0.0 and self._fill_salary_tilt == 0.0:
                chosen_pid = int(cands[int(rng.integers(len(cands)))])
            else:
                if self._fill_weight_power != 0.0:
                    cw_dict = dict(zip(p_ids_pos, p_w_pos))
                    raw_cw = np.array(
                        [cw_dict.get(pid, 1.0 / len(cands)) for pid in cands],
                        dtype=np.float64,
                    )
                    if self._fill_weight_power != 1.0:
                        raw_cw = np.power(raw_cw, self._fill_weight_power)
                else:
                    raw_cw = np.ones(len(cands), dtype=np.float64)
                if self._fill_salary_tilt != 0.0:
                    sal_arr = np.array([pid_salary[pid] for pid in cands], dtype=np.float64)
                    raw_cw = raw_cw * np.power(sal_arr / sal_arr.max(), self._fill_salary_tilt)
                cw_sum = raw_cw.sum()
                if cw_sum <= 0:
                    chosen_pid = int(cands[int(rng.integers(len(cands)))])
                else:
                    raw_cw /= cw_sum
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
        if self._spend_up_prob > 0.0 and rng.random() < self._spend_up_prob:
            all_ids = self._spend_up(all_ids)
        # Full-stack floor guard (fill_slots empty when primary+secondary cover all 8 slots).
        # Use the team's computed effective floor (may be lower than configured for cheap
        # teams). Do NOT use salary_floor_val — dynamic floor relief can lower it further
        # and must not override this hard minimum.
        _hard_floor = self._team_salary_floor.get(primary_team, self._salary_floor)
        if _hard_floor is not None and sum(pid_salary[p] for p in all_ids) < _hard_floor:
            return None, '', ''
        if not _is_valid_field_lineup(all_ids, pmeta):
            return None, '', ''
        return all_ids, primary_team, secondary_team

    def _spend_up(self, ids: list[int]) -> list[int]:
        """Greedy budget-exhausting pass: repeatedly replace a player with the
        costliest alternative that fits the leftover budget, preferring an
        exact leftover match so the lineup lands on the cap.

        Two swap tiers:
          1. same-team same-position — structure-preserving for any player
             (stack counts, roster multiset, pitcher rules all unchanged);
          2. any-team same-position, but only for *singleton* batters (their
             team has exactly one rostered hitter, i.e. fill slots) — carries
             no stack structure; team cap and pitcher-conflict re-checked.
        """
        pid_salary = self._pid_salary
        pid_team = self._pid_team
        pmeta = self._pmeta
        leftover = int(SALARY_CAP) - sum(pid_salary[p] for p in ids)
        if leftover <= 0:
            return ids
        ids = list(ids)
        used = set(ids)
        hitter_counts: dict[str, int] = {}
        pitcher_opps: set[str] = set()
        for pid in ids:
            if pmeta[pid]["position"] == "P":
                opp = self._pid_opponent.get(pid, "")
                if opp:
                    pitcher_opps.add(opp)
            else:
                t = pid_team[pid]
                hitter_counts[t] = hitter_counts.get(t, 0) + 1

        while leftover > 0:
            best = None  # (is_exact, gain, idx, new_pid)

            def _consider(i: int, cand: int) -> None:
                nonlocal best
                gain = pid_salary[cand] - pid_salary[ids[i]]
                if gain <= 0 or gain > leftover or cand in used:
                    return
                key = (gain == leftover, gain)
                if best is None or key > (best[0], best[1]):
                    best = (gain == leftover, gain, i, cand)

            for i, pid in enumerate(ids):
                m = pmeta[pid]
                pos = m["position"]
                # Tier 1: same team, same position (always safe).
                for cand in self._team_pos_players.get((m["team"], pos), ()):
                    _consider(i, cand)
                # Tier 2: singleton batters may swap across teams.
                if pos != "P" and hitter_counts.get(pid_team[pid], 0) == 1:
                    cand_ids, _ = self._pos_pools.get(pos, ([], None))
                    for cand in cand_ids:
                        ct = pid_team[cand]
                        if ct == pid_team[pid]:
                            continue  # tier 1 covered same-team
                        if ct in pitcher_opps:
                            continue
                        if hitter_counts.get(ct, 0) + 1 > _MAX_HITTERS_PER_TEAM:
                            continue
                        _consider(i, cand)

            if best is None:
                break
            _, gain, i, cand = best
            old = ids[i]
            if pmeta[old]["position"] != "P":
                ot, nt = pid_team[old], pid_team[cand]
                if ot != nt:
                    hitter_counts[ot] -= 1
                    hitter_counts[nt] = hitter_counts.get(nt, 0) + 1
            used.discard(old)
            used.add(cand)
            ids[i] = cand
            leftover -= gain
        return ids
