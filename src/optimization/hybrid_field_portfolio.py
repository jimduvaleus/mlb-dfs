"""Hybrid-field simulation-driven portfolio constructor.

Algorithm (three phases):

  Phase 1 (pre-scored — free):
    Consume ContestScorer's robust_payout matrix.  Cull to candidates whose
    mean net payout (across all Phase-1 sims) is > 0.  Sort descending by EV.

  Phase 2 (fast-track — no re-scoring):
    Add the best lineup L*.  Add every remaining +EV candidate that shares zero
    players with L* — they are safe additions because (a) they are independently
    profitable and (b) they add no overlap risk with the anchor lineup.

  Phase 3+ (hybrid-field cycles):
    Repeat until the portfolio is full or the pool is exhausted:
      1. Build a hybrid opponent field of (N − k) random opponent lineups plus
         the k portfolio lineups already selected.  This makes the EV signal
         self-cannibalization-aware: our own entries occupy real field slots, so
         a new correlated candidate must beat our existing lineups to cash.
      2. Run a completely independent fresh simulation (n_hybrid_sims rows) so
         that surviving candidates must be robustly profitable across multiple
         independent draws, not just the Phase-1 sample.
      3. Score all remaining candidates against the hybrid field.
      4. Cull to +EV; add the best lineup and any zero-overlap-with-best
         fast-tracks.  Zero-overlap check is against L* of the current cycle
         only (fast and unlikely to introduce serious correlation risk given
         the hybrid-field re-scoring next cycle).
"""
import logging
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.contest import ContestSimulator
from src.optimization.gpp_portfolio import (
    _build_payout_lookup,
    _compute_payout_from_sorted_field,
)
from src.optimization.lineup import Lineup
from src.optimization.ownership import compute_heuristic_ownership
from src.simulation.engine import SimulationEngine

logger = logging.getLogger(__name__)

_BATCH = 500  # candidates per Numba kernel call


class HybridFieldPortfolioSelector:
    """Simulation-driven portfolio selector with self-cannibalization awareness.

    Parameters
    ----------
    candidates:
        Full candidate pool (output of CandidateGenerator).
    robust_payout:
        (M, n_sims) float32 — per-lineup net payout from ContestScorer Phase 1.
        Positive values indicate the lineup is +EV against the Phase-1 fields.
    sim_engine:
        Initialised SimulationEngine; called each cycle to generate fresh sims.
    players_df:
        Player pool used for computing heuristic ownership (cand_players_df).
    payout_arr:
        (total_entries,) float32 — GROSS payout per rank from the contest
        payout table (entry fee NOT subtracted; the selector handles that).
    portfolio_size:
        Target number of lineups.
    n_field_lineups:
        Total hybrid field size N (opponents + own entries = N throughout).
    n_hybrid_sims:
        Number of independent simulations generated per hybrid cycle.
    field_players_df:
        Optional broader player pool for field lineup generation (sim_players_df).
        Defaults to players_df if not supplied.
    ownership_vec:
        Per-player ownership weights aligned with players_df row order.
        Computed via compute_heuristic_ownership if not supplied.
    team_totals:
        Optional {team: implied_total} for ownership model.
    entry_fee:
        Dollar cost per lineup entry (used to compute net payout lookup).
    rng_seed:
        Base seed for reproducible field generation across cycles.
    """

    def __init__(
        self,
        candidates: list[Lineup],
        robust_payout: np.ndarray,
        sim_engine: SimulationEngine,
        players_df: pd.DataFrame,
        payout_arr: np.ndarray,
        portfolio_size: int,
        n_field_lineups: int = 5_000,
        n_hybrid_sims: int = 10_000,
        field_players_df: Optional[pd.DataFrame] = None,
        ownership_vec: Optional[np.ndarray] = None,
        team_totals: Optional[dict] = None,
        entry_fee: float = 4.0,
        max_correlation: float = 0.9,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._candidates = candidates
        self._robust_payout = np.asarray(robust_payout, dtype=np.float32)
        self._sim_engine = sim_engine
        self._portfolio_size = portfolio_size
        self._n_field = n_field_lineups
        self._n_hybrid_sims = n_hybrid_sims
        self._max_correlation = max_correlation
        self._rng = np.random.default_rng(rng_seed)

        # Column map: player_id → column index in every fresh sim_matrix.
        # Stable because sim_engine.players_df is fixed.
        self._player_ids: list[int] = sim_engine.players_df["player_id"].tolist()
        self._col_map: dict[int, int] = {
            pid: i for i, pid in enumerate(self._player_ids)
        }

        # Pitcher IDs — used by the fast-track overlap rule.
        self._pitcher_ids: frozenset[int] = frozenset(
            int(pid) for pid in players_df.loc[
                players_df["position"] == "P", "player_id"
            ]
        )

        # Pre-build col_lineups for all candidates (constant across cycles).
        self._col_lineups = self._build_col_lineups(candidates)  # (M, 10) int32

        # Payout lookup for N = n_field_lineups (field size stays fixed each cycle).
        payout_arr_f32 = np.ascontiguousarray(payout_arr.astype(np.float32))
        self._payout_lookup = _build_payout_lookup(
            payout_arr_f32, N=n_field_lineups, entry_fee=entry_fee
        )

        # Field generation components.
        self._cs = ContestSimulator()
        self._field_players_df = (
            field_players_df if field_players_df is not None else players_df
        )
        if ownership_vec is None:
            ownership_vec = compute_heuristic_ownership(players_df, team_totals)
        self._ownership_vec = ownership_vec
        self._field_ownership_vec = (
            compute_heuristic_ownership(self._field_players_df, team_totals)
            if field_players_df is not None
            else ownership_vec
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> list[tuple[Lineup, float]]:
        """Run the three-phase selection and return (lineup, ev) pairs."""
        t_start = time.perf_counter()
        M = len(self._candidates)
        ev_means = self._robust_payout.mean(axis=1)  # (M,) — mean net $ per sim

        # ---- Phase 1: cull to +EV ----------------------------------------
        pos_mask = ev_means > 0.0
        pool_indices = np.where(pos_mask)[0]  # indices into self._candidates
        pct_pos = float(pos_mask.mean() * 100)
        logger.info(
            "HybridField Phase 1 cull: %d / %d candidates are +EV (%.1f%%)",
            len(pool_indices), M, pct_pos,
        )

        if len(pool_indices) == 0:
            logger.warning(
                "No +EV candidates after Phase 1; returning empty portfolio."
            )
            return []

        # Sort pool by Phase-1 mean EV descending.
        pool_indices = pool_indices[np.argsort(ev_means[pool_indices])[::-1]]

        portfolio: list[tuple[Lineup, float]] = []

        # ---- Phase 2: best + zero-overlap fast-tracks --------------------
        best_idx = int(pool_indices[0])
        best_lu = self._candidates[best_idx]
        best_pids = frozenset(best_lu.player_ids)
        portfolio.append((best_lu, float(ev_means[best_idx])))

        # Fast-track set grows cumulatively within this cycle: each new addition
        # must be zero-overlap with all lineups added so far in this cycle.
        # This prevents the batch from filling the portfolio with 500 lineups
        # that don't overlap L* but overlap heavily with each other.
        fast_track_sets: list[frozenset] = [best_pids]

        remaining_indices: list[int] = []
        for idx in pool_indices[1:]:
            lu = self._candidates[int(idx)]
            pids = frozenset(lu.player_ids)
            if self._is_fast_trackable(pids, fast_track_sets):
                portfolio.append((lu, float(ev_means[idx])))
                fast_track_sets.append(pids)
            else:
                remaining_indices.append(int(idx))

        n_fast = len(portfolio) - 1  # zero-overlap fast-tracks added
        n_phase1_pos = int(len(pool_indices))
        logger.info(
            "HybridField Phase 2: +1 best, +%d zero-overlap fast-tracks = %d portfolio, "
            "%d remaining in pool",
            n_fast, len(portfolio), len(remaining_indices),
        )

        if progress_cb is not None:
            progress_cb({
                "portfolio_current": min(len(portfolio), self._portfolio_size),
                "portfolio_total": self._portfolio_size,
                "cycle": 0,
                "n_added": len(portfolio),
                "n_remaining": len(remaining_indices),
                "n_ev_survivors": n_phase1_pos,
                "cycle_wall_s": 0.0,
            })

        # ---- Phase 3+: hybrid-field cycles --------------------------------
        cycle = 0
        while len(portfolio) < self._portfolio_size and remaining_indices:
            if stop_check is not None and stop_check():
                logger.info("HybridField: stop requested at start of cycle %d.", cycle + 1)
                break

            cycle += 1
            n_portfolio = len(portfolio)
            n_opp = max(0, self._n_field - n_portfolio)

            _t_cycle = time.perf_counter()

            # 1. Fresh independent simulations.
            fresh_results = self._sim_engine.simulate(self._n_hybrid_sims)
            sim_mat = np.ascontiguousarray(
                fresh_results.results_matrix.astype(np.float32)
            )  # (n_hybrid_sims, n_players)

            # 2. Build hybrid field: n_opp opponents + portfolio lineups.
            if n_opp > 0:
                opp_lineups = self._cs.generate_field(
                    self._field_players_df,
                    self._field_ownership_vec,
                    n_lineups=n_opp,
                    rng_seed=int(self._rng.integers(0, 2**31)),
                )
                port_arr = np.array(
                    [list(lu.player_ids) for lu, _ in portfolio], dtype=np.int64
                )  # (n_portfolio, 10)
                hybrid_field = np.concatenate([opp_lineups, port_arr], axis=0)
            else:
                # Portfolio fills all field slots.
                hybrid_field = np.array(
                    [list(lu.player_ids) for lu, _ in portfolio], dtype=np.int64
                )

            # 3. Score hybrid field → sort for binary search.
            hybrid_scores = self._cs.score_field(
                hybrid_field, sim_mat, self._col_map
            )  # (n_hybrid_sims, n_field_actual)
            field_sorted = np.ascontiguousarray(
                np.sort(hybrid_scores, axis=1)
            )  # (n_hybrid_sims, n_field_actual) ascending

            # 4+5. Score candidates and compute mean net payout in batches.
            # Also compute Pearson correlation of each candidate's score vector
            # against each committed portfolio lineup's score vector, to cull
            # near-duplicates that the EV filter misses (a lineup with 9/10
            # overlapping players still looks +EV because the hybrid field only
            # displaces 1 of 5000 opponent slots).
            M_rem = len(remaining_indices)
            rem_col_lineups = self._col_lineups[remaining_indices]  # (M_rem, 10)
            invalid = (rem_col_lineups == -1).any(axis=1)
            safe_cols = np.where(rem_col_lineups == -1, 0, rem_col_lineups)

            # Portfolio score vectors for this cycle's fresh sims.
            # (n_hybrid_sims, n_portfolio) → normalise columns → used for correlation.
            port_col_lineups = np.array(
                [
                    [self._col_map.get(int(pid), 0) for pid in lu.player_ids]
                    for lu, _ in portfolio
                ],
                dtype=np.int32,
            )  # (n_portfolio, 10)
            port_scores = sim_mat[:, port_col_lineups].sum(axis=2)  # (n_hybrid_sims, n_portfolio)
            port_mean = port_scores.mean(axis=0)          # (n_portfolio,)
            port_std = port_scores.std(axis=0)
            port_std = np.where(port_std == 0, 1.0, port_std)
            port_norm = ((port_scores - port_mean) / port_std).astype(np.float32)  # (n_hybrid_sims, n_portfolio)

            hybrid_ev = np.empty(M_rem, dtype=np.float32)
            corr_culled = np.zeros(M_rem, dtype=bool)

            for start in range(0, M_rem, _BATCH):
                end = min(start + _BATCH, M_rem)
                # (n_hybrid_sims, BATCH, 10) → sum → (n_hybrid_sims, BATCH) → T
                batch_scores = (
                    sim_mat[:, safe_cols[start:end]].sum(axis=2).T.astype(np.float32)
                )  # (BATCH, n_hybrid_sims)
                if invalid[start:end].any():
                    batch_scores[invalid[start:end]] = 0.0

                # Pearson correlation: (BATCH, n_portfolio)
                b_mean = batch_scores.mean(axis=1, keepdims=True)
                b_std = batch_scores.std(axis=1, keepdims=True)
                b_std = np.where(b_std == 0, 1.0, b_std)
                b_norm = (batch_scores - b_mean) / b_std   # (BATCH, n_hybrid_sims)
                corr = (b_norm @ port_norm) / self._n_hybrid_sims  # (BATCH, n_portfolio)
                corr_culled[start:end] = corr.max(axis=1) > self._max_correlation

                batch_payout = _compute_payout_from_sorted_field(
                    np.ascontiguousarray(batch_scores),
                    field_sorted,
                    self._payout_lookup,
                )
                hybrid_ev[start:end] = batch_payout.mean(axis=1)

            if invalid.any():
                hybrid_ev[invalid] = np.finfo(np.float32).min

            # 6. Cull to +EV and low-correlation; re-sort.
            n_corr_culled = int(corr_culled.sum())
            surviving_mask = (hybrid_ev > 0.0) & ~corr_culled
            if not surviving_mask.any():
                logger.info(
                    "HybridField cycle %d: no +EV low-correlation candidates remain; stopping.", cycle
                )
                break

            # Build sorted list of (original_index, cycle_ev) descending by EV.
            surviving_order = np.argsort(hybrid_ev)[::-1]
            surviving: list[tuple[int, float]] = [
                (remaining_indices[i], float(hybrid_ev[i]))
                for i in surviving_order
                if surviving_mask[i]
            ]

            # 7. Add best + zero-overlap fast-tracks (cumulative within this cycle).
            best_orig_idx, best_cycle_ev = surviving[0]
            best_lu = self._candidates[best_orig_idx]
            best_pids = frozenset(best_lu.player_ids)
            portfolio.append((best_lu, best_cycle_ev))
            cycle_fast_sets: list[frozenset] = [best_pids]

            new_remaining: list[int] = []
            for orig_idx, _ev in surviving[1:]:
                lu = self._candidates[orig_idx]
                pids = frozenset(lu.player_ids)
                if self._is_fast_trackable(pids, cycle_fast_sets):
                    portfolio.append((lu, _ev))
                    cycle_fast_sets.append(pids)
                else:
                    new_remaining.append(orig_idx)

            remaining_indices = new_remaining

            _cycle_wall = time.perf_counter() - _t_cycle
            _n_added = len(portfolio) - n_portfolio
            logger.info(
                "HybridField cycle %d: +%d lineups → portfolio=%d, "
                "remaining=%d, +EV_survivors=%d, corr_culled=%d, cycle_wall=%.2fs",
                cycle, _n_added, len(portfolio),
                len(remaining_indices), len(surviving), n_corr_culled, _cycle_wall,
            )

            if progress_cb is not None:
                progress_cb({
                    "portfolio_current": min(len(portfolio), self._portfolio_size),
                    "portfolio_total": self._portfolio_size,
                    "cycle": cycle,
                    "n_added": _n_added,
                    "n_remaining": len(remaining_indices),
                    "n_ev_survivors": len(surviving),
                    "n_corr_culled": n_corr_culled,
                    "cycle_wall_s": round(_cycle_wall, 2),
                })

        result = portfolio[: self._portfolio_size]
        logger.info(
            "HybridField done: %d lineups in %.1fs (%d cycles)",
            len(result), time.perf_counter() - t_start, cycle,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_fast_trackable(
        self, pids: frozenset[int], existing_sets: list[frozenset[int]]
    ) -> bool:
        """True if pids qualifies for the zero-overlap fast-track vs every set in existing_sets.

        Rules (relaxed from strict 10-player disjoint):
          - 0 shared hitters  (all 8 hitter slots must differ)
          - ≤1 shared pitcher (at least 1 of the 2 pitchers must be unique)
        """
        p_hitters = pids - self._pitcher_ids
        p_pitchers = pids & self._pitcher_ids
        for other in existing_sets:
            if not p_hitters.isdisjoint(other - self._pitcher_ids):
                return False
            if len(p_pitchers & (other & self._pitcher_ids)) >= 2:
                return False
        return True

    def _build_col_lineups(self, candidates: list[Lineup]) -> np.ndarray:
        """Map each candidate's player_ids to sim_matrix column indices.

        Returns (M, 10) int32; -1 for any player_id not in sim_results.
        """
        M = len(candidates)
        col_lineups = np.full((M, 10), -1, dtype=np.int32)
        for i, lu in enumerate(candidates):
            for j, pid in enumerate(lu.player_ids):
                col_lineups[i, j] = self._col_map.get(int(pid), -1)
        return col_lineups
