"""GPP portfolio construction via candidate scoring and simulation-coverage selection.

Two-stage pipeline for the `leverage_surplus` objective:

  ContestScorer   — scores each candidate against K simulated opponent fields,
                    returning a robust_payout matrix (M × n_sims) that averages
                    over field compositions to reduce overfitting.

  EVPortfolioSelector — assembles a portfolio by simulation-coverage selection:
                         lineup #1 is the highest-EV candidate across all sims;
                         each subsequent lineup is the highest-EV candidate on
                         the subset of sims not yet "covered" by prior lineups.
                         A sim is covered when the selected lineup beats ≥
                         coverage_percentile fraction of the field in that sim.

Performance notes
-----------------
The Numba kernel _compute_payout_from_sorted_field is a module-level
@njit(parallel=True, cache=True) function following the same pattern as
optimizer.py. Compiled artifacts land in __pycache__; first-run JIT adds ~5–15 s.
"""
import logging
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from numba import njit, prange

from src.optimization.contest import ContestSimulator
from src.optimization.lineup import Lineup
from src.optimization.ownership import compute_heuristic_ownership
from src.simulation.results import SimulationResults

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Numba kernels (module-level, compiled once per process)            #
# ------------------------------------------------------------------ #

@njit(parallel=True, cache=True)
def _compute_payout_from_sorted_field(
    cand_scores_batch: np.ndarray,  # (BATCH, n_sims) float32, C-contiguous
    field_sorted: np.ndarray,       # (n_sims, N) float32, sorted along axis=1
    payout_arr: np.ndarray,         # (total_entries,) float32 — payout_arr[rank-1] = $ for that rank
) -> np.ndarray:                    # (BATCH, n_sims) float32
    """Compute dollar payout for each (candidate, sim) pair.

    For each candidate b and simulation s:
      lo          = number of simulated field lineups beaten (binary search)
      real_rank   = (N - lo) * total_entries // N + 1   (scales sim field → real contest)
      payout      = payout_arr[real_rank - 1]  (0 for ranks beyond the payout table)
    """
    BATCH = cand_scores_batch.shape[0]
    n_sims = cand_scores_batch.shape[1]
    N = field_sorted.shape[1]
    T = len(payout_arr)
    out = np.zeros((BATCH, n_sims), dtype=np.float32)
    for b in prange(BATCH):
        for s in range(n_sims):
            score = cand_scores_batch[b, s]
            lo = 0
            hi = N
            while lo < hi:
                mid = (lo + hi) >> 1
                if field_sorted[s, mid] <= score:
                    lo = mid + 1
                else:
                    hi = mid
            # Map simulated percentile to real contest rank (1-indexed).
            # rank=1 when lo=N (beat entire field); rank=T+1 when lo=0 (beat nobody).
            real_rank = (N - lo) * T // N + 1
            if real_rank <= T:
                out[b, s] = payout_arr[real_rank - 1]
    return out




# ------------------------------------------------------------------ #
#  ContestScorer                                                       #
# ------------------------------------------------------------------ #

class ContestScorer:
    """Score a candidate pool against K simulated opponent fields.

    Returns robust_payout (M, n_sims) float32 where each cell is the mean
    dollar payout across K independent field samples, using the real contest
    payout table to convert simulated percentile to expected dollars.

    Parameters
    ----------
    sim_results : SimulationResults
    players_df : must include player_id, position, salary, team, game, mean
    n_field_lineups : N — field lineups per sample
    n_field_samples : K — independent field draws (anti-overfitting)
    payout_arr : (total_entries,) float32 — payout_arr[rank-1] = dollar payout for that rank.
                 If None, loaded from data/payout_structures/dk_classic_gpp.json.
    field_rng_seed : base seed; sample k uses seed field_rng_seed + k
    ownership_vec : if None, computed from players_df via compute_heuristic_ownership
    team_totals : optional {team: implied_total} for Model D ownership
    candidate_batch_size : BATCH — candidates processed simultaneously (memory control)
    portfolio_size : unused, kept for API compatibility
    """

    def __init__(
        self,
        sim_results: SimulationResults,
        players_df: pd.DataFrame,
        field_players_df: Optional[pd.DataFrame] = None,
        n_field_lineups: int = 5_000,
        n_field_samples: int = 3,
        payout_arr: Optional[np.ndarray] = None,
        field_rng_seed: int = 42,
        ownership_vec: Optional[np.ndarray] = None,
        team_totals: Optional[dict] = None,
        candidate_batch_size: int = 500,
        portfolio_size: int = 0,
        cand_excluded_player_ids: Optional[set] = None,
        preloaded_field: Optional[list[np.ndarray]] = None,
    ) -> None:
        self._sim_results = sim_results
        self._players_df = players_df
        self._n_field = n_field_lineups
        self._n_k = n_field_samples
        self._field_seed = field_rng_seed
        self._batch_size = candidate_batch_size
        self._portfolio_size = portfolio_size
        self._cand_excluded_pids: set[int] = (
            {int(p) for p in cand_excluded_player_ids} if cand_excluded_player_ids else set()
        )
        self._preloaded_field = preloaded_field
        # Populated after score_candidates() for use by the coverage selector.
        self.last_field_sorted: Optional[np.ndarray] = None
        self.last_col_lineups: Optional[np.ndarray] = None
        self.last_raw_field_list: Optional[list[np.ndarray]] = None

        if payout_arr is None:
            from src.optimization.payout import load_payout_structure, payout_table_to_array
            structure = load_payout_structure("dk_classic_gpp")
            payout_arr = payout_table_to_array(structure).astype(np.float32)
        self._payout_arr = np.ascontiguousarray(payout_arr.astype(np.float32))

        self._sim_matrix = sim_results.results_matrix.astype(np.float32)
        self._col_map: dict[int, int] = {
            pid: i for i, pid in enumerate(sim_results.player_ids)
        }

        if ownership_vec is None:
            ownership_vec = compute_heuristic_ownership(players_df, team_totals)
        self._ownership_vec = ownership_vec

        # Field lineup generation uses the full (unfiltered) player pool so that
        # opponent field lineups reflect the real DFS player universe, not just
        # the players we chose to include in our own candidate pool.
        if field_players_df is not None:
            self._field_players_df = field_players_df
            self._field_ownership_vec = compute_heuristic_ownership(field_players_df, team_totals)
        else:
            self._field_players_df = players_df
            self._field_ownership_vec = self._ownership_vec

        self._cs = ContestSimulator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidates: list[Lineup],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        field_progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[list[Lineup], np.ndarray]:
        """Compute robust_payout for all candidates.

        Parameters
        ----------
        candidates : list of M Lineup objects
        progress_cb : optional callable(batches_done, total_batches)
        field_progress_cb : optional callable(n_done, n_total) — total is K × N_field

        Returns
        -------
        tuple of (candidates, robust_payout), robust_payout shape (M, n_sims) float32.

        Side effects
        ------------
        Sets self.last_field_sorted (n_sims × N_field float32, field sample 0)
        and self.last_col_lineups (M × 10 int32) for use by EVPortfolioSelector.
        """
        n_sims = self._sim_matrix.shape[0]
        _t_phase = time.perf_counter()

        # --- Generate (or inject cached) field samples ---
        field_sorted_list: list[np.ndarray] = []
        if self._preloaded_field is not None:
            logger.info(
                "Using %d preloaded field samples from cache.", len(self._preloaded_field)
            )
            for _ki, raw in enumerate(self._preloaded_field):
                _t_fs = time.perf_counter()
                field_sorted_list.append(self._build_field_sorted(raw))
                logger.info(
                    "  [TIMING] _build_field_sorted sample %d/%d (cached): %.3fs",
                    _ki + 1, len(self._preloaded_field), time.perf_counter() - _t_fs,
                )
            self._n_k = len(self._preloaded_field)
            self._n_field = self._preloaded_field[0].shape[0] if self._preloaded_field else self._n_field
        else:
            logger.info(
                "Generating %d field samples (N=%d each)...",
                self._n_k, self._n_field,
            )
            raw_list: list[np.ndarray] = []
            n_total_field = self._n_k * self._n_field
            for k in range(self._n_k):
                seed = self._field_seed + k
                offset = k * self._n_field
                def _field_cb(n_done: int, _n: int, _offset: int = offset) -> None:
                    if field_progress_cb is not None:
                        field_progress_cb(_offset + n_done, n_total_field)
                _t_gen = time.perf_counter()
                raw = self._cs.generate_field(
                    self._field_players_df, self._field_ownership_vec,
                    n_lineups=self._n_field, rng_seed=seed,
                    progress_cb=_field_cb if field_progress_cb is not None else None,
                )
                logger.info(
                    "  [TIMING] generate_field %d/%d: %.3fs (%d lineups)",
                    k + 1, self._n_k, time.perf_counter() - _t_gen, raw.shape[0],
                )
                raw_list.append(raw)
                _t_fs = time.perf_counter()
                field_sorted_list.append(self._build_field_sorted(raw))
                logger.info(
                    "  [TIMING] _build_field_sorted %d/%d (fresh): %.3fs",
                    k + 1, self._n_k, time.perf_counter() - _t_fs,
                )
                if field_progress_cb is not None:
                    field_progress_cb(offset + len(raw), n_total_field)
            self.last_raw_field_list = raw_list

        logger.info("[TIMING] Total field phase: %.3fs", time.perf_counter() - _t_phase)

        # Combine all K field samples into one sorted pool for coverage computation.
        # Beat rate = fraction of K×N field lineups beaten, consistent with how
        # robust_payout averages payout across the same K samples.
        _t_concat = time.perf_counter()
        self.last_field_sorted = np.sort(
            np.concatenate(field_sorted_list, axis=1), axis=1
        )  # (n_sims, K * N_field)
        logger.info(
            "[TIMING] concat+sort field_sorted %s: %.3fs",
            self.last_field_sorted.shape, time.perf_counter() - _t_concat,
        )

        _t_col = time.perf_counter()
        M = len(candidates)
        col_lineups = self._build_col_lineups(candidates)  # (M, 10) int32
        self.last_col_lineups = col_lineups
        logger.info("[TIMING] _build_col_lineups M=%d: %.3fs", M, time.perf_counter() - _t_col)

        # --- Score the (possibly enriched) candidate pool against all K fields ---
        robust_payout = np.zeros((M, n_sims), dtype=np.float32)
        n_batches = (M + self._batch_size - 1) // self._batch_size
        logger.info(
            "Scoring %d candidates in %d batches of %d...",
            M, n_batches, self._batch_size,
        )

        _t_scoring = time.perf_counter()
        for batch_idx, start in enumerate(range(0, M, self._batch_size)):
            end = min(start + self._batch_size, M)
            batch_cols = col_lineups[start:end]  # (batch, 10)

            # Score this batch against the sim matrix: (batch, n_sims) float32
            cand_scores_batch = (
                self._sim_matrix[:, batch_cols].sum(axis=2).T.astype(np.float32)
            )  # (batch, n_sims)

            # Accumulate payout over K field samples
            batch_payout = np.zeros((end - start, n_sims), dtype=np.float32)
            for field_sorted in field_sorted_list:
                payout_k = _compute_payout_from_sorted_field(
                    cand_scores_batch, field_sorted, self._payout_arr,
                )
                batch_payout += payout_k

            robust_payout[start:end] = batch_payout / self._n_k

            if progress_cb is not None:
                progress_cb(batch_idx + 1, n_batches)
            if stop_check is not None and stop_check():
                logger.info("ContestScorer: stop requested after batch %d/%d.", batch_idx + 1, n_batches)
                break

        logger.info("[TIMING] Numba scoring loop total: %.3fs", time.perf_counter() - _t_scoring)
        logger.info(
            "[TIMING] score_candidates total: %.3fs (field=%.3fs, scoring=%.3fs)",
            time.perf_counter() - _t_phase,
            time.perf_counter() - _t_phase - (time.perf_counter() - _t_scoring),
            time.perf_counter() - _t_scoring,
        )
        return candidates, robust_payout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_col_lineups(self, candidates: list[Lineup]) -> np.ndarray:
        """Convert candidate player_ids to sim_matrix column indices.

        Invalid candidates (player_id not in sim_results) get column -1 for
        that slot; scoring will produce 0 for those slots, which is safe.
        """
        M = len(candidates)
        col_lineups = np.zeros((M, 10), dtype=np.int32)
        for i, lu in enumerate(candidates):
            for j, pid in enumerate(lu.player_ids):
                col_lineups[i, j] = self._col_map.get(int(pid), -1)
                if col_lineups[i, j] == -1:
                    logger.warning(
                        "Candidate %d: player_id %d not in sim_results; score = 0",
                        i, pid,
                    )
        return col_lineups

    def _build_field_sorted(self, field_lineups: np.ndarray) -> np.ndarray:
        """Score field lineups and sort per simulation for binary search.

        Returns (n_sims, N_valid) float32, sorted along axis=1 (ascending).
        """
        field_scores = self._cs.score_field(
            field_lineups, self._sim_matrix, self._col_map,
        )  # (n_sims, N_valid) float32
        return np.ascontiguousarray(np.sort(field_scores, axis=1))



# ------------------------------------------------------------------ #
#  EVPortfolioSelector                                                 #
# ------------------------------------------------------------------ #

class EVPortfolioSelector:
    """Simulation-coverage portfolio selection from a pre-scored candidate pool.

    Round 0: pick the candidate with the highest mean EV across all sims.
    Round k: pick the candidate with the highest mean EV on sims not yet
             "covered" by any prior lineup.

    A sim is covered when the selected lineup beats ≥ coverage_percentile
    fraction of the field in that sim, as reported by beat_rate_fn.

    Parameters
    ----------
    robust_payout : (M, n_sims) float32 from ContestScorer.score_candidates()
    candidates : list of M Lineup objects (same order as robust_payout rows)
    portfolio_size : number of lineups to select
    holdout_fraction : fraction of sims held out for OOS evaluation (0 = disabled)
    rng_seed : used only to split train/holdout sims reproducibly
    """

    def __init__(
        self,
        robust_payout: np.ndarray,
        candidates: list[Lineup],
        portfolio_size: int = 10,
        holdout_fraction: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._M = len(candidates)
        self._candidates = candidates
        self._portfolio_size = min(portfolio_size, self._M)
        self._robust_payout = np.ascontiguousarray(
            robust_payout.astype(np.float32)
        )  # (M, n_sims)

        n_sims = robust_payout.shape[1]
        if holdout_fraction > 0.0:
            rng = np.random.default_rng(rng_seed)
            perm = rng.permutation(n_sims)
            n_holdout = max(1, int(n_sims * holdout_fraction))
            self._train_idx = np.sort(perm[n_holdout:]).astype(np.int64)
            self._holdout_idx = np.sort(perm[:n_holdout]).astype(np.int64)
        else:
            self._train_idx = np.arange(n_sims, dtype=np.int64)
            self._holdout_idx = np.array([], dtype=np.int64)

        self._selected_indices: list[int] = []
        self._best_payout_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        progress_cb: Optional[Callable[[int, int, float, int, float], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        beat_rate_fn: Optional[Callable[[int], np.ndarray]] = None,
        coverage_percentile: float = 0.5,
    ) -> list[tuple[Lineup, float]]:
        """Run simulation-coverage selection.

        Parameters
        ----------
        progress_cb : optional callable(round_k, candidate_index, lineup_ev,
                      n_covered, pct_covered)
        beat_rate_fn : callable(candidate_idx) -> (n_sims,) float32 in [0, 1].
                       Returns fraction of field lineups beaten per sim for the
                       given candidate. Required for coverage tracking; if None,
                       all sims remain active every round (first-lineup-only coverage).
        coverage_percentile : sim is marked covered when beat_rate >= this value.

        Returns
        -------
        list of (Lineup, lineup_ev) tuples in selection order, where lineup_ev
        is the mean robust_payout on uncovered sims at the time of selection.
        """
        # Avoid a redundant copy of robust_payout when there is no holdout split.
        # Fancy indexing with np.arange(n) always materialises a full copy; skip it
        # when train_idx covers the whole matrix.
        n_sims_total = self._robust_payout.shape[1]
        if len(self._train_idx) == n_sims_total:
            train_payout = self._robust_payout          # view — no copy
        else:
            train_payout = self._robust_payout[:, self._train_idx]  # (M, n_train)
        n_train = train_payout.shape[1]
        logger.info(
            "[TIMING] EVPortfolioSelector.select() start — train_payout shape %s (%.1f MB)",
            train_payout.shape, train_payout.nbytes / 1e6,
        )

        selected: list[tuple[Lineup, float]] = []
        selected_pid_sets: list[frozenset] = []
        remaining = np.arange(self._M, dtype=np.int32)

        # active_mask tracks which train-split sims are not yet covered.
        active_mask = np.ones(n_train, dtype=bool)
        n_covered = 0

        # Incremental EV: precompute the per-candidate sum over all train sims once,
        # then subtract covered-sim contributions as coverage accrues.  Each round
        # becomes O(M) instead of O(M × n_active), avoiding repeated 1800 MB copies.
        _t_setup = time.perf_counter()
        _ev_sum_all = train_payout.sum(axis=1).astype(np.float64)  # (M,) immutable baseline
        _ev_sum = _ev_sum_all.copy()                                # (M,) mutable running sum
        _n_active = float(n_train)
        logger.info("[TIMING] ev_sum precompute: %.3fs", time.perf_counter() - _t_setup)

        _t_select_start = time.perf_counter()
        _t_ev_total = 0.0
        _t_beat_total = 0.0

        for k in range(self._portfolio_size):
            if len(remaining) == 0:
                break

            # EV per round: O(M) — just index the running sum vector.
            _t_ev = time.perf_counter()
            n_active = int(_n_active)
            if _n_active > 0:
                ev = (_ev_sum[remaining] / _n_active).astype(np.float32)
            else:
                # All sims covered: fall back to global mean as tiebreaker.
                ev = (_ev_sum_all[remaining] / float(n_train)).astype(np.float32)
            _dt_ev = time.perf_counter() - _t_ev
            _t_ev_total += _dt_ev

            # Walk candidates in descending EV order, skipping duplicates.
            order = np.argsort(ev)[::-1]
            lineup = None
            best_local = -1
            best_global = -1
            lineup_ev = 0.0
            for rank_pos in order:
                cand_local = int(rank_pos)
                cand_global = int(remaining[cand_local])
                cand = self._candidates[cand_global]
                pid_set = frozenset(cand.player_ids)
                if pid_set in selected_pid_sets:
                    logger.warning(
                        "Round %d: candidate %d is a duplicate; skipping.", k, cand_global
                    )
                    remaining = np.delete(remaining, cand_local)
                    # Shift remaining order indices after deletion.
                    order = order[order != rank_pos]
                    order = np.where(order > cand_local, order - 1, order)
                    continue
                lineup = cand
                best_local = cand_local
                best_global = cand_global
                lineup_ev = float(ev[rank_pos])
                break

            if lineup is None or len(remaining) == 0:
                break

            selected.append((lineup, lineup_ev))
            selected_pid_sets.append(frozenset(lineup.player_ids))
            self._selected_indices.append(best_global)
            remaining = np.delete(remaining, best_local)

            # Mark sims covered by this lineup and update running EV sum.
            _t_beat = time.perf_counter()
            if beat_rate_fn is not None:
                beat_rates = beat_rate_fn(best_global)  # (n_sims,) float32
                # beat_rates is over the full sim set; index into train split.
                beat_rates_train = beat_rates[self._train_idx]
                newly_covered = active_mask & (beat_rates_train >= coverage_percentile)
                if newly_covered.any():
                    # Subtract newly covered sims from running sum: O(M × n_newly_covered)
                    # rather than recomputing the full O(M × n_active) slice next round.
                    covered_idx = np.where(newly_covered)[0]
                    _ev_sum -= train_payout[:, covered_idx].sum(axis=1).astype(np.float64)
                    active_mask[newly_covered] = False
                    _n_active = float(int(active_mask.sum()))
                n_covered = int((~active_mask).sum())
            _dt_beat = time.perf_counter() - _t_beat
            _t_beat_total += _dt_beat

            pct_covered = n_covered / n_train * 100
            logger.info(
                "  Round %d: lineup_ev=%.4f, covered=%d/%d (%.1f%%) | "
                "ev_compute=%.3fs (n_active=%d), beat_rate=%.3fs",
                k, lineup_ev, n_covered, n_train, pct_covered,
                _dt_ev, n_active, _dt_beat,
            )

            if progress_cb is not None:
                progress_cb(k, best_global, lineup_ev, n_covered, pct_covered)
            if stop_check is not None and stop_check():
                logger.info("EVPortfolioSelector: stop requested after round %d.", k)
                break

        _t_select_total = time.perf_counter() - _t_select_start
        logger.info(
            "[TIMING] EVPortfolioSelector.select() done — total=%.3fs, "
            "ev_compute_total=%.3fs, beat_rate_total=%.3fs, other=%.3fs",
            _t_select_total, _t_ev_total, _t_beat_total,
            _t_select_total - _t_ev_total - _t_beat_total,
        )

        self._best_payout_train = np.zeros(n_train, dtype=np.float32)
        return selected

    def holdout_score(self) -> Optional[float]:
        """Mean payout on held-out sims for the selected portfolio.

        Returns None if holdout_fraction was 0 or select() has not been called.
        """
        if len(self._holdout_idx) == 0 or not self._selected_indices:
            return None
        holdout_payout = self._robust_payout[self._selected_indices[0], self._holdout_idx]
        for idx in self._selected_indices[1:]:
            holdout_payout = np.maximum(
                holdout_payout,
                self._robust_payout[idx, self._holdout_idx],
            )
        return float(holdout_payout.mean())
