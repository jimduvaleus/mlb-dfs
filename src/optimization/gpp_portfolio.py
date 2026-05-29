"""GPP portfolio construction via candidate scoring and simulation-coverage selection.

Two-stage pipeline for the `leverage_surplus` objective:

  ContestScorer   — scores each candidate against K simulated opponent fields,
                    returning a robust_payout matrix (M × n_sims) that averages
                    over field compositions to reduce overfitting.

  MeanVariancePortfolioSelector — assembles a portfolio via simulated annealing
                                   on the (M × n_sims) robust_payout matrix.
                                   Objective: mean(max_k payout_k) - alpha * std(max_k payout_k)
                                   where alpha = (10 - risk) / 10. Risk=0 (Shaidy default)
                                   maximises diversity; risk=10 maximises concentration.

  EVPortfolioSelector — legacy greedy coverage-based selector (preserved for reference).

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

def _build_payout_lookup(gross_payout_arr: np.ndarray, N: int, entry_fee: float = 4.0) -> np.ndarray:
    """Precompute a net payout lookup of size N+1 from the real gross payout array.

    Each entry lookup[lo] is the average GROSS payout over the fractionally-weighted
    band of real ranks corresponding to beating lo of N simulated field lineups,
    minus the entry fee.

    The prize pool scales naturally with the simulated field size: a 5,001-entry
    simulated contest drawn from a 14,863-entry real structure gets a prize pool of
    $50,000 × (5001/14863) ≈ $16,820 — the same economics as if DK ran that
    smaller contest with the same payout shape.

    Example (N=5_000, T=5_001, prize_pool=$16_800, entry_fee=$4):
      - Each bin covers exactly 1 real rank (T == N+1)
      - lookup[5000] (beat all)  = $2,000 - $4 = +$1,996 net
      - lookup[4100] (min cash)  = $8    - $4 = +$4.00 net
      - lookup[4099] (no cash)   = $0    - $4 = -$4.00 net
      - lookup[0]    (beat none) = $0    - $4 = -$4.00 net

    Parameters
    ----------
    gross_payout_arr : (T,) array — GROSS prize per rank (entry fee NOT subtracted)
    N : number of simulated field lineups; lookup will have N+1 entries (lo=0..N)
    entry_fee : $4 for a standard DK GPP; subtracted from each slot after scaling
    """
    T = len(gross_payout_arr)
    ga = gross_payout_arr.astype(np.float64)
    cum = np.empty(T + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(ga, out=cum[1:])

    band = T / (N + 1)
    gross_lookup = np.empty(N + 1, dtype=np.float64)

    for lo in range(N + 1):
        k = N - lo             # 0-indexed rank: k=0=best, k=N=worst
        r_lo = k * band
        r_hi = (k + 1) * band

        i0 = int(r_lo);  f0 = r_lo - i0
        i1 = int(r_hi);  f1 = r_hi - i1
        if i0 >= T: i0, f0 = T - 1, 0.0
        if i1 >= T: i1, f1 = T - 1, 0.0

        gross_lookup[lo] = (cum[i1] + f1 * ga[i1] - cum[i0] - f0 * ga[i0]) / band

    return (gross_lookup - entry_fee).astype(np.float32)


@njit(parallel=True, cache=True)
def _compute_payout_from_sorted_field(
    cand_scores_batch: np.ndarray,  # (BATCH, n_sims) float32, C-contiguous
    field_sorted: np.ndarray,       # (n_sims, N) float32, sorted along axis=1
    payout_lookup: np.ndarray,      # (N+1,) float32 — precomputed by _build_payout_lookup
) -> np.ndarray:                    # (BATCH, n_sims) float32
    """Compute dollar payout for each (candidate, sim) pair via precomputed lookup.

    For each candidate b and simulation s:
      lo      = number of simulated field lineups beaten (binary search, O(log N))
      payout  = payout_lookup[lo]   (O(1) table lookup)

    The lookup is indexed directly by lo so there is no floating-point arithmetic
    in the inner loop.  Aggregate preservation and edge cases are handled entirely
    in _build_payout_lookup at construction time.
    """
    BATCH = cand_scores_batch.shape[0]
    n_sims = cand_scores_batch.shape[1]
    N = field_sorted.shape[1]
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
            out[b, s] = payout_lookup[lo]
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
            structure = load_payout_structure("dk_classic_gpp_5001")
            payout_arr = payout_table_to_array(structure).astype(np.float32)
            self._entry_fee: float = float(structure.get("entry_fee", 4.0))
        else:
            # Caller is responsible for passing the GROSS payout array and setting
            # entry_fee via the ContestScorer.entry_fee attribute if needed.
            self._entry_fee = 4.0  # default for DK Classic GPP
        self._payout_arr = np.ascontiguousarray(payout_arr.astype(np.float32))
        self._payout_lookup = _build_payout_lookup(
            self._payout_arr, N=n_field_lineups, entry_fee=self._entry_fee
        )

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
                    cand_scores_batch, field_sorted, self._payout_lookup,
                )
                batch_payout += payout_k

            robust_payout[start:end] = batch_payout / self._n_k

            if progress_cb is not None:
                progress_cb(batch_idx + 1, n_batches)
            if stop_check is not None and stop_check():
                logger.info("ContestScorer: stop requested after batch %d/%d.", batch_idx + 1, n_batches)
                break

        logger.info("[TIMING] Numba scoring loop total: %.3fs", time.perf_counter() - _t_scoring)

        # Zero out robust_payout for any candidate whose col_lineups contains -1
        # (a player absent from sim_results). Without this, numpy's -1 index wraps
        # to the last column and inflates the candidate's score, potentially
        # causing it to be selected into the portfolio.
        invalid_mask = (col_lineups == -1).any(axis=1)
        if invalid_mask.any():
            robust_payout[invalid_mask] = 0.0
            logger.warning(
                "%d candidate(s) contain player_ids not in sim_results "
                "(loaded from cache for a different player pool?); zeroed out.",
                int(invalid_mask.sum()),
            )

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


# ------------------------------------------------------------------ #
#  MeanVariancePortfolioSelector                                       #
# ------------------------------------------------------------------ #

class MeanVariancePortfolioSelector:
    """Mean-variance portfolio selection via simulated annealing.

    Finds K lineups from M candidates maximising the Markowitz objective:
        f(S) = mean_s[sum_{k∈S} payout_k(s)] - alpha * std_s[sum_{k∈S} payout_k(s)]
    where alpha = (10 - risk) / 10.

    risk=0  → alpha=1.0 → maximum diversity (Shaidy's default)
    risk=10 → alpha=0.0 → pure mean-EV maximisation

    Under sum, mean[sum_k] = sum_k mean[k] and the benefit of diversification
    flows through std[sum], which decreases as pairwise lineup correlations fall.
    A single-swap proposal reduces to new_pp = portfolio_payout - old + new, so
    there is no top-2 tracking and no periodic rebuild.

    Parameters
    ----------
    robust_payout : (M, n_sims) float32 from ContestScorer.score_candidates()
    candidates : list of M Lineup objects (same order as robust_payout rows)
    portfolio_size : number of lineups to select
    risk : float 0–10; 0 = max diversity, 10 = max concentration
    n_iter : SA iterations per restart
    n_restarts : number of independent SA chains; best result returned
    holdout_fraction : fraction of sims held out for OOS evaluation
    rng_seed : base seed; restart r uses seed + r
    """

    def __init__(
        self,
        robust_payout: np.ndarray,
        candidates: list,
        portfolio_size: int = 10,
        risk: float = 0.0,
        n_iter: int = 10_000,
        n_restarts: int = 3,
        holdout_fraction: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._M = len(candidates)
        self._K = min(portfolio_size, self._M)
        self._candidates = candidates
        self._alpha = (10.0 - float(risk)) / 10.0
        self._n_iter = n_iter
        self._n_restarts = n_restarts
        self._rng_seed = rng_seed
        self._robust_payout = np.ascontiguousarray(robust_payout.astype(np.float32))

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

        n_train = len(self._train_idx)
        n_total = self._robust_payout.shape[1]
        if n_train == n_total:
            self._train_payout = self._robust_payout          # view, no copy
        else:
            self._train_payout = self._robust_payout[:, self._train_idx]

        self._selected_indices: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        progress_cb: Optional[Callable] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list:
        """Run SA portfolio selection.

        Parameters
        ----------
        progress_cb : callable(iteration, total_iterations, temperature, current_f, acceptance_rate)
        stop_check : callable() -> bool

        Returns
        -------
        list of (Lineup, mean_payout) tuples in selection order
        """
        best_selected: list[int] = []
        best_f = -np.inf
        total_iter = self._n_restarts * self._n_iter

        for restart_idx in range(self._n_restarts):
            if stop_check is not None and stop_check():
                break
            seed = (self._rng_seed + restart_idx) if self._rng_seed is not None else None
            rng = np.random.default_rng(seed)

            init = self._greedy_init(rng) if restart_idx == 0 else self._random_init(rng)
            selected, f = self._run_annealing(
                init, rng, progress_cb, stop_check, restart_idx, total_iter
            )
            if f > best_f:
                best_f = f
                best_selected = selected[:]

        self._selected_indices = best_selected
        return [
            (self._candidates[i], float(self._robust_payout[i].mean()))
            for i in best_selected
        ]

    def holdout_score(self) -> Optional[float]:
        """Mean of sum_k payout_k on holdout sims. None if no holdout split."""
        if len(self._holdout_idx) == 0 or not self._selected_indices:
            return None
        hp = self._robust_payout[self._selected_indices, :][:, self._holdout_idx].sum(axis=0)
        return float(hp.mean())

    def find_replacement(
        self,
        current_portfolio_indices: list[int],
        exclude_index: int,
        additional_excluded: Optional[set] = None,
    ) -> int:
        """Find the best replacement for a removed lineup.

        Uses the full robust_payout matrix (not the train split) so replacement
        quality is evaluated on all sims consistently with external reporting.

        Parameters
        ----------
        current_portfolio_indices : candidate indices of the remaining portfolio
            (should NOT include exclude_index)
        exclude_index : candidate index being removed
        additional_excluded : extra indices to skip (previously discarded lineups)

        Returns
        -------
        int : candidate index of the best replacement
        """
        remaining = [i for i in current_portfolio_indices if i != exclude_index]

        if len(remaining) == 0:
            means = self._robust_payout.mean(axis=1)
            return int(np.argmax(means))

        current_sum = self._robust_payout[remaining].sum(axis=0)  # (n_sims,)

        excluded_set: set[int] = set(current_portfolio_indices) | {exclude_index}
        if additional_excluded:
            excluded_set |= additional_excluded
        avail = np.array(
            [i for i in range(self._M) if i not in excluded_set], dtype=np.int32
        )
        if len(avail) == 0:
            return int(np.argmax(self._robust_payout.mean(axis=1)))

        # (n_avail, n_sims): portfolio payout after adding each candidate
        candidate_sum = current_sum[np.newaxis, :] + self._robust_payout[avail]
        means = candidate_sum.mean(axis=1)
        stds = candidate_sum.std(axis=1)
        f_vals = means - self._alpha * stds
        return int(avail[int(np.argmax(f_vals))])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_f(self, portfolio_payout: np.ndarray) -> float:
        m = float(portfolio_payout.mean())
        s = float(portfolio_payout.std())
        return m - self._alpha * s

    def _calibrate_temperature(
        self,
        selected: list[int],
        portfolio_payout: np.ndarray,
        rng: np.random.Generator,
        n_probes: int = 50,
    ) -> float:
        """Estimate T_start via random walk so initial acceptance rate ≈ 50%."""
        remaining = list(set(range(self._M)) - set(selected))
        if not remaining:
            return 1.0
        current_f = self._compute_f(portfolio_payout)
        deltas = []
        for _ in range(n_probes):
            old_local = int(rng.integers(0, self._K))
            old_global = selected[old_local]
            new_global = remaining[int(rng.integers(0, len(remaining)))]
            new_pp = portfolio_payout - self._train_payout[old_global] + self._train_payout[new_global]
            deltas.append(abs(self._compute_f(new_pp) - current_f))
        median_delta = float(np.median(deltas)) if deltas else 1.0
        return median_delta / np.log(2) if median_delta > 1e-12 else 1.0

    def _greedy_init(self, rng: np.random.Generator) -> list[int]:
        """Incremental greedy: repeatedly add the lineup maximising f(current_sum + payout_new).

        Each step is O(M × n_train) vectorised in batches of 500 to cap peak memory.
        Accounts for correlation: a lineup redundant with those already selected won't
        be picked even if its individual EV is high.
        """
        selected: list[int] = []
        remaining = list(range(self._M))
        n_train = self._train_payout.shape[1]
        portfolio_sum = np.zeros(n_train, dtype=np.float32)
        batch_size = 500

        for _ in range(self._K):
            if not remaining:
                break
            avail = np.array(remaining, dtype=np.int32)
            f_vals = np.empty(len(avail), dtype=np.float32)
            for b_start in range(0, len(avail), batch_size):
                batch = avail[b_start: b_start + batch_size]
                csum = portfolio_sum + self._train_payout[batch]   # (batch, n_train)
                f_vals[b_start: b_start + len(batch)] = (
                    csum.mean(axis=1) - self._alpha * csum.std(axis=1)
                )
            best_local = int(np.argmax(f_vals))
            best_global = int(avail[best_local])
            selected.append(best_global)
            portfolio_sum = portfolio_sum + self._train_payout[best_global]
            remaining.pop(best_local)

        return selected

    def _random_init(self, rng: np.random.Generator) -> list[int]:
        """Soft-weighted random init proportional to individual mean payout.

        Weights are shifted by the minimum so the best candidates still get
        more weight even when all individual means are negative (typical after
        entry-fee subtraction in a GPP with a high contest rake).
        """
        means = self._train_payout.mean(axis=1)
        weights = means - means.min() + 1e-9   # always positive, relative ordering preserved
        weights /= weights.sum()
        return list(rng.choice(self._M, size=self._K, replace=False, p=weights))

    def _run_annealing(
        self,
        init_selected: list[int],
        rng: np.random.Generator,
        progress_cb: Optional[Callable],
        stop_check: Optional[Callable[[], bool]],
        restart_idx: int,
        total_iter: int,
    ) -> tuple[list[int], float]:
        """One SA restart. Returns (best_selected, best_f)."""
        selected = list(init_selected)
        remaining = list(set(range(self._M)) - set(selected))

        # Portfolio sum payout vector: sum_k payout_k across all train sims.
        portfolio_payout = self._train_payout[selected].sum(axis=0).copy()
        current_f = self._compute_f(portfolio_payout)
        best_f = current_f
        best_selected = selected[:]

        if not remaining:
            return best_selected, best_f

        T_start = self._calibrate_temperature(selected, portfolio_payout, rng)

        acceptance_window: list[int] = []
        iter_offset = restart_idx * self._n_iter

        for iteration in range(self._n_iter):
            if stop_check is not None and stop_check():
                break

            T = T_start * (1.0 - iteration / self._n_iter)

            old_local = int(rng.integers(0, self._K))
            old_global = selected[old_local]
            new_local = int(rng.integers(0, len(remaining)))
            new_global = remaining[new_local]

            new_pp = portfolio_payout - self._train_payout[old_global] + self._train_payout[new_global]
            new_f = self._compute_f(new_pp)
            delta_f = new_f - current_f

            accepted = delta_f > 0
            if not accepted and T > 1e-12:
                accepted = rng.random() < np.exp(delta_f / T)

            acceptance_window.append(1 if accepted else 0)
            if len(acceptance_window) > 200:
                acceptance_window.pop(0)

            if accepted:
                portfolio_payout = new_pp
                current_f = new_f
                selected[old_local] = new_global
                remaining[new_local] = old_global

                if current_f > best_f:
                    best_f = current_f
                    best_selected = selected[:]

            if progress_cb is not None and iteration % 250 == 0:
                acc_rate = sum(acceptance_window) / len(acceptance_window) if acceptance_window else 0.0
                progress_cb(
                    iter_offset + iteration,
                    total_iter,
                    float(T),
                    float(current_f),
                    acc_rate,
                    float(portfolio_payout.mean()),
                    float(portfolio_payout.std()),
                    restart_idx,
                )

        return best_selected, best_f
