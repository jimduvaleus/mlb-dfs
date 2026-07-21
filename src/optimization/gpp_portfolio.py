"""GPP portfolio construction via candidate scoring and simulation-coverage selection.

  ContestScorer                — scores each candidate against K simulated opponent fields,
                                 returning a robust_payout matrix (M × n_sims).
  DeterminantPortfolioSelector — greedy portfolio assembly via incremental determinant
                                 maximisation, balancing EV and lineup diversity.

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

# Seed offset for the second-stage fresh field draws (rescore_fresh_fields).
# Must exceed any plausible n_field_samples so the fresh seeds never collide
# with the first-stage seeds field_rng_seed + 0..K-1.
_FRESH_FIELD_SEED_OFFSET = 100_003


# ------------------------------------------------------------------ #
#  Numba kernels (module-level, compiled once per process)            #
# ------------------------------------------------------------------ #

def _band_average(gross_payout_arr: np.ndarray, N: int) -> np.ndarray:
    """Band-average a per-rank gross payout array onto N+1 lookup slots.

    Each entry lookup[lo] is the average payout over the fractionally-weighted
    band of real ranks corresponding to beating lo of N simulated field lineups.

    The prize pool scales naturally with the simulated field size: a 5,001-entry
    simulated contest drawn from a 14,863-entry real structure gets a prize pool of
    $50,000 × (5001/14863) ≈ $16,820 — the same economics as if DK ran that
    smaller contest with the same payout shape.
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

    return gross_lookup


def _build_payout_lookup(gross_payout_arr: np.ndarray, N: int, entry_fee: float = 4.0) -> np.ndarray:
    """Precompute a net payout lookup of size N+1 from the real gross payout array.

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
    return (_band_average(gross_payout_arr, N) - entry_fee).astype(np.float32)


def _build_dilutable_lookup(
    gross_payout_arr: np.ndarray, N: int, min_gross_payout: float
) -> np.ndarray:
    """Band-average only the ranks whose gross prize is >= min_gross_payout.

    This is the portion of the payout the duplicate penalty dilutes: near the
    min-cash plateau the payout curve is flat, so tying with a duplicate barely
    changes the prize; at the steep top of the curve splitting is real money.
    The threshold is applied per REAL rank before banding so a lookup band that
    straddles the threshold is diluted only for its qualifying fraction.
    """
    dilutable = np.where(
        gross_payout_arr.astype(np.float64) >= float(min_gross_payout),
        gross_payout_arr.astype(np.float64), 0.0,
    )
    return _band_average(dilutable, N)


def _payout_cumsum(lookup: np.ndarray) -> np.ndarray:
    """(N+2,) float64 prefix sum of a (N+1,) lookup, cs[i] = sum(lookup[:i]).

    Lets the kernel average any contiguous slot range lo..hi in O(1):
    mean = (cs[hi+1] - cs[lo]) / (hi - lo + 1). float64 so 25k-entry sums do
    not lose cents to float32 cancellation.
    """
    cs = np.zeros(len(lookup) + 1, dtype=np.float64)
    np.cumsum(lookup.astype(np.float64), out=cs[1:])
    return cs


@njit(parallel=True, cache=True)
def _compute_payout_from_sorted_field(
    cand_scores_batch: np.ndarray,  # (BATCH, n_sims) float32, C-contiguous
    field_sorted: np.ndarray,       # (n_sims, N) float32, sorted along axis=1
    payout_cumsum: np.ndarray,      # (N+2,) float64 — _payout_cumsum(net lookup)
    dilute_cumsum: np.ndarray,      # (N+2,) float64 — _payout_cumsum(dilutable lookup)
    dupe_scale: np.ndarray,         # (BATCH,) float32 — 1/(1+E[dupes]); 1.0 = no penalty
) -> np.ndarray:                    # (BATCH, n_sims) float32
    """Compute dollar payout for each (candidate, sim) pair with exact tie splitting.

    For each candidate b and simulation s two binary searches bound the tie group:
      lo = number of field lineups strictly below the candidate's score
      hi = number of field lineups at or below it
    The candidate finishes anywhere in the (hi - lo + 1)-slot tie band with equal
    probability, so its payout is the band mean — an O(1) prefix-sum difference.
    With no ties (hi == lo) this reduces exactly to the old single-lookup path.

    The duplicate penalty subtracts (1 - dupe_scale[b]) of the dilutable (top-band
    gross) portion of the same slot range, i.e. top payouts scale by 1/(1+E[dupes]).
    """
    BATCH = cand_scores_batch.shape[0]
    n_sims = cand_scores_batch.shape[1]
    N = field_sorted.shape[1]
    out = np.zeros((BATCH, n_sims), dtype=np.float32)
    for b in prange(BATCH):
        dilution = 1.0 - np.float64(dupe_scale[b])
        for s in range(n_sims):
            score = cand_scores_batch[b, s]
            # lower bound: count of field scores strictly < score
            lo = 0
            hi = N
            while lo < hi:
                mid = (lo + hi) >> 1
                if field_sorted[s, mid] < score:
                    lo = mid + 1
                else:
                    hi = mid
            lo_idx = lo
            # upper bound: count of field scores <= score
            hi = N
            while lo < hi:
                mid = (lo + hi) >> 1
                if field_sorted[s, mid] <= score:
                    lo = mid + 1
                else:
                    hi = mid
            hi_idx = lo
            total = payout_cumsum[hi_idx + 1] - payout_cumsum[lo_idx]
            if dilution > 0.0:
                total -= dilution * (dilute_cumsum[hi_idx + 1] - dilute_cumsum[lo_idx])
            out[b, s] = total / (hi_idx - lo_idx + 1)
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
    field_ownership_vec : ownership for field_players_df (drives generate_field()).
                 If None, computed from field_players_df via compute_heuristic_ownership
                 (uncalibrated). Callers that have a fitted isotonic calibrator
                 (data/processed/ownership_calibrator.json) should pass the
                 calibrated vector here so the simulated opponent field reflects
                 the same magnitude-corrected ownership used elsewhere.
    team_totals : optional {team: implied_total} for Model D ownership
    candidate_batch_size : BATCH — candidates processed simultaneously (memory control)
    portfolio_size : unused, kept for API compatibility
    dupe_penalty : when True, each candidate's top-band payouts are diluted by
                 1/(1+E[dupes]) where E[dupes] comes from a log-linear model:
                 log E[dupes] = dupe_intercept
                                + dupe_log_own_coef * Σ log(ownership_i)
                                - dupe_salary_coef  * (unused salary / $100)
                                + dupe_stack_coef   * (primary stack size - 4)
                 Default coefficients were fitted by scripts/fit_dupe_model.py
                 (zero-truncated Poisson GLM on 32 archived DK contest
                 standings, 2026-07-04); the intercept is calibrated to the
                 reference 14,863-entry contest. Re-run the fit as the archive
                 grows. Dilution applies only to lookup bands whose gross prize
                 is >= dupe_min_gross_payout (flat near-min-cash bands are
                 unaffected by splitting).
    salary_cap : platform salary cap used for the unused-salary dupe feature
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
        field_ownership_vec: Optional[np.ndarray] = None,
        team_totals: Optional[dict] = None,
        candidate_batch_size: int = 500,
        portfolio_size: int = 0,
        cand_excluded_player_ids: Optional[set] = None,
        preloaded_field: Optional[list[np.ndarray]] = None,
        field_source: str = "simulated",
        historical_archive_root: Optional["Path"] = None,
        historical_n_slates: int = 10,
        exclude_slate_date: Optional[str] = None,
        dupe_penalty: bool = False,
        dupe_intercept: float = 3.698,
        dupe_log_own_coef: float = 0.212,
        dupe_salary_coef: float = 0.089,
        dupe_stack_coef: float = 0.024,
        dupe_min_gross_payout: float = 15.0,
        salary_cap: float = 50_000.0,
        compute_tail_metrics: bool = False,
        tail_ev_min_gross: float = 100.0,
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
        self.last_raw_field_list: Optional[list[np.ndarray]] = None
        # Per-sample sorted fields, retained so score_batch() can score
        # additional candidates against the exact same fields.
        self._field_sorted_list: Optional[list[np.ndarray]] = None

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
        self._payout_cumsum = _payout_cumsum(self._payout_lookup)

        self._dupe_penalty = bool(dupe_penalty)
        self._dupe_intercept = float(dupe_intercept)
        self._dupe_log_own_coef = float(dupe_log_own_coef)
        self._dupe_salary_coef = float(dupe_salary_coef)
        self._dupe_stack_coef = float(dupe_stack_coef)
        self._salary_cap = float(salary_cap)
        if self._dupe_penalty:
            self._dilute_cumsum = _payout_cumsum(_build_dilutable_lookup(
                self._payout_arr, N=n_field_lineups,
                min_gross_payout=dupe_min_gross_payout,
            ))
        else:
            # All-zero dilutable portion: the kernel's dilution term vanishes.
            self._dilute_cumsum = np.zeros_like(self._payout_cumsum)

        # Tail-metric state (ceiling-first redesign). tail-EV is the expected
        # GROSS dollars from the steep top payout bands only (ranks paying
        # >= tail_ev_min_gross) — a ranking currency for ceiling, deliberately
        # blind to the min-cash plateau that dominates mean EV. Reuses the
        # dilutable-lookup builder (same "zero below a gross threshold, then
        # band-average" transform) and the same tie-splitting kernel; the
        # tail band is passed as its own dilutable portion so dupe dilution
        # applies to it exactly as it does to mean EV's top bands.
        self._compute_tail_metrics = bool(compute_tail_metrics)
        self._tail_ev_min_gross = float(tail_ev_min_gross)
        if self._compute_tail_metrics:
            self._tail_cumsum = _payout_cumsum(_build_dilutable_lookup(
                self._payout_arr, N=n_field_lineups,
                min_gross_payout=self._tail_ev_min_gross,
            ).astype(np.float32))
        else:
            self._tail_cumsum = None
        # Populated by _score_col_lineups() when compute_tail_metrics is on,
        # aligned to that call's candidate order (mined after score_candidates,
        # fresh after rescore_fresh_fields).
        self.last_tail_ev: Optional[np.ndarray] = None
        self.last_p_beat99: Optional[np.ndarray] = None
        self.last_p_beat999: Optional[np.ndarray] = None
        # Round-10 coverage selector: when retain_beat999_worlds is set (the
        # pipeline flips it on just for the fresh re-score), _score_col_lineups
        # also keeps the per-world beat-p999 booleans, bit-packed to
        # (M, K × ceil(n_sims/8)) uint8 on last_beat999_bits — one bit per
        # (field sample, sim) world, ~n_K × M × n_sims/8 bytes.
        self.retain_beat999_worlds: bool = False
        self.last_beat999_bits: Optional[np.ndarray] = None
        # E[dupes] per candidate (log-linear model) — computed on every
        # score_candidates() call regardless of dupe_penalty, so the
        # win-equity currency (p_beat999 / (1+E[dupes])) is always dumpable.
        self.last_e_dupes: Optional[np.ndarray] = None

        self._sim_matrix = sim_results.results_matrix.astype(np.float32)
        self._col_map: dict[int, int] = {
            pid: i for i, pid in enumerate(sim_results.player_ids)
        }

        # NOTE: these None-fallbacks compute ownership from players_df means
        # as-is. The production pipeline always passes both vectors (computed
        # on restore_fitted_mean_scale'd means — see projection_calibration);
        # the fallbacks exist for tests/ad-hoc use with synthetic means.
        if ownership_vec is None:
            ownership_vec = compute_heuristic_ownership(players_df, team_totals)
        self._ownership_vec = ownership_vec

        # Field lineup generation uses the full (unfiltered) player pool so that
        # opponent field lineups reflect the real DFS player universe, not just
        # the players we chose to include in our own candidate pool.
        if field_players_df is not None:
            self._field_players_df = field_players_df
            self._field_ownership_vec = (
                field_ownership_vec if field_ownership_vec is not None
                else compute_heuristic_ownership(field_players_df, team_totals)
            )
        else:
            self._field_players_df = players_df
            self._field_ownership_vec = (
                field_ownership_vec if field_ownership_vec is not None else self._ownership_vec
            )

        self._cs = ContestSimulator()

        # Historical field mode ----------------------------------------
        self._field_source = field_source
        self._hist_distributions: list = []
        if field_source == "historical":
            if historical_archive_root is None:
                logger.warning(
                    "ContestScorer: field_source='historical' but no archive root "
                    "provided — falling back to simulated."
                )
                self._field_source = "simulated"
            else:
                from pathlib import Path as _Path
                from src.optimization.historical_field import load_historical_distributions
                self._hist_distributions = load_historical_distributions(
                    _Path(historical_archive_root),
                    n_slates=historical_n_slates,
                    exclude_date=exclude_slate_date,
                )
                if not self._hist_distributions:
                    logger.warning(
                        "ContestScorer: no historical distributions found in %s "
                        "(n_slates=%d, exclude=%s) — falling back to simulated.",
                        historical_archive_root, historical_n_slates, exclude_slate_date,
                    )
                    self._field_source = "simulated"
                else:
                    logger.info(
                        "ContestScorer: historical field mode — %d distributions loaded.",
                        len(self._hist_distributions),
                    )

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
        Sets self.last_raw_field_list for field caching.
        """
        n_sims = self._sim_matrix.shape[0]
        _t_phase = time.perf_counter()

        # --- Generate (or inject cached) field samples ---
        field_sorted_list: list[np.ndarray] = []
        if self._preloaded_field is not None and self._field_source == "historical":
            logger.warning(
                "ContestScorer: ignoring cached field lineups — field_source='historical' "
                "requires real score distributions, not model-generated lineup arrays."
            )
            self._preloaded_field = None
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
        elif self._field_source == "historical":
            from src.optimization.historical_field import (
                estimate_current_slate_ref,
                build_historical_field_samples,
            )
            logger.info(
                "Generating %d historical field samples (N=%d each, "
                "%d source distributions)...",
                self._n_k, self._n_field, len(self._hist_distributions),
            )
            _t_ref = time.perf_counter()
            current_ref = estimate_current_slate_ref(
                self._sim_matrix, self._field_ownership_vec,
            )
            logger.info(
                "  [TIMING] estimate_current_slate_ref: %.3fs  ref=%.2f",
                time.perf_counter() - _t_ref, current_ref,
            )
            rng_hist = np.random.default_rng(self._field_seed)
            _t_build = time.perf_counter()
            field_sorted_list = build_historical_field_samples(
                self._hist_distributions,
                n_field=self._n_field,
                n_sims=n_sims,
                current_ref=current_ref,
                rng=rng_hist,
                K=self._n_k,
            )
            logger.info(
                "  [TIMING] build_historical_field_samples: %.3fs",
                time.perf_counter() - _t_build,
            )
            # No raw lineup arrays in historical mode — field caching is inapplicable.
            self.last_raw_field_list = None
            if field_progress_cb is not None:
                field_progress_cb(self._n_k * self._n_field, self._n_k * self._n_field)
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

        # last_field_sorted (combined K-sample sorted pool) was used by the legacy
        # EVPortfolioSelector which is no longer active. Skip the concat+sort to
        # avoid a ~1.4 GB peak allocation (K*N_field float32) on top of the already
        # large _field_sorted_list and robust_payout.
        self.last_field_sorted = None
        self._field_sorted_list = field_sorted_list

        _t_col = time.perf_counter()
        M = len(candidates)
        col_lineups = self._build_col_lineups(candidates)  # (M, 10) int32
        logger.info("[TIMING] _build_col_lineups M=%d: %.3fs", M, time.perf_counter() - _t_col)

        _t_scoring = time.perf_counter()
        dupe_scale = self._compute_dupe_scale(candidates)
        robust_payout = self._score_col_lineups(col_lineups, dupe_scale, progress_cb, stop_check)
        logger.info("[TIMING] Numba scoring loop total: %.3fs", time.perf_counter() - _t_scoring)

        logger.info(
            "[TIMING] score_candidates total: %.3fs (field=%.3fs, scoring=%.3fs)",
            time.perf_counter() - _t_phase,
            time.perf_counter() - _t_phase - (time.perf_counter() - _t_scoring),
            time.perf_counter() - _t_scoring,
        )
        return candidates, robust_payout

    def score_batch(
        self,
        candidates: list[Lineup],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Score additional candidates against the fields built by score_candidates().

        Used by candidate-pool refinement: mutants must be scored against the
        exact same K field samples as the original pool so their robust_payout
        rows are directly comparable.

        Returns robust_payout of shape (len(candidates), n_sims) float32.
        """
        if self._field_sorted_list is None:
            raise RuntimeError("score_batch() requires a prior score_candidates() call.")

        col_lineups = self._build_col_lineups(candidates)
        dupe_scale = self._compute_dupe_scale(candidates)
        return self._score_col_lineups(col_lineups, dupe_scale, progress_cb, stop_check)

    def rescore_fresh_fields(
        self,
        candidates: list[Lineup],
        n_samples: int,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        field_progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Re-score candidates against n_samples freshly drawn opponent fields.

        Second scoring stage against fields never seen during pool mining
        (generation + refinement), so a candidate whose first-stage EV was
        luck against the particular K field draws regresses back to its true
        EV here (winner's-curse control). Fresh draws use a seed range
        disjoint from the first-stage seeds.

        Side effect: the fresh fields replace the cached first-stage fields,
        so any later score_batch() call scores against the same fresh fields
        the selector consumed.
        """
        n_sims = self._sim_matrix.shape[0]
        _t0 = time.perf_counter()
        fresh_seed = self._field_seed + _FRESH_FIELD_SEED_OFFSET
        # Release the stage-1 fields up front: at production scale each sorted
        # field is ~n_sims × N_field × 4 bytes, so holding both generations
        # simultaneously would roughly double the field memory peak.
        self._field_sorted_list = None
        field_sorted_list: list[np.ndarray] = []
        if self._field_source == "historical":
            from src.optimization.historical_field import (
                estimate_current_slate_ref,
                build_historical_field_samples,
            )
            current_ref = estimate_current_slate_ref(
                self._sim_matrix, self._field_ownership_vec,
            )
            field_sorted_list = build_historical_field_samples(
                self._hist_distributions,
                n_field=self._n_field,
                n_sims=n_sims,
                current_ref=current_ref,
                rng=np.random.default_rng(fresh_seed),
                K=n_samples,
            )
            if field_progress_cb is not None:
                field_progress_cb(n_samples * self._n_field, n_samples * self._n_field)
        else:
            logger.info(
                "Generating %d fresh field samples (N=%d each) for re-scoring...",
                n_samples, self._n_field,
            )
            n_total_field = n_samples * self._n_field
            for k in range(n_samples):
                offset = k * self._n_field
                def _field_cb(n_done: int, _n: int, _offset: int = offset) -> None:
                    if field_progress_cb is not None:
                        field_progress_cb(_offset + n_done, n_total_field)
                raw = self._cs.generate_field(
                    self._field_players_df, self._field_ownership_vec,
                    n_lineups=self._n_field, rng_seed=fresh_seed + k,
                    progress_cb=_field_cb if field_progress_cb is not None else None,
                )
                field_sorted_list.append(self._build_field_sorted(raw))
                if field_progress_cb is not None:
                    field_progress_cb(offset + len(raw), n_total_field)

        self._field_sorted_list = field_sorted_list
        self._n_k = len(field_sorted_list)
        logger.info(
            "[TIMING] Fresh field phase (K=%d): %.3fs",
            self._n_k, time.perf_counter() - _t0,
        )

        col_lineups = self._build_col_lineups(candidates)
        dupe_scale = self._compute_dupe_scale(candidates)
        return self._score_col_lineups(col_lineups, dupe_scale, progress_cb, stop_check)

    def _score_col_lineups(
        self,
        col_lineups: np.ndarray,
        dupe_scale: np.ndarray,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Score (M, 10) column lineups against the cached K sorted fields."""
        n_sims = self._sim_matrix.shape[0]
        field_sorted_list = self._field_sorted_list
        M = col_lineups.shape[0]
        robust_payout = np.zeros((M, n_sims), dtype=np.float32)
        # Tail metrics are reduced (M,) accumulators — never a second
        # (M, n_sims) matrix, which would double the scoring memory peak.
        tail_ev = np.zeros(M, dtype=np.float64) if self._compute_tail_metrics else None
        p_beat99 = np.zeros(M, dtype=np.float64) if self._compute_tail_metrics else None
        p_beat999 = np.zeros(M, dtype=np.float64) if self._compute_tail_metrics else None
        if self._compute_tail_metrics:
            # kth field lineup such that beating it means beating >= 99% of N.
            _p99_col = int(np.ceil(0.99 * self._n_field)) - 1
            # ... and >= 99.9% of N (the real-contest money zone: ~30% of
            # prize mass sits above p99.9).
            _p999_col = int(np.ceil(0.999 * self._n_field)) - 1
        beat_bits = None
        if self._compute_tail_metrics and self.retain_beat999_worlds:
            _n_bytes = (n_sims + 7) // 8
            beat_bits = np.zeros((M, len(field_sorted_list) * _n_bytes), dtype=np.uint8)
        n_batches = (M + self._batch_size - 1) // self._batch_size
        logger.info(
            "Scoring %d candidates in %d batches of %d...",
            M, n_batches, self._batch_size,
        )

        for batch_idx, start in enumerate(range(0, M, self._batch_size)):
            end = min(start + self._batch_size, M)
            batch_cols = col_lineups[start:end]  # (batch, 10)

            # Score this batch against the sim matrix: (batch, n_sims) float32
            cand_scores_batch = (
                self._sim_matrix[:, batch_cols].sum(axis=2).T.astype(np.float32)
            )  # (batch, n_sims)

            # Accumulate payout over K field samples
            batch_payout = np.zeros((end - start, n_sims), dtype=np.float32)
            batch_dupe_scale = np.ascontiguousarray(dupe_scale[start:end])
            for _field_k, field_sorted in enumerate(field_sorted_list):
                payout_k = _compute_payout_from_sorted_field(
                    cand_scores_batch, field_sorted,
                    self._payout_cumsum, self._dilute_cumsum, batch_dupe_scale,
                )
                batch_payout += payout_k
                if self._compute_tail_metrics:
                    # Same kernel over the tail-band-only lookup; the tail band
                    # doubles as its own dilutable portion so E[dupes] dilutes
                    # it exactly as it dilutes mean EV's top bands.
                    tail_k = _compute_payout_from_sorted_field(
                        cand_scores_batch, field_sorted,
                        self._tail_cumsum, self._tail_cumsum, batch_dupe_scale,
                    )
                    tail_ev[start:end] += tail_k.mean(axis=1)
                    thr = field_sorted[:, _p99_col]  # (n_sims,)
                    p_beat99[start:end] += (cand_scores_batch >= thr[None, :]).mean(axis=1)
                    thr999 = field_sorted[:, _p999_col]  # (n_sims,)
                    _beat999 = cand_scores_batch >= thr999[None, :]
                    p_beat999[start:end] += _beat999.mean(axis=1)
                    if beat_bits is not None:
                        beat_bits[start:end, _field_k * _n_bytes:(_field_k + 1) * _n_bytes] = (
                            np.packbits(_beat999, axis=1)
                        )

            robust_payout[start:end] = batch_payout / self._n_k

            if progress_cb is not None:
                progress_cb(batch_idx + 1, n_batches)
            if stop_check is not None and stop_check():
                logger.info("ContestScorer: stop requested after batch %d/%d.", batch_idx + 1, n_batches)
                break

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

        if self._compute_tail_metrics:
            tail_ev /= self._n_k
            p_beat99 /= self._n_k
            p_beat999 /= self._n_k
            if invalid_mask.any():
                tail_ev[invalid_mask] = 0.0
                p_beat99[invalid_mask] = 0.0
                p_beat999[invalid_mask] = 0.0
            self.last_tail_ev = tail_ev
            self.last_p_beat99 = p_beat99
            self.last_p_beat999 = p_beat999
        if beat_bits is not None:
            if invalid_mask.any():
                beat_bits[invalid_mask] = 0
            self.last_beat999_bits = beat_bits

        return robust_payout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_dupe_scale(self, candidates: list[Lineup]) -> np.ndarray:
        """(M,) float32 top-band payout multipliers 1/(1+E[dupes]) per candidate.

        E[dupes] is the log-linear model documented on the class. Ownership
        comes from the field vector (the real-universe %drafted estimate that
        drives opponent field generation), not the candidate pool's internal
        sampling weights. E[dupes] is always computed and stored on
        self.last_e_dupes (the win-equity currency needs it); the returned
        payout scale is all ones when the penalty is disabled.
        """
        M = len(candidates)
        fdf = self._field_players_df
        pids = fdf["player_id"].astype(int).to_numpy()
        own_map = dict(zip(pids, np.asarray(self._field_ownership_vec, dtype=np.float64)))
        sal_map = dict(zip(pids, fdf["salary"].astype(float).to_numpy()))
        team_map = dict(zip(pids, fdf["team"]))
        pos_map = dict(zip(pids, fdf["position"]))

        scale = np.ones(M, dtype=np.float32)
        e_dupes_all = np.zeros(M, dtype=np.float64)
        for i, lu in enumerate(candidates):
            sum_log_own = 0.0
            salary_used = 0.0
            team_counts: dict[str, int] = {}
            for pid in lu.player_ids:
                pid = int(pid)
                own = float(np.clip(own_map.get(pid, 0.01), 1e-4, 0.95))
                sum_log_own += np.log(own)
                salary_used += sal_map.get(pid, 0.0)
                if pos_map.get(pid, "") != "P":
                    t = team_map.get(pid, "")
                    if t:
                        team_counts[t] = team_counts.get(t, 0) + 1
            primary_stack = max(team_counts.values()) if team_counts else 0
            unused_hundreds = max(self._salary_cap - salary_used, 0.0) / 100.0
            log_dupes = (
                self._dupe_intercept
                + self._dupe_log_own_coef * sum_log_own
                - self._dupe_salary_coef * unused_hundreds
                + self._dupe_stack_coef * (primary_stack - 4)
            )
            e_dupes = float(np.exp(np.clip(log_dupes, -20.0, 10.0)))
            e_dupes_all[i] = e_dupes
            scale[i] = np.float32(1.0 / (1.0 + e_dupes))

        logger.info(
            "Dupe model: E[dupes] min=%.3f  p50=%.3f  p90=%.3f  max=%.1f  "
            "(top-band payout scale p50=%.3f, penalty %s)",
            float(e_dupes_all.min()), float(np.percentile(e_dupes_all, 50)),
            float(np.percentile(e_dupes_all, 90)), float(e_dupes_all.max()),
            float(np.percentile(scale, 50)),
            "on" if self._dupe_penalty else "off",
        )
        self.last_e_dupes = e_dupes_all
        if not self._dupe_penalty:
            return np.ones(M, dtype=np.float32)
        return scale

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
#  DeterminantPortfolioSelector                                        #
# ------------------------------------------------------------------ #

class DeterminantPortfolioSelector:
    """Greedy portfolio construction via incremental correlation-distance maximisation.

    At each step scores every remaining +EV candidate on a weighted combination of:
      EVn — normalised mean dollar payout (robust_payout.mean(axis=1))
      Dn  — normalised correlation distance between the candidate's payout and
            the running portfolio's aggregate payout (the sum of the payout
            vectors of the lineups already selected)

    For standardised (zero-mean, unit-variance) random variables X, Y:
      E[(X-Y)^2] = 2(1 - corr(X,Y))
    so (1 - r) / 2 is the natural, normalised [0, 1] distance implied by a
    correlation — monotonically *decreasing* in r, not in |r|. A candidate
    anti-correlated with the portfolio (r -> -1) is strictly farther
    (Dn -> 1) than an uncorrelated one (r = 0, Dn = 0.5), which is strictly
    farther than a co-correlated one (r -> 1, Dn -> 0). This is deliberately
    different from a partial-variance/Schur-complement formulation (1 - r^2),
    which is symmetric in the sign of r and treats a hedge as no better than
    an equally-sized co-movement.

    Generalising from a single reference lineup to a k-lineup running
    portfolio: let P_k = sum of the k already-selected (standardised) payout
    vectors. The candidate's correlation with the portfolio's aggregate
    payout is
      r_c = Cov(x_c, P_k) / sqrt(Var(P_k)) = (sum_i r_{c,i}) / sqrt(sum_{i,j} r_{i,j})
    (sums over the selected set i, j; each x is standardised so Var(x_c) = 1).
    Both the numerator and Var(P_k) update in O(k) per step as lineups are
    added — no matrix inverse, no Schur complement.

    No fresh simulations, no hybrid fields, no correlation thresholds —
    the Phase-1 robust_payout matrix is the only input required.

    Parameters
    ----------
    robust_payout : (M, n_sims) float32 — net dollar payout per candidate per sim
    candidates    : list of M Lineup objects (same ordering as robust_payout rows)
    portfolio_size : number of lineups to select
    risk          : 1..5; EVw is linearly interpolated between evw_base (risk=1) and
                    evw_max (risk=5)
    evw_base      : EVw at risk=1 (most diversity-heavy)
    evw_max       : EVw at risk=5 (most EV-heavy)
    ev_floor      : candidates with mean robust_payout below this $ amount are
                    culled from the pool before selection begins
    """

    def __init__(
        self,
        robust_payout: np.ndarray,
        candidates: list[Lineup],
        portfolio_size: int,
        risk: float = 5.0,
        evw_base: float = 0.10,
        evw_max: float = 0.40,
        ev_floor: float = 0.20,
        rng_seed: Optional[int] = None,  # unused; kept for API consistency
        precomputed: Optional[tuple] = None,
        ev_override: Optional[np.ndarray] = None,
        cash_anchor_fraction: float = 0.0,
    ) -> None:
        # robust_payout may be None when `precomputed` is supplied (external
        # pool mode passes (pool_idx, ROI vector, corr) directly): the greedy
        # loop runs entirely off the precompute tuple, and pool_ev_vals is
        # then whatever EV currency the caller chose (ev_floor is unused).
        self._robust_payout = (
            None if robust_payout is None else np.asarray(robust_payout, dtype=np.float32)
        )
        self._candidates = candidates
        self._portfolio_size = portfolio_size
        self._evw = self.evw_for_risk(risk, evw_base, evw_max)
        self._dew = 1.0 - self._evw
        self._ev_floor = ev_floor
        # Optional shared precompute from precompute_pool() — must have been
        # built from the same robust_payout/ev_floor. The risk sweep passes
        # one precompute to all five selectors instead of repeating the
        # M×M correlation matmul per risk.
        self._precomputed = precomputed
        # Ceiling-first redesign (Phase 3): when ev_override is given (length
        # = len(candidates), aligned with robust_payout rows), the greedy
        # score's EV term ranks by it (e.g. fresh tail-EV) instead of mean
        # dollar EV. The floor cull, the diversity/determinant term, and the
        # reported per-lineup EVs stay on mean dollar EV. The first
        # ceil(cash_anchor_fraction × portfolio_size) picks keep the mean-EV
        # basis — a cash-anchor block preserving mean EV's proven cash-band
        # signal inside an otherwise tail-ranked portfolio.
        self._ev_override = (
            np.asarray(ev_override, dtype=np.float64) if ev_override is not None else None
        )
        self._cash_anchor_fraction = float(cash_anchor_fraction)

    @staticmethod
    def evw_for_risk(risk: float, evw_base: float = 0.10, evw_max: float = 0.40) -> float:
        """EVw = evw_base at risk=1, evw_max at risk=5; linear in between.

        Single source of truth for the EVw/DEw split — also used by callers
        (e.g. PipelineRunner's risk-sweep logging) that need the weight
        without constructing a selector.
        """
        t = (risk - 1) / 4.0
        return float(np.clip(evw_base + t * (evw_max - evw_base), 0.0, 1.0))

    @staticmethod
    def precompute_pool(
        robust_payout: np.ndarray, ev_floor: float,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Floor cull + payout correlation matrix — the risk-independent part
        of select(). A risk sweep over the same robust_payout/ev_floor should
        compute this once and hand it to every selector: the M×M matmul is
        the dominant selection cost (~23s at pool=11k) and is identical for
        every risk, since risk only changes the EVw/DEw blend.

        Returns (pool_idx, pool_ev_vals float64, corr_matrix float32), or
        None when no candidate clears the floor.
        """
        robust_payout = np.asarray(robust_payout, dtype=np.float32)
        n_sims = robust_payout.shape[1]
        pool_ev = robust_payout.mean(axis=1)
        pool_idx = np.where(pool_ev >= ev_floor)[0]
        if len(pool_idx) == 0:
            return None
        pool_ev_vals = pool_ev[pool_idx].astype(np.float64)

        # Memory budget: pool_payout (M×S×8) → norm_payout reuses the same
        # buffer → matmul peak is norm_payout(M×S×8) + result(M×M×8). /=
        # avoids a second M×M allocation.
        _t = time.perf_counter()
        pool_payout = robust_payout[pool_idx].astype(np.float64)  # (M_pool, n_sims)
        mu = pool_payout.mean(axis=1, keepdims=True)
        std = pool_payout.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        norm_payout = (pool_payout - mu) / std  # float64; skip float32 round-trip
        del pool_payout, mu, std

        corr_matrix = norm_payout @ norm_payout.T  # single float64 copy during matmul
        corr_matrix /= n_sims                       # in-place: no second M×M allocation
        corr_matrix = corr_matrix.astype(np.float32)
        del norm_payout

        logger.info(
            "[TIMING] DeterminantSelector precompute: %.2fs  corr_matrix %s (%.1f MB)",
            time.perf_counter() - _t, corr_matrix.shape, corr_matrix.nbytes / 1e6,
        )
        return pool_idx, pool_ev_vals, corr_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> list[tuple[Lineup, float]]:
        """Run greedy Det-EV selection and return (lineup, mean_ev) pairs."""
        t0 = time.perf_counter()

        # --- Floor cull + correlation precompute (shared across a risk sweep
        # when the caller passed precompute_pool() output) ---
        pre = self._precomputed
        if pre is None:
            if self._robust_payout is None:
                raise ValueError(
                    "robust_payout is required when precomputed is not supplied"
                )
            pre = self.precompute_pool(self._robust_payout, self._ev_floor)
        if pre is None:
            logger.warning("No candidates with EV >= $%.2f; returning empty portfolio.", self._ev_floor)
            return []
        pool_idx, pool_ev_vals, corr_matrix = pre

        logger.info(
            "DeterminantSelector: %d / %d candidates have EV >= $%.2f",
            len(pool_idx), len(self._candidates), self._ev_floor,
        )

        M_pool = len(pool_idx)
        evw, dew = self._evw, self._dew

        # Selection-EV basis per pick: mean dollar EV for the cash-anchor
        # block (and everywhere when no override), the override vector after.
        if self._ev_override is not None:
            override_vals = self._ev_override[pool_idx].astype(np.float64)
            n_anchor = int(np.ceil(self._cash_anchor_fraction * self._portfolio_size))
        else:
            override_vals = None
            n_anchor = self._portfolio_size

        def _sel_ev_vals(step: int) -> np.ndarray:
            return pool_ev_vals if (override_vals is None or step <= n_anchor) else override_vals

        # --- Greedy selection ---
        selected_in_pool: list[int] = []    # indices into pool (0..M_pool-1)
        remaining_mask = np.ones(M_pool, dtype=bool)

        # Step 1: highest EV on the step-1 basis
        first = int(np.argmax(_sel_ev_vals(1)))
        selected_in_pool.append(first)
        remaining_mask[first] = False
        # Var(P_1) in standardised units — a single lineup's self-correlation.
        portfolio_var = 1.0

        if progress_cb is not None:
            progress_cb({
                "step": 1,
                "portfolio_size": self._portfolio_size,
                "lineup_ev": float(pool_ev_vals[first]),
                "distance": 1.0,
                "score": 1.0,
                "n_remaining": int(remaining_mask.sum()),
            })

        # Steps 2…portfolio_size
        for step in range(2, self._portfolio_size + 1):
            if stop_check is not None and stop_check():
                logger.info("DeterminantSelector: stop requested at step %d.", step)
                break

            remaining_pool_idx = np.where(remaining_mask)[0]  # (M_rem,)
            M_rem = len(remaining_pool_idx)
            if M_rem == 0:
                break

            # Correlation of each remaining candidate with each selected lineup
            R = corr_matrix[np.ix_(remaining_pool_idx, selected_in_pool)].astype(np.float64)  # (M_rem, k)

            # Correlation-distance to the running portfolio's aggregate payout
            # P_k = sum of the selected (standardised) payout vectors. Since
            # each is unit-variance, Cov(x_c, P_k) = sum_i r_{c,i} and
            # Var(P_k) = sum_{i,j} r_{i,j} = portfolio_var (maintained
            # incrementally below), so r_c = Cov / sqrt(Var(P_k)) directly —
            # no matrix inverse needed.
            cov_to_portfolio = R.sum(axis=1)                         # (M_rem,)
            r_to_portfolio = cov_to_portfolio / np.sqrt(portfolio_var)
            r_to_portfolio = np.clip(r_to_portfolio, -1.0, 1.0)       # fp-noise guard
            distance = (1.0 - r_to_portfolio) / 2.0                   # (M_rem,) in [0, 1]

            # Normalised EV (relative to current remaining pool).
            # Shift up by |min_ev| when negatives are present so EVn stays in
            # [0, 1]; shift is normalization-only and does not affect stored EVs.
            ev_rem = _sel_ev_vals(step)[remaining_pool_idx]
            min_ev = ev_rem.min()
            shift = -min_ev if min_ev < 0.0 else 0.0
            ev_shifted = ev_rem + shift
            max_ev_shifted = ev_shifted.max()
            EVn = ev_shifted / max_ev_shifted if max_ev_shifted > 1e-12 else np.ones(M_rem)

            # distance is already a bounded, self-normalised [0, 1] quantity
            # (unlike partial_var, it needs no per-step max-rescaling).
            DEn = distance

            score = evw * EVn + dew * DEn
            best_in_rem = int(np.argmax(score))
            best_pool = int(remaining_pool_idx[best_in_rem])

            selected_in_pool.append(best_pool)
            remaining_mask[best_pool] = False

            # Var(P_{k+1}) = Var(P_k) + Var(x_new) + 2*Cov(P_k, x_new)
            #              = portfolio_var + 1 + 2*cov_to_portfolio[picked]
            portfolio_var += 1.0 + 2.0 * float(cov_to_portfolio[best_in_rem])
            portfolio_var = max(portfolio_var, 1e-10)  # clamp for numerical stability

            logger.debug(
                "Det-EV step %d: pool_idx=%d  ev=$%.4f  distance=%.4f  score=%.4f",
                step, pool_idx[best_pool],
                float(pool_ev_vals[best_pool]), float(distance[best_in_rem]),
                float(score[best_in_rem]),
            )

            if progress_cb is not None:
                progress_cb({
                    "step": step,
                    "portfolio_size": self._portfolio_size,
                    "lineup_ev": float(pool_ev_vals[best_pool]),
                    "distance": float(distance[best_in_rem]),
                    "score": float(score[best_in_rem]),
                    "n_remaining": int(remaining_mask.sum()),
                })

        result = [
            (self._candidates[pool_idx[i]], float(pool_ev_vals[i]))
            for i in selected_in_pool
        ]
        logger.info(
            "DeterminantSelector done: %d lineups in %.1fs",
            len(result), time.perf_counter() - t0,
        )
        return result


@njit(parallel=True, cache=True)
def _kelly_gains_all(payout: np.ndarray, denom: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Marginal expected-log-growth of each candidate against the current
    portfolio: mean_w[log1p(payout[m, w] / denom[w])] for unmasked rows.
    denom[w] = bankroll + portfolio payout in world w (always > 0)."""
    M, n_sims = payout.shape
    out = np.full(M, -np.inf)
    for m in prange(M):
        if not mask[m]:
            continue
        s = 0.0
        for w in range(n_sims):
            p = payout[m, w]
            if p != 0.0:
                s += np.log1p(p / denom[w])
        out[m] = s / n_sims
    return out


@njit(cache=True)
def _kelly_gain_one(pay_row: np.ndarray, denom: np.ndarray) -> float:
    n_sims = pay_row.shape[0]
    s = 0.0
    for w in range(n_sims):
        p = pay_row[w]
        if p != 0.0:
            s += np.log1p(p / denom[w])
    return s / n_sims


class KellyPortfolioSelector:
    """Round-10 selector: greedy expected-log-growth (fractional Kelly).

    Objective: maximize E_w[log(B + portfolio_payout_w)] over sim worlds w,
    exactly, on the same fresh robust_payout matrix the Det selector
    consumes. Concavity makes marginal gains shrink in worlds the portfolio
    already covers — world-partitioning derived from the objective rather
    than proxied by a covariance determinant. Marginal gains are also
    monotonically non-increasing as the portfolio grows, so lazy greedy
    (stale gains as upper bounds, re-evaluate the heap top) is exact.

    The first ceil(cash_anchor_fraction × portfolio_size) picks are pure
    mean-EV rank (the cash-anchor block, mirroring the det path's anchor
    semantics), with the anchors' payouts entering the portfolio state
    before the Kelly picks begin.

    B must exceed the maximum possible portfolio loss (fee × size) or the
    all-entries-lose world sends log to a domain error; the caller enforces
    the bankroll table's mult > 1.
    """

    def __init__(
        self,
        robust_payout: np.ndarray,
        candidates: list[Lineup],
        portfolio_size: int,
        bankroll: float,
        ev_floor: float = 0.20,
        cash_anchor_fraction: float = 0.0,
    ) -> None:
        self._robust_payout = np.asarray(robust_payout, dtype=np.float32)
        self._candidates = candidates
        self._portfolio_size = portfolio_size
        self._bankroll = float(bankroll)
        self._ev_floor = float(ev_floor)
        self._cash_anchor_fraction = float(cash_anchor_fraction)

    def select(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> list[tuple[Lineup, float]]:
        t0 = time.perf_counter()
        pool_ev = self._robust_payout.mean(axis=1).astype(np.float64)
        pool_idx = np.where(pool_ev >= self._ev_floor)[0]
        if len(pool_idx) == 0:
            logger.warning("KellySelector: no candidates with EV >= $%.2f.", self._ev_floor)
            return []
        # float32 rows with float64 accumulators/denominator: the log1p sums
        # are float64 either way, and the float64 copy would be ~3 GB at
        # production pool sizes.
        payout = np.ascontiguousarray(self._robust_payout[pool_idx])  # (M_pool, n_sims) f32
        ev_vals = pool_ev[pool_idx]
        M_pool, n_sims = payout.shape
        size = min(self._portfolio_size, M_pool)
        n_anchor = int(np.ceil(self._cash_anchor_fraction * self._portfolio_size))

        denom = np.full(n_sims, self._bankroll, dtype=np.float64)
        selected: list[int] = []
        remaining = np.ones(M_pool, dtype=bool)

        # Cash-anchor block: pure mean-EV picks; their payouts still update
        # the portfolio state so the Kelly picks respond to them.
        anchor_order = np.argsort(ev_vals)[::-1]
        for i in anchor_order[: min(n_anchor, size)]:
            selected.append(int(i))
            remaining[int(i)] = False
            denom += payout[int(i)]

        # Kelly picks, lazy greedy.
        gains = _kelly_gains_all(payout, denom, remaining)
        stale = np.zeros(M_pool, dtype=bool)
        while len(selected) < size:
            if stop_check is not None and stop_check():
                logger.info("KellySelector: stop requested at step %d.", len(selected) + 1)
                break
            while True:
                best = int(np.argmax(gains))
                if gains[best] == -np.inf:
                    break
                if stale[best]:
                    gains[best] = _kelly_gain_one(payout[best], denom)
                    stale[best] = False
                    continue
                break
            if gains[best] == -np.inf:
                break
            selected.append(best)
            remaining[best] = False
            denom += payout[best]
            gains[best] = -np.inf
            stale[remaining] = True
            if progress_cb is not None:
                progress_cb({
                    "step": len(selected),
                    "portfolio_size": size,
                    "lineup_ev": float(ev_vals[best]),
                    "n_remaining": int(remaining.sum()),
                })

        result = [(self._candidates[pool_idx[i]], float(ev_vals[i])) for i in selected]
        logger.info(
            "KellySelector done: %d lineups (anchor=%d, B=$%.0f) in %.1fs",
            len(result), min(n_anchor, size), self._bankroll, time.perf_counter() - t0,
        )
        return result


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class CoveragePortfolioSelector:
    """Round-10 selector: greedy weighted max-coverage on tail worlds.

    Each candidate carries a bit-vector over worlds (sim × fresh-field
    pairs): 1 where the candidate beats the simulated field's p99.9. Greedy
    picks the candidate covering the most uncovered worlds (submodular ⇒
    (1−1/e) guarantee); ties — including the everything-covered endgame —
    break by the tie_break vector (win_equity), then mean EV. The first
    ceil(cash_anchor_fraction × portfolio_size) picks are pure mean-EV rank
    (cash-anchor block); worlds they cover count as covered.
    """

    def __init__(
        self,
        robust_payout: np.ndarray,
        candidates: list[Lineup],
        portfolio_size: int,
        beat999_bits: np.ndarray,      # (M, n_bytes) uint8, packed world bits
        tie_break: Optional[np.ndarray] = None,   # (M,) higher = preferred
        ev_floor: float = 0.20,
        cash_anchor_fraction: float = 0.0,
    ) -> None:
        self._robust_payout = np.asarray(robust_payout, dtype=np.float32)
        self._candidates = candidates
        self._portfolio_size = portfolio_size
        self._bits = np.asarray(beat999_bits, dtype=np.uint8)
        self._tie_break = tie_break
        self._ev_floor = float(ev_floor)
        self._cash_anchor_fraction = float(cash_anchor_fraction)

    def select(
        self,
        stop_check: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
    ) -> list[tuple[Lineup, float]]:
        t0 = time.perf_counter()
        pool_ev = self._robust_payout.mean(axis=1).astype(np.float64)
        pool_idx = np.where(pool_ev >= self._ev_floor)[0]
        if len(pool_idx) == 0:
            logger.warning("CoverageSelector: no candidates with EV >= $%.2f.", self._ev_floor)
            return []
        bits = self._bits[pool_idx]                    # (M_pool, n_bytes)
        ev_vals = pool_ev[pool_idx]
        if self._tie_break is not None:
            tb = np.asarray(self._tie_break, dtype=np.float64)[pool_idx]
        else:
            tb = ev_vals
        M_pool = len(pool_idx)
        size = min(self._portfolio_size, M_pool)
        n_anchor = int(np.ceil(self._cash_anchor_fraction * self._portfolio_size))

        uncovered = np.full(bits.shape[1], 0xFF, dtype=np.uint8)
        selected: list[int] = []
        remaining = np.ones(M_pool, dtype=bool)

        def _cover(i: int) -> None:
            np.bitwise_and(uncovered, np.bitwise_not(bits[i]), out=uncovered)

        anchor_order = np.argsort(ev_vals)[::-1]
        for i in anchor_order[: min(n_anchor, size)]:
            selected.append(int(i))
            remaining[int(i)] = False
            _cover(int(i))

        # Tie-break rank: primary tb, secondary mean EV — encode as a single
        # dense rank so np.lexsort runs once, not per pick.
        tb_rank = np.lexsort((ev_vals, tb))            # ascending; higher = better
        tb_score = np.empty(M_pool, dtype=np.float64)
        tb_score[tb_rank] = np.arange(M_pool, dtype=np.float64)

        while len(selected) < size:
            if stop_check is not None and stop_check():
                logger.info("CoverageSelector: stop requested at step %d.", len(selected) + 1)
                break
            new_bits = np.bitwise_and(bits, uncovered[None, :])
            gains = _POPCOUNT_LUT[new_bits].sum(axis=1).astype(np.float64)  # (M_pool,)
            gains[~remaining] = -1.0
            # Ties (incl. the all-covered endgame where every gain is 0)
            # break by tb_score, which is < 1 in units of whole worlds.
            best = int(np.argmax(gains + tb_score / M_pool))
            if gains[best] < 0:
                break
            selected.append(best)
            remaining[best] = False
            _cover(best)
            if progress_cb is not None:
                progress_cb({
                    "step": len(selected),
                    "portfolio_size": size,
                    "lineup_ev": float(ev_vals[best]),
                    "worlds_gained": int(gains[best]),
                    "n_remaining": int(remaining.sum()),
                })

        result = [(self._candidates[pool_idx[i]], float(ev_vals[i])) for i in selected]
        logger.info(
            "CoverageSelector done: %d lineups (anchor=%d, %d world-bytes) in %.1fs",
            len(result), min(n_anchor, size), bits.shape[1], time.perf_counter() - t0,
        )
        return result
