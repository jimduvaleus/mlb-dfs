"""GPP portfolio construction via candidate scoring and marginal-EV selection.

Two-stage pipeline for the `leverage_surplus` objective:

  ContestScorer   — scores each candidate against K simulated opponent fields,
                    returning a robust_payout matrix (M × n_sims) that averages
                    over field compositions to reduce overfitting.

  EVPortfolioSelector — greedily assembles a portfolio by iterative marginal-EV
                         maximization: each lineup added is the one that contributes
                         the most *new* expected payout above what prior lineups
                         already capture.

Performance notes
-----------------
The Numba kernels (_compute_payout_from_sorted_field, _compute_marginal_ev) are
module-level @njit(parallel=True, cache=True) functions following the same pattern
as optimizer.py. Compiled artifacts land in __pycache__; first-run JIT adds ~5–15 s.
"""
import logging
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


@njit(parallel=True, cache=True)
def _compute_marginal_ev(
    robust_payout: np.ndarray,  # (M, n_sims) float32, C-contiguous
    best_payout: np.ndarray,    # (n_sims,) float32
    remaining: np.ndarray,      # (M_rem,) int32 — indices into robust_payout rows
) -> np.ndarray:                # (M_rem,) float64
    """Compute per-candidate marginal EV above an existing portfolio.

    marginal_ev[i] = mean_s(max(0, robust_payout[remaining[i], s] - best_payout[s]))
    """
    n_rem = remaining.shape[0]
    n_sims = best_payout.shape[0]
    out = np.zeros(n_rem, dtype=np.float64)
    for i in prange(n_rem):
        c = remaining[i]
        total = 0.0
        for s in range(n_sims):
            diff = robust_payout[c, s] - best_payout[s]
            if diff > 0.0:
                total += diff
        out[i] = total / n_sims
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
    portfolio_size : used to determine N for field lineup injection (0 = disabled)
    inject_field_lineups : if True and portfolio_size > 0, score field sample 0 lineups
                           and promote any that crack the top portfolio_size EV into
                           the candidate pool, replacing the bottom performers
    """

    def __init__(
        self,
        sim_results: SimulationResults,
        players_df: pd.DataFrame,
        n_field_lineups: int = 5_000,
        n_field_samples: int = 3,
        payout_arr: Optional[np.ndarray] = None,
        field_rng_seed: int = 42,
        ownership_vec: Optional[np.ndarray] = None,
        team_totals: Optional[dict] = None,
        candidate_batch_size: int = 500,
        portfolio_size: int = 0,
        inject_field_lineups: bool = True,
    ) -> None:
        self._sim_results = sim_results
        self._players_df = players_df
        self._n_field = n_field_lineups
        self._n_k = n_field_samples
        self._field_seed = field_rng_seed
        self._batch_size = candidate_batch_size
        self._portfolio_size = portfolio_size
        self._inject_field_lineups = inject_field_lineups
        self.last_injected_count: int = 0

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

        self._cs = ContestSimulator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidates: list[Lineup],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> tuple[list[Lineup], np.ndarray]:
        """Compute robust_payout for all candidates.

        If inject_field_lineups is enabled, scores field sample 0 lineups
        alongside the original candidates and promotes any that crack the top
        portfolio_size EV into the pool (replacing the bottom performers).
        The enriched pool is then scored against all K field samples normally.

        Parameters
        ----------
        candidates : list of M Lineup objects
        progress_cb : optional callable(batches_done, total_batches)

        Returns
        -------
        tuple of (candidates, robust_payout) where candidates may be the
        enriched list if injection occurred, and robust_payout has shape
        (len(candidates), n_sims) float32.
        """
        n_sims = self._sim_matrix.shape[0]

        # --- Generate all K field samples (raw arrays + sorted scoring arrays) ---
        logger.info(
            "Generating %d field samples (N=%d each)...",
            self._n_k, self._n_field,
        )
        field_raw_list: list[np.ndarray] = []
        field_sorted_list: list[np.ndarray] = []
        for k in range(self._n_k):
            seed = self._field_seed + k
            raw = self._cs.generate_field(
                self._players_df, self._ownership_vec,
                n_lineups=self._n_field, rng_seed=seed,
            )
            field_raw_list.append(raw)
            field_sorted_list.append(self._build_field_sorted(raw))
            logger.info("  Field %d/%d: %d lineups", k + 1, self._n_k, raw.shape[0])

        # --- Optional injection: promote top field lineups into candidate pool ---
        if self._inject_field_lineups and self._portfolio_size > 0 and self._n_k > 0:
            candidates = self._inject_top_field_lineups(
                candidates, field_raw_list[0], field_sorted_list[0],
            )
            # last_injected_count is set inside _inject_top_field_lineups

        M = len(candidates)
        col_lineups = self._build_col_lineups(candidates)  # (M, 10) int32

        # --- Score the (possibly enriched) candidate pool against all K fields ---
        robust_payout = np.zeros((M, n_sims), dtype=np.float32)
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

    def _score_raw_lineups_against_field(
        self,
        raw_lineups: np.ndarray,   # (N, 10) int64 player_ids
        field_sorted: np.ndarray,  # (n_sims, F) float32 sorted
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score raw (N, 10) lineup arrays against a sorted field array.

        Returns
        -------
        payout : (N, n_sims) float32 — zero for lineups with unmapped players
        valid_mask : (N,) bool
        """
        N = raw_lineups.shape[0]
        n_sims = self._sim_matrix.shape[0]

        col = np.full((N, 10), -1, dtype=np.int32)
        valid_mask = np.ones(N, dtype=bool)
        for i in range(N):
            for j in range(10):
                c = self._col_map.get(int(raw_lineups[i, j]), -1)
                if c == -1:
                    valid_mask[i] = False
                    break
                col[i, j] = c

        payout = np.zeros((N, n_sims), dtype=np.float32)
        valid_idx = np.where(valid_mask)[0]
        for start in range(0, len(valid_idx), self._batch_size):
            end = min(start + self._batch_size, len(valid_idx))
            vi = valid_idx[start:end]
            scores = self._sim_matrix[:, col[vi]].sum(axis=2).T.astype(np.float32)
            payout[vi] = _compute_payout_from_sorted_field(scores, field_sorted, self._payout_arr)

        return payout, valid_mask

    def _inject_top_field_lineups(
        self,
        candidates: list[Lineup],
        field_lineups: np.ndarray,  # (N_field, 10) int64
        field_sorted: np.ndarray,   # (n_sims, N_valid) float32 sorted
    ) -> list[Lineup]:
        """Promote top-performing field lineups into the candidate pool.

        Scores both the original candidates and the field sample 0 lineups
        against field_sorted. Any field lineup whose mean $EV exceeds that of
        its paired bottom candidate is substituted in. Up to portfolio_size
        substitutions are made.
        """
        N_inject = self._portfolio_size
        M = len(candidates)
        n_sims = self._sim_matrix.shape[0]

        # Score original candidates against field 0
        orig_col = self._build_col_lineups(candidates)
        orig_payout = np.zeros((M, n_sims), dtype=np.float32)
        for start in range(0, M, self._batch_size):
            end = min(start + self._batch_size, M)
            scores = self._sim_matrix[:, orig_col[start:end]].sum(axis=2).T.astype(np.float32)
            orig_payout[start:end] = _compute_payout_from_sorted_field(
                scores, field_sorted, self._payout_arr,
            )
        cand_mean_ev = orig_payout.mean(axis=1)  # (M,)

        # Score field lineups against field 0 (they compete against themselves,
        # but N=5000 makes the self-inclusion effect negligible)
        field_payout, field_valid = self._score_raw_lineups_against_field(
            field_lineups, field_sorted,
        )
        field_mean_ev = field_payout.mean(axis=1)  # (N_field,); invalid → 0.0

        # Rank field lineups best→worst; rank candidates worst→best
        top_field = np.argsort(field_mean_ev)[::-1][:N_inject]
        bottom_cands = np.argsort(cand_mean_ev)[:N_inject]

        existing_pid_sets = {frozenset(lu.player_ids) for lu in candidates}
        new_candidates = list(candidates)
        injected = 0

        for rank in range(min(N_inject, len(top_field))):
            fi = top_field[rank]
            ci = bottom_cands[rank]

            if not field_valid[fi]:
                continue

            field_ev = float(field_mean_ev[fi])
            cand_ev = float(cand_mean_ev[ci])
            if field_ev <= cand_ev:
                break  # sorted descending; nothing further will beat it either

            raw_pids = [int(p) for p in field_lineups[fi]]
            pid_set = frozenset(raw_pids)
            if pid_set in existing_pid_sets:
                continue  # already in pool

            existing_pid_sets.discard(frozenset(new_candidates[ci].player_ids))
            existing_pid_sets.add(pid_set)
            new_candidates[ci] = Lineup(player_ids=raw_pids)
            injected += 1

        self.last_injected_count = injected
        if injected > 0:
            logger.info(
                "Injected %d field lineup(s) into candidate pool (replaced %d bottom candidates).",
                injected, injected,
            )
        else:
            logger.info("No field lineups qualified for injection (all below bottom candidate EV).")

        return new_candidates


# ------------------------------------------------------------------ #
#  EVPortfolioSelector                                                 #
# ------------------------------------------------------------------ #

class EVPortfolioSelector:
    """Greedy marginal-EV portfolio selection from a pre-scored candidate pool.

    Round 0: pick the candidate with the highest mean EV across all sims.
    Round k: pick the candidate that maximizes the *additional* EV above what
             prior portfolio lineups already capture — i.e. the one maximizing
             mean_s(max(0, robust_payout[c, s] - best_payout[s])).

    This naturally produces diverse portfolios: a second lineup that wins in the
    same sims as the first adds little marginal EV.

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
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> list[tuple[Lineup, float]]:
        """Run greedy selection.

        Parameters
        ----------
        progress_cb : optional callable(round_k, candidate_index, marginal_ev)

        Returns
        -------
        list of (Lineup, marginal_ev) tuples in selection order.
        """
        train_payout = self._robust_payout[:, self._train_idx]  # (M, n_train)
        n_train = train_payout.shape[1]
        best_payout = np.zeros(n_train, dtype=np.float32)

        selected: list[tuple[Lineup, float]] = []
        selected_pid_sets: list[frozenset] = []
        remaining = np.arange(self._M, dtype=np.int32)

        for k in range(self._portfolio_size):
            if len(remaining) == 0:
                break

            if k == 0:
                ev = train_payout[remaining, :].mean(axis=1)  # (M_rem,)
                best_local = int(np.argmax(ev))
                marginal = float(ev[best_local])
            else:
                marginal_ev = _compute_marginal_ev(
                    train_payout, best_payout, remaining
                )
                best_local = int(np.argmax(marginal_ev))
                marginal = float(marginal_ev[best_local])

            best_global = int(remaining[best_local])
            lineup = self._candidates[best_global]
            pid_set = frozenset(lineup.player_ids)

            # Skip exact duplicates (shouldn't happen but guards edge cases).
            if pid_set in selected_pid_sets:
                logger.warning(
                    "Round %d: candidate %d is a duplicate; skipping.", k, best_global
                )
                remaining = np.delete(remaining, best_local)
                continue

            selected.append((lineup, marginal))
            selected_pid_sets.append(pid_set)
            self._selected_indices.append(best_global)

            best_payout = np.maximum(best_payout, train_payout[best_global, :])
            remaining = np.delete(remaining, best_local)

            if progress_cb is not None:
                progress_cb(k, best_global, marginal)
            if stop_check is not None and stop_check():
                logger.info("EVPortfolioSelector: stop requested after round %d.", k)
                break

        self._best_payout_train = best_payout
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
