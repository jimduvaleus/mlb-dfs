"""Runnable MRP entry point: a drop-in alternative to `allocate_contests`.

Returns the same `ExternalAllocation(portfolio, entry_plan, unfilled)` that
`external_pool.allocate_contests` returns, so everything downstream --
`_external_assignments`, `write_upload_files`, the portfolio JSON -- works
unchanged. The point is that adopting MRP is a swap at ONE call site, and
backing it out is the same swap in reverse.

WHAT IT DOES DIFFERENTLY. Production fills contests one at a time in a fixed
order (entry fee desc, prize pool asc), coupling them only through a shared
removal mask, and scores each candidate against the field ALONE. Here every
pick is the best (candidate, contest) pair in the whole slate by marginal
dollars, and a candidate is scored against `opponents UNION our own entries`,
so a near-duplicate of something already selected is penalised mechanically
rather than by a correlation heuristic.

A/B SUPPORT is the reason `preassigned` exists. The live comparison splits each
contest's purchased slots between production and MRP, and both halves are OURS
-- they compete with each other for the same prizes. Passing production's picks
as `preassigned` commits them into each contest's state before the greedy runs,
so MRP sees them as incumbents (displacing its candidates, and losing value
when MRP outscores them) instead of pretending it has the contest to itself.
Getting this wrong would flatter MRP by exactly the interaction term the whole
build is about.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Callable, Optional

import numpy as np

from src.api.external_pool import (
    ExternalAllocation,
    _lineup_indicator_matrix,
    compute_lineup_scores,
    compute_proj_score_floor,
    implied_field_size,
)
from src.optimization.contest import ContestSimulator
from src.optimization.mrp.allocator import AllocationRules, allocate
from src.optimization.mrp.delta_reward import ContestDeltaState
from src.optimization.payout import nearest_payout_structure, payout_table_to_array

# Matches _TOPN_FIELD_POOL_CAP: one field-lineup pool is generated per slate and
# subsampled per contest, rather than a fresh field per contest.
_FIELD_POOL_CAP = 25_000
# The dominant retained array is n_above_field, uint16 (M x S) PER CONTEST, and
# the global greedy needs every contest's state alive at once. At M=5,100 and
# 10 contests that is ~100 MB per 1,000 worlds, so the world axis is capped
# (evenly strided, same device as external_pool_corr_max_sims) rather than
# letting n_sims multiply by the contest count.
_DEFAULT_MAX_SIMS_PER_CONTEST = 12_500


@dataclass
class MRPConfig:
    gamma_in: int = 7
    gamma_out: int = 8
    allow_cross_contest_duplicates: bool = False
    smooth_tau_scale: float = 0.0
    field_pool_size: int = _FIELD_POOL_CAP
    max_sims_per_contest: int = _DEFAULT_MAX_SIMS_PER_CONTEST
    seed: int = 42
    # Haugh & Singal line 2: generate along the mean-variance frontier and add
    # the result to the candidate pool. dR cannot select what is not in the
    # pool, and the pool measured as NOT spanning this region on 9/9 slates.
    # Off by default -- it costs CP-SAT solve time and changes the pool.
    frontier_enabled: bool = False
    # Wall clock is dominated by SAMPLING (~10s at 30k) plus whatever
    # n_anchors costs (~2.6-8s each). Scoring, lambda calibration and mutation
    # are together under a second.
    frontier_n_lambdas: int = 12
    # THE DIVERSITY KNOB. Top N per (lambda, primary stack team), so no single
    # team can crowd out the rest however much the objective favours it. The
    # previous solve-then-mutate build produced 100% one-team lineups without
    # it. per_team x n_teams is the effective cap per lambda.
    frontier_per_team: int = 8
    # Candidates drawn from CandidateGenerator's team-round-robin sampler and
    # ranked by the exact objective. 30k costs ~10s and covers every team.
    frontier_sample_n: int = 30_000
    # Exact CP-SAT solves at spread lambdas, keeping the true frontier tip in
    # the pool (sampling lands ~4% short) and standing as a drift check.
    # 0 makes the run completely solver-free.
    frontier_n_anchors: int = 2
    frontier_n_generations: int = 2
    frontier_mutants_per_parent: int = 4
    # Minimum lineup salary for generated candidates. Set this to the floor
    # SaberSim was given, so generated lineups occupy the same salary regime
    # as the external pool they are merged into. 0 disables.
    frontier_salary_floor: float = 47_500.0
    frontier_solver_timeout_s: float = 8.0

    def rules(self) -> AllocationRules:
        return AllocationRules(
            gamma_in=self.gamma_in,
            gamma_out=self.gamma_out,
            allow_cross_contest_duplicates=self.allow_cross_contest_duplicates,
        )


@dataclass
class MRPDiagnostics:
    """Everything needed to explain a portfolio after the fact."""

    per_contest: list = _field(default_factory=list)
    total_reward: float = 0.0
    n_unfilled: int = 0
    # Overlap caps loosened to fill purchased slots, and the pre-flight verdict.
    relaxations: list = _field(default_factory=list)
    preflight: dict = _field(default_factory=dict)
    # Pool-wide floor cull, empty when disabled.
    floor: dict = _field(default_factory=dict)
    # Line-2 frontier generation, empty when disabled.
    frontier: dict = _field(default_factory=dict)

    def warnings(self) -> list[str]:
        """User-facing warnings, worst first. Empty when nothing went wrong."""
        out = []
        if self.n_unfilled:
            out.append(
                f"{self.n_unfilled} purchased {'entry' if self.n_unfilled == 1 else 'entries'} "
                f"could NOT be filled - the candidate pool has fewer distinct lineups than "
                f"slots. Those entry fees are spent and unused.")
        if self.relaxations:
            by_rule: dict = {}
            for r in self.relaxations:
                by_rule.setdefault(r["rule"] if isinstance(r, dict) else r.rule, set()).add(
                    r["contest_id"] if isinstance(r, dict) else r.contest_id)
            parts = [f"{rule} in {len(cids)} contest{'s' if len(cids) != 1 else ''}"
                     for rule, cids in sorted(by_rule.items())]
            out.append(
                "Overlap limits were relaxed to fill every purchased slot ("
                + ", ".join(parts)
                + "). The pool could not supply enough distinct lineups at the "
                  "configured caps, so entries overlap more than intended.")
        pf = self.preflight
        if pf and not pf.get("ok", True):
            out.append(
                f"Pre-flight: at gamma_in={pf.get('gamma_in')} the pool supports about "
                f"{pf.get('capacity')} mutually-compatible lineups but the largest contest "
                f"needs {pf.get('required')}.")
        return out

    def summary(self) -> str:
        lines = [f"MRP: {len(self.per_contest)} contests, "
                 f"R(S) = ${self.total_reward:,.2f}, unfilled {self.n_unfilled}"
                 + (f", {len(self.relaxations)} overlap relaxations"
                    if self.relaxations else "")]
        f = self.frontier
        if f.get("skipped"):
            lines.append(f"  line-2 frontier: SKIPPED ({f['skipped']})")
        elif f:
            lines.append(
                f"  line-2 frontier: generated {f.get('n_generated', 0)}, "
                f"kept {f.get('n_kept', 0)} after exact-dupe drop, "
                f"{f.get('n_surviving_floor', 0)} past the floor, "
                f"PICKED {f.get('n_picked', 0)}  "
                f"[lambda {f.get('lambda_min', 0):.4g}-{f.get('lambda_max', 0):.4g}, "
                f"{f.get('n_cov_pairs', 0)} cov pairs, "
                f"sigma_dG blended over {f.get('sigma_dG_contests', 0)} contests "
                f"(min corr {f.get('sigma_dG_min_corr', float('nan')):.3f})]")
        for c in self.per_contest:
            lines.append(
                f"  {c['contest_name'][:34]:34s} k={c['k']:3d} "
                f"field~{c['field_size']:6,d} R=${c['reward']:9,.2f} "
                f"first dR=${c['first_delta']:7.3f} last dR=${c['last_delta']:7.3f}"
            )
        return "\n".join(lines)


def _world_slice(n_sims: int, cap: int) -> np.ndarray:
    """Evenly strided world indices. Strided rather than a leading block so the
    subsample cannot align with any contiguous split used elsewhere."""
    if cap <= 0 or cap >= n_sims:
        return np.arange(n_sims)
    step = int(np.ceil(n_sims / cap))
    return np.arange(0, n_sims, step)


def _floor_keep_indices(
    n: int,
    floor_scores: Optional[np.ndarray],
    percentile: float,
    preassigned: Optional[dict],
    exempt_idx: Optional[set] = None,
) -> tuple[np.ndarray, dict]:
    """(keep_idx, diag) for the pool-wide floor cull.

    Delegates the cutoff to `external_pool.compute_proj_score_floor` and
    reproduces `allocate_contests`' mask expression exactly, so the two paths
    cull the same lineups from the same basis rather than merely similar ones
    (`tests/test_mrp_floor.py` asserts the agreement on a shared basis).
    """
    all_idx = np.arange(n)
    if floor_scores is None or percentile <= 0:
        return all_idx, {}
    basis = np.asarray(floor_scores, dtype=np.float64)
    if basis.shape != (n,):
        raise ValueError(
            f"floor_scores has shape {basis.shape}, expected ({n},) to align "
            "with pool.lineups"
        )
    floor = compute_proj_score_floor(basis, percentile)
    if floor is None:
        return all_idx, {}
    cutoff, n_culled = floor
    keep = np.isfinite(basis) & (basis >= cutoff)
    n_exempt = 0
    for idxs in (preassigned or {}).values():
        for j in idxs:
            if not keep[int(j)]:
                keep[int(j)] = True
                n_exempt += 1
    # Frontier lineups are exempt for a different reason than preassigned ones.
    # The floor basis is SaberSim's own "99th" column, which a generated lineup
    # by definition does not have; the fallback (compute_pool_ceiling_proxy)
    # assumes the 10 rostered players are independent, so it understates
    # exactly the correlated-stack variance line 2 maximises -- it would cull
    # the high-lambda lineups the frontier exists to produce. dR still has to
    # want them; this only declines to pre-judge them on a biased basis.
    n_frontier_exempt = 0
    for j in (exempt_idx or ()):
        if not keep[int(j)]:
            keep[int(j)] = True
            n_frontier_exempt += 1
    keep_idx = np.flatnonzero(keep)
    return keep_idx, {
        "cutoff": float(cutoff),
        "percentile": float(percentile),
        "pool_size": n,
        "n_culled": int(n - len(keep_idx)),
        # Counts non-finite entries too, which includes the NaN pad the
        # frontier merge adds -- read n_culled for the applied figure.
        "n_culled_before_exempt": int(n_culled),
        "n_preassigned_exempt": n_exempt,
        "n_frontier_exempt": n_frontier_exempt,
    }


def _frontier_augment(
    pool,
    players_df,
    sim_results,
    sim_matrix,
    field_pool_scores,
    groups: list,
    cfg: "MRPConfig",
    floor_scores: Optional[np.ndarray],
    rng,
    progress_cb=None,
    stop_check=None,
):
    """Haugh & Singal line 2: generate along the frontier, merge into the pool.

    Returns `(pool, n_frontier, diag, floor_scores)` -- `pool` augmented,
    `n_frontier` the count of survivors appended at the END of `pool.lineups`,
    and `floor_scores` padded to match.

    WHY IT RUNS HERE. `sigma_dG` needs the simulated field, and the frontier
    lineups must exist before anything is scored or culled, so this sits
    between the field build and the floor -- which is why
    `allocate_marginal_reward` builds its field pool earlier than it used to.

    SIGMA_dG IS BLENDED ACROSS CONTESTS, weighted by purchased entries. The
    paper's formulation is single-contest; dR is not, so the generator sees
    every contest's payout cutoff in proportion to the exposure we actually
    bought there. See the inline note below for the measured size of the
    effect and for `sigma_min_corr`, which makes a genuinely divergent slate
    visible instead of silent.

    Fails SOFT. This is an off-by-default generator, so a slate that cannot
    supply what it needs (no `eligible_positions`, no contests, a solver that
    returns nothing) leaves the pool untouched and records why, rather than
    taking down a run that would otherwise have produced a portfolio.
    """
    from src.api.external_pool import ExternalPool
    from src.optimization.mrp.field_covariance import (
        field_order_statistics,
        payout_weighted_sigma,
        player_field_covariance,
        tier_boundary_ranks,
    )
    from src.optimization.mrp.frontier_qp import frontier_lineups, restrict_to_playable
    from src.optimization.mrp.lineup_variance import unit_covariance_pairs

    n_real = len(pool.lineups)
    if "eligible_positions" not in getattr(players_df, "columns", []):
        return pool, 0, {"skipped": "players_df has no eligible_positions"}, floor_scores
    if not groups:
        return pool, 0, {"skipped": "no contests"}, floor_scores

    F_pool = field_pool_scores.shape[1]

    # SIGMA_dG IS BLENDED ACROSS CONTESTS, WEIGHTED BY ENTRIES. It used to be
    # taken from the largest contest alone, which is arbitrary: "largest field"
    # is not "where our money is", and on a slate whose biggest contest holds
    # two entries it would tune generation to a contest we barely play. dR
    # allocates across all of them, so the generator should see all of them.
    #
    # Weighting by purchased entries is the natural choice -- it is the
    # exposure each contest's payout cutoff actually carries.
    #
    # SIZE OF THE EFFECT, measured on 08/18 (7 contests, 496-8,000 entrants):
    # per-player sigma_dG correlates 0.93-0.99 across contests and LINEUP
    # scores 0.984-0.9996, with 85-98 of the top 100 shared against the
    # largest contest. So this is a robustness fix, not a large one -- the
    # same approximate stability Assumption 5.2 asserts across tiers appears
    # to hold across contest sizes. `sigma_min_corr` below reports the worst
    # per-contest agreement with the blend so a slate where it does NOT hold
    # is visible rather than silent.
    thr_parts, meta, specs = [], [], []
    for g in groups:
        f_size = int(np.clip(int(implied_field_size(g)) or F_pool, 1, F_pool))
        structure, _approx = nearest_payout_structure(g.contest_name, n_entries=f_size)
        payout_arr = payout_table_to_array(structure)
        ranks = tier_boundary_ranks(payout_arr)
        if ranks.size == 0:
            continue
        cols = (np.arange(F_pool) if f_size >= F_pool
                else rng.choice(F_pool, size=f_size, replace=False))
        field_sorted = np.sort(field_pool_scores[:, cols], axis=1)
        thr_c = field_order_statistics(field_sorted, ranks)
        thr_parts.append(thr_c)
        # Steps R_d - R_{d+1}: formulation (2)'s tier weights, and exactly what
        # line 4 sums over. Kept here so the same order statistics serve both
        # sigma_dG and the lambda* search instead of being rebuilt.
        amounts = payout_arr[np.clip(ranks - 1, 0, len(payout_arr) - 1)]
        steps_c = amounts - np.concatenate((amounts[1:], [0.0]))
        specs.append((g.contest_id, thr_c, steps_c, len(g.entries)))
        meta.append((payout_arr, ranks, f_size, len(g.entries)))
        del field_sorted
    if not thr_parts:
        return pool, 0, {"skipped": "no contest has a paying payout table"}, floor_scores

    # One covariance pass over the CONCATENATED thresholds rather than one per
    # contest: player_field_covariance is a chunked matmul over the world axis,
    # so a single (n_players x sum_T) call costs far less than N of them and
    # touches the big sim array once.
    block = player_field_covariance(sim_matrix, np.concatenate(thr_parts, axis=1))
    del thr_parts
    sigma_vec = np.zeros(block.shape[0], dtype=np.float64)
    per_contest, weights, off = [], [], 0
    for payout_arr, ranks, _f_size, k in meta:
        s_c = payout_weighted_sigma(block[:, off:off + len(ranks)], payout_arr, ranks)
        off += len(ranks)
        per_contest.append(s_c)
        w = float(max(k, 1))
        weights.append(w)
        sigma_vec += w * s_c
    sigma_vec /= max(sum(weights), 1.0)
    del block

    # Worst agreement between any single contest and the blend. Near 1.0 means
    # the contests want the same lineups and the blend costs nothing; a low
    # value means this slate's contests genuinely disagree and the blend is
    # serving none of them well.
    sigma_min_corr = 1.0
    if len(per_contest) > 1 and np.std(sigma_vec) > 0:
        cs = [float(np.corrcoef(s_c, sigma_vec)[0, 1])
              for s_c in per_contest if np.std(s_c) > 0]
        if cs:
            sigma_min_corr = min(cs)

    sigma_dG = {int(pid): float(sigma_vec[i])
                for i, pid in enumerate(sim_results.player_ids)}

    var_by_pid, cov_by_pair = unit_covariance_pairs(
        sim_matrix, sim_results.player_ids, players_df,
    )
    # Drop the doubly-dominated before solving: they carry most of the pair
    # variables and can never be optimal at any lambda. Recompute the pairs on
    # the restricted set so the model is not handed variables it cannot use.
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    gen_df, restrict_diag = restrict_to_playable(players_df, pool_pids)
    gen_pids = {int(x) for x in gen_df["player_id"]}
    cov_by_pair = {k: v for k, v in cov_by_pair.items()
                   if k[0] in gen_pids and k[1] in gen_pids}

    if progress_cb is not None:
        progress_cb({"stage": "mrp_frontier_start",
                     # The SEARCH grid line 4 chooses from -- NOT the number of
                     # operating points generated at, which is the count of
                     # distinct lambda* and is not known until line 4 has run.
                     "n_lambda_search": cfg.frontier_n_lambdas,
                     "per_team": cfg.frontier_per_team,
                     "n_sample": cfg.frontier_sample_n,
                     "n_pairs": len(cov_by_pair)})

    # min_uniques/min_stack mirror the sigma_frontier precedent so frontier
    # output is as playable as anything else this repo generates.
    generated, lambdas, lam_diag = frontier_lineups(
        gen_df, var_by_pid, cov_by_pair, sigma_dG,
        specs, sim_matrix, {int(p): i for i, p in enumerate(sim_results.player_ids)},
        n_lambdas=cfg.frontier_n_lambdas,
        per_team=cfg.frontier_per_team,
        sample_n=cfg.frontier_sample_n,
        n_anchors=cfg.frontier_n_anchors,
        n_generations=cfg.frontier_n_generations,
        mutants_per_parent=cfg.frontier_mutants_per_parent,
        min_uniques=3, min_stack=4,
        # Deliberately NOT optimizer.salary_floor (49,500 here): that is a
        # holdover from the internal optimizer and is not the floor the
        # external pool was built under. Generated lineups have to sit in the
        # same salary regime as the SaberSim lineups they compete with in the
        # pool, so this mirrors whatever floor SaberSim was given. Without any
        # floor the sampler leaves thousands unspent -- a shape no other stage
        # of the funnel produces, and unused salary is a dupe-model feature.
        salary_floor=(cfg.frontier_salary_floor
                      if cfg.frontier_salary_floor and cfg.frontier_salary_floor > 0
                      else None),
        timeout_s=cfg.frontier_solver_timeout_s,
        seed=cfg.seed,
        progress_cb=(lambda d, t, n: progress_cb(
            {"stage": "mrp_frontier", "done": d, "total": t,
             "n_lineups": n})) if progress_cb else None,
        stop_check=stop_check,
    )
    if not generated:
        return pool, 0, {**lam_diag, "skipped": "generator returned no lineups",
                         "n_generated": 0}, floor_scores

    # EXACT duplicates only -- the 9/10 near-duplicate cull is deliberately NOT
    # applied to frontier lineups. Shape mutants differ from their parent by a
    # single player, which is precisely what that cull targets: it removed 73%
    # of them (520 -> 140) when tried. It is also redundant here, because dR's
    # demotion term already prices near-duplicates mechanically -- an entry
    # that cannot also take first place is penalised by the objective itself,
    # which is the stated reason this module exists rather than a correlation
    # heuristic. Exact duplicates still go, since two identical entries in one
    # contest is a real error rather than a diversity question.
    #
    # CONSEQUENCE, stated because it is easy to miss: `gamma_out: 8` is
    # documented as a no-op against a pool through the 9/10 cull. With the
    # frontier exempt, part of the pool is no longer 9/10-deduped, so gamma_out
    # can start binding on frontier-vs-frontier pairs.
    existing = {frozenset(lu.player_ids) for lu in pool.lineups}
    kept, kept_lambdas = [], []
    for lu, lam in zip(generated, lambdas):
        key = frozenset(lu.player_ids)
        if key in existing:
            continue
        existing.add(key)
        kept.append(lu)
        kept_lambdas.append(lam)
    survivors = list(pool.lineups) + kept
    n_frontier = len(kept)

    augmented = ExternalPool(
        lineups=survivors, contests=pool.contests,
        n_dropped_unknown_players=pool.n_dropped_unknown_players,
        n_dropped_duplicates=pool.n_dropped_duplicates,
        n_dropped_near_duplicates=pool.n_dropped_near_duplicates,
        source_paths=pool.source_paths,
    )

    # Pad the caller's floor basis to the new length. NaN rather than a
    # computed ceiling: these indices are exempt from the cull anyway, and
    # compute_proj_score_floor drops non-finite entries before taking its
    # percentile, so the cutoff stays computed on the REAL pool alone.
    if floor_scores is not None and n_frontier:
        floor_scores = np.concatenate([
            np.asarray(floor_scores, dtype=np.float64),
            np.full(n_frontier, np.nan, dtype=np.float64),
        ])

    diag = {
        "n_generated": len(generated),
        "n_kept": int(n_frontier),
        "n_dropped_duplicate": int(len(generated) - n_frontier),
        "n_real": int(n_real),
        "lambda_min": float(min(kept_lambdas)) if kept_lambdas else 0.0,
        "lambda_max": float(max(kept_lambdas)) if kept_lambdas else 0.0,
        "n_lambdas_represented": len(set(kept_lambdas)),
        "sigma_dG_contests": len(meta),
        "sigma_dG_min_corr": round(float(sigma_min_corr), 4),
        **lam_diag,
        "n_cov_pairs": len(cov_by_pair),
        **restrict_diag,
    }
    if progress_cb is not None:
        progress_cb({"stage": "mrp_frontier_done", **diag})
    return augmented, int(n_frontier), diag, floor_scores


def allocate_marginal_reward(
    pool,
    players_df,
    sim_results,
    groups: list,
    cfg: Optional[MRPConfig] = None,
    *,
    preassigned: Optional[dict] = None,
    floor_scores: Optional[np.ndarray] = None,
    proj_score_floor_percentile: float = 0.0,
    progress_cb: Optional[Callable[[dict], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> tuple[ExternalAllocation, MRPDiagnostics]:
    """Allocate every purchased entry by marginal expected dollars.

    Parameters
    ----------
    pool : ExternalPool -- the candidate pool (already through the 9/10 cull).
    groups : list[ContestGroup] -- purchased entries per contest. Entry counts
        are exogenous; MRP decides only WHICH lineup fills each slot.
    preassigned : {contest_id: [pool index]} already committed to that contest
        (an A/B's production half). Those entries become incumbents and their
        pool indices are removed from consideration.
    floor_scores : (M,) per-lineup floor basis aligned to `pool.lineups` --
        pass `compute_pool_ceiling_scores(pool, players_df)`, the same array
        every other ev_type floors on.
    proj_score_floor_percentile : cull the bottom N% of `floor_scores`
        pool-wide before any scoring, with `allocate_contests`' exact
        semantics (`isfinite(basis) & basis >= cutoff`, so a lineup with no
        finite score is culled too). 0 disables. Applied by SUBSETTING the
        candidate axis rather than masking it, so the (M x S) per-contest
        rank arrays shrink with the cull instead of carrying dead columns.
        `preassigned` indices are exempt: an A/B's other arm already bought
        those entries, so they stay as incumbents whatever their ceiling. So
        are line-2 frontier lineups when `cfg.frontier_enabled` -- they have no
        SaberSim ceiling column and the fallback proxy is biased against them
        (see `_floor_keep_indices`).
    """
    cfg = cfg or MRPConfig()
    groups = [g for g in groups if g.entries]
    if not pool.lineups or not groups:
        n_missing = sum(len(g.entries) for g in groups)
        return ExternalAllocation(
            portfolio=[], entry_plan=[],
            unfilled=[e for g in groups for e in g.entries],
        ), MRPDiagnostics(n_unfilled=n_missing)

    # SIM SUBSTRATE FIRST. The field pool used to be built after the floor
    # cull, but sigma_dG needs the simulated field and the frontier's lineups
    # must exist before anything is scored or culled, so the field moves ahead
    # of both. The two orderings the old comments justified are preserved
    # below: the floor still runs BEFORE the pre-flight, and the (M x S)
    # candidate-score array is still built AFTER the floor.
    rng = np.random.default_rng(cfg.seed)
    keep = _world_slice(sim_results.results_matrix.shape[0], cfg.max_sims_per_contest)
    sim_matrix = sim_results.results_matrix.astype(np.float32)[keep]
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    own = players_df["ownership"].to_numpy(dtype=np.float64)

    sim = ContestSimulator()
    field_pool = sim.generate_field(
        players_df, own, n_lineups=min(cfg.field_pool_size, _FIELD_POOL_CAP),
        rng_seed=cfg.seed, stop_check=stop_check,
    )
    field_pool_scores = sim.score_field(field_pool, sim_matrix, col_map)   # (S, F_pool)
    F_pool = field_pool_scores.shape[1]

    # LINE-2 FRONTIER. Adds candidates dR could not otherwise select; a no-op
    # when disabled. Runs while sim_matrix is still alive because Sigma_delta
    # is estimated from it.
    frontier_diag: dict = {}
    n_frontier = 0
    if cfg.frontier_enabled:
        pool, n_frontier, frontier_diag, floor_scores = _frontier_augment(
            pool, players_df, sim_results, sim_matrix, field_pool_scores,
            groups, cfg, floor_scores, rng, progress_cb, stop_check,
        )
    del sim_matrix

    # POOL-WIDE FLOOR, before the pre-flight: the cull changes how many
    # mutually-compatible lineups the pool can supply, so a capacity verdict
    # taken on the uncut pool would answer a question about a pool that is
    # not the one being allocated from. Frontier lineups sit at the END of
    # pool.lineups and are exempt -- see _floor_keep_indices.
    frontier_idx = (set(range(len(pool.lineups) - n_frontier, len(pool.lineups)))
                    if n_frontier else None)
    keep_idx, floor_diag = _floor_keep_indices(
        len(pool.lineups), floor_scores, proj_score_floor_percentile, preassigned,
        exempt_idx=frontier_idx,
    )
    lineups = [pool.lineups[i] for i in keep_idx]
    if not lineups:
        n_missing = sum(len(g.entries) for g in groups)
        return ExternalAllocation(
            portfolio=[], entry_plan=[],
            unfilled=[e for g in groups for e in g.entries],
        ), MRPDiagnostics(n_unfilled=n_missing, floor=floor_diag,
                          frontier=frontier_diag)
    # Pool indices -> candidate-axis positions, for `preassigned` in and picks
    # back out. Every downstream index is a position in `lineups` from here.
    pos_of = {int(orig): new for new, orig in enumerate(keep_idx)}
    preassigned = {
        cid: [pos_of[int(j)] for j in idxs if int(j) in pos_of]
        for cid, idxs in (preassigned or {}).items()
    }
    # Frontier membership on the CANDIDATE axis, so a pick can be attributed
    # back to the generator after the floor has renumbered everything.
    is_frontier = np.zeros(len(lineups), dtype=bool)
    for j in (frontier_idx or ()):
        if int(j) in pos_of:
            is_frontier[pos_of[int(j)]] = True
    if floor_diag.get("n_culled") and progress_cb is not None:
        progress_cb({"stage": "mrp_floor", **floor_diag})

    # PRE-FLIGHT, before any simulation-sized work: can the pool even supply
    # the biggest contest under this gamma_in? Composition-only, so it costs
    # ~a second and fails fast instead of ten minutes in.
    _max_slots = max(len(g.entries) for g in groups)
    preflight = preflight_overlap_capacity(
        lineups, sim_results.player_ids, _max_slots,
        cfg.gamma_in, roster_size=cfg.rules().roster_size,
    )
    if progress_cb is not None:
        progress_cb({"stage": "mrp_preflight", **preflight})

    scores_full = compute_lineup_scores(lineups, sim_results)             # (M, S)
    cand_scores = np.ascontiguousarray(scores_full[:, keep])
    del scores_full
    indicator = _lineup_indicator_matrix(lineups, sim_results.player_ids)

    states, slots, meta = {}, {}, {}
    for i, g in enumerate(groups):
        if stop_check is not None and stop_check():
            break
        implied = int(implied_field_size(g)) or F_pool
        f_size = int(np.clip(implied, 1, F_pool))
        structure, approx = nearest_payout_structure(g.contest_name, n_entries=f_size)
        payout_arr = payout_table_to_array(structure)

        cols = (np.arange(F_pool) if f_size >= F_pool
                else rng.choice(F_pool, size=f_size, replace=False))
        field_sorted = np.sort(field_pool_scores[:, cols], axis=1)
        states[g.contest_id] = ContestDeltaState(
            cand_scores, field_sorted, payout_arr,
            smooth_tau_scale=cfg.smooth_tau_scale,
        )
        del field_sorted
        slots[g.contest_id] = len(g.entries)
        meta[g.contest_id] = {
            "contest_name": g.contest_name, "field_size": f_size,
            "payout_approx": bool(approx), "k": len(g.entries),
        }
        if progress_cb is not None:
            progress_cb({"stage": "mrp_build", "done": i + 1, "total": len(groups)})
    del field_pool_scores

    # A/B: our OTHER arm's entries are incumbents, not absent.
    for cid, idxs in (preassigned or {}).items():
        st = states.get(cid)
        if st is None:
            continue
        for j in idxs:
            st.commit(int(j))
        slots[cid] = max(0, slots[cid] - len(idxs))

    res = allocate(
        states, slots, indicator, cfg.rules(),
        progress_cb=(lambda d, t: progress_cb({"stage": "mrp_pick", "done": d, "total": t}))
        if progress_cb else None,
    )

    picks_by_contest = res.by_contest()
    portfolio, entry_plan, unfilled, from_generated = [], [], [], []
    delta_by = {}
    for p in res.picks:
        delta_by.setdefault(p.contest_id, []).append(p.delta)
    for g in groups:
        picks = picks_by_contest.get(g.contest_id, [])
        n_pre = len((preassigned or {}).get(g.contest_id, []))
        take = g.entries[n_pre:n_pre + len(picks)]
        for idx, ent in zip(picks, take):
            portfolio.append((lineups[idx], float(delta_by[g.contest_id].pop(0))))
            entry_plan.append(ent)
            from_generated.append(bool(is_frontier[idx]))
        unfilled.extend(g.entries[n_pre + len(picks):])

    if frontier_diag:
        # What the generator actually bought: how many of its lineups dR chose
        # to spend a purchased entry on. Zero is a real answer -- the frontier
        # is exempt from the floor, not from having to win on marginal dollars.
        frontier_diag = {
            **frontier_diag,
            "n_picked": int(sum(1 for pk in res.picks if is_frontier[pk.candidate])),
            "n_surviving_floor": int(is_frontier.sum()),
        }
    diag = MRPDiagnostics(
        n_unfilled=len(unfilled),
        floor=floor_diag,
        relaxations=[{"contest_id": r.contest_id, "rule": r.rule,
                      "frm": r.frm, "to": r.to, "step": r.step}
                     for r in res.relaxations],
        preflight=preflight,
        frontier=frontier_diag,
    )
    for cid, st in states.items():
        d = [p.delta for p in res.picks if p.contest_id == cid]
        diag.per_contest.append({
            **meta[cid], "contest_id": cid, "reward": st.reward(),
            "first_delta": d[0] if d else float("nan"),
            "last_delta": d[-1] if d else float("nan"),
        })
    diag.total_reward = float(sum(c["reward"] for c in diag.per_contest))
    return ExternalAllocation(portfolio=portfolio, entry_plan=entry_plan,
                              unfilled=unfilled,
                              from_generated=from_generated), diag


def publish_portfolio(
    alloc: ExternalAllocation,
    diag: MRPDiagnostics,
    players_df,
    slate_path,
    output_dir,
    platform: str = "draftkings",
    active_risk: float = 1.0,
    backup: bool = True,
) -> dict:
    """Write the portfolio where the UI's Portfolio tab reads it.

    `GET /api/portfolio/sweep` serves exactly one path --
    `<output_dir>/portfolio_sweep_<platform>.json` -- and drops the payload
    unless `slate_fingerprint` matches the CURRENT DKSalaries file. So making
    MRP visible in the tab means writing that file, with that fingerprint;
    there is no side channel.

    The per-lineup contest mapping the tab renders comes from four fields
    (`upload_tag`, `entry_fee`, `contest_name`, `entry_sort_order`), and
    `PortfolioTable` hides the entry column entirely when `upload_tag` is
    missing -- so they are populated here the same way
    `PipelineRunner._build_external_entry_map` does, from the entry plan, in
    fill order.

    TWO ARTIFACTS, because two different endpoints serve the portfolio:
    `GET /api/portfolio/sweep` reads the JSON, and `GET /api/portfolio?platform=`
    reads `portfolio_<platform>.csv` via `_load_portfolio_from_csv`. Writing
    only one leaves the two endpoints disagreeing about what the current
    portfolio is.

    The THIRD downstream consumer is late swap, which does not read either:
    `late_swap.scan_swap_entry_files` globs `outputs/*Entries*.csv`, i.e. the
    `upload_*.csv` files. Those are written by the caller (see
    scripts/run_mrp.py), and they MUST land in this same directory -- otherwise
    the Portfolio tab shows MRP while the Late Swap tab edits production's
    submitted lineups, which is a real-money hazard, not a cosmetic
    inconsistency.

    THIS OVERWRITES PRODUCTION'S SHIPPED PORTFOLIO. Those files are what the tab
    shows, what `activate_risk` re-reads, and what the archive later grades as
    "production", so `backup=True` copies each existing file to
    `*.prod-backup.*` first. `mode` is written as "marginal_reward" (the
    endpoint already surfaces `mode` and `ev_type`) so a published MRP
    portfolio is never mistaken for a production one.
    """
    import json
    import shutil
    from pathlib import Path as _Path

    from src.api.pipeline import (
        PipelineRunner,
        _extract_upload_tag,
        _shorten_contest_name,
    )
    from src.api.slate_exclusions import compute_file_fingerprint

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    own_by_id = None
    if "ownership" in players_df.columns:
        # build_external_players_df carries ownership in PERCENTAGE POINTS,
        # which is the scale _serialize_portfolio documents for this argument.
        own_by_id = {int(p): float(o) for p, o in
                     zip(players_df["player_id"], players_df["ownership"])}

    lineups = PipelineRunner._serialize_portfolio(
        alloc.portfolio, players_df, mean_ev_from_score=True,
        ownership_by_id=own_by_id,
    )
    for i, (file_path, rec) in enumerate(alloc.entry_plan):
        if i < len(lineups):
            lineups[i].update({
                "upload_tag": _extract_upload_tag(_Path(file_path).name),
                "entry_fee": rec.entry_fee_raw,
                "contest_name": _shorten_contest_name(rec.contest_name),
                "entry_sort_order": i,
            })

    sweep_path = out / f"portfolio_sweep_{platform}.json"
    csv_path = out / f"portfolio_{platform}.csv"
    backed_up = []
    if backup:
        for src, dst in ((sweep_path, out / f"portfolio_sweep_{platform}.prod-backup.json"),
                         (csv_path, out / f"portfolio_{platform}.prod-backup.csv")):
            if src.exists():
                shutil.copy2(src, dst)
                backed_up.append(str(dst))

    # GET /api/portfolio?platform= reads this CSV, not the sweep JSON.
    PipelineRunner._format_portfolio_df(
        alloc.portfolio, players_df, mean_ev_from_score=True,
    ).to_csv(csv_path, index=False)

    payload = {
        "slate_fingerprint": compute_file_fingerprint(_Path(slate_path)),
        "active_risk": active_risk,
        "mode": "marginal_reward",
        "ev_type": "delta_reward",
        "mrp_total_reward": diag.total_reward,
        "mrp_per_contest": diag.per_contest,
        "sweep": [{"risk": active_risk, "lineups": lineups}],
    }
    sweep_path.write_text(json.dumps(payload))
    return {"sweep_path": str(sweep_path), "csv_path": str(csv_path),
            "backup_paths": backed_up, "n_lineups": len(lineups)}


def check_payout_coverage(groups: list, field_pool_cap: int = _FIELD_POOL_CAP) -> list[dict]:
    """Resolve each contest's payout table and flag approximate matches.

    A pure inspection of what `allocate_marginal_reward` WILL use, so a caller
    can gate on it before committing to a run. No side effects, no blocking --
    whether an approximate table is acceptable is the caller's decision.

    WHY THIS DESERVES A GATE AT ALL. `nearest_payout_structure` never returns
    None: an unregistered contest silently falls back to the closest-size table
    of ANY registered type. That is tolerable for a per-contest allocator, where
    a wrong curve misranks candidates inside one contest. It is not tolerable
    here, because dR is denominated in DOLLARS and the greedy compares marginal
    dollars ACROSS contests -- so a wrong table does not just misrank within a
    contest, it misallocates entries between them, silently and slate-wide.

    The backtest path already treats this as a hard stop (`load_real_contests`
    raises SystemExit on an unmapped contest, and PROSPECTIVE_PROTOCOL says not
    to silence it). This gives the live path the same detector.

    Returns one row per contest with `exact=False` where the fallback fired.
    """
    from src.optimization.payout import structure_for_contest

    rows = []
    for g in groups:
        implied = int(implied_field_size(g)) or field_pool_cap
        f_size = int(np.clip(implied, 1, field_pool_cap))
        structure, approx = nearest_payout_structure(g.contest_name, n_entries=f_size)
        exact = structure_for_contest(g.contest_name, n_entries=f_size) is not None
        payout_total = float(payout_table_to_array(structure).sum())
        rows.append({
            "contest_id": g.contest_id,
            "contest_name": g.contest_name,
            "k": len(g.entries),
            "implied_field_size": f_size,
            "table_name": structure.get("name", "?"),
            "table_entries": int(structure.get("total_entries", 0)),
            "table_entry_fee": float(structure.get("entry_fee", 0.0)),
            "table_prize_pool": payout_total,
            "entry_fee": g.entry_fee_cents / 100.0,
            "exact": bool(exact and not approx),
        })
    return rows


def describe_payout_fallbacks(rows: list[dict]) -> str:
    """Human-readable summary of the approximate matches, for a prompt."""
    bad = [r for r in rows if not r["exact"]]
    if not bad:
        return ""
    lines = [f"{len(bad)} of {len(rows)} contests have no registered payout table "
             f"and will fall back to another contest's structure:"]
    for r in bad:
        lines.append(
            f"  {r['contest_name']}  ({r['k']} entries, ~{r['implied_field_size']:,} field, "
            f"${r['entry_fee']:.2f} entry)\n"
            f"      -> will use: {r['table_name']} "
            f"({r['table_entries']:,} entries, ${r['table_entry_fee']:.2f} entry, "
            f"${r['table_prize_pool']:,.0f} prize pool)"
        )
    lines.append(
        "Marginal reward is denominated in dollars and compares contests against "
        "each other, so a wrong payout table misallocates entries BETWEEN contests, "
        "not just within one. Register the real table in data/payout_structures/ "
        "and add it to payout.CONTEST_STRUCTURES to fix this properly."
    )
    return "\n".join(lines)


def preflight_overlap_capacity(
    pool_lineups: list,
    player_ids: list,
    max_slots: int,
    gamma_in: int,
    roster_size: int = 10,
    probe_cap: int = 0,
) -> dict:
    """Can the pool actually supply `max_slots` lineups under this gamma_in?

    Greedily builds a mutually-compatible set (pairwise overlap <= gamma_in) and
    stops as soon as it has enough, so the common case costs almost nothing.
    Composition only -- no sims, no fields -- so this can run BEFORE the
    expensive work rather than discovering starvation ten minutes in.

    A greedy set is a LOWER bound on what is achievable (finding the maximum is
    an independent-set problem), which is the right direction: if greedy already
    clears the requirement, the real allocator certainly can.

    Measured on a live 10,054-lineup / 226-player pool, greedy reached 200+ at
    every gamma_in down to 4, 264 at 3, 76 at 2, and only fell short at 1. So on
    a normal slate this check passes trivially; it is aimed at short slates and
    hand-tightened caps.
    """
    cap = probe_cap or max(max_slots, 1)
    M = len(pool_lineups)
    if M == 0 or max_slots <= 0:
        return {"ok": M >= max_slots, "capacity": M, "required": max_slots,
                "gamma_in": gamma_in, "probe_exhaustive": True}
    if gamma_in >= roster_size:
        return {"ok": M >= max_slots, "capacity": M, "required": max_slots,
                "gamma_in": gamma_in, "probe_exhaustive": True}

    I = _lineup_indicator_matrix(pool_lineups, player_ids)
    max_ov = np.zeros(M, dtype=np.int16)
    taken = np.zeros(M, dtype=bool)
    n = 0
    while n < cap:
        ok = (max_ov <= gamma_in) & ~taken
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            break
        j = int(idx[0])
        taken[j] = True
        n += 1
        np.maximum(max_ov, (I.T @ I[:, j]).astype(np.int16), out=max_ov)
    return {"ok": n >= max_slots, "capacity": n, "required": max_slots,
            "gamma_in": gamma_in, "probe_exhaustive": n < cap}
