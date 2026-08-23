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
    keep_idx = np.flatnonzero(keep)
    return keep_idx, {
        "cutoff": float(cutoff),
        "percentile": float(percentile),
        "pool_size": n,
        "n_culled": int(n - len(keep_idx)),
        "n_culled_before_exempt": int(n_culled),
        "n_preassigned_exempt": n_exempt,
    }


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
        those entries, so they stay as incumbents whatever their ceiling.
    """
    cfg = cfg or MRPConfig()
    groups = [g for g in groups if g.entries]
    if not pool.lineups or not groups:
        n_missing = sum(len(g.entries) for g in groups)
        return ExternalAllocation(
            portfolio=[], entry_plan=[],
            unfilled=[e for g in groups for e in g.entries],
        ), MRPDiagnostics(n_unfilled=n_missing)

    # POOL-WIDE FLOOR, before the pre-flight: the cull changes how many
    # mutually-compatible lineups the pool can supply, so a capacity verdict
    # taken on the uncut pool would answer a question about a pool that is
    # not the one being allocated from.
    keep_idx, floor_diag = _floor_keep_indices(
        len(pool.lineups), floor_scores, proj_score_floor_percentile, preassigned,
    )
    lineups = [pool.lineups[i] for i in keep_idx]
    if not lineups:
        n_missing = sum(len(g.entries) for g in groups)
        return ExternalAllocation(
            portfolio=[], entry_plan=[],
            unfilled=[e for g in groups for e in g.entries],
        ), MRPDiagnostics(n_unfilled=n_missing, floor=floor_diag)
    # Pool indices -> candidate-axis positions, for `preassigned` in and picks
    # back out. Every downstream index is a position in `lineups` from here.
    pos_of = {int(orig): new for new, orig in enumerate(keep_idx)}
    preassigned = {
        cid: [pos_of[int(j)] for j in idxs if int(j) in pos_of]
        for cid, idxs in (preassigned or {}).items()
    }
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

    rng = np.random.default_rng(cfg.seed)
    scores_full = compute_lineup_scores(lineups, sim_results)             # (M, S)
    keep = _world_slice(scores_full.shape[1], cfg.max_sims_per_contest)
    cand_scores = np.ascontiguousarray(scores_full[:, keep])
    del scores_full
    indicator = _lineup_indicator_matrix(lineups, sim_results.player_ids)

    sim_matrix = sim_results.results_matrix.astype(np.float32)[keep]
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    own = players_df["ownership"].to_numpy(dtype=np.float64)

    sim = ContestSimulator()
    field_pool = sim.generate_field(
        players_df, own, n_lineups=min(cfg.field_pool_size, _FIELD_POOL_CAP),
        rng_seed=cfg.seed, stop_check=stop_check,
    )
    field_pool_scores = sim.score_field(field_pool, sim_matrix, col_map)   # (S, F_pool)
    del sim_matrix
    F_pool = field_pool_scores.shape[1]

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
    portfolio, entry_plan, unfilled = [], [], []
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
        unfilled.extend(g.entries[n_pre + len(picks):])

    diag = MRPDiagnostics(
        n_unfilled=len(unfilled),
        floor=floor_diag,
        relaxations=[{"contest_id": r.contest_id, "rule": r.rule,
                      "frm": r.frm, "to": r.to, "step": r.step}
                     for r in res.relaxations],
        preflight=preflight,
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
                              unfilled=unfilled), diag


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
