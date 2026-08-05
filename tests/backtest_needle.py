"""Needle-in-a-haystack backtest: how often does a selection approach's
submitted portfolio contain one of the pool's own top-10-by-real-score
lineups?

The external-pool pipeline culls/diversifies thousands of SaberSim candidate
lineups down to a small submitted portfolio. The objective this measures
isn't mean EV -- it's "did we catch at least one of the 10 best lineups the
pool could possibly have produced", where "best" is realized score (real
per-player FPTS, known only after the slate concludes) *within that same
pool*, not a real DK contest field. So a needle's identity needs nothing but
the pool's lineups and real per-player FPTS -- no payout table, no contest
field size.

SLATE UNIVERSE: checked every archive dir directly, not assumed.

  Tier 1 (TIER1_SLATES) -- named per-contest standings zips present (the same
  9 slates as tests/bt_core.py's BACKTEST_SLATES). Every arm runs here,
  including the contest-relative ones (p_win_rank, prod_p_win).

  Tier 2 (TIER2_SLATES) -- only a generic contest-standings-*.zip (no named
  per-contest zips, so no real payout table), but the pool CSV and
  portfolio_sweep_draftkings.json (real budget) are both present. Real FPTS
  is still fully recoverable from the generic zip's Player/FPTS side table
  (see bt_core.discover_fpts_zips) -- just not split per real contest, so
  only the contest-agnostic arms run here.

  Excluded: 07172026 (no lineups_*.csv at all -- no haystack), 07182026 (no
  DKSalaries/pool/budget), 07222026e (pool present but no standings zip and
  no portfolio_sweep -- no ground truth). 07232026 has no archive dir.

PPD-flagged slates (data/slate_exclusions.json's game_ppd_pcts, matched by
recomputing compute_slate_id from each slate's own DKSalaries game list):
07272026 (CLE@CIN 45%) and 08022026 (PHI@BAL 35%), both Tier 2. Per
tests/backtest.py's own precedent, no special grading logic is needed for
these -- real FPTS already reflects whatever actually happened that day --
they are simply reported as a separate ex-PPD slate subset alongside the
full-universe rate.

    source venv/bin/activate
    python tests/backtest_needle.py                  # all usable slates, full arm roster
    python tests/backtest_needle.py 07282026 07292026 # subset
    BT_ARMS=random,ceiling_rank,stack_max python tests/backtest_needle.py
    python tests/backtest_needle.py status            # per-slate tier/PPD/budget audit
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.bt_core import (  # noqa: E402
    LIVE_CFG, build_slate_context, discover_fpts_zips, load_real_contests,
    verify_slate,
)
from tests.backtest_lab import _rank_norm  # noqa: E402

OUT_DIR = PROJECT_ROOT / "tests" / "backtest_output"
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
NEEDLE_SUMMARY = OUT_DIR / "needle_summary.csv"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
SEEDS = [int(s) for s in os.environ.get("BT_SEEDS", "42").split(",")]
FLOOR_PCT = 30.0
ADMIT_N = 2000

TIER1_SLATES = [
    "07222026", "07242026", "07252026", "07262026",
    "07282026", "07292026", "07302026", "07312026",
    "08012026", "08032026",
]
TIER2_SLATES = ["07192026", "07202026", "07212026", "07272026", "08022026"]
ALL_SLATES = TIER1_SLATES + TIER2_SLATES

CONTEST_AGNOSTIC_ARMS = [
    "random", "proj_top", "ceiling_rank", "own_fade", "ceiling_contrarian",
    "coverage_light", "stack_max",
    # hybrids of the arms above
    "ceiling_stack", "coverage_ceiling", "ensemble_top3",
    # no-admit-window variants of the two strongest single currencies
    "ceiling_rank_noadmit", "stack_max_noadmit",
    # hard portfolio-level overlap cap, all 10 roster slots
    "overlap8", "overlap6", "overlap4", "overlap2",
    # same, restricted to the 8 hitter slots (excludes the 2 pitcher slots)
    "overlap_hit6", "overlap_hit4", "overlap_hit2",
    # sim-world coverage: outcome diversity, not composition diversity
    "sim_coverage",
    # k-means on a multi-axis structural feature vector, forced representation
    "cluster_diverse",
    # proportional spread across ownership deciles, whole-budget barbell
    "own_barbell",
]
CONTEST_RELATIVE_ARMS = ["p_win_rank", "prod_p_win"]
ALL_ARMS = CONTEST_AGNOSTIC_ARMS + CONTEST_RELATIVE_ARMS


# ---------------------------------------------------------------------------
# Slate bookkeeping: tier, budget, PPD
# ---------------------------------------------------------------------------

def slate_budget(d: Path) -> int:
    """Total real entries submitted that slate, pooled across every contest --
    one line off portfolio_sweep_draftkings.json (entry counts are
    risk-invariant, verified against every archived sweep file)."""
    sw = json.loads((d / "portfolio_sweep_draftkings.json").read_text())
    r1 = next((x for x in sw["sweep"] if x["risk"] == 1.0), sw["sweep"][0])
    return len(r1["lineups"])


def is_ppd_slate(d: Path) -> tuple[bool, dict]:
    """(flagged, game_ppd_pcts) via compute_slate_id(DKSalaries games) against
    data/slate_exclusions.json -- the same detector the live pipeline uses."""
    from src.api.slate_exclusions import EXCLUSIONS_PATH, compute_slate_id
    from src.ingestion.dk_slate import DraftKingsSlateIngestor

    slate_df = DraftKingsSlateIngestor(str(d / "DKSalaries.csv")).get_slate_dataframe()
    slate_id = compute_slate_id(sorted(set(slate_df["game"])))
    if not EXCLUSIONS_PATH.exists():
        return False, {}
    data = json.loads(EXCLUSIONS_PATH.read_text())
    for key, entry in data.items():
        if key.split(":", 1)[0] == slate_id:
            pcts = entry.get("game_ppd_pcts") or {}
            if pcts:
                return True, pcts
    return False, {}


def cmd_status() -> None:
    print(f"{'slate':>10}  {'tier':>4}  {'budget':>7}  ppd")
    for slate in ALL_SLATES:
        d = ARCHIVE_DIR / slate
        tier = 1 if slate in TIER1_SLATES else 2
        try:
            budget = slate_budget(d)
        except Exception as exc:
            budget = f"ERR({exc})"
        ppd, pcts = is_ppd_slate(d)
        print(f"{slate:>10}  {tier:>4}  {str(budget):>7}  {pcts if ppd else ''}")


# ---------------------------------------------------------------------------
# Per-slate context: pool, real FPTS, sim-derived currencies -- uniform across
# both tiers (no contest/payout data needed for anything here).
# ---------------------------------------------------------------------------

def load_pool_context(slate: str, seed: int = 42, calib: bool = False) -> dict:
    from src.api import external_pool as ep
    from src.api.pipeline import PipelineRunner
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    from src.models.copula import EmpiricalCopula
    from src.simulation.engine import SimulationEngine
    from src.simulation.results import SimulationResults

    d = ARCHIVE_DIR / slate
    found = ep.discover_external_files(str(d))
    if not found["lineups_paths"] or not found["projections_path"]:
        raise SystemExit(f"{slate}: no lineups_*.csv / projections CSV pair found.")

    slate_df = DraftKingsSlateIngestor(str(d / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    if not pool.lineups:
        raise SystemExit(f"{slate}: every lineup dropped (unknown player ids).")
    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )

    # Real per-player FPTS: discover_fpts_zips works whether this slate has
    # named per-contest zips (Tier 1) or only the generic
    # contest-standings-*.zip (Tier 2) -- verify_slate/resolve_duplicate_names
    # only ever read `contest_id` off these dicts (see bt_core.py).
    real = discover_fpts_zips(d)
    raw = pd.read_csv(d / "DKSalaries.csv")
    nm = raw[["ID", "Name"]].astype({"ID": str})
    fpts = verify_slate(d, real, nm)
    actual = np.array([
        sum(fpts.get(int(p), float("nan")) for p in lu.player_ids)
        for lu in pool.lineups
    ], dtype=np.float64)
    ok = np.isfinite(actual)

    copula = EmpiricalCopula(str(PROJECT_ROOT / LIVE_CFG["paths"]["copula"]))
    if calib:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=True, scratch_prob=0.02,
            mean_calib_batter=0.88, mean_calib_pitcher=1.0,
        )
    else:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=False, scratch_prob=0.0,
            mean_calib_batter=1.0, mean_calib_pitcher=1.0,
        )
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)

    # Same cache key convention as bt_core.build_slate_context, so Tier-1
    # slates hit the oracle's already-built sim cache instead of resimulating.
    SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SIM_CACHE_DIR / f"{slate}_{N_SIMS}_{seed}_calib{calib}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
    else:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(N_SIMS)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache_path,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )

    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
    proj_score = ep.compute_pool_proj_scores(pool.lineups, players_df)
    own_sum = ep.compute_pool_ownership(pool.lineups, players_df)
    sim_p99 = np.percentile(lineup_scores, 99, axis=1)

    return dict(
        slate=slate, dir=d, pool=pool, players_df=players_df,
        actual=actual, ok=ok, proj_score=proj_score, own_sum=own_sum,
        sim_p99=sim_p99, lineup_scores=lineup_scores,
        budget=slate_budget(d),
    )


def needle_indices(ctx: dict, n: int = 10) -> np.ndarray:
    idx = np.where(ctx["ok"])[0]
    return idx[np.argsort(-ctx["actual"][idx])[:n]]


def primary_stack_teams(ctx: dict) -> np.ndarray:
    """Each lineup's largest same-team hitter group's team -- the categorical
    label select_stack_diverse maximizes distinctness over."""
    players_df = ctx["players_df"]
    team_by_id = players_df.set_index("player_id")["team"].astype(str)
    pos_by_id = players_df.set_index("player_id")["position"].astype(str)
    out = np.empty(len(ctx["pool"].lineups), dtype=object)
    for i, lu in enumerate(ctx["pool"].lineups):
        pids = [int(p) for p in lu.player_ids]
        is_pitcher = (pos_by_id.reindex(pids).fillna("") == "P").to_numpy()
        teams = team_by_id.reindex(pids)[~is_pitcher]
        vc = teams.value_counts()
        out[i] = vc.index[0] if len(vc) else ""
    return out


def lineup_salary_and_maxstack(ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """(salary_sum, max_stack) per pool lineup -- the two structural features
    select_cluster_diverse's feature vector needs beyond what's already in
    ctx (proj_score, own_sum, sim_p99)."""
    players_df = ctx["players_df"]
    sal_by_id = players_df.set_index("player_id")["salary"].astype(float)
    team_by_id = players_df.set_index("player_id")["team"].astype(str)
    pos_by_id = players_df.set_index("player_id")["position"].astype(str)
    M = len(ctx["pool"].lineups)
    salary_sum = np.zeros(M)
    max_stack = np.zeros(M)
    for i, lu in enumerate(ctx["pool"].lineups):
        pids = [int(p) for p in lu.player_ids]
        salary_sum[i] = sal_by_id.reindex(pids).sum()
        is_pitcher = (pos_by_id.reindex(pids).fillna("") == "P").to_numpy()
        vc = team_by_id.reindex(pids)[~is_pitcher].value_counts()
        max_stack[i] = vc.iloc[0] if len(vc) else 0
    return salary_sum, max_stack


def _kmeans(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm, numpy-only (no sklearn dependency in this venv)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for it in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            pts = X[labels == c]
            if len(pts):
                centers[c] = pts.mean(axis=0)
    return labels


def select_cluster_diverse(ctx: dict, sel: np.ndarray, cull: np.ndarray, *,
                           floor_pct=FLOOR_PCT, admit_n=ADMIT_N,
                           n_clusters=30) -> list[int]:
    """New arm: k-means (numpy Lloyd's algorithm, see _kmeans) on a per-
    lineup structural/distributional feature vector -- proj_score,
    ownership, salary, simulated ceiling and spread, max stack size --
    standardized, then max distinct-cluster coverage ranked by `sel` within
    the admit window. Same "force representation, then rank" shape as
    select_stack_diverse, but partitions on a genuinely different,
    multi-axis feature space instead of a single team label: two lineups
    from different teams can land in the same cluster (similar overall
    shape), and two same-team lineups can land in different clusters
    (different salary/ownership/ceiling profile)."""
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    if len(rem) == 0:
        return []
    k = min(k, len(rem))

    if "salary_sum" not in ctx:
        ctx["salary_sum"], ctx["max_stack"] = lineup_salary_and_maxstack(ctx)
    sim_std = ctx["lineup_scores"].std(axis=1)

    feats = np.stack([
        ctx["proj_score"][rem], ctx["own_sum"][rem], ctx["salary_sum"][rem],
        ctx["sim_p99"][rem], sim_std[rem], ctx["max_stack"][rem],
    ], axis=1).astype(np.float64)
    mu, sigma = feats.mean(axis=0), feats.std(axis=0)
    sigma[sigma == 0] = 1.0
    labels = _kmeans((feats - mu) / sigma, n_clusters)

    order = np.argsort(-sel[rem])
    picks: list[int] = []
    used_clusters: set = set()
    leftover: list[int] = []
    for i in order:
        c = int(labels[i])
        if c not in used_clusters:
            picks.append(int(rem[i]))
            used_clusters.add(c)
        else:
            leftover.append(int(rem[i]))
        if len(picks) == k:
            break
    if len(picks) < k:
        picks.extend(leftover[:k - len(picks)])
    return picks


def select_ownership_barbell(ctx: dict, sel: np.ndarray, cull: np.ndarray, *,
                             floor_pct=FLOOR_PCT, admit_n=ADMIT_N,
                             n_tiers=10) -> list[int]:
    """New arm: forced PROPORTIONAL spread across ownership deciles (roughly
    budget/n_tiers picks from each), ranked by `sel` within each tier --
    unlike own_fade (monotonic, always lowest-owned first) or stack_max/
    cluster_diverse (force representation only in the first ~15-30% of
    budget, then fall back to a flat ranking for the rest), this spends the
    ENTIRE budget maintaining a barbell across the ownership spectrum: some
    chalk, some contrarian, by construction, not just at the margin."""
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    if len(rem) == 0:
        return []
    k = min(k, len(rem))

    own = ctx["own_sum"][rem]
    edges = np.quantile(own, np.linspace(0, 1, n_tiers + 1))
    edges[-1] += 1e-9
    tier_id = np.clip(np.searchsorted(edges, own, side="right") - 1, 0, n_tiers - 1)

    base, extra = divmod(k, n_tiers)
    tier_budget = [base + (1 if t < extra else 0) for t in range(n_tiers)]

    picks: list[int] = []
    leftover: list[int] = []
    for t in range(n_tiers):
        in_tier = np.where(tier_id == t)[0]
        order = in_tier[np.argsort(-sel[rem[in_tier]])]
        take = tier_budget[t]
        picks.extend(int(rem[i]) for i in order[:take])
        leftover.extend(int(rem[i]) for i in order[take:])
    if len(picks) < k:
        leftover.sort(key=lambda i: -sel[i])
        picks.extend(leftover[: k - len(picks)])
    return picks[:k]


def composition_overlap_fn(ctx: dict):
    """Lazy per-admit-window player-overlap "correlation" for coverage_light,
    adapted from tests/backtest_lab.py's _composition_overlap_fn to take the
    pool's player_ids directly instead of an oracle {slate}_real.npz (which
    only exists for Tier-1 slates) -- same one-hot-incidence-matrix mechanism,
    works for either tier."""
    import scipy.sparse as sp

    pids = np.array([[int(p) for p in lu.player_ids] for lu in ctx["pool"].lineups])
    roster_size = pids.shape[1]
    uniq, inv = np.unique(pids, return_inverse=True)
    inv = np.asarray(inv).reshape(pids.shape)
    rows = np.repeat(np.arange(pids.shape[0]), roster_size)
    H = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, inv.ravel())),
        shape=(pids.shape[0], len(uniq)),
    )

    def overlap(rem: np.ndarray) -> np.ndarray:
        hs = H[rem]
        return np.asarray((hs @ hs.T).toarray(), dtype=np.float64) / roster_size

    return overlap


# ---------------------------------------------------------------------------
# Selection: one pooled "contest" of size ctx["budget"] (grading is confirmed
# pooled across contests anyway, and Tier-2 slates have no real per-contest
# split to fill from). Mirrors tests/backtest_lab.py's select_greedy body
# collapsed to a single contest -- evw<1.0 still delegates to the real
# DeterminantPortfolioSelector rather than reimplementing diversity.
# ---------------------------------------------------------------------------

def proj_floor_mask(ctx: dict, floor_pct: float) -> np.ndarray:
    proj = ctx["proj_score"]
    mask = ctx["ok"] & np.isfinite(proj)
    if floor_pct > 0:
        mask &= proj >= np.percentile(proj[np.isfinite(proj)], floor_pct)
    return mask


def pick_top(ctx: dict, sel: np.ndarray, cull: np.ndarray, *, floor_pct=FLOOR_PCT,
            admit_n=ADMIT_N, evw=1.0, corr_fn=None, rng=None) -> list[int]:
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    k = min(k, len(rem))
    if k == 0:
        return []
    if rng is not None:
        return list(map(int, rem[rng.choice(len(rem), size=k, replace=False)]))
    if evw >= 1.0:
        return list(map(int, rem[np.argsort(-sel[rem])[:k]]))
    from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
    sub_corr = corr_fn(rem)
    s = DeterminantPortfolioSelector(
        robust_payout=None, candidates=list(range(len(sel))), portfolio_size=k,
        risk=3.0, evw_base=evw, evw_max=evw, ev_floor=float("-inf"),
        precomputed=(rem, sel[rem].astype(np.float64), np.ascontiguousarray(sub_corr)),
    )
    return [int(i) for i, _ in s.select()]


def select_stack_diverse(ctx: dict, sel: np.ndarray, cull: np.ndarray, *,
                         floor_pct=FLOOR_PCT, admit_n=ADMIT_N) -> list[int]:
    """New arm: max distinct-primary-team coverage. Not expressible as a
    single EV-vector-plus-correlation call (it's a categorical-distinctness
    constraint over the whole portfolio, not a pairwise redundancy term), so
    it gets its own small selection loop rather than a pick_top call."""
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    if len(rem) == 0:
        return []
    order = rem[np.argsort(-sel[rem])]
    teams = ctx["stack_team"][order]
    picks: list[int] = []
    used_teams: set = set()
    leftover: list[int] = []
    for idx, team in zip(order, teams):
        if team not in used_teams:
            picks.append(int(idx))
            used_teams.add(team)
        else:
            leftover.append(int(idx))
        if len(picks) == k:
            break
    if len(picks) < k:
        picks.extend(leftover[:k - len(picks)])
    return picks


def select_sim_coverage(ctx: dict, sel: np.ndarray, cull: np.ndarray, *,
                        floor_pct=FLOOR_PCT, admit_n=ADMIT_N, top_k=10) -> list[int]:
    """New arm: greedy max-coverage of per-simulated-world top-`top_k`
    finishers within the pool's OWN simulated score matrix -- self-
    referential (no opponent field needed, unlike p_win). Targets outcome
    diversity directly (does the portfolio contain a top finisher in as many
    distinct plausible simulated worlds as possible) instead of composition
    (team/player overlap) as a proxy for it -- the natural next hypothesis
    given that no pre-slate feature tested so far (proj/own/ceiling/salary/
    stack shape) separates a true-surprise needle from noise: hedging across
    simulated *scenarios* doesn't need to know in advance which scenario is
    which.

    Standard greedy submodular max-coverage ((1-1/e) guarantee): each world
    is "covered" once any portfolio lineup is among its top-`top_k`
    simulated finishers; each step picks whichever remaining candidate adds
    the most newly-covered worlds. Falls back to ranking by `sel` once no
    candidate offers any further marginal coverage (this always exhausts
    well before `top_k * n_sims` worlds are covered in practice, since top-K
    membership concentrates in a shrinking set of high-mean lineups)."""
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    if len(rem) == 0:
        return []
    k = min(k, len(rem))

    sub_scores = ctx["lineup_scores"][rem]                          # (R, S)
    R, S = sub_scores.shape
    kk = min(top_k, R)
    topk_local = np.argpartition(-sub_scores, kk - 1, axis=0)[:kk]  # (kk, S), values in [0, R)

    remaining_gain = np.bincount(topk_local.ravel(), minlength=R).astype(np.int64)
    covered = np.zeros(S, dtype=bool)
    avail = np.ones(R, dtype=bool)
    picked_local: list[int] = []

    for _ in range(k):
        cand_gain = np.where(avail, remaining_gain, -1)
        best = int(np.argmax(cand_gain))
        if cand_gain[best] <= 0:
            break
        picked_local.append(best)
        avail[best] = False
        my_worlds = np.where((topk_local == best).any(axis=0) & ~covered)[0]
        if len(my_worlds):
            covered[my_worlds] = True
            np.subtract.at(remaining_gain, topk_local[:, my_worlds].ravel(), 1)

    if len(picked_local) < k:
        picked_set = set(picked_local)
        fallback = [i for i in np.argsort(-sel[rem]) if i not in picked_set]
        picked_local.extend(fallback[: k - len(picked_local)])

    return [int(rem[i]) for i in picked_local]


def select_overlap_capped(ctx: dict, sel: np.ndarray, cull: np.ndarray, *,
                          floor_pct=FLOOR_PCT, admit_n=ADMIT_N,
                          max_overlap=6, hitters_only=False) -> list[int]:
    """New arm: hard portfolio-level cap on shared players between any two
    picks (max_overlap out of 10 roster slots, or of the 8 hitter slots when
    hitters_only, excluding the 2 pitcher slots -- DK Classic MLB roster
    order here is P,P,C,1B,2B,3B,SS,OF,OF,OF, confirmed positional and
    consistent across every lineup in a pool).

    Different mechanism from stack_max/coverage_light's soft correlation
    term: this is a hard constraint checked against EVERY already-picked
    lineup, not a squared-correlation penalty. Walks the admit window in
    `sel` rank order, skipping (not permanently discarding) any candidate
    that would exceed the cap against some already-picked lineup; if the cap
    leaves the budget under-filled once the ranked window is exhausted, the
    remainder is topped up ignoring the cap (in `sel` order) rather than
    leaving the portfolio smaller than its real budget.
    """
    mask = proj_floor_mask(ctx, floor_pct)
    k = min(ctx["budget"], int(mask.sum()))
    if k <= 0:
        return []
    rem = np.where(mask & np.isfinite(sel))[0]
    if admit_n > 0 and len(rem) > admit_n:
        rem = np.sort(rem[np.argsort(-cull[rem])[:admit_n]])
    if len(rem) == 0:
        return []
    order = rem[np.argsort(-sel[rem])]
    lo, hi = (2, 10) if hitters_only else (0, 10)
    lineups = ctx["pool"].lineups
    id_sets = {int(i): frozenset(int(p) for p in lineups[i].player_ids[lo:hi]) for i in order}

    picks: list[int] = []
    picked_sets: list[frozenset] = []
    leftover: list[int] = []
    for idx in order:
        cand = id_sets[int(idx)]
        if all(len(cand & p) <= max_overlap for p in picked_sets):
            picks.append(int(idx))
            picked_sets.append(cand)
        else:
            leftover.append(int(idx))
        if len(picks) == k:
            break
    if len(picks) < k:
        picks.extend(leftover[:k - len(picks)])
    return picks


def select_ensemble(ctx: dict, sub_arms: list[str], *, rng=None) -> list[int]:
    """New arm: split the budget evenly across a few strategies instead of
    committing it all to one currency's ranking. The individual arms miss on
    different slates (see the arm x slate hit pivot), so a strategy-diverse
    split can cover more of the needle set than any single one, at the cost
    of each sub-block being less deep into its own currency's ranking."""
    n = len(sub_arms)
    saved_budget = ctx["budget"]
    picks: set = set()
    try:
        for i, arm in enumerate(sub_arms):
            share = saved_budget // n + (1 if i < saved_budget % n else 0)
            ctx["budget"] = share
            picks.update(run_arm(ctx, arm, rng=rng))
    finally:
        ctx["budget"] = saved_budget
    return list(picks)


# ---------------------------------------------------------------------------
# Contest-relative arms (Tier 1 only): faithful production replay of p_win,
# needing real named per-contest zips for a field size / payout shape.
# ---------------------------------------------------------------------------

def build_pwin_context(slate: str, seed: int, calib: bool = False, want_corr: bool = True) -> dict:
    """Built once per (slate, seed) and shared by both p_win_rank and
    prod_p_win -- the expensive part (real field generation, and for
    prod_p_win the (M,M) correlation matrix) must not be paid twice per
    slate just because two arms both want it."""
    d = ARCHIVE_DIR / slate
    real = load_real_contests(d)
    return build_slate_context(
        d, seed, calib, real, n_sims=N_SIMS, sharpness=SHARPNESS,
        sim_cache_dir=SIM_CACHE_DIR, want_corr=want_corr, want_pwin=True,
    )


def pwin_picks(ctx: dict, arm: str) -> list[int]:
    proj = ctx["proj_scores"]
    mask = np.isfinite(proj)
    if mask.sum() > 0:
        floor = np.percentile(proj[mask], FLOOR_PCT)
        mask &= proj >= floor
    picks: set = set()
    for c in ctx["contests"]:
        k = int(c["k"])
        if k <= 0:
            continue
        cid = c["contest_id"]
        cull, sel = ctx["p_win_cull"][cid], ctx["p_win_select"][cid]
        rem = np.where(mask & np.isfinite(sel))[0]
        if len(rem) > ADMIT_N:
            rem = np.sort(rem[np.argsort(-cull[rem])[:ADMIT_N]])
        kk = min(k, len(rem))
        if kk == 0:
            continue
        if arm == "p_win_rank":
            chosen = rem[np.argsort(-sel[rem])[:kk]]
        else:
            from src.optimization.gpp_portfolio import DeterminantPortfolioSelector
            sub_corr = ctx["corr"][np.ix_(rem, rem)]
            s = DeterminantPortfolioSelector(
                robust_payout=None, candidates=list(range(len(sel))), portfolio_size=kk,
                risk=3.0, evw_base=0.25, evw_max=0.25, ev_floor=float("-inf"),
                precomputed=(rem, sel[rem].astype(np.float64), np.ascontiguousarray(sub_corr)),
            )
            chosen = np.array([i for i, _ in s.select()], dtype=np.int64)
        picks.update(int(i) for i in chosen)
        mask[chosen] = False
    return sorted(picks)


# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------

def run_arm(ctx: dict, arm: str, *, rng=None) -> list[int]:
    proj = ctx["proj_score"]
    if arm == "random":
        return pick_top(ctx, proj, proj, admit_n=0, rng=rng)
    if arm == "proj_top":
        return pick_top(ctx, proj, proj)
    if arm == "ceiling_rank":
        return pick_top(ctx, ctx["sim_p99"], ctx["sim_p99"])
    if arm == "own_fade":
        neg_own = -ctx["own_sum"]
        return pick_top(ctx, neg_own, neg_own)
    if arm == "ceiling_contrarian":
        cc = _rank_norm(ctx["sim_p99"]) - _rank_norm(ctx["own_sum"])
        return pick_top(ctx, cc, cc)
    if arm == "coverage_light":
        comp_fn = composition_overlap_fn(ctx)
        return pick_top(ctx, proj, proj, evw=0.0, corr_fn=comp_fn)
    if arm == "stack_max":
        if "stack_team" not in ctx:
            ctx["stack_team"] = primary_stack_teams(ctx)
        return select_stack_diverse(ctx, proj, proj)
    if arm == "ceiling_stack":
        if "stack_team" not in ctx:
            ctx["stack_team"] = primary_stack_teams(ctx)
        return select_stack_diverse(ctx, ctx["sim_p99"], ctx["sim_p99"])
    if arm == "coverage_ceiling":
        comp_fn = composition_overlap_fn(ctx)
        return pick_top(ctx, ctx["sim_p99"], ctx["sim_p99"], evw=0.3, corr_fn=comp_fn)
    if arm == "ensemble_top3":
        return select_ensemble(ctx, ["stack_max", "ceiling_rank", "coverage_light"])
    if arm == "ceiling_rank_noadmit":
        return pick_top(ctx, ctx["sim_p99"], ctx["sim_p99"], admit_n=0)
    if arm == "stack_max_noadmit":
        if "stack_team" not in ctx:
            ctx["stack_team"] = primary_stack_teams(ctx)
        return select_stack_diverse(ctx, proj, proj, admit_n=0)
    if arm.startswith("overlap_hit"):
        cap = int(arm[len("overlap_hit"):])
        return select_overlap_capped(ctx, ctx["sim_p99"], ctx["sim_p99"],
                                     max_overlap=cap, hitters_only=True)
    if arm.startswith("overlap"):
        cap = int(arm[len("overlap"):])
        return select_overlap_capped(ctx, ctx["sim_p99"], ctx["sim_p99"],
                                     max_overlap=cap, hitters_only=False)
    if arm == "sim_coverage":
        return select_sim_coverage(ctx, proj, proj)
    if arm == "cluster_diverse":
        return select_cluster_diverse(ctx, proj, proj)
    if arm == "own_barbell":
        return select_ownership_barbell(ctx, proj, proj)
    raise ValueError(f"not a contest-agnostic arm: {arm}")


def grade_arm(needles: np.ndarray, picks: list[int]) -> dict:
    hit_idx = set(picks) & set(needles.tolist())
    best_rank = None
    if hit_idx:
        pos = {int(v): r + 1 for r, v in enumerate(needles)}
        best_rank = min(pos[i] for i in hit_idx)
    return {
        "needle_hit": bool(hit_idx),
        "n_needles_hit": len(hit_idx),
        "best_needle_rank": best_rank,
        "budget_used": len(picks),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def jeffreys_ci(hits: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    lo = beta.ppf(0.025, hits + 0.5, n - hits + 0.5) if n > 0 else 0.0
    hi = beta.ppf(0.975, hits + 0.5, n - hits + 0.5) if n > 0 else 0.0
    return float(lo), float(hi)


def append_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if NEEDLE_SUMMARY.exists():
        df = pd.concat([pd.read_csv(NEEDLE_SUMMARY), df], ignore_index=True)
    df.to_csv(NEEDLE_SUMMARY, index=False)


def print_report(df: pd.DataFrame) -> None:
    ppd_slates = set(df.loc[df["is_ppd_slate"], "slate"])
    print("\n-- slate x arm hit pivot --")
    piv = df.pivot_table(index="slate", columns="arm", values="needle_hit",
                         aggfunc="max")
    # pandas' DataFrame.replace hits an internal block-manager bug on some
    # single-row/single-dtype pivots (IndexError in replace_list) -- build
    # the Y/./blank display directly instead of routing through .replace().
    disp = np.where(piv.isna(), "", np.where(piv.fillna(False).to_numpy(), "Y", "."))
    print(pd.DataFrame(disp, index=piv.index, columns=piv.columns).to_string())

    print("\n-- rate (all vs ex-PPD), Jeffreys 95% CI --")
    rows = []
    for arm, g in df.groupby("arm"):
        by_slate = g.groupby("slate")["needle_hit"].max()
        n_all, hits_all = len(by_slate), int(by_slate.sum())
        lo_a, hi_a = jeffreys_ci(hits_all, n_all)
        ex = by_slate[~by_slate.index.isin(ppd_slates)]
        n_ex, hits_ex = len(ex), int(ex.sum())
        lo_e, hi_e = jeffreys_ci(hits_ex, n_ex)
        rows.append({
            "arm": arm, "n_all": n_all, "hits_all": hits_all,
            "rate_all": hits_all / n_all if n_all else float("nan"),
            "ci_all": f"[{lo_a:.2f},{hi_a:.2f}]",
            "n_expd": n_ex, "hits_expd": hits_ex,
            "rate_expd": hits_ex / n_ex if n_ex else float("nan"),
            "ci_expd": f"[{lo_e:.2f},{hi_e:.2f}]",
        })
    rate_df = pd.DataFrame(rows).sort_values("rate_all", ascending=False)
    print(rate_df.to_string(index=False))

    if "random" in set(df["arm"]):
        print("\n-- paired bootstrap vs random (resampled by slate) --")
        base = df[df["arm"] == "random"].groupby("slate")["needle_hit"].max()
        rng = np.random.default_rng(0)
        boot_rows = []
        for arm, g in df.groupby("arm"):
            if arm == "random":
                continue
            by_slate = g.groupby("slate")["needle_hit"].max()
            common = sorted(set(by_slate.index) & set(base.index))
            if not common:
                continue
            a = by_slate.loc[common].to_numpy(dtype=float)
            b = base.loc[common].to_numpy(dtype=float)
            d = a - b
            n = len(d)
            bs = d[rng.integers(0, n, size=(20000, n))].mean(axis=1)
            boot_rows.append({
                "arm": arm, "n_slates": n, "mean_delta": float(d.mean()),
                "d_lo95": float(np.percentile(bs, 2.5)),
                "d_hi95": float(np.percentile(bs, 97.5)),
            })
        print(pd.DataFrame(boot_rows).to_string(index=False))


def _done_keys() -> set:
    """{(int(slate), arm, seed)} already recorded in needle_summary.csv --
    lets a run resume after an interrupt/timeout instead of redoing (and
    re-appending duplicate rows for) work that's already on disk. slate is
    normalized to int since a plain DataFrame/CSV round-trip silently drops
    the leading zero (07242026 -> 7242026) on the existing data."""
    if not NEEDLE_SUMMARY.exists():
        return set()
    df = pd.read_csv(NEEDLE_SUMMARY)
    return set(zip(df["slate"].astype(int), df["arm"], df["seed"].astype(int)))


def main() -> None:
    if sys.argv[1:2] == ["status"]:
        cmd_status()
        return

    arms = os.environ["BT_ARMS"].split(",") if os.environ.get("BT_ARMS") else ALL_ARMS
    slates = [s for s in sys.argv[1:] if s.isdigit()] or ALL_SLATES
    done = set() if os.environ.get("BT_FORCE") else _done_keys()

    rows = []
    run_ts = pd.Timestamp.utcnow().isoformat()
    for slate in slates:
        d = ARCHIVE_DIR / slate
        tier = 1 if slate in TIER1_SLATES else 2
        try:
            budget = slate_budget(d)
        except Exception as exc:
            print(f"  {slate}: skipped, can't read budget ({exc})")
            continue
        ppd, pcts = is_ppd_slate(d)

        t0 = time.time()
        ctx_cache: dict = {}
        slate_rows = []
        for seed in SEEDS:
            arms_here = [a for a in arms if a in CONTEST_AGNOSTIC_ARMS
                        and (int(slate), a, seed) not in done]
            pwin_arms_here = [a for a in arms if a in CONTEST_RELATIVE_ARMS and tier == 1
                              and (int(slate), a, seed) not in done]
            if not arms_here and not pwin_arms_here:
                continue

            ctx = ctx_cache.get(seed)
            if (arms_here or pwin_arms_here) and ctx is None:
                ctx = load_pool_context(slate, seed=seed, calib=False)
                ctx_cache[seed] = ctx
            needles = needle_indices(ctx) if ctx is not None else None

            for arm in arms_here:
                rng = np.random.default_rng(seed) if arm == "random" else None
                picks = run_arm(ctx, arm, rng=rng)
                grade = grade_arm(needles, picks)
                slate_rows.append({
                    "slate": slate, "arm": arm, "seed": seed, "budget_n": budget,
                    **grade, "is_ppd_slate": ppd, "ppd_detail": json.dumps(pcts),
                    "tier": tier, "run_ts": run_ts,
                })

            if pwin_arms_here:
                # Built once per (slate, seed) and shared by both p_win_rank
                # and prod_p_win -- see build_pwin_context's docstring.
                pctx = build_pwin_context(
                    slate, seed, want_corr=("prod_p_win" in pwin_arms_here),
                )
                for arm in pwin_arms_here:
                    picks = pwin_picks(pctx, arm)
                    grade = grade_arm(needles, picks)
                    slate_rows.append({
                        "slate": slate, "arm": arm, "seed": seed, "budget_n": budget,
                        **grade, "is_ppd_slate": ppd, "ppd_detail": json.dumps(pcts),
                        "tier": tier, "run_ts": run_ts,
                    })
        if slate_rows:
            print(f"  {slate}: {time.time() - t0:.0f}s", flush=True)
            append_summary(slate_rows)
            rows.extend(slate_rows)
        else:
            print(f"  {slate}: already done, skipped", flush=True)

    if not rows:
        print("no rows produced")
        return
    df = pd.DataFrame(rows)
    print_report(df)
    print(f"\nappended {len(rows)} rows -> {NEEDLE_SUMMARY}")


if __name__ == "__main__":
    main()
