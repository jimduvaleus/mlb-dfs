"""Select (not generate) a needlunchmoney-emulating 150-lineup portfolio out
of SaberSim's own pre-built lineup export pool, and grade it against that
contest's REAL archived field.

The from-scratch approach (`scripts/emulate_needlunchmoney.py`, tuning
`CandidateGenerator`'s ownership-fade/GROUP_FRACTIONS) worked but fought the
generator's own construction mechanics the whole way, and even where
structural fidelity was good, never closed the real dollar gap. This script
takes a different approach: SELECT 150 lineups out of the archived
SaberSim lineup pool (~8,000 candidates/slate after dedup) with a bespoke
greedy selector whose objective is DOMINATED by matching needlunchmoney's
revealed portfolio statistics, using the pool's own EV signals only as a
secondary/floor input -- per the reasoning that a naive top-EV read of the
same pool is presumably available to anyone and wouldn't explain his edge.

Reused unchanged from `scripts/emulate_needlunchmoney.py`: TARGET,
FIELD_BASELINE_OWN, find_target_contest, field_players_for_contest,
measure_structure, print_structure_comparison, grade_our_portfolio,
resolve_name_to_id, the incremental-checkpoint-and-resume main() pattern.

Reused unchanged from `src/api/external_pool.py`: discover_external_files,
parse_lineup_pool (dedup: exact player-set match, then 9/10-overlap
near-duplicate removal -- already correct, do not reimplement),
parse_player_projections, build_external_players_df.

Genuinely new: the feature-recovery step (parse_lineup_pool discards
Proj Score/percentiles/Ownership/EV-bucket columns, keeping only player
IDs -- re-read the raw CSVs positionally to recover them, joined back onto
the deduped candidates by player-id-set), the per-candidate feature matrix,
and the greedy multi-term selector itself.

IMPORTANT: the selector's steering ownership signal is PROJECTED ownership
("My Own", from `parse_player_projections`), never realized/post-contest
ownership -- a real bettor only has projected ownership at lock time, and
this session's own earlier pitcher-usage-by-ownership-quartile analysis
(which found real pros touch 100% of confirmed starters, steeply weighted
by ownership) was itself built on projected ownership. Realized ownership
is used only downstream, in the reused `measure_structure`/reporting step,
matching how the targets were themselves originally derived.

Usage
-----
    source venv/bin/activate
    python scripts/select_needlunchmoney_pool.py --slate 08032026 --dry-run
    python scripts/select_needlunchmoney_pool.py
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.external_pool import (  # noqa: E402
    discover_external_files, parse_lineup_pool, parse_player_projections,
    build_external_players_df,
)
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.optimization.lineup import Lineup  # noqa: E402

from tests.bt_core import BACKTEST_SLATES  # noqa: E402
from scripts.emulate_needlunchmoney import (  # noqa: E402
    TENTH_SLATE, TARGET, FIELD_BASELINE_OWN,
    find_target_contest, field_players_for_contest, measure_structure,
    grade_our_portfolio,
)
from scripts.analyze_rival_portfolio import (  # noqa: E402
    pitcher_names, overlap_profile, team_map, primary_teams,
)

# Split hitter/pitcher ownership targets (both REALIZED, matching how
# chalk_index/FIELD_BASELINE_OWN were derived -- see module docstring for
# why the selector itself steers on PROJECTED ownership, an approximation
# to these realized-derived numbers).
TARGET_HITTER_OWN = 6.5
TARGET_PITCHER_OWN = 27.9
TARGET_MEAN_OVERLAP = TARGET["mean_overlap"]
TARGET_GROUP_FRACTIONS = TARGET["group_fractions"]
# Pooled fallbacks for the top-of-order / clustering terms (from
# scripts/analyze_player_pick_quality.py's 9-slate pooled measurement,
# 2026-08-06): his hitters average batting order 3.94 (vs field 4.11, our
# prior unweighted selection 4.41 -- 9/9 slates lower than ours) and his
# primary-stack order clustering (excess_gap, see that script's docstring)
# averages 2.370, significantly tighter than a random draw AND tighter than
# ours (2.536, p=0.0024). Real per-slate LOSO targets are preferred
# (load_needlunchmoney_actuals_loso); these are only the no-actuals fallback.
TARGET_MEAN_ORDER = 3.94
TARGET_EXCESS_GAP = 2.37
# Not a per-slate LOSO target (no per-day SS-Proj actuals collector exists
# yet) -- pooled reference from scripts/analyze_sp_signal.py's 10-slate Part
# 2 measurement, for the diagnostic print only.
REFERENCE_HIS_PITCHER_SS_PROJ = 16.65
# Both real pros' averaged pitcher usage-count-per-150 rate, by ownership
# quartile among CONFIRMED STARTERS ONLY (see module docstring / this
# session's pitcher-usage-by-ownership-quartile analysis).
# Superseded design, kept for reference/comparison: a flat per-quartile
# target (every pitcher IN a quartile got the SAME absolute target count)
# turned out badly flawed two ways -- (1) it doesn't scale with pool size
# (a fixed 47.5-per-pitcher target for ~5 top-quartile pitchers can exceed
# the whole slate's 300-slot pitcher budget on a big slate, or leave slack
# unused on a small one), and (2) it can't concentrate WITHIN a quartile
# (the single best-owned arm and the 5th-best "top25%" arm got identical
# targets, when real usage almost certainly concentrates hard on the very
# best arm). Replaced by PITCHER_COVERAGE_ALPHA below: target_count[pid]
# is now proportional to own[pid]**ALPHA, normalized to sum to exactly
# 2*portfolio_size across the slate's own confirmed-starter pool -- this
# self-normalizes to both pool size AND the slate's own achievable
# ownership range (a real, confirmed problem on 07222026: the pool's max
# achievable pitcher ownership was 24.5%, below TARGET_PITCHER_OWN=27.9,
# making that absolute target structurally unreachable regardless of
# weight -- a share-based target has no such failure mode).
# PITCHER_QUARTILE_TARGET_RATE = {
#     "bottom25": 1.8 / 150, "25-50": 1.9 / 150, "50-75": 13.3 / 150, "top25": 47.5 / 150,
# }
PITCHER_COVERAGE_ALPHA = 3.0
# Independent exponent for the SS-Proj quality basis in
# compute_pitcher_target_shares -- 0 disables it (pure ownership, the
# original behavior). NOT a blend weight: a linear own/quality blend was
# tried first and confirmed not to work (see that function's docstring).
# Set via scripts/sweep_pitcher_quality_blend.py, 2026-08-06: alpha_own
# calibration hits its 0.0 floor and pitcher_own runs away once quality and
# ownership are correlated enough on a given slate (07302026: alpha=4.0 ->
# pitcher_own 35.1 vs target 21.9) -- 2.0 was the best balance found across
# 3 probed slates: real gains on 2/3 (+13.3%/+2.9% relative SS-Proj vs
# baseline), one still right on its ownership target (08032026: 23.75 vs
# 23.76), one accepting a real but bounded overshoot (07302026: 29.87 vs
# 21.93) -- per explicit user direction that ownership precision may give
# ground for quality.
PITCHER_QUALITY_ALPHA = 2.0
LARGE_SLATE_MIN_GAMES = 5

DEFAULT_WEIGHTS = {
    "ev": 0.5,
    "hitter_own": 7.0,  # overridden per-slate by calibrate_hitter_own_weight when calibrate=True
    "pitcher_coverage": 3.0,
    "diversity": 1.0,
    "team": 0.05,
    "stack": 0.5,
    "pitcher_pair": 60.0,
    # order needed a much heavier weight than cluster to move the needle:
    # sweep found weight=20 nearly exact-matches LOSO target mean_order on
    # 2/3 probed slates (07302026 4.23->3.94 exact; 08032026 4.64->3.92 vs
    # target 3.90) and closes most of the gap on the third (07252026
    # 4.48->4.15 vs target 3.92). cluster saturates near its target by
    # weight~2 already -- pushing it higher (5-10) doesn't tighten
    # excess_gap further and drags mean_order the wrong way as a side
    # effect (08032026: cluster_w=10 -> mean_order 4.72 vs 4.64 at
    # cluster_w=2, same excess_gap either way), so left at the prior
    # default. Any residual order/hitter_own interaction gets absorbed
    # automatically by calibrate_hitter_own_weight, which runs against
    # these order/cluster weights already fixed.
    "order": 20.0,
    "cluster": 2.0,
}


# ---------------------------------------------------------------------------
# Per-slate actual targets (NOT the pooled 10-slate/443-handle averages in
# TARGET/TARGET_HITTER_OWN/TARGET_PITCHER_OWN/TARGET_MEAN_OVERLAP)
# ---------------------------------------------------------------------------

def load_needlunchmoney_actuals(archive_dir: Path, contest_name: str) -> dict:
    """needlunchmoney's ACTUAL stats on this ONE slate/contest -- the right
    calibration/measurement benchmark, confirmed directly this session: on
    07222026 our selector's chalk_index (-1.61) looked like a bad miss
    against the pooled TARGET (-0.84), but his own REAL chalk_index that
    exact slate was -1.565 (nearly identical to ours) -- the pooled -0.84
    is a 10-slate average and was simply the wrong target for that slate,
    since 07222026 ran contrarian for the WHOLE FIELD that day, him
    included. Some slates are naturally chalkier/flatter/more spread for
    everyone; comparing against a fixed cross-slate (or, for mean_overlap,
    cross-slate-and-cross-pool-size) constant chases a phantom gap instead
    of the real one. Returns {} (caller falls back to the pooled TARGET
    constants) if no needlunchmoney rows exist for this slate/contest.
    """
    lineups_csv = PROJECT_ROOT / "outputs" / "profitable_entrants_lineups.csv"
    if not lineups_csv.exists():
        return {}
    df = pd.read_csv(lineups_csv, dtype={"slate": str})
    mine = df[(df.handle == "needlunchmoney") & (df.slate == archive_dir.name)
              & (df.contest == contest_name)].copy()
    if mine.empty:
        return {}
    mine["names"] = mine["names"].apply(ast.literal_eval)

    chalk_index = float(mine["realized_own_mean"].mean() - FIELD_BASELINE_OWN)
    lus = list(mine["names"])
    ov = overlap_profile(lus)

    pitchers = pitcher_names(archive_dir)
    tmap = team_map(archive_dir)
    found = discover_external_files(str(archive_dir))
    raw = pd.read_csv(found["projections_path"])
    name_to_own = raw.set_index("Name")["My Own"].to_dict()
    name_to_order = raw.set_index("Name")["Order"].to_dict()
    hitter_owns, pitcher_owns = [], []
    hitter_orders = []
    for names in lus:
        h = [n for n in names if n not in pitchers]
        p = [n for n in names if n in pitchers]
        hitter_owns.append(np.nanmean([name_to_own.get(n, np.nan) for n in h]))
        pitcher_owns.append(np.nanmean([name_to_own.get(n, np.nan) for n in p]))
        hitter_orders.extend(name_to_order.get(n, np.nan) for n in h)

    primaries = primary_teams(lus, tmap, pitchers)
    n_primary_teams = len(set(t for t in primaries if t))
    pitcher_pair = 0
    excess_gaps = []
    for names, prim in zip(lus, primaries):
        if prim and any(n in pitchers and tmap.get(n) == prim for n in names):
            pitcher_pair += 1
        if prim:
            orders = sorted(name_to_order.get(n) for n in names
                             if n not in pitchers and tmap.get(n) == prim
                             and name_to_order.get(n) is not None and not np.isnan(name_to_order.get(n)))
            if len(orders) >= 2:
                excess_gaps.append((orders[-1] - orders[0]) - (len(orders) - 1))

    return {
        "chalk_index": chalk_index,
        "mean_overlap": float(ov["mean_overlap"]),
        "hitter_own": float(np.nanmean(hitter_owns)),
        "pitcher_own": float(np.nanmean(pitcher_owns)),
        "n_primary_teams": n_primary_teams,
        "pitcher_pair_rate": pitcher_pair / len(lus),
        "mean_order": float(np.nanmean(hitter_orders)),
        "mean_excess_gap": float(np.mean(excess_gaps)) if excess_gaps else float("nan"),
        "n": len(mine),
    }


def load_needlunchmoney_actuals_loso(current_slate: str) -> dict:
    """Leave-one-slate-out version of load_needlunchmoney_actuals: averages
    (n-weighted) needlunchmoney's real per-slate stats across every OTHER
    backtest slate, excluding `current_slate` entirely. This -- not the
    same-slate version -- is what run_one_slate must use for its
    construction/calibration targets: a real bettor never knows
    needlunchmoney's realized chalk_index/hitter_own/etc. for TODAY before
    locking lineups, so steering today's portfolio (or today's alpha/weight
    search) at today's own realized answer is look-ahead, full stop. Using
    his real submissions from OTHER days to inform today's target is not --
    that's the entire premise of "emulate his revealed statistical
    signature" -- so this stays deliberately rich (per-slate, n-weighted)
    rather than collapsing to the single pooled TARGET constant, which is
    itself a valid (if coarser) LOSO-safe fallback and is what an empty
    return here falls back to (see load_needlunchmoney_actuals's docstring
    for why the pooled constant alone is often the wrong per-slate target).
    The same-slate load_needlunchmoney_actuals() call stays in use
    elsewhere for POST-HOC reporting only (print_structure_vs_actuals),
    which is pure grading, not construction, and is fine to view in
    hindsight."""
    all_slates = list(BACKTEST_SLATES) + [TENTH_SLATE]
    rows = []
    for slate in all_slates:
        if slate == current_slate:
            continue
        adir = PROJECT_ROOT / "archive" / slate
        if not adir.exists():
            continue
        try:
            rc = find_target_contest(adir)
        except ValueError:
            continue
        a = load_needlunchmoney_actuals(adir, rc["contest"])
        if a:
            rows.append(a)
    if not rows:
        return {}
    total_n = sum(r["n"] for r in rows)
    out = {key: sum(r[key] * r["n"] for r in rows) / total_n
           for key in ("chalk_index", "mean_overlap", "hitter_own", "pitcher_own",
                        "pitcher_pair_rate", "n_primary_teams", "mean_order")}
    # mean_excess_gap can be NaN on a slate with no >=2-hitter primary stacks
    # (never observed but defensive); weight only over slates that have it.
    eg_rows = [r for r in rows if not np.isnan(r.get("mean_excess_gap", np.nan))]
    if eg_rows:
        eg_n = sum(r["n"] for r in eg_rows)
        out["mean_excess_gap"] = sum(r["mean_excess_gap"] * r["n"] for r in eg_rows) / eg_n
    else:
        out["mean_excess_gap"] = float("nan")
    out["n"] = total_n
    out["n_slates"] = len(rows)
    return out


# ---------------------------------------------------------------------------
# Stage 0: EV bucket resolution
# ---------------------------------------------------------------------------

def resolve_ev_column(n_games: int, n_field: int) -> str:
    size_tier = "Large Slate" if n_games >= LARGE_SLATE_MIN_GAMES else "Small Slate"
    if n_field < 1_000:
        bucket = "100-1k"
    elif n_field < 10_000:
        bucket = "1k-10k"
    elif n_field < 50_000:
        bucket = "10k-50k"
    else:
        bucket = "50k+"
    return f"{size_tier} | {bucket}"


# ---------------------------------------------------------------------------
# Stage 1: load pool + recover discarded feature columns
# ---------------------------------------------------------------------------

def load_pool(archive_dir: Path):
    found = discover_external_files(str(archive_dir))
    if not found["lineups_paths"]:
        raise FileNotFoundError(f"no lineup pool found in {archive_dir}")

    slate_df = DraftKingsSlateIngestor(str(archive_dir / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}

    pool = parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)

    proj_ext = parse_player_projections(found["projections_path"])
    pool_pids = {pid for lu in pool.lineups for pid in lu.player_ids}
    runner = PipelineRunner(config_path=str(PROJECT_ROOT / "config.yaml"))
    players_df = build_external_players_df(slate_df, proj_ext, pool_pids, runner._derive_opponent)

    raw = pd.read_csv(found["projections_path"])
    # SaberSim's Pos column is inconsistent across export files -- some days
    # "P", others "SP"/"RP" (same gotcha documented in
    # src/api/external_pool.py::parse_sabersim_projections).
    confirmed_mask = raw["Pos"].astype(str).isin(["P", "SP", "RP"]) & (raw["Status"] == "Confirmed")
    confirmed_starter_ids = set(
        pd.to_numeric(raw.loc[confirmed_mask, "DFS ID"], errors="coerce").dropna().astype(int)
    )

    # SaberSim's own raw point projection ("SS Proj", pre-blend, distinct
    # from "My Proj") -- not carried by parse_player_projections, so pulled
    # from `raw` (already loaded above) and merged onto the local proj_ext
    # copy as extra columns. Purely additive: doesn't change proj_ext's
    # existing columns/shape for any other caller. Motivation: real
    # confirmed starters he rosters run ~19-20% ahead of ours on SS Proj/K
    # despite similar ownership match -- see project-sp-signal-investigation
    # memory -- so pitcher_target_share needs a quality basis, not just
    # ownership.
    quality_cols = raw[["DFS ID", "SS Proj", "K"]].rename(
        columns={"DFS ID": "player_id", "SS Proj": "ss_proj", "K": "k_proj"})
    quality_cols["player_id"] = pd.to_numeric(quality_cols["player_id"], errors="coerce")
    quality_cols = quality_cols.dropna(subset=["player_id"])
    quality_cols["player_id"] = quality_cols["player_id"].astype(int)
    quality_cols = quality_cols.drop_duplicates(subset=["player_id"], keep="first")
    proj_ext = proj_ext.merge(quality_cols, on="player_id", how="left")

    # Feature recovery: re-read the raw pool CSV(s) POSITIONALLY for the 10
    # slot-id columns (pandas mangles duplicate P/P.1/OF/OF.1/OF.2 header
    # names but preserves column ORDER, so positional access is safe --
    # confirmed working precedent: analyze_rival_portfolio.py::_our_pool).
    # First-occurrence-in-file-order wins, replicating parse_lineup_pool's
    # own dedup identity rule so the feature row picked for a surviving
    # candidate is provably the same physical CSV row it came from.
    row_by_key = {}
    for p in found["lineups_paths"]:
        df = pd.read_csv(p)
        for row in df.itertuples(index=False):
            key = frozenset(int(row[j]) for j in range(10))
            if key not in row_by_key:
                row_by_key[key] = dict(zip(df.columns, row))

    feat_rows = [row_by_key[frozenset(lu.player_ids)] for lu in pool.lineups]
    features_df = pd.DataFrame(feat_rows)

    n_games = slate_df["game"].nunique()
    return pool, proj_ext, features_df, confirmed_starter_ids, n_games, players_df


# ---------------------------------------------------------------------------
# Stage 2: vectorized per-candidate feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(pool_lineups: list, features_df: pd.DataFrame, proj_ext: pd.DataFrame,
                          confirmed_starter_ids: set, ev_col: str) -> dict:
    id_to_team = proj_ext.set_index("player_id")["team"].to_dict()
    id_to_own = proj_ext.set_index("player_id")["ownership"].to_dict()  # projected, 0-100 scale
    id_to_order = proj_ext.set_index("player_id")["order"].to_dict()
    id_to_ss_proj = proj_ext.set_index("player_id")["ss_proj"].to_dict()

    pids_arr = np.array([lu.player_ids for lu in pool_lineups], dtype=np.int64)  # (M, 10)
    own_lookup = np.vectorize(lambda pid: id_to_own.get(int(pid), np.nan))
    own_arr = own_lookup(pids_arr).astype(np.float64)
    hitter_own = np.nanmean(own_arr[:, 2:10], axis=1)
    pitcher_own = np.nanmean(own_arr[:, 0:2], axis=1)

    ss_proj_lookup = np.vectorize(lambda pid: id_to_ss_proj.get(int(pid), np.nan))
    pitcher_ss_proj = np.nanmean(ss_proj_lookup(pids_arr[:, 0:2]).astype(np.float64), axis=1)

    order_lookup = np.vectorize(lambda pid: id_to_order.get(int(pid), np.nan))
    hitter_orders = order_lookup(pids_arr[:, 2:10]).astype(np.float64)  # (M, 8)
    hitter_mean_order = np.nanmean(hitter_orders, axis=1)

    team_lookup = np.vectorize(lambda pid: id_to_team.get(int(pid), ""))
    hitter_teams = team_lookup(pids_arr[:, 2:10])  # (M, 8)
    primary_team = np.empty(len(pool_lineups), dtype=object)
    primary_size = np.zeros(len(pool_lineups), dtype=np.int64)
    # Order clustering within the primary stack: (max-min)-(k-1), 0 for a
    # perfectly contiguous run like 3-4-5-6, growing with dispersion like
    # 1-2-5-7 -- see analyze_player_pick_quality.py's docstring. Computed in
    # the same per-row loop as primary_team/size (already O(M) Python, no
    # separate pass needed).
    primary_excess_gap = np.zeros(len(pool_lineups), dtype=np.float64)
    for i, row in enumerate(hitter_teams):
        vc = pd.Series(row).value_counts()
        t = vc.index[0]
        primary_team[i] = t
        primary_size[i] = int(vc.iloc[0])
        stack_orders = sorted(o for o, team in zip(hitter_orders[i], row) if team == t and not np.isnan(o))
        if len(stack_orders) >= 2:
            primary_excess_gap[i] = (stack_orders[-1] - stack_orders[0]) - (len(stack_orders) - 1)

    pitcher_teams = team_lookup(pids_arr[:, 0:2])  # (M, 2)
    pitcher_pair_flag = (pitcher_teams[:, 0] == primary_team) | (pitcher_teams[:, 1] == primary_team)

    # Ownership-proportional pitcher usage target, normalized to the slate's
    # own confirmed-starter pool -- see PITCHER_COVERAGE_ALPHA comment above
    # for why this replaced the flat-per-quartile scheme. conf_own (raw)
    # is kept in feat so callers can recompute shares at a different alpha
    # cheaply (see compute_pitcher_target_shares/calibrate_pitcher_coverage_
    # alpha) without rebuilding the whole feature matrix.
    conf_own = {pid: id_to_own[pid] for pid in confirmed_starter_ids if pid in id_to_own}
    conf_quality = {pid: id_to_ss_proj[pid] for pid in confirmed_starter_ids
                     if pid in id_to_ss_proj and not np.isnan(id_to_ss_proj[pid])}

    ev = pd.to_numeric(features_df[ev_col], errors="coerce").to_numpy()
    ev_norm = (ev - np.nanmin(ev)) / (np.nanmax(ev) - np.nanmin(ev) + 1e-9)
    ev_norm = np.nan_to_num(ev_norm, nan=0.0)

    n_teams_slate = len(set(id_to_team.values()) - {""})

    return {
        "pids_arr": pids_arr, "hitter_own": hitter_own, "pitcher_own": pitcher_own,
        "pitcher_ss_proj": pitcher_ss_proj,
        "primary_team": primary_team, "primary_size": primary_size,
        "hitter_mean_order": hitter_mean_order, "primary_excess_gap": primary_excess_gap,
        "conf_own": conf_own, "conf_quality": conf_quality,
        "pitcher_target_share": compute_pitcher_target_shares(
            conf_own, PITCHER_COVERAGE_ALPHA, conf_quality, PITCHER_QUALITY_ALPHA),
        "ev_norm": ev_norm,
        "n_teams_slate": n_teams_slate, "pitcher_pair_flag": pitcher_pair_flag,
    }


def compute_pitcher_target_shares(conf_own: dict, alpha_own: float, conf_quality: dict | None = None,
                                   alpha_quality: float = 0.0) -> dict:
    """{player_id: share of the 2*portfolio_size pitcher-slot budget}:
    weight[pid] = own_norm[pid]**alpha_own * quality_norm[pid]**alpha_quality
    (both min-max normalized within the confirmed-starter pool), normalized
    to sum to 1.0. alpha_quality=0 (the original default) reproduces pure-
    ownership behavior exactly, since quality_norm**0 == 1 for everyone.

    TWO INDEPENDENT exponents, not a single blended-then-exponentiated
    basis -- an earlier linear-blend version (`quality_blend`, own_norm and
    quality_norm mixed before exponentiating) was tried and confirmed NOT
    to work: SS Proj (SaberSim's raw point projection, the quality signal --
    see project-sp-signal-investigation memory) is far more tightly
    clustered among confirmed starters than ownership is, so after min-max
    normalizing, the quality basis comes out nearly flat. Blending a flat
    basis into ownership dilutes concentration rather than redirecting it,
    and re-sharpening via ONE shared alpha just converges back toward
    roughly the same ownership-driven picks -- measured directly (3-slate
    probe, 2026-08-06): the linear-blend version only closed 0.5-3.3% of
    the ~19% pitcher-quality gap, and pinning alpha fixed while raising
    blend made BOTH ownership and quality worse, not better. A second,
    independent alpha_quality avoids this: it can sharpen concentration on
    high-quality arms specifically, without being diluted by the flat
    quality basis or fighting the ownership exponent for the same knob.
    Cheap enough to recompute per alpha probe during calibration -- no need
    to touch the rest of the feature matrix.
    """
    if not conf_own:
        return {}
    conf_ids = list(conf_own.keys())
    own_vals = np.clip(np.array([conf_own[pid] for pid in conf_ids], dtype=np.float64), 1e-3, None)
    own_norm = own_vals / own_vals.max()
    own_component = own_norm ** alpha_own

    if conf_quality and alpha_quality > 0:
        q_vals = np.array([conf_quality.get(pid, np.nan) for pid in conf_ids], dtype=np.float64)
        q_valid = ~np.isnan(q_vals)
        q_norm = np.ones_like(own_norm)  # neutral (contributes nothing) for missing quality
        if q_valid.any():
            qmin, qmax = np.nanmin(q_vals), np.nanmax(q_vals)
            q_norm[q_valid] = np.clip((q_vals[q_valid] - qmin) / max(qmax - qmin, 1e-9), 1e-3, None)
        quality_component = q_norm ** alpha_quality
    else:
        quality_component = np.ones_like(own_norm)

    raw_weight = own_component * quality_component
    shares = raw_weight / raw_weight.sum()
    return dict(zip(conf_ids, shares))


# ---------------------------------------------------------------------------
# Stage 3: bespoke greedy multi-term selector
# ---------------------------------------------------------------------------

def select_portfolio(pool_lineups: list, feat: dict, confirmed_starter_ids: set,
                      weights: dict, portfolio_size: int = 150,
                      target_hitter_own: float = TARGET_HITTER_OWN,
                      target_mean_overlap: float = TARGET_MEAN_OVERLAP,
                      target_pitcher_pair_rate: float = TARGET["pitcher_pair_rate"],
                      target_mean_order: float = TARGET_MEAN_ORDER,
                      target_excess_gap: float = TARGET_EXCESS_GAP,
                      pitcher_target_share: dict | None = None) -> list:
    """target_hitter_own/target_mean_overlap/target_pitcher_pair_rate
    default to the pooled 10-slate constants but should normally be
    overridden per-slate by the caller with load_needlunchmoney_actuals()'s
    numbers -- see that function's docstring for why the pooled constants
    are the wrong target for any single slate. pitcher_target_share
    defaults to feat's own (built at PITCHER_COVERAGE_ALPHA) but
    calibrate_pitcher_coverage_alpha passes a recomputed one per alpha
    probe without touching the rest of feat."""
    M = len(pool_lineups)
    pids_arr = feat["pids_arr"]
    if pitcher_target_share is None:
        pitcher_target_share = feat["pitcher_target_share"]

    # Player-indicator matrix over ALL 10 slots (matches overlap_profile's
    # own definition), for the running diversity term -- pattern from
    # analyze_rival_portfolio.py::greedy_diverse/indicator, extended from
    # "always minimize" to "steer toward TARGET_MEAN_OVERLAP".
    all_ids = sorted(set(pids_arr.reshape(-1).tolist()))
    id_to_col = {pid: i for i, pid in enumerate(all_ids)}
    m = np.zeros((M, len(all_ids)), dtype=np.float32)
    rows = np.repeat(np.arange(M), 10)
    cols = np.array([id_to_col[pid] for pid in pids_arr.reshape(-1)])
    m[rows, cols] = 1.0

    pitcher_target_count = {
        pid: 2 * portfolio_size * pitcher_target_share.get(pid, 0.0)
        for pid in confirmed_starter_ids
    }
    pitcher_usage = {pid: 0 for pid in confirmed_starter_ids}
    conf_pitcher_arr = np.array([
        [pid if pid in confirmed_starter_ids else -1 for pid in pids_arr[i, 0:2]]
        for i in range(M)
    ])

    selected: list[int] = []
    remaining = np.ones(M, dtype=bool)
    hitter_own_sum = 0.0
    overlap_with_selected = np.zeros(M, dtype=np.float32)
    total_overlap_selected = 0.0
    team_count: dict = {}
    group_count = {5: 0, 4: 0, 3: 0}
    n_teams_slate = max(feat["n_teams_slate"], 1)
    pitcher_pair_count = 0
    order_sum = 0.0
    excess_gap_sum = 0.0

    for k in range(portfolio_size):
        idx = np.where(remaining)[0]

        hyp_hitter_mean = (hitter_own_sum + feat["hitter_own"][idx]) / (k + 1)
        hitter_term = -np.abs(hyp_hitter_mean - target_hitter_own) / max(target_hitter_own, 1e-6)

        hyp_order_mean = (order_sum + feat["hitter_mean_order"][idx]) / (k + 1)
        order_term = -np.abs(hyp_order_mean - target_mean_order) / max(target_mean_order, 1e-6)

        hyp_gap_mean = (excess_gap_sum + feat["primary_excess_gap"][idx]) / (k + 1)
        cluster_term = -np.abs(hyp_gap_mean - target_excess_gap) / max(target_excess_gap, 1e-6)

        if k == 0:
            diversity_term = np.zeros(len(idx))
        else:
            hyp_total = total_overlap_selected + overlap_with_selected[idx]
            hyp_n_pairs = k * (k + 1) / 2
            hyp_overlap_mean = hyp_total / hyp_n_pairs
            diversity_term = -np.abs(hyp_overlap_mean - target_mean_overlap) / max(target_mean_overlap, 1e-6)

        p_ids = conf_pitcher_arr[idx]  # (len(idx), 2)
        resid0 = np.array([max(0.0, pitcher_target_count.get(p, 0.0) - pitcher_usage.get(p, 0.0))
                            if p != -1 else 0.0 for p in p_ids[:, 0]])
        resid1 = np.array([max(0.0, pitcher_target_count.get(p, 0.0) - pitcher_usage.get(p, 0.0))
                            if p != -1 else 0.0 for p in p_ids[:, 1]])
        pitcher_cov = resid0 + resid1
        pc_max = pitcher_cov.max()
        pitcher_cov_norm = pitcher_cov / pc_max if pc_max > 0 else pitcher_cov

        teams = feat["primary_team"][idx]
        denom = max(k, 1)
        target_share = 1.0 / n_teams_slate
        # Rescaled to 1 - (current_share / target_share): 0 at exactly the
        # uniform-coverage share, +1 for a never-yet-used team, negative once
        # a team is over-represented relative to uniform. Raw (unscaled)
        # target_share - current_share was tiny (~0.04-0.07 typically, one
        # order of magnitude below the other terms' -1..1 range) so the
        # "team" weight barely moved anything -- confirmed by per-slate
        # entropy/max-exposure analysis showing our exposure concentrates
        # harder on high-EV/high-ownership teams than needlunchmoney's,
        # whose team choice tracks no team-quality signal (Vegas total,
        # ownership, value) we could find and is close to uniform across the
        # whole slate instead (see project-team-selection-signal memory).
        team_term = np.array([
            1.0 - (team_count.get(t, 0) / denom) / target_share for t in teams
        ])

        sizes = feat["primary_size"][idx]
        stack_term = np.zeros(len(idx))
        for j, s in enumerate(sizes):
            if s in TARGET_GROUP_FRACTIONS:
                cur_frac = group_count[s] / denom
                stack_term[j] = TARGET_GROUP_FRACTIONS[s] - cur_frac

        hyp_pp_frac = (pitcher_pair_count + feat["pitcher_pair_flag"][idx].astype(np.float64)) / (k + 1)
        pitcher_pair_term = -np.abs(hyp_pp_frac - target_pitcher_pair_rate)

        score = (
            weights["ev"] * feat["ev_norm"][idx]
            + weights["hitter_own"] * hitter_term
            + weights["pitcher_coverage"] * pitcher_cov_norm
            + weights["diversity"] * diversity_term
            + weights["team"] * team_term
            + weights["stack"] * stack_term
            + weights["pitcher_pair"] * pitcher_pair_term
            + weights["order"] * order_term
            + weights["cluster"] * cluster_term
        )

        best = int(idx[np.argmax(score)])
        selected.append(best)
        remaining[best] = False

        hitter_own_sum += feat["hitter_own"][best]
        order_sum += feat["hitter_mean_order"][best]
        excess_gap_sum += feat["primary_excess_gap"][best]
        if k > 0:
            total_overlap_selected += float(overlap_with_selected[best])
        overlap_with_selected += m @ m[best]

        t = feat["primary_team"][best]
        team_count[t] = team_count.get(t, 0) + 1
        s = int(feat["primary_size"][best])
        if s in group_count:
            group_count[s] += 1
        for pid in pids_arr[best, 0:2]:
            pid = int(pid)
            if pid in pitcher_usage:
                pitcher_usage[pid] += 1
        if bool(feat["pitcher_pair_flag"][best]):
            pitcher_pair_count += 1

    portfolio = [(pool_lineups[i], 0.0) for i in selected]  # (Lineup, ev) pairs, matching build_portfolio's return shape
    return portfolio, selected


def calibrate_pitcher_coverage_alpha(pool_lineups: list, feat: dict, confirmed_starter_ids: set,
                                      base_weights: dict, target_pitcher_own: float,
                                      quality_alpha: float = PITCHER_QUALITY_ALPHA,
                                      grid: tuple = (0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 15.0)
                                      ) -> tuple[float, list[dict]]:
    """Per-slate PITCHER_COVERAGE_ALPHA search -- REAL probes through
    select_portfolio, not the pure share-weighted-average analytical
    shortcut this function started with.

    That shortcut was confirmed WRONG this session: on 07292026 it predicted
    pitcher_own~26.02 (matching target) at the alpha it chose, but the REAL
    selected portfolio's pitcher_own came out to 31.67 -- a 5.65-point miss
    that was the dominant driver of that slate's chalk_index gap (target
    -1.00, achieved +0.85/+1.11 across runs). The analytical version ignores
    how pitcher_coverage_term's GREEDY, residual-based mechanism interacts
    with the other scoring terms (EV, diversity, etc. also influence which
    candidates -- and therefore which pitchers -- actually get picked); only
    running the real selector captures that. Same plain-grid-plus-bracket-
    interpolation pattern as calibrate_hitter_own_weight, for the same
    reason: no assumed monotonic direction.
    """
    target = target_pitcher_own

    def probe(alpha: float) -> float:
        share = compute_pitcher_target_shares(feat["conf_own"], alpha, feat.get("conf_quality"), quality_alpha)
        portfolio, sel_idx = select_portfolio(pool_lineups, feat, confirmed_starter_ids, base_weights,
                                               pitcher_target_share=share)
        return float(feat["pitcher_own"][sel_idx].mean())

    trace = [{"a": a, "pitcher_own": probe(a)} for a in grid]

    best_bracket = None
    for p, q in zip(trace, trace[1:]):
        rp, rq = p["pitcher_own"] - target, q["pitcher_own"] - target
        if rp == 0:
            return p["a"], trace
        if rq == 0:
            return q["a"], trace
        if rp * rq < 0:
            width = abs(q["a"] - p["a"])
            if best_bracket is None or width < best_bracket[0]:
                best_bracket = (width, p, q)

    if best_bracket is not None:
        _, p, q = best_bracket
        t = (target - p["pitcher_own"]) / (q["pitcher_own"] - p["pitcher_own"])
        best_a = p["a"] + t * (q["a"] - p["a"])
    else:
        best_a = min(trace, key=lambda t: abs(t["pitcher_own"] - target))["a"]
    return float(best_a), trace


def calibrate_hitter_own_weight(pool_lineups: list, feat: dict, confirmed_starter_ids: set,
                                 players_df: pd.DataFrame, archive_dir: Path,
                                 field_players_df: pd.DataFrame, base_weights: dict,
                                 target_chalk_index: float, target_hitter_own: float,
                                 target_mean_overlap: float, target_pitcher_pair_rate: float,
                                 pitcher_target_share: dict | None = None,
                                 grid: tuple = (0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 30.0)
                                 ) -> tuple[float, list[dict]]:
    """Per-slate hitter_own weight search against the BLENDED chalk_index.

    target_chalk_index/target_hitter_own/target_mean_overlap MUST be this
    slate's ACTUAL needlunchmoney numbers (load_needlunchmoney_actuals), not
    the pooled 10-slate constants -- confirmed directly this session that
    the pooled chalk_index target (-0.84) was simply the wrong number for
    07222026 specifically (his own real chalk_index that slate was -1.565,
    nearly identical to what an "unfixable" run had already produced).

    Plain grid search over `grid`, picking whichever point (or linear
    interpolation between the two closest-bracketing points, if any pair
    does bracket the target) lands nearest target_chalk_index -- NOT an
    expanding-bracket search assuming a fixed monotonic direction. That
    assumption ("more hitter_own weight -> more contrarian chalk_index")
    was confirmed WRONG on 07222026 after alpha calibration changed the
    pitcher-side baseline: there, chalk_index at w=0.5 (-2.31) was more
    negative than at w=8.0 (-1.92) -- the opposite of the 08032026 relationship
    the old assumption was built from. hitter_own_term only pulls the
    selection toward target_hitter_own; whether that makes the BLENDED
    chalk_index more or less negative depends on which side of
    target_hitter_own the zero-weight/natural trajectory already sits on,
    which varies by slate (and now also depends on the calibrated alpha) --
    not something safe to hardcode a sign for. select_portfolio is cheap
    (no simulation, no ContestScorer pass), so a 9-point grid costs a few
    seconds, not minutes.
    """
    target = target_chalk_index

    def probe(w: float) -> float:
        weights = dict(base_weights)
        weights["hitter_own"] = w
        portfolio, _ = select_portfolio(pool_lineups, feat, confirmed_starter_ids, weights,
                                         target_hitter_own=target_hitter_own,
                                         target_mean_overlap=target_mean_overlap,
                                         target_pitcher_pair_rate=target_pitcher_pair_rate,
                                         pitcher_target_share=pitcher_target_share)
        s = measure_structure(portfolio, players_df, archive_dir, field_players_df)
        return s["chalk_index"]

    trace = [{"w": w, "chalk_index": probe(w)} for w in grid]

    # Prefer interpolating between adjacent grid points that bracket the
    # target (opposite-sign residual), tightest bracket wins; fall back to
    # the single closest point if nothing brackets.
    best_bracket = None
    for a, b in zip(trace, trace[1:]):
        ra, rb = a["chalk_index"] - target, b["chalk_index"] - target
        if ra == 0:
            return a["w"], trace
        if rb == 0:
            return b["w"], trace
        if ra * rb < 0:
            width = abs(b["w"] - a["w"])
            if best_bracket is None or width < best_bracket[0]:
                best_bracket = (width, a, b)

    if best_bracket is not None:
        _, a, b = best_bracket
        t = (target - a["chalk_index"]) / (b["chalk_index"] - a["chalk_index"])
        best_w = a["w"] + t * (b["w"] - a["w"])
    else:
        best_w = min(trace, key=lambda t: abs(t["chalk_index"] - target))["w"]
    return float(best_w), trace


def print_structure_vs_actuals(measured: dict, label: str, actuals: dict) -> None:
    """Same layout as emulate_needlunchmoney.py::print_structure_comparison,
    but every parenthetical is this slate's ACTUAL needlunchmoney number
    (falling back to the pooled TARGET constant only when no actuals exist
    for this slate/contest) -- NOT reused unchanged, because that function's
    targets are hardcoded to the pooled constants and would silently mislabel
    every number here now that the selector itself calibrates against
    per-slate actuals."""
    print(f"\n--- structural match: {label} ---")
    print(f"  n_lineups           {measured['n_lineups']}")
    ov_t = actuals.get("mean_overlap", TARGET["mean_overlap"])
    ci_t = actuals.get("chalk_index", TARGET["chalk_index"])
    nt_t = actuals.get("n_primary_teams", TARGET["n_primary_teams"])
    pp_t = actuals.get("pitcher_pair_rate", TARGET["pitcher_pair_rate"])
    tag = "his actual" if actuals else "POOLED FALLBACK"
    print(f"  mean_overlap         {measured['mean_overlap']:.2f}  ({tag} {ov_t:.2f})")
    print(f"  chalk_index          {measured['chalk_index']:+.2f}  ({tag} {ci_t:+.2f})")
    print(f"  n_primary_teams      {measured['n_primary_teams']}  ({tag} {nt_t})")
    print(f"  pitcher_pair_rate    {measured['pitcher_pair_rate']:.3f}  ({tag} {pp_t:.3f})")
    gf = measured["group_fractions_realized"]
    print(f"  group_fractions      5:{gf.get(5,0):.2f} 4:{gf.get(4,0):.2f} 3:{gf.get(3,0):.2f}"
          f"  (pooled target 5:0.44 4:0.45 3:0.11 -- not slate-calibrated, deprioritized per user)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_one_slate(slate: str, weights: dict, calibrate: bool = True) -> dict | None:
    archive_dir = PROJECT_ROOT / "archive" / slate
    if not archive_dir.exists():
        print(f"skipping {slate}: no archive dir")
        return None
    try:
        real_contest = find_target_contest(archive_dir)
    except ValueError as exc:
        print(f"skipping {slate}: {exc}")
        return None

    print(f"\n=== {slate} ({real_contest['contest']}, n_field={real_contest['n_field']:,}) ===")

    pool, proj_ext, features_df, confirmed_starter_ids, n_games, players_df = load_pool(archive_dir)
    print(f"  pool: {len(pool.lineups):,} lineups after dedup "
          f"({pool.n_dropped_duplicates} exact-dup, {pool.n_dropped_near_duplicates} near-dup dropped)")
    print(f"  confirmed starters: {len(confirmed_starter_ids)}")

    ev_col = resolve_ev_column(n_games, real_contest["n_field"])
    print(f"  n_games={n_games}  ev_col={ev_col!r}")

    feat = build_feature_matrix(pool.lineups, features_df, proj_ext, confirmed_starter_ids, ev_col)
    field_players_df = field_players_for_contest(archive_dir, real_contest)

    # LOSO target: needlunchmoney's real stats averaged across every OTHER
    # backtest slate, excluding this one -- never this slate's own realized
    # numbers. Same-slate actuals (below, `actuals_display`) are loaded
    # separately and used ONLY for post-hoc printing, never as a
    # construction/calibration input -- see load_needlunchmoney_actuals_loso's
    # docstring for why same-slate targets are look-ahead.
    actuals = load_needlunchmoney_actuals_loso(slate)
    target_chalk_index = actuals.get("chalk_index", TARGET["chalk_index"])
    target_hitter_own = actuals.get("hitter_own", TARGET_HITTER_OWN)
    target_mean_overlap = actuals.get("mean_overlap", TARGET_MEAN_OVERLAP)
    target_pitcher_own = actuals.get("pitcher_own", TARGET_PITCHER_OWN)
    target_n_primary_teams = actuals.get("n_primary_teams", TARGET["n_primary_teams"])
    target_pitcher_pair_rate = actuals.get("pitcher_pair_rate", TARGET["pitcher_pair_rate"])
    target_mean_order = actuals.get("mean_order", TARGET_MEAN_ORDER)
    target_excess_gap = actuals.get("mean_excess_gap", TARGET_EXCESS_GAP)
    if np.isnan(target_excess_gap):
        target_excess_gap = TARGET_EXCESS_GAP
    src = (f"LOSO n={actuals['n']} across {actuals['n_slates']} other slates" if actuals
           else "POOLED FALLBACK (no other-slate actuals found)")
    print(f"  targets ({src}): chalk_index={target_chalk_index:+.2f}  "
          f"hitter_own={target_hitter_own:.2f}  pitcher_own={target_pitcher_own:.2f}  "
          f"mean_overlap={target_mean_overlap:.2f}  n_primary_teams={target_n_primary_teams:.1f}  "
          f"pitcher_pair_rate={target_pitcher_pair_rate:.3f}  mean_order={target_mean_order:.2f}  "
          f"excess_gap={target_excess_gap:.2f}")

    pitcher_target_share = feat["pitcher_target_share"]
    if calibrate:
        alpha, a_trace = calibrate_pitcher_coverage_alpha(pool.lineups, feat, confirmed_starter_ids,
                                                           weights, target_pitcher_own)
        a_trace_str = "  ".join(f"{t['a']:.1f}->{t['pitcher_own']:.2f}" for t in a_trace)
        print(f"  pitcher_coverage alpha calibration: {a_trace_str}  -> chose {alpha:.2f}")
        pitcher_target_share = compute_pitcher_target_shares(
            feat["conf_own"], alpha, feat.get("conf_quality"), PITCHER_QUALITY_ALPHA)

        w_hitter_own, trace = calibrate_hitter_own_weight(
            pool.lineups, feat, confirmed_starter_ids, players_df, archive_dir,
            field_players_df, weights, target_chalk_index, target_hitter_own, target_mean_overlap,
            target_pitcher_pair_rate, pitcher_target_share=pitcher_target_share)
        trace_str = "  ".join(f"{t['w']:.1f}->{t['chalk_index']:+.2f}" for t in trace)
        print(f"  hitter_own weight calibration: {trace_str}  -> chose {w_hitter_own:.2f}")
        weights = dict(weights)
        weights["hitter_own"] = w_hitter_own

    portfolio, selected_idx = select_portfolio(pool.lineups, feat, confirmed_starter_ids, weights,
                                                target_hitter_own=target_hitter_own,
                                                target_mean_overlap=target_mean_overlap,
                                                target_pitcher_pair_rate=target_pitcher_pair_rate,
                                                target_mean_order=target_mean_order,
                                                target_excess_gap=target_excess_gap,
                                                pitcher_target_share=pitcher_target_share)

    structure = measure_structure(portfolio, players_df, archive_dir, field_players_df)
    structure["hitter_own_weight_used"] = weights["hitter_own"]
    structure["n_games"] = int(n_games)

    # Post-hoc only, never seen by construction/calibration above.
    actuals_display = load_needlunchmoney_actuals(archive_dir, real_contest["contest"])
    print_structure_vs_actuals(structure, slate, actuals_display)
    print(f"  selector-side (projected) hitter_own mean: {feat['hitter_own'][selected_idx].mean():.2f}"
          f"  (LOSO target {target_hitter_own:.2f}, his actual this slate "
          f"{actuals_display.get('hitter_own', float('nan')):.2f})   "
          f"pitcher_own mean: {feat['pitcher_own'][selected_idx].mean():.2f}"
          f"  (LOSO target {target_pitcher_own:.2f}, his actual this slate "
          f"{actuals_display.get('pitcher_own', float('nan')):.2f})")
    print(f"  selector-side mean_order: {feat['hitter_mean_order'][selected_idx].mean():.2f}"
          f"  (LOSO target {target_mean_order:.2f}, his actual this slate "
          f"{actuals_display.get('mean_order', float('nan')):.2f})   "
          f"excess_gap: {feat['primary_excess_gap'][selected_idx].mean():.2f}"
          f"  (LOSO target {target_excess_gap:.2f}, his actual this slate "
          f"{actuals_display.get('mean_excess_gap', float('nan')):.2f})")
    print(f"  selector-side pitcher SS-Proj quality: {feat['pitcher_ss_proj'][selected_idx].mean():.2f}"
          f"  (his pooled reference {REFERENCE_HIS_PITCHER_SS_PROJ:.2f}, from "
          f"project-sp-signal-investigation; not a per-slate LOSO target)")

    graded = grade_our_portfolio(portfolio, players_df, archive_dir, real_contest)
    graded["slate"] = slate
    graded["contest"] = real_contest["contest"]
    return {"structure": structure, "graded": graded}


GRADED_CSV = "select_needlunchmoney_pool_graded.csv"
STRUCTURE_CSV = "select_needlunchmoney_pool_structure.csv"


def print_summary(graded_all: pd.DataFrame) -> None:
    print("\n\n===== POOL-SELECTED PORTFOLIO vs needlunchmoney (real) =====")
    fees, won = graded_all.fee.sum(), graded_all.won.sum()
    per_slate = graded_all.groupby("slate").apply(
        lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
    print(f"  entries {len(graded_all):,}  slates {graded_all.slate.nunique()}  "
          f"fees ${fees:,.0f}  won ${won:,.0f}  ROI {100*(won-fees)/fees:+.1f}%  "
          f"$/entry {(won-fees)/len(graded_all):+.2f}")
    gmax = graded_all.won.max()
    drop_max = (won - gmax - fees) / len(graded_all)
    print(f"  cash% {100*(graded_all.won>0).mean():.1f}  top1% {100*graded_all.top1.mean():.2f}  "
          f"top01% {100*graded_all.top01.mean():.3f}  drop_max $/entry {drop_max:+.2f}")
    if len(per_slate) > 1:
        print(f"  pos_slates {int((per_slate>0).sum())}/{len(per_slate)}  "
              f"LOSO_min {min((per_slate.sum()-x)/(len(per_slate)-1) for x in per_slate):.2f}")
    else:
        print(f"  ({len(per_slate)} slate so far -- LOSO not meaningful yet)")

    report_path = PROJECT_ROOT / "outputs" / "profitable_entrants_report.csv"
    if report_path.exists():
        rep = pd.read_csv(report_path)
        row = rep[rep.handle == "needlunchmoney"]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  needlunchmoney (real, all 10 slates): cash%_lift {r['cash%_lift']:+.2f}  "
                  f"top1%_lift {r['top1%_lift']:+.3f}  $/entry {r['$/entry']:+.2f}  "
                  f"drop_max $/entry {r['drop_max_$/entry']:+.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slate", default=None, help="single MMDDYYYY (default: all 10)")
    p.add_argument("--dry-run", action="store_true", help="one slate, no checkpoint loop")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--report-only", action="store_true")
    for term in DEFAULT_WEIGHTS:
        p.add_argument(f"--w-{term.replace('_','-')}", type=float, default=None,
                        help=f"override weight for '{term}' (default {DEFAULT_WEIGHTS[term]})")
    args = p.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    for term in DEFAULT_WEIGHTS:
        override = getattr(args, f"w_{term}")
        if override is not None:
            weights[term] = override

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    graded_path = out_dir / GRADED_CSV
    structure_path = out_dir / STRUCTURE_CSV

    if args.report_only:
        if not graded_path.exists():
            print(f"no checkpoint at {graded_path} yet")
            return
        print_summary(pd.read_csv(graded_path, dtype={"slate": str}))
        return

    slates = [args.slate] if args.slate else list(BACKTEST_SLATES) + [TENTH_SLATE]

    graded_rows, structure_rows = [], []
    done_slates = set()
    if graded_path.exists() and not args.no_resume and not args.dry_run:
        prior = pd.read_csv(graded_path, dtype={"slate": str})
        graded_rows.append(prior)
        done_slates = set(prior.slate.unique())
        if structure_path.exists():
            structure_rows.append(pd.read_csv(structure_path, dtype={"slate": str}))
        if done_slates:
            print(f"resuming: skipping {sorted(done_slates)}")

    for slate in slates:
        if slate in done_slates:
            continue
        r = run_one_slate(slate, weights)
        if r is None:
            continue
        graded_rows.append(r["graded"])
        structure_rows.append(pd.DataFrame([{**r["structure"], "slate": slate,
                                              "group_fractions_realized": str(r["structure"]["group_fractions_realized"])}]))
        if not args.dry_run:
            pd.concat(graded_rows, ignore_index=True).to_csv(graded_path, index=False)
            pd.concat(structure_rows, ignore_index=True).to_csv(structure_path, index=False)
            print(f"  checkpointed {slate} -> outputs/{GRADED_CSV}")
        if args.dry_run:
            break

    if args.dry_run or not graded_rows:
        return

    print_summary(pd.concat(graded_rows, ignore_index=True))


if __name__ == "__main__":
    main()
