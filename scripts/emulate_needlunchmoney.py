"""Construct a needlunchmoney-style 150-lineup portfolio for Rally Cap/Relay
Throw on each archived slate, and grade it against that contest's REAL field.

needlunchmoney is DK-restricted from sub-$3 contests for excess lifetime
profitability -- independent, strong evidence he's a genuine long-run
skilled winner. `scripts/analyze_profitable_entrants.py` already measured his
revealed construction stats from his real 150-max play in Rally Cap/Relay
Throw across all 10 archived slates (see `outputs/profitable_entrants_report.csv`,
handle "needlunchmoney"): mean_overlap 1.40, chalk_index -0.84 (mild fade
below the field's own ~11.6% mean rostered ownership), ~18.4 distinct
primary-stack teams/slate, and a ~44% five-man / 45% four-man / 11% three-man
primary-stack-size mix (vs `CandidateGenerator`'s default 62/31/7).

This script builds our own candidate pool and portfolio tuned toward those
targets, entirely by composing existing standalone pipeline stages
(ingestor -> SaberSim projections -> `PipelineRunner._build_players_df` ->
`SimulationEngine` -> `CandidateGenerator` -> `ContestScorer` ->
`DeterminantPortfolioSelector`) rather than `PipelineRunner.run()`, which is
a single ~2000-line method with no injection point for a custom generator or
a real per-contest payout table (it always scores against a generic
~5,001-entry reference, never the contest's actual field/payout curve).
`_apply_exclusions`/`_apply_twitter_overrides` are intentionally skipped:
they read LIVE persisted state (`data/slate_exclusions.json`) that has no
historical entries for an archived slate, so both are a no-op for this use
case, not a shortcut. Market-implied quantile grids are also intentionally
skipped -- that parquet is a live, globally-shared file, not archived per
slate, so applying it here would contaminate a historical reconstruction
with unrelated present-day data.

Two genuinely new mechanisms, neither of which exists anywhere else in this
codebase:
  - `fade_ownership()` -- CandidateGenerator's stack-sampling weight is
    literally the raw ownership vector (chalk-SEEKING by construction, see
    candidate_generator.py:1089/1178); there is no existing fade knob, so a
    power-transformed COPY of the ownership vector is fed to the generator's
    constructor while the real vector still drives players_df/scoring/
    measurement.
  - Grading a freshly-constructed (not-in-the-real-standings) 150-lineup
    portfolio against a real contest: `tests/bt_core.py::verify_slate` +
    `grade_portfolio` already solve this (per-player realized FPTS -> summed
    per lineup -> joint self-tie-splitting insertion into the real field) --
    no new grading math, just glue.

Usage
-----
    source venv/bin/activate
    python scripts/emulate_needlunchmoney.py --slate 08032026 --dry-run   # tune loop
    python scripts/emulate_needlunchmoney.py                              # full 10-slate run
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.factory import get_ingestor  # noqa: E402
from src.platforms.base import Platform  # noqa: E402
from src.api.external_pool import parse_sabersim_projections, discover_external_files  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.models.batter_model import BatterPCAModel  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.optimization.candidate_generator import CandidateGenerator  # noqa: E402
from src.optimization.gpp_portfolio import ContestScorer, DeterminantPortfolioSelector  # noqa: E402
from src.optimization.payout import structure_for_contest, payout_table_to_array  # noqa: E402

from tests.bt_core import (  # noqa: E402
    BACKTEST_SLATES, load_real_contests, verify_slate, grade_portfolio,
)
from scripts.analyze_rival_portfolio import (  # noqa: E402
    parse_standings_rows, overlap_profile, exposure_profile, stack_profile,
    primary_teams, team_map, pitcher_names,
)

TENTH_SLATE = "08032026"
TARGET_CONTESTS = ("Rally Cap", "Relay Throw")

# needlunchmoney's revealed targets (outputs/profitable_entrants_report.csv,
# Rally Cap/Relay Throw, 10 slates) -- see module docstring.
TARGET = {
    "mean_overlap": 1.40,
    "chalk_index": -0.84,
    "n_primary_teams": 18.4,
    "group_fractions": {5: 0.44, 4: 0.45, 3: 0.11},
    "pitcher_pair_rate": 0.132,
}

# chalk_index is defined against REALIZED (post-contest DK %Drafted)
# ownership, not our own pre-lock projected ownership -- and against the
# field-lineup-average baseline `chalk_profile()` used when it derived
# needlunchmoney's -0.84 target (outputs/profitable_entrants_features.csv,
# field_baseline_own column, pooled across the 443-handle cohort's real
# lineups). Reuse that exact constant so our own portfolio's chalk_index is
# comparable to his, not measured against a different reference.
FIELD_BASELINE_OWN = 11.628136900033793


# ---------------------------------------------------------------------------
# Stage 1: players_df + sim, without PipelineRunner.run()
# ---------------------------------------------------------------------------

def build_players_df_and_sim(archive_dir: Path, n_sims: int, rng_seed: int = 42):
    """(players_df, sim_results) for one archived slate, composed from the
    same standalone stages PipelineRunner.run() calls internally (lines
    310-453), minus the exclusions/twitter-override/quantile-grid steps
    documented above as no-ops or intentionally-skipped for this use case."""
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    runner = PipelineRunner(config_path=str(PROJECT_ROOT / "config.yaml"))

    slate_path = str(archive_dir / "DKSalaries.csv")
    ingestor = get_ingestor(Platform.DRAFTKINGS, slate_path)
    slate_df = ingestor.get_slate_dataframe()

    found = discover_external_files(str(archive_dir))
    if found["projections_path"] is None:
        raise FileNotFoundError(f"no SaberSim projections file found in {archive_dir}")
    proj_df = parse_sabersim_projections(found["projections_path"])

    players_df = runner._build_players_df(slate_df, proj_df)
    if "name" in slate_df.columns:
        players_df = players_df.merge(
            slate_df[["player_id", "name"]], on="player_id", how="left"
        )

    paths = cfg.get("paths", {})
    copula = EmpiricalCopula(str(PROJECT_ROOT / paths["copula"]))
    pca_model, score_grid = None, None
    pca_path = PROJECT_ROOT / (paths.get("batter_pca_model") or "data/processed/batter_pca_model.npz")
    grid_path = PROJECT_ROOT / (paths.get("batter_score_grid") or "data/processed/batter_score_grid.npy")
    if pca_path.exists() and grid_path.exists():
        pca_model = BatterPCAModel.load(str(pca_path))
        score_grid = np.load(grid_path)

    engine = SimulationEngine(
        copula, players_df, batter_pca_model=pca_model, score_grid=score_grid,
        quantile_grids=None,
    )
    sim_results = engine.simulate(n_sims)
    return players_df, sim_results


# ---------------------------------------------------------------------------
# Stage 2: ownership-fade transform (new)
# ---------------------------------------------------------------------------

def fade_ownership(own_raw: np.ndarray, gamma: float) -> np.ndarray:
    """Power-transform a COPY of the ownership vector for use only as
    CandidateGenerator's sampling weight. gamma=1.0 reproduces today's raw
    chalk-seeking behavior; gamma in (0,1) flattens toward uniform; gamma<0
    inverts into genuine fade (low-owned players sample MORE). Renormalization
    is already handled downstream by _build_pos_pools, so no rescaling here."""
    eps = 1e-4
    return np.power(np.clip(own_raw, eps, None), gamma)


# ---------------------------------------------------------------------------
# Stage 3+4: tuned generation, real-contest scoring, selection
# ---------------------------------------------------------------------------

def build_portfolio(players_df: pd.DataFrame, sim_results, real_contest: dict,
                     gamma: float, risk: float, n_candidates: int, rng_seed: int,
                     n_field_samples: int = 3):
    own_raw = players_df["ownership"].fillna(0.0).to_numpy()
    own_faded = fade_ownership(own_raw, gamma)

    gen = CandidateGenerator(
        players_df, own_faded, rng_seed=rng_seed, salary_floor=None,
    )
    gen.GROUP_FRACTIONS = dict(TARGET["group_fractions"])
    candidates = gen.generate(n_candidates=n_candidates)

    structure = structure_for_contest(real_contest["contest"].lower(),
                                       n_entries=real_contest["n_field"])
    payout_arr = payout_table_to_array(structure).astype(np.float32)

    # n_field_samples is the dominant cost driver against a real field this
    # large (up to ~29k for Rally Cap) -- kept low during calibration, full
    # (3) for the final build that's actually graded/reported.
    scorer = ContestScorer(
        sim_results, players_df, field_players_df=players_df,
        n_field_lineups=structure["total_entries"], n_field_samples=n_field_samples,
        payout_arr=payout_arr, ownership_vec=own_raw, field_ownership_vec=own_raw,
    )
    candidates, robust_payout = scorer.score_candidates(candidates)

    sel = DeterminantPortfolioSelector(
        robust_payout, candidates, portfolio_size=150, risk=risk,
    )
    return sel.select()  # list[(Lineup, mean_ev)]


def calibrate_gamma(players_df: pd.DataFrame, sim_results, real_contest: dict,
                     archive_dir: Path, field_players_df: pd.DataFrame, risk: float,
                     rng_seed: int, n_candidates_calib: int = 3000,
                     initial_bracket: tuple = (0.5, 4.0),
                     gamma_floor: float = -2.0, gamma_ceiling: float = 14.0,
                     max_expansions: int = 3) -> tuple[float, list[dict]]:
    """Per-slate gamma search: a FIXED gamma didn't transfer across slates in
    the first full run (chalk_index swung -3.21 to +3.27 with gamma locked at
    2.3) -- the right fade strength apparently depends on something slate-
    specific (candidate hypothesis: slate size / number of games, since a
    smaller player pool concentrates ownership differently).

    Expanding-bracket search, not a fixed-bracket interpolation: a first cut
    of this (fixed bracket 0.5-4.0, extrapolated beyond it when needed)
    still landed badly off-target on 07252026 -- both bracket ends read
    chalk -4.37/-3.06, nowhere near the -0.84 target, so the "interpolated"
    gamma was actually a risky extrapolation past every point actually
    tested. Instead: probe the initial bracket, and if the target isn't
    actually BETWEEN the two observed chalk_index values (same-sign
    residual on both ends), push the offending end further out (doubling
    the gap, up to gamma_floor/ceiling) and re-probe, up to
    max_expansions times, before ever interpolating. Only interpolates
    WITHIN two points that actually bracket the target -- no extrapolation.
    chalk_index empirically increases (gets chalkier) with gamma across
    every slate probed so far, so the search always pushes hi up / lo down
    in the direction that should close the gap.

    n_field_samples=1 and reduced n_candidates during every probe (real
    n_field_samples=3 is the dominant cost against a real field this large,
    up to ~29k for Rally Cap) -- kept cheap since each expansion is one more
    full ContestScorer pass.
    """
    target = TARGET["chalk_index"]

    def probe(g: float) -> float:
        portfolio = build_portfolio(players_df, sim_results, real_contest, g, risk,
                                     n_candidates_calib, rng_seed, n_field_samples=1)
        s = measure_structure(portfolio, players_df, archive_dir, field_players_df)
        return s["chalk_index"]

    lo, hi = initial_bracket
    c_lo, c_hi = probe(lo), probe(hi)
    trace = [{"gamma": lo, "chalk_index": c_lo}, {"gamma": hi, "chalk_index": c_hi}]

    expansions = 0
    while (c_lo - target) * (c_hi - target) > 0 and expansions < max_expansions:
        step = max(hi - lo, 2.0) * 2.0
        if c_hi < target:
            # both readings too negative (too contrarian) -- discard lo,
            # push a new probe further out past hi
            lo, c_lo = hi, c_hi
            new_hi = min(hi + step, gamma_ceiling)
            if new_hi == hi:
                break  # already pinned at the ceiling, can't expand further
            hi = new_hi
            c_hi = probe(hi)
        else:
            # both readings too positive (too chalky) -- discard hi,
            # push a new probe further out past lo
            hi, c_hi = lo, c_lo
            new_lo = max(lo - step, gamma_floor)
            if new_lo == lo:
                break  # already pinned at the floor, can't expand further
            lo = new_lo
            c_lo = probe(lo)
        trace.append({"gamma": lo, "chalk_index": c_lo})
        trace.append({"gamma": hi, "chalk_index": c_hi})
        expansions += 1

    if c_hi == c_lo:
        best_gamma = (lo + hi) / 2
    else:
        t = float(np.clip((target - c_lo) / (c_hi - c_lo), 0.0, 1.0))
        best_gamma = lo + t * (hi - lo)
    return float(best_gamma), trace


# ---------------------------------------------------------------------------
# Stage 5: measure structural stats, same functions used on needlunchmoney
# ---------------------------------------------------------------------------

def lineup_names(portfolio: list, players_df: pd.DataFrame) -> list[tuple]:
    id_to_name = players_df.set_index("player_id")["name"].to_dict()
    out = []
    for lu, _ev in portfolio:
        names = tuple(sorted(id_to_name.get(pid) for pid in lu.player_ids))
        if any(n is None for n in names):
            continue
        out.append(names)
    return out


def field_players_for_contest(archive_dir: Path, real_contest: dict) -> pd.DataFrame:
    """field_players_df (real, POST-contest DK %Drafted) for the target
    contest -- needed because chalk_index is defined against realized
    ownership, not our own pre-lock projection (see FIELD_BASELINE_OWN)."""
    import zipfile, csv, io
    z = archive_dir / f"{real_contest['contest_id'].split(':', 1)[1]}.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(zf.read(name).decode("utf-8-sig", errors="replace"))))
    _, field_players_df = parse_standings_rows(rows)
    return field_players_df


def measure_structure(portfolio: list, players_df: pd.DataFrame, archive_dir: Path,
                       field_players_df: pd.DataFrame) -> dict:
    lus = lineup_names(portfolio, players_df)
    tmap = team_map(archive_dir)
    pitchers = pitcher_names(archive_dir)

    ov = overlap_profile(lus)
    stk = stack_profile(lus, tmap, pitchers)
    pr = primary_teams(lus, tmap, pitchers)
    n_teams = len(set(p for p in pr if p))

    # chalk_index: REALIZED field %Drafted per rostered player (matching how
    # needlunchmoney's own -0.84 target was computed), not our projected
    # ownership -- an earlier version of this function compared projected
    # ownership against a uniform pool-wide average and produced numbers
    # that looked chalky even at an aggressively faded gamma; that was a
    # measurement bug (wrong ownership source AND wrong baseline), not a
    # tuning failure. See FIELD_BASELINE_OWN.
    field_own = field_players_df.set_index("player")["pct_drafted"].to_dict()
    own_means = [
        float(np.nanmean([field_own.get(n, np.nan) for n in lu])) for lu in lus
    ]
    chalk_index = float(np.nanmean(own_means)) - FIELD_BASELINE_OWN

    pitcher_pair = 0
    for lu in lus:
        team_series = pd.Series([tmap.get(x) for x in lu if x not in pitchers and tmap.get(x)])
        if team_series.empty:
            continue
        top_team = team_series.value_counts().index[0]
        if any(x in pitchers and tmap.get(x) == top_team for x in lu):
            pitcher_pair += 1

    size_frac = {}
    for k in (5, 4, 3):
        keys = [s for s in stk["shape_counts"].index if s.startswith(f"{k}-") or s == str(k)]
        size_frac[k] = float(stk["shape_counts"].reindex(keys).sum())

    return {
        "n_lineups": len(lus),
        "mean_overlap": ov["mean_overlap"],
        "n_primary_teams": n_teams,
        "chalk_index": chalk_index,
        "pitcher_pair_rate": pitcher_pair / max(len(lus), 1),
        "group_fractions_realized": size_frac,
    }


def print_structure_comparison(measured: dict, label: str) -> None:
    print(f"\n--- structural match: {label} ---")
    if "gamma_used" in measured:
        print(f"  n_games={measured.get('n_games')}  gamma_used={measured['gamma_used']:.2f}")
    print(f"  n_lineups           {measured['n_lineups']}")
    print(f"  mean_overlap         {measured['mean_overlap']:.2f}  (target {TARGET['mean_overlap']:.2f})")
    print(f"  chalk_index          {measured['chalk_index']:+.2f}  (target {TARGET['chalk_index']:+.2f})")
    print(f"  n_primary_teams      {measured['n_primary_teams']}  (target ~{TARGET['n_primary_teams']:.1f})")
    print(f"  pitcher_pair_rate    {measured['pitcher_pair_rate']:.3f}  (target {TARGET['pitcher_pair_rate']:.3f})")
    gf = measured["group_fractions_realized"]
    print(f"  group_fractions      5:{gf.get(5,0):.2f} 4:{gf.get(4,0):.2f} 3:{gf.get(3,0):.2f}"
          f"  (target 5:0.44 4:0.45 3:0.11)")


# ---------------------------------------------------------------------------
# Stage 6: grading against the real field
# ---------------------------------------------------------------------------

def grade_our_portfolio(portfolio: list, players_df: pd.DataFrame, archive_dir: Path,
                         real_contest: dict) -> pd.DataFrame:
    """Grading result PLUS the actual roster (`names`) per lineup, in the
    same schema as outputs/profitable_entrants_lineups.csv (`handle`,
    `names`, `slate`, `contest`, ...) so every analysis function already
    built against real pros' lineups (pair_stats, pitcher usage-by-
    ownership-quartile, chalk_profile) works unchanged on our own
    constructed portfolios -- just concat and filter by handle=="ours".
    Persisting this (not just the graded $/rank) is what makes direct
    gap-finding against a known-good portfolio possible without an
    expensive rebuild every time."""
    dk_salaries = pd.read_csv(archive_dir / "DKSalaries.csv")
    real_contests = load_real_contests(archive_dir)
    fpts_map = verify_slate(archive_dir, real_contests, dk_salaries)
    id_to_name = players_df.set_index("player_id")["name"].to_dict()

    actual_scores = np.array([
        sum(fpts_map.get(pid, np.nan) for pid in lu.player_ids) for lu, _ev in portfolio
    ])
    gross, rank = grade_portfolio(actual_scores, real_contest["sorted_scores"],
                                   real_contest["payout_arr"])
    names = [tuple(sorted(id_to_name.get(pid) for pid in lu.player_ids)) for lu, _ev in portfolio]
    return pd.DataFrame({
        "handle": "ours", "names": names,
        "points": actual_scores, "won": gross, "rank": rank,
        "fee": real_contest["fee"], "n_field": real_contest["n_field"],
        "top1": (rank <= max(1, real_contest["n_field"] // 100)).astype(int),
        "top01": (rank <= max(1, real_contest["n_field"] // 1000)).astype(int),
    })


def find_target_contest(archive_dir: Path) -> dict:
    for c in load_real_contests(archive_dir):
        if c["contest"] in TARGET_CONTESTS:
            return c
    raise ValueError(f"no Rally Cap/Relay Throw contest found in {archive_dir}")


# ---------------------------------------------------------------------------
# Duplicate-name (Muncy/Fermin) fix for reconstructing REAL entries from
# name-only standings text
# ---------------------------------------------------------------------------

def resolve_name_to_id(archive_dir: Path, dk_salaries: pd.DataFrame) -> dict:
    """Name -> DK player ID, correcting a naive `set_index("Name")["ID"]
    .to_dict()` for the rare DKSalaries name shared by two different real
    MLB players that slate (e.g. two "Max Muncy"s, LAD and ATH). A plain
    dict silently collapses duplicates to whichever row happens to appear
    LAST in DKSalaries.csv -- confirmed on 07282026 this picked the BENCHED
    Oakland Muncy (0 real FPTS) for every real entry naming "Max Muncy",
    when DK's own reported entry totals show the entry meant the PLAYING
    Dodgers Muncy (21 FPTS): a real lineup naming a player almost certainly
    meant whoever was actually starting that day, not their same-named
    benched counterpart. `tests/bt_core.py::resolve_duplicate_names` already
    solves the harder id->fpts side of this (see
    project-duplicate-name-fpts-resolution.md); this reuses the same
    projections-export `Order` signal to pick the right id for the
    name->id side, which that function doesn't need since it works
    entirely in id-space.
    """
    dup_names = set(dk_salaries["Name"][dk_salaries["Name"].duplicated(keep=False)])
    name_to_id = dk_salaries.set_index("Name")["ID"].to_dict()
    if not dup_names:
        return name_to_id
    from src.api import external_pool as ep
    found = ep.discover_external_files(str(archive_dir))
    proj = ep.parse_player_projections(found["projections_path"])
    order_by_id = dict(zip(proj["player_id"].astype("Int64"), proj["order"]))
    for name in dup_names:
        ids = dk_salaries.loc[dk_salaries["Name"] == name, "ID"].tolist()
        played = [i for i in ids if pd.notna(order_by_id.get(i, pd.NA))]
        if len(played) == 1:
            name_to_id[name] = played[0]
        # else ambiguous (both or neither played that day) -- leave the
        # arbitrary last-wins mapping; bounded, documented residual risk,
        # not fixable from the archive alone (see the memory above).
    return name_to_id


# ---------------------------------------------------------------------------
# Verification: reproduce needlunchmoney's own known real results
# ---------------------------------------------------------------------------

def sanity_check_grading_glue(archive_dir: Path, real_contest: dict) -> None:
    """Re-grade needlunchmoney's OWN real 150 lineups through verify_slate +
    grade_portfolio, with his rows excluded from sorted_real first (else his
    own entries get double-counted -- once already resident in the field
    array, once again via joint self-tie-split insertion). Compare against
    the won/rank already stored in outputs/profitable_entrants_lineups.csv
    (computed via the single-entry payout_for) as a correctness check on the
    grading glue before trusting it on our own constructed lineups."""
    lineups_csv = PROJECT_ROOT / "outputs" / "profitable_entrants_lineups.csv"
    df = pd.read_csv(lineups_csv, dtype={"slate": str})
    df["names"] = df["names"].apply(ast.literal_eval)
    mine = df[(df.handle == "needlunchmoney") & (df.slate == archive_dir.name)
              & (df.contest == real_contest["contest"])]
    if mine.empty:
        print(f"  (sanity check skipped: no needlunchmoney rows for {archive_dir.name}/{real_contest['contest']})")
        return

    z = archive_dir / f"{real_contest['contest_id'].split(':', 1)[1]}.zip"
    import zipfile, csv, io
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(io.StringIO(zf.read(name).decode("utf-8-sig", errors="replace"))))
    entries, _ = parse_standings_rows(rows)
    others = entries[entries.handle.str.lower() != "needlunchmoney"]
    sorted_excl = np.sort(others["points"].dropna().to_numpy())

    dk_salaries = pd.read_csv(archive_dir / "DKSalaries.csv")
    real_contests = load_real_contests(archive_dir)
    fpts_map = verify_slate(archive_dir, real_contests, dk_salaries)
    name_to_id = resolve_name_to_id(archive_dir, dk_salaries)

    actual_scores = np.array([
        sum(fpts_map.get(name_to_id.get(n), np.nan) for n in names) for names in mine["names"]
    ])
    gross, rank = grade_portfolio(actual_scores, sorted_excl, real_contest["payout_arr"])

    d_won = np.nanmean(np.abs(gross - mine["won"].to_numpy()))
    d_rank = np.nanmean(np.abs(rank - mine["rank_check"].to_numpy()))
    print(f"  sanity check ({archive_dir.name}/{real_contest['contest']}, n={len(mine)}): "
          f"mean |Δwon|={d_won:.3f}  mean |Δrank|={d_rank:.2f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_one_slate(slate: str, gamma: float | None, risk: float, n_candidates: int,
                   n_sims: int, rng_seed: int, verify: bool) -> dict | None:
    """gamma=None triggers per-slate auto-calibration (see calibrate_gamma) --
    a FIXED gamma tuned on one slate did not transfer to the others (chalk_index
    swung -3.21 to +3.27 across the 10 slates at a locked gamma=2.3)."""
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
    if verify:
        sanity_check_grading_glue(archive_dir, real_contest)

    players_df, sim_results = build_players_df_and_sim(archive_dir, n_sims, rng_seed)
    field_players_df = field_players_for_contest(archive_dir, real_contest)
    n_games = players_df["game"].nunique()

    if gamma is None:
        gamma_used, trace = calibrate_gamma(players_df, sim_results, real_contest,
                                             archive_dir, field_players_df, risk, rng_seed)
        trace_str = "  ".join(f"{t['gamma']:.1f}->{t['chalk_index']:+.2f}" for t in trace)
        print(f"  gamma calibration (n_games={n_games}): {trace_str}  -> chose {gamma_used:.1f}")
    else:
        gamma_used = gamma

    portfolio = build_portfolio(players_df, sim_results, real_contest, gamma_used, risk,
                                 n_candidates, rng_seed)
    structure = measure_structure(portfolio, players_df, archive_dir, field_players_df)
    structure["n_games"] = int(n_games)
    structure["gamma_used"] = gamma_used
    print_structure_comparison(structure, slate)

    graded = grade_our_portfolio(portfolio, players_df, archive_dir, real_contest)
    graded["slate"] = slate
    graded["contest"] = real_contest["contest"]
    return {"structure": structure, "graded": graded}


GRADED_CSV = "emulate_needlunchmoney_graded.csv"
STRUCTURE_CSV = "emulate_needlunchmoney_structure.csv"


def print_summary(graded_all: pd.DataFrame) -> None:
    print("\n\n===== OUR CONSTRUCTED PORTFOLIO vs needlunchmoney (real) =====")
    fees, won = graded_all.fee.sum(), graded_all.won.sum()
    per_slate = graded_all.groupby("slate").apply(
        lambda x: (x.won.sum() - x.fee.sum()) / len(x), include_groups=False)
    print(f"  entries {len(graded_all):,}  slates {graded_all.slate.nunique()}  "
          f"fees ${fees:,.0f}  won ${won:,.0f}  ROI {100*(won-fees)/fees:+.1f}%  "
          f"$/entry {(won-fees)/len(graded_all):+.2f}")
    print(f"  cash% {100*(graded_all.won>0).mean():.1f}  top1% {100*graded_all.top1.mean():.2f}  "
          f"top01% {100*graded_all.top01.mean():.3f}")
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
    p.add_argument("--gamma", type=float, default=None,
                    help="ownership-fade exponent; default None triggers per-slate "
                         "auto-calibration (see calibrate_gamma) since a fixed value "
                         "did not transfer across slates. Pass a number to override "
                         "and skip calibration for all slates.")
    p.add_argument("--risk", type=float, default=1.5, help="DeterminantPortfolioSelector risk (1-5)")
    p.add_argument("--n-candidates", type=int, default=15_000)
    p.add_argument("--n-sims", type=int, default=10_000)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="structural match only, no grading loop over all slates")
    p.add_argument("--no-verify", action="store_true", help="skip the grading-glue sanity check")
    p.add_argument("--no-resume", action="store_true",
                    help="reprocess every slate even if already checkpointed in outputs/" + GRADED_CSV)
    p.add_argument("--report-only", action="store_true",
                    help="just print the summary from whatever's already checkpointed, run nothing")
    args = p.parse_args()

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

    # Resume: skip slates already checkpointed from a prior (possibly
    # timed-out or crashed) run, so a re-run doesn't repeat expensive work.
    graded_rows = []
    structure_rows = []
    done_slates = set()
    if graded_path.exists() and not args.no_resume:
        prior = pd.read_csv(graded_path, dtype={"slate": str})
        graded_rows.append(prior)
        done_slates = set(prior.slate.unique())
        if structure_path.exists():
            structure_rows.append(pd.read_csv(structure_path, dtype={"slate": str}))
        if done_slates:
            print(f"resuming: {len(done_slates)} slate(s) already checkpointed, skipping: "
                  f"{sorted(done_slates)}")

    for slate in slates:
        if slate in done_slates:
            continue
        r = run_one_slate(slate, args.gamma, args.risk, args.n_candidates,
                           args.n_sims, args.rng_seed, verify=not args.no_verify)
        if r is None:
            continue
        graded_rows.append(r["graded"])
        structure_rows.append(pd.DataFrame([{**r["structure"], "slate": slate,
                                              "group_fractions_realized": str(r["structure"]["group_fractions_realized"])}]))
        # Checkpoint after EVERY slate, not just at the end -- a timeout or
        # crash partway through must not lose already-completed slates.
        pd.concat(graded_rows, ignore_index=True).to_csv(graded_path, index=False)
        pd.concat(structure_rows, ignore_index=True).to_csv(structure_path, index=False)
        print(f"  checkpointed {slate} -> outputs/{GRADED_CSV}")
        if args.dry_run:
            break

    if args.dry_run or not graded_rows:
        return

    print_summary(pd.concat(graded_rows, ignore_index=True))

    print(f"\nwrote outputs/emulate_needlunchmoney_graded.csv")


if __name__ == "__main__":
    main()
