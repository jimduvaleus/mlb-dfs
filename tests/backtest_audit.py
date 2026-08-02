"""Phase 2 model-error audit: how wrong is the simulator, and do its
currencies carry any realized selection signal?

Four subcommands, each printing a compact table and appending a tidy CSV
under tests/backtest_output/audit/:

  pit       calibration ladder -- PIT of realized score against the sim
            distribution, tail exceedance rates, hitter/pitcher decomposition.
  varcomp   variance decomposition of z = (realized - sim_mean) / sim_std
            into slate / primary-stack-team / SP1 components.
  signal    entry-weighted decile lift for sim currencies vs model-free
            controls, plus a head-to-head against proj_score per slate.
  crowding  real vs simulated opponent-field crowding (dupes, stack
            concentration, effective distinct count) at the top-1% band,
            plus a consensus-proximity cross-check against audit 1's
            per-slate model accuracy.
  all       runs 1->4, sharing one build_slate_context per slate (the
            expensive part -- field generation/scoring inside it is
            unconditional, see bt_core.build_slate_context) so the shared
            path is paid once, not once per subcommand.

Everything here is read-only against the existing harness: no existing file
is modified. tests/bt_core.py and tests/backtest_lab.py are imported, and one
of backtest_lab's own commands (cmd_currencies) is reused for the `signal`
decile machinery via a temporary, restored, in-process monkeypatch of its
_candidate_currencies helper (adds the one extra currency -- own_prod_log --
the audit needs and that function doesn't already compute; nothing is
written back to backtest_lab.py).

Usage (seed 42 / calib=False everywhere except the stated stability checks):

    source venv/bin/activate
    python tests/backtest_audit.py all
    python tests/backtest_audit.py pit
"""
import csv as csv_mod
import io
import re
import sys
import zipfile
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.backtest_lab as lab  # noqa: E402
from tests.bt_core import (  # noqa: E402
    BACKTEST_SLATES, LIVE_CFG, _PWIN_FIELD_CAP, build_slate_context,
    hitter_only_lineup_scores, load_real_contests, verify_slate,
)
from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _decile_agg, _decile_consistency, _sign_p,
)

ARCHIVE_DIR = PROJECT_ROOT / "archive"
SIM_CACHE_DIR = PROJECT_ROOT / "tests" / "backtest_output" / "sim_cache"
AUDIT_DIR = ORACLE_DIR.parent / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMS = int(LIVE_CFG["simulation"]["n_sims"])
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
SEED = 42
STABILITY_SLATE = "07292026"
STABILITY_SEEDS = (42, 137, 4242)
N_BOOT = 2000

# Standings-zip Lineup column tokenizer: "1B Name 2B Name ... P Name P Name SS Name".
_POS_TOKEN_RE = re.compile(
    r"(1B|2B|3B|C|OF|P|SS)\s+(.+?)(?=\s+(?:1B|2B|3B|C|OF|P|SS)\s+|$)"
)


def _slate_list() -> list:
    return [s for s in BACKTEST_SLATES if (ORACLE_DIR / f"{s}_real.npz").exists()]


# ---------------------------------------------------------------------------
# Shared grouping helpers (used by varcomp and crowding)
# ---------------------------------------------------------------------------

def _primary_stack(pid_row, pos_by_id: dict, team_by_id: dict) -> tuple:
    """(team, hitter_count) for a lineup's dominant batter team -- "hitter"
    means position != "P" via the same players_df convention every other
    module in this repo uses (build_external_players_df/hitter_only_lineup_scores)."""
    hitters = [int(p) for p in pid_row if pos_by_id.get(int(p)) != "P"]
    if not hitters:
        return "", 0
    teams = [team_by_id.get(p, "") for p in hitters]
    team, cnt = Counter(teams).most_common(1)[0]
    return team, cnt


def _sp1(pid_row, pos_by_id: dict, sal_by_id: dict) -> int:
    """The lineup's SP1: highest-salary rostered pitcher, ties broken by the
    lower player_id -- deterministic, stated per the plan's requirement.
    Returns -1 if the lineup rosters no pitcher (shouldn't happen for a
    valid 10-man DK classic roster)."""
    pitchers = sorted(
        (int(p) for p in pid_row if pos_by_id.get(int(p)) == "P"),
        key=lambda p: (-sal_by_id.get(p, 0.0), p),
    )
    return pitchers[0] if pitchers else -1


# ---------------------------------------------------------------------------
# 1/2/4 shared per-slate work: one build_slate_context call each.
# ---------------------------------------------------------------------------

def _slate_workunit(slate: str, seed: int = SEED, calib: bool = False) -> dict:
    """Cached wrapper around _slate_workunit_uncached: the expensive part
    (one build_slate_context call -- field generation/scoring inside it is
    unconditional regardless of which currencies a caller asked for, see
    build_slate_context's docstring) is paid once per (slate, seed, calib)
    and its lightweight summaries are cached to
    audit/{slate}_workunit_s{seed}_c{calib}.npz, so a later run (e.g. a
    separate process invocation covering a different slate range, to stay
    under a wall-clock budget) reloads instantly instead of recomputing."""
    cache = AUDIT_DIR / f"{slate}_workunit_s{seed}_c{int(calib)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        pit = {k[len("pit__"):]: z[k].item() for k in z.files if k.startswith("pit__")}
        pit["slate"] = str(pit["slate"])
        simc = {k[len("simc__"):]: z[k].item() for k in z.files if k.startswith("simc__")}
        simc["slate"] = str(simc["slate"])
        return dict(
            pit=pit, z=z["z"], ok=z["ok"], team=z["team"], stack_n=z["stack_n"],
            sp1=z["sp1"], sim_std=z["sim_std"], sim_crowding=simc,
        )
    bundle = _slate_workunit_uncached(slate, seed, calib)
    out = {f"pit__{k}": np.asarray(v) for k, v in bundle["pit"].items()}
    out.update({f"simc__{k}": np.asarray(v) for k, v in bundle["sim_crowding"].items()})
    for k in ("z", "ok", "team", "stack_n", "sp1", "sim_std"):
        out[k] = bundle[k]
    np.savez_compressed(cache, **out)
    return bundle


def _slate_workunit_uncached(slate: str, seed: int = SEED, calib: bool = False) -> dict:
    """Everything audits 1 (pit), 2 (varcomp) and 4 (crowding's simulated
    field) need from ONE build_slate_context call. The (M, n_sims) score
    matrices are reduced to summaries before returning -- keeping them alive
    for all 9 slates at once (each up to ~1GB) is not needed and not safe to
    assume fits in memory.
    """
    d = ARCHIVE_DIR / slate
    real = load_real_contests(d)
    ctx = build_slate_context(
        d, seed, calib, real, n_sims=N_SIMS, sharpness=SHARPNESS,
        sim_cache_dir=SIM_CACHE_DIR, want_corr=False, want_pwin=False,
    )
    realz = np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False)
    pool_pids = np.array(
        [[int(p) for p in lu.player_ids] for lu in ctx["pool"].lineups], dtype=np.int64,
    )
    assert np.array_equal(realz["player_ids"], pool_pids), (
        f"{slate}: pool order drift between {slate}_real.npz and a fresh "
        "build_slate_context call -- the two are supposed to be deterministic "
        "reruns of the same parse, investigate before trusting anything here."
    )
    actual = realz["actual_score"]
    ok = np.isfinite(actual)

    players_df = ctx["players_df"]
    pos_by_id = dict(zip(players_df["player_id"].astype(int), players_df["position"]))
    team_by_id = dict(zip(players_df["player_id"].astype(int), players_df["team"]))
    sal_by_id = dict(zip(players_df["player_id"].astype(int),
                        players_df["salary"].astype(float)))

    nm = pd.read_csv(d / "DKSalaries.csv")[["ID", "Name"]].astype({"ID": str})
    fpts = verify_slate(d, real, nm)
    hitter_actual = np.array([
        sum(fpts.get(int(p), float("nan")) for p in row if pos_by_id.get(int(p)) != "P")
        for row in pool_pids
    ])
    pitcher_actual = actual - hitter_actual  # NaN propagates from either side

    scores = ctx["lineup_scores"]                                    # (M, S) float32
    hitter_scores = hitter_only_lineup_scores(ctx["pool"], ctx["sim_results"], players_df)
    pitcher_scores = scores - hitter_scores

    pit = (scores <= actual[:, None]).mean(axis=1)
    pit_h = (hitter_scores <= hitter_actual[:, None]).mean(axis=1)
    pit_p = (pitcher_scores <= pitcher_actual[:, None]).mean(axis=1)
    p90, p95, p99 = np.percentile(scores, [90, 95, 99], axis=1)
    # "sim-field p99": the 99th percentile, across sim worlds, of that
    # world's own pool-best score -- a field-level ceiling check (does the
    # model's simulated ceiling for the WHOLE pool, not any one lineup,
    # reach as high as what actually happened), distinct from the per-lineup
    # PIT/tail rates above.
    per_world_max = scores.max(axis=0)

    sim_mean = scores.mean(axis=1).astype(np.float64)
    sim_std = scores.std(axis=1).astype(np.float64)
    z = np.full(len(actual), np.nan)
    zmask = ok & (sim_std > 0)
    z[zmask] = (actual[zmask] - sim_mean[zmask]) / sim_std[zmask]

    team, stack_n, sp1 = [], [], []
    for row in pool_pids:
        t, n = _primary_stack(row, pos_by_id, team_by_id)
        team.append(t)
        stack_n.append(n)
        sp1.append(_sp1(row, pos_by_id, sal_by_id))
    team = np.array(team)
    stack_n = np.array(stack_n, dtype=np.int64)
    sp1 = np.array(sp1, dtype=np.int64)

    pit_summary = dict(
        slate=slate, seed=seed, n=int(ok.sum()),
        mean_pit=float(np.nanmean(np.where(ok, pit, np.nan))),
        mean_pit_hitter=float(np.nanmean(np.where(ok, pit_h, np.nan))),
        mean_pit_pitcher=float(np.nanmean(np.where(ok, pit_p, np.nan))),
        tail90=float(np.mean((actual > p90)[ok])),
        tail95=float(np.mean((actual > p95)[ok])),
        tail99=float(np.mean((actual > p99)[ok])),
        max_realized=float(np.nanmax(actual)),
        simfield_p99=float(np.percentile(per_world_max, 99)),
    )
    pit_summary["gap"] = pit_summary["max_realized"] - pit_summary["simfield_p99"]

    # --- crowding: regenerate the opponent field build_slate_context already
    # paid to generate+score internally (field_A/field_scores_A), since it
    # doesn't return the compositions -- deterministic given the same
    # (players_df, ownership, n_lineups, rng_seed), so this reproduces it
    # bit-for-bit rather than resampling something different.
    from src.optimization.contest import ContestSimulator

    own_vec = players_df["ownership"].astype(float).to_numpy()
    field_n = int(min(max(5_000, max(c["n_field"] for c in ctx["contests"])), _PWIN_FIELD_CAP))
    cs = ContestSimulator()
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed)
    col_map = {int(p): i for i, p in enumerate(ctx["sim_results"].player_ids)}
    valid = np.array([all(int(p) in col_map for p in row) for row in field_A])
    field_valid = field_A[valid]
    mean_field_scores = ctx["field_scores_A"].mean(axis=0).astype(np.float64)
    if field_valid.shape[0] != mean_field_scores.shape[0]:
        raise SystemExit(
            f"{slate}: regenerated field_A ({field_valid.shape[0]}) doesn't match "
            f"ctx['field_scores_A'] ({mean_field_scores.shape[0]}) -- generate_field "
            "isn't reproducing deterministically from (players_df, ownership, "
            "n_lineups, rng_seed) as assumed; investigate before trusting crowding."
        )
    band = max(1, round(0.01 * field_valid.shape[0]))
    order = np.argsort(-mean_field_scores)[:band]
    top_ids = field_valid[order]
    sim_keys, sim_teams, sim_counts = [], [], []
    for row in top_ids:
        t, c = _primary_stack(row, pos_by_id, team_by_id)
        sim_teams.append(t)
        sim_counts.append(c)
        sim_keys.append("|".join(str(int(p)) for p in sorted(int(x) for x in row)))
    sim_crowd = _crowding_metrics(
        np.array(sim_keys), np.array(sim_teams), np.array(sim_counts, dtype=np.int64), band,
    )
    sim_crowd.update(slate=slate, n_field=int(field_valid.shape[0]), band=band)

    return dict(
        pit=pit_summary, z=z, ok=ok, team=team, stack_n=stack_n, sp1=sp1,
        sim_std=sim_std, sim_crowding=sim_crowd,
    )


def _run_core(seed: int = SEED, calib: bool = False, slates=None):
    """One _slate_workunit pass over every slate -- the shared substrate for
    pit/varcomp/crowding. Returns (pit_df, z_info, sim_crowd_df)."""
    slates = slates or _slate_list()
    pit_rows, sim_crowd_rows = [], []
    z_info = {}
    for s in slates:
        bundle = _slate_workunit(s, seed, calib)
        pit_rows.append(bundle["pit"])
        z_info[s] = {k: bundle[k] for k in ("z", "ok", "team", "stack_n", "sp1", "sim_std")}
        sim_crowd_rows.append(bundle["sim_crowding"])
        print(f"    {s} (seed {seed}): done", flush=True)
    return pd.DataFrame(pit_rows), z_info, pd.DataFrame(sim_crowd_rows)


# ---------------------------------------------------------------------------
# 1. pit -- calibration ladder
# ---------------------------------------------------------------------------

def print_pit(main_df: pd.DataFrame, stab_df: pd.DataFrame) -> None:
    print(f"\n===== 1. PIT CALIBRATION LADDER (seed {SEED}, calib=False, "
          f"{len(main_df)} slates) =====")
    print("mean_pit ~ 0.5 under a well-calibrated model (each lineup's PIT is U(0,1))")
    print("tail90/95/99 should be ~ 0.10/0.05/0.01 if the sim's tail is calibrated.")
    print(main_df.round(4).to_string(index=False))

    print(f"\n-- seed stability on {STABILITY_SLATE} ({STABILITY_SEEDS}) --")
    print(stab_df.round(4).to_string(index=False))


def compute_pit_all(seed: int = SEED, run_core=None):
    if run_core is None:
        pit_df, _, _ = _run_core(seed)
    else:
        pit_df, _, _ = run_core
    stab_rows = []
    for s in STABILITY_SEEDS:
        if s == seed and STABILITY_SLATE in set(pit_df["slate"]):
            stab_rows.append(pit_df[pit_df.slate == STABILITY_SLATE].iloc[0].to_dict())
            continue
        bundle = _slate_workunit(STABILITY_SLATE, s, False)
        stab_rows.append(bundle["pit"])
    stab_df = pd.DataFrame(stab_rows)
    return pit_df, stab_df


def cmd_pit(run_core=None) -> pd.DataFrame:
    pit_df, stab_df = compute_pit_all(SEED, run_core)
    print_pit(pit_df, stab_df)
    combined = pd.concat(
        [pit_df, stab_df[stab_df.seed != SEED]], ignore_index=True,
    )
    combined.to_csv(AUDIT_DIR / "pit.csv", index=False)
    print(f"\nwrote {AUDIT_DIR / 'pit.csv'}")
    return pit_df


# ---------------------------------------------------------------------------
# 2. varcomp -- error variance decomposition
# ---------------------------------------------------------------------------

def _mom_component(group: np.ndarray, z: np.ndarray, mask: np.ndarray, min_cell: int = 2):
    """Method-of-moments between-group variance component for one slate:
    var(cell means) - mean(within-cell var / n_cell). Cells with < min_cell
    lineups are dropped (within-cell variance undefined at n=1). Returns
    None if fewer than 2 cells survive."""
    df = pd.DataFrame({"g": np.asarray(group)[mask], "z": np.asarray(z)[mask]})
    stats = df.groupby("g")["z"].agg(["mean", "var", "count"])
    stats = stats[stats["count"] >= min_cell]
    if len(stats) < 2:
        return None
    var_of_means = float(stats["mean"].var(ddof=1))
    mean_within = float((stats["var"] / stats["count"]).mean())
    return dict(component=var_of_means - mean_within, stats=stats,
                n_cells=len(stats), n_obs=int(stats["count"].sum()))


def _bootstrap_slate_means(slate_means: np.ndarray, n_reps: int = N_BOOT, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(slate_means)
    idx = rng.integers(0, n, size=(n_reps, n))
    return slate_means[idx].var(axis=1, ddof=1)


def _bootstrap_component(stats_list: list, n_reps: int = N_BOOT, seed: int = 1):
    """Resample teams/pitchers WITHIN each slate (with replacement, same
    count), recompute that slate's MoM component, average across slates --
    one bootstrap draw of the overall component. 2000 reps -> percentile CI."""
    rng = np.random.default_rng(seed)
    dfs = [r["stats"] for r in stats_list]
    reps = np.empty(n_reps)
    for i in range(n_reps):
        comps = []
        for stats in dfs:
            n_cells = len(stats)
            idx = rng.integers(0, n_cells, size=n_cells)
            rs = stats.iloc[idx]
            vm = rs["mean"].var(ddof=1)
            mw = (rs["var"] / rs["count"]).mean()
            comps.append(vm - mw)
        reps[i] = np.mean(comps)
    return reps


def compute_varcomp(z_info: dict = None, n_reps: int = N_BOOT) -> dict:
    if z_info is None:
        _, z_info, _ = _run_core(SEED)
    slates = sorted(z_info)
    slate_means, all_sim_std = [], []
    team_stats_list, pitcher_stats_list = [], []
    for s in slates:
        info = z_info[s]
        z, ok = info["z"], info["ok"]
        team, stack_n, sp1, sim_std = info["team"], info["stack_n"], info["sp1"], info["sim_std"]
        slate_means.append(float(np.nanmean(z[ok])))
        all_sim_std.append(sim_std[ok])

        team_mask = ok & (stack_n >= 4)
        r = _mom_component(team, z, team_mask)
        if r is not None:
            r["slate"] = s
            team_stats_list.append(r)

        pitch_mask = ok & (sp1 >= 0)
        r2 = _mom_component(sp1, z, pitch_mask)
        if r2 is not None:
            r2["slate"] = s
            pitcher_stats_list.append(r2)

    slate_means = np.array(slate_means)
    sigma2_slate = float(np.var(slate_means, ddof=1))
    ci_slate = np.percentile(_bootstrap_slate_means(slate_means, n_reps), [2.5, 97.5])

    sigma2_team = (float(np.mean([r["component"] for r in team_stats_list]))
                   if team_stats_list else np.nan)
    ci_team = (np.percentile(_bootstrap_component(team_stats_list, n_reps), [2.5, 97.5])
               if team_stats_list else (np.nan, np.nan))

    sigma2_pitcher = (float(np.mean([r["component"] for r in pitcher_stats_list]))
                      if pitcher_stats_list else np.nan)
    ci_pitcher = (np.percentile(_bootstrap_component(pitcher_stats_list, n_reps), [2.5, 97.5])
                  if pitcher_stats_list else (np.nan, np.nan))

    median_sim_std = float(np.median(np.concatenate(all_sim_std)))

    def _row(name, sigma2, ci, n_slates, n_cells, n_obs):
        sigma_z = float(np.sqrt(max(sigma2, 0.0))) if np.isfinite(sigma2) else np.nan
        ci_lo = float(np.sqrt(max(ci[0], 0.0))) if np.isfinite(ci[0]) else np.nan
        ci_hi = float(np.sqrt(max(ci[1], 0.0))) if np.isfinite(ci[1]) else np.nan
        return dict(component=name, raw_var=sigma2, sigma_z=sigma_z,
                    ci_lo_z=ci_lo, ci_hi_z=ci_hi,
                    sigma_fpts=sigma_z * median_sim_std if np.isfinite(sigma_z) else np.nan,
                    n_slates=n_slates, n_cells=n_cells, n_obs=n_obs)

    rows = [
        _row("slate", sigma2_slate, ci_slate, len(slates), len(slates),
             int(sum(z_info[s]["ok"].sum() for s in slates))),
        _row("team (batter, stack>=4)", sigma2_team, ci_team, len(team_stats_list),
             sum(r["n_cells"] for r in team_stats_list) if team_stats_list else 0,
             sum(r["n_obs"] for r in team_stats_list) if team_stats_list else 0),
        _row("pitcher (SP1)", sigma2_pitcher, ci_pitcher, len(pitcher_stats_list),
             sum(r["n_cells"] for r in pitcher_stats_list) if pitcher_stats_list else 0,
             sum(r["n_obs"] for r in pitcher_stats_list) if pitcher_stats_list else 0),
    ]
    table = pd.DataFrame(rows)
    return dict(table=table, median_sim_std=median_sim_std,
                team_stats_list=team_stats_list, pitcher_stats_list=pitcher_stats_list)


def print_varcomp(res: dict) -> None:
    print(f"\n===== 2. ERROR VARIANCE DECOMPOSITION (z = (realized - sim_mean)/sim_std) =====")
    print(f"median sim_std across all gradeable lineups: {res['median_sim_std']:.2f} FPTS\n")
    print("raw_var can be negative (method-of-moments) -- that means the between-cell")
    print("spread is no bigger than pure sampling noise predicts, i.e. no real effect;")
    print("sigma_z/sigma_fpts clip it at 0 for display, ci_lo/ci_hi likewise.\n")
    print(res["table"].round(4).to_string(index=False))


def cmd_varcomp(z_info: dict = None) -> dict:
    res = compute_varcomp(z_info)
    print_varcomp(res)
    res["table"].to_csv(AUDIT_DIR / "varcomp.csv", index=False)
    print(f"\nwrote {AUDIT_DIR / 'varcomp.csv'}")
    return res


# ---------------------------------------------------------------------------
# 3. signal -- currency signal vs model-free controls
# ---------------------------------------------------------------------------

SIGNAL_SIM_CURRENCIES = ["p_win", "ev_dollars", "ev_tail", "p_cash"]
SIGNAL_FREE_CURRENCIES = ["proj_score", "neg_own", "own_prod_log", "saber_roi"]


@contextmanager
def _patched_currencies():
    """Temporarily wraps backtest_lab._candidate_currencies to add
    own_prod_log (the one model-free currency in the plan that
    _candidate_currencies doesn't already compute), so cmd_currencies'
    existing decile-cell-building loop can be reused unmodified for it too.
    Restored in `finally` -- no permanent change to the imported module."""
    orig = lab._candidate_currencies

    def _ext(sd):
        out = dict(orig(sd))
        out["own_prod_log"] = np.broadcast_to(
            np.asarray(sd.feats["own_prod_log"], dtype=np.float64), (len(sd.cids), sd.M),
        )
        return out

    lab._candidate_currencies = _ext
    try:
        yield
    finally:
        lab._candidate_currencies = orig


def _decile_lift_df(seed: int, calib: bool = False) -> pd.DataFrame:
    """Reruns backtest_lab.cmd_currencies (unmodified) for one seed under
    the own_prod_log patch above, and reads back the CSV it writes -- avoids
    reimplementing its decile-cell construction loop."""
    buf = io.StringIO()
    with _patched_currencies(), redirect_stdout(buf):
        lab.cmd_currencies(seed=seed, calib=calib)
    # dtype=str on slate: it's an all-digit string ("07222026") that pandas'
    # sniffer otherwise infers as int64, silently dropping the leading zero
    # and breaking every downstream join keyed on the slate string.
    df = pd.read_csv(lab.ORACLE_DIR.parent / "lab_decile_lift.csv", dtype={"slate": str})
    df["seed"] = seed
    return df


def _per_slate_topbot(df: pd.DataFrame, metric: str, n_deciles: int = 10) -> dict:
    """{currency: {slate: top-minus-bottom decile contrast}} -- the raw
    per-slate values _decile_consistency reduces to n_pos/sign_p/LOSO,
    extracted via the same _decile_agg call (not a reimplementation of the
    decile bucketing/weighting itself) because the head-to-head sign test
    needs the per-slate numbers, not the summary."""
    out = {}
    for (cur, s), g in df.groupby(["currency", "slate"]):
        a = _decile_agg(g.assign(slate=s))[metric]
        if len(a) < n_deciles:
            continue
        out.setdefault(cur, {})[s] = a.iloc[-1] - a.iloc[0]
    return out


def compute_signal(seeds=(42, 137, 4242), calib: bool = False):
    dfs = [_decile_lift_df(s, calib) for s in seeds]
    full = pd.concat(dfs, ignore_index=True)
    want = SIGNAL_SIM_CURRENCIES + SIGNAL_FREE_CURRENCIES
    present = set(full["currency"].unique())
    keep = [c for c in want if c in present]
    missing = [c for c in want if c not in present]
    df = full[full["currency"].isin(keep)].copy()

    decile_rows = []
    topbot_by_seed = {}
    for seed in seeds:
        sub = df[df.seed == seed]
        for metric in ("$/entry", "top1%"):
            cons = _decile_consistency(sub, metric, 10)
            for cur in keep:
                if cur not in cons.index:
                    continue
                row = cons.loc[cur]
                decile_rows.append({
                    "seed": seed, "metric": metric, "currency": cur,
                    "top-bot": row["top-bot"], "n_pos": row["n_pos"],
                    "sign_p": row["sign_p"], "LOSO_lo": row["LOSO_lo"],
                    "LOSO_hi": row["LOSO_hi"],
                })
        topbot_by_seed[seed] = _per_slate_topbot(sub, "$/entry", 10)

    decile_df = pd.DataFrame(decile_rows)

    h2h_rows = []
    sim_currencies = [c for c in SIGNAL_SIM_CURRENCIES if c in keep]
    for seed in seeds:
        per = topbot_by_seed.get(seed, {})
        proj = per.get("proj_score")
        if not proj:
            continue
        for cur in sim_currencies:
            if cur not in per:
                continue
            common = sorted(set(per[cur]) & set(proj))
            if not common:
                continue
            diff = np.array([per[cur][s] - proj[s] for s in common])
            n_pos = int((diff > 0).sum())
            h2h_rows.append({
                "seed": seed, "currency": cur, "n_slates": len(diff),
                "mean_diff_$/entry": float(diff.mean()),
                "n_pos": f"{n_pos}/{len(diff)}",
                "sign_p": _sign_p(n_pos, len(diff)),
            })
    h2h_df = pd.DataFrame(h2h_rows)
    if missing:
        print(f"    note: currencies unavailable on every slate, skipped: {missing}")
    return decile_df, h2h_df, topbot_by_seed


def print_signal(decile_df: pd.DataFrame, h2h_df: pd.DataFrame) -> None:
    print(f"\n===== 3. CURRENCY SIGNAL vs MODEL-FREE CONTROLS (3 seeds) =====")
    print("top-bot = entry-weighted top-minus-bottom decile contrast (see backtest_lab")
    print("_decile_consistency); n_pos/sign_p/LOSO gate a pooled gradient against being")
    print("one slate's luck.\n")
    for metric in ("$/entry", "top1%"):
        print(f"-- {metric} --")
        sub = decile_df[decile_df.metric == metric].drop(columns="metric")
        piv = sub.pivot(index="currency", columns="seed")
        print(piv.round(4).to_string())
        print()

    print("-- head-to-head: sim currency's per-slate top-bot MINUS proj_score's --")
    print("(does the sim beat the best model-free control, per slate, at the same seed?)")
    if h2h_df.empty:
        print("  (no overlapping slates -- proj_score or sim currencies unavailable)")
    else:
        print(h2h_df.round(4).to_string(index=False))


def cmd_signal():
    decile_df, h2h_df, topbot_by_seed = compute_signal()
    print_signal(decile_df, h2h_df)
    decile_df.to_csv(AUDIT_DIR / "signal.csv", index=False)
    h2h_df.to_csv(AUDIT_DIR / "signal_h2h.csv", index=False)
    print(f"\nwrote {AUDIT_DIR / 'signal.csv'} and {AUDIT_DIR / 'signal_h2h.csv'}")
    return decile_df, h2h_df, topbot_by_seed


# ---------------------------------------------------------------------------
# 4. crowding -- consensus-crowding audit
# ---------------------------------------------------------------------------

def _parse_lineup_string(s: str) -> list:
    return [(m.group(1), m.group(2).strip()) for m in _POS_TOKEN_RE.finditer(s)]


def _crowding_metrics(keys: np.ndarray, teams: np.ndarray, counts: np.ndarray,
                      band_size: int, min_stack: int = 4) -> dict:
    """Crowding metrics for one band of `band_size` entries (already ordered
    best-first; only the first band_size rows of each array are used).

      dupe_rate   fraction of the band that shares its exact 10-player
                  roster with >= 1 other band member.
      max_mult    largest exact-duplicate group size in the band.
      stack_conc  fraction of the band whose primary stack (>= min_stack
                  hitters from one team) IS the band's modal such team.
      eff_frac    1/sum(p_l^2) (effective distinct lineup count, by exact
                  roster) normalized by band_size -- 1.0 = every entry
                  distinct, -> 0 as the band collapses onto few rosters.
    """
    keys, teams, counts = keys[:band_size], teams[:band_size], counts[:band_size]
    n = len(keys)
    if n == 0:
        return dict(dupe_rate=np.nan, max_mult=0, stack_conc=np.nan, eff_frac=np.nan, n=0)
    uniq_keys, mult = np.unique(keys, return_counts=True)
    per_row_mult = mult[np.searchsorted(uniq_keys, keys)]
    dupe_rate = float((per_row_mult > 1).sum()) / n
    max_mult = int(mult.max())

    stacked = teams[counts >= min_stack]
    if len(stacked):
        modal_team, modal_n = Counter(stacked).most_common(1)[0]
        stack_conc = modal_n / n
    else:
        stack_conc = 0.0

    p_l = mult.astype(np.float64) / n
    eff = 1.0 / float(np.sum(p_l ** 2))
    eff_frac = eff / n
    return dict(dupe_rate=dupe_rate, max_mult=max_mult, stack_conc=stack_conc,
               eff_frac=eff_frac, n=n)


def _parse_real_field(slate: str) -> dict:
    """{contest_id: {key, team, count}} -- per real standings zip, every
    entry's exact 10-name roster key (dupe detection), primary hitter team
    and hitter-count (stack concentration), in the zip's own row order
    (verified rank-ascending/best-first for every contest checked). Cached
    to audit/{slate}_fieldlineups.npz since parsing every zip's Lineup
    column is the expensive part of this command and is a pure function of
    the archive.

    Ambiguous name->team lookups (a DKSalaries name shared by two teams that
    slate) are resolved to whichever row's TeamAbbrev appears first --
    second-order for stack metrics per the plan.
    """
    cache = AUDIT_DIR / f"{slate}_fieldlineups.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        cids = [str(x) for x in z["contest_id"]]
        return {cid: dict(key=z[f"key_{j}"], team=z[f"team_{j}"], count=z[f"count_{j}"])
                for j, cid in enumerate(cids)}

    d = ARCHIVE_DIR / slate
    real = load_real_contests(d)
    raw = pd.read_csv(d / "DKSalaries.csv")
    name_team = {}
    for r in raw.itertuples():
        name_team.setdefault(r.Name, r.TeamAbbrev)

    out = {"contest_id": np.array([c["contest_id"] for c in real])}
    result = {}
    for j, c in enumerate(real):
        stem = c["contest_id"].split(":", 1)[1]
        with zipfile.ZipFile(d / f"{stem}.zip") as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv_mod.reader(
                zf.read(name).decode("utf-8-sig", errors="replace").splitlines()))
        keys, teams, counts = [], [], []
        for r in rows[1:]:
            if not r or not r[0].strip().isdigit():
                continue
            parsed = _parse_lineup_string(r[5])
            names = sorted(nm for _, nm in parsed)
            hitters = [nm for pos, nm in parsed if pos != "P"]
            tc = Counter(name_team.get(nm, "") for nm in hitters)
            team, cnt = tc.most_common(1)[0] if tc else ("", 0)
            keys.append("|".join(names))
            teams.append(team)
            counts.append(cnt)
        out[f"key_{j}"] = np.array(keys)
        out[f"team_{j}"] = np.array(teams)
        out[f"count_{j}"] = np.array(counts, dtype=np.int64)
        result[c["contest_id"]] = dict(key=out[f"key_{j}"], team=out[f"team_{j}"],
                                       count=out[f"count_{j}"])
    np.savez_compressed(cache, **out)
    return result


def _real_crowding_for_slate(slate: str) -> pd.DataFrame:
    field = _parse_real_field(slate)
    d = ARCHIVE_DIR / slate
    real = load_real_contests(d)
    try:
        realz = np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False)
        k_by_cid = dict(zip([str(x) for x in realz["contest_id"]], realz["k"]))
    except FileNotFoundError:
        k_by_cid = {}

    rows = []
    for c in real:
        cid = c["contest_id"]
        fl = field[cid]
        n_field = len(fl["key"])
        band = max(1, round(0.01 * n_field))
        m = _crowding_metrics(fl["key"], fl["team"], fl["count"], band)

        payout, sorted_scores = c["payout_arr"], c["sorted_scores"]
        n_paying = int((payout > 0).sum())
        if n_paying > 0 and len(sorted_scores) > 0:
            n_paying = min(n_paying, len(sorted_scores))
            paying_scores = sorted_scores[-n_paying:]
            uniq, cnt = np.unique(sorted_scores, return_counts=True)
            idx = np.searchsorted(uniq, paying_scores)
            tie_width = float(cnt[idx].mean())
        else:
            tie_width = np.nan

        rows.append({
            "slate": slate, "contest_id": cid, "contest": c["contest"],
            "n_field": n_field, "band": band, "k": int(k_by_cid.get(cid, 0)),
            "dupe_rate": m["dupe_rate"], "max_mult": m["max_mult"],
            "stack_conc": m["stack_conc"], "eff_frac": m["eff_frac"],
            "tie_width": tie_width,
        })
    return pd.DataFrame(rows)


def _consensus_row(slate: str) -> dict:
    from scipy.stats import spearmanr

    sd = lab.load_slate(slate, SEED, False)
    ev = sd.currency("ev_dollars", "B")
    k = sd.k.astype(np.float64)
    w_ev = (ev * k[:, None]).sum(axis=0) / k.sum() if k.sum() > 0 else ev.mean(axis=0)
    own = sd.feats["own_sum"]
    mask = sd.ok & np.isfinite(own) & np.isfinite(w_ev)
    if mask.sum() < 10:
        rho, p = np.nan, np.nan
    else:
        rho, p = spearmanr(own[mask], w_ev[mask])
    return {"slate": slate, "spearman_own_ev": float(rho), "p": float(p), "n": int(mask.sum())}


def compute_crowding(pit_df: pd.DataFrame, sim_crowd_df: pd.DataFrame,
                     pwin_topbot_by_slate: dict):
    slates = sorted(sim_crowd_df["slate"].unique())
    contest_rows, real_summary_rows = [], []
    for s in slates:
        cdf = _real_crowding_for_slate(s)
        contest_rows.append(cdf)
        real_summary_rows.append({
            "slate": s,
            "real_dupe_rate": cdf["dupe_rate"].mean(),
            "real_max_mult": cdf["max_mult"].max(),
            "real_stack_conc": cdf["stack_conc"].mean(),
            "real_eff_frac": cdf["eff_frac"].mean(),
            "real_tie_width": cdf["tie_width"].mean(),
            "n_contests": len(cdf),
        })
        print(f"    {s}: real field parsed ({len(cdf)} contests)", flush=True)
    contest_df = pd.concat(contest_rows, ignore_index=True)
    real_summary = pd.DataFrame(real_summary_rows).set_index("slate")
    sim_summary = sim_crowd_df.set_index("slate")[
        ["dupe_rate", "max_mult", "stack_conc", "eff_frac", "band", "n_field"]
    ].add_prefix("sim_")
    summary_df = real_summary.join(sim_summary).reset_index()

    pit_by_slate = pit_df.set_index("slate")["mean_pit"].to_dict()
    consensus_rows = []
    for s in slates:
        crow = _consensus_row(s)
        acc = abs(pit_by_slate.get(s, np.nan) - 0.5)
        crow.update(model_accuracy=acc, pwin_topbot=pwin_topbot_by_slate.get(s, np.nan))
        consensus_rows.append(crow)
    consensus_df = pd.DataFrame(consensus_rows)
    return contest_df, summary_df, consensus_df


def print_crowding(contest_df: pd.DataFrame, summary_df: pd.DataFrame,
                   consensus_df: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    print(f"\n===== 4. CONSENSUS-CROWDING: REAL vs SIMULATED TOP-1% BAND =====")
    print("real_* = per-slate mean across that slate's real contests (equal weight per")
    print("contest); sim_* = one ownership-sampled opponent field per slate, ranked by")
    print("mean sim score, same top-1% band size. Gap = field-model's crowding bias.\n")
    cols = ["slate", "real_dupe_rate", "sim_dupe_rate", "real_stack_conc", "sim_stack_conc",
            "real_eff_frac", "sim_eff_frac", "real_tie_width", "n_contests", "sim_n_field"]
    print(summary_df[[c for c in cols if c in summary_df.columns]].round(4).to_string(index=False))

    print(f"\n-- consensus-proximity (n=9, report don't oversell) --")
    print(consensus_df.round(4).to_string(index=False))
    valid = consensus_df.dropna(subset=["model_accuracy", "pwin_topbot"])
    if len(valid) >= 3:
        rho, p = spearmanr(valid["model_accuracy"], valid["pwin_topbot"])
        acc_med, lift_med = valid["model_accuracy"].median(), valid["pwin_topbot"].median()
        concordant = int((np.sign(valid["model_accuracy"] - acc_med)
                         == np.sign(valid["pwin_topbot"] - lift_med)).sum())
        print(f"\nSpearman(model_accuracy=|mean_pit-0.5|, p_win top-bot lift) = "
              f"{rho:.3f} (p={p:.3f}, n={len(valid)})")
        print(f"median-split concordance (same side of both medians): "
              f"{concordant}/{len(valid)}")
    else:
        print("\n(fewer than 3 slates with both quantities finite -- skipping correlation)")


def cmd_crowding(pit_df=None, sim_crowd_df=None, topbot_by_seed=None):
    if pit_df is None or sim_crowd_df is None:
        full_pit_df, _, full_sim_crowd_df = _run_core(SEED)
        pit_df = full_pit_df if pit_df is None else pit_df
        sim_crowd_df = full_sim_crowd_df if sim_crowd_df is None else sim_crowd_df
    if topbot_by_seed is None:
        _, _, topbot_by_seed = compute_signal(seeds=(SEED,))

    contest_df, summary_df, consensus_df = compute_crowding(
        pit_df, sim_crowd_df, topbot_by_seed.get(SEED, {}).get("p_win", {}),
    )
    print_crowding(contest_df, summary_df, consensus_df)
    contest_df.to_csv(AUDIT_DIR / "crowding_contests.csv", index=False)
    summary_df.to_csv(AUDIT_DIR / "crowding.csv", index=False)
    consensus_df.to_csv(AUDIT_DIR / "crowding_consensus.csv", index=False)
    print(f"\nwrote {AUDIT_DIR / 'crowding.csv'} (+ crowding_contests.csv, "
          f"crowding_consensus.csv)")
    return contest_df, summary_df, consensus_df


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------

def cmd_all():
    print("===== building shared per-slate substrate (pit + varcomp + sim-crowding) =====")
    pit_df, z_info, sim_crowd_df = _run_core(SEED)

    stab_df = compute_pit_all(SEED, (pit_df, z_info, sim_crowd_df))[1]
    print_pit(pit_df, stab_df)
    pd.concat([pit_df, stab_df[stab_df.seed != SEED]], ignore_index=True).to_csv(
        AUDIT_DIR / "pit.csv", index=False)
    print(f"wrote {AUDIT_DIR / 'pit.csv'}")

    varcomp_res = compute_varcomp(z_info)
    print_varcomp(varcomp_res)
    varcomp_res["table"].to_csv(AUDIT_DIR / "varcomp.csv", index=False)
    print(f"wrote {AUDIT_DIR / 'varcomp.csv'}")

    decile_df, h2h_df, topbot_by_seed = compute_signal()
    print_signal(decile_df, h2h_df)
    decile_df.to_csv(AUDIT_DIR / "signal.csv", index=False)
    h2h_df.to_csv(AUDIT_DIR / "signal_h2h.csv", index=False)
    print(f"wrote {AUDIT_DIR / 'signal.csv'} and {AUDIT_DIR / 'signal_h2h.csv'}")

    contest_df, summary_df, consensus_df = compute_crowding(
        pit_df, sim_crowd_df, topbot_by_seed.get(SEED, {}).get("p_win", {}),
    )
    print_crowding(contest_df, summary_df, consensus_df)
    contest_df.to_csv(AUDIT_DIR / "crowding_contests.csv", index=False)
    summary_df.to_csv(AUDIT_DIR / "crowding.csv", index=False)
    consensus_df.to_csv(AUDIT_DIR / "crowding_consensus.csv", index=False)
    print(f"wrote {AUDIT_DIR / 'crowding.csv'} (+ crowding_contests.csv, crowding_consensus.csv)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "pit":
        cmd_pit()
    elif cmd == "varcomp":
        cmd_varcomp()
    elif cmd == "signal":
        cmd_signal()
    elif cmd == "crowding":
        cmd_crowding()
    elif cmd == "all":
        cmd_all()
    else:
        raise SystemExit(f"unknown command {cmd!r} (pit|varcomp|signal|crowding|all)")


if __name__ == "__main__":
    main()
