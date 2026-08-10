"""Cross-arm portfolio comparison: player-exposure overlap and across-worlds
score correlation between self_play, p_win (production), and the
leverage-family arms (leverage_rank_only, team_diverse_leverage).

Motivated by the question "which arms behave differently, and would running
several together hedge each other, vs. which are just redundant copies of
the same bet?" -- rate-ladder tables (hit99/hit99.9/cash%/$/entry) say which
arm wins on THIS realized archive, but not whether two arms are making the
same bet dressed differently. Two metrics here answer that directly:

  - player exposure correlation: pooled per-player entry-count vectors,
    Pearson r between arms. High r = same players, just re-shuffled into
    lineups differently. Low/negative r = genuinely different bets.
  - across-worlds portfolio score correlation: each arm's WHOLE portfolio's
    total simulated score in each of n_sims Monte Carlo worlds (same shared
    sim_results per slate, so both arms are scored against the identical set
    of realized worlds), Pearson r across those worlds. This is the "do these
    two arms win and lose together" question -- literal hedge/redundancy
    measurement, not just static roster overlap.
  - team-concentration structure (teams/slate, top-team share%): the same
    metric compare_diverse_arms.py already reports for the leverage-family
    arms + production, extended here to self_play/p_win so it's directly
    comparable -- answers "is this arm's edge (if any) concentrated on a
    handful of teams that happened to boom on these specific slates, or
    genuinely spread" (see project memory project-leverage-a1-adjudication-
    result's concentration finding for why this matters alongside any raw
    rate/dollar comparison).
  - player-COMBO concentration (2-player and 3-player): a finer-grained cut
    than team-level -- two lineups can share most of a roster without
    sharing the same PRIMARY team (e.g. same pitcher + same 3-stack from a
    secondary team), which team-concentration alone can't see. Counts every
    2-/3-player combination across an arm's whole portfolio and reports the
    single most-repeated combo's share of all lineups, pooled correctly
    across slates the same way team concentration is (merge combo counts
    across slates FIRST, then take max/total -- summing each slate's own
    top-combo% would be wrong whenever a different combo is "top" on
    different slates).

self_play/p_win portfolios are built fresh (same machinery as
scripts/eval_self_play_selector.py) since results.csv doesn't retain the
actual Lineup objects, only graded outcomes. leverage_rank_only/
team_diverse_leverage come from tests/backtest_lab.py's oracle-cached
substrate -- do NOT run this while tests/backtest_oracle.py's rebuild is in
flight, it reads the exact ORACLE_DIR files that rebuild overwrites.

Scoped to a small representative SLATES subset by default (self_play/p_win
construction costs ~5-8 min/slate) -- pass slates on the command line to
widen it.

Usage
-----
    source venv/bin/activate
    python scripts/compare_arm_portfolios.py [slate ...]
"""
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.optimization import self_play  # noqa: E402
from tests.bt_core import (  # noqa: E402
    LIVE_CFG, _FakeGroup, build_slate_context, load_real_contests, prod_order,
)
from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, _candidate_currencies, load_leverage, load_slate, select_greedy,
    select_team_diverse_leverage,
)

N_SIMS = int(os.environ.get("BT_NSIMS", LIVE_CFG["simulation"]["n_sims"]))
SHARPNESS = float(LIVE_CFG["gpp"].get("external_pool_pwin_sharpness", 0.05))
POOL_SIZE = int(os.environ.get("SELF_PLAY_POOL_SIZE", self_play._SELF_PLAY_POOL_CAP))
SEED = 42
PWIN_ADMIT_FLOOR, PWIN_ADMIT_MULT, PWIN_EVW = 2000, 0.0, 0.25  # flat-2000 production baseline

OUT_DIR = PROJECT_ROOT / "outputs" / "self_play_eval"
SIM_CACHE_DIR = OUT_DIR / "sim_cache"
OUT_CSV = OUT_DIR / "arm_portfolio_overlap.csv"

# Spread across the archive rather than adjacent days, and keep it cheap by
# default -- self_play/p_win construction is the expensive part (~5-8 min/slate).
DEFAULT_SLATES = ["07222026", "07282026", "08032026"]


def _self_play_and_pwin_portfolios(slate: str):
    """-> (portfolios, ctx) where portfolios = {"self_play"|"p_win":
    {contest_id: [frozenset(player_id), ...]}}, ctx is the bt_core slate
    context (players_df/sim_results reused for the world-correlation calc)."""
    d = PROJECT_ROOT / "archive" / slate
    real = load_real_contests(d)
    ctx = build_slate_context(
        d, SEED, False, real, n_sims=N_SIMS, sharpness=SHARPNESS,
        sim_cache_dir=SIM_CACHE_DIR,
    )
    own_vec = ctx["players_df"]["ownership"].astype(float).to_numpy()
    sp_ctx = self_play.build_self_play_context(
        ctx["sim_results"], ctx["players_df"], own_vec, ctx["pool"],
        base_pool_size=POOL_SIZE, base_pool_seed=SEED,
    )
    contests = [c for c in ctx["contests"] if c["k"] > 0]
    prize_pool = {c["contest_id"]: float(c["payout_arr"].sum()) for c in contests}
    order = prod_order([c["contest_id"] for c in contests], [c["fee"] for c in contests], prize_pool)
    ordered = [contests[i] for i in order]

    alloc_sp = ep.self_play_allocate_contests(ordered, sp_ctx, rng_seed=SEED)
    groups = [_FakeGroup(c["contest_id"], c["k"]) for c in ordered]
    alloc_pw = ep.allocate_contests(
        ctx["pool"], ctx["corr"], groups, risk=3.0,
        evw_base=PWIN_EVW, evw_max=PWIN_EVW, ev_type="p_win",
        p_win_cull=ctx["p_win_cull"], p_win_select=ctx["p_win_select"],
        p_win_admit_n=PWIN_ADMIT_FLOOR, p_win_admit_multiplier=PWIN_ADMIT_MULT,
    )

    def _to_dict(entry_plan, portfolio):
        out: dict[str, list] = {}
        for (cid, _e), (lu, _roi) in zip(entry_plan, portfolio):
            out.setdefault(cid, []).append(frozenset(int(p) for p in lu.player_ids))
        return out

    return {
        "self_play": _to_dict(alloc_sp.entry_plan, alloc_sp.portfolio),
        "p_win": _to_dict(alloc_pw.entry_plan, alloc_pw.portfolio),
    }, ctx


def _leverage_portfolios(slate: str):
    """-> {"leverage_rank_only"|"team_diverse_leverage":
    {contest_id: [frozenset(player_id), ...]}} from the oracle-cached pool."""
    sd = load_slate(slate, SEED, False)
    with np.load(ORACLE_DIR / f"{slate}_real.npz", allow_pickle=False) as z:
        pids_arr = z["player_ids"]

    lev = load_leverage(slate)["leverage_ratio_mean"]
    picks_rank = select_greedy(sd, lev, lev, floor_pct=30.0, admit_n=2000)
    picks_diverse = select_team_diverse_leverage(sd)

    def _to_dict(picks):
        return {cid: [frozenset(int(p) for p in pids_arr[i]) for i in idxs]
                for cid, idxs in picks.items()}

    return {
        "leverage_rank_only": _to_dict(picks_rank),
        "team_diverse_leverage": _to_dict(picks_diverse),
    }


def _exposure_vector(port: dict, universe: list) -> np.ndarray:
    counts: dict[int, int] = {}
    for lineups in port.values():
        for fs in lineups:
            for p in fs:
                counts[p] = counts.get(p, 0) + 1
    return np.array([counts.get(p, 0) for p in universe], dtype=float)


def _world_score_vector(port: dict, sim_results, pid_to_col: dict) -> np.ndarray:
    """Sum, over every entry in the portfolio, that lineup's per-sim-world
    score -- a length-n_sims vector: this portfolio's total score in each
    Monte Carlo world."""
    mat = sim_results.results_matrix  # (n_sims, n_players)
    total = np.zeros(mat.shape[0], dtype=np.float64)
    for lineups in port.values():
        for fs in lineups:
            cols = [pid_to_col[p] for p in fs if p in pid_to_col]
            if cols:
                total += mat[:, cols].sum(axis=1)
    return total


def _primary_team_map(slate: str) -> tuple[dict, dict]:
    """Same convention as inspect_arm_portfolios.py::team_and_position_maps
    -- DKSalaries Position is SP/RP, never plain "P", so pitcher detection
    must substring-match (gotcha bitten twice historically, see
    project-rival-portfolio-shaidyadvice)."""
    sal = pd.read_csv(PROJECT_ROOT / "archive" / slate / "DKSalaries.csv")
    team_map = dict(zip(sal["ID"].astype(int), sal["TeamAbbrev"].astype(str)))
    is_pitcher = sal["Position"].astype(str).str.contains("P")
    pos_map = dict(zip(sal["ID"].astype(int), is_pitcher))
    return team_map, pos_map


def _primary_team_of_lineup(player_ids, team_map: dict, pos_map: dict) -> str:
    hitters = [p for p in player_ids if not pos_map.get(int(p), False)]
    teams = pd.Series([team_map.get(int(p), "?") for p in hitters])
    return teams.value_counts().idxmax()


def _team_counts_by_arm(portfolios: dict, slate: str) -> dict:
    """arm -> {team: count} for this one slate -- the raw counts, not yet
    summarized, so callers can either report per-slate stats directly or
    MERGE counts across slates before computing a top-team share (summing
    per-slate top-team% would be wrong: a different team can be "top" each
    slate, so the correct pooled share needs one accumulated {team: count}
    dict across all slates first, exactly like compare_diverse_arms.py's
    structure() function -- see _pooled_team_concentration below)."""
    team_map, pos_map = _primary_team_map(slate)
    out: dict[str, dict[str, int]] = {}
    for arm, port in portfolios.items():
        team_counts: dict[str, int] = {}
        for lineups in port.values():
            for fs in lineups:
                t = _primary_team_of_lineup(fs, team_map, pos_map)
                team_counts[t] = team_counts.get(t, 0) + 1
        out[arm] = team_counts
    return out


def _team_concentration(counts: dict) -> pd.DataFrame:
    """Per-slate summary from one slate's _team_counts_by_arm output."""
    rows = []
    for arm, team_counts in counts.items():
        total = sum(team_counts.values())
        rows.append(dict(
            arm=arm, n_teams=len(team_counts),
            top_team_share_pct=100 * max(team_counts.values()) / total if total else float("nan"),
            n_entries=total,
        ))
    return pd.DataFrame(rows)


def _pooled_team_concentration(all_counts: list) -> pd.DataFrame:
    """Correctly pooled across slates: merge each arm's {team: count} dicts
    from every slate FIRST, then take max/total on the merged dict -- not an
    average of each slate's own top-team%, which would be wrong whenever a
    different team is "top" on different slates."""
    arms = set()
    for counts in all_counts:
        arms.update(counts.keys())
    rows = []
    per_slate_n_teams: dict[str, list] = {a: [] for a in arms}
    for arm in arms:
        merged: dict[str, int] = {}
        for counts in all_counts:
            tc = counts.get(arm, {})
            per_slate_n_teams[arm].append(len(tc))
            for t, c in tc.items():
                merged[t] = merged.get(t, 0) + c
        total = sum(merged.values())
        rows.append(dict(
            arm=arm,
            mean_teams_per_slate=float(np.mean(per_slate_n_teams[arm])) if per_slate_n_teams[arm] else float("nan"),
            pooled_top_team_share_pct=100 * max(merged.values()) / total if total else float("nan"),
            pooled_n_teams=len(merged), pooled_entries=total,
        ))
    return pd.DataFrame(rows)


def _combo_counts_by_arm(portfolios: dict, k: int) -> dict:
    """arm -> {frozenset(k player_ids): count} across every lineup in that
    arm's WHOLE portfolio this slate (every contest combined) -- a finer
    cut than primary-team: two lineups sharing a pitcher + 3-stack from a
    SECONDARY team look identical here even if their primary teams differ."""
    out: dict[str, dict[frozenset, int]] = {}
    for arm, port in portfolios.items():
        counts: dict[frozenset, int] = {}
        for lineups in port.values():
            for fs in lineups:
                for combo in combinations(sorted(fs), k):
                    key = frozenset(combo)
                    counts[key] = counts.get(key, 0) + 1
        out[arm] = counts
    return out


def _combo_concentration(counts: dict, n_lineups: dict) -> pd.DataFrame:
    """Per-slate summary from one slate's _combo_counts_by_arm output."""
    rows = []
    for arm, c in counts.items():
        n = n_lineups.get(arm, 0)
        top = max(c.values()) if c else 0
        rows.append(dict(
            arm=arm, n_distinct_combos=len(c), top_combo_count=top,
            top_combo_share_pct=100 * top / n if n else float("nan"), n_lineups=n,
        ))
    return pd.DataFrame(rows)


def _pooled_combo_concentration(all_counts: list, all_n_lineups: list) -> pd.DataFrame:
    """Correctly pooled across slates -- same principle as
    _pooled_team_concentration: merge {combo: count} dicts across slates
    FIRST, then take max/total, since a different combo can be "top" on
    different slates."""
    arms = set()
    for counts in all_counts:
        arms.update(counts.keys())
    rows = []
    for arm in arms:
        merged: dict[frozenset, int] = {}
        total_n = 0
        for counts, n_lineups in zip(all_counts, all_n_lineups):
            for combo, c in counts.get(arm, {}).items():
                merged[combo] = merged.get(combo, 0) + c
            total_n += n_lineups.get(arm, 0)
        top = max(merged.values()) if merged else 0
        rows.append(dict(
            arm=arm, pooled_n_distinct_combos=len(merged), pooled_top_combo_count=top,
            pooled_top_combo_share_pct=100 * top / total_n if total_n else float("nan"),
            pooled_n_lineups=total_n,
        ))
    return pd.DataFrame(rows)


def run_slate(slate: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    print(f"{slate}: building self_play/p_win portfolios (fresh construction)...")
    sp_pw, ctx = _self_play_and_pwin_portfolios(slate)
    print(f"  done in {time.time() - t0:.0f}s")

    lev = _leverage_portfolios(slate)
    portfolios = {**sp_pw, **lev}

    sal = pd.read_csv(PROJECT_ROOT / "archive" / slate / "DKSalaries.csv")
    all_ids = sal["ID"].astype(int).tolist()

    sim_results = ctx["sim_results"]
    pid_to_col = {int(pid): j for j, pid in enumerate(sim_results.player_ids)}

    exposure = {arm: _exposure_vector(port, all_ids) for arm, port in portfolios.items()}
    world_score = {arm: _world_score_vector(port, sim_results, pid_to_col)
                    for arm, port in portfolios.items()}

    arms = list(portfolios.keys())
    rows = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            exp_r = float(np.corrcoef(exposure[a], exposure[b])[0, 1])
            world_r = float(np.corrcoef(world_score[a], world_score[b])[0, 1])
            rows.append(dict(slate=slate, arm_a=a, arm_b=b,
                              exposure_corr=exp_r, world_score_corr=world_r))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    team_counts = _team_counts_by_arm(portfolios, slate)
    print(_team_concentration(team_counts).round(3).to_string(index=False))

    n_lineups = {arm: sum(len(v) for v in port.values()) for arm, port in portfolios.items()}
    pair_counts = _combo_counts_by_arm(portfolios, 2)
    triple_counts = _combo_counts_by_arm(portfolios, 3)
    print(_combo_concentration(pair_counts, n_lineups).round(3).to_string(index=False))
    print(_combo_concentration(triple_counts, n_lineups).round(3).to_string(index=False))
    print(f"{slate}: total {time.time() - t0:.0f}s")
    return df, team_counts, pair_counts, triple_counts, n_lineups


def main() -> None:
    slates = [s for s in sys.argv[1:] if s.isdigit()] or DEFAULT_SLATES
    all_rows = []
    all_team_counts = []
    all_pair_counts = []
    all_triple_counts = []
    all_n_lineups = []
    for slate in slates:
        try:
            df, team_counts, pair_counts, triple_counts, n_lineups = run_slate(slate)
            all_rows.append(df)
            all_team_counts.append(team_counts)
            all_pair_counts.append(pair_counts)
            all_triple_counts.append(triple_counts)
            all_n_lineups.append(n_lineups)
        except FileNotFoundError as e:
            print(f"{slate}: skipping, missing oracle/archive file: {e}")
    if not all_rows:
        return
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n===== pooled across {len(slates)} slate(s) =====")
    print(combined.groupby(["arm_a", "arm_b"])[["exposure_corr", "world_score_corr"]]
          .mean().round(3).to_string())

    pooled_conc = _pooled_team_concentration(all_team_counts)
    pooled_conc.to_csv(OUT_DIR / "arm_team_concentration.csv", index=False)
    print(f"\n===== team concentration, correctly pooled across {len(slates)} slate(s) =====")
    print(pooled_conc.sort_values("pooled_top_team_share_pct").round(3).to_string(index=False))

    pooled_pairs = _pooled_combo_concentration(all_pair_counts, all_n_lineups)
    pooled_triples = _pooled_combo_concentration(all_triple_counts, all_n_lineups)
    pooled_pairs.to_csv(OUT_DIR / "arm_pair_concentration.csv", index=False)
    pooled_triples.to_csv(OUT_DIR / "arm_triple_concentration.csv", index=False)
    print(f"\n===== 2-player combo concentration, correctly pooled across {len(slates)} slate(s) =====")
    print(pooled_pairs.sort_values("pooled_top_combo_share_pct").round(3).to_string(index=False))
    print(f"\n===== 3-player combo concentration, correctly pooled across {len(slates)} slate(s) =====")
    print(pooled_triples.sort_values("pooled_top_combo_share_pct").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
