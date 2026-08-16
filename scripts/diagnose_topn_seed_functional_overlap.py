"""Is the topn_coverage selector's cross-seed variation REAL, or just
relabeling among near-equivalent lineups?

`scripts/eval_topn_payout_ladder.py` reported 41.2% literal pick-identity
overlap between seeds 42 and 137 and I called the selector "not
reproducible." That framing is wrong on its own: the eligible pool holds
~7,800 candidates and `parse_lineup_pool`'s near-duplicate cull only removes
9/10-player matches, so large families of 8/10-equivalent lineups survive by
design. Two runs drawing different members of the same family would show low
IDENTITY overlap while being the same portfolio in every way that affects
results.

So identity overlap is a lower bound on similarity, not a measure of it. This
script measures what actually matters:

  nn_overlap    for each lineup in run A, the largest player-intersection
                with ANY lineup in run B (0-10). If most lineups have a 9/10
                or 10/10 twin, the portfolios are the same object wearing
                different labels.
  exposure_rho  Spearman correlation of per-PLAYER exposure between runs,
                and the max absolute per-player exposure gap. Composition is
                what drives results; identity is not.
  stack_rho     same for per-team hitter-stack exposure.
  objective     worlds_claimed and mean ceiling, already known to be close.

A shuffled control (a random same-size draw from the eligible pool) is
reported for every metric, because nn_overlap in particular is only
interpretable against the pool's own baseline similarity -- in a pool this
correlated, even unrelated portfolios share a lot of players.

Usage
-----
    source venv/bin/activate
    python scripts/diagnose_topn_seed_functional_overlap.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import parse_entry_file  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402

RAW_DIR = os.environ.get("TOPN_REQ_RAW", str(PROJECT_ROOT / "data" / "raw"))
LADDER_DIR = PROJECT_ROOT / "outputs" / "topn_payout_ladder"
SHARED = PROJECT_ROOT / "outputs" / "topn_dupe_discount"
PAIRS = [
    ("baseline_s42", "baseline_s137", "single bar, seed 42 vs 137"),
    ("ladder1_s42", "ladder1_s137", "ladder@1,   seed 42 vs 137"),
    ("baseline_s42", "ladder1_s42", "single bar vs ladder@1 (same seed)"),
]


def nn_overlap(A: list, B: list) -> np.ndarray:
    """(len(A),) largest player-intersection of each A lineup with any B lineup."""
    Bs = [set(lu.player_ids) for lu in B]
    return np.array([max(len(set(a.player_ids) & b) for b in Bs) for a in A])


def exposures(lineups: list, pids: np.ndarray) -> np.ndarray:
    idx = {int(p): i for i, p in enumerate(pids)}
    out = np.zeros(len(pids))
    for lu in lineups:
        for p in lu.player_ids:
            j = idx.get(int(p))
            if j is not None:
                out[j] += 1
    return out / max(1, len(lineups))


def stack_exposure(lineups: list, team_of: dict, pos_of: dict) -> dict:
    counts: dict[str, int] = {}
    for lu in lineups:
        teams = {}
        for p in lu.player_ids:
            if pos_of.get(int(p), "") != "P":
                t = team_of.get(int(p))
                if t:
                    teams[t] = teams.get(t, 0) + 1
        for t, c in teams.items():
            if c >= 3:  # a real stack, not an incidental single bat
                counts[t] = counts.get(t, 0) + 1
    return {t: c / max(1, len(lineups)) for t, c in counts.items()}


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp, paths = cfg["gpp"], cfg["paths"]
    seed = int(gpp.get("rng_seed") or 42)

    found = ep.discover_external_files(RAW_DIR)
    slate_df = DraftKingsSlateIngestor(str(PROJECT_ROOT / paths["dk_slate"])).get_slate_dataframe()
    pool = ep.parse_lineup_pool(
        found["lineups_paths"], set(slate_df["player_id"].astype(int)), require_roi_blocks=False,
    )
    proj_ext = ep.parse_player_projections(found["projections_path"])
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, {int(p) for lu in pool.lineups for p in lu.player_ids},
        PipelineRunner._derive_opponent,
    )
    all_file_entries = []
    for p in sorted(Path(RAW_DIR).glob("*Entries.csv")):
        recs = parse_entry_file(p)
        if recs:
            all_file_entries.append((p, recs))
    groups = ep.group_and_match_contests(all_file_entries, pool)

    fp_size = int(gpp.get("external_pool_topn_field_pool_size", 25_000))
    n_sims = max(int(cfg["simulation"].get("n_sims", 25_000)), ep.topn_total_sims_needed(
        groups, fp_size,
        float(gpp.get("external_pool_topn_sims_per_contest_fraction", 0.5)),
        int(gpp.get("external_pool_topn_sims_min", 0)),
        float(gpp.get("external_pool_topn_sims_reference_field_size", 0.0)),
        float(gpp.get("external_pool_topn_sims_power", 0.0)),
    ))
    with np.load(SHARED / "sim_cache" / f"{found['projections_path'].stem}_{n_sims}_{seed}.npz") as z:
        sim_results = SimulationResults(
            [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
        )
    from src.optimization.leverage import compute_generation_ownership_vec
    gen_own = compute_generation_ownership_vec(
        pool.lineups, sim_results, players_df,
        field_size=float(ep.pwin_field_size(groups, floor=int(gpp.get("n_field_lineups", 5_000)))),
        blend_weight=float(gpp.get("external_pool_topn_generated_leverage_weight", 0.0)),
        sharpness=float(gpp.get("external_pool_pwin_sharpness", 0.05)),
    )
    alloc_pool, _ = ep.augment_topn_pool_with_generated(
        pool, players_df, gen_own,
        int(gpp.get("external_pool_topn_generated_pool_size", 0)), seed + 1,
    )
    lineups_all = alloc_pool.lineups
    floor_scores = ep.compute_pool_ceiling_scores(alloc_pool, players_df)
    floor = ep.compute_proj_score_floor(
        floor_scores, float(gpp.get("external_pool_proj_score_pct", 30.0)))
    elig = np.where(np.isfinite(floor_scores) & (floor_scores >= floor[0]))[0]

    pids = players_df["player_id"].astype(int).to_numpy()
    team_of = dict(zip(pids, players_df["team"]))
    pos_of = dict(zip(pids, players_df["position"]))
    name_of = dict(zip(pids, players_df["name"]))

    def load(arm):
        return [lineups_all[i] for i in np.load(LADDER_DIR / f"pick_idx_{arm}.npy")]

    rng = np.random.default_rng(7)
    rows = []
    for a_arm, b_arm, label in PAIRS:
        A, B = load(a_arm), load(b_arm)
        # Control: two independent random draws from the eligible pool, same
        # sizes -- the pool's own baseline similarity.
        Ca = [lineups_all[i] for i in rng.choice(elig, size=len(A), replace=False)]
        Cb = [lineups_all[i] for i in rng.choice(elig, size=len(B), replace=False)]

        ident = len(set(map(id, A)) & set(map(id, B))) / len(A)
        nn = nn_overlap(A, B)
        nn_ctl = nn_overlap(Ca, Cb)
        ea, eb = exposures(A, pids), exposures(B, pids)
        eca, ecb = exposures(Ca, pids), exposures(Cb, pids)
        used = (ea + eb) > 0
        rho = spearmanr(ea[used], eb[used]).statistic
        rho_ctl = spearmanr(eca[(eca + ecb) > 0], ecb[(eca + ecb) > 0]).statistic
        gap = np.abs(ea - eb)
        worst = int(np.argmax(gap))

        sa, sb = stack_exposure(A, team_of, pos_of), stack_exposure(B, team_of, pos_of)
        teams = sorted(set(sa) | set(sb))
        va, vb = [sa.get(t, 0) for t in teams], [sb.get(t, 0) for t in teams]
        srho = spearmanr(va, vb).statistic if len(teams) > 2 else float("nan")

        rows.append({
            "pair": label,
            "identity_overlap": round(ident, 3),
            "nn>=9": round(float((nn >= 9).mean()), 3),
            "nn>=8": round(float((nn >= 8).mean()), 3),
            "nn_mean": round(float(nn.mean()), 2),
            "nn_mean_ctl": round(float(nn_ctl.mean()), 2),
            "expo_rho": round(float(rho), 3),
            "expo_rho_ctl": round(float(rho_ctl), 3),
            "max_player_gap": round(float(gap.max()), 3),
            "max_gap_player": name_of.get(int(pids[worst]), "?"),
            "stack_rho": round(float(srho), 3),
        })
        print(f"{label}: identity {ident * 100:.1f}%  nn>=9 {(nn >= 9).mean() * 100:.0f}%  "
              f"nn_mean {nn.mean():.2f} (ctl {nn_ctl.mean():.2f})  "
              f"expo_rho {rho:.3f} (ctl {rho_ctl:.3f})  "
              f"max player gap {gap.max() * 100:.1f}pp ({name_of.get(int(pids[worst]), '?')})")

    out = pd.DataFrame(rows)
    out.to_csv(LADDER_DIR / "seed_functional_overlap.csv", index=False)
    print(f"\n{out.to_string(index=False)}")
    print(f"\nsaved: {LADDER_DIR / 'seed_functional_overlap.csv'}")


if __name__ == "__main__":
    main()
