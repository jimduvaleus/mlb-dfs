"""ShaidyAdvice's real 150-entry portfolios, measured with the same N_eff ruler.

[[scripts/eval_portfolio_neff.py]] establishes how many independent bets OUR
shipped portfolios amount to. This asks the only question that makes that
number mean anything: what does the SAME ruler read on a portfolio built by
the process we are trying to understand?

The archived standings zips carry every entrant's full lineup, so for a
max-entry professional they are the finished product of his whole process --
19 complete 150-entry portfolios across 13 slates (Rally Cap / Bat Flip /
Relay Throw). Same slate, same sims, same opponent field, same payout table,
same metric. The only thing that differs is who built the portfolio.

WHY THIS NEEDS ITS OWN SIMS. `players_df` in the main script is built from the
SaberSim pool's players only. Up to 26% of his lineups roster at least one
player our pool never used, and dropping those is NOT neutral -- it discards
precisely his most contrarian entries, biasing his measured diversity
DOWNWARD, which is the direction that would flatter us. So the player set here
is the UNION (our pool + every player he rostered that slate) and the sim is
rebuilt on it. Our own portfolio is re-measured on that same union sim rather
than being carried over from the main run: a number computed against a
different sim is not comparable, and mixing them would be the same error in a
subtler form.

K-MATCHING. N_eff grows sublinearly in k, so comparing his 150 against our ~90
directly would hand him a win on portfolio size alone. Every arm is therefore
also reported at a common k (the smaller of the two), averaged over
`RIV_DRAWS` random subsets.

Arms: shaidy (his real entries), ours (the shipped portfolio at its active
risk), random (uniform from our pool), proj (top-k by projection).

Checkpoint / resume per CLAUDE.md: one row per (slate, contest, arm, space)
appended to outputs/rival_neff/results.csv.

Usage
-----
    source venv/bin/activate
    python scripts/eval_rival_neff.py            # all 19 archived portfolios
    python scripts/eval_rival_neff.py 07222026   # one slate

Env vars
--------
    RIV_ENTRANT  standings handle to grade (default ShaidyAdvice)
    RIV_DRAWS    subsample draws for the k-matched figure (default 25)
    RIV_NSIMS / RIV_FIELD / RIV_FORCE  as in eval_portfolio_neff.py
"""
import csv
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from src.api import external_pool as ep  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pne", PROJECT_ROOT / "scripts" / "eval_portfolio_neff.py")
pne = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pne)
_spec2 = importlib.util.spec_from_file_location(
    "arp", PROJECT_ROOT / "scripts" / "analyze_rival_portfolio.py")
arp = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(arp)

ENTRANT = os.environ.get("RIV_ENTRANT", "ShaidyAdvice")
DRAWS = int(os.environ.get("RIV_DRAWS", "25"))
N_SIMS = int(os.environ.get("RIV_NSIMS", "25000"))
FIELD_N = int(os.environ.get("RIV_FIELD", "10000"))
FORCE = os.environ.get("RIV_FORCE") == "1"
SEED = 42
ARMS = ["shaidy", "ours", "random", "proj"]

OUT_DIR = PROJECT_ROOT / "outputs" / "rival_neff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def rival_lineups(adir: Path, zip_stem: str, entrant: str) -> list[tuple]:
    """His entries for one contest, as tuples of DK display names."""
    with zipfile.ZipFile(adir / f"{zip_stem}.zip") as zf:
        n = [x for x in zf.namelist() if x.endswith(".csv")][0]
        rows = list(csv.reader(io.StringIO(
            zf.read(n).decode("utf-8-sig", errors="replace"))))
    e, _ = arp.parse_standings_rows(rows)
    return list(e[e.handle == entrant]["names"])


def name_index(adir: Path) -> tuple[dict, set]:
    """(name -> player_id, ambiguous names). A name mapping to two ids on one
    slate cannot be resolved from the standings line alone (which carries no
    team), so those lineups are dropped and COUNTED rather than guessed --
    see [[project-duplicate-name-fpts-resolution]]."""
    sal = pd.read_csv(adir / "DKSalaries.csv")
    n2i, dup = {}, set()
    for nm, pid in zip(sal["Name"].astype(str).str.strip(), sal["ID"].astype(int)):
        if nm in n2i and n2i[nm] != pid:
            dup.add(nm)
        n2i[nm] = pid
    return n2i, dup


def union_sim(slate: str, adir: Path, players_df, proj_ext, cfg):
    """Sim over the UNION player set. Cached under a distinct key so it can
    never be confused with the pool-only sims the main script uses."""
    cache = OUT_DIR / f"sim_union_{slate}_{N_SIMS}_{SEED}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float32)
    gpp, paths = cfg["gpp"], cfg["paths"]
    grids = ep.build_quantile_grids(
        proj_ext,
        zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
        scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
        mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
        mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)),
    )
    engine = SimulationEngine(
        EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
        batter_pca_model=None, score_grid=None, quantile_grids=grids,
    )
    st = np.random.get_state()
    np.random.seed(SEED)
    sr = engine.simulate(N_SIMS)
    np.random.set_state(st)
    np.savez_compressed(cache, player_ids=np.asarray(sr.player_ids, dtype=np.int64),
                        results_matrix=sr.results_matrix.astype(np.float32))
    return sr.player_ids, sr.results_matrix.astype(np.float32)


def matched_n_eff(X: np.ndarray, k: int, rng) -> float:
    """Mean N_eff over `DRAWS` random size-k subsets. N_eff is sublinear in k,
    so this is what makes two portfolios of different size comparable."""
    if X.shape[0] <= k:
        return pne.n_eff(X)
    return float(np.mean([pne.n_eff(X[rng.choice(X.shape[0], k, replace=False)])
                          for _ in range(DRAWS)]))


def run_pair(slate: str, zip_stem: str, cfg: dict) -> list[dict]:
    adir = PROJECT_ROOT / "archive" / slate
    n2i, dup = name_index(adir)
    his_names = rival_lineups(adir, zip_stem, ENTRANT)
    if len(his_names) < 20:
        print(f"  {zip_stem}: only {len(his_names)} entries for {ENTRANT} -- skipped")
        return []
    his_ids, n_amb = [], 0
    for t in his_names:
        if any(nm in dup or nm not in n2i for nm in t):
            n_amb += 1
            continue
        his_ids.append([n2i[nm] for nm in t])

    found = ep.discover_external_files(str(adir))
    slate_df = DraftKingsSlateIngestor(str(adir / "DKSalaries.csv")).get_slate_dataframe()
    pool = ep.parse_lineup_pool(found["lineups_paths"],
                                set(slate_df["player_id"].astype(int)),
                                require_roi_blocks=False)
    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    union = pool_pids | {int(p) for ids in his_ids for p in ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, union, PipelineRunner._derive_opponent)

    pid, mat = union_sim(slate, adir, players_df, proj_ext, cfg)
    col_map = {int(p): i for i, p in enumerate(pid)}

    # Any of his players still missing after the union rebuild means no
    # projection existed for them -- report, never silently drop.
    his_ok = [ids for ids in his_ids if all(int(p) in col_map for p in ids)]
    n_nosim = len(his_ids) - len(his_ok)
    if not his_ok:
        print(f"  {zip_stem}: no usable rival lineups")
        return []

    sw = json.loads((adir / "portfolio_sweep_draftkings.json").read_text())
    risk = sw.get("active_risk")
    ent = [e for e in sw.get("sweep", []) if e.get("risk") == risk] or sw.get("sweep", [])
    ours, ours_contest = [], []
    for lu in (ent[0]["lineups"] if ent else []):
        ids = [int(p["player_id"]) for p in lu["players"]]
        if all(i in col_map for i in ids):
            ours.append(ids)
            ours_contest.append(lu["contest_name"])
    if len(ours) < 20:
        print(f"  {zip_stem}: only {len(ours)} of our lineups mapped -- skipped")
        return []

    proj = ep.compute_pool_proj_scores(pool.lineups, players_df)
    rng = np.random.default_rng(int(slate))
    all_cols = pne.lineup_cols([lu.player_ids for lu in pool.lineups], col_map)
    n_take = len(his_ok)
    sets = {
        "shaidy": pne.lineup_cols(his_ok, col_map),
        "ours": pne.lineup_cols(ours, col_map),
        "random": all_cols[rng.choice(len(pool.lineups), min(n_take, len(pool.lineups)),
                                      replace=False)],
        "proj": all_cols[np.argsort(-np.nan_to_num(proj, nan=-np.inf))[:n_take]],
    }
    scores = {a: pne.score_lineups(mat, c) for a, c in sets.items()}
    k_match = min(s.shape[0] for s in scores.values())
    print(f"  {zip_stem}: shaidy {len(his_ok)} ({n_amb} ambiguous, {n_nosim} unprojected), "
          f"ours {len(ours)}, k_match {k_match}")

    rows = []
    for arm, sc in scores.items():
        v, vm = pne.n_eff(sc), matched_n_eff(sc, k_match, np.random.default_rng(7))
        rows.append(dict(slate=slate, contest=zip_stem, arm=arm, space="score",
                         k=sc.shape[0], k_match=k_match,
                         n_eff=round(v, 3), frac=round(v / sc.shape[0], 4),
                         n_eff_matched=round(vm, 3),
                         frac_matched=round(vm / k_match, 4)))
        print(f"    score  {arm:<7} k={sc.shape[0]:>3}  N_eff={v:6.2f} ({v/sc.shape[0]:5.1%})"
              f"   k-matched N_eff={vm:6.2f} ({vm/k_match:5.1%})")

    # --- payout space, in the contest he actually entered -------------------
    real = pne.load_real_contests_tolerant(adir)
    want = arp  # noqa
    from bt_core import ZIP_TO_CONTEST
    display = ZIP_TO_CONTEST.get(zip_stem)
    match = [c for c in real if c["contest"] == display]
    if not match:
        print(f"    (no payout table for {display} -- score space only)")
        return rows
    c = max(match, key=lambda v: v["n_field"])

    own = players_df["ownership"].astype(float).to_numpy()
    fpool = pne.load_field(slate, players_df, own)
    fcols = np.array([[col_map[int(p)] for p in r] for r in fpool], dtype=np.int32)
    # One field_ranks call for every arm at once: its cost is the (S x F)
    # field scoring, which does not depend on how many of our rows ride along.
    order = list(scores)
    stacked = np.concatenate([scores[a] for a in order], axis=0)
    t0 = time.time()
    frac_all = pne.field_ranks(stacked, mat, fcols)
    print(f"    [field ranks {time.time()-t0:.0f}s]")
    off = np.cumsum([0] + [scores[a].shape[0] for a in order])

    for i, arm in enumerate(order):
        sc = scores[arm]
        fr = frac_all[off[i]:off[i + 1]]
        sel = (np.flatnonzero(np.array(ours_contest) == display) if arm == "ours"
               else np.arange(sc.shape[0]))
        if len(sel) < 2:
            continue
        dollars = pne.payout_series(fr[sel], sc[sel], c["n_field"], c["payout_arr"])
        live = int((dollars.std(axis=1) > 0).sum())
        row = dict(slate=slate, contest=zip_stem, arm=arm, space="payout",
                   k=len(sel), k_match=k_match, n_field=c["n_field"],
                   mean_gross=round(float(dollars.mean()), 3),
                   p_cash=round(float((dollars > 0).mean()), 4))
        parts = []
        for sp, X in pne.return_spaces(dollars, fr[sel]).items():
            v = pne.n_eff(X)
            row[f"n_eff_{sp}"] = round(v, 3)
            row[f"frac_{sp}"] = round(v / max(live, 1), 4)
            parts.append(f"{sp}={v:6.2f}")
        rows.append(row)
        print(f"    payout {arm:<7} k={len(sel):>3} " + "  ".join(parts))
    return rows


def discover_pairs(slates: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for d in sorted((PROJECT_ROOT / "archive").iterdir()):
        if not d.is_dir() or (slates and d.name not in slates):
            continue
        if not (d / "portfolio_sweep_draftkings.json").exists():
            continue
        for z in sorted(d.glob("*.zip")):
            if z.name.startswith("contest-standings"):
                continue
            try:
                if len(rival_lineups(d, z.stem, ENTRANT)) >= 20:
                    pairs.append((d.name, z.stem))
            except Exception:
                continue
    return pairs


def main() -> None:
    slates = [a for a in sys.argv[1:] if not a.startswith("-")]
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    pairs = discover_pairs(slates)
    print(f"{len(pairs)} archived {ENTRANT} portfolios: "
          + ", ".join(f"{s}/{z}" for s, z in pairs))
    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        prev = pd.read_csv(RESULTS_CSV, dtype={"slate": str})
        done = set(zip(prev["slate"], prev["contest"]))
    for slate, zs in pairs:
        if (slate, zs) in done:
            print(f"{slate}/{zs}: already done, skipping")
            continue
        print(f"=== {slate}/{zs} ===")
        t0 = time.time()
        try:
            rows = run_pair(slate, zs, cfg)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        if rows:
            df = pd.DataFrame(rows)
            if RESULTS_CSV.exists():
                old = pd.read_csv(RESULTS_CSV, dtype={"slate": str})
                old = old[~((old["slate"] == slate) & (old["contest"] == zs))]
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
        print(f"  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
