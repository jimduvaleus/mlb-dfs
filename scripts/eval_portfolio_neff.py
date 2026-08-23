"""How many INDEPENDENT bets is a shipped portfolio actually making?

The effective-degrees-of-freedom number (a physicist's inverse participation
ratio; the same quantity as an effective sample size). For a portfolio of k
lineups whose returns have correlation matrix C:

    N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 = k^2 / ||C||_F^2

since trace(C) = k for a correlation matrix. k independent lineups give
N_eff = k; k identical ones give N_eff = 1. No eigendecomposition needed --
||C||_F^2 = sum_ij C_ij^2 is the whole computation.

WHY THIS AND NOT Dn. The production selector's `Dn` (gpp_portfolio.py) is an
INCREMENTAL, per-pick quantity: it scores the candidate under consideration
against the picks already made. It answers "is this next lineup redundant?".
It cannot answer "is the finished portfolio concentrated?", because it is
never evaluated on the assembled set. N_eff is that missing portfolio-level
scalar, and it is the natural unit for a risk dial: "risk 0" is plausibly
just a high N_eff target.

TWO SPACES, and the difference between them is the point.

  score space   correlation of lineup FPTS across sim worlds. Contest-blind.
                This is what the production diversity term operates on.
  payout space  correlation of realized DOLLARS across sim worlds, under a
                real contest's real payout table, with our own entries
                inserted into the field so they displace each other. This is
                the space ShaidyAdvice describes working in ("it's not
                diversifying your exposures, it is diversifying your
                returns" / "the entire return distribution correlated to
                every other lineup"), and the payout ladder is violently
                nonlinear, so the two numbers are not interchangeable.

Payout space is computed PER CONTEST, because payouts only compare within a
contest -- and because self-displacement ("you can't take first twice") is a
within-contest effect. Our k_c entries for contest c are ranked against the
simulated opponent field AND against each other.

BASELINES ARE MANDATORY. N_eff alone is uninterpretable: 20-of-89 is only
meaningful against what an unselected draw and a pure-EV greedy would score
on the same slate, same k, same contest. Three references per slate:

    random    uniform draw from the pool. NO selection pressure -- the
              "what does the pool itself give you for free" line.
    proj      top-k by projected score. The naive-greedy floor.
    prj_own   top-k by the leverage currency. A selected-but-not-diversity-
              aware arm, to separate "selection" from "diversity selection".

WHAT THIS IS NOT. A measurement, not a challenger. It scores portfolios that
were already shipped; it changes nothing and predicts nothing on its own. A
low N_eff is only bad if the diversity was supposed to be there -- which is
exactly the claim this exists to check.

Checkpoint / resume per CLAUDE.md: one row per (slate, space, arm, contest)
appended to outputs/portfolio_neff/results.csv; slates already on disk are
skipped unless PNE_FORCE=1.

Sims and opponent-field pools are read from outputs/winspace_validity/ when
present (identical construction, seed 42, n_sims 25000) so this re-simulates
nothing that branch already paid for.

Usage
-----
    source venv/bin/activate
    python scripts/eval_portfolio_neff.py 08152026 08162026 ...
    python scripts/eval_portfolio_neff.py --all

Env vars
--------
    PNE_FIELD    opponent field pool size (default 10000)
    PNE_NSIMS    sim worlds (default 25000)
    PNE_CHUNK    sim worlds per field-scoring chunk (default 1000)
    PNE_FORCE    "1" re-runs slates already in results.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import external_pool as ep  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from bt_core import load_real_contests  # noqa: E402


def load_real_contests_tolerant(adir: Path) -> list[dict]:
    """`load_real_contests` per zip instead of per slate.

    The shared loader deliberately SystemExit's on the first zip whose payout
    table is not registered -- correct for the backtest, where silently
    dropping a contest would drop real entries from a graded result. Here the
    unit of analysis is one contest, so an unregistered table should cost that
    contest and nothing else. Each zip is isolated in its own temp dir named
    for the slate (so `contest_id` still comes out as `slate:stem`) and run
    through the verified loader unchanged; failures are reported, never
    swallowed silently.
    """
    import shutil
    import tempfile
    out = []
    for z in sorted(adir.glob("*.zip")):
        if z.name.startswith("contest-standings"):
            continue
        with tempfile.TemporaryDirectory() as td:
            one = Path(td) / adir.name
            one.mkdir()
            shutil.copy2(z, one / z.name)
            try:
                out.extend(load_real_contests(one))
            except SystemExit as e:
                print(f"    skip {z.name}: {e}")
    return out

FIELD_N = int(os.environ.get("PNE_FIELD", "10000"))
N_SIMS = int(os.environ.get("PNE_NSIMS", "25000"))
CHUNK = int(os.environ.get("PNE_CHUNK", "1000"))
FORCE = os.environ.get("PNE_FORCE") == "1"
SEED = 42

OUT_DIR = PROJECT_ROOT / "outputs" / "portfolio_neff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
WS_DIR = PROJECT_ROOT / "outputs" / "winspace_validity"   # shared sim/field caches


def _append_and_reload(csv_path: Path, slate: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype={"slate": str})
        old = old[old["slate"] != slate]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path, dtype={"slate": str})


def _done_slates(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path, dtype={"slate": str})["slate"].unique())


def n_eff(X: np.ndarray) -> float:
    """N_eff = k^2 / ||C||_F^2 over the rows of X (k, S).

    Rows with zero variance carry no information and would make C undefined;
    they are dropped rather than being handed a 0/0 correlation. That is the
    normal case in payout space, where a lineup can miss the money in every
    simulated world and its dollar series is identically zero -- such a
    lineup is not an independent bet, it is not a bet at all.
    """
    A = np.asarray(X, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    sd = np.sqrt((A * A).sum(axis=1))
    live = sd > 0
    if live.sum() < 2:
        return float(live.sum())
    A = A[live] / sd[live, None]
    C = A @ A.T
    k = A.shape[0]
    return float(k * k / (C * C).sum())


def lineup_cols(player_id_lists, col_map) -> np.ndarray:
    return np.array([[col_map[int(p)] for p in ids] for ids in player_id_lists],
                    dtype=np.int32)


def score_lineups(mat: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(k, S) simulated scores, batched over lineups (not worlds) so the
    (S, k, 10) gather intermediate never materializes -- CLAUDE.md's
    memory-conscious rule; at k=200/S=25000 the unbatched form is 2GB."""
    S = mat.shape[0]
    out = np.empty((cols.shape[0], S), dtype=np.float32)
    for i0 in range(0, cols.shape[0], 64):
        i1 = min(i0 + 64, cols.shape[0])
        out[i0:i1] = mat[:, cols[i0:i1]].sum(axis=2).T
    return out


def field_ranks(our_scores: np.ndarray, mat: np.ndarray, fcols: np.ndarray) -> np.ndarray:
    """(k, S) count of opponent-field entries strictly above each of our
    lineups, as a FRACTION of the field pool. Chunked over sim worlds: the
    full (S, F) field-score array is 1GB at S=25k/F=10k and is only ever
    needed one chunk at a time (CLAUDE.md)."""
    k, S = our_scores.shape
    F = fcols.shape[0]
    frac = np.empty((k, S), dtype=np.float32)
    for s0 in range(0, S, CHUNK):
        s1 = min(s0 + CHUNK, S)
        fs = ep._score_field_cols_batched(mat[s0:s1], fcols)     # (c, F)
        fs.sort(axis=1)
        for j in range(s1 - s0):
            n_below = np.searchsorted(fs[j], our_scores[:, s0 + j], side="left")
            frac[:, s0 + j] = (F - n_below) / F
        del fs
    return frac


def payout_series(frac_above: np.ndarray, our_scores: np.ndarray,
                  n_field: int, payout_arr: np.ndarray) -> np.ndarray:
    """(k, S) gross dollars per simulated world for one contest's k entries.

    Rank = (opponent entries above, scaled from the field pool to the
    contest's REAL field size) + (our OWN entries above). The second term is
    self-displacement -- "you can't take first twice" -- which is the whole
    reason a portfolio's dollar correlation is not just its score
    correlation, and it is a within-contest effect only.

    The own-above count is chunked over worlds: the natural (k, k, S)
    pairwise-comparison array is 1GB of bool at k=200/S=25000, and it is
    rebuilt for every arm and every contest (CLAUDE.md's repeated-loop case).
    """
    k, S = our_scores.shape
    L = len(payout_arr)
    out = np.zeros((k, S), dtype=np.float64)
    for s0 in range(0, S, CHUNK):
        s1 = min(s0 + CHUNK, S)
        sub = our_scores[:, s0:s1]
        own_above = (sub[:, None, :] < sub[None, :, :]).sum(axis=1)
        rank0 = np.rint(frac_above[:, s0:s1] * n_field).astype(np.int64) + own_above
        paid = rank0 < L
        blk = out[:, s0:s1]
        blk[paid] = payout_arr[np.clip(rank0[paid], 0, L - 1)]
    return out


def return_spaces(dollars: np.ndarray, frac_above: np.ndarray) -> dict:
    """The functionals of "return" an N_eff can be measured on.

    They are NOT interchangeable, and the spread between them is the main
    result this script produces. Pearson correlation weights a series by where
    its variance lives, and in a GPP the dollar variance lives almost entirely
    in the jackpot tail -- ranks two correlated lineups essentially never
    occupy together. So raw dollars reports even a portfolio of near-clones as
    nearly independent. That is a true statement about dollar VARIANCE and a
    useless one about redundancy.

      dollars    raw gross $. The literal mean-variance quantity, and the one
                 ShaidyAdvice's "entire return distribution" describes.
      log1p$     tail-compressed dollars: still payout-shaped, but no longer
                 dominated by the handful of first-place worlds.
      cash       paid/not-paid indicator. The FLOOR. This is the space the
                 stated goal lives in -- "reduce the risk that none of your
                 lineups cash at all in the same day".
      pctile     finishing percentile vs the field. Payout-blind but
                 contest-aware; the best-conditioned of the four.
    """
    return {
        "dollars": dollars,
        "log1p$": np.log1p(dollars),
        "cash": (dollars > 0).astype(np.float64),
        "pctile": (1.0 - frac_above).astype(np.float64),
    }


def load_sims(slate: str, players_df, proj_ext, cfg):
    """Reuse the winspace_validity cache when it exists (same seed, same
    n_sims, same grid construction); otherwise build and cache our own."""
    shared = WS_DIR / f"sim_{slate}_{N_SIMS}_{SEED}.npz"
    local = OUT_DIR / f"sim_{slate}_{N_SIMS}_{SEED}.npz"
    for c in (shared, local):
        if c.exists():
            with np.load(c) as z:
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
    np.savez_compressed(local, player_ids=np.asarray(sr.player_ids, dtype=np.int64),
                        results_matrix=sr.results_matrix.astype(np.float32))
    return sr.player_ids, sr.results_matrix.astype(np.float32)


def load_field(slate: str, players_df, own) -> np.ndarray:
    shared = WS_DIR / f"field_{slate}_{FIELD_N}_{SEED}.npy"
    local = OUT_DIR / f"field_{slate}_{FIELD_N}_{SEED}.npy"
    for c in (shared, local):
        if c.exists():
            return np.load(c)
    fpool = ep.build_topn_field_pool(players_df, own, FIELD_N, SEED)
    np.save(local, fpool)
    return fpool


def run_slate(slate: str, cfg: dict) -> list[dict]:
    adir = PROJECT_ROOT / "archive" / slate
    found = ep.discover_external_files(str(adir))
    slate_df = DraftKingsSlateIngestor(str(adir / "DKSalaries.csv")).get_slate_dataframe()
    pool = ep.parse_lineup_pool(found["lineups_paths"],
                                set(slate_df["player_id"].astype(int)),
                                require_roi_blocks=False)
    proj_ext = ep.parse_player_projections(found["projections_path"])
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, {int(p) for lu in pool.lineups for p in lu.player_ids},
        PipelineRunner._derive_opponent)

    pid, mat = load_sims(slate, players_df, proj_ext, cfg)
    col_map = {int(p): i for i, p in enumerate(pid)}
    S = mat.shape[0]

    # --- shipped portfolio, straight from the archived sweep -----------------
    sw = json.loads((adir / "portfolio_sweep_draftkings.json").read_text())
    risk = sw.get("active_risk")
    ent = [e for e in sw.get("sweep", []) if e.get("risk") == risk] or sw.get("sweep", [])
    ship_raw = ent[0]["lineups"] if ent else []
    ship, ship_contest, n_unmapped = [], [], 0
    for lu in ship_raw:
        ids = [int(p["player_id"]) for p in lu["players"]]
        if any(i not in col_map for i in ids):
            n_unmapped += 1
            continue
        ship.append(ids)
        ship_contest.append(lu["contest_name"])
    if len(ship) < 10:
        raise RuntimeError(f"{slate}: only {len(ship)} shipped lineups mapped to sim columns")
    k = len(ship)
    print(f"  shipped k={k} at risk={risk} ({n_unmapped} unmapped), pool={len(pool.lineups)}")

    ship_scores = score_lineups(mat, lineup_cols(ship, col_map))

    # --- baseline arms, same k ----------------------------------------------
    all_cols = lineup_cols([lu.player_ids for lu in pool.lineups], col_map)
    proj = ep.compute_pool_proj_scores(pool.lineups, players_df)
    ownv = ep.compute_pool_ownership(pool.lineups, players_df)
    prj_own = ep.compute_prj_own_ev(proj, ownv, float(FIELD_N))
    rng = np.random.default_rng(int(slate))
    arms_idx = {
        "random": rng.choice(len(pool.lineups), size=min(k, len(pool.lineups)), replace=False),
        "proj": np.argsort(-np.nan_to_num(proj, nan=-np.inf))[:k],
        "prj_own": np.argsort(-np.nan_to_num(prj_own, nan=-np.inf))[:k],
    }
    arm_scores = {"shipped": ship_scores}
    for name, idx in arms_idx.items():
        arm_scores[name] = score_lineups(mat, all_cols[idx])

    rows = []
    for arm, sc in arm_scores.items():
        v = n_eff(sc)
        rows.append(dict(slate=slate, space="score", arm=arm, contest="ALL",
                         k=sc.shape[0], k_live=sc.shape[0], n_field=np.nan,
                         n_eff_score=round(v, 3),
                         frac_score=round(v / sc.shape[0], 4)))
        print(f"    score  {arm:<8} k={sc.shape[0]:>4}  N_eff={v:7.2f}  ({v/sc.shape[0]:.1%})")

    # --- payout space, per real contest -------------------------------------
    real = load_real_contests_tolerant(adir)
    if not real:
        print("  no gradeable real contests -- score space only")
        return rows
    by_display: dict[str, list[dict]] = {}
    for c in real:
        by_display.setdefault(c["contest"], []).append(c)

    own = players_df["ownership"].astype(float).to_numpy()
    fpool = load_field(slate, players_df, own)
    fcols = np.array([[col_map[int(p)] for p in r] for r in fpool], dtype=np.int32)

    # Rank-against-the-field does NOT depend on the contest (only the payout
    # lookup does), so it is paid once per arm rather than once per arm per
    # contest -- the single dominant cost in here, ~30s a call.
    t0 = time.time()
    arm_frac = {arm: field_ranks(sc, mat, fcols) for arm, sc in arm_scores.items()}
    print(f"    [field ranks, {len(arm_scores)} arms: {time.time() - t0:.0f}s]")

    ship_contest = np.array(ship_contest)
    for display, variants in by_display.items():
        sel = np.flatnonzero(ship_contest == display)
        if len(sel) < 2:
            continue
        c = max(variants, key=lambda v: v["n_field"])   # largest variant of a shared name
        kc = len(sel)
        for arm, sc in arm_scores.items():
            pick = sel if arm == "shipped" else np.arange(kc)
            sub, frac = sc[pick], arm_frac[arm][pick]
            dollars = payout_series(frac, sub, c["n_field"], c["payout_arr"])
            live = int((dollars.std(axis=1) > 0).sum())
            row = dict(slate=slate, space="payout", arm=arm, contest=display,
                       k=kc, k_live=live, n_field=c["n_field"],
                       mean_gross=round(float(dollars.mean()), 3),
                       p_cash=round(float((dollars > 0).mean()), 4))
            parts = []
            for sp_name, X in return_spaces(dollars, frac).items():
                v = n_eff(X)
                row[f"n_eff_{sp_name}"] = round(v, 3)
                row[f"frac_{sp_name}"] = round(v / max(live, 1), 4)
                parts.append(f"{sp_name}={v:6.2f}")
            rows.append(row)
            print(f"    payout {arm:<8} {display:<14} k={kc:>3} " + "  ".join(parts))
    return rows


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--all" in sys.argv:
        args = sorted(p.name for p in (PROJECT_ROOT / "archive").iterdir()
                      if p.is_dir() and (p / "portfolio_sweep_draftkings.json").exists()
                      and (WS_DIR / f"sim_{p.name}_{N_SIMS}_{SEED}.npz").exists())
    if not args:
        raise SystemExit(__doc__)
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    done = set() if FORCE else _done_slates(RESULTS_CSV)
    for slate in args:
        if slate in done:
            print(f"{slate}: already done, skipping")
            continue
        print(f"=== {slate} ===")
        t0 = time.time()
        try:
            rows = run_slate(slate, cfg)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        _append_and_reload(RESULTS_CSV, slate, rows)
        print(f"  [{time.time() - t0:.0f}s, {len(rows)} rows]")


if __name__ == "__main__":
    main()
