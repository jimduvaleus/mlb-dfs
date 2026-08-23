"""Does ShaidyAdvice's PORTFOLIO look like his stated THEORY, against real peers?

The theory reconstructed from his own descriptions (archive/shaidy*.txt) is
specific and therefore falsifiable:

  1. the optimisation happens in RETURN space, not roster space -- "my stuff
     doesn't know your lineup... I just know how much money each of your
     lineups makes", "it's not diversifying your exposures, it is
     diversifying your returns"
  2. so his portfolio may be CHALKY in exposures while still spread in
     returns -- "diversifying your returns can sometimes still have you very
     chalky on the very chalky plays because they're very good"
  3. no self-cannibalisation -- "you can't take first twice"
  4. lower variance is preferred at equal EV -- "given two lineups with the
     same $EV, a portfolio process prefers the more owned ones"

Every one of those predicts a measurable signature. The problem is the
control: N_eff alone says nothing without something to compare against, and
synthetic arms drawn from our own pool answer "different from us", which is
not the question. The right control is sitting in the standings zips -- 32 to
91 OTHER entrants at 100+ entries in the same contest, i.e. dozens of rival
professional processes graded on the same field, same sims, same payout
table. If his stated theory is what he actually runs, he should be an
OUTLIER among them in return-space spread specifically, and NOT in roster
spread. If he is unremarkable on both, the theory does not describe the
output.

Metrics per entrant, all portfolio-level:

  n_eff_{dollars,log1p$,cash,pctile}   return-space spread (prediction 1)
  n_eff_score                          FPTS-space spread
  overlap                              mean pairwise player overlap 0-10:
                                       ROSTER space (prediction 2)
  max_expo / own_sum                   chalkiness (prediction 2)
  self_dupes                           identical lineups within the
                                       portfolio (prediction 3)

STREAMING DESIGN. A dense (n_pros * 150, n_sims) score array is ~1.4GB and the
field-rank array another 1.4GB (CLAUDE.md). But N_eff needs only WITHIN-
portfolio cross-products, so the pass accumulates per-entrant (k, k) sums of
x and x x^T per world chunk and never materialises a full per-world array.
Cost is dominated by scoring the opponent field once per chunk, which is
shared by every entrant -- so grading 90 portfolios costs barely more than
grading one.

Checkpoint / resume per CLAUDE.md: one row per (slate, contest, entrant)
appended to outputs/pro_neff/results.csv.

Usage
-----
    source venv/bin/activate
    python scripts/eval_pro_neff.py                 # every archived contest
    python scripts/eval_pro_neff.py 07222026

Env vars
--------
    PRO_MIN      minimum entries to count as a max-entry portfolio (default 100)
    PRO_NSIMS / PRO_FIELD / PRO_CHUNK / PRO_FORCE
"""
import csv
import importlib.util
import io
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
from bt_core import ZIP_TO_CONTEST  # noqa: E402

_s = importlib.util.spec_from_file_location(
    "pne", PROJECT_ROOT / "scripts" / "eval_portfolio_neff.py")
pne = importlib.util.module_from_spec(_s); _s.loader.exec_module(pne)
_s2 = importlib.util.spec_from_file_location(
    "arp", PROJECT_ROOT / "scripts" / "analyze_rival_portfolio.py")
arp = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(arp)

MIN_ENTRIES = int(os.environ.get("PRO_MIN", "100"))
N_SIMS = int(os.environ.get("PRO_NSIMS", "25000"))
FIELD_N = int(os.environ.get("PRO_FIELD", "10000"))
CHUNK = int(os.environ.get("PRO_CHUNK", "500"))
FORCE = os.environ.get("PRO_FORCE") == "1"
SEED = 42
SPACES = ["dollars", "log1p$", "cash", "pctile"]

OUT_DIR = PROJECT_ROOT / "outputs" / "pro_neff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"


def read_standings(adir: Path, stem: str):
    with zipfile.ZipFile(adir / f"{stem}.zip") as zf:
        n = [x for x in zf.namelist() if x.endswith(".csv")][0]
        rows = list(csv.reader(io.StringIO(
            zf.read(n).decode("utf-8-sig", errors="replace"))))
    return arp.parse_standings_rows(rows)


def name_index(adir: Path):
    sal = pd.read_csv(adir / "DKSalaries.csv")
    n2i, dup = {}, set()
    for nm, pid in zip(sal["Name"].astype(str).str.strip(), sal["ID"].astype(int)):
        if nm in n2i and n2i[nm] != pid:
            dup.add(nm)
        n2i[nm] = pid
    return n2i, dup


class Accum:
    """Streaming (k,k) cross-product accumulators -- one per return space."""

    def __init__(self, k: int):
        self.n = 0
        self.s1 = {sp: np.zeros(k) for sp in SPACES}
        self.s2 = {sp: np.zeros((k, k)) for sp in SPACES}

    def add(self, blocks: dict, c: int):
        self.n += c
        for sp, X in blocks.items():
            Xd = X.astype(np.float64)
            self.s1[sp] += Xd.sum(axis=1)
            self.s2[sp] += Xd @ Xd.T

    def n_eff(self, sp: str) -> tuple[float, int]:
        n = self.n
        m = self.s1[sp] / n
        cov = self.s2[sp] / n - np.outer(m, m)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        live = sd > 1e-12
        if live.sum() < 2:
            return float(live.sum()), int(live.sum())
        cov = cov[np.ix_(live, live)]
        sd = sd[live]
        C = cov / np.outer(sd, sd)
        k = C.shape[0]
        return float(k * k / (C * C).sum()), k


def roster_stats(cols: np.ndarray) -> dict:
    """Roster-space (as opposed to return-space) structure."""
    k = cols.shape[0]
    ind = np.zeros((k, cols.max() + 1), dtype=np.float32)
    for i, r in enumerate(cols):
        ind[i, r] = 1.0
    ov = ind @ ind.T
    iu = np.triu_indices(k, 1)
    expo = ind.mean(axis=0)
    return dict(overlap=float(ov[iu].mean()),
                overlap_p90=float(np.percentile(ov[iu], 90)),
                max_expo=float(expo.max()),
                n_players_used=int((expo > 0).sum()))


def run_contest(slate: str, stem: str, cfg: dict) -> list[dict]:
    adir = PROJECT_ROOT / "archive" / slate
    display = ZIP_TO_CONTEST.get(stem)
    real = [c for c in pne.load_real_contests_tolerant(adir) if c["contest"] == display]
    if not real:
        print(f"  {stem}: no payout table for {display} -- skipped")
        return []
    contest = max(real, key=lambda v: v["n_field"])

    e, _ = read_standings(adir, stem)
    counts = e.handle.value_counts()
    pros = [h for h in counts.index if counts[h] >= MIN_ENTRIES]
    if "ShaidyAdvice" not in pros:
        print(f"  {stem}: ShaidyAdvice not a max-entry entrant -- skipped")
        return []
    n2i, dup = name_index(adir)

    port, dropped = {}, {}
    for h in pros:
        ids = []
        raw = list(e[e.handle == h]["names"])
        kept = [t for t in raw if not any(nm in dup or nm not in n2i for nm in t)]
        for t in kept:
            ids.append([n2i[nm] for nm in t])
        if len(ids) >= MIN_ENTRIES * 0.8:
            port[h] = ids
            # self-dupes counts IDENTICAL lineups among the kept entries only.
            # Counting them as (raw - unique_kept) would fold ambiguous-name
            # drops into the duplicate total and inflate it several-fold.
            dropped[h] = (len(raw) - len(kept), len(kept) - len(set(kept)))

    found = ep.discover_external_files(str(adir))
    slate_df = DraftKingsSlateIngestor(str(adir / "DKSalaries.csv")).get_slate_dataframe()
    proj_ext = ep.parse_player_projections(found["projections_path"])
    union = {int(p) for v in port.values() for ids in v for p in ids}
    pool = ep.parse_lineup_pool(found["lineups_paths"],
                                set(slate_df["player_id"].astype(int)),
                                require_roi_blocks=False)
    union |= {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, union, PipelineRunner._derive_opponent)

    cache = OUT_DIR / f"sim_{slate}_{stem}_{N_SIMS}_{SEED}.npz"
    if cache.exists():
        with np.load(cache) as z:
            pid = [int(p) for p in z["player_ids"]]
            mat = z["results_matrix"].astype(np.float32)
    else:
        gpp, paths = cfg["gpp"], cfg["paths"]
        grids = ep.build_quantile_grids(
            proj_ext,
            zero_inflate=bool(gpp.get("external_pool_zero_inflate", False)),
            scratch_prob=float(gpp.get("external_pool_scratch_prob", 0.02)),
            mean_calib_batter=float(gpp.get("external_pool_mean_calib_batter", 1.0)),
            mean_calib_pitcher=float(gpp.get("external_pool_mean_calib_pitcher", 1.0)))
        engine = SimulationEngine(
            EmpiricalCopula(str(PROJECT_ROOT / paths["copula"])), players_df,
            batter_pca_model=None, score_grid=None, quantile_grids=grids)
        st = np.random.get_state(); np.random.seed(SEED)
        sr = engine.simulate(N_SIMS); np.random.set_state(st)
        pid, mat = sr.player_ids, sr.results_matrix.astype(np.float32)
        np.savez_compressed(cache, player_ids=np.asarray(pid, dtype=np.int64),
                            results_matrix=mat)
    col_map = {int(p): i for i, p in enumerate(pid)}

    usable = {h: v for h, v in port.items()
              if all(int(p) in col_map for ids in v for p in ids)}
    if "ShaidyAdvice" not in usable:
        print(f"  {stem}: ShaidyAdvice has unprojected players -- skipped")
        return []
    handles = sorted(usable)
    cols = {h: pne.lineup_cols(usable[h], col_map) for h in handles}
    print(f"  {stem}: {len(handles)} max-entry portfolios, "
          f"{sum(c.shape[0] for c in cols.values())} lineups, n_field={contest['n_field']:,}")

    own = players_df["ownership"].astype(float).to_numpy()
    fpool = pne.load_field(slate, players_df, own)
    fcols = np.array([[col_map[int(p)] for p in r] for r in fpool], dtype=np.int32)

    accs = {h: Accum(cols[h].shape[0]) for h in handles}
    sums = {h: np.zeros(cols[h].shape[0]) for h in handles}   # score-space mean
    sq = {h: np.zeros((cols[h].shape[0],) * 2) for h in handles}
    S, F = mat.shape[0], fcols.shape[0]
    pay, nf = contest["payout_arr"], contest["n_field"]
    L = len(pay)
    t0 = time.time()
    for s0 in range(0, S, CHUNK):
        s1 = min(s0 + CHUNK, S)
        sub = mat[s0:s1]
        fs = ep._score_field_cols_batched(sub, fcols)
        fs.sort(axis=1)
        c = s1 - s0
        for h in handles:
            sc = sub[:, cols[h]].sum(axis=2).T.astype(np.float32)      # (k, c)
            sums[h] += sc.sum(axis=1)
            sq[h] += sc.astype(np.float64) @ sc.astype(np.float64).T
            frac = np.empty_like(sc)
            for j in range(c):
                frac[:, j] = (F - np.searchsorted(fs[j], sc[:, j], side="left")) / F
            own_above = (sc[:, None, :] < sc[None, :, :]).sum(axis=1)
            rank0 = np.rint(frac * nf).astype(np.int64) + own_above
            paid = rank0 < L
            dollars = np.zeros(sc.shape, dtype=np.float64)
            dollars[paid] = pay[np.clip(rank0[paid], 0, L - 1)]
            accs[h].add({"dollars": dollars, "log1p$": np.log1p(dollars),
                         "cash": (dollars > 0).astype(np.float64),
                         "pctile": (1.0 - frac).astype(np.float64)}, c)
        del fs
    print(f"    [streamed {S} worlds in {time.time() - t0:.0f}s]")

    rows = []
    for h in handles:
        k = cols[h].shape[0]
        m = sums[h] / S
        cov = sq[h] / S - np.outer(m, m)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        live = sd > 1e-12
        Cs = (cov[np.ix_(live, live)] / np.outer(sd[live], sd[live]))
        ns = float(live.sum() ** 2 / (Cs * Cs).sum()) if live.sum() > 1 else 0.0
        r = dict(slate=slate, contest=stem, entrant=h, k=k,
                 n_field=nf, n_eff_score=round(ns, 3),
                 frac_score=round(ns / k, 4),
                 dropped_ambiguous=dropped[h][0], self_dupes=dropped[h][1])
        r.update({f"{kk}": round(vv, 4) for kk, vv in roster_stats(cols[h]).items()})
        for sp in SPACES:
            v, kl = accs[h].n_eff(sp)
            r[f"n_eff_{sp}"] = round(v, 3)
            r[f"frac_{sp}"] = round(v / max(kl, 1), 4)
        rows.append(r)
    return rows


def main() -> None:
    slates = [a for a in sys.argv[1:] if not a.startswith("-")]
    with open(PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    done = set()
    if RESULTS_CSV.exists() and not FORCE:
        p = pd.read_csv(RESULTS_CSV, dtype={"slate": str})
        done = set(zip(p["slate"], p["contest"]))
    targets = []
    for d in sorted((PROJECT_ROOT / "archive").iterdir()):
        if not d.is_dir() or (slates and d.name not in slates):
            continue
        for z in sorted(d.glob("*.zip")):
            if z.name.startswith("contest-standings"):
                continue
            try:
                e, _ = read_standings(d, z.stem)
            except Exception:
                continue
            c = e.handle.value_counts()
            if "ShaidyAdvice" in c.index and c["ShaidyAdvice"] >= MIN_ENTRIES:
                targets.append((d.name, z.stem))
    print(f"{len(targets)} contests with a ShaidyAdvice max-entry portfolio")
    for slate, stem in targets:
        if (slate, stem) in done:
            print(f"{slate}/{stem}: already done, skipping")
            continue
        print(f"=== {slate}/{stem} ===")
        t0 = time.time()
        try:
            rows = run_contest(slate, stem, cfg)
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}")
            continue
        if rows:
            df = pd.DataFrame(rows)
            if RESULTS_CSV.exists():
                old = pd.read_csv(RESULTS_CSV, dtype={"slate": str})
                old = old[~((old["slate"] == slate) & (old["contest"] == stem))]
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(RESULTS_CSV, index=False)
        print(f"  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
