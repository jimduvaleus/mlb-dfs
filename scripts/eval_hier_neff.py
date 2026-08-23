"""Two-level N_eff: is the ceiling edge BETWEEN teams or WITHIN them?

Portfolio-level N_eff (scripts/eval_portfolio_neff.py) is a single global
number, and it cannot distinguish two very differently-behaved designs:

  A. spread evenly over few teams, one build per team
  B. spread evenly over MANY teams, several near-variant builds per team

Both can post identical global N_eff, identical mean pairwise overlap and
identical exposure concentration, while behaving completely differently
conditional on a team actually booming. The reverse-engineering work
(project-rival-portfolio-shaidyadvice) measured exactly that split by
OVERLAP -- ShaidyAdvice at within-team 3.88 / between-team 0.93 versus ours
at 3.02 / 0.55 -- i.e. we are more spread on BOTH axes. This does the same
decomposition in the space that matters for the payout, and ties it to the
outcome the whole thing is about.

THE HYPOTHESIS BEING TESTED. If the boom team is unpredictable (established:
no team-quality signal predicts stack picks, no per-lineup value signal
predicts hit99), then P(some entry lands top-1%) factors roughly as

    P(hold the boom team) x P(hold a GOOD build of it | holding it)

The first term wants BETWEEN-team spread; the second wants WITHIN-team
CLUSTERING on the best builds, because when a team goes off you do not know
which 5 of its 9 hitters carried it. Those pull in opposite directions, which
is why a single global diversity number cannot express the design. Prediction:
across the peer cohort, top-1% rate should be associated with LOW within-block
N_eff (tight clusters) at comparable between-block N_eff.

DECOMPOSITION. Both halves come from the one (k,k) covariance the streaming
pass already builds, given block labels b(i) = the lineup's primary HITTER
stack team:

  within    per block, N_eff on that block's own correlation submatrix;
            reported entry-weighted as a fraction of block size.
  between   covariance of the block-MEAN series, which is just the block-
            averaged sub-blocks of the same covariance matrix
            (Cov[mean_b, mean_b'] = mean of Cov_ij over i in b, j in b'),
            then N_eff over the B blocks. No second pass needed.

Also emitted per portfolio: `top1_rate` (mean over entries of P(finishing in
the field's top 1%)) -- the outcome variable -- plus the overlap
decomposition, so the numbers can be checked against the recorded 3.88/0.93.

Reuses the sims cached by scripts/eval_pro_neff.py; simulates nothing.

Usage
-----
    source venv/bin/activate
    python scripts/eval_hier_neff.py
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

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from src.api import external_pool as ep  # noqa: E402

_s = importlib.util.spec_from_file_location(
    "pne", PROJECT_ROOT / "scripts" / "eval_portfolio_neff.py")
pne = importlib.util.module_from_spec(_s); _s.loader.exec_module(pne)
_s2 = importlib.util.spec_from_file_location(
    "arp", PROJECT_ROOT / "scripts" / "analyze_rival_portfolio.py")
arp = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(arp)

MIN_ENTRIES = int(os.environ.get("HIER_MIN", "100"))
CHUNK = int(os.environ.get("HIER_CHUNK", "500"))
N_SIMS, FIELD_N, SEED = 25000, 10000, 42
PRO_DIR = PROJECT_ROOT / "outputs" / "pro_neff"
OUT_DIR = PROJECT_ROOT / "outputs" / "hier_neff"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = OUT_DIR / "results.csv"


def n_eff_from_corr(C: np.ndarray) -> float:
    k = C.shape[0]
    return float(k * k / (C * C).sum()) if k else 0.0


def cov_to_corr(cov: np.ndarray):
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    live = sd > 1e-12
    if live.sum() < 2:
        return None, live
    c = cov[np.ix_(live, live)] / np.outer(sd[live], sd[live])
    return c, live


def decompose(cov: np.ndarray, labels: np.ndarray) -> dict:
    """within / between N_eff from one covariance matrix + block labels."""
    out = {}
    C, live = cov_to_corr(cov)
    out["global"] = n_eff_from_corr(C) if C is not None else 0.0
    out["global_frac"] = out["global"] / max(int(live.sum()), 1)

    blocks = [np.flatnonzero(labels == b) for b in pd.unique(labels)]
    blocks = [b for b in blocks if len(b) >= 2]
    num = den = 0.0
    for b in blocks:
        Cb, lb = cov_to_corr(cov[np.ix_(b, b)])
        if Cb is None:
            continue
        v = n_eff_from_corr(Cb)
        num += v            # entry-weighted: sum of N_eff over sum of k
        den += int(lb.sum())
    out["within_frac"] = num / den if den else np.nan
    out["n_blocks_ge2"] = len(blocks)

    # between: covariance of block means, straight from the same matrix
    ub = pd.unique(labels)
    idx = [np.flatnonzero(labels == b) for b in ub]
    B = len(ub)
    M = np.empty((B, B))
    for i in range(B):
        for j in range(B):
            M[i, j] = cov[np.ix_(idx[i], idx[j])].mean()
    Cb, lb = cov_to_corr(M)
    out["between"] = n_eff_from_corr(Cb) if Cb is not None else 0.0
    out["between_frac"] = out["between"] / max(int(lb.sum()), 1)
    out["n_teams"] = B
    return out


def overlap_split(cols: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    k = cols.shape[0]
    ind = np.zeros((k, int(cols.max()) + 1), dtype=np.float32)
    for i, r in enumerate(cols):
        ind[i, r] = 1.0
    ov = ind @ ind.T
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(k, 1)
    m = same[iu]
    o = ov[iu]
    return (float(o[m].mean()) if m.any() else np.nan,
            float(o[~m].mean()) if (~m).any() else np.nan)


def run_contest(slate: str, stem: str) -> list[dict]:
    adir = PROJECT_ROOT / "archive" / slate
    cache = PRO_DIR / f"sim_{slate}_{stem}_{N_SIMS}_{SEED}.npz"
    if not cache.exists():
        print(f"  no cached sim for {slate}/{stem} -- run eval_pro_neff first")
        return []
    with np.load(cache) as z:
        pid = [int(p) for p in z["player_ids"]]
        mat = z["results_matrix"].astype(np.float32)
    col_map = {int(p): i for i, p in enumerate(pid)}

    with zipfile.ZipFile(adir / f"{stem}.zip") as zf:
        n = [x for x in zf.namelist() if x.endswith(".csv")][0]
        rows = list(csv.reader(io.StringIO(
            zf.read(n).decode("utf-8-sig", errors="replace"))))
    e, _ = arp.parse_standings_rows(rows)
    counts = e.handle.value_counts()
    pros = [h for h in counts.index if counts[h] >= MIN_ENTRIES]
    if "ShaidyAdvice" not in pros:
        return []

    sal = pd.read_csv(adir / "DKSalaries.csv")
    n2i, dup = {}, set()
    for nm, p in zip(sal["Name"].astype(str).str.strip(), sal["ID"].astype(int)):
        if nm in n2i and n2i[nm] != p:
            dup.add(nm)
        n2i[nm] = p
    tmap = arp.team_map(adir)
    pitchers = arp.pitcher_names(adir)

    port = {}
    for h in pros:
        raw = list(e[e.handle == h]["names"])
        kept = [t for t in raw
                if not any(nm in dup or nm not in n2i for nm in t)
                and all(n2i[nm] in col_map for nm in t)]
        if len(kept) < MIN_ENTRIES * 0.8:
            continue
        if len(kept) - len(set(kept)) > 0.5 * len(kept):
            continue                      # degenerate: near-total duplication
        port[h] = kept
    if "ShaidyAdvice" not in port:
        return []

    handles = sorted(port)
    cols = {h: np.array([[col_map[n2i[nm]] for nm in t] for t in port[h]],
                        dtype=np.int32) for h in handles}
    labels = {h: np.array(arp.primary_teams(port[h], tmap, pitchers))
              for h in handles}
    print(f"  {stem}: {len(handles)} portfolios")

    own = None
    fpool = np.load(PROJECT_ROOT / "outputs" / "winspace_validity" /
                    f"field_{slate}_{FIELD_N}_{SEED}.npy") \
        if (PROJECT_ROOT / "outputs" / "winspace_validity" /
            f"field_{slate}_{FIELD_N}_{SEED}.npy").exists() else None
    if fpool is None:
        fpool = np.load(PROJECT_ROOT / "outputs" / "portfolio_neff" /
                        f"field_{slate}_{FIELD_N}_{SEED}.npy")
    fcols = np.array([[col_map[int(p)] for p in r] for r in fpool], dtype=np.int32)

    S, F = mat.shape[0], fcols.shape[0]
    s1a = {h: np.zeros(cols[h].shape[0]) for h in handles}
    s2a = {h: np.zeros((cols[h].shape[0],) * 2) for h in handles}
    s1s = {h: np.zeros(cols[h].shape[0]) for h in handles}
    s2s = {h: np.zeros((cols[h].shape[0],) * 2) for h in handles}
    top1 = {h: np.zeros(cols[h].shape[0]) for h in handles}
    # SUM vs MAX. `top1` is a per-ENTRY rate and is therefore maximised by
    # CONCENTRATION -- pile every entry onto the single highest-ceiling team
    # and it goes up (a 1-team portfolio in the 07/22 pilot scored 0.0185 that
    # way). The design question here is the opposite one: does the portfolio
    # hold AT LEAST ONE top-1% entry, which is what coverage buys and what
    # actually cashes a GPP. Both are recorded; only the max statistic tests
    # the hypothesis.
    anyt1 = {h: 0.0 for h in handles}
    anyt01 = {h: 0.0 for h in handles}
    t0 = time.time()
    for a in range(0, S, CHUNK):
        b = min(a + CHUNK, S)
        sub = mat[a:b]
        fs = ep._score_field_cols_batched(sub, fcols)
        fs.sort(axis=1)
        c = b - a
        for h in handles:
            sc = sub[:, cols[h]].sum(axis=2).T.astype(np.float32)
            q = np.empty_like(sc)
            for j in range(c):
                q[:, j] = np.searchsorted(fs[j], sc[:, j], side="left") / F
            s1s[h] += sc.sum(axis=1); s2s[h] += sc.astype(np.float64) @ sc.astype(np.float64).T
            qd = q.astype(np.float64)
            s1a[h] += qd.sum(axis=1); s2a[h] += qd @ qd.T
            top1[h] += (q >= 0.99).sum(axis=1)
            anyt1[h] += float((q >= 0.99).any(axis=0).sum())
            anyt01[h] += float((q >= 0.999).any(axis=0).sum())
        del fs
    print(f"    [{time.time() - t0:.0f}s]")

    out = []
    for h in handles:
        k = cols[h].shape[0]
        r = dict(slate=slate, contest=stem, entrant=h, k=k,
                 top1_rate=round(float(top1[h].mean() / S), 5),
                 p_any_top1=round(float(anyt1[h] / S), 5),
                 p_any_top01=round(float(anyt01[h] / S), 5))
        for tag, s1, s2 in (("pctile", s1a[h], s2a[h]), ("score", s1s[h], s2s[h])):
            m = s1 / S
            cov = s2 / S - np.outer(m, m)
            d = decompose(cov, labels[h])
            for kk, vv in d.items():
                r[f"{tag}_{kk}"] = round(float(vv), 4) if vv == vv else np.nan
        wi, be = overlap_split(cols[h], labels[h])
        r["overlap_within"] = round(wi, 3)
        r["overlap_between"] = round(be, 3)
        out.append(r)
    return out


def main() -> None:
    slates = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = []
    for f in sorted(PRO_DIR.glob("sim_*.npz")):
        parts = f.stem.split("_")
        slate, stem = parts[1], "_".join(parts[2:-2])
        if not slates or slate in slates:
            targets.append((slate, stem))
    print(f"{len(targets)} contests")
    allrows = []
    for slate, stem in targets:
        print(f"=== {slate}/{stem} ===")
        try:
            allrows += run_contest(slate, stem)
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}")
    if allrows:
        pd.DataFrame(allrows).to_csv(RESULTS, index=False)
        print(f"\nwrote {len(allrows)} rows -> {RESULTS}")


if __name__ == "__main__":
    main()
