"""How much does `gpp.per_contest_shortlist` cost, at real per-contest shapes?

Sizes the shortlist against measurement rather than a guess. It is a purely
COMPUTATIONAL guard -- a smaller value only removes options the arms would
otherwise have had -- so the question is never "which M is best" but "how large
can M be inside the memory and wall-clock budget", and that is what this
measures. Re-run it whenever n_sims, the arm set, or the contest mix moves.

Kernels are the production ones (`contest_payout_matrix`, `ContestDeltaState`,
`KellyPortfolioSelector`); only the scores and fields are synthetic, since cost
depends on shape and dtype rather than on the numbers being realistic.

Outer loop is the CONTEST, not M: the sorted field is (S x F) and at F=17,835 /
S=25,000 that is 1.78GB, so it gets built once and every M reads it. Which is
also the shape of the real code (`select_per_contest_multi_arm` puts contests
outside for the same reason).
"""
import csv
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.multi_contest import (                      # noqa: E402
    contest_beat_bits, contest_payout_matrix,
)
from src.optimization.mrp.delta_reward import ContestDeltaState   # noqa: E402
from src.optimization.gpp_portfolio import KellyPortfolioSelector  # noqa: E402

S = 25_000
MS = [4_000, 8_000, 12_000, 16_000, 20_000]
OUT = Path("outputs") / "bench_per_contest_shortlist.csv"

# (name, field_size, n_entries, fee, top_prize) -- the real 08/25 entries files.
CONTESTS = [
    ("Four-Seamer", 5_945, 20, 4.0, 2_000.0),
    ("mini-MAX", 17_835, 60, 1.0, 1_500.0),
    ("Chin Music", 2_378, 2, 5.0, 1_000.0),
    ("Base Hit", 980, 2, 12.0, 1_000.0),
    ("Solo Shot", 7_134, 24, 1.0, 600.0),
    ("Pickoff", 792, 2, 3.0, 200.0),
]


def rss_gb():
    """CURRENT RSS, not ru_maxrss.

    ru_maxrss is a high-water mark for the whole process, so one transient
    allocation early on pins every later reading to it -- which is what
    happened on the first attempt, where building the candidate matrix through
    a float64 intermediate parked the mark at 5.8GB and made every M identical.
    Current RSS, sampled while the arrays that scale with M are alive, is the
    number the sizing decision needs.
    """
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE") / 1024**3


def normal32(rng, shape, loc, scale, rows=2_000):
    """(shape) float32 normals WITHOUT a full float64 intermediate.

    `rng.normal(size=(20000, 25000))` materialises 4GB of float64 before the
    cast to float32 halves it. At these shapes that transient is larger than
    anything the benchmark is trying to measure.
    """
    out = np.empty(shape, dtype=np.float32)
    for i in range(0, shape[0], rows):
        j = min(i + rows, shape[0])
        out[i:j] = rng.normal(loc, scale, size=(j - i, shape[1]))
    return out


def ladder(field_size, top_prize, fee):
    """Top-heavy ~20%-paying ladder of roughly the right shape and total."""
    n_paid = max(int(field_size * 0.20), 2)
    r = np.arange(1, n_paid + 1, dtype=np.float64)
    arr = top_prize * r ** -1.35
    arr = np.maximum(arr, fee * 1.5)
    return arr


def done_rows():
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {(r["contest"], int(r["M"])) for r in csv.DictReader(f)}


def append(row):
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_rows()
    rng = np.random.default_rng(0)
    maxM = max(MS)
    print(f"allocating cand_scores ({maxM} x {S}) float32 = "
          f"{maxM * S * 4 / 1024**3:.2f} GB")
    cand = normal32(rng, (maxM, S), 140.0, 25.0)

    for name, F, k, fee, top in CONTESTS:
        if all((name, M) in done for M in MS):
            print(f"{name}: done, skipping")
            continue
        t0 = time.perf_counter()
        field = normal32(rng, (S, F), 140.0, 25.0)
        field.sort(axis=1)
        print(f"{name}: field ({S} x {F}) built+sorted in "
              f"{time.perf_counter() - t0:.1f}s  RSS={rss_gb():.2f} GB")
        pay_arr = ladder(F, top, fee)

        for M in MS:
            if (name, M) in done:
                continue
            sub = np.ascontiguousarray(cand[:M])
            t = time.perf_counter()
            payout = contest_payout_matrix(sub, [field], pay_arr, fee,
                                           cand_chunk=2_000)
            t_pay = time.perf_counter() - t

            # Kelly reads only the (M, S) payout matrix.
            t = time.perf_counter()
            KellyPortfolioSelector(
                payout, list(range(M)), portfolio_size=k,
                bankroll=fee * k * 2.0, ev_floor=float("-inf"),
                cash_anchor_fraction=0.25,
            ).select()
            t_kelly = time.perf_counter() - t

            # dR materialises four (M, S) rank/tie arrays from the sorted field.
            t = time.perf_counter()
            st = ContestDeltaState(sub, field, pay_arr.astype(np.float64),
                                   chunk=512)
            t_dr_state = time.perf_counter() - t
            t = time.perf_counter()
            taken = np.zeros(M, dtype=bool)
            for _ in range(k):
                g = st.marginal_gains()
                g[taken] = -np.inf
                j = int(np.argmax(g))
                taken[j] = True
                st.commit(j)
            t_dr_greedy = time.perf_counter() - t
            peak = rss_gb()   # every M-scaled array still alive
            del st, payout, sub

            row = {
                "contest": name, "F": F, "k": k, "M": M,
                "t_payout": round(t_pay, 2), "t_kelly": round(t_kelly, 2),
                "t_dr_state": round(t_dr_state, 2),
                "t_dr_greedy": round(t_dr_greedy, 2),
                "t_total": round(t_pay + t_kelly + t_dr_state + t_dr_greedy, 2),
                "peak_rss_gb": round(peak, 2),
            }
            append(row)
            print(f"  M={M:>6}: payout {t_pay:6.1f}s  kelly {t_kelly:5.1f}s  "
                  f"dR state {t_dr_state:6.1f}s  dR greedy {t_dr_greedy:6.1f}s "
                  f" peak {peak:.2f} GB")
        del field

    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
