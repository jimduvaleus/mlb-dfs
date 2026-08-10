"""Grid-probe LeveragePortfolioSelector's two uncalibrated constants
(target_anchor_c, coverage_weight) against the 10-slate archive, via
tests/backtest_lab.py's cached oracle tables (no fresh sim/pool work needed
-- select_leverage/grade_joint run entirely off tests/backtest_output/oracle/
{slate}_leverage.npz, built by tests/backtest_oracle.py::build_leverage).

Why this needed its own harness
--------------------------------
The Phase D real-data check (tests/backtest_lab.py leverage_arms, memory
project-leverage-anchor-calibration-todo) found the RAW leverage currency
ranks #1 of ~20 currencies on top1%/top01% decile lift (LOSO-robust), but
the full LeveragePortfolioSelector (band-widening subsetting + regret-
minimization) loses to plain top-K ranking on the same currency -- most
likely because target_anchor_c=1.0 (default) was shown, in an earlier
smoke test, to admit the ENTIRE pool at even mini-MAX's 17,835-entry field
size (no gatekeeping at all), and coverage_weight=0.5 was never tuned
either. This sweeps both against real per-contest field sizes and real
payout outcomes -- the same category of calibration p_win's `sharpness` and
select_needlunchmoney_pool.py's PITCHER_COVERAGE_ALPHA both needed before
being trusted.

Usage
-----
    source venv/bin/activate
    python scripts/sweep_leverage_selector.py
    python scripts/sweep_leverage_selector.py --anchors 1,10,100 --weights 0,0.5,1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.backtest_lab import (  # noqa: E402
    ORACLE_DIR, grade_joint, load_leverage, load_slate, print_report,
    report, select_greedy, select_leverage,
)

SLATES = [
    "07222026", "07242026", "07252026", "07262026", "07282026",
    "07292026", "07302026", "07312026", "08012026", "08032026",
]
DEFAULT_ANCHORS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def admissible_fraction_report(anchors: tuple, seed: int = 42, calib: bool = False) -> None:
    """Mean (admissible subset size / pool size) per anchor, averaged over
    every (slate, contest) -- the diagnostic the calibration memo asked
    for: is a given target_anchor_c actually gatekeeping anything, or (like
    the default 1.0) admitting the whole pool regardless of field size?"""
    from src.optimization.gpp_portfolio import LeveragePortfolioSelector

    print("\n===== admissible-subset fraction by target_anchor_c "
          f"(mean over {len(SLATES)} slates x contests) =====")
    for c in anchors:
        fracs = []
        for s in SLATES:
            if not (ORACLE_DIR / f"{s}_leverage.npz").exists():
                continue
            lev = load_leverage(s)
            M = lev["p_opt"].shape[1]
            sd = load_slate(s, seed, calib)
            for ci in range(len(lev["contest_id"])):
                sel = LeveragePortfolioSelector(
                    candidates=list(range(M)), portfolio_size=10,
                    p_opt=lev["p_opt"][ci], optimal_ownership=lev["optimal_ownership"][ci],
                    leverage_diff=lev["leverage_diff"][ci], leverage_ratio=lev["leverage_ratio"][ci],
                    player_indicator=np.zeros((1, M)),  # unused by _admissible_subset
                    field_size=float(sd.n_field[ci]), target_anchor_c=c,
                )
                fracs.append(len(sel._admissible_subset()) / M)
        print(f"  target_anchor_c={c:>10g}   mean admitted fraction = {np.mean(fracs):.4f}")


def run_sweep(anchors: tuple, weights: tuple, seed: int = 42, calib: bool = False) -> pd.DataFrame:
    slates = [s for s in SLATES if (ORACLE_DIR / f"{s}_leverage.npz").exists()]
    frames = []
    for s in slates:
        sd = load_slate(s, seed, calib)
        curs_leverage = load_leverage(s)["leverage_ratio_mean"]
        picks = select_greedy(sd, curs_leverage, curs_leverage, floor_pct=30.0, admit_n=2000)
        frames.append(grade_joint(sd, picks, "leverage_rank_only"))
        for c in anchors:
            for w in weights:
                name = f"anchor{c:g}_cov{w:g}"
                picks = select_leverage(sd, target_anchor_c=c, coverage_weight=w)
                frames.append(grade_joint(sd, picks, name))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", default=",".join(str(a) for a in DEFAULT_ANCHORS))
    p.add_argument("--weights", default=",".join(str(w) for w in DEFAULT_WEIGHTS))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    anchors = tuple(float(x) for x in args.anchors.split(","))
    weights = tuple(float(x) for x in args.weights.split(","))

    admissible_fraction_report(anchors, args.seed)

    print(f"\nRunning {len(anchors)}x{len(weights)}={len(anchors) * len(weights)} "
          f"combos x {len(SLATES)} slates ...")
    df = run_sweep(anchors, weights, args.seed)
    out = PROJECT_ROOT / "outputs" / "leverage_selector_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    res = report(df, baseline="leverage_rank_only")
    top_by_top1 = res.sort_values("top1%", ascending=False).head(15)
    print_report(
        pd.concat([top_by_top1,
                  res[res.arm == "leverage_rank_only"]]).drop_duplicates("arm"),
        f"TOP 15 COMBOS BY top1% + leverage_rank_only baseline ({len(SLATES)} slates)",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
