"""Grade lineups against a real archived contest field, marginal or as a portfolio.

Two modes, and the difference between them is the whole point of this module.

MARGINAL (`grade_marginal`) scores each candidate as if it were the *only* extra
entry: rank it against the real field alone, apply the ladder, average over
worlds. This is what `prelock_quadrant_pool.py` has always done and it is the
right question for comparing per-lineup currencies (ceiling, ownership,
leverage) -- the +7.7% / +50.6% quadrant numbers came from it.

It is also structurally blind to self-competition. Two IDENTICAL lineups grade
identically under it, because neither can see the other. Any selector whose
entire job is to stop your own entries from crowding each other is therefore
unfalsifiable in marginal mode.

PORTFOLIO (`grade_portfolio`) inserts all K entries at once. The field becomes
`n_field + K`, and each entry is ranked against the real field PLUS its own
team-mates, so demotion -- your lineup pushing your other lineup down a rung --
is priced. That is the mode a 150-entry portfolio is actually played in, and the
only one in which a diversity/self-competition objective can be measured at all.

Both modes share `build_field` so the field construction and payout ladder can
never drift between them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_contest_sim_roi import parse_payout_table  # noqa: E402

_POS = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
_SPLIT = re.compile(r"\s*\b(" + "|".join(_POS) + r")\b\s+")

# Per-world value offset used to fold a per-world searchsorted into ONE global
# call. Must exceed any realistic lineup score by enough that world w's block
# cannot overlap world w+1's.
_WORLD_OFFSET = 1e6


def parse_names(lineup_raw: str) -> list[str]:
    return _SPLIT.split(lineup_raw)[1:][1::2]


def build_field(
    entries_csv: str,
    payout_table: str,
    pid_index: dict[int, int],
    name_to_id: dict[str, int],
    n_players: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(Ffield, payout, n_field, n_paid) for one archived contest.

    Ffield is (n_field, n_players) float32 indicator; payout is indexed by
    0-based rank and cut to the field size, so only the unpaid tail shortens.
    """
    entries = pd.read_csv(entries_csv)
    entries["names"] = entries["lineup_raw"].map(parse_names)
    n_field = len(entries)
    payout = parse_payout_table(Path(payout_table).read_text(), n_field)
    n_paid = int((payout > 0).sum())

    Ffield = np.zeros((n_field, n_players), dtype=np.float32)
    for r, ns in enumerate(entries["names"]):
        for n in ns:
            Ffield[r, pid_index[name_to_id[n]]] = 1.0
    return Ffield, payout, n_field, n_paid


def _field_ranks(
    s: np.ndarray, Ffield: np.ndarray, X: np.ndarray, n_field: int,
) -> np.ndarray:
    """(c, K) count of REAL field entries scoring above each of X's lineups.

    One global searchsorted rather than a per-world Python loop: each world's
    block is offset into its own disjoint value range, so the concatenation of
    per-world sorted fields is already globally sorted.
    """
    c = s.shape[0]
    FS = np.sort((s @ Ffield.T).astype(np.float64), axis=1)      # ascending
    XS = (s @ X.T).astype(np.float64)                            # (c, K)
    offs = (np.arange(c) * _WORLD_OFFSET)[:, None]
    idx = np.searchsorted((FS + offs).ravel(), (XS + offs).ravel(), side="right")
    n_le = idx - np.repeat(np.arange(c) * n_field, X.shape[0])
    return (n_field - n_le).reshape(c, -1), XS


def grade_marginal(
    engine,
    X: np.ndarray,
    Ffield: np.ndarray,
    payout: np.ndarray,
    n_sims: int,
    sim_batch: int = 25_000,
    chunk: int = 500,
    seed: int = 42,
    used_cols: Optional[np.ndarray] = None,
    progress: bool = True,
) -> np.ndarray:
    """Mean gross $ per lineup, each graded as an independent extra entry.

    Every lineup is ranked against the real field ONLY -- never against the
    others in X -- so this says nothing about how they would interact if all of
    them were entered together. Use `grade_portfolio` for that.
    """
    n_field = Ffield.shape[0]
    n_paid = int((payout > 0).sum())
    gross = np.zeros(X.shape[0], dtype=np.float64)
    np.random.seed(seed)
    done = 0
    while done < n_sims:
        b = min(sim_batch, n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix.astype(np.float32)
        if used_cols is not None:
            sc = sc[:, used_cols]
        for st in range(0, b, chunk):
            s = sc[st:st + chunk]
            rank, _ = _field_ranks(s, Ffield, X, n_field)
            pay = np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0)
            gross += pay.sum(axis=0)
        done += b
        if progress:
            print(f"      marginal {done:,}/{n_sims:,}")
        del sim, sc
    return gross / n_sims


def grade_portfolio(
    engine,
    X: np.ndarray,
    Ffield: np.ndarray,
    payout: np.ndarray,
    n_sims: int,
    sim_batch: int = 25_000,
    chunk: int = 500,
    seed: int = 42,
    used_cols: Optional[np.ndarray] = None,
    progress: bool = True,
) -> np.ndarray:
    """Mean gross $ per lineup with ALL K entries inserted simultaneously.

    The field is `n_field + K`, and a lineup's rank counts both the real field
    entries above it and its own team-mates above it. The second half is the
    demotion term: two co-moving entries push each other down a rung in exactly
    the worlds where they both do well, which is the cost a diversity objective
    exists to avoid. Summing the return gives portfolio gross; the caller
    subtracts K * fee.

    Ties are ignored deliberately -- simulated scores are continuous floats, so
    exact ties have probability zero (unlike the realized standings, where DK
    ties must be split).
    """
    n_field = Ffield.shape[0]
    K = X.shape[0]
    n_paid = int((payout > 0).sum())
    gross = np.zeros(K, dtype=np.float64)
    np.random.seed(seed)
    done = 0
    while done < n_sims:
        b = min(sim_batch, n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix.astype(np.float32)
        if used_cols is not None:
            sc = sc[:, used_cols]
        for st in range(0, b, chunk):
            s = sc[st:st + chunk]
            n_above_field, XS = _field_ranks(s, Ffield, X, n_field)
            # Self-competition: how many of OUR OWN entries outscore each one.
            order = np.argsort(-XS, axis=1, kind="stable")
            own_rank = np.empty_like(order)
            np.put_along_axis(
                own_rank, order,
                np.broadcast_to(np.arange(K), XS.shape), axis=1)
            rank = n_above_field + own_rank          # 0-based in the joint field
            pay = np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0)
            gross += pay.sum(axis=0)
        done += b
        if progress:
            print(f"      portfolio {done:,}/{n_sims:,}")
        del sim, sc
    return gross / n_sims


def grade_portfolios_multi(
    engine,
    portfolios: dict,
    Ffield: np.ndarray,
    payout: np.ndarray,
    n_sims: int,
    sim_batch: int = 25_000,
    chunk: int = 500,
    seed: int = 42,
    used_cols: Optional[np.ndarray] = None,
    progress: bool = True,
) -> tuple[dict, np.ndarray]:
    """Grade many portfolios in ONE simulation pass.

    ({label: mean gross $ per lineup, portfolio mode}, marginal gross for the
    stacked union of all lineups in `portfolios` iteration order).

    Grading each arm separately re-simulates the same worlds once per arm and
    re-sorts the same field once per arm — 18 full passes for 9 arms x 2 modes.
    The field sort is the expensive shared term, so it is done once per world
    chunk and every arm reads it. Marginal mode needs no per-arm state at all
    (each lineup is ranked against the field alone), so all arms are stacked
    into one matrix and graded together.
    """
    n_field = Ffield.shape[0]
    n_paid = int((payout > 0).sum())
    labels = list(portfolios)
    mats = [np.ascontiguousarray(portfolios[k]) for k in labels]
    sizes = [m.shape[0] for m in mats]
    stacked = np.vstack(mats)
    gross_port = {k: np.zeros(sz, dtype=np.float64) for k, sz in zip(labels, sizes)}
    gross_marg = np.zeros(stacked.shape[0], dtype=np.float64)

    np.random.seed(seed)
    done = 0
    while done < n_sims:
        b = min(sim_batch, n_sims - done)
        sim = engine.simulate(b)
        sc = sim.results_matrix.astype(np.float32)
        if used_cols is not None:
            sc = sc[:, used_cols]
        for st in range(0, b, chunk):
            s = sc[st:st + chunk]
            c = s.shape[0]
            # Sorted field: computed ONCE per chunk, read by every arm.
            FS = np.sort((s @ Ffield.T).astype(np.float64), axis=1)
            offs = (np.arange(c) * _WORLD_OFFSET)[:, None]
            flat = (FS + offs).ravel()
            del FS

            # Marginal: one stacked searchsorted for every lineup of every arm.
            XS_all = (s @ stacked.T).astype(np.float64)
            idx = np.searchsorted(flat, (XS_all + offs).ravel(), side="right")
            n_le = idx - np.repeat(np.arange(c) * n_field, stacked.shape[0])
            rank_m = (n_field - n_le).reshape(c, -1)
            gross_marg += np.where(
                rank_m < n_paid, payout[np.clip(rank_m, 0, n_paid - 1)], 0.0
            ).sum(axis=0)

            # Portfolio: the own-rank term is per-arm, so split the same ranks.
            off = 0
            for k, sz in zip(labels, sizes):
                sl = slice(off, off + sz)
                XS = XS_all[:, sl]
                order = np.argsort(-XS, axis=1, kind="stable")
                own_rank = np.empty_like(order)
                np.put_along_axis(
                    own_rank, order,
                    np.broadcast_to(np.arange(sz), XS.shape), axis=1)
                rank = rank_m[:, sl] + own_rank
                gross_port[k] += np.where(
                    rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0
                ).sum(axis=0)
                off += sz
            del XS_all, idx, n_le, rank_m, flat
        done += b
        if progress:
            print(f"      grading {done:,}/{n_sims:,}")
        del sim, sc
    return ({k: v / n_sims for k, v in gross_port.items()}, gross_marg / n_sims)


def summarize(
    per_lineup_gross: np.ndarray, entry_fee: float, label: str = "",
) -> dict:
    """Portfolio-level and per-lineup rollup of a graded set."""
    k = len(per_lineup_gross)
    cost = entry_fee * k
    total = float(per_lineup_gross.sum())
    net = per_lineup_gross - entry_fee
    return {
        "label": label,
        "n": k,
        "cost": cost,
        "gross": total,
        "net": total - cost,
        "roi": (total - cost) / cost if cost else float("nan"),
        "mean_lineup_net": float(net.mean()),
        "pct_lineups_positive": 100.0 * float((net > 0).mean()),
        "best_lineup_net": float(net.max()),
        "worst_lineup_net": float(net.min()),
    }
