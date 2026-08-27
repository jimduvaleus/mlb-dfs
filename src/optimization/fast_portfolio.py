"""Build a 150-lineup portfolio from pre-lock inputs, priced for self-competition.

WHAT PROBLEM THIS SOLVES. Entering 150 lineups into one contest means they
compete with each other: two near-identical entries do not cover two outcomes,
they crowd the same rank band in the same worlds. Diversity is not the goal, it
is one proxy for the goal. This module builds a pool, gates it on ceiling and
ownership, and then hands the survivors to any of several selection objectives
that differ precisely in HOW they price that crowding -- from exact demotion
(dR) through log-concavity (Kelly) and best-entry coverage (E[max]) down to the
correlation and world-coverage proxies.

INFORMATION BOUNDARY. Everything here is available before lock: projections, the
copula simulation, projected ownership, the published payout ladder, and a field
SIMULATED from projected ownership. The realized field, actual %Drafted and
actual results are never touched -- those belong only to the grading harness in
`scripts/portfolio_grading.py`.

THE ONE RULE THE GATE EXISTS TO ENFORCE. Ceiling is a CONSTRAINT, never a term.
Measured on the 08/25/2026 slate against two real fields, `z(ceiling) - z(own)`
returned -15.8% and pure lowest-ownership -99.1%, because ownership can always
be bought down by rostering worse players and a difference of z-scores happily
pays ceiling for it. The conjunction -- gate on ceiling, THEN rank by ownership
among the survivors -- returned +7.7% and +50.6%. `conjunctive_gate` is written
so the degenerate form is not expressible.

ARRAY CONVENTIONS, which the selectors disagree about if you let them:
    cand_scores / cand_payout : (M, S)   candidates x sim worlds
    field_sorted              : (S, F)   sim worlds x field entries, ASCENDING
    indicator                 : (M, P)   candidates x players
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as _dcfield
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.candidate_generator import CandidateGenerator, physical_cores
from src.optimization.gpp_portfolio import (
    CoveragePortfolioSelector,
    DeterminantPortfolioSelector,
    EMaxPortfolioSelector,
    KellyPortfolioSelector,
)
from src.optimization.lineup import Lineup, normalize_eligible_positions
from src.optimization.optimal_lineups import (
    generate_sim_optimal_lineups,
    stratified_sim_sample,
)

logger = logging.getLogger(__name__)

# Per-world value offset that folds a per-world searchsorted into one global
# call; must exceed any realistic lineup score so world w's block cannot
# overlap world w+1's.
_WORLD_OFFSET = 1e6


@dataclass
class FastPortfolioConfig:
    # --- generation ---
    n_candidates: int = 30_000
    salary_floor: float = 48_700.0        # real entrants play near the cap
    n_anchors: int = 800                  # per-world ILP seeds; THE budget dial
    anchor_salary_floor: float = 49_500.0
    anchor_seed_sims: int = 10_000
    anchor_min_stack: int = 5
    mutants_per_anchor: int = 10
    cbc_workers: int = 0                  # 0 = auto (physical cores)
    mutant_workers: int = 0               # 0 = auto (physical - 1)

    # --- ceiling ---
    ceiling_worlds: int = 25_000
    ceiling_world_batch: int = 5_000      # sim-buffer fill size only
    ceiling_cand_block: int = 5_000
    ceiling_bar_pct: float = 99.5         # pool-relative bar defining coverage bits

    # --- conjunctive gate ---
    ceiling_gate_pool_pct: float = 95.0   # keep the top (100-x)% by ceiling
    own_gate_pct: float = 40.0            # percentile among GATE-A SURVIVORS only
    target_shortlist: int = 4_000
    min_shortlist: int = 1_500

    # --- contest context ---
    field_size: int = 15_000
    contest_worlds: int = 12_500

    # --- selection ---
    portfolio_size: int = 150
    kelly_bankroll_mult: float = 3.0      # B = mult * size * fee; must exceed max loss
    det_risk: float = 1.0
    det_lane_fraction: float = 0.20
    corr_max_sims: int = 12_500

    seed: int = 11

    def __post_init__(self) -> None:
        if self.ceiling_worlds < 1 or self.ceiling_world_batch < 1:
            raise ValueError("ceiling_worlds and ceiling_world_batch must be >= 1")


# ---------------------------------------------------------------------------
# Stage 1-4: pool
# ---------------------------------------------------------------------------

def build_pool(
    players_df: pd.DataFrame,
    engine,
    own_pct: np.ndarray,
    cfg: FastPortfolioConfig,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Lineup]:
    """Sampler + per-world ILP ceiling anchors + shape-preserving mutants.

    The sampler alone is not enough: measured on 08/25, a raw 20k
    CandidateGenerator pool had mean lineup ceiling 172.9 against the real
    field's 192.2, with only 4.2% of it clearing the field's MEDIAN. The ILP
    anchors are per-world optima by construction and are what put mass in the
    part of the ceiling distribution the gate then selects from.
    """
    say = progress or (lambda m: None)
    # SimulationEngine.simulate() draws from the GLOBAL numpy RNG, so the
    # anchor seed-sim below — and therefore which worlds get ILP-solved, which
    # anchors come back, and every mutant grown off them — is unreproducible
    # unless the global stream is pinned here. The sampler is already seeded
    # through the constructor, which is why only part of the pool drifted:
    # measured 08/26, two runs at identical config shared 148/150 lineups on an
    # ownership-ranked arm (sampler-dominated) but 7/150 on a random draw
    # (pool-index-dependent). Cross-run comparisons were meaningless until this.
    np.random.seed(cfg.seed)
    gen = CandidateGenerator(
        players_df, own_pct, rng_seed=cfg.seed, salary_floor=cfg.salary_floor,
    )
    cands = list(gen.generate(n_candidates=cfg.n_candidates))
    say(f"{len(cands):,} from the sampler")

    if cfg.n_anchors > 0:
        seed_sim = engine.simulate(max(cfg.anchor_seed_sims, cfg.n_anchors * 5))
        df_ilp = players_df.copy()
        df_ilp["eligible_positions"] = [
            normalize_eligible_positions(e, p)
            for e, p in zip(players_df["eligible_positions"], players_df["position"])
        ]
        idx = [
            i for i, _ in stratified_sim_sample(
                seed_sim.results_matrix, cfg.n_anchors,
                np.random.default_rng(cfg.seed),
            )
        ]
        seen = {frozenset(int(p) for p in lu.player_ids) for lu in cands}
        anchors = generate_sim_optimal_lineups(
            df_ilp, seed_sim.results_matrix, list(seed_sim.player_ids), idx,
            min_stack=cfg.anchor_min_stack, salary_floor=cfg.anchor_salary_floor,
            seen=seen, n_workers=cfg.cbc_workers or None,
        )
        say(f"+{len(anchors):,} per-world ILP ceiling anchors")
        cands += list(anchors)

        if cfg.mutants_per_anchor > 0 and anchors:
            seen2 = {frozenset(int(p) for p in lu.player_ids) for lu in cands}
            muts = gen.generate_shape_mutants(
                anchors, n_per_parent=cfg.mutants_per_anchor, seen=seen2,
                rng_seed=cfg.seed + 5, salary_floor=cfg.salary_floor,
                n_workers=cfg.mutant_workers,
            )
            say(f"+{len(muts):,} shape-preserving mutants of those anchors")
            cands += list(muts)
        del seed_sim
    return cands


# Ownership currency floor, matching leverage.py's OWN_FLOOR_PCT: a projected
# 0.0% would send the log form to -inf and let one unrostered player dominate
# the whole lineup's score.
_OWN_FLOOR_PCT = 0.1


def ownership_currency(C: np.ndarray, own_pct: np.ndarray, metric: str) -> np.ndarray:
    """(M,) per-lineup ownership score; LOWER is more contrarian in both forms.

    The two forms are not variants of one idea, they measure different things:

      sum : Sum p_i -- the expected number of players shared with ONE random
            field entry. Dominated by HIGH-owned players; a single 60%-owned
            pitcher counts as much as twelve 5%-owned bats. Reads "how much
            chalk am I carrying".

      log : Sum log p_i -- the log-probability a random field entry is EXACTLY
            this lineup under independence, i.e. duplication probability.
            Dominated by LOW-owned players; a 0.5% player contributes -5.3
            against a 60% player's -0.51. Reads "how hard am I to duplicate".

    They correlate only 0.854 on a real pool, and the log form was the stronger
    predictor of realized ROI in both archived contests (partial rho against
    ceiling: -0.627 / -0.645, versus -0.448 / -0.537 for the sum). That is
    evidence the thing that matters is RARITY rather than aggregate
    chalk-avoidance -- consistent with the self-competition story, where what
    hurts is other people holding your exact bet.

    Not offered as a default because the measurement above was made on MARGINAL
    ROI; whether it survives portfolio-mode grading is a separate question.
    """
    if metric == "sum":
        return C @ own_pct
    if metric == "log":
        return C @ np.log(np.clip(own_pct, _OWN_FLOOR_PCT, None))
    raise ValueError(f"unknown ownership metric {metric!r} (want 'sum' or 'log')")


def indicator_matrix(candidates: list[Lineup], pid_index: dict[int, int]) -> np.ndarray:
    """(M, P) float32 roster indicator."""
    A = np.zeros((len(candidates), len(pid_index)), dtype=np.float32)
    for r, lu in enumerate(candidates):
        for p in lu.player_ids:
            A[r, pid_index[int(p)]] = 1.0
    return A


# ---------------------------------------------------------------------------
# Stage 5: ceiling + coverage bits, one streaming pass
# ---------------------------------------------------------------------------

def lineup_ceilings(
    engine,
    C: np.ndarray,
    cfg: FastPortfolioConfig,
    progress: Optional[Callable[[str], None]] = None,
    gate_currency: str = "abs",
    world_chunk: int = 2_500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(ceiling p99.9 (M,), coverage bits (M, W/8), sim32 (W, P), bar).

    GATE CURRENCY. `abs` scores a lineup by p99.9 of its raw POINTS; `rank`
    scores it by how many worlds it clears the POOL's OWN percentile for that
    world -- a bar that floats with the run environment. The ladder pays rank,
    not points: in a quiet world every score is low INCLUDING the field's, so a
    lineup can win on 150 where an explosive world needs 210, and an absolute
    bar discards it. Measured 08/26 against mean marginal payout across four
    ladders (235 -> 11,437 entries), rank beat points on every one --
    Spearman 0.959/0.966/0.941/0.884 vs 0.890/0.899/0.854/0.788, and top-5%
    recall of the genuinely best-paying lineups 0.561/0.577/0.424/0.238 vs
    0.426/0.444/0.348/0.210. The edge is largest in SMALL fields, where winning
    is a local question, and shrinks as the field grows and an absolute outlier
    is needed anyway.

    `rank` costs a second pass: a per-world bar needs every candidate present
    for that world, which is the opposite chunking axis from the per-lineup
    percentile, so the score matrix is traversed twice (~+20s at 38k x 12.5k).

    The world batch is purely how the retained sim buffer is filled -- packing
    runs over the WHOLE world axis per candidate block, so neither the batch
    nor the world count has to be a multiple of 8 (an earlier guard demanded
    both; `n_bytes` now uses ceiling division instead, which is what packbits
    actually returns).

    One pass over `ceiling_worlds` yields both currencies the downstream arms
    need, so the world set can never drift between the gate and the selection.
    The retained (W, P) float32 sim buffer is 111 MB at 25,000 x 1,108 and is
    reused for the contest context -- cheaper than re-simulating and, more
    importantly, keeps every stage on the SAME worlds.

    Chunked over candidates: the (batch x M) score block is the large
    transient, so it is cut at `ceiling_cand_block` per CLAUDE.md's matrix rule
    (measured 0.5 GB peak at 5,000 x 5,000).
    """
    say = progress or (lambda m: None)
    M = C.shape[0]
    W = cfg.ceiling_worlds
    B = cfg.ceiling_world_batch
    # Ceiling division, matching what np.packbits actually returns: a world
    # count that is not a multiple of 8 still occupies a whole final byte, and
    # `W // 8` silently under-allocates it. The padding bits are zero, so they
    # read as "not covered" and never inflate a coverage count.
    n_bytes = -(-W // 8)

    ceiling = np.empty(M, dtype=np.float64)
    bits = np.zeros((M, n_bytes), dtype=np.uint8)
    sim32 = np.empty((W, C.shape[1]), dtype=np.float32)

    np.random.seed(cfg.seed)
    for w0 in range(0, W, B):
        b = min(B, W - w0)
        sim = engine.simulate(b)
        sim32[w0:w0 + b] = sim.results_matrix.astype(np.float32)
        del sim
        say(f"ceiling sim {w0 + b:,}/{W:,}")

    # The bar defining a "covered" world is pool-relative and slate-absolute:
    # one number, derived from the pool's own score distribution, so it needs no
    # field. Taken from the first batch to avoid a second full pass.
    probe = sim32[:min(B, W)] @ C[:min(cfg.ceiling_cand_block, M)].T
    bar = float(np.percentile(probe, cfg.ceiling_bar_pct))
    del probe

    # Pass 1, chunked over CANDIDATES: the per-lineup points percentile.
    for c0 in range(0, M, cfg.ceiling_cand_block):
        c1 = min(c0 + cfg.ceiling_cand_block, M)
        S = sim32 @ C[c0:c1].T                      # (W, blk) float32
        ceiling[c0:c1] = np.percentile(S, 99.9, axis=0)
        if gate_currency == "abs":
            bits[c0:c1] = np.packbits((S >= bar).T, axis=1)
        del S
        say(f"ceiling block {c1:,}/{M:,}")

    if gate_currency == "rank":
        # Pass 2, chunked over WORLDS: a bar that floats with each world's run
        # environment. The (M, W) boolean is materialised once and packed at
        # the end rather than packed per chunk, so no chunk size has to land on
        # a byte boundary (483 MB at 38,650 x 12,500).
        beat = np.zeros((M, W), dtype=bool)
        for w0 in range(0, W, world_chunk):
            w1 = min(w0 + world_chunk, W)
            blk = sim32[w0:w1] @ C.T                # (c, M) float32
            bar_w = np.percentile(blk, cfg.ceiling_bar_pct, axis=1)
            beat[:, w0:w1] = (blk >= bar_w[:, None]).T
            del blk, bar_w
            say(f"rank worlds {w1:,}/{W:,}")
        ceiling = beat.sum(axis=1).astype(np.float64)
        bits = np.packbits(beat, axis=1)
        del beat
    elif gate_currency != "abs":
        raise ValueError(f"gate_currency must be 'abs' or 'rank', got {gate_currency!r}")
    return ceiling, bits, sim32, bar


# ---------------------------------------------------------------------------
# Stage 6: the conjunctive gate
# ---------------------------------------------------------------------------

def conjunctive_gate(
    ceiling: np.ndarray,
    own_sum: np.ndarray,
    cfg: FastPortfolioConfig,
) -> tuple[np.ndarray, dict]:
    """Hard ceiling cut, THEN an ownership cut among its survivors.

    The ordering is the whole design. Ownership is only ever compared among
    lineups that have already proven their ceiling, so "less owned" can never be
    purchased with quality -- which is exactly the failure mode that made
    z(ceiling) - z(own) return -15.8%.

    Ceiling is NEVER relaxed to hit a size target; only the ownership
    percentile widens. A caller who wants a bigger shortlist gets more of the
    ownership distribution, not a lower ceiling bar.
    """
    if not np.isfinite(own_sum).all() or float(np.std(own_sum)) <= 0.0:
        raise ValueError(
            "own_sum is constant or non-finite — the projected-ownership column "
            "is missing or all-zero, which would make the ownership gate a "
            "silent no-op. Check players_df['ownership']."
        )
    c_star = float(np.percentile(ceiling, cfg.ceiling_gate_pool_pct))
    gate_a = np.where(ceiling >= c_star)[0]
    if len(gate_a) == 0:
        raise ValueError("ceiling gate admitted nothing")

    own_a = own_sum[gate_a]
    pct = float(cfg.own_gate_pct)
    order = np.argsort(own_a, kind="stable")
    keep = max(cfg.min_shortlist, int(np.ceil(len(gate_a) * pct / 100.0)))
    keep = min(keep, len(gate_a), cfg.target_shortlist)
    shortlist = gate_a[order[:keep]]
    o_star = float(own_a[order[keep - 1]])

    assert ceiling[shortlist].min() >= c_star - 1e-9
    assert own_sum[shortlist].max() <= o_star + 1e-9
    diag = {
        "c_star": c_star,
        "o_star": o_star,
        "n_gate_a": int(len(gate_a)),
        "n_shortlist": int(len(shortlist)),
        "pool_pct_admitted": 100.0 * len(shortlist) / len(ceiling),
        "shortlist_ceiling_mean": float(ceiling[shortlist].mean()),
        "shortlist_own_mean": float(own_sum[shortlist].mean()),
    }
    return shortlist, diag


def random_shortlist(n_pool: int, size: int, seed: int) -> tuple[np.ndarray, dict]:
    """An UNBIASED sample of the pool -- no ceiling cut, no ownership cut.

    The control for "do the objectives find the good quadrant unaided?". It has
    to be a random draw rather than a widened gate, because `conjunctive_gate`
    finishes by taking the least-owned `target_shortlist` of the ceiling
    survivors: relaxing the ceiling cut alone makes gate A admit more than the
    shortlist cap, at which point that ownership sort silently becomes the
    binding gate and the ablation measures nothing. (Not hypothetical -- the
    first ownership-gate ablation was clean only because gate A happened to
    admit 1,932 against a 4,000 cap.)
    """
    rng = np.random.default_rng(seed)
    take = min(size, n_pool)
    idx = np.sort(rng.choice(n_pool, take, replace=False))
    return idx, {"mode": "random", "n_shortlist": int(take),
                 "c_star": float("nan"), "o_star": float("nan"),
                 "n_gate_a": int(n_pool),
                 "pool_pct_admitted": 100.0 * take / n_pool}


def anchor_ceiling_reference(ceiling: np.ndarray, anchor_slice: slice) -> float:
    """Field-free cross-check on C*: the ILP anchors' own median ceiling.

    The anchors are per-world optima, so their ceiling distribution is a
    model-internal statement about what a good lineup looks like on this slate,
    with no reference to any field. If C* lands far from it, the pool-percentile
    calibration is suspect on this slate and should be logged as such.
    """
    seg = ceiling[anchor_slice]
    return float(np.median(seg)) if seg.size else float("nan")


# ---------------------------------------------------------------------------
# Stage 7-8: contest context
# ---------------------------------------------------------------------------

def candidate_payout_matrix(
    cand_scores: np.ndarray,
    field_sorted: np.ndarray,
    payout: np.ndarray,
    world_chunk: int = 1_000,
) -> np.ndarray:
    """(M, S) gross $ per candidate per world against the simulated field.

    This is the MARGINAL payout each candidate would earn entered alone. Kelly,
    E[max], Coverage and Determinant all consume it; only dR goes deeper and
    re-ranks incumbents, which is why dR needs `field_sorted` itself.

    Chunked over worlds: the searchsorted operand is (chunk x F) and the result
    (M x chunk), so nothing (M x S)-shaped beyond the output is ever built.
    """
    M, S = cand_scores.shape
    F = field_sorted.shape[1]
    n_paid = int((payout > 0).sum())
    out = np.zeros((M, S), dtype=np.float32)
    for w0 in range(0, S, world_chunk):
        w1 = min(w0 + world_chunk, S)
        c = w1 - w0
        fs = field_sorted[w0:w1].astype(np.float64)          # (c, F) ascending
        cs = cand_scores[:, w0:w1].T.astype(np.float64)      # (c, M)
        offs = (np.arange(c) * _WORLD_OFFSET)[:, None]
        idx = np.searchsorted((fs + offs).ravel(), (cs + offs).ravel(), side="right")
        n_le = idx - np.repeat(np.arange(c) * F, M)
        rank = (F - n_le).reshape(c, M)
        pay = np.where(rank < n_paid, payout[np.clip(rank, 0, n_paid - 1)], 0.0)
        out[:, w0:w1] = pay.T.astype(np.float32)
        del fs, cs, idx, n_le, rank, pay
    return out


# ---------------------------------------------------------------------------
# Stage 9-11: the arms
# ---------------------------------------------------------------------------

def _wrap(picks: list[tuple[Lineup, float]]) -> list[Lineup]:
    return [lu for lu, _ in picks]


def select_dr(cand_scores, field_sorted, payout, candidates, cfg, progress=None):
    """Exact marginal reward with the demotion term -- the direct objective.

    dR(j|S) = E[payout(j)] + sum_i E[payout(i | j) - payout(i)]. The second term
    is self-competition itself, priced rather than proxied, and it is what makes
    the objective submodular.
    """
    from src.optimization.mrp.delta_reward import ContestDeltaState
    say = progress or (lambda m: None)
    st = ContestDeltaState(cand_scores, field_sorted, payout, chunk=512)
    picks: list[int] = []
    # `marginal_gains()` is defined over EVERY candidate including the ones
    # already committed -- dR(j | S) for j already in S is a perfectly
    # well-formed number, just not one we may act on. Without this mask the
    # greedy can re-pick an incumbent, which is the one thing this objective
    # exists to prevent: entering the same lineup twice is maximal
    # self-competition. Caught 08/26 when a deduped pool still yielded a
    # repeated entry in the dR arm.
    taken = np.zeros(len(candidates), dtype=bool)
    for k in range(cfg.portfolio_size):
        g = np.asarray(st.marginal_gains(), dtype=np.float64).copy()
        g[taken] = -np.inf
        j = int(np.argmax(g))
        if not np.isfinite(g[j]):
            say(f"dR exhausted the pool at {k} picks")
            break
        taken[j] = True
        st.commit(j)
        picks.append(j)
        if progress and (k + 1) % 25 == 0:
            say(f"dR {k + 1}/{cfg.portfolio_size}  reward ${st.reward():,.0f}")
    return [candidates[i] for i in picks], {"reward": float(st.reward())}


def select_kelly(cand_payout, candidates, cfg, entry_fee):
    bankroll = cfg.kelly_bankroll_mult * cfg.portfolio_size * entry_fee
    sel = KellyPortfolioSelector(
        cand_payout, candidates, portfolio_size=cfg.portfolio_size,
        bankroll=bankroll, ev_floor=float("-inf"),
    )
    return _wrap(sel.select()), {"bankroll": bankroll}


def select_emax(cand_payout, candidates, cfg):
    sel = EMaxPortfolioSelector(
        cand_payout, candidates, portfolio_size=cfg.portfolio_size,
        ev_floor=float("-inf"), baseline=0.0,
    )
    return _wrap(sel.select()), {}


def select_coverage(cand_payout, bits, own_sum, candidates, cfg):
    # ev_floor MUST be -inf: the class default of 0.20 is applied as
    # `pool_ev >= ev_floor` and would cull the entire pool here.
    sel = CoveragePortfolioSelector(
        cand_payout, candidates, portfolio_size=cfg.portfolio_size,
        beat999_bits=bits, tie_break=-own_sum, ev_floor=float("-inf"),
    )
    out = _wrap(sel.select())
    if not out:
        raise RuntimeError("CoverageSelector returned nothing — check ev_floor")
    return out, {}


def select_determinant(cand_payout, corr, own_sum, candidates, cfg):
    M = len(candidates)
    ev_override = np.full(M, np.nan, dtype=np.float64)
    ev_override[:] = -own_sum                      # higher = less owned
    sel = DeterminantPortfolioSelector(
        None, candidates, portfolio_size=cfg.portfolio_size,
        risk=cfg.det_risk, ev_floor=float("-inf"),
        precomputed=(np.arange(M), cand_payout.mean(axis=1).astype(np.float64), corr),
        ev_override=ev_override,
        lane_fraction=cfg.det_lane_fraction, lane_evw=0.0,
        rank_normalize=True,      # mandatory at 150; cardinal Dn saturates
    )
    return _wrap(sel.select()), {}


def select_gate_then_own(own_sum, candidates, cfg):
    """The measured incumbent: gate (already applied), then lowest ownership."""
    order = np.argsort(own_sum, kind="stable")[:cfg.portfolio_size]
    return [candidates[i] for i in order], {}
