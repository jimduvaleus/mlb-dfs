"""Self-play portfolio-construction primitives (Phase 1 prototype).

For one contest, iteratively scores real dollar ROI for every remaining
candidate lineup against `opponents U own_selections_so_far` (using the
contest's real payout table via the same tie-splitting kernel ContestScorer
uses), adds the single best lineup, and repeats until the contest's entry
count is filled. Diversity is a byproduct of genuinely competing against our
own prior picks each round, rather than a separate correlation-based penalty.

NO BATCHING: exactly one lineup is admitted per round, always. An earlier
version admitted several per round (batched) for large contests to cut
round count, but that reintroduces the exact failure mode self-play exists to
avoid -- the top-B candidates by ROI against one static field snapshot are
likely to be correlated/near-duplicate variants of whatever made #1 score
well, admitted together without ever having to prove themselves against each
other. The honest framing: wanting k lineups from a contest is better
understood as k separate, cheaper draws (fewer sims each) than fewer,
larger-batched draws at higher precision -- see round_n_sims below, which is
the actual lever for keeping many single-pick rounds affordable.

`opponents` is a FIXED-SIZE bucket for the whole build of one contest (never
replaced -- an earlier "swap out the lowest-ROI field member" design was
rejected because it biases field strength upward every round for no reason
grounded in how a real contest fills); the specific lineups populating it are
refreshed every round by subsampling from one large base opponent pool
generated once per SLATE (see build_base_opponent_pool / _SELF_PLAY_POOL_CAP)
-- opponent-lineup generation (stack-aware, salary-cap-respecting rejection
sampling in ContestSimulator.generate_field) is the expensive step, and reused
across every contest on that slate rather than regenerated per contest.

SHORTLIST RESTRICTION (round 1+): the plan's original step 8 wanted to defer
this until round-by-round evidence justified it, logged as a byproduct of
running the unrestricted algorithm. That evidence-gathering run turned out to
be the thing that's intractable to run: at production scale
(n_sims=25,000, ~34k-lineup candidate universe) a single full re-score of the
whole candidate universe took ~150-165s -- repeated every round for every
contest, one slate alone was projected at several hours (see
scripts/profile_self_play_round_cost.py). So round 0 of every contest still
scores the FULL remaining candidate universe (the one unavoidable full pass,
needed to establish who's actually in contention), but rounds 1+ only
re-score the top `shortlist_size` candidates from the previous round's
ranking -- a ~100x reduction in per-round cost for the typical shortlist_size
default. This is a real, accepted tradeoff (a candidate outside the shortlist
can never be discovered in a later round, even if a shifted field would have
favored it) rather than a validated-safe approximation; if the shortlist runs
low on remaining eligible members before a contest's k entries are filled, a
fresh full re-score refills it rather than under-filling silently.

PRECISION REFINEMENT (run_contest_precision_refinement, gated on field_size >=
_REFINEMENT_MIN_FIELD_SIZE): a bounded post-pass, run once per qualifying
contest after its round loop finishes, that corrects specifically for the
2,000-sim ranking noise documented at _PRECISE_N_SIMS_DEFAULT above. It scores
a widened candidate pool (the round loop's final shortlist, plus SaberSim-
99th/dk_std-ceiling-promoted candidates the cheap pass may never have seen)
at much higher sim depth, and iteratively swaps in any candidate that beats
the current weakest pick -- up to _REFINEMENT_MAX_SWAPS times, independent of
contest size. This is a local hill-climb, not a re-run of the sequential
round loop at high precision (that would cost roughly contest-size-many
precise field rebuilds, not a handful) and not a joint re-optimization of the
whole k-lineup set (a harder combinatorial problem this doesn't attempt).
Small contests are skipped entirely: their "top" percentile is a much higher
base rate to begin with (hit99.9 in a 470-entry field is just 1st place),
already resolvable at cheap precision, so the refinement cost buys little
there -- skipping them is what makes the higher precise_n_sims affordable on
the contests where the tail-rarity problem is actually severe.

See /home/jduvaleus/.claude/plans/reactive-yawning-cookie.md for the full
design writeup and the two-agent codebase exploration it's built on.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.optimization.contest import ContestSimulator
from src.optimization.gpp_portfolio import (
    _build_payout_lookup,
    _compute_payout_from_sorted_field,
    _payout_cumsum,
)
from src.optimization.lineup import Lineup

# scripts/profile_self_play_costs.py: generate_field ran ~540-585 lineups/sec
# (one-time, per-slate cost); 30k lineups ~55s, comfortably covers the
# largest real DK contest registered in payout.py (Rally Cap, 29,411
# entries). Memory: (n_sims, B) float32 field-score views peak at
# n_sims=10_000 * B=30_000 * 4B =~ 1.2GB, well under the 2.5GB budget
# _PWIN_FIELD_CAP was chosen for at 25k (src/api/external_pool.py:1278).
_SELF_PLAY_POOL_CAP = 30_000

# scripts/profile_self_play_round_cost.py at the full production n_sims=25,000:
# _merge_and_sort_field alone took ~13s/round on the slate's biggest contest
# (N=17,735), independent of how many candidates get scored -- the shortlist
# restriction above doesn't touch this cost at all, and a full slate (~10
# contests x ~15 rounds) was projected at several HOURS, far outside the
# ~10-15 minute real submission window this has to fit into. The round loop
# only needs sims to RANK candidates during construction (final grading uses
# real historical outcomes, which by definition don't exist yet at selection
# time -- there is no "free" way to skip simulation here, only a noisier one);
# this trades ranking precision for a ~12.5x cutdown, the same kind of
# noise-for-speed trade replay_slate.py's --screen mode already makes (10k
# vs. the 25k production default) -- REVISIT if a full run leaves time to
# spare against the submission window.
_ROUND_N_SIMS_DEFAULT = 2_000

# scripts/probe_sim_count_tail_sensitivity.py (2026-08-08): at n_sims=2,000,
# candidate ROI rankings for the rare, extreme-tail events hit99.9 depends on
# are dominated by sampling noise, not signal -- of the top-10 candidates by
# 2,000-sim ROI on the slate's biggest contest, only 3 were still top-10 at
# 25,000 sims, and one TRUE top-10 candidate (by the 25k estimate) ranked
# 740th out of 4,049 at 2k. hit99 (~1-in-100) is fine at 2k (~20 expected
# hits); hit99.9 in a large field (~1-in-1,000 to 1-in-tens-of-thousands) is
# not (often <1 expected hit per candidate). PRECISION REFINEMENT below
# exists to correct for this, targeted only where it matters.
_PRECISE_N_SIMS_DEFAULT = 20_000

# hit99.9 in a small field is a much higher base rate to begin with (top 1
# spot of 470 vs top ~18 of 17,835) -- already close to what cheap sims can
# resolve, so the precision pass buys little there for its ~10-25s/iteration
# cost. Gating on field size (not k) because field size is what drives event
# rarity, the actual reason precision matters -- see run_contest_precision_refinement.
_REFINEMENT_MIN_FIELD_SIZE = 2_000

# Bounded local hill-climb, not a full re-optimization -- see
# run_contest_precision_refinement. Independent of contest size/k by design,
# so cost stays predictable regardless of how large a contest is.
_REFINEMENT_MAX_SWAPS = 8

# Per source (external by its own "99th" column, generated by the dk_std-based
# ceiling_proxy) -- how many of the slate's best-looking-for-ceiling candidates
# get added to every contest's refinement pool regardless of whether the cheap
# round loop's shortlist already found them.
_PROMOTE_K_DEFAULT = 300

# glibc's malloc RAISES its mmap threshold after large frees (dynamic
# adjustment), pushing SUBSEQUENT similarly-sized allocations into the
# arena/brk heap instead of mmap -- arena memory is only returned to the OS
# via malloc_trim, not automatically on free(). Diagnosed 2026-08-08: after
# chunking _precise_scores_for to score only what each contest's refinement
# actually needs (no more persistent (M, precise_n_sims) array), a live
# single-slate profile still showed ~6.7GB of RSS growth during
# self_play_allocate_contests with no single persistently-held array behind
# it -- consistent with exactly this pattern across many differently-sized
# large-array alloc/free cycles (opponents_scores alone ranges from a few
# hundred KB to multiple GB depending on field size: this session's test
# slate spanned 476-23,781 implied entries, and this codebase runs contests
# bigger than that on other slates). Pinning the threshold low forces every
# one of these multi-MB+ arrays through mmap consistently, which glibc
# unmaps (truly returns to the OS) immediately on free -- addressing the
# mechanism, not just nudging its symptom after the fact.
_MMAP_THRESHOLD_BYTES = 256 * 1024
_malloc_tuned = False


def _tune_malloc_for_large_arrays() -> None:
    """Pin glibc's mmap threshold low -- see _MMAP_THRESHOLD_BYTES's comment.
    Idempotent (only calls mallopt once per process) and a no-op outside
    glibc (e.g. a non-Linux dev machine) or if libc can't be loaded."""
    global _malloc_tuned
    if _malloc_tuned:
        return
    _malloc_tuned = True
    try:
        libc = ctypes.CDLL("libc.so.6")
        M_MMAP_THRESHOLD = -3  # glibc malloc.h mallopt() param id
        libc.mallopt(ctypes.c_int(M_MMAP_THRESHOLD), ctypes.c_int(_MMAP_THRESHOLD_BYTES))
    except OSError:
        pass


def _release_free_memory() -> None:
    """Ask glibc to return freed ARENA memory to the OS -- a secondary net
    alongside _tune_malloc_for_large_arrays's fixed mmap threshold (which
    handles most of this on its own by keeping large arrays out of the
    arena in the first place; this covers whatever still lands there, e.g.
    allocations below the pinned threshold, or before it was set). Meant to
    be called between contests, once each one's big transient arrays have
    gone out of scope. No-op outside glibc."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def build_base_opponent_pool(
    players_df, ownership_vec: np.ndarray, n_lineups: int, rng_seed: int,
) -> np.ndarray:
    """(n_lineups, 10) int64 player-id rows, one ContestSimulator.generate_field()
    call. Meant to be called exactly once per slate -- see SelfPlaySlateContext."""
    return ContestSimulator().generate_field(
        players_df, ownership_vec, n_lineups=n_lineups, rng_seed=rng_seed,
    )


def _merge_and_sort_field(opponent_scores: np.ndarray, own_scores: np.ndarray) -> np.ndarray:
    """opponent_scores (n_sims, F) + own_scores (n_sims, m) -> (n_sims, F+m)
    sorted ascending along axis=1 -- the entire "score against opponents U
    own-selections-so-far" composition step. No need to touch ContestScorer's
    internals: this is what its `field_sorted` argument expects.

    A partition-based alternative (avoiding the full sort) was built and
    benchmarked 2026-08-08 (gpp_portfolio._tier_boundaries/_tier_partition/
    _compute_payout_tiered, kept there as tested-but-unused) and did NOT pay
    off: at mini-max's real scale (N~17,763, n_sims=20,000), np.partition's
    wall-clock cost -- even for a single cash-line target -- came in close to
    or ABOVE the full np.sort it was meant to avoid (2.0s partition vs 2.6s
    sort for one target; 8.3s vs 2.6s for the ~398 tiers a real payout table's
    banded lookup actually produces). numpy's sort is apparently well enough
    optimized relative to its partition implementation, at this array shape,
    that the textbook O(N) vs O(N log N) advantage doesn't show up in
    practice. Revisit only with real evidence a different approach changes
    that, not by re-deriving the theory again."""
    if own_scores.shape[1] == 0:
        merged = opponent_scores
    elif opponent_scores.shape[1] == 0:
        merged = own_scores
    else:
        merged = np.concatenate([opponent_scores, own_scores], axis=1)
    return np.ascontiguousarray(np.sort(merged, axis=1)).astype(np.float32)


def _score_against_field(
    cand_scores: np.ndarray, field_sorted: np.ndarray,
    payout_cumsum: np.ndarray, dilute_cumsum: np.ndarray,
    batch_size: int = 2_000,
) -> np.ndarray:
    """Mean-over-sims dollar ROI for every row of cand_scores (n_cand, n_sims)
    against field_sorted, chunked through _compute_payout_from_sorted_field
    to bound peak memory (mirrors ContestScorer's candidate_batch_size)."""
    n_cand = cand_scores.shape[0]
    out = np.empty(n_cand, dtype=np.float64)
    for lo in range(0, n_cand, batch_size):
        hi = min(lo + batch_size, n_cand)
        batch = np.ascontiguousarray(cand_scores[lo:hi])
        dupe_scale = np.ones(batch.shape[0], dtype=np.float32)
        payouts = _compute_payout_from_sorted_field(
            batch, field_sorted, payout_cumsum, dilute_cumsum, dupe_scale,
        )
        out[lo:hi] = payouts.mean(axis=1)
    return out


@dataclass
class SelfPlaySlateContext:
    """Per-slate state shared across every contest's self-play build.

    `lineups`/`source`/`scores` cover the WHOLE candidate universe: external
    pool lineups first, then the generated base opponent pool -- both are
    eligible to be picked into our own portfolio (a generated lineup scoring
    better than every external candidate is expected to happen often, since
    we're vastly outnumbered by generated lineups). Only the generated slice
    is ever eligible to serve as an *opponent* (external pool lineups are
    SaberSim's curated picks, not a representative sample of the field the
    ownership-weighted generator models) -- see opponent_mask below.

    PRE-LOCK STALENESS is NOT an external-only problem, despite how that might
    read at first: `generated` is built from `players_df`, which in a backtest
    comes from the archived projections file -- and that file is captured at
    essentially the SAME pre-lock moment as the external lineup export (verified
    2026-08-08: `MLB_..._640pm_DK_Main.csv` and `lineups_..._640pm.csv` are ~1s
    apart in the 07222026 archive). A since-scratched player still carries a
    nonzero projected mean/ownership in that frozen snapshot, so ownership-
    weighted generation can and does seed him into `generated` lineups too --
    scoring near-zero against real archived FPTS just like a stale `external`
    pick would. Neither source has a real informational edge over the other
    here; the actual asymmetry is both of them vs. the REAL FIELD, whose
    entrants (including our own accounts) could make post-lock swaps that
    neither pool reflects. A generated-vs-external gap in backtest results
    should NOT be attributed to this staleness effect -- both suffer it
    roughly equally -- see parse_lineup_pool's PRE-LOCK SNAPSHOT LIMITATION.
    """
    lineups: list                 # list[Lineup], external ++ generated, concatenated
    source: np.ndarray            # (M,) str, "external" | "generated"
    scores: np.ndarray            # (M, n_sims) float32 -- ep.compute_lineup_scores, cheap tier
    n_external: int
    n_sims: int
    # Precise-tier sim world subset (see PRECISION REFINEMENT), kept as the
    # SimulationResults object rather than a precomputed (M, precise_n_sims)
    # score matrix for the WHOLE candidate universe -- that matrix would be
    # ~3GB+ at production scale (M~40k lineups x precise_n_sims=20k) and,
    # unlike `scores` above, is used by AT MOST a few thousand lineups per
    # contest (final_shortlist_idx U promoted_idx U own_idx, not the whole
    # universe), so precomputing and holding it for the entire slate wastes
    # memory nothing downstream needs -- see _precise_scores_for, which
    # scores exactly the lineups a given refinement call needs, on demand.
    # Diagnosed 2026-08-08: this was the dominant contributor to a live
    # single-slate profile peaking at 12.2GB RSS (scripts/
    # profile_self_play_memory.py) before this fix.
    precise_sim: Optional["SimulationResults"] = None
    precise_n_sims: Optional[int] = None
    promoted_idx: Optional[np.ndarray] = None  # indices into lineups: top-K by external "99th" / generated ceiling_proxy

    def new_opponent_mask(self) -> np.ndarray:
        """(M,) bool, True only for the generated slice -- the initial
        pre-poaching set of opponent-eligible lineups."""
        mask = np.zeros(len(self.lineups), dtype=bool)
        mask[self.n_external:] = True
        return mask

    def new_candidate_mask(self) -> np.ndarray:
        """(M,) bool, True for every lineup not yet placed in ANY contest --
        the cross-contest removal mask, shared across the whole slate."""
        return np.ones(len(self.lineups), dtype=bool)


def build_self_play_context(
    sim_results, players_df, ownership_vec: np.ndarray, external_pool,
    base_pool_size: int = _SELF_PLAY_POOL_CAP, base_pool_seed: int = 42,
    round_n_sims: Optional[int] = _ROUND_N_SIMS_DEFAULT, round_sims_seed: int = 42,
    precise_n_sims: Optional[int] = _PRECISE_N_SIMS_DEFAULT,
    promote_k: int = _PROMOTE_K_DEFAULT,
) -> SelfPlaySlateContext:
    """One-time-per-slate setup: generate the base opponent pool and score
    the whole candidate universe (external U generated) in one pass, at both
    the cheap (round-loop) and precise (refinement) sim depths.

    `round_n_sims` (default 2,000, see _ROUND_N_SIMS_DEFAULT) and
    `precise_n_sims` (default 20,000, see _PRECISE_N_SIMS_DEFAULT) both
    subsample `sim_results` down to that many sim worlds before scoring, so
    every downstream cost that scales with n_sims scales with the smaller
    number for the round loop, the larger for precision refinement. Drawn
    from ONE shared permutation of sim indices so the cheap set is a strict
    subset of the precise set (simulating "more", not "different") -- not
    required for correctness, just the cleaner way to think about what
    refinement adds. `ep.compute_lineup_scores` is a single BLAS matmul over
    the WHOLE candidate universe regardless of n_sims (unlike the round
    loop's per-candidate binary-search kernel), so covering everyone at both
    depths here is cheap -- the expensive part was never this step, it's
    _score_against_field's per-candidate work, which precision refinement
    deliberately keeps scoped to a small candidate pool rather than the
    whole universe. Pass
    `round_n_sims=None`/`precise_n_sims=None` to use every sim in
    `sim_results` unchanged for that tier (only tractable for small slates
    or controlled experiments).

    `promote_k` (default 300, see _PROMOTE_K_DEFAULT) is how many candidates
    per source get added to `promoted_idx`: external candidates ranked by
    their own SaberSim "99th" column (src.api.external_pool.parse_pool_p99),
    generated candidates by `ep.compute_pool_ceiling_proxy` (no native
    ceiling column of their own -- see that function's docstring for the
    independence-assumption caveat). `promoted_idx` is independent of which
    contest is being refined; every qualifying contest's refinement pool
    includes it."""
    from src.api import external_pool as ep  # local: avoid api<->optimization import cycle
    from src.simulation.results import SimulationResults

    _tune_malloc_for_large_arrays()

    total_sims = sim_results.results_matrix.shape[0]
    perm = np.random.default_rng(round_sims_seed).permutation(total_sims)

    def _subsample(n: Optional[int]):
        if n is None or n >= total_sims:
            return sim_results
        idx = np.sort(perm[:n])
        return SimulationResults(sim_results.player_ids, sim_results.results_matrix[idx])

    cheap_sim = _subsample(round_n_sims)
    precise_sim = _subsample(precise_n_sims) if precise_n_sims is not None else None

    base_pool = build_base_opponent_pool(players_df, ownership_vec, base_pool_size, base_pool_seed)
    generated_lineups = [Lineup(player_ids=[int(p) for p in row]) for row in base_pool]
    n_external = len(external_pool.lineups)
    all_lineups = list(external_pool.lineups) + generated_lineups
    source = np.array(["external"] * n_external + ["generated"] * len(generated_lineups))

    # Cheap-tier scores cover the WHOLE universe -- every round loop needs
    # to rank the entire remaining candidate pool at least once (round 0).
    # Precise-tier scores are NOT precomputed for the whole universe (that
    # would be the ~3GB+ array _precise_scores_for's docstring warns about)
    # -- only precise_sim itself is kept here; run_contest_precision_
    # refinement scores just the few thousand lineups it actually needs,
    # per contest, via _precise_scores_for.
    scores = ep.compute_lineup_scores(all_lineups, cheap_sim).astype(np.float32)

    # Promotion set: external by SaberSim's own "99th" column, generated by
    # the dk_std-based ceiling proxy (neither needs any of the sims above).
    promoted_parts = []
    if n_external > 0:
        valid_ids = set(players_df["player_id"].astype(int))
        p99_lookup = ep.parse_pool_p99(external_pool.source_paths, valid_ids)
        p99 = np.array([
            p99_lookup.get(frozenset(lu.player_ids), np.nan) for lu in external_pool.lineups
        ])
        finite = np.flatnonzero(np.isfinite(p99))
        top_ext = finite[np.argsort(-p99[finite])[:promote_k]]
        promoted_parts.append(top_ext)
    if generated_lineups:
        ceiling_proxy = ep.compute_pool_ceiling_proxy(generated_lineups, players_df)
        top_gen = n_external + np.argsort(-ceiling_proxy)[:promote_k]
        promoted_parts.append(top_gen)
    promoted_idx = (
        np.unique(np.concatenate(promoted_parts)) if promoted_parts else np.empty(0, dtype=np.int64)
    )

    return SelfPlaySlateContext(
        lineups=all_lineups, source=source, scores=scores,
        n_external=n_external, n_sims=scores.shape[1],
        precise_sim=precise_sim,
        precise_n_sims=(precise_sim.results_matrix.shape[0] if precise_sim is not None else None),
        promoted_idx=promoted_idx,
    )


@dataclass
class SelfPlayContestResult:
    own_idx: list                 # indices into ctx.lineups, in fill order (pick order)
    own_roi: list                 # float, parallel to own_idx -- ROI at the moment of picking
    round_log: pd.DataFrame       # per-round operational log, see run_contest_self_play
    final_shortlist_idx: np.ndarray = None  # last round's shortlist -- refinement's starting pool


def run_contest_self_play(
    ctx: SelfPlaySlateContext,
    contest_id: str,
    k: int,
    field_size: float,
    payout_arr: np.ndarray,
    entry_fee: float,
    candidate_mask: np.ndarray,          # (M,) bool, mutated in place -- cross-contest removal
    opponent_available_mask: np.ndarray, # (M,) bool, mutated in place -- poaching removal
    rng: np.random.Generator,
    shortlist_size: int = 1_000,
    refresh_every: int = 5,
) -> SelfPlayContestResult:
    """Fill one contest's k entries via the self-play round loop. Mutates
    candidate_mask/opponent_available_mask in place so the caller's shared
    masks reflect this contest's picks.

    Round 0 always scores the FULL remaining candidate universe. Rounds 1+
    are restricted to the top `shortlist_size` candidates from the previous
    round's ranking (intersected with candidate_mask, since this contest's
    own picks shrink it every round) -- see the module docstring's "SHORTLIST
    RESTRICTION" note for why this is unconditional rather than
    evidence-gated. If the shortlist would run out before k is filled, a
    fresh full re-score replaces it rather than under-filling silently.
    `round_log` records, per round after round 0, whether it was a full or
    restricted re-score and how many candidates were scored -- an
    operational record of the algorithm's actual behavior, not a
    counterfactual comparison (there's no full-universe ranking to compare
    a restricted round against once restriction is unconditional).

    `refresh_every` draws a fresh opponent subsample only every N rounds
    (round 0 always draws fresh) instead of every round; between refreshes
    the SAME drawn indices are reused, filtered each round against
    opponent_available_mask so a cached opponent that gets poached into
    own_idx (ours or another contest's, mid-window) drops out rather than
    being double-counted -- it would otherwise show up both in the stale
    cached opponent set and in own_scores_so_far. Note this mainly cuts the
    number of rng.choice draws and, when a poach shrinks the cached set
    below n_opponents mid-window, opponents that round; the field STILL gets
    fully re-sorted every round regardless (own_scores_so_far grows every
    round, so there's no way to skip that part without a proper incremental
    merge, which wasn't judged worth building given round_n_sims's cut
    already brought that sort to ~1s at n_sims=2,000) -- so this is a
    smaller, secondary lever on top of round_n_sims, not an independent
    equal-sized win."""
    k = int(k)
    field_size = int(field_size)
    n_opponents = max(field_size - k, 0)

    own_idx: list[int] = []
    own_roi: list[float] = []
    round_rows: list[dict] = []
    shortlist_idx: Optional[np.ndarray] = None  # None -> next round does a full re-score
    chosen_opp: np.ndarray = np.empty(0, dtype=np.int64)
    remaining = k
    round_num = 0

    while remaining > 0:
        opp_pool_idx = np.flatnonzero(opponent_available_mask)
        if n_opponents > 0 and len(opp_pool_idx) < n_opponents:
            raise ValueError(
                f"{contest_id}: opponent pool exhausted ({len(opp_pool_idx)} available, "
                f"{n_opponents} needed) -- base_pool_size is too small for this slate's "
                "contest sizes, or too many generated lineups have been poached."
            )
        if n_opponents > 0 and round_num % refresh_every == 0:
            chosen_opp = rng.choice(opp_pool_idx, size=n_opponents, replace=False)
        elif n_opponents > 0:
            chosen_opp = chosen_opp[opponent_available_mask[chosen_opp]]  # drop any poached since the last draw
        opponents_scores = ctx.scores[chosen_opp].T if n_opponents > 0 else \
            np.zeros((ctx.n_sims, 0), dtype=np.float32)
        own_scores_so_far = ctx.scores[own_idx].T if own_idx else \
            np.zeros((ctx.n_sims, 0), dtype=np.float32)
        field_sorted = _merge_and_sort_field(opponents_scores, own_scores_so_far)

        # N = field_sorted.shape[1] grows every round as own_idx grows, so the
        # payout lookup MUST be rebuilt against the real payout_arr each time
        # (see plan's correctness note) -- _build_payout_lookup/_band_average
        # already interpolate the real fixed-size curve down to any N.
        N = field_sorted.shape[1]
        lookup = _build_payout_lookup(payout_arr, N=N, entry_fee=entry_fee)
        cumsum = _payout_cumsum(lookup)
        dilute_cumsum = np.zeros_like(cumsum)

        full_rescore = shortlist_idx is None
        if not full_rescore:
            score_idx = shortlist_idx[candidate_mask[shortlist_idx]]
            if score_idx.size < remaining:
                full_rescore = True  # shortlist can't cover what's left -- replenish it
        if full_rescore:
            score_idx = np.flatnonzero(candidate_mask)
        if score_idx.size == 0:
            break  # candidate universe exhausted -- caller reports the shortfall

        roi = _score_against_field(ctx.scores[score_idx], field_sorted, cumsum, dilute_cumsum)
        order = np.argsort(-roi)
        ranked_idx = score_idx[order]
        picked = int(ranked_idx[0])  # exactly one admission per round -- see NO BATCHING above

        if round_num > 0:
            round_rows.append({
                "contest_id": contest_id, "round": round_num,
                "full_rescore": full_rescore, "n_scored": int(score_idx.size),
            })

        own_idx.append(picked)
        own_roi.append(float(roi[order[0]]))
        candidate_mask[picked] = False
        opponent_available_mask[picked] = False  # no-op for external-sourced picks

        shortlist_idx = ranked_idx[:shortlist_size]
        remaining -= 1
        round_num += 1

    if shortlist_idx is None:
        shortlist_idx = np.empty(0, dtype=np.int64)
    return SelfPlayContestResult(
        own_idx=own_idx, own_roi=own_roi, round_log=pd.DataFrame(round_rows),
        final_shortlist_idx=shortlist_idx,
    )


def _precise_scores_for(ctx: SelfPlaySlateContext, idx: np.ndarray) -> np.ndarray:
    """On-demand precise-tier scoring for exactly the given lineup indices --
    see SelfPlaySlateContext.precise_sim's docstring for why this replaces a
    single (M, precise_n_sims) array held for the whole slate. `idx` is
    typically a few hundred to a few thousand lineups (one contest's
    refinement pool or opponent draw), never the whole candidate universe,
    so this stays cheap: ep.compute_lineup_scores is a single BLAS matmul,
    already noted elsewhere as cheap regardless of call count."""
    from src.api import external_pool as ep  # local: avoid api<->optimization import cycle

    lineups = [ctx.lineups[int(i)] for i in idx]
    return ep.compute_lineup_scores(lineups, ctx.precise_sim).astype(np.float32)


def run_contest_precision_refinement(
    ctx: SelfPlaySlateContext,
    contest_id: str,
    own_idx: list,
    own_roi: list,
    final_shortlist_idx: np.ndarray,
    field_size: float,
    k: int,
    payout_arr: np.ndarray,
    entry_fee: float,
    candidate_mask: np.ndarray,          # (M,) bool, mutated in place
    opponent_available_mask: np.ndarray, # (M,) bool, mutated in place
    rng: np.random.Generator,
    max_swaps: int = _REFINEMENT_MAX_SWAPS,
) -> tuple:
    """Bounded post-pass over one contest's cheap-precision picks -- see the
    module docstring's PRECISION REFINEMENT note for the design rationale.
    No-op (returns inputs unchanged, empty log) if `ctx.precise_sim` is
    None. Mutates candidate_mask/opponent_available_mask in place, same
    contract as run_contest_self_play, INCLUDING freeing a swapped-out
    candidate back to both masks -- it's no longer part of any portfolio,
    so it becomes available to whichever contest processes next.

    The opponent field is drawn ONCE for the whole pass, not once per
    iteration -- unlike the main round loop (whose `refresh_every` matters
    because it runs for potentially dozens of rounds and staleness
    compounds), refinement is a short, bounded local correction (<= max_swaps
    iterations), so one draw for the whole pass is a reasonable trade. This
    also matters for cost: field-sorting a large contest's opponent pool
    (e.g. mini-max's ~17,700 opponents) dominates each iteration's cost far
    more than the actual candidate scoring does, so paying it once instead of
    per-iteration is most of the reason refinement is affordable at all on
    the biggest fields -- see scripts/eval_self_play_selector.py timing notes.

    Each of up to `max_swaps` iterations:
      1. Identify the weakest current pick by its EXISTING (cheap-tier, or a
         prior iteration's precise-tier) own_roi -- no extra scoring needed
         for this step; the point of precision is verifying/replacing that
         pick, not re-deriving which one looks weakest.
      2. Score the refinement pool (final shortlist U promoted_idx U current
         own_idx, restricted to lineups still available or already ours)
         against the (fixed) opponents U (own_idx minus the weakest pick),
         at precise depth -- one _merge_and_sort_field + one
         _score_against_field call, reusing exactly the round loop's own
         field-composition logic.
      3. If the pool's best scorer isn't the weakest pick itself, swap it in.
      4. Stop early (before max_swaps) if a round finds no improving swap.

    own_roi entries end up a MIX of cheap- and precise-tier values after
    this runs (only swapped-into entries get refreshed) -- deliberately not
    reconciled to a uniform tier, since a full precise leave-one-out re-score
    of every final pick would itself cost k more precise field rebuilds per
    contest, which is exactly the per-round cost this whole design avoids.
    Callers that need to know which is which should consult the returned
    log (each swapped-in entry's contest/index appears there) rather than
    assume own_roi's precision uniformly."""
    if ctx.precise_sim is None or not own_idx:
        return own_idx, own_roi, pd.DataFrame()

    field_size = int(field_size)
    k = int(k)
    n_opponents = max(field_size - k, 0)

    promoted_idx = ctx.promoted_idx if ctx.promoted_idx is not None else np.empty(0, dtype=np.int64)
    pool_idx = np.unique(np.concatenate([
        final_shortlist_idx, promoted_idx, np.array(own_idx, dtype=np.int64),
    ]))
    eligible = candidate_mask.copy()
    eligible[own_idx] = True  # current picks must be able to "defend their spot"
    pool_idx = pool_idx[eligible[pool_idx]]

    own_idx = list(own_idx)
    own_roi = list(own_roi)
    swap_rows: list[dict] = []

    # One opponent draw for the whole bounded pass -- see docstring above.
    opp_pool_idx = np.flatnonzero(opponent_available_mask)
    if n_opponents > 0 and len(opp_pool_idx) < n_opponents:
        return own_idx, own_roi, pd.DataFrame()  # not enough opponents to build a precise field at all
    chosen_opp = (
        rng.choice(opp_pool_idx, size=n_opponents, replace=False)
        if n_opponents > 0 else np.empty(0, dtype=np.int64)
    )
    opponents_scores = _precise_scores_for(ctx, chosen_opp).T if n_opponents > 0 else \
        np.zeros((ctx.precise_n_sims, 0), dtype=np.float32)
    # pool_idx is fixed for the whole pass (see above), so its precise scores
    # are worth computing once rather than re-scoring the same lineups on
    # every iteration.
    pool_scores_precise = _precise_scores_for(ctx, pool_idx) if pool_idx.size else \
        np.zeros((0, ctx.precise_n_sims), dtype=np.float32)

    for it in range(max_swaps):
        if pool_idx.size == 0:
            break
        worst_local = int(np.argmin(own_roi))
        worst_idx = int(own_idx[worst_local])
        own_arr = np.array(own_idx, dtype=np.int64)
        rest = np.delete(own_arr, worst_local)
        rest_scores = _precise_scores_for(ctx, rest).T if rest.size else \
            np.zeros((ctx.precise_n_sims, 0), dtype=np.float32)
        field_sorted = _merge_and_sort_field(opponents_scores, rest_scores)

        N = field_sorted.shape[1]
        lookup = _build_payout_lookup(payout_arr, N=N, entry_fee=entry_fee)
        cumsum = _payout_cumsum(lookup)
        dilute_cumsum = np.zeros_like(cumsum)

        roi_pool = _score_against_field(
            pool_scores_precise, field_sorted, cumsum, dilute_cumsum,
        )
        best_local = int(np.argmax(roi_pool))
        best_idx = int(pool_idx[best_local])
        best_roi = float(roi_pool[best_local])

        if best_idx == worst_idx:
            break  # weakest pick still wins its own slot -- converged, nothing left to fix

        own_idx[worst_local] = best_idx
        own_roi[worst_local] = best_roi
        candidate_mask[worst_idx] = True
        candidate_mask[best_idx] = False
        if ctx.source[worst_idx] == "generated":
            opponent_available_mask[worst_idx] = True
        if ctx.source[best_idx] == "generated":
            opponent_available_mask[best_idx] = False

        swap_rows.append({
            "contest_id": contest_id, "iteration": it,
            "swapped_out_idx": worst_idx, "swapped_out_source": str(ctx.source[worst_idx]),
            "swapped_in_idx": best_idx, "swapped_in_source": str(ctx.source[best_idx]),
            "swapped_in_precise_roi": best_roi,
        })

    # Explicit del on the pass's big arrays (opponents_scores/
    # pool_scores_precise can be multiple GB for a large-field contest,
    # see _MMAP_THRESHOLD_BYTES's comment) rather than relying on function-
    # return scoping -- frees them here, before the caller's post-contest
    # _release_free_memory() runs, instead of at whatever point Python
    # would otherwise get around to it.
    del opponents_scores, pool_scores_precise
    return own_idx, own_roi, pd.DataFrame(swap_rows)
