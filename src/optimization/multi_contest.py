"""Group an entries file into contests, resolve each ladder, and order them.

WHY THIS EXISTS. The selection objectives (Kelly, dR, E[max], coverage) are
contest-aware: the payout ladder and field size are the only contest-specific
inputs they see, and they respond strongly — measured 08/26 on one slate, the
same candidate pool produced portfolios ~19 ownership points chalkier for a
235-entry contest than for an 11,437-entry one, unprompted. A single entries
file routinely spans contests that differ by orders of magnitude in both (a real
one here holds six, $2K to $20K pools, fees $1 to $12, entry caps 1 to 150), so
building ONE portfolio against ONE ladder and spreading it across all of them
throws that away.

That was harmless while every contest used the same fixed reference curve. It is
not harmless now.

FILL ORDER. Contests select from a shared pool in order, so earlier contests get
first refusal on scarce material. The order is:

  1. descending TOP PRIZE
  2. descending implied field size
  3. descending dollars at risk (entries x fee)
  4. original file/row order

Top prize leads because that is what a marginal improvement in lineup quality is
worth: a $50K-to-1st contest and a $2K-to-1st contest can have similar field
sizes while the first is worth 25x more per lineup. Field size is the scarcity
term and belongs as the tiebreak rather than the primary — a large field wants
the thin high-ceiling/low-ownership corner (0.63-0.67% of real entries), while a
small field wants the abundant chalky region, so ordering by size alone hands
scarce material to contests that do not want it.

Top prize comes from the RESOLVED payout structure, never from the contest name.
The name's leading token is the prize POOL ("MLB $20K Four-Seamer"), and the
bracketed "[$25K to 1st]" is not what dk_entries parses — so the structure is
both the correct source and an exact one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as _field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContestSlot:
    """One contest from the entries files, with its resolved ladder."""
    contest_id: str
    contest_name: str
    entry_fee: float
    n_entries: int
    structure: dict
    payout_arr: np.ndarray
    field_size: int
    is_approximate: bool
    first_seen: int                      # original file/row order, for stability
    entries: list = _field(default_factory=list)   # [(file_path, EntryRecord)]
    # Prize pool advertised in the contest NAME, in dollars ("MLB $8K Chin
    # Music" -> 8000.0) -- the quantity `_resolve` matches size variants on,
    # and the only statement of what the contest actually pays that does not
    # come from a table. None when the name carried no "$<n>[K|M]" token. Kept
    # distinct from `prize_pool` (the RESOLVED table's sum) precisely so a
    # borrowed ladder can be told apart from the contest it was borrowed for.
    advertised_pool: Optional[float] = None

    @property
    def top_prize(self) -> float:
        return float(self.payout_arr.max()) if self.payout_arr.size else 0.0

    @property
    def prize_pool(self) -> float:
        return float(self.payout_arr.sum())

    @property
    def dollars_at_risk(self) -> float:
        return self.entry_fee * self.n_entries

    @property
    def implied_field_size(self) -> float:
        """Field implied by the contest's own economics, not the table's."""
        return self.prize_pool / self.entry_fee if self.entry_fee else float("inf")

    def sort_key(self) -> tuple:
        # Negated for descending; first_seen ascending last so ties are stable.
        return (-self.top_prize, -self.implied_field_size,
                -self.dollars_at_risk, self.first_seen)


def resolve_contest_slots(
    all_file_entries: list[tuple[Any, list]],
    field_size_override: int = 0,
) -> list[ContestSlot]:
    """Group entries by contest, resolve each ladder, return them in fill order.

    Entries with a blank contest name are skipped: real DK entry files carry
    trailing instruction/blank rows, and treating those as a contest would
    invent a phantom slot with no ladder.
    """
    from src.optimization.payout import (
        nearest_payout_structure, payout_table_to_array, structure_for_contest,
    )

    def _resolve(name: str, pool_dollars: Optional[float],
                 implied: Optional[int]) -> tuple[dict, bool]:
        """Pick the right SIZE VARIANT of a registered contest.

        DK runs the same contest at several sizes and advertises the pool in
        the name ("MLB $20K Four-Seamer" vs the $15K version), so the pool is
        the discriminator and it is stated exactly. Matching instead on
        entries back-solved as pool/fee is wrong by the rake: a $15,000 pool at
        a $1 entry needs ~17,600 entries at 15% rake, not 15,000, and that gap
        picked the wrong variant for 5 of 6 contests on a real entries file --
        a $15K mini-MAX resolved to a $12,000 table, a $20K Four-Seamer to
        $15,000. Every objective here is denominated in dollars, so the ladder's
        magnitude is not cosmetic.

        Falls back to nearest-by-entries when the name matches no registered
        contest, which `nearest_payout_structure` already flags as approximate.
        """
        from src.optimization.payout import (
            CONTEST_STRUCTURES, _contest_structure_key, load_payout_structure,
        )
        key = _contest_structure_key(name)
        if key is not None and pool_dollars:
            best, best_gap = None, None
            for fk in CONTEST_STRUCTURES[key]:
                st = load_payout_structure(fk)
                gap = abs(float(payout_table_to_array(st).sum()) - pool_dollars)
                if best_gap is None or gap < best_gap:
                    best, best_gap = st, gap
            if best is not None:
                # Within 1% of the advertised pool is the same contest; further
                # than that means no captured variant really matches it.
                return best, bool(best_gap > 0.01 * pool_dollars)
        st, approx = nearest_payout_structure(name, n_entries=implied)
        exact = structure_for_contest(name, n_entries=implied) is not None
        return st, bool(approx or not exact)

    groups: dict[str, dict] = {}
    idx = 0
    for file_path, records in all_file_entries:
        for rec in records:
            name = (getattr(rec, "contest_name", "") or "").strip()
            if not name:
                idx += 1
                continue
            key = str(getattr(rec, "contest_id", "") or name)
            g = groups.setdefault(key, {
                "name": name,
                "fee": float(getattr(rec, "entry_fee_cents", 0) or 0) / 100.0,
                "first_seen": idx,
                "entries": [],
            })
            g["entries"].append((file_path, rec))
            idx += 1

    slots: list[ContestSlot] = []
    for key, g in groups.items():
        n = len(g["entries"])
        # The contest's own implied field, used to pick between same-name size
        # variants (DK runs e.g. Bat Flip at both 9,803 and 11,437 entries).
        implied = None
        rec0 = g["entries"][0][1]
        pool_c = getattr(rec0, "prize_pool_cents", None)
        fee_c = getattr(rec0, "entry_fee_cents", None)
        if pool_c and fee_c:
            implied = int(pool_c / fee_c)
        if field_size_override:
            implied = int(field_size_override)
        pool_dollars = (float(pool_c) / 100.0) if pool_c else None
        struct, approx = _resolve(g["name"], pool_dollars, implied)
        arr = payout_table_to_array(struct)
        slots.append(ContestSlot(
            contest_id=key, contest_name=g["name"], entry_fee=g["fee"],
            n_entries=n, structure=struct, payout_arr=arr,
            field_size=int(struct.get("total_entries", 0) or 0),
            is_approximate=bool(approx),
            first_seen=g["first_seen"], entries=g["entries"],
            advertised_pool=pool_dollars,
        ))

    slots.sort(key=lambda s: s.sort_key())
    return slots


def describe_slots(slots: list[ContestSlot]) -> str:
    """Human-readable fill order — logged per run because first pick is worth
    something and the ordering should be visible rather than implicit."""
    lines = [
        f"{'#':>2}  {'contest':<40} {'entries':>7} {'fee':>7} "
        f"{'top prize':>10} {'field':>8}  table"
    ]
    for i, s in enumerate(slots, 1):
        lines.append(
            f"{i:>2}  {s.contest_name[:40]:<40} {s.n_entries:>7,} "
            f"${s.entry_fee:>6,.2f} ${s.top_prize:>9,.0f} {s.field_size:>8,}  "
            f"{s.structure.get('name', '?')}"
            + ("  [APPROXIMATE]" if s.is_approximate else "")
        )
    return "\n".join(lines)


def select_per_contest_multi_arm(
    slots: list[ContestSlot],
    n_candidates: int,
    make_field,
    prepare_fn,
    arm_fns: dict,
    exclude_used: bool = True,
    progress=None,
    contest_done=None,
    narrow_fn=None,
    stop_check=None,
) -> tuple[dict, dict]:
    """Fill every contest for every selection arm, contests on the OUTER loop.

    ({arm_id: {contest_id: [candidate index, ...]}}, {contest_id: diagnostics}).

    WHY CONTESTS ARE THE OUTER LOOP. Everything expensive here is a function of
    the CONTEST alone, not the arm: building the contest's sorted field
    ((n_sims x F) float32, ~1.8 GB and a full sort at a 17,835-entry contest)
    and running the rank->payout kernel over the shortlist. The arms then differ
    only in how they read the resulting matrix. Running arms on the outside --
    the obvious shape, since the sweep is one portfolio per arm -- rebuilds all
    of that once per arm: at six contests and six arm keys that is 36 field
    builds instead of 6, which measured out to tens of minutes of pure
    recomputation. `prepare_fn` is therefore called once per contest and its
    result shared across every arm.

    Each arm keeps its OWN `used` set, so arms never constrain each other -- an
    arm's portfolio must be internally disjoint across contests (see
    `select_per_contest`), but two arms are separate candidate portfolios and
    are expected to overlap.

    `prepare_fn(field, slot) -> ctx` builds the shared per-contest context.
    `arm_fns[arm_id](ctx, avail_idx, k, slot) -> [idx]` returns indices into the
    candidate pool, drawn from `avail_idx`.
    `narrow_fn(ctx, avail_idx, k, slot) -> avail_idx` optionally trims the
    candidate set THIS contest is chosen from, after the used-set exclusion and
    before any arm runs. It is applied identically to every arm, so the arms
    still see one common menu and remain comparable. None disables it.
    `contest_done(index, slot)` fires once per contest AFTER every arm has
    filled it -- the only point at which that contest's cost is fully realized,
    which is what a caller driving a progress bar needs.

    `stop_check()` is polled at every boundary that precedes real work: before a
    contest's field build, again once `prepare_fn` has returned (that call is
    the minutes-long half of a contest and can only report a stop by finishing
    early with a matrix nobody should read), and before each arm fills. A stop
    RETURNS what is already filled rather than raising -- the caller flattens a
    partial portfolio and the run reports `stopped`, which is the contract every
    other stage honours. A contest that never ran contributes no picks at all,
    so the positional flatten downstream truncates there instead of sliding a
    later contest's lineups onto its entries.
    """
    say = progress or (lambda m: None)
    stopped = stop_check if stop_check is not None else (lambda: False)
    used: dict = {arm: set() for arm in arm_fns}
    out: dict = {arm: {} for arm in arm_fns}
    diag: dict[str, dict] = {}
    M = int(n_candidates)
    for i, slot in enumerate(slots, 1):
        if stopped():
            say(f"stop requested — {i - 1} of {len(slots)} contests filled")
            break
        say(f"[{i}/{len(slots)}] {slot.contest_name} — {slot.n_entries} entries, "
            f"field {slot.field_size:,}, top ${slot.top_prize:,.0f}, "
            f"{M:,} candidates shortlisted")
        ctx = prepare_fn(make_field(slot.field_size), slot)
        if stopped():
            # The field sort and the payout kernel are the expensive half of a
            # contest, and a Stop clicked inside them is only observable once
            # they return. Drop the context unread: whatever it holds was cut
            # short, so pricing an arm against it would pick lineups off a
            # half-built matrix.
            del ctx
            say(f"stop requested during contest {i}/{len(slots)} — "
                f"{i - 1} contests filled")
            break
        for arm, fn in arm_fns.items():
            if stopped():
                break
            u = used[arm]
            avail = (
                np.array([j for j in range(M) if j not in u], dtype=np.int64)
                if exclude_used else np.arange(M, dtype=np.int64)
            )
            if narrow_fn is not None:
                avail = narrow_fn(ctx, avail, slot.n_entries, slot)
            if len(avail) < slot.n_entries:
                raise RuntimeError(
                    f"arm {arm!r}: contest {slot.contest_name!r} needs "
                    f"{slot.n_entries} lineups but only {len(avail)} remain "
                    f"unused of {M}. Widen the shortlist."
                )
            picks = [int(p) for p in fn(ctx, avail, slot.n_entries, slot)]
            picks = picks[:slot.n_entries]
            if exclude_used:
                u.update(picks)
            out[arm][slot.contest_id] = picks
        diag[slot.contest_id] = {
            "contest_name": slot.contest_name,
            "n_entries": slot.n_entries,
            "field_size": slot.field_size,
            "entry_fee": slot.entry_fee,
            "top_prize": slot.top_prize,
            "prize_pool": slot.prize_pool,
            "table_name": str(slot.structure.get("name", "?")),
            "n_available": int(M - min(len(u) for u in used.values())) if used else M,
            "pool_consumed_pct": (
                100.0 * max(len(u) for u in used.values()) / M if used and M else 0.0
            ),
            "is_approximate": slot.is_approximate,
        }
        del ctx
        if contest_done is not None:
            contest_done(i, slot)
    return out, diag


def select_per_contest(
    slots: list[ContestSlot],
    candidates: list,
    cand_scores: np.ndarray,
    make_field,
    select_fn,
    exclude_used: bool = True,
    progress=None,
) -> tuple[dict, dict]:
    """Select each contest's own slice against its OWN ladder, in fill order.

    ({contest_id: [candidate index, ...]}, {contest_id: diagnostics}).

    Contests draw from one shared candidate pool, so `exclude_used` makes the
    slices disjoint: a lineup entered in two contests concentrates risk with no
    diversification benefit, and measured on 08/26 rebuilding the second
    contest's portfolio under that constraint raised P(at least one contest
    profits) by 4.1 points and P(at least one doubles) by 2.9, at 5.7% of mean
    EV. Order therefore matters -- earlier contests get first refusal -- which
    is why `resolve_contest_slots` sorts by top prize.

    `cand_scores` is (M, S) on the shared sim worlds and is CONTEST-INDEPENDENT:
    only the field and the ladder change per contest, so the expensive candidate
    scoring is done once by the caller and only the field build plus a rank->
    payout pass is repeated.

    `make_field(field_size)` returns this contest's field, in whatever form
    `select_fn` consumes -- `SharedFieldPool.fields` returns a GENERATOR of
    (S, F) ascending-sorted fields so only one is ever resident. It is passed
    through opaquely; nothing here inspects it.

    `select_fn(field, payout_arr, avail_idx, k, slot) -> [idx]` returns indices
    INTO `candidates`, drawn from `avail_idx`.

    Single-arm wrapper over `select_per_contest_multi_arm` so the fill-order and
    disjointness rules have exactly one implementation.
    """
    picks, diag = select_per_contest_multi_arm(
        slots,
        len(candidates),
        make_field,
        prepare_fn=lambda field, slot: field,
        arm_fns={"_": lambda ctx, avail, k, slot: select_fn(
            ctx, slot.payout_arr, avail, k, slot,
        )},
        exclude_used=exclude_used,
        progress=progress,
    )
    return picks["_"], diag


def assign_per_contest(slots: list[ContestSlot], picks: dict, candidates: list,
                       evs=None) -> dict:
    """{file_path: [(EntryRecord, Lineup), ...]} — contest C's lineups onto
    contest C's entries.

    Replaces the global cross-contest sort in `assign_lineups_to_entries`, which
    ranks ONE portfolio by strength and gives the strongest lineup to the
    easiest contest. That is the right rule when a single portfolio is spread
    across contests; once each contest has selected its own slice against its
    own ladder there is nothing left to rank across contests, and keeping both
    rules would have them silently fight.
    """
    result: dict = {}
    for slot in slots:
        idxs = picks.get(slot.contest_id, [])
        for (file_path, rec), j in zip(slot.entries, idxs):
            lu = candidates[j]
            result.setdefault(file_path, []).append((rec, lu))
    return result


# ---------------------------------------------------------------------------
# Per-contest field and payout machinery
# ---------------------------------------------------------------------------

class SharedFieldPool:
    """One raw opponent-field pool, subsampled down to each contest's size.

    WHY SUBSAMPLE RATHER THAN REGENERATE. Field entries are i.i.d. draws from
    the same projected-ownership model, so a random F-subset of an N-entry field
    IS a valid F-entry field -- there is nothing about a 234-entry contest that
    makes its entrants differently distributed from a 17,835-entry one under
    this model. Regenerating per contest would cost a full `generate_field` per
    contest (the largest block in the field phase) and buy only independent
    noise.

    It also buys COMMON RANDOM NUMBERS, which is the part that matters for the
    thing this whole path exists to show. The measured ~19-45 point ownership
    spread between a small and a large contest is a real response to the payout
    ladder; if each contest drew its own field, part of that spread would be
    field-draw noise and the two would be inseparable. Subsets are NESTED (a
    fixed permutation per sample, first n taken), so a small contest's field is
    literally a subset of a large one's and the ladder is the only thing that
    differs.

    Raw field arrays are (N, 10) player-id integers -- ~1.4 MB at N=17,835, so
    holding them across the whole selection phase is free. It is the SCORED,
    per-world-sorted field that is expensive ((n_sims x F) float32, ~1.8 GB at
    25k sims x 17,835), which is why `fields()` is a generator: exactly one
    sorted field exists at a time.
    """

    def __init__(self, raw_fields: list, sort_fn, seed: int = 0) -> None:
        self._raw = [np.asarray(r) for r in raw_fields]
        self._sort_fn = sort_fn
        self.capacity = min((r.shape[0] for r in self._raw), default=0)
        self._perm = [
            np.random.default_rng([seed, k]).permutation(r.shape[0])
            for k, r in enumerate(self._raw)
        ]

    @property
    def n_samples(self) -> int:
        return len(self._raw)

    def fields(self, field_size: int):
        """Yield each sample's (n_sims, F) ascending-sorted field, one at a time."""
        n = int(min(field_size, self.capacity))
        if n <= 0:
            raise ValueError(f"field pool is empty (capacity={self.capacity})")
        for raw, perm in zip(self._raw, self._perm):
            sub = raw if n >= raw.shape[0] else raw[np.sort(perm[:n])]
            yield self._sort_fn(sub)


def scale_dupes_for_field(e_dupes: np.ndarray, field_size: int) -> np.ndarray:
    """(M,) float32 top-band payout scale 1/(1+E[dupes]) at THIS contest's size.

    The dupe GLM is fitted with a `log(n_entries / DUPE_REF_FIELD_SIZE)` offset,
    so its prediction is E[copies] in a reference-size contest and E[copies] is
    linear in field size. `ContestScorer` never rescales because its whole run
    is one contest; this path spans contests two orders of magnitude apart in
    size, where using the unscaled value would overstate duplication in a
    234-entry contest by ~60x. See `gpp_portfolio.DUPE_REF_FIELD_SIZE`.
    """
    from src.optimization.gpp_portfolio import DUPE_REF_FIELD_SIZE
    scaled = np.asarray(e_dupes, dtype=np.float64) * (
        float(field_size) / DUPE_REF_FIELD_SIZE
    )
    return (1.0 / (1.0 + np.maximum(scaled, 0.0))).astype(np.float32)


def contest_payout_matrix(
    cand_scores: np.ndarray,
    fields,
    payout_arr: np.ndarray,
    entry_fee: float,
    e_dupes: Optional[np.ndarray] = None,
    dupe_min_gross_payout: float = 15.0,
    cand_chunk: int = 2_000,
    stop_check=None,
) -> np.ndarray:
    """(M, S) NET dollars per candidate per sim world against ONE contest.

    Runs the production scoring kernel (`_compute_payout_from_sorted_field`) --
    exact tie splitting, the same band-averaged lookup, the same dupe dilution
    -- rather than a second rank->payout implementation, so a candidate's EV
    here is denominated identically to the funnel's `robust_payout`. Only the
    ladder, the field size and the fee change.

    `fields` is an iterable of (n_sims, F) ascending-sorted fields, consumed
    lazily and averaged: peak memory is one sorted field plus the (M, S) output,
    never K of either. Chunked over candidates for the same reason.

    `stop_check()` is polled per candidate chunk, and a stop returns the matrix
    PART-BUILT: at production scale this loop is the minutes-long block a Stop
    click most often lands in, and the only useful thing to do with it then is
    abandon it. Callers must poll stop_check themselves on return and discard
    the result -- returning early beats raising, which would turn a user Stop
    into an error path in every caller.
    """
    from src.optimization.gpp_portfolio import (
        _build_dilutable_lookup, _build_payout_lookup, _payout_cumsum,
        _compute_payout_from_sorted_field,
    )
    cand_scores = np.ascontiguousarray(cand_scores, dtype=np.float32)
    M, S = cand_scores.shape
    out = np.zeros((M, S), dtype=np.float32)
    payout_cumsum = None
    dilute_cumsum = None
    dupe_scale = None
    width = None
    n_k = 0
    for field_sorted in fields:
        F = int(field_sorted.shape[1])
        if width is None:
            width = F
            payout_cumsum = _payout_cumsum(
                _build_payout_lookup(payout_arr, N=F, entry_fee=entry_fee)
            )
            if e_dupes is None:
                dilute_cumsum = np.zeros_like(payout_cumsum)
                dupe_scale = np.ones(M, dtype=np.float32)
            else:
                dilute_cumsum = _payout_cumsum(_build_dilutable_lookup(
                    payout_arr, N=F, min_gross_payout=dupe_min_gross_payout,
                ).astype(np.float32))
                dupe_scale = scale_dupes_for_field(e_dupes, F)
        elif F != width:
            # The lookups are built once against the first field's width; a
            # differently-sized field would be scored against the wrong ladder
            # banding and silently misprice every candidate.
            raise ValueError(
                f"field width changed mid-contest ({width} -> {F}); every "
                "sample of one contest must be subsampled to the same size"
            )
        for c0 in range(0, M, cand_chunk):
            if stop_check is not None and stop_check():
                return out
            c1 = min(c0 + cand_chunk, M)
            out[c0:c1] += _compute_payout_from_sorted_field(
                np.ascontiguousarray(cand_scores[c0:c1]),
                field_sorted,
                payout_cumsum,
                dilute_cumsum,
                np.ascontiguousarray(dupe_scale[c0:c1]),
            )
        n_k += 1
        del field_sorted
    if n_k == 0:
        raise ValueError("contest_payout_matrix: no field samples were supplied")
    if n_k > 1:
        out /= n_k
    return out


def contest_beat_bits(
    cand_scores: np.ndarray,
    fields: list,
    quantile: float = 0.999,
) -> np.ndarray:
    """(M, K*ceil(S/8)) uint8 — packed "beat the field's `quantile`" world bits.

    The coverage selector partitions sim worlds by which candidates reach the
    money zone in them, so its bits must be derived from THIS contest's field:
    a 234-entry contest's p99.9 is a different bar from an 11,437-entry one's.

    Layout and threshold column match `ContestScorer._score_col_lineups`
    exactly (`ceil(q*N)-1`, `np.packbits` per field sample, samples
    concatenated along axis 1) so the selector cannot tell the two sources
    apart.
    """
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    M, S = cand_scores.shape
    n_bytes = (S + 7) // 8
    out = np.zeros((M, len(fields) * n_bytes), dtype=np.uint8)
    for k, field_sorted in enumerate(fields):
        col = int(np.ceil(quantile * field_sorted.shape[1])) - 1
        thr = field_sorted[:, max(col, 0)]                      # (S,)
        out[:, k * n_bytes:(k + 1) * n_bytes] = np.packbits(
            cand_scores >= thr[None, :], axis=1,
        )
    return out


def contest_ev_means(
    cand_scores: np.ndarray,
    fields,
    payout_arr: np.ndarray,
    entry_fee: float,
    e_dupes: Optional[np.ndarray] = None,
    dupe_min_gross_payout: float = 15.0,
    cand_chunk: int = 2_000,
    stop_check=None,
) -> np.ndarray:
    """(M,) mean net dollars per candidate against ONE contest.

    The mean-only sibling of `contest_payout_matrix`, for ranking rather than
    selection: it accumulates per-candidate means chunk by chunk and never
    holds an (M, S) payout matrix. Used to rank the whole pool against every
    contest's own ladder, which is the input the shortlist union needs.

    `stop_check` behaves exactly as in `contest_payout_matrix`: polled per
    candidate chunk, and a stop returns a part-built ranking for the caller to
    discard, never a raise.
    """
    from src.optimization.gpp_portfolio import (
        _build_dilutable_lookup, _build_payout_lookup, _payout_cumsum,
        _compute_payout_from_sorted_field,
    )
    cand_scores = np.ascontiguousarray(cand_scores, dtype=np.float32)
    M = cand_scores.shape[0]
    out = np.zeros(M, dtype=np.float64)
    payout_cumsum = dilute_cumsum = dupe_scale = None
    width = None
    n_k = 0
    for field_sorted in fields:
        F = int(field_sorted.shape[1])
        if width is None:
            width = F
            payout_cumsum = _payout_cumsum(
                _build_payout_lookup(payout_arr, N=F, entry_fee=entry_fee)
            )
            if e_dupes is None:
                dilute_cumsum = np.zeros_like(payout_cumsum)
                dupe_scale = np.ones(M, dtype=np.float32)
            else:
                dilute_cumsum = _payout_cumsum(_build_dilutable_lookup(
                    payout_arr, N=F, min_gross_payout=dupe_min_gross_payout,
                ).astype(np.float32))
                dupe_scale = scale_dupes_for_field(e_dupes, F)
        elif F != width:
            raise ValueError(
                f"field width changed mid-contest ({width} -> {F})"
            )
        for c0 in range(0, M, cand_chunk):
            if stop_check is not None and stop_check():
                return out / n_k if n_k else out
            c1 = min(c0 + cand_chunk, M)
            out[c0:c1] += _compute_payout_from_sorted_field(
                np.ascontiguousarray(cand_scores[c0:c1]),
                field_sorted, payout_cumsum, dilute_cumsum,
                np.ascontiguousarray(dupe_scale[c0:c1]),
            ).mean(axis=1)
        n_k += 1
        del field_sorted
    if n_k == 0:
        raise ValueError("contest_ev_means: no field samples were supplied")
    return out / n_k


def union_shortlist(ev_by_contest: list, cap: int) -> np.ndarray:
    """Shortlist indices: each contest's own top candidates, round-robin.

    WHY A UNION AND NOT A RANKING. Cutting the pool by ONE contest's mean EV is
    a composition bias, not a threshold -- measured on a real 15k pool,
    Spearman(mean EV, ownership) = +0.356 against a 5,945-entry reference, and
    a plain top-4,000 cut took 118 of 1,500 from the least-owned decile against
    603 from the ninth. That 5:1 skew runs toward chalk, which is precisely the
    material a 17,835-entry contest does not want, so the contests that differ
    most from the reference are the ones starved.

    Ownership stratification fixes that, but by assuming ownership IS the axis
    contests disagree on. Contests can just as easily diverge on stack shape,
    salary usage or correlation structure, and a summed-ownership band is blind
    to all three. Ranking every candidate against every contest's OWN ladder
    measures the disagreement instead of proxying it.

    Contests take turns contributing their next-best candidate not already
    chosen, which is self-balancing in exactly the right way: when contests
    agree the union grows slowly and each one gets deeper coverage of the same
    material, and when they disagree it fills with each contest's own
    preferences -- which is when that matters most. Every contest contributes
    equally regardless of how many entries it holds, because the shortlist is a
    menu, not an allocation.
    """
    if not ev_by_contest:
        raise ValueError("union_shortlist: no per-contest rankings supplied")
    M = len(ev_by_contest[0])
    cap = int(min(cap, M))
    orders = [
        np.argsort(-np.asarray(ev, dtype=np.float64), kind="stable")
        for ev in ev_by_contest
    ]
    sel: set = set()
    ptr = [0] * len(orders)
    while len(sel) < cap:
        progressed = False
        for ci, order in enumerate(orders):
            p = ptr[ci]
            while p < M and int(order[p]) in sel:
                p += 1
            if p < M:
                sel.add(int(order[p]))
                ptr[ci] = p + 1
                progressed = True
                if len(sel) >= cap:
                    break
            else:
                ptr[ci] = M
        if not progressed:
            break
    return np.array(sorted(sel), dtype=np.int64)
