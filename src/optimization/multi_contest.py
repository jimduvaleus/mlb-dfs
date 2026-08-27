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

    `make_field(field_size) -> (S, F)` ascending-sorted field scores.
    `select_fn(cand_payout, field_sorted, payout_arr, avail_idx, k) -> [idx]`
    returns indices INTO `candidates`, drawn from `avail_idx`.
    """
    say = progress or (lambda m: None)
    used: set[int] = set()
    out: dict[str, list[int]] = {}
    diag: dict[str, dict] = {}
    M = len(candidates)
    for i, slot in enumerate(slots, 1):
        avail = np.array([j for j in range(M) if j not in used], dtype=np.int64) \
            if exclude_used else np.arange(M, dtype=np.int64)
        if len(avail) < slot.n_entries:
            raise RuntimeError(
                f"contest {slot.contest_name!r} needs {slot.n_entries} lineups but "
                f"only {len(avail)} remain unused of {M}. Widen the shortlist or "
                "raise portfolio size."
            )
        say(f"[{i}/{len(slots)}] {slot.contest_name} — {slot.n_entries} entries, "
            f"field {slot.field_size:,}, top ${slot.top_prize:,.0f}, "
            f"{len(avail):,}/{M:,} candidates available")
        field_sorted = make_field(slot.field_size)
        picks = select_fn(field_sorted, slot.payout_arr, avail, slot.n_entries,
                          slot)
        picks = [int(p) for p in picks][:slot.n_entries]
        if exclude_used:
            used.update(picks)
        out[slot.contest_id] = picks
        diag[slot.contest_id] = {
            "contest_name": slot.contest_name,
            "n_entries": slot.n_entries,
            "field_size": slot.field_size,
            "top_prize": slot.top_prize,
            "n_available": int(len(avail)),
            "pool_consumed_pct": 100.0 * len(used) / M,
            "is_approximate": slot.is_approximate,
        }
        del field_sorted
    return out, diag


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
