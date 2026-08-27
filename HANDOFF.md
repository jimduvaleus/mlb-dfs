# Handoff — contest-aware portfolio selection (updated 2026-08-27)

Delete this file once the work is merged. It exists because several *negative*
results here are worth more than the code, and none of them are recoverable from
reading it.

## State: seven commits + uncommitted wiring

Seven commits on `parallel-shape-mutants`; the per-contest **wiring** described
under "Wiring, as built" is in the working tree, not yet committed.
**1,249 tests pass**, `tsc` clean.

    e1f3825  Fix three silent defects in candidate generation
    f3eace3  Add E[max] selector, fast portfolio builder, per-contest selection
    c29863f  Make GPP selection contest-aware; add Kelly/dR/E[max] arms
    92b27a0  Surface selection arms and contest identity in the UI
    475cb44  Add the measurement harnesses, and this note
    ae896bf  Correct the sweep-dimension note
    67ac4bc  Correct HANDOFF's own state section

`e1f3825` stands alone and is worth keeping regardless of whether the rest
lands. Note `config.yaml` is gitignored — new keys live in
`config.example.yaml`, and any machine with an older `config.yaml` picks up
defaults from `models.py`, not from the file. That now includes the four
`gpp.per_contest_*` keys.

## What was built

**New objectives, wired into the pipeline sweep.** `gpp.selector_mode` is now a
comma list (`kelly,dr,emax,coverage`, default `kelly,dr`), rendered as
checkboxes in ConfigForm. Sweep keys double as arm ids: det 1-5, kelly 11-15,
coverage 23, E[max] 31, dR 41 — `PortfolioTable.armLabel()` maps them to names,
keep it in sync with `pipeline.py`.

**Real contest ladders.** The GPP path hardcoded `dk_classic_gpp_5001` — one
5,001-entry curve for every contest. `gpp.contest_structure` /
`contest_field_size` now resolve the real ladder, and an unregistered contest
gates the run through the existing `PayoutFallbackDialog` (payload shape matches
`mrp_payout_fallback`, one dialog serves both).

**Per-contest selection** — `src/optimization/multi_contest.py`, now WIRED into
`pipeline._run` (see "Wiring, as built" below). Groups an entries file by
contest, resolves each ladder, orders them, and selects each contest's own
disjoint slice.

**Fixes found along the way** (each was silent, each found by checking an
invariant rather than by anything failing):
- `CandidateGenerator.generate` produced 1.97% duplicate lineups — now
  `dedupe=True` by default. Any pre-08/26 replay baseline used a pool with
  duplicates; `dedupe=False` reproduces it.
- `generate_shape_mutants` returned **0** mutants when `eligible_positions` was
  a string rather than a list — strictly worse than omitting the column (104).
  Fixed by `normalize_eligible_positions` in `lineup.py`.
- `generate_sim_optimal_lineups` opened 20 CBC threads on 8 physical cores.
  Measured 743/218/142/139/139 ms per solve at 1/4/8/16/20 workers — flat past
  the physical count. Now capped via `physical_cores()`.
- Payout structure resolution matched size variants by back-solved entries,
  which ignores rake — wrong for 5 of 6 real contests (a $15K mini-MAX resolved
  to a $12,000 table). Now matches on the **advertised pool**, which the contest
  name states exactly.
- Ladder bands and DK tie-splitting were hardcoded to a 670-payer contest.
- `n_bytes = W // 8` under-allocated the coverage bitset for any world count not
  divisible by 8.

## Settled findings

Four contests, two slates: Main Event Warm Up (3,335 @ $333), Bat Flip
(10,170 @ $17), Hot Corner (594 @ $3, 5-max), Skipper (234 @ $25, 1-max).

- **Self-competition objectives beat ownership-ranked and correlation selection
  by 12-24 points of portfolio ROI**, consistently across all four.
- **Kelly / dR / coverage are statistically tied** (median per-arm sd 0.45 ME /
  1.13 BF over 5 grading seeds; pairwise z < 2). Do not claim a winner.
  E[max] sits a rung below (z 2.4-3.0).
- **Contest-awareness falls out of the payout math**: ~19 points of ownership
  swing from a 235-entry to an 11,437-entry contest, unprompted. Per-contest
  selection reproduced a 45-point spread across six real contests in one run.
- **Coverage is structurally contest-blind** (slope 0.00 across a 63x field-size
  range). It ties on large contests and was right on Hot Corner *by accident*.
  Treat with care on small fields.
- **Marginal grading is blind to self-competition** — 50 identical lineups grade
  **+7.6% marginal, -60.1% portfolio**. Always grade in portfolio mode
  (`scripts/portfolio_grading.py`).

## Negative results — do not redo these

- **Generation is not the constraint.** 2.5x the ILP anchors: **+0.18 pts**.
  Dropping all 7,850 mutants: **~0**. Ownership headroom was never binding.
- **Rank-denominated gate currency: -0.03 pts**, despite ranking lineups far
  more like the payout does (Spearman 0.94-0.97 vs 0.79-0.90, +13pt recall).
  Available as `--gate-currency rank`; it *does* help det-style selectors (+3.6
  to +9.1) which cannot price rank themselves.
- **Perfect ownership projection is worth ~+1.4 to +1.9 pts** (oracle field),
  against 16 points for the objective choice. Feeding realized ownership to the
  pool and gate as well is usually *worse*.
- **Anchor world deciles do not matter.** corr(decile, ceiling) = -0.097. A
  quiet world is quiet for everyone including the field, so a quiet-world
  optimum is not a low-ceiling lineup.

Pattern worth internalising: **two strong proxy improvements both converted to
nothing.** A 150-lineup portfolio draws from ~750 candidates above its own mean
ceiling — 5:1 coverage. Improving *what reaches* the objectives does not bind.

## Wiring, as built (2026-08-27)

Item 1 is done. Per-contest selection **replaces** the single-ladder path
whenever the slate has entry files; with none, the old path runs unchanged,
which is what keeps `replay_slate.py` reproducible (archives carry no
`*Entries.csv`). No feature flag — that was a deliberate call.

Four things turned out differently from the plan above, and each is the
interesting part:

1. **Contests had to become the OUTER loop.** `select_per_contest` iterates
   contests for one arm. Everything expensive is a function of the contest
   alone — the (n_sims x F) sorted field and the payout kernel over the
   shortlist — so running arms outside rebuilt all of it per arm: 36 field
   builds instead of 6 at six contests x six arm keys, tens of minutes of pure
   recomputation. `select_per_contest_multi_arm` now owns the fill loop and
   `select_per_contest` is a single-arm wrapper over it, so the fill-order and
   disjointness rules still have exactly one implementation. Each arm keeps its
   own `used` set: slices are disjoint *within* an arm, arms never constrain
   each other.
2. **One field pool, subsampled per contest**, not one field per contest
   (`SharedFieldPool`). Field entries are i.i.d. draws from the same ownership
   model, so a random F-subset IS a valid F-field; regenerating buys only
   independent noise at a full `generate_field` per contest. Subsets are
   NESTED, which matters more than the speed: with independent draws, part of
   the measured ownership spread between a small and a large contest would be
   field-draw noise and the two would be inseparable. Raw arrays are ~1.4 MB
   and held for the whole phase; only the scored, sorted field (~1.8 GB at 25k
   sims x 17,835) is rebuilt, one at a time, via a generator.
3. **The dupe model had to be rescaled per contest.** `DUPE_REF_FIELD_SIZE`'s
   own docstring says any caller applying the model across contests of
   different sizes must scale by `field_size / N_REF`, and `ContestScorer`
   never does because its whole run is one contest. Unscaled, a 234-entry
   contest's duplication was overstated ~60x. `scale_dupes_for_field`.
4. **The EV floor is off during per-contest selection.** `gpp.ev_floor` is a
   dollar threshold calibrated on the reference ladder; across contests from $1
   to $25 fees it is either trivially cleared or culls the entire pool. The
   funnel already gated on it twice, which is where a fee-denominated floor
   belongs. Same reasoning `fast_portfolio.py`'s arm wrappers already used.

Also settled along the way:

- **`portfolio_size` is no longer the per-contest count** — each contest takes
  exactly the entries the file gives it. `portfolio_size` now only bounds the
  funnel; `gpp.per_contest_shortlist` (default 4,000) bounds selection.
- **Assignment is positional.** `_assign_positional` replaces
  `assign_lineups_to_entries`, and the diversity reorder is skipped: it exists
  to line lineup strength up with entry FEE across a fee-heterogeneous slate,
  and within a contest every entry has the same fee while across contests each
  slice was already chosen for its own ladder.
- **Contest identity rides on the entry map**, which is merged into every
  lineup row live, across the whole sweep, and again on restore from disk — so
  the UI groups per contest with no new plumbing and it survives a restart.
- **`replace_lineup` swaps in place** on this path instead of appending and
  reordering. Caveat now stated in the code: the replacement is ranked on the
  reference ladder, because the contest's field is freed by then. It is a
  like-for-like swap, not a re-selection.
- Two guards fall back rather than crash after the whole funnel has run: a
  contest whose resolved table has no entry count disables the path entirely,
  and a pool too small to fill every entry disjointly turns disjointness off
  for that run with a loud warning.

**The sweep stayed ONE-dimensional**, as the corrected note predicted. A
"portfolio" is and always was "the set of lineups across all contests we will
submit"; the old flow spread a single portfolio across contests too, via
`assign_lineups_to_entries`. Per-contest selection does not create a new object,
it just makes each contest's slice actually chosen for that contest. So
`activate_sweep_risk` works unchanged (still worth renaming to `activate_arm`,
cosmetic only).

**Display.** The arm buttons show the ownership RANGE across contests, not the
mean (with the per-contest breakdown in the tooltip); a summary table above the
cards gives one row per contest plus the headline spread; the card list is
divided by contest. The `complete`/`stopped` SSE payloads carry a `contests`
list.

**Tests: 1,249 pass** (was 1,202), `tsc` clean. 47 new across
`tests/test_multi_contest.py` (fill order, disjointness, arm independence,
field-pool nesting, dupe scaling, payout-matrix invariants) and
`tests/test_per_contest_pipeline.py` (every arm end-to-end through a real
`ContestScorer` and real ladders, entry order, positional assignment, the
summary).

## Open work, in order

1. **First real UI run.** Still not done, and still the gap that matters:
   `replay_slate.py` cannot exercise any of this, because no archive has both
   `market_odds_projections.csv` and a `*Entries.csv`. Everything above is
   covered by tests against real ladders and a real `ContestScorer`, which is
   not the same as having run it once at production scale. Watch RSS on the
   first run — the per-contest peak is one sorted field (~1.8 GB at 25k sims
   and a 17,835-entry contest) plus the shortlist payout matrix.
2. **Multi-slate validation.** Everything still rests on two slates.
   `replay_slate.py --all --variants` is what would turn any of this into a
   design principle — and note it exercises the SINGLE-LADDER path, so it
   validates the funnel, not the per-contest selection.
3. **Cross-contest risk** — measured, not built. Rebuilding contest B under an
   exact-duplicate constraint raised P(at least one contest profits) **+4.1
   pts** and P(at least one doubles) **+2.9**, costing 5.7% of mean EV.
   `gamma_out` below 9 buys nothing beyond the duplicate rule.
   `per_contest_disjoint` (default on) already delivers this within a run.
4. **Cross-contest overlap reporting.** `overlap_profile` /
   `cluster_decomposition` mean different things within versus across contests:
   within, overlap is self-competition; across, entries never compete, so it is
   bankroll variance — which is why `gamma_out` is documented as "NOT an EV
   rule". Those helpers live only in `scripts/analyze_rival_portfolio.py`, not
   in the pipeline or UI, so nothing currently conflates the two — but nothing
   reports the cross-contest figure either.

## Gotchas

- Coverage silently yields nothing if `gpp.compute_tail_metrics` is off. On the
  per-contest path it recomputes its own beat-p99.9 bits from each contest's
  field (`contest_beat_bits`) rather than reusing the fresh pass's, so it does
  not depend on `retain_beat999_worlds` there — but it still needs a field, so
  it forces the sorted field to be materialized rather than streamed.
- dR is the only arm with real memory cost — it retains the sorted field
  (~0.6-1.2 GB) the pipeline otherwise frees; `gpp.dr_shortlist` caps it. On the
  per-contest path `gpp.dr_shortlist` is NOT the cap — `per_contest_shortlist`
  is, and it applies to every arm.
- Per-contest peak memory is one sorted field (~1.8 GB at 25k sims and a
  17,835-entry contest) plus the shortlist payout matrix, one contest at a time.
  `per_contest_field_samples > 1` does not multiply that (samples are streamed),
  but it does multiply the field build and the per-contest score+sort.
- Any new `gpp.*` config key must land in **three** places (`config.yaml`,
  `models.py`, `types.ts`) or Pydantic drops it silently on save.
- Grading noise: gaps under ~1.3 pts (ME) / ~3.2 pts (BF) are unresolved at 40k
  sims. Use `scripts/regrade_portfolios.py` to size it before believing a gap.
