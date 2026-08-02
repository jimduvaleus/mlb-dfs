# Prospective evaluation protocol (season program, started 2026-08-02)

The archive is the only instrument that can eventually resolve selection skill
(9 slates today; effects below ~±0.5pp on top-1% rate are invisible; dollar
ROI is decided by single outlier wins). It grows **only forward** — DraftKings
stops serving contest standings roughly **10 days** after a contest ends, so a
missed day is unrecoverable (this already cost slate 07/21). The protocol has
two halves: a daily capture checklist and a weekly walk-forward evaluation
with pre-registered rules.

## Daily capture checklist (per slate entered)

1. **Standings zips, per contest, named**: download every entered contest's
   standings ("Export lineups" zip) into `archive/MMDDYYYY/`, named after the
   contest (e.g. `four-seamer.zip`). The generic `contest-standings-*.zip` is
   NOT sufficient — `bt_core.load_real_contests` skips it. Deadline: within
   10 days, but do it next morning.
2. **Payout tables**: if a contest has no entry in `data/payout_structures/`
   (new contest type or new size variant), capture the payout page. The
   detector is `load_real_contests` raising SystemExit on the new slate —
   don't silence it; add the JSON (see `dk_relay_throw_9803.json` pattern)
   and register it in `src/optimization/payout.py` `CONTEST_STRUCTURES`.
3. **Slate inputs** (normally captured by the live run automatically —
   verify they exist): `DKSalaries.csv`, the SaberSim export
   (`MLB_*_DK_*.csv`), the candidate pool files (`lineups_*.csv`), and
   `portfolio_sweep_draftkings.json` (the shipped portfolio — the thing the
   backtest grades as "production").
4. **SaberSim ROI/Dupes blocks**: exports that include per-contest
   ROI/Win%/Sim Dupes columns are strictly more valuable (the dupe model can
   only be fitted on slates that have them — currently 4). If obtainable,
   prefer that export flavor.
5. **PPD slates**: if a game postpones after lock, still archive everything
   and note the PPD in a `NOTES.txt` in the slate dir; the slate is excluded
   from grading (realized scores are distorted) but inputs remain useful.

## Weekly walk-forward evaluation

1. Add new gradeable slates via the `BT_SLATES` env override (do not edit
   `bt_core.BACKTEST_SLATES` weekly; commit the constant only at milestones).
2. Build oracle tables for the new slates:
   `python tests/backtest_oracle.py <slates>` (then `field <slates>` if not
   auto-built), all three seeds.
3. `python tests/backtest_lab.py verify` — must be green before anything is
   read.
4. Run the standing arm set + audit refresh; outputs land in a dated
   `tests/backtest_output/evidence/YYYYMMDD/` directory.
5. Adjudicate ONLY with the pre-registered protocol: paired bootstrap
   $/entry, LOSO_min, drop_max, entry-weighted rate ladder, exact sign
   tests, 3 seeds (42/137/4242), joint grading (`grade_joint`) for headline
   numbers. Gates G1–G5 as recorded in the plan/evidence log.

## Mining control

**No new arm, currency, or hypothesis is run against the archive without a
dated entry in `EVIDENCE_LOG.md` first** (date, hypothesis, exact arm
config, which gates apply, what result would count as failure). ~100
retrospective configurations have already been exhausted on the current
9 slates; anything new mined from the same 9 slates carries near-zero
evidentiary weight. New evidence comes from new slates.

Promotion/demotion decisions are made by the user from the evidence
package; nothing auto-promotes.
