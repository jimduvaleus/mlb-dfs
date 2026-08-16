"""One-off: what portfolio would the LIVE topn_coverage selector have picked
had we max-entered ONLY the Bat Flip contest on the 08092026 archived slate
(150 entries -- Bat Flip's real DK per-user cap, confirmed by the user) and
no other contest, using current config.yaml settings -- then grade every
hypothetical entry against REAL fantasy points and print the top 10.

Mirrors src/api/pipeline.py's `elif _ev_type == "topn_coverage":` branch
line for line (field pool build, optional generated-pool augmentation,
allocate_contests_topn_coverage call) but against a single hand-built
ContestGroup instead of the slate's real `groups` list, since Bat Flip
wasn't actually entered that day (archive/08092026/portfolio_sweep_
draftkings.json's real sweep has no Bat Flip line item to pull `k` from).

Real per-player FPTS come from the existing post-contest-analysis path
(scripts/analyze_candidate_pool.py::load_contest_player_fpts, which reads
archive/<slate>/contest_player_fpts.json directly) rather than
tests/bt_core.py's stricter verify_slate/load_real_contests, which
currently can't even load this slate (no payout table exactly matches Bat
Flip's real 10,294-entry field -- that machinery also isn't needed here
anyway, since allocate_contests_topn_coverage never reads a payout table).

Usage
-----
    source venv/bin/activate
    python scripts/simulate_bat_flip_max_entry.py
"""
import csv
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.api import external_pool as ep  # noqa: E402
from src.api.dk_entries import EntryRecord  # noqa: E402
from src.api.pipeline import PipelineRunner  # noqa: E402
from src.ingestion.dk_slate import DraftKingsSlateIngestor  # noqa: E402
from src.models.copula import EmpiricalCopula  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.simulation.results import SimulationResults  # noqa: E402
from analyze_candidate_pool import load_contest_player_fpts  # noqa: E402

SLATE = "08092026"
CONTEST_ID = "bat-flip"
CONTEST_NAME = "Bat Flip"
MAX_ENTRIES = 150  # real DK per-user cap for this contest (confirmed, not derived)

SIM_CACHE_DIR = PROJECT_ROOT / "outputs" / "bat_flip_sim_cache"
SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class _SingleContestGroup:
    """Minimal ep.ContestGroup stand-in -- allocate_contests_topn_coverage
    only reads .contest_id/.entries and (via implied_field_size) .prize_
    pool_cents/.entry_fee_cents. Mirrors scripts/stress_test_topn_coverage_
    memory.py's _StressGroup, the established template for this."""
    def __init__(self, contest_id: str, contest_name: str, n_field: int, k: int):
        self.contest_id = contest_id
        self.contest_name = contest_name
        self.entry_fee_cents = 100
        self.prize_pool_cents = int(round(n_field * 100 * (1 - ep._DK_RAKE)))
        self.single_entry_tag = False
        self.roi_key = ""
        self.roi_fallback = True
        self.entries = [
            (Path("x/Entries.csv"), _make_entry(contest_id, contest_name, f"e{i}"))
            for i in range(k)
        ]


def _make_entry(contest_id: str, contest_name: str, entry_id: str) -> EntryRecord:
    return EntryRecord(
        entry_id=entry_id, contest_name=contest_name, contest_id=contest_id,
        entry_fee_cents=100, entry_fee_raw="$1", prize_pool_cents=None,
    )


def _real_field_size(archive_dir: Path) -> int:
    """True Bat Flip field size straight from the standings zip -- same
    'count real entry rows' logic tests/bt_core.py::load_real_contests
    uses, without that function's payout-table match guard (which fails
    for this slate -- no captured table exactly matches a 10,294 field)."""
    with zipfile.ZipFile(archive_dir / "bat-flip.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        rows = list(csv.reader(zf.read(name).decode("utf-8-sig", errors="replace").splitlines()))
    body = [r for r in rows[1:] if r and r[0].strip().isdigit()]
    return len(body)


def main() -> None:
    archive_dir = PROJECT_ROOT / "archive" / SLATE
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    gpp_cfg = cfg["gpp"]
    sim_cfg = cfg["simulation"]
    paths_cfg = cfg["paths"]

    print(f"slate={SLATE} contest={CONTEST_NAME} max_entries={MAX_ENTRIES}")
    print(f"external_pool_ev_type={gpp_cfg.get('external_pool_ev_type')}")

    # --- Load real slate inputs, exactly like pipeline.py::_run_external ---
    found = ep.discover_external_files(str(archive_dir))
    if not found["lineups_paths"] or not found["projections_path"]:
        raise SystemExit(f"no lineups_*.csv / projections CSV pair in {archive_dir}")

    slate_df = DraftKingsSlateIngestor(str(archive_dir / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}
    pid_to_name = dict(zip(slate_df["player_id"].astype(int), slate_df["name"]))
    pid_to_pos = dict(zip(slate_df["player_id"].astype(int), slate_df["position"]))

    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    print(f"external pool: {len(pool.lineups):,} lineups "
          f"({pool.n_dropped_near_duplicates:,} near-duplicates removed)")

    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )
    print(f"players_df: {len(players_df):,} players")

    # --- Simulate (cached across re-runs of this script) --------------------
    n_sims = int(sim_cfg.get("n_sims", 25_000))
    cache_path = SIM_CACHE_DIR / f"{SLATE}_{n_sims}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
        print(f"simulation loaded from cache ({n_sims:,} sims)")
    else:
        t0 = time.time()
        copula = EmpiricalCopula(str(PROJECT_ROOT / paths_cfg["copula"]))
        grids = ep.build_quantile_grids(
            proj_ext,
            zero_inflate=bool(gpp_cfg.get("external_pool_zero_inflate", False)),
            scratch_prob=float(gpp_cfg.get("external_pool_scratch_prob", 0.02)),
            mean_calib_batter=float(gpp_cfg.get("external_pool_mean_calib_batter", 1.0)),
            mean_calib_pitcher=float(gpp_cfg.get("external_pool_mean_calib_pitcher", 1.0)),
        )
        engine = SimulationEngine(
            copula, players_df, batter_pca_model=None, score_grid=None, quantile_grids=grids,
        )
        sim_results = engine.simulate(n_sims)
        np.savez_compressed(
            cache_path,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )
        print(f"simulated {n_sims:,} worlds in {time.time() - t0:.1f}s")

    proj_scores = ep.compute_pool_proj_scores(pool.lineups, players_df)

    # --- The single hypothetical contest: max-entered, nothing else ---------
    n_field = _real_field_size(archive_dir)
    print(f"Bat Flip real field size: {n_field:,}")
    group = _SingleContestGroup(f"{SLATE}:{CONTEST_ID}", CONTEST_NAME, n_field, MAX_ENTRIES)

    # --- Mirrors pipeline.py's topn_coverage branch verbatim ----------------
    _topn_field_pool_size = int(gpp_cfg.get(
        "external_pool_topn_field_pool_size", ep._TOPN_FIELD_POOL_CAP,
    ))
    _topn_rank = int(gpp_cfg.get("external_pool_topn_rank", 10))
    _topn_percentile_floor = float(gpp_cfg.get("external_pool_topn_percentile_floor", 0.001))
    _topn_field_samples = int(gpp_cfg.get("external_pool_topn_field_samples", 5))
    _topn_sims_fraction = float(gpp_cfg.get("external_pool_topn_sims_per_contest_fraction", 0.5))
    _topn_sims_min = int(gpp_cfg.get("external_pool_topn_sims_min", 0))
    _topn_sims_ref = float(gpp_cfg.get("external_pool_topn_sims_reference_field_size", 0.0))
    _topn_sims_power = float(gpp_cfg.get("external_pool_topn_sims_power", 0.0))
    _proj_score_floor_pct = float(gpp_cfg.get("external_pool_proj_score_pct", 0.0))
    _rng_seed = int(gpp_cfg.get("rng_seed") or 42)

    own_vec = players_df["ownership"].astype(float).to_numpy()

    print(f"building opponent field pool ({_topn_field_pool_size:,} lineups)...")
    t0 = time.time()
    field_lineups = ep.build_topn_field_pool(players_df, own_vec, _topn_field_pool_size, _rng_seed)
    print(f"  {field_lineups.shape[0]:,} lineups, {time.time() - t0:.1f}s")

    pool_for_alloc = pool
    is_generated = None
    proj_scores_for_alloc = proj_scores
    _topn_generated_pool_size = int(gpp_cfg.get("external_pool_topn_generated_pool_size", 0))
    if _topn_generated_pool_size > 0:
        n_real = len(pool.lineups)
        t0 = time.time()
        pool_for_alloc, generated_kept = ep.augment_topn_pool_with_generated(
            pool, players_df, own_vec, _topn_generated_pool_size, _rng_seed + 1,
        )
        is_generated = np.zeros(len(pool_for_alloc.lineups), dtype=bool)
        is_generated[n_real:] = True
        proj_scores_for_alloc = ep.compute_pool_proj_scores(pool_for_alloc.lineups, players_df)
        print(f"pool augmented: +{len(generated_kept):,} generated candidates "
              f"(of {_topn_generated_pool_size:,} requested) -> "
              f"{len(pool_for_alloc.lineups):,} total, {time.time() - t0:.1f}s")

    print(f"filling {MAX_ENTRIES} entries for {CONTEST_NAME}...")
    t0 = time.time()

    def _progress(info: dict) -> None:
        if info.get("event") == "contest_done":
            print(f"  filled {info['n_filled']}/{info['k']} "
                  f"({info['n_relaxations']} relaxations, "
                  f"{info['n_generated_picks']} generated picks) "
                  f"in {info['elapsed_s']:.1f}s")

    alloc = ep.allocate_contests_topn_coverage(
        pool_for_alloc, sim_results, [group], field_lineups,
        proj_scores=proj_scores_for_alloc,
        proj_score_floor_percentile=_proj_score_floor_pct,
        topn_rank=_topn_rank, topn_percentile_floor=_topn_percentile_floor,
        field_samples=_topn_field_samples,
        sims_per_contest_fraction=_topn_sims_fraction,
        sims_min=_topn_sims_min, sims_reference_field_size=_topn_sims_ref,
        sims_power=_topn_sims_power, rng_seed=_rng_seed,
        is_generated=is_generated, progress_cb=_progress,
    )
    print(f"total allocation time: {time.time() - t0:.1f}s")
    if alloc.unfilled:
        print(f"WARNING: {len(alloc.unfilled)} entries left unfilled (pool exhausted)")

    # --- Grade against real results ------------------------------------------
    fpts_map = load_contest_player_fpts(archive_dir)
    graded = []
    for lu, pick_ev in alloc.portfolio:
        actual = sum(fpts_map.get(int(p), float("nan")) for p in lu.player_ids)
        graded.append((lu, pick_ev, actual))
    n_gradeable = sum(1 for _, _, a in graded if not np.isnan(a))
    print(f"graded {n_gradeable}/{len(graded)} hypothetical entries "
          f"(ungradeable = a rostered player missing from contest_player_fpts.json)")

    graded_valid = [g for g in graded if not np.isnan(g[2])]
    graded_valid.sort(key=lambda g: -g[2])

    print(f"\nTop 10 hypothetical {CONTEST_NAME} entries by REAL fantasy points "
          f"(of {MAX_ENTRIES} max-entered):")
    print(f"{'rank':>4}  {'actual_fpts':>11}  {'pick_ev':>9}  lineup")
    for i, (lu, pick_ev, actual) in enumerate(graded_valid[:10], 1):
        roster = " / ".join(
            f"{pid_to_pos.get(int(p), '?')} {pid_to_name.get(int(p), int(p))}"
            for p in lu.player_ids
        )
        print(f"{i:>4}  {actual:>11.2f}  {pick_ev:>9.0f}  {roster}")


if __name__ == "__main__":
    main()
