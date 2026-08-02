"""Shared core for the real-money backtest harness.

Everything in here was lifted verbatim out of tests/backtest.py so that other
tools can import it -- backtest.py parses sys.argv at import time and
SystemExit's, so it can never be imported. The functions themselves are
unchanged; the only difference is that values backtest.py held in module
globals (N_SIMS, SIM_CACHE_DIR, SHARPNESS, the diversity-corr sweep list) are
now explicit keyword arguments, so a caller with different settings can't
silently inherit backtest.py's.

New here, and the reason the extraction was worth doing:

  * `grade_pool` -- the vectorized form of `grade_pick`. Grading every lineup
    in the pool (~10k) against every real contest, rather than only the ~150
    a strategy picked, turns strategy evaluation into an index lookup. See
    tests/backtest_oracle.py.

  * `accumulate_currencies` -- one chunked pass over the simulated worlds that
    produces EVERY field-relative currency at once (p_win, expected dollars
    under the contest's real payout table, P(cash), tail-band EV, P(beat the
    field's p99)) instead of compute_p_win's single one. The expensive part --
    turning (M, S) simulated lineup scores into per-world percentiles against
    the opponent field -- is paid once and shared.

REAL DATA LIMIT: DK stops serving contest standings roughly 10 days out, so
the archive can only ever grow forward. archive/07212026 has all 10 named
zips but every one of them is a 0-member file (the download was attempted too
late); it is unrecoverable, not merely unmapped. Eight slates is the ceiling
today.
"""
import csv as csv_mod
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config.yaml") as _f:
    LIVE_CFG = yaml.safe_load(_f)

# Slates with intact (non-empty) standings zips.
BACKTEST_SLATES = [
    "07222026", "07242026", "07252026", "07262026",
    "07282026", "07292026", "07302026", "07312026",
    "08012026",
]

# standings zip stem (lowercased, no extension) -> display key matching
# portfolio_sweep_draftkings.json's contest_name AND the payout registry's
# lowercased contest key (structure_for_contest lowercases internally).
ZIP_TO_CONTEST = {
    "4k-base-hit": "Base Hit", "6k-base-hit": "Base Hit",
    "10k-base-hit": "Base Hit", "base-hit": "Base Hit",
    "bat-flip": "Bat Flip", "chin-music": "Chin Music",
    "five-tool-player": "Five-Tool Player", "four-seamer": "Four-Seamer",
    "hot-corner": "Hot Corner", "mini-max": "mini-MAX",
    "pickoff": "Pickoff", "rally-cap": "Rally Cap",
    "relay-throw": "Relay Throw", "skipper": "Skipper",
    "solo-shot": "Solo Shot", "moonshot": "Moonshot",
    "knuckleball": "Knuckleball",
    "5k-base-hit": "Base Hit",
    # 07/25 archive: misnamed file, 2,972 entries matches four-seamer-payout.txt
    # exactly and no four-seamer.zip exists otherwise -- see git log f20550a.
    "sour-seamer": "Four-Seamer",
}


# ---------------------------------------------------------------------------
# Real-field / real-money grading
# ---------------------------------------------------------------------------

def load_real_contests(d: Path) -> list[dict]:
    """[{contest, contest_id, n_field, fee, sorted_scores, payout_arr}] --
    one entry per real standings zip in this slate's archive dir.

    `contest-standings-*.zip` is always excluded: on every slate checked
    it's a byte-identical (or, once, a stale same-night pre-stat-correction)
    duplicate of one of the named zips, kept around only because the
    Analyze Contest UI feature expects a file with that name to exist.
    """
    from src.optimization.payout import payout_table_to_array, structure_for_contest

    out = []
    for z in sorted(d.glob("*.zip")):
        if z.name.startswith("contest-standings"):
            continue
        stem = z.stem.lower()
        contest = ZIP_TO_CONTEST.get(stem)
        if contest is None:
            raise SystemExit(
                f"unmapped zip {z.name} in {d.name} -- add it to ZIP_TO_CONTEST. "
                "Silently skipping a contest would drop real entries from the backtest."
            )
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                raise SystemExit(
                    f"{z.name} in {d.name} is an empty/corrupt zip (0 members). "
                    "DK stops serving standings ~10 days out -- this slate is "
                    "unrecoverable, exclude it rather than grading a partial one."
                )
            rows = list(csv_mod.reader(
                zf.read(names[0]).decode("utf-8-sig", errors="replace").splitlines()
            ))
        body = rows[1:]
        scores = sorted(float(r[4]) for r in body if r and r[0].strip().isdigit())
        n_field = len(scores)
        struct = structure_for_contest(contest, n_entries=n_field)
        table_n = struct["total_entries"] if struct is not None else None
        # DK's payout tiers are fixed to the contest's DESIGNED/advertised
        # size at creation, not to how many people actually showed up that
        # day -- a guaranteed contest that under-fills still pays from the
        # full-size table (confirmed 2026-07-31: two 07/22 captures had the
        # table's OWN stated header size, not that day's real fill, and were
        # byte-identical to already-registered tables at the larger size).
        # So table_n >= n_field is expected and fine; only a real MISMATCH
        # (table smaller than the real field, or a big relative gap --
        # meaning nearest-match grabbed the wrong contest's table) is an error.
        gap = None if table_n is None else table_n - n_field
        if struct is None or gap < 0 or gap > max(50, round(0.05 * table_n)):
            raise SystemExit(
                f"no matching payout table for {contest!r} at real fill n={n_field:,} "
                f"(zip {z.name} in {d.name}, nearest table n={table_n}) -- "
                "capture and register the right one."
            )
        if gap > 0:
            print(f"    note: {contest} real fill {n_field:,} < designed table size "
                  f"{table_n:,} (gap {gap}, {100*gap/table_n:.1f}%) -- using the full-size "
                  "table, DK's payout tiers don't shrink for an under-filled guarantee",
                  flush=True)
        out.append({
            "contest": contest,
            "contest_id": f"{d.name}:{stem}",
            "n_field": n_field,
            "fee": struct["entry_fee"],
            "sorted_scores": np.array(scores, dtype=np.float64),
            "payout_arr": payout_table_to_array(struct),
        })
    return out


def resolve_duplicate_names(d: Path, real: list[dict], nm: pd.DataFrame) -> dict:
    """{player_id: fpts} for DKSalaries names shared by two real players that
    slate, where the pairing can be established beyond doubt.

    verify_slate drops both ids of any shared name because the standings
    zip's Player/FPTS side table carries no id column. That is the safe
    default, but it is often needlessly lossy: on 07/26 it discarded 520 pool
    lineups (6% of the pool, on the slate that decides the whole dollar
    comparison) and on 07/25 another 379.

    Two facts make the pairing recoverable:

      * Different contests on the same slate roster different players, so
        pooling every zip's side table gives the SET of distinct FPTS values
        the shared name produced -- e.g. 07/26's Max Muncy yields {0, 4}
        across the eleven contests even though most list only one row.
      * The projections export says who was actually in a lineup. A player
        with no batting `Order` did not play and therefore scored 0.

    So when exactly one id of a shared name was in a lineup and exactly one
    was not, and 0 is among the observed values, the non-playing id takes the
    0 and the playing id takes the other value. Anything less clear-cut --
    both played, neither played, more than two values -- resolves to nothing
    and those ids stay excluded, because assigning the wrong player's score
    would silently mis-grade real money.

    KNOWN LIMIT: if both players actually played and both scored nonzero,
    nothing in the archived files distinguishes them and those lineups stay
    ungradeable. Resolving that case needs an outside source (an MLB box
    score for the date). It has not come up yet: across all 8 slates the only
    shared names are Max Muncy (LAD/ATH) and Jose Fermin (STL/LAA), and in
    every instance either one of them was out or both produced the same
    score. 07/31 is the one unresolvable Muncy -- and no pool lineup that
    slate rosters either id, so it costs nothing.
    """
    from src.api import external_pool as ep

    dup_names = sorted(set(nm["Name"][nm["Name"].duplicated(keep=False)]))
    if not dup_names:
        return {}
    try:
        found = ep.discover_external_files(str(d))
        proj = ep.parse_player_projections(found["projections_path"])
    except Exception:
        return {}
    order_by_id = dict(zip(proj["player_id"].astype("Int64"), proj["order"]))

    # every distinct FPTS this name produced, pooled across all contests
    vals_by_name: dict[str, set] = {n: set() for n in dup_names}
    for c in real:
        z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv_mod.reader(
                zf.read(name).decode("utf-8-sig", errors="replace").splitlines()
            ))
        for r in rows[1:]:
            if len(r) > 10 and r[7].strip() in vals_by_name and r[10] not in ("", "FPTS"):
                vals_by_name[r[7].strip()].add(float(r[10]))

    out: dict = {}
    for name in dup_names:
        ids = [int(i) for i in nm.loc[nm["Name"] == name, "ID"]]
        vals = vals_by_name.get(name, set())
        if len(vals) == 1:
            # both ids produced the same number -- which is which cannot
            # matter, so this is safe regardless of who played.
            out.update({i: next(iter(vals)) for i in ids})
            continue
        if len(vals) != 2 or 0.0 not in vals or len(ids) != 2:
            continue
        played = [i for i in ids if pd.notna(order_by_id.get(i, pd.NA))]
        benched = [i for i in ids if i not in played]
        if len(played) == 1 and len(benched) == 1:
            out[benched[0]] = 0.0
            out[played[0]] = next(v for v in vals if v != 0.0)
    return out


def verify_slate(d: Path, real: list[dict], nm: pd.DataFrame) -> dict:
    """{player_id: actual_fpts} rebuilt fresh from every real zip's embedded
    per-player FPTS side table (never trusts a separately-archived
    contest_player_fpts.json -- that file was found stale once already,
    see git log). Every zip must agree with every other on every
    unambiguous player or this raises: a mismatch across INDEPENDENTLY
    downloaded real files means one of them is wrong (a DK retroactive
    stat correction between download times, most often) and grading
    against it would silently score real money wrong.

    DKSalaries names that map to more than one player_id that slate (two
    real MLB players who happen to share a name) can't be resolved from
    the zip's Player/FPTS side table alone (no ID column), so they are left
    out of the first pass. resolve_duplicate_names then adds back the ones
    that can be paired beyond doubt by pooling every contest's side table
    and consulting who was actually in a lineup; whatever it can't establish
    stays out, and our lineups rostering those ids drop out of grading (NaN
    actual_score) rather than risk crediting the wrong player's score.
    """
    dup_names = set(nm["Name"][nm["Name"].duplicated(keep=False)])
    id_by_name = {r.Name: str(r.ID) for r in nm.itertuples() if r.Name not in dup_names}
    merged: dict[str, float] = {}
    for c in real:
        z = d / f"{c['contest_id'].split(':', 1)[1]}.zip"
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv_mod.reader(
                zf.read(name).decode("utf-8-sig", errors="replace").splitlines()
            ))
        body = rows[1:]
        emb = {
            r[7].strip(): float(r[10]) for r in body
            if len(r) > 10 and r[7].strip() and r[10] not in ("", "FPTS")
        }
        for pname, fp in emb.items():
            pid = id_by_name.get(pname)
            if pid is None:
                continue
            if pid in merged and abs(merged[pid] - fp) >= 0.011:
                raise SystemExit(
                    f"{d.name}: real zips disagree on {pname} ({pid}) -- "
                    f"{merged[pid]} vs {fp} from {c['contest_id']}. "
                    "Independently downloaded real files should never differ; "
                    "investigate before trusting this slate."
                )
            merged[pid] = fp
    out = {int(k): v for k, v in merged.items()}
    # Recover the shared-name ids that CAN be paired beyond doubt; the rest
    # stay out, as before. See resolve_duplicate_names.
    out.update(resolve_duplicate_names(d, real, nm))
    return out


def grade_pick(actual_score: float, sorted_real: np.ndarray, payout_arr: np.ndarray):
    """(gross_$, rank) inserting our lineup as one more competitor in the
    real field, ties split evenly across the tie band (including us)."""
    n_field = len(sorted_real)
    right = int(np.searchsorted(sorted_real, actual_score, side="right"))
    left = int(np.searchsorted(sorted_real, actual_score, side="left"))
    n_above, n_tied = n_field - right, right - left
    rank = n_above + 1
    lo, hi = n_above, min(n_above + n_tied + 1, len(payout_arr))
    band = payout_arr[lo:hi] if lo < len(payout_arr) else np.array([])
    return (float(band.mean()) if len(band) else 0.0), rank


def grade_pool(actual_scores: np.ndarray, sorted_real: np.ndarray,
               payout_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `grade_pick` over a whole pool: (gross_$ (M,), rank (M,)).

    Identical semantics -- our lineup inserted as one more competitor in the
    real field, ties split evenly across the tie band including us -- computed
    for every candidate at once via a prefix sum over the payout array instead
    of a per-lineup slice-and-mean. NaN actual scores (ambiguous-name drops,
    see verify_slate) come back as gross=NaN / rank=-1 so callers can't grade
    them by accident.

    This is the substrate the whole fast evaluation rests on: with it, a
    strategy's realized dollars are an index lookup rather than a pipeline run.
    """
    a = np.asarray(actual_scores, dtype=np.float64)
    finite = np.isfinite(a)
    safe = np.where(finite, a, 0.0)

    n_field = len(sorted_real)
    L = len(payout_arr)
    right = np.searchsorted(sorted_real, safe, side="right")
    left = np.searchsorted(sorted_real, safe, side="left")
    n_above = n_field - right
    n_tied = right - left
    rank = n_above + 1

    cum = np.concatenate(([0.0], np.cumsum(payout_arr, dtype=np.float64)))
    lo = np.clip(n_above, 0, L)
    hi = np.clip(n_above + n_tied + 1, 0, L)
    width = hi - lo
    gross = np.where(width > 0, (cum[hi] - cum[lo]) / np.maximum(width, 1), 0.0)

    gross = np.where(finite, gross, np.nan)
    rank = np.where(finite, rank, -1)
    return gross, rank.astype(np.int64)


# ---------------------------------------------------------------------------
# Pipeline replication (real production functions only)
# ---------------------------------------------------------------------------

_PWIN_FIELD_CAP = 25_000


def build_slate_context(d: Path, seed: int, calibrated: bool, real: list[dict], *,
                        n_sims: int, sharpness: float, sim_cache_dir: Path,
                        want_hitter_corr: bool = False, want_corr: bool = True,
                        want_pwin: bool = True):
    """Everything needed to run every arm for one (slate, seed, calibration)
    combination: pool, players_df, sim_results, corr, proj_scores, p_win
    cull/select dicts (two-stage), and the per-contest real allocation
    (sizes/fees keyed to `real`, splitting a display name shared by more
    than one real contest that slate -- e.g. "Base Hit" covering both a
    $4K and $10K variant -- proportionally by real field size)."""
    from src.api import external_pool as ep
    from src.api.pipeline import PipelineRunner
    from src.ingestion.dk_slate import DraftKingsSlateIngestor
    from src.models.copula import EmpiricalCopula
    from src.optimization.contest import ContestSimulator
    from src.simulation.engine import SimulationEngine
    from src.simulation.results import SimulationResults

    raw_dir = str(d)
    found = ep.discover_external_files(raw_dir)
    if not found["lineups_paths"] or not found["projections_path"]:
        raise SystemExit(f"{d.name}: no lineups_*.csv / projections CSV pair found.")

    slate_df = DraftKingsSlateIngestor(str(d / "DKSalaries.csv")).get_slate_dataframe()
    valid_ids = {int(p) for p in slate_df["player_id"]}
    pool = ep.parse_lineup_pool(found["lineups_paths"], valid_ids, require_roi_blocks=False)
    if not pool.lineups:
        raise SystemExit(f"{d.name}: every lineup dropped (unknown player ids).")
    proj_ext = ep.parse_player_projections(found["projections_path"])
    pool_pids = {int(p) for lu in pool.lineups for p in lu.player_ids}
    players_df = ep.build_external_players_df(
        slate_df, proj_ext, pool_pids, PipelineRunner._derive_opponent,
    )

    copula = EmpiricalCopula(str(PROJECT_ROOT / LIVE_CFG["paths"]["copula"]))
    if calibrated:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=True, scratch_prob=0.02,
            mean_calib_batter=0.88, mean_calib_pitcher=1.0,
        )
    else:
        grids = ep.build_quantile_grids(
            proj_ext, zero_inflate=False, scratch_prob=0.0,
            mean_calib_batter=1.0, mean_calib_pitcher=1.0,
        )
    engine = SimulationEngine(copula, players_df, batter_pca_model=None,
                              score_grid=None, quantile_grids=grids)

    sim_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = sim_cache_dir / f"{d.name}_{n_sims}_{seed}_calib{calibrated}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            sim_results = SimulationResults(
                [int(p) for p in z["player_ids"]], z["results_matrix"].astype(np.float64),
            )
    else:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        sim_results = engine.simulate(n_sims)
        np.random.set_state(rng_state)
        np.savez_compressed(
            cache_path,
            player_ids=np.asarray(sim_results.player_ids, dtype=np.int64),
            results_matrix=sim_results.results_matrix.astype(np.float32),
        )

    lineup_scores = ep.compute_lineup_scores(pool.lineups, sim_results)
    # (M, M) float32 -- 400 MB at M=10k. Skippable for callers that only want
    # the currencies (the oracle build), which is the whole cost difference.
    corr = (ep.compute_pool_corr(pool.lineups, sim_results, scores=lineup_scores)
            if want_corr else None)
    proj_scores = ep.compute_pool_proj_scores(pool.lineups, players_df)

    hitter_corr = None
    if want_hitter_corr:
        hitter_corr = hitter_only_corr(pool, sim_results, players_df)

    # --- real per-contest sizes: split a shared display name proportionally
    # by real field size (e.g. "Base Hit" covering both a $4K and $10K
    # contest the same slate) ---
    import json
    sw = json.loads((d / "portfolio_sweep_draftkings.json").read_text())
    r1 = next(x for x in sw["sweep"] if x["risk"] == 1.0)
    display_sizes: dict[str, int] = {}
    for lu in r1["lineups"]:
        display_sizes[lu["contest_name"]] = display_sizes.get(lu["contest_name"], 0) + 1
    by_display: dict[str, list[dict]] = {}
    for c in real:
        by_display.setdefault(c["contest"], []).append(c)
    contests: list[dict] = []
    for display, n_total in display_sizes.items():
        variants = by_display.get(display)
        if not variants:
            continue  # no real zip for this display name -- can't grade it
        if len(variants) == 1:
            contests.append({**variants[0], "k": n_total})
            continue
        total_field = sum(v["n_field"] for v in variants)
        alloc = [int(round(n_total * v["n_field"] / total_field)) for v in variants]
        alloc[-1] += n_total - sum(alloc)  # fix rounding drift on the last one
        for v, k in zip(variants, alloc):
            if k > 0:
                contests.append({**v, "k": k})

    # --- p_win: two-stage winner's-curse guard, mirrors pipeline.py exactly ---
    own_vec = players_df["ownership"].astype(float).to_numpy()
    col_map = {int(p): i for i, p in enumerate(sim_results.player_ids)}
    n_half = n_sims // 2
    sims_A = sim_results.results_matrix[:n_half]
    sims_B = sim_results.results_matrix[n_half:2 * n_half]
    scores_A, scores_B = lineup_scores[:, :n_half], lineup_scores[:, n_half:2 * n_half]

    field_n = int(min(max(5_000, max(c["n_field"] for c in contests)), _PWIN_FIELD_CAP))
    cs = ContestSimulator()
    field_A = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed)
    field_B = cs.generate_field(players_df, own_vec, n_lineups=field_n, rng_seed=seed + 1)
    field_scores_A = cs.score_field(field_A, sims_A, col_map)
    field_scores_B = cs.score_field(field_B, sims_B, col_map)

    # Ground-truth field size (from the real zip), not an implied estimate --
    # strictly better information than production has at run time.
    exponents = {c["contest_id"]: max(1.0, sharpness * c["n_field"]) for c in contests}
    # The dominant cost of a context build is np.power(q, exponent) over
    # (M, n_sims) once per contest. accumulate_currencies produces the same
    # arrays (verified bit-identical) as a by-product of the pass it has to
    # make anyway, so callers using it skip this to avoid paying twice.
    p_win_cull = ep.compute_p_win(scores_A, field_scores_A, exponents) if want_pwin else None
    p_win_select = ep.compute_p_win(scores_B, field_scores_B, exponents) if want_pwin else None

    return dict(
        pool=pool, corr=corr, hitter_corr=hitter_corr, proj_scores=proj_scores,
        players_df=players_df, contests=contests,
        p_win_cull=p_win_cull, p_win_select=p_win_select,
        lineup_scores=lineup_scores, sim_results=sim_results,
        scores_A=scores_A, scores_B=scores_B,
        field_scores_A=field_scores_A, field_scores_B=field_scores_B,
        exponents=exponents,
    )


class _FakeGroup:
    """Minimal stand-in for ep.ContestGroup: allocate_contests's p_win
    branch only reads .contest_id and len(.entries). We know real per-
    contest entry counts from portfolio_sweep_draftkings.json directly
    (the pipeline's actual submitted counts -- overflow entries some days
    come from a second account and shouldn't be replicated here), so a
    dummy .entries list of the right length is sufficient and exact.

    Entries are tagged (contest_id, j) rather than bare ints: when a
    contest's pool is exhausted before it fills, allocate_contests reports
    the shortfall via a flat `unfilled` list with no group boundary
    markers -- globally-unique entries let run_arm trace an unfilled
    placeholder back to the contest it belongs to (see the partial-fill
    trap this guards against: an arm that silently drops entries must not
    look "better" per-entry just because it graded fewer of them)."""
    def __init__(self, contest_id: str, k: int):
        self.contest_id = contest_id
        self.entries = [(contest_id, j) for j in range(k)]


def weighted_proj_scores(ctx: dict, pitcher_weight: float) -> np.ndarray:
    """Pool proj-score sum with each pitcher's projected mean multiplied by
    `pitcher_weight` before summing (hitters unweighted) -- pitcher_weight=1.0
    reproduces ctx["proj_scores"] exactly. Motivated by a direct measurement
    against real outcomes across all 8 archived slates: pitcher projected
    mean correlates with real FPTS more than 2x as strongly as hitter
    projected mean does (r=0.415 vs 0.167; see memory
    project-cull-calibration-followup). The floor currently sums both
    equally, trusting a noisy hitter signal exactly as much as a much more
    reliable pitcher one -- this tests whether weighting the sum toward the
    more trustworthy half changes what the floor admits/culls."""
    from src.api import external_pool as ep

    if pitcher_weight == 1.0:
        return ctx["proj_scores"]
    players_df = ctx["players_df"]
    w = np.where(players_df["position"].to_numpy() == "P", pitcher_weight, 1.0)
    weighted_means = players_df["mean"].to_numpy(dtype=np.float64) * w
    tmp = players_df.copy()
    tmp["mean"] = weighted_means
    return ep.compute_pool_proj_scores(ctx["pool"].lineups, tmp)


def hitter_only_lineup_scores(pool, sim_results, players_df) -> np.ndarray:
    """(M, n_sims) lineup scores identical to ep.compute_lineup_scores, but
    with every pitcher's simulated points zeroed first -- pitcher marker is
    players_df["position"] == "P" (build_external_players_df), the same
    convention src/simulation/engine.py and ContestSimulator's
    _build_pos_pools use everywhere else. Zeroes sim columns and reuses the
    public compute_lineup_scores matmul unchanged (mirrors compute_pool_corr's
    own approach) rather than reimplementing the indicator matmul."""
    from src.api import external_pool as ep
    from src.simulation.results import SimulationResults

    pitcher_ids = set(int(p) for p in
                      players_df.loc[players_df["position"] == "P", "player_id"])
    hitter_matrix = sim_results.results_matrix.copy()
    pitcher_cols = [i for i, pid in enumerate(sim_results.player_ids)
                    if int(pid) in pitcher_ids]
    hitter_matrix[:, pitcher_cols] = 0.0
    hitter_sim = SimulationResults(sim_results.player_ids, hitter_matrix)
    return ep.compute_lineup_scores(pool.lineups, hitter_sim)


def hitter_only_corr(pool, sim_results, players_df) -> np.ndarray:
    """(M, M) correlation of hitter-only simulated lineup scores -- same
    shape/semantics as ep.compute_pool_corr's full-lineup matrix, restricted
    to the 9 hitter slots. Motivated by pitcher proj correlating with real
    FPTS 2.5x as strongly as hitter proj (r=0.415 vs 0.167, all 8 archived
    slates) -- if pitchers are the reliably-forecastable half, diversity
    effort should concentrate on the unpredictable hitter half instead of
    diversifying against the pitcher slot's contribution to the full sum."""
    from src.api import external_pool as ep

    scores = hitter_only_lineup_scores(pool, sim_results, players_df)
    return ep.compute_pool_corr(pool.lineups, sim_results, scores=scores)


# ---------------------------------------------------------------------------
# One-pass currency accumulator
# ---------------------------------------------------------------------------

def _quantile_bin_edges(contests: list[dict], extra_log_bins: int = 96) -> np.ndarray:
    """Bin edges in u = 1 - q (i.e. normalized rank, 0 = best) chosen so that
    every contest's payout tier boundary falls exactly on an edge.

    Expected dollars is then EXACT rather than discretized: within a bin the
    payout is constant for every contest by construction, so
    E[$] = sum_b P(u in bin_b) * amount_b. The extra log-spaced bins only
    sharpen the p_win/p_beat99 estimates, which do vary inside a bin.
    """
    edges = {0.0, 1.0}
    for c in contests:
        n = float(c["n_field"])
        pay = c["payout_arr"]
        # tier boundaries = every rank where the payout changes, in u-space.
        change = np.flatnonzero(np.diff(pay)) + 1  # 0-indexed rank of new value
        for r in change:
            edges.add(min(float(r) / n, 1.0))
        edges.add(min(float(len(pay)) / n, 1.0))
    # log-spaced refinement over the top of the field, where q**n and the
    # steep payout tiers both live.
    edges.update(np.geomspace(1e-6, 1.0, extra_log_bins).tolist())
    return np.unique(np.clip(np.array(sorted(edges), dtype=np.float64), 0.0, 1.0))


def accumulate_currencies(pool_scores: np.ndarray, field_scores: np.ndarray,
                          contests: list[dict], exponents: dict[str, float],
                          chunk: int = 1000) -> dict:
    """Every field-relative currency in one pass over the simulated worlds.

    `pool_scores` (M, S) and `field_scores` (S, F) are exactly what
    compute_p_win takes. Per world we compute q = the candidate's percentile
    against the opponent field, then accumulate:

      * `p_win[cid]`        -- mean(q ** exponents[cid]), bit-identical to
                               ep.compute_p_win (same clip, same chunking).
      * `hist` (M, B)       -- counts of u = 1 - q per bin, the sufficient
                               statistic from which every payout-shaped
                               currency below is derived exactly.
      * `ev_dollars[cid]`   -- E[gross $] under that contest's REAL payout
                               table. This is the objective the backtest
                               actually grades, and unlike p_win it has no
                               free parameter and adapts to each contest's
                               payout SHAPE (which prior work established is
                               a property of contest design, not field size).
      * `p_cash[cid]`       -- P(finish in a paying rank).
      * `ev_tail[cid]`      -- E[$ | only ranks paying >= 10x entry fee],
                               the steep band, blind to the min-cash plateau
                               that dominates mean EV.
      * `p_beat99`          -- P(q >= 0.99), contest-independent.
      * `sim_mean/std/p99`  -- field-blind moments, for diagnostics.

    Deriving the payout currencies from a shared histogram rather than a
    per-contest gather is what makes this affordable: the (M, S) percentile
    work is done once for all contests instead of once per contest.
    """
    M, S = pool_scores.shape
    F = field_scores.shape[1]
    edges = _quantile_bin_edges(contests)
    B = len(edges) - 1

    p_win_acc = {k: np.zeros(M, dtype=np.float64) for k in exponents}
    hist = np.zeros((M, B), dtype=np.float64)
    row_off = (np.arange(M, dtype=np.int64) * B)[:, None]
    n_done = 0

    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        c = s1 - s0
        fs_chunk = np.sort(field_scores[s0:s1], axis=1)          # (c, F)
        q = np.empty((M, c), dtype=np.float32)
        for j in range(c):
            q[:, j] = np.searchsorted(fs_chunk[j], pool_scores[:, s0 + j], side="left") / F
        np.clip(q, 0.5 / (F + 1), 1.0 - 0.5 / (F + 1), out=q)
        qd = q.astype(np.float64)
        for key, n_exp in exponents.items():
            p_win_acc[key] += np.power(qd, n_exp).sum(axis=1)

        u = 1.0 - qd
        bidx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, B - 1)
        flat = (bidx + row_off).ravel()
        hist += np.bincount(flat, minlength=M * B).reshape(M, B)
        n_done += c

    if n_done == 0:
        raise ValueError("accumulate_currencies: no worlds processed")
    for key in p_win_acc:
        p_win_acc[key] /= n_done
    prob = hist / n_done                                          # (M, B)

    # Bin -> payout, per contest. Every bin lies inside one tier by
    # construction (see _quantile_bin_edges), so evaluating at the bin's
    # lower edge is exact, not an approximation.
    lo_edges = edges[:-1]
    ev_dollars, p_cash, ev_tail = {}, {}, {}
    for c in contests:
        pay = c["payout_arr"]
        n = float(c["n_field"])
        rank_idx = np.clip((lo_edges * n).astype(np.int64), 0, len(pay) - 1)
        amt = np.where(lo_edges * n < len(pay), pay[rank_idx], 0.0)
        ev_dollars[c["contest_id"]] = prob @ amt
        p_cash[c["contest_id"]] = prob @ (amt > 0).astype(np.float64)
        tail_amt = np.where(amt >= 10.0 * c["fee"], amt, 0.0)
        ev_tail[c["contest_id"]] = prob @ tail_amt

    beat99 = prob @ (lo_edges < 0.01).astype(np.float64)
    return dict(
        p_win=p_win_acc, ev_dollars=ev_dollars, p_cash=p_cash, ev_tail=ev_tail,
        p_beat99=beat99, hist=prob, edges=edges,
        sim_mean=pool_scores.mean(axis=1).astype(np.float64),
        sim_std=pool_scores.std(axis=1).astype(np.float64),
        sim_p99=np.percentile(pool_scores, 99, axis=1).astype(np.float64),
    )
