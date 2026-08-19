"""Tests for the runnable MRP entry point.

The contract that matters is that this is a DROP-IN for
`external_pool.allocate_contests`: same ExternalAllocation shape, portfolio and
entry_plan parallel, unfilled accounted for. If that holds, adopting or backing
out MRP is a one-line swap at a single call site.

The `preassigned` tests carry the most weight. In the live A/B both halves of a
contest are OURS and compete for the same prizes, so production's entries must
enter MRP's state as incumbents. Getting that wrong would flatter MRP by
exactly the self-competition term the whole build exists to add.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api.external_pool import ContestGroup, ExternalPool
from src.optimization.lineup import Lineup
from src.optimization.mrp.runner import MRPConfig, allocate_marginal_reward
from src.simulation.results import SimulationResults

TEAMS = [("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC"),
         ("EEE", "FFF"), ("FFF", "EEE")]
# Two of each infield slot and four outfielders per team. A thinner roster
# (one of each) makes ContestSimulator's stacked sampler reject nearly every
# attempt -- a random 5+3 team split almost never covers C/1B/2B/3B/SS/OFx3 --
# which quietly produced a 2-lineup field and made these tests vacuous.
ROSTER = ("P", "P", "C", "C", "1B", "1B", "2B", "2B", "3B", "3B",
          "SS", "SS", "OF", "OF", "OF", "OF")


def _players_df():
    rows, pid = [], 0
    for team, opp in TEAMS:
        for pos in ROSTER:
            pid += 1
            rows.append({
                "player_id": pid, "name": f"p{pid}", "position": pos,
                "eligible_positions": [pos], "team": team, "opponent": opp,
                "game": f"{team}@{opp}", "salary": 3000 + (pid % 7) * 300,
                "mean": 8.0 + (pid % 11) * 0.6, "std_dev": 5.0,
                "ownership": 5.0 + (pid % 9),
            })
    return pd.DataFrame(rows)


def _legal_lineups(df, n, rng):
    """n DK-legal lineups: 2 P from one game, 8 hitters from another game."""
    out, seen = [], set()
    p_ids = df[(df.position == "P") & (df.team.isin(["AAA", "BBB"]))].player_id.tolist()
    hit = df[(df.position != "P") & (df.team.isin(["CCC", "DDD"]))]
    for _ in range(n * 60):
        if len(out) >= n:
            break
        pids = list(rng.choice(p_ids, size=2, replace=False))
        for pos, cnt in (("C", 1), ("1B", 1), ("2B", 1), ("3B", 1), ("SS", 1), ("OF", 3)):
            avail = hit[hit.position == pos].player_id.tolist()
            pids += list(rng.choice(avail, size=cnt, replace=False))
        key = frozenset(int(p) for p in pids)
        if len(key) == 10 and key not in seen:
            seen.add(key)
            out.append(Lineup(player_ids=[int(p) for p in pids]))
    return out


def _fixture(n_pool=40, n_sims=300, seed=0):
    rng = np.random.default_rng(seed)
    df = _players_df()
    pids = df.player_id.tolist()
    mat = rng.normal(df["mean"].to_numpy(), 5.0, size=(n_sims, len(pids)))
    sim = SimulationResults(player_ids=pids, results_matrix=mat)
    lineups = _legal_lineups(df, n_pool, rng)
    pool = ExternalPool(lineups=lineups, contests={}, source_paths=[],
                        n_dropped_unknown_players=0, n_dropped_duplicates=0,
                        n_dropped_near_duplicates=0)
    return df, sim, pool


def _group(cid, name, k, fee_cents=400, pool_cents=400_000):
    return ContestGroup(
        contest_id=cid, contest_name=name, entry_fee_cents=fee_cents,
        prize_pool_cents=pool_cents, single_entry_tag=False,
        entries=[(f"/tmp/{cid}.csv", f"entry{j}") for j in range(k)],
    )


CFG = MRPConfig(field_pool_size=400, max_sims_per_contest=300, seed=1)


def test_returns_a_drop_in_external_allocation():
    df, sim, pool = _fixture()
    groups = [_group("c1", "Four-Seamer", 5), _group("c2", "Base Hit", 4)]

    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)

    assert len(alloc.portfolio) == len(alloc.entry_plan), "must stay parallel"
    assert len(alloc.portfolio) + len(alloc.unfilled) == 9, "every purchased slot accounted for"
    for lu, delta in alloc.portfolio:
        assert isinstance(lu, Lineup) and len(lu.player_ids) == 10
        assert np.isfinite(delta)
    assert diag.total_reward >= 0
    assert len(diag.per_contest) == 2
    assert "Four-Seamer" in diag.summary()


def test_entries_are_drawn_from_the_right_contest_in_order():
    df, sim, pool = _fixture()
    groups = [_group("c1", "Four-Seamer", 3), _group("c2", "Base Hit", 2)]
    alloc, _ = allocate_marginal_reward(pool, df, sim, groups, CFG)

    files = [e[0] for e in alloc.entry_plan]
    assert files.count("/tmp/c1.csv") == 3
    assert files.count("/tmp/c2.csv") == 2


def test_no_lineup_is_used_in_two_contests_by_default():
    df, sim, pool = _fixture()
    groups = [_group("c1", "Four-Seamer", 5), _group("c2", "Base Hit", 5)]
    alloc, _ = allocate_marginal_reward(pool, df, sim, groups, CFG)

    keys = [frozenset(lu.player_ids) for lu, _ in alloc.portfolio]
    assert len(keys) == len(set(keys))


def test_preassigned_consumes_slots_and_never_reuses_those_lineups():
    df, sim, pool = _fixture()
    groups = [_group("c1", "Four-Seamer", 6)]
    pre = {"c1": [0, 1]}

    alloc, _ = allocate_marginal_reward(pool, df, sim, groups, CFG, preassigned=pre)

    assert len(alloc.portfolio) == 4, "two slots were already spent by the other arm"
    picked = {id(lu) for lu, _ in alloc.portfolio}
    assert id(pool.lineups[0]) not in picked
    assert id(pool.lineups[1]) not in picked


def test_preassigned_entries_are_treated_as_competing_incumbents():
    """The A/B correctness property. Committing the other arm's entries must
    LOWER the marginal value MRP sees, because those entries take prize mass
    it would otherwise have had. If dR came back unchanged, production's half
    was being ignored and the comparison would be rigged in MRP's favour."""
    df, sim, pool = _fixture()
    groups = [_group("c1", "Four-Seamer", 3)]

    alone, _ = allocate_marginal_reward(pool, df, sim, groups, CFG)
    with_pre, _ = allocate_marginal_reward(
        pool, df, sim, [_group("c1", "Four-Seamer", 6)], CFG,
        preassigned={"c1": list(range(3))},
    )

    best_alone = alone.portfolio[0][1]
    best_after = with_pre.portfolio[0][1]
    assert best_after < best_alone, (
        f"incumbents did not reduce marginal value ({best_after} vs {best_alone})"
    )


def test_unfilled_is_reported_when_the_pool_cannot_cover_the_slots():
    df, sim, pool = _fixture(n_pool=6)
    groups = [_group("c1", "Four-Seamer", 20)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)

    assert len(alloc.unfilled) > 0
    assert diag.n_unfilled == len(alloc.unfilled)
    assert len(alloc.portfolio) <= 6


def test_empty_pool_returns_everything_unfilled():
    df, sim, pool = _fixture()
    pool.lineups = []
    groups = [_group("c1", "Four-Seamer", 4)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)
    assert alloc.portfolio == [] and len(alloc.unfilled) == 4


def test_gamma_in_is_enforced_end_to_end():
    df, sim, pool = _fixture(n_pool=60)
    groups = [_group("c1", "Four-Seamer", 5)]
    cfg = MRPConfig(field_pool_size=400, max_sims_per_contest=300, seed=1, gamma_in=6)
    alloc, _ = allocate_marginal_reward(pool, df, sim, groups, cfg)

    sets = [set(lu.player_ids) for lu, _ in alloc.portfolio]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert len(sets[a] & sets[b]) <= 6


def test_world_capping_does_not_change_the_interface():
    df, sim, pool = _fixture(n_sims=600)
    groups = [_group("c1", "Four-Seamer", 4)]
    cfg = MRPConfig(field_pool_size=300, max_sims_per_contest=150, seed=1)
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, cfg)
    assert len(alloc.portfolio) == 4
    assert diag.per_contest[0]["field_size"] > 0


# ---------------------------------------------------------------------------
# Publishing to the UI's Portfolio tab
# ---------------------------------------------------------------------------

def _entry_rec(cid, name, j):
    from src.api.dk_entries import EntryRecord
    return EntryRecord(entry_id=f"e{cid}{j}", contest_name=name, contest_id=cid,
                       entry_fee_cents=400, entry_fee_raw="$4",
                       prize_pool_cents=400_000, slot_players=[])


def _real_group(cid, name, k, tmp_path):
    f = tmp_path / f"{cid}DKEntries.csv"
    f.write_text("stub")
    return ContestGroup(
        contest_id=cid, contest_name=name, entry_fee_cents=400,
        prize_pool_cents=400_000, single_entry_tag=False,
        entries=[(f, _entry_rec(cid, name, j)) for j in range(k)],
    )


def test_publish_writes_what_the_portfolio_tab_reads(tmp_path):
    """GET /api/portfolio/sweep serves exactly one path and validates the slate
    fingerprint, and PortfolioTable hides the entry column unless upload_tag is
    set -- so this pins the whole contract in one place."""
    from src.api.slate_exclusions import compute_file_fingerprint
    from src.optimization.mrp.runner import publish_portfolio

    df, sim, pool = _fixture()
    df = df.copy()
    df["slot"] = 1
    groups = [_real_group("c1", "MLB $20K Four-Seamer [20 Entry Max]", 3, tmp_path),
              _real_group("c2", "MLB $10K Base Hit [Single Entry]", 2, tmp_path)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)

    salaries = tmp_path / "DKSalaries.csv"
    salaries.write_text("Name,ID\nx,1\n")
    res = publish_portfolio(alloc, diag, df, salaries, tmp_path)

    payload = json.loads((tmp_path / "portfolio_sweep_draftkings.json").read_text())
    assert payload["slate_fingerprint"] == compute_file_fingerprint(salaries)
    assert payload["mode"] == "marginal_reward", "must be distinguishable from production"
    assert payload["ev_type"] == "delta_reward"
    assert len(payload["sweep"]) == 1

    lineups = payload["sweep"][0]["lineups"]
    assert len(lineups) == len(alloc.portfolio) == res["n_lineups"]
    for i, lr in enumerate(lineups):
        # The four fields the tab needs to render lineup -> contest.
        assert lr["upload_tag"], "PortfolioTable hides the entry column without this"
        assert lr["entry_fee"] == "$4"
        assert lr["contest_name"]
        assert lr["entry_sort_order"] == i
        assert len(lr["players"]) == 10

    names = {lr["contest_name"] for lr in lineups}
    assert len(names) == 2, "both contests must be represented and labelled"


def test_publish_also_writes_the_csv_the_other_endpoint_reads(tmp_path):
    """Two endpoints serve the portfolio: /api/portfolio/sweep reads the JSON,
    /api/portfolio?platform= reads portfolio_<platform>.csv via
    _load_portfolio_from_csv. Writing only one leaves them disagreeing."""
    import pandas as _pd

    from src.optimization.mrp.runner import publish_portfolio

    df, sim, pool = _fixture()
    df = df.copy()
    df["slot"] = 1
    groups = [_real_group("c1", "MLB $20K Four-Seamer [20 Entry Max]", 3, tmp_path)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)
    salaries = tmp_path / "DKSalaries.csv"
    salaries.write_text("Name,ID\nx,1\n")

    res = publish_portfolio(alloc, diag, df, salaries, tmp_path)

    csv_path = Path(res["csv_path"])
    assert csv_path.exists()
    got = _pd.read_csv(csv_path)
    assert "lineup" in got.columns, "_load_portfolio_from_csv groups by 'lineup'"
    assert got["lineup"].nunique() == len(alloc.portfolio)


def test_publish_backs_up_an_existing_production_portfolio(tmp_path):
    """That file is production's shipped portfolio and is later graded as
    'production'; overwriting it without a copy would destroy real-money
    provenance."""
    from src.optimization.mrp.runner import publish_portfolio

    df, sim, pool = _fixture()
    df = df.copy()
    df["slot"] = 1
    groups = [_real_group("c1", "MLB $20K Four-Seamer [20 Entry Max]", 2, tmp_path)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)

    existing = tmp_path / "portfolio_sweep_draftkings.json"
    existing.write_text(json.dumps({"mode": "external", "sweep": [{"risk": 1.0, "lineups": []}]}))
    salaries = tmp_path / "DKSalaries.csv"
    salaries.write_text("Name,ID\nx,1\n")

    res = publish_portfolio(alloc, diag, df, salaries, tmp_path)

    assert res["backup_paths"], "no backup was taken"
    sweep_backup = next(b for b in res["backup_paths"] if b.endswith(".json"))
    backup = json.loads(Path(sweep_backup).read_text())
    assert backup["mode"] == "external", "the production payload must survive intact"
    assert json.loads(existing.read_text())["mode"] == "marginal_reward"


def test_publish_can_be_told_not_to_back_up(tmp_path):
    from src.optimization.mrp.runner import publish_portfolio

    df, sim, pool = _fixture()
    df = df.copy()
    df["slot"] = 1
    groups = [_real_group("c1", "MLB $20K Four-Seamer [20 Entry Max]", 2, tmp_path)]
    alloc, diag = allocate_marginal_reward(pool, df, sim, groups, CFG)
    (tmp_path / "portfolio_sweep_draftkings.json").write_text("{}")
    salaries = tmp_path / "DKSalaries.csv"
    salaries.write_text("Name,ID\nx,1\n")

    res = publish_portfolio(alloc, diag, df, salaries, tmp_path, backup=False)
    assert res["backup_paths"] == []
