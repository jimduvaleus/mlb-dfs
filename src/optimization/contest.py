"""Contest EV evaluator for GPP portfolio post-processing.

Generates a simulated field of opponent lineups using heuristic ownership
probabilities, then computes per-lineup cash rate and EV stability metrics
(ev_gap: sim-variance check; field_gap: field-sampling check).
"""
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.optimization.lineup import Lineup, ROSTER_REQUIREMENTS, SALARY_CAP
from src.optimization.ownership import compute_heuristic_ownership

logger = logging.getLogger(__name__)

# DK roster slots in order
_SLOTS = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
_SLOT_ELIG = {
    "P": {"P"},
    "C": {"C"},
    "1B": {"1B"},
    "2B": {"2B"},
    "3B": {"3B"},
    "SS": {"SS"},
    "OF": {"OF"},
}
_MAX_HITTERS_PER_TEAM = 5
_MIN_GAMES = 2


def _load_field_calibration() -> dict:
    """Load calibration params from data/processed/contest_stats.json if present."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "contest_stats.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f).get("calibration_params", {})
    except Exception:
        return {}


_CAL = _load_field_calibration()
# P(4-batter primary stack | lineup is stacked). Calibrated from 10-slate contest
# analysis: real DK entries use 5-stacks ~73.5% of the time among stacked lineups.
_STACK_SIZE_4_PROB: float = float(_CAL.get("stack_size_4_prob", 0.265))
# P(secondary stack ≥2 | lineup is primary-stacked). Empirically ~82% of stacked
# lineups also have ≥2 batters from a second team.
_SECONDARY_STACK_PROB: float = float(_CAL.get("secondary_stack_prob", 0.822))
# P(secondary size == 2 | has secondary stack). Roughly 52.5% use 2-man secondary,
# the rest use 3-man.
_SECONDARY_SIZE_2_PROB: float = float(_CAL.get("secondary_size_2_prob", 0.525))
# Minimum total salary for a generated field lineup. Rejects lineups below the
# empirical p10 of real contest entries (~$49,200), rounded to the nearest $500.
_FIELD_SALARY_FLOOR: float = float(_CAL.get("salary_floor_field", 49_000))


def compute_emergent_ownership(
    players_df: pd.DataFrame,
    team_totals: dict[str, float] | None = None,
    n_sims: int = 10_000,
    rng_seed: int | None = 42,
) -> np.ndarray:
    """Estimate per-player ownership by simulating n_sims realistic DFS lineups.

    Team selection is driven by team_totals (implied runs). Player selection
    within stacks uses sqrt(projected_fpts). The salary floor, stacking
    structure, and calibrated probabilities are the same as generate_field().

    Returns array shape (len(players_df),) with same semantics as
    compute_heuristic_ownership(): values sum to slot_count per position group.
    """
    rng = np.random.default_rng(rng_seed)
    df = players_df.reset_index(drop=True)

    proj_w = np.sqrt(np.maximum(df["mean"].values.astype(float), 0.0))
    pos_pools = _build_pos_pools(df, proj_w)
    pmeta = _player_meta_from_df(df)

    team_batters: dict[str, list[int]] = {}
    team_stack_w: dict[str, float] = {}
    for pos_name in ("C", "1B", "2B", "3B", "SS", "OF"):
        p_ids, _ = pos_pools.get(pos_name, ([], np.array([])))
        for pid in p_ids:
            t = pmeta[pid]["team"]
            team_batters.setdefault(t, []).append(pid)
    for team in team_batters:
        team_stack_w[team] = float((team_totals or {}).get(team, 4.5))

    p_sals = sorted([v["salary"] for v in pmeta.values() if v["position"] == "P"], reverse=True)
    b_sals = sorted([v["salary"] for v in pmeta.values() if v["position"] != "P"], reverse=True)
    effective_floor = (
        _FIELD_SALARY_FLOOR if (sum(p_sals[:2]) + sum(b_sals[:8])) >= _FIELD_SALARY_FLOOR else 0.0
    )

    sp = float(_CAL.get("stack_probability", 0.775))
    pid_to_idx = {int(pid): i for i, pid in enumerate(df["player_id"].tolist())}
    counts = np.zeros(len(df), dtype=float)
    n_generated = 0
    attempts = 0

    while n_generated < n_sims and attempts < n_sims * 200:
        attempts += 1
        try:
            ids = (
                _sample_stacked_lineup(rng, pos_pools, pmeta, team_batters, team_stack_w)
                if rng.random() < sp
                else _sample_random_lineup(rng, pos_pools, pmeta)
            )
        except Exception:
            ids = None
        if ids is not None and effective_floor > 0:
            if sum(pmeta[pid]["salary"] for pid in ids) < effective_floor:
                ids = None
        if ids is not None:
            for pid in ids:
                if (idx := pid_to_idx.get(int(pid))) is not None:
                    counts[idx] += 1
            n_generated += 1

    if n_generated < n_sims:
        logger.warning("compute_emergent_ownership: only %d / %d sims", n_generated, n_sims)

    _SLOT_COUNTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    result = np.zeros(len(df), dtype=float)
    positions = df["position"].values
    for pos, n_slots in _SLOT_COUNTS.items():
        mask = positions == pos
        if not mask.any():
            continue
        pos_total = counts[mask].sum()
        result[mask] = (
            counts[mask] / pos_total * n_slots if pos_total > 0 else np.full(mask.sum(), n_slots / mask.sum())
        )
    return result


def _build_pos_pools(
    players_df: pd.DataFrame,
    ownership_vec: np.ndarray,
) -> dict[str, tuple[list[int], np.ndarray]]:
    """Return {position: (player_id_list, weight_array)} for fast sampling."""
    pools: dict[str, tuple[list[int], np.ndarray]] = {}
    df = players_df.reset_index(drop=True)
    for pos in df["position"].unique():
        mask = df["position"] == pos
        pids = df.loc[mask, "player_id"].tolist()
        w = ownership_vec[mask.values]
        w = w / w.sum() if w.sum() > 0 else np.ones(len(pids)) / len(pids)
        pools[pos] = (pids, w)
    return pools


def _player_meta_from_df(players_df: pd.DataFrame) -> dict:
    """Build {player_id: {salary, position, team, game, opponent}} dict."""
    meta = {}
    df = players_df.reset_index(drop=True)
    has_game = "game" in df.columns
    for _, row in df.iterrows():
        pid = int(row["player_id"])
        game_str = str(row["game"]) if has_game else ""
        team = str(row["team"])
        opp = ""
        if game_str and "@" in game_str:
            away, home = game_str.split("@", 1)
            opp = home if team == away else away
        meta[pid] = {
            "salary": float(row["salary"]),
            "position": str(row["position"]),
            "team": team,
            "game": game_str,
            "opponent": opp,
        }
    return meta


def _is_valid_field_lineup(
    ids: list[int],
    pmeta: dict,
) -> bool:
    """Quick validity check for field lineups (no bipartite matching needed)."""
    if len(ids) != 10 or len(set(ids)) != 10:
        return False
    salary = sum(pmeta[pid]["salary"] for pid in ids)
    if salary > SALARY_CAP:
        return False
    # Team hitter cap
    team_counts: dict[str, int] = {}
    for pid in ids:
        if pmeta[pid]["position"] != "P":
            t = pmeta[pid]["team"]
            team_counts[t] = team_counts.get(t, 0) + 1
            if team_counts[t] > _MAX_HITTERS_PER_TEAM:
                return False
    # Min games
    games = {pmeta[pid]["game"] for pid in ids if pmeta[pid]["game"]}
    if len(games) < _MIN_GAMES:
        return False
    # Pitcher-batter conflict: no pitcher vs a batter from opponent's team
    batter_teams = {pmeta[pid]["team"] for pid in ids if pmeta[pid]["position"] != "P"}
    for pid in ids:
        if pmeta[pid]["position"] == "P":
            opp = pmeta[pid]["opponent"]
            if opp and opp in batter_teams:
                return False
    return True


def _sample_stacked_lineup(
    rng: np.random.Generator,
    pos_pools: dict[str, tuple[list[int], np.ndarray]],
    pmeta: dict,
    team_batters: dict[str, list[int]],
    team_weights: dict[str, float],
) -> Optional[list[int]]:
    """Try to build one stacked lineup (4-5 batters from a primary team)."""
    # Weight teams by sum of their batter ownership
    teams = list(team_weights.keys())
    tw = np.array([team_weights[t] for t in teams])
    tw = tw / tw.sum()
    primary_team = teams[int(rng.choice(len(teams), p=tw))]
    primary_pool = team_batters.get(primary_team, [])
    if len(primary_pool) < 4:
        return None

    stack_size = 4 if rng.random() < _STACK_SIZE_4_PROB else 5
    if len(primary_pool) < stack_size:
        stack_size = len(primary_pool)

    # Ownership weights within primary stack
    all_pids = pos_pools.get("C", ([], []))[0] + pos_pools.get("1B", ([], []))[0] + \
               pos_pools.get("2B", ([], []))[0] + pos_pools.get("3B", ([], []))[0] + \
               pos_pools.get("SS", ([], []))[0] + pos_pools.get("OF", ([], []))[0]
    pid_to_ow: dict[int, float] = {}
    for pos_name in ("C", "1B", "2B", "3B", "SS", "OF"):
        p_ids, p_w = pos_pools.get(pos_name, ([], np.array([])))
        for pid, w in zip(p_ids, p_w):
            pid_to_ow[pid] = w

    stack_pool = [pid for pid in primary_pool if pid in pid_to_ow]
    if len(stack_pool) < stack_size:
        return None
    sw = np.array([pid_to_ow[pid] for pid in stack_pool])
    sw = sw / sw.sum()
    chosen_idx = rng.choice(len(stack_pool), size=stack_size, replace=False, p=sw)
    stack_ids = set(int(stack_pool[i]) for i in chosen_idx)

    # Fill 2 pitchers (not from primary team's opponent)
    p_ids, p_w = pos_pools.get("P", ([], np.array([])))
    excluded_pitcher_opps = {pmeta[pid]["opponent"] for pid in stack_ids}
    pitcher_pool = [pid for pid in p_ids if pmeta[pid]["team"] not in excluded_pitcher_opps]
    if len(pitcher_pool) < 2:
        return None
    pitcher_w_raw = np.array([pid_to_ow.get(pid, 1.0 / len(p_ids)) for pid in pitcher_pool])
    # Use the pitcher ownership from pos_pools
    p_w_dict = dict(zip(p_ids, p_w))
    pitcher_w = np.array([p_w_dict.get(pid, 1.0 / len(pitcher_pool)) for pid in pitcher_pool])
    pitcher_w = pitcher_w / pitcher_w.sum()
    pit_idx = rng.choice(len(pitcher_pool), size=2, replace=False, p=pitcher_w)
    pitcher_ids = [int(pitcher_pool[i]) for i in pit_idx]

    used = set(pitcher_ids) | stack_ids

    # Secondary stack: fill 2-3 batter slots from a second team before the
    # generic remaining-positions pass.
    secondary_ids: list[int] = []
    if rng.random() < _SECONDARY_STACK_PROB and len(teams) > 1:
        sec_teams = [t for t in teams if t != primary_team]
        if sec_teams:
            sec_tw = np.array([team_weights[t] for t in sec_teams])
            sec_tw = sec_tw / sec_tw.sum()
            secondary_team = sec_teams[int(rng.choice(len(sec_teams), p=sec_tw))]
            secondary_target = 2 if rng.random() < _SECONDARY_SIZE_2_PROB else 3

            for pos_name in ("C", "1B", "2B", "3B", "SS", "OF"):
                if len(secondary_ids) >= secondary_target:
                    break
                p_ids_pos, p_w_pos = pos_pools.get(pos_name, ([], np.array([])))
                covered = sum(
                    1 for pid in (stack_ids | set(secondary_ids))
                    if pmeta[pid]["position"] == pos_name
                )
                need = ROSTER_REQUIREMENTS.get(pos_name, 0) - covered
                if need <= 0:
                    continue
                sec_cands = [
                    pid for pid in p_ids_pos
                    if pmeta[pid]["team"] == secondary_team and pid not in used
                ]
                if not sec_cands:
                    continue
                cw_dict = dict(zip(p_ids_pos, p_w_pos))
                cw = np.array([cw_dict.get(pid, 1.0 / len(sec_cands)) for pid in sec_cands])
                cw = cw / cw.sum()
                n_pick = min(need, secondary_target - len(secondary_ids), len(sec_cands))
                chosen = [
                    int(sec_cands[i])
                    for i in rng.choice(len(sec_cands), size=n_pick, replace=False, p=cw)
                ]
                secondary_ids.extend(chosen)
                used.update(chosen)

    all_stacked = stack_ids | set(secondary_ids)
    remaining: list[int] = []
    for pos_name in ("C", "1B", "2B", "3B", "SS", "OF"):
        p_ids_pos, p_w_pos = pos_pools.get(pos_name, ([], np.array([])))
        already_covered = sum(
            1 for pid in all_stacked
            if pmeta[pid]["position"] == pos_name
        )
        need = ROSTER_REQUIREMENTS.get(pos_name, 0) - already_covered
        if need <= 0:
            continue
        cands = [pid for pid in p_ids_pos if pid not in used]
        if len(cands) < need:
            return None
        cw_raw = p_w_pos
        cw_dict = dict(zip(p_ids_pos, cw_raw))
        cw = np.array([cw_dict.get(pid, 1.0 / len(cands)) for pid in cands])
        cw = cw / cw.sum()
        chosen = [int(cands[i]) for i in rng.choice(len(cands), size=need, replace=False, p=cw)]
        remaining.extend(chosen)
        used.update(chosen)

    all_ids = pitcher_ids + list(stack_ids) + secondary_ids + remaining
    if len(set(all_ids)) != 10 or len(all_ids) != 10:
        return None
    if not _is_valid_field_lineup(all_ids, pmeta):
        return None
    return all_ids


def _sample_random_lineup(
    rng: np.random.Generator,
    pos_pools: dict[str, tuple[list[int], np.ndarray]],
    pmeta: dict,
) -> Optional[list[int]]:
    """Sample one lineup by filling slots independently weighted by ownership."""
    ids: list[int] = []
    used: set[int] = set()
    for slot in _SLOTS:
        eligible = _SLOT_ELIG.get(slot, {slot})
        cands: list[int] = []
        weights: list[float] = []
        for pos_name in eligible:
            p_ids, p_w = pos_pools.get(pos_name, ([], np.array([])))
            for pid, w in zip(p_ids, p_w):
                if pid not in used:
                    cands.append(pid)
                    weights.append(float(w))
        if not cands:
            return None
        w_arr = np.array(weights)
        w_arr = w_arr / w_arr.sum()
        chosen = int(cands[rng.choice(len(cands), p=w_arr)])
        ids.append(chosen)
        used.add(chosen)
    if not _is_valid_field_lineup(ids, pmeta):
        return None
    return ids


class ContestSimulator:
    """Generates simulated GPP fields and evaluates portfolio EV against them."""

    def generate_field(
        self,
        players_df: pd.DataFrame,
        ownership_vec: np.ndarray,
        n_lineups: int = 10_000,
        rng_seed: Optional[int] = None,
        stack_probability: float = 0.75,
        max_attempts_per_lineup: int = 200,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        progress_chunk: int = 500,
    ) -> np.ndarray:
        """Generate n_lineups opponent lineups sampled by ownership.

        Returns
        -------
        np.ndarray, shape (n_generated, 10), dtype int64 — player_ids per lineup.
        """
        rng = np.random.default_rng(rng_seed)
        pmeta = _player_meta_from_df(players_df)
        pos_pools = _build_pos_pools(players_df, ownership_vec)

        # Build team batter groups and team stack weights
        team_batters: dict[str, list[int]] = {}
        team_stack_w: dict[str, float] = {}
        for pos_name in ("C", "1B", "2B", "3B", "SS", "OF"):
            p_ids, p_w = pos_pools.get(pos_name, ([], np.array([])))
            for pid, w in zip(p_ids, p_w):
                t = pmeta[pid]["team"]
                team_batters.setdefault(t, []).append(pid)
                team_stack_w[t] = team_stack_w.get(t, 0.0) + float(w)

        # Empirical salary floor: only apply when the player pool can achieve it.
        # If the best-case lineup salary (top-2 pitchers + top-8 batters) falls below
        # _FIELD_SALARY_FLOOR, the pool is a toy/test fixture and we skip the floor.
        pitcher_sals = sorted(
            [v["salary"] for v in pmeta.values() if v["position"] == "P"], reverse=True
        )
        batter_sals = sorted(
            [v["salary"] for v in pmeta.values() if v["position"] != "P"], reverse=True
        )
        max_feasible_sal = sum(pitcher_sals[:2]) + sum(batter_sals[:8])
        effective_floor = _FIELD_SALARY_FLOOR if max_feasible_sal >= _FIELD_SALARY_FLOOR else 0.0

        field: list[list[int]] = []
        total_attempts = 0
        max_total = n_lineups * max_attempts_per_lineup

        while len(field) < n_lineups and total_attempts < max_total:
            total_attempts += 1
            try:
                if rng.random() < stack_probability:
                    ids = _sample_stacked_lineup(rng, pos_pools, pmeta, team_batters, team_stack_w)
                else:
                    ids = _sample_random_lineup(rng, pos_pools, pmeta)
            except Exception:
                ids = None

            if ids is not None:
                if effective_floor > 0:
                    total_sal = sum(pmeta[pid]["salary"] for pid in ids)
                    if total_sal < effective_floor:
                        ids = None
            if ids is not None:
                field.append(ids)
                if progress_cb is not None and len(field) % progress_chunk == 0:
                    progress_cb(len(field), n_lineups)

        if len(field) < n_lineups:
            logger.warning(
                "generate_field: only produced %d / %d lineups after %d attempts",
                len(field), n_lineups, total_attempts,
            )

        return np.array(field, dtype=np.int64)

    def score_field(
        self,
        field_lineups: np.ndarray,
        sim_matrix: np.ndarray,
        col_map: dict[int, int],
        batch_size: int = 500,
    ) -> np.ndarray:
        """Score all field lineups against the simulation matrix.

        Parameters
        ----------
        field_lineups : shape (n_field, 10) — player_ids
        sim_matrix : shape (n_sims, n_players)
        col_map : {player_id: col_index}

        Returns
        -------
        np.ndarray, shape (n_sims, n_field), dtype float32
        """
        n_field = field_lineups.shape[0]
        n_sims = sim_matrix.shape[0]

        # Convert player_ids to column indices; drop lineups with unmapped players
        col_lineups = np.full((n_field, 10), -1, dtype=np.int32)
        valid_mask = np.ones(n_field, dtype=bool)
        for i, lineup in enumerate(field_lineups):
            for j, pid in enumerate(lineup):
                col = col_map.get(int(pid), -1)
                if col == -1:
                    valid_mask[i] = False
                    break
                col_lineups[i, j] = col

        valid_lineups = col_lineups[valid_mask]
        n_valid = valid_lineups.shape[0]
        field_scores = np.zeros((n_sims, n_valid), dtype=np.float32)

        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            batch_cols = valid_lineups[start:end]  # (batch, 10)
            # sim_matrix[:, batch_cols] has shape (n_sims, batch, 10)
            field_scores[:, start:end] = sim_matrix[:, batch_cols].sum(axis=2)

        return field_scores

    def eval_portfolio(
        self,
        portfolio: list[tuple],
        players_df: pd.DataFrame,
        sim_results,
        team_totals: Optional[dict[str, float]] = None,
        n_field_lineups: int = 10_000,
        field_seed: int = 42,
        cash_threshold: float = 0.74,
    ) -> pd.DataFrame:
        """Evaluate each lineup's EV and stability metrics against a simulated field.

        Parameters
        ----------
        portfolio : list of (Lineup, score) tuples
        players_df : must include player_id, position, mean, salary, team, game
        sim_results : SimulationResults (used both for field generation and scoring)
        team_totals : optional {team: implied_total} for Model D ownership
        n_field_lineups : field sample size
        field_seed : base RNG seed for field generation

        Returns
        -------
        DataFrame with columns: lineup_index, cash_rate, beat_pct,
        ev_gap, field_gap, fragile
        """
        sim_matrix = sim_results.results_matrix.astype(np.float32)
        col_map = {pid: i for i, pid in enumerate(sim_results.player_ids)}
        n_sims = sim_matrix.shape[0]

        ownership_vec = compute_heuristic_ownership(players_df, team_totals)

        logger.info("Generating field A (%d lineups, seed=%d)...", n_field_lineups, field_seed)
        field_a = self.generate_field(
            players_df, ownership_vec, n_lineups=n_field_lineups, rng_seed=field_seed
        )
        logger.info("Generating field B (%d lineups, seed=%d)...", n_field_lineups, field_seed + 1)
        field_b = self.generate_field(
            players_df, ownership_vec, n_lineups=n_field_lineups, rng_seed=field_seed + 1
        )

        logger.info("Scoring fields against %d sims...", n_sims)
        scores_a = self.score_field(field_a, sim_matrix, col_map)  # (n_sims, n_field_a)
        scores_b = self.score_field(field_b, sim_matrix, col_map)  # (n_sims, n_field_b)

        # Sim split: first half = "train", second half = "eval"
        split = n_sims // 2
        scores_a_s1 = scores_a[:split]
        scores_a_s2 = scores_a[split:]

        records = []
        for i, (lineup, _) in enumerate(portfolio):
            lineup_cols = [col_map[pid] for pid in lineup.player_ids if pid in col_map]
            if len(lineup_cols) != 10:
                records.append({
                    "lineup_index": i + 1,
                    "cash_rate": None,
                    "beat_pct": None,
                    "ev_gap": None,
                    "field_gap": None,
                    "fragile": None,
                })
                continue

            lineup_scores = sim_matrix[:, lineup_cols].sum(axis=1)  # (n_sims,)

            # beat_pct: mean fraction of field A beaten per sim (full sims)
            beat_a = (lineup_scores[:, None] > scores_a).mean(axis=1)   # (n_sims,)
            beat_pct_a = float(beat_a.mean())
            # cash_rate: fraction of sims where lineup beats >= cash_threshold of field
            # (e.g. 0.74 for a top-26% GPP payout structure)
            cash_rate = float((beat_a >= cash_threshold).mean())

            # beat_pct vs field B (same sims, different field) → field_gap
            beat_b = (lineup_scores[:, None] > scores_b).mean(axis=1)
            beat_pct_b = float(beat_b.mean())

            # beat_pct on sim halves (same field A, different sims) → ev_gap
            beat_a_s1 = (lineup_scores[:split, None] > scores_a_s1).mean(axis=1)
            beat_pct_s1 = float(beat_a_s1.mean())
            beat_a_s2 = (lineup_scores[split:, None] > scores_a_s2).mean(axis=1)
            beat_pct_s2 = float(beat_a_s2.mean())

            denom_ev = max(beat_pct_a, 1e-6)
            denom_field = max(beat_pct_a, 1e-6)
            ev_gap = abs(beat_pct_s1 - beat_pct_s2) / denom_ev
            field_gap = abs(beat_pct_a - beat_pct_b) / denom_field
            fragile = ev_gap > 0.15 or field_gap > 0.15

            records.append({
                "lineup_index": i + 1,
                "cash_rate": round(cash_rate, 4),
                "beat_pct": round(beat_pct_a, 4),
                "ev_gap": round(ev_gap, 4),
                "field_gap": round(field_gap, 4),
                "fragile": fragile,
            })

        return pd.DataFrame(records)
