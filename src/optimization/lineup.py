from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# DraftKings Classic MLB roster requirements — preserved for backward compat
# and still used directly by BasinHoppingOptimizer (Phase 7 will wire platforms).
ROSTER_REQUIREMENTS: Dict[str, int] = {
    'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3,
}
SALARY_CAP: float = 50_000.0
MAX_HITTERS_PER_TEAM: int = 5
MIN_GAMES: int = 2

# Type alias: player_id -> {position, eligible_positions, salary, team, game}
PlayerMeta = Dict[int, Dict]

# Expanded slot list matching ROSTER_REQUIREMENTS; built once at import time.
SLOTS: List[str] = []
for _pos, _cnt in ROSTER_REQUIREMENTS.items():
    SLOTS.extend([_pos] * _cnt)


@dataclass
class Lineup:
    """Represents a single DraftKings or FanDuel classic MLB lineup."""

    player_ids: List[int]

    def score(
        self,
        sim_matrix: np.ndarray,
        col_map: Dict[int, int],
        target: float,
    ) -> float:
        """Return P(lineup total >= target) estimated over all simulation rows."""
        cols = [col_map[pid] for pid in self.player_ids]
        totals = sim_matrix[:, cols].sum(axis=1)
        return float((totals >= target).mean())

    def is_valid(
        self,
        player_meta: PlayerMeta,
        salary_floor: Optional[float] = None,
        rules=None,        # Optional[RosterRules] — imported lazily to avoid
        slot_eligibility=None,  # Optional[dict]    — circular import risk
    ) -> bool:
        """
        Check roster construction constraints for a lineup.

        Parameters
        ----------
        player_meta:
            Per-player metadata dict (player_id → dict with position,
            eligible_positions, salary, team, opponent, game).
        salary_floor:
            Minimum total salary allowed (optional).
        rules:
            A :class:`src.platforms.base.RosterRules` instance.
            Defaults to ``DK_ROSTER`` when not provided.
        slot_eligibility:
            Mapping of slot label → set of eligible player positions.
            Required for platforms with compound slots (FD's ``C/1B``,
            ``UTIL``).  Defaults to the DK identity mapping when not provided.

        Backward compatibility
        ---------------------
        Existing callers that pass no ``rules`` or ``slot_eligibility`` get
        the same DK Classic behaviour as before Phase 4.
        """
        # Lazy imports to avoid making every module depend on src.platforms
        # at import time; also keeps the legacy call sites unchanged.
        if rules is None:
            from src.platforms.draftkings import DK_ROSTER
            rules = DK_ROSTER
        if slot_eligibility is None:
            from src.platforms.registry import get_slot_eligibility
            from src.platforms.base import Platform
            slot_eligibility = get_slot_eligibility(Platform.DRAFTKINGS)

        r = rules
        se = slot_eligibility
        slots = list(r.slots)
        roster_size = r.roster_size

        if len(self.player_ids) != roster_size or len(set(self.player_ids)) != roster_size:
            return False

        rows = [player_meta[pid] for pid in self.player_ids if pid in player_meta]
        if len(rows) != roster_size:
            return False

        # ------------------------------------------------------------------
        # Slot assignment via bipartite matching (Hopcroft-Karp style DFS).
        #
        # For DK, each slot label is an exact position name, so
        # se.get(slot_pos, {slot_pos}) == {slot_pos} and the behaviour is
        # identical to the pre-Phase-4 code.
        #
        # For FD, compound labels expand:
        #   'C/1B'  → {'C', '1B'}
        #   'UTIL'  → {'C', '1B', '2B', '3B', 'SS', 'OF'}
        # ------------------------------------------------------------------
        rows_list = list(rows)

        def _elig(r_: Dict) -> set:
            ep = r_.get('eligible_positions')
            return set(ep) if ep else {r_['position']}

        match_slot = [-1] * len(slots)

        def _try_assign(player_idx: int, elig: set, visited: set) -> bool:
            for j, slot_pos in enumerate(slots):
                slot_positions = se.get(slot_pos, {slot_pos})
                if elig & slot_positions and j not in visited:
                    visited.add(j)
                    if match_slot[j] == -1 or _try_assign(
                        match_slot[j], _elig(rows_list[match_slot[j]]), visited
                    ):
                        match_slot[j] = player_idx
                        return True
            return False

        matched = sum(
            1 for i, row in enumerate(rows_list)
            if _try_assign(i, _elig(row), set())
        )
        if matched != roster_size:
            return False

        # Salary bounds
        total_salary = sum(row['salary'] for row in rows)
        if total_salary > r.salary_cap:
            return False
        if salary_floor is not None and total_salary < salary_floor:
            return False

        # Max hitters from one team (pitchers excluded from this count)
        hitter_team: Dict[str, int] = {}
        for row in rows:
            if row['position'] != 'P':
                t = row['team']
                hitter_team[t] = hitter_team.get(t, 0) + 1
        if hitter_team and max(hitter_team.values()) > r.max_hitters_per_team:
            return False

        # At least min_games distinct games (only enforced when game info present)
        games = {row.get('game', '') for row in rows if row.get('game', '')}
        if games and len(games) < r.min_games:
            return False

        # No pitcher may oppose any batter in the same lineup
        pitcher_opponents = {
            row['opponent'] for row in rows
            if row['position'] == 'P' and row.get('opponent')
        }
        batter_teams = {row['team'] for row in rows if row['position'] != 'P'}
        if pitcher_opponents & batter_teams:
            return False

        # Both pitchers must be from different teams (DK has 2 P slots;
        # for FD the roster has only 1 P slot so this list has length 1
        # and the check is naturally skipped).
        pitcher_teams = [row['team'] for row in rows if row['position'] == 'P']
        if len(pitcher_teams) == 2 and pitcher_teams[0] == pitcher_teams[1]:
            return False

        return True
