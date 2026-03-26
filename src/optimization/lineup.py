from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# DraftKings Classic MLB roster requirements
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
    """Represents a single DraftKings classic MLB lineup (10 players)."""

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
    ) -> bool:
        """Check all DraftKings Classic constraints."""
        if len(self.player_ids) != 10:
            return False

        rows = [player_meta[pid] for pid in self.player_ids if pid in player_meta]
        if len(rows) != 10:
            return False

        # Verify a valid slot assignment exists via bipartite matching.
        # Each player can fill any slot whose position label is in their
        # eligible_positions list (falls back to [position] if absent).
        rows_list = list(rows)

        def _elig(r: Dict) -> set:
            ep = r.get('eligible_positions')
            return set(ep) if ep else {r['position']}

        match_slot = [-1] * len(SLOTS)

        def _try_assign(player_idx: int, elig: set, visited: set) -> bool:
            for j, slot_pos in enumerate(SLOTS):
                if slot_pos in elig and j not in visited:
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
        if matched != 10:
            return False

        # Salary bounds
        total_salary = sum(r['salary'] for r in rows)
        if total_salary > SALARY_CAP:
            return False
        if salary_floor is not None and total_salary < salary_floor:
            return False

        # Max 5 hitters from one team (pitchers do not count toward this limit)
        hitter_team: Dict[str, int] = {}
        for r in rows:
            if r['position'] != 'P':
                t = r['team']
                hitter_team[t] = hitter_team.get(t, 0) + 1
        if hitter_team and max(hitter_team.values()) > MAX_HITTERS_PER_TEAM:
            return False

        # At least 2 different games (only enforced when game info is present)
        games = {r.get('game', '') for r in rows if r.get('game', '')}
        if games and len(games) < MIN_GAMES:
            return False

        return True
