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

# Type alias: player_id -> {position, salary, team, game}
PlayerMeta = Dict[int, Dict]


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

        # Position counts must match roster requirements exactly
        pos_counts: Dict[str, int] = {}
        for r in rows:
            pos = r['position']
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for pos, req in ROSTER_REQUIREMENTS.items():
            if pos_counts.get(pos, 0) != req:
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
