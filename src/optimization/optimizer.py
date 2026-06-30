"""Lineup construction helpers shared across the optimization pipeline."""
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.optimization.lineup import PlayerMeta, SLOTS


def _build_player_meta(players_df: pd.DataFrame) -> PlayerMeta:
    """Convert a players DataFrame to a fast dict-based lookup."""
    has_game = 'game' in players_df.columns
    has_eligible = 'eligible_positions' in players_df.columns
    meta: PlayerMeta = {}
    for _, row in players_df.iterrows():
        pos = row['position']
        elig = list(row['eligible_positions']) if has_eligible else []
        if not elig:
            elig = [pos]
        team = row['team']
        game_str = str(row['game']) if has_game else ''
        if has_game and '@' in game_str:
            away, home = game_str.split('@', 1)
            opponent = home if team == away else away
        else:
            opponent = ''
        meta[int(row['player_id'])] = {
            'position': pos,
            'eligible_positions': elig,
            'salary': float(row['salary']),
            'team': team,
            'opponent': opponent,
            'game': game_str,
        }
    return meta


def _compute_slot_assignment(
    ids: List[int],
    player_meta: PlayerMeta,
    slots: Optional[List[str]] = None,
    slot_eligibility: Optional[Dict[str, set]] = None,
) -> Tuple[List[int], List[int]]:
    """Compute a valid player→slot bipartite matching via augmenting-path DFS.

    Returns ``(slot_to_pidx, pidx_to_slot)`` where:
      ``slot_to_pidx[j]`` = index into *ids* matched to slots[j]  (-1 = free)
      ``pidx_to_slot[i]`` = slot index matched to ids[i]          (-1 = unmatched)

    Raises RuntimeError if no full matching exists — only call with lineups
    that have already passed Lineup.is_valid().
    """
    _slots = slots if slots is not None else SLOTS
    _se: Dict[str, set] = slot_eligibility if slot_eligibility is not None else {}
    n = len(ids)
    slot_to_pidx: List[int] = [-1] * len(_slots)
    pidx_to_slot: List[int] = [-1] * n

    def _elig(pidx: int) -> List[int]:
        meta = player_meta.get(ids[pidx])
        if meta is None:
            return []
        ep = meta.get('eligible_positions') or [meta['position']]
        ep_set = set(ep)
        return [j for j, s in enumerate(_slots) if ep_set & _se.get(s, {s})]

    def _dfs(pidx: int, visited: set) -> bool:
        for j in _elig(pidx):
            if j not in visited:
                visited.add(j)
                occ = slot_to_pidx[j]
                if occ == -1 or _dfs(occ, visited):
                    slot_to_pidx[j] = pidx
                    pidx_to_slot[pidx] = j
                    return True
        return False

    for i in range(n):
        if not _dfs(i, set()):
            raise RuntimeError(
                f"_compute_slot_assignment: could not match player index {i} "
                f"(id={ids[i]}); lineup appears invalid."
            )
    return slot_to_pidx, pidx_to_slot
