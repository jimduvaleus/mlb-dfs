"""
Canonical data structures for platform-specific rules.

No call sites change in Phase 1 — these dataclasses are definitions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Platform(str, Enum):
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"


@dataclass(frozen=True)
class ScoringRules:
    """
    Weights for every stat that contributes to fantasy points.

    Batter weights
    --------------
    single, double, triple, home_run, rbi, run, walk, hbp, sb

    Pitcher weights
    ---------------
    win, er, so, ip, h, bb, hb, cg, cgs, nh
    (FanDuel omits h/bb/hb/cg/cgs/nh; set those to 0.0.)

    FanDuel also awards a Quality Start (qs) bonus — DK has no equivalent.
    """

    # Batter
    single: float
    double: float
    triple: float
    home_run: float
    rbi: float
    run: float
    walk: float
    hbp: float
    sb: float

    # Pitcher
    win: float
    er: float
    so: float
    ip: float
    h: float = 0.0
    bb: float = 0.0
    hb: float = 0.0
    cg: float = 0.0
    cgs: float = 0.0
    nh: float = 0.0
    qs: float = 0.0  # quality start (FD only)

    def batter_points(
        self,
        single: float = 0,
        double: float = 0,
        triple: float = 0,
        home_run: float = 0,
        rbi: float = 0,
        run: float = 0,
        walk: float = 0,
        hbp: float = 0,
        sb: float = 0,
    ) -> float:
        return (
            single * self.single
            + double * self.double
            + triple * self.triple
            + home_run * self.home_run
            + rbi * self.rbi
            + run * self.run
            + walk * self.walk
            + hbp * self.hbp
            + sb * self.sb
        )

    def pitcher_points(
        self,
        win: float = 0,
        er: float = 0,
        so: float = 0,
        ip: float = 0,
        h: float = 0,
        bb: float = 0,
        hb: float = 0,
        cg: float = 0,
        cgs: float = 0,
        nh: float = 0,
        qs: float = 0,
    ) -> float:
        return (
            win * self.win
            + er * self.er
            + so * self.so
            + ip * self.ip
            + h * self.h
            + bb * self.bb
            + hb * self.hb
            + cg * self.cg
            + cgs * self.cgs
            + nh * self.nh
            + qs * self.qs
        )


@dataclass(frozen=True)
class RosterRules:
    """
    Roster construction constraints for a single platform + contest type.

    slots            — ordered list of slot labels (length == roster_size).
                       DK: ['P','P','C','1B','2B','3B','SS','OF','OF','OF']
                       FD: ['P','C/1B','2B','3B','SS','OF','OF','OF','UTIL']
    salary_cap       — maximum total salary allowed.
    max_hitters_per_team — DK=5, FD=4.
    min_games        — minimum distinct games required in the lineup.
    roster_size      — total number of players (derived from slots).
    """

    slots: tuple  # immutable so the dataclass stays frozen/hashable
    salary_cap: float
    max_hitters_per_team: int
    min_games: int

    @property
    def roster_size(self) -> int:
        return len(self.slots)
