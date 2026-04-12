"""
Factory functions for retrieving platform-specific rules by Platform enum value.
"""

from src.platforms.base import Platform, RosterRules, ScoringRules
from src.platforms.draftkings import DK_ROSTER, DK_SCORING
from src.platforms.fanduel import FD_ROSTER, FD_SCORING, FD_SLOT_ELIGIBILITY

# DK slots are all exact-match (no compound labels), so eligibility is 1:1.
_DK_SLOT_ELIGIBILITY: dict[str, set[str]] = {
    slot: {slot} for slot in dict.fromkeys(DK_ROSTER.slots)
}


def get_scoring(platform: Platform) -> ScoringRules:
    if platform == Platform.DRAFTKINGS:
        return DK_SCORING
    if platform == Platform.FANDUEL:
        return FD_SCORING
    raise ValueError(f"Unknown platform: {platform!r}")


def get_roster(platform: Platform) -> RosterRules:
    if platform == Platform.DRAFTKINGS:
        return DK_ROSTER
    if platform == Platform.FANDUEL:
        return FD_ROSTER
    raise ValueError(f"Unknown platform: {platform!r}")


def get_slot_eligibility(platform: Platform) -> dict[str, set[str]]:
    """
    Returns a mapping of slot label → set of eligible player positions.

    DraftKings slots are all single-position labels so every slot maps to
    exactly one position.  FanDuel compound slots (C/1B, UTIL) expand to
    multiple positions.
    """
    if platform == Platform.DRAFTKINGS:
        return _DK_SLOT_ELIGIBILITY
    if platform == Platform.FANDUEL:
        return FD_SLOT_ELIGIBILITY
    raise ValueError(f"Unknown platform: {platform!r}")
