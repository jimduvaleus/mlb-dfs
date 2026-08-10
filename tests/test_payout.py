"""Tests for src/optimization/payout.py's nearest_payout_structure -- the
approximate-match fallback self_play's live wiring needs (unlike
structure_for_contest's other callers, self_play's round loop cannot skip a
contest with no payout table, it needs SOME real curve every round)."""
from src.optimization.payout import (
    CONTEST_STRUCTURES,
    load_payout_structure,
    nearest_payout_structure,
    structure_for_contest,
)


def test_real_dk_entries_file_names_match_via_substring():
    # Confirmed via a live smoke test 2026-08-08: real DK entries files carry
    # full display names ("MLB $10K Skipper [Single Entry]"), not the short
    # keys CONTEST_STRUCTURES is registered under -- every one of these must
    # resolve via substring matching, not the (previously exact-only) match.
    real_names = [
        "MLB $10K Skipper [Single Entry]",
        "MLB $175K Bat Flip [$50K to 1st]",
        "MLB $4K Base Hit [Single Entry $1K to 1st]",
        "MLB $80K Rally Cap [$20K to 1st]",
        "MLB $3K Five-Tool Player [5 Entry Max]",
        "MLB $10K Chin Music [Single Entry]",
        "MLB $20K Four-Seamer [20 Entry Max]",
        "MLB $2K Pickoff [Single Entry]",
        "MLB $3K Hot Corner [5 Entry Max]",
        "MLB $6K Solo Shot",
        "MLB $20K mini-MAX [150 Entry Max]",
    ]
    for name in real_names:
        struct, is_approx = nearest_payout_structure(name, 5000)
        assert is_approx is False, f"{name!r} should have matched exactly via substring"
        assert struct is not None


def test_substring_match_picks_closest_size_variant():
    struct, is_approx = nearest_payout_structure("MLB $20K Four-Seamer [20 Entry Max]", 5945)
    assert is_approx is False
    assert struct["total_entries"] == structure_for_contest("four-seamer", 5945)["total_entries"]


def test_ambiguous_substring_match_is_not_treated_as_exact():
    # Contrived name containing two different registered keys -- must not
    # silently pick one; falls through to the cross-type closest-size path.
    struct, is_approx = nearest_payout_structure("MLB Skipper meets Chin Music Special", 1000)
    assert is_approx is True
    assert struct is not None


def test_known_name_is_not_approximate():
    struct, is_approx = nearest_payout_structure("mini-max", 14268)
    assert is_approx is False
    assert struct["total_entries"] == structure_for_contest("mini-max", 14268)["total_entries"]


def test_unknown_name_falls_back_to_closest_size_across_all_types():
    # "chin music" 's smallest real variant is 1,189 entries -- an unknown
    # name with a nearby implied size should match it (or something equally
    # close) from across the WHOLE registry, not just fail.
    struct, is_approx = nearest_payout_structure("Totally New Contest Type", 1189)
    assert is_approx is True
    assert struct["total_entries"] is not None

    # The match must actually be the closest total_entries across every
    # registered variant, not just any one of them.
    all_structs = [
        load_payout_structure(k) for keys in CONTEST_STRUCTURES.values() for k in keys
    ]
    best = min(all_structs, key=lambda s: abs(int(s["total_entries"]) - 1189))
    assert struct["total_entries"] == best["total_entries"]


def test_none_n_entries_falls_back_to_smallest_registered_variant():
    struct, is_approx = nearest_payout_structure("Totally New Contest Type", None)
    assert is_approx is True
    all_structs = [
        load_payout_structure(k) for keys in CONTEST_STRUCTURES.values() for k in keys
    ]
    smallest = min(int(s["total_entries"]) for s in all_structs)
    assert struct["total_entries"] == smallest


def test_zero_or_negative_n_entries_treated_like_none():
    struct_zero, approx_zero = nearest_payout_structure("Totally New Contest Type", 0.0)
    struct_none, approx_none = nearest_payout_structure("Totally New Contest Type", None)
    assert approx_zero is True and approx_none is True
    assert struct_zero["total_entries"] == struct_none["total_entries"]


def test_unknown_name_never_raises_or_returns_none():
    struct, is_approx = nearest_payout_structure("", -5.0)
    assert struct is not None
    assert is_approx is True
