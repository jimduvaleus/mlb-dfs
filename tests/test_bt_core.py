"""Unit tests for tests/bt_core.py's own-entry exclusion (see load_real_contests
docstring -- our real historical submissions sit inside the standings zips we
grade against, and an unfiltered field lets a hypothetical candidate split a
prize with our own real entry for the same lineup)."""
from tests.bt_core import OWN_USERNAMES, _is_own_entry


def test_single_entry_exact_match():
    for u in OWN_USERNAMES:
        assert _is_own_entry(u)


def test_multi_entry_suffix_stripped():
    for u in OWN_USERNAMES:
        assert _is_own_entry(f"{u} (1/150)")
        assert _is_own_entry(f"{u} (72/72)")


def test_case_insensitive():
    assert _is_own_entry("EdgelessCart")
    assert _is_own_entry("EDUVALEUS (2/4)")


def test_other_real_usernames_not_matched():
    assert not _is_own_entry("mattyrob23")
    assert not _is_own_entry("horse7887 (2/4)")


def test_substring_does_not_false_positive():
    # A real username containing one of ours as a substring must NOT match --
    # only the exact pre-suffix username counts.
    assert not _is_own_entry("xedgelesscart99")
    assert not _is_own_entry("edgelesscartfan (1/2)")
