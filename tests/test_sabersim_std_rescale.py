"""A hand-edited projection must move the spread with it.

SaberSim's `dk_std` is the spread its own simulation produced around
`dk_points`. Editing `My Proj` on the user's board does not regenerate it, so
importing the edited mean against the untouched std leaves an inconsistent
pair -- and the coefficient of variation moves by the size of the edit, in the
wrong direction.

That is not hypothetical: on the 08/24 slate a team edited down 6.3% arrived
with a CV 6.7% too high, which moved it from 7th to 3rd on the slate for
variance-per-unit-projection and made a variance-seeking objective favour it.
The bug is silent -- every column is individually valid.

`build_quantile_grids(rescale_to_file_mean=True)` fixes the same leak on the
grid path; this covers the Gaussian-fallback path so the two agree, and a
player just below the grid-validity threshold responds to an edit the same way
as one just above it.
"""
import numpy as np
import pandas as pd
import pytest

from src.api.external_pool import _STD_RESCALE_DEADBAND, _std_scaled_to_edited_mean


def _frame(my_proj, dk_points, dk_std):
    return pd.DataFrame({"My Proj": my_proj, "dk_points": dk_points, "dk_std": dk_std})


def _run(df):
    return _std_scaled_to_edited_mean(
        df, "dk_std", "dk_points", pd.to_numeric(df["My Proj"], errors="coerce"),
    )


def test_unedited_rows_are_untouched():
    df = _frame([7.0, 9.5, 3.2], [7.0, 9.5, 3.2], [6.9, 8.1, 4.0])
    np.testing.assert_allclose(_run(df), df["dk_std"])


def test_editing_the_mean_down_scales_the_spread_down():
    """The property the whole file exists for: CV is preserved."""
    df = _frame([6.57], [7.01], [6.93])
    got = float(_run(df).iloc[0])
    assert got == pytest.approx(6.93 * (6.57 / 7.01))
    # SaberSim's own CV, recovered exactly.
    assert got / 6.57 == pytest.approx(6.93 / 7.01)


def test_editing_up_scales_up():
    df = _frame([8.0], [7.0], [6.0])
    assert float(_run(df).iloc[0]) == pytest.approx(6.0 * 8.0 / 7.0)


def test_opposing_team_shifts_are_honoured():
    """SaberSim re-subsamples whole GAME sims when a team projection moves, so
    editing one team shifts its opponent's numbers slightly, in both
    directions. On 08/24 a 4.2-7.8% cut to nine PIT batters moved all ten SD
    batters by -1.1% to +1.35%. Those are real distribution changes, so the
    rescale must key on the columns disagreeing rather than on which team
    anyone thinks was edited."""
    df = _frame([8.010, 5.520, 6.410], [8.100, 5.447, 6.401], [7.0, 6.0, 6.5])
    got = _run(df)
    assert not np.allclose(got, df["dk_std"]), "opponent shifts must not be ignored"
    # Direction follows the sign of the shift, per player.
    assert got.iloc[0] < 7.0     # Tatis edited down
    assert got.iloc[1] > 6.0     # Bogaerts edited up


def test_display_rounding_is_a_no_op():
    """`My Proj` is stored to fewer decimals than `dk_points`, so the two
    disagree at the 4th decimal on nearly every row. Measured on 08/24:
    rounding tops out at 0.091%, the smallest real edit is 0.14%. Scaling on
    rounding would perturb every replay of every slate for no reason."""
    df = _frame([7.0004, 9.4996], [7.0, 9.5], [6.9, 8.1])
    np.testing.assert_allclose(_run(df), df["dk_std"])


def test_the_deadband_does_not_swallow_a_real_edit():
    edited = 7.0 * (1.0 + _STD_RESCALE_DEADBAND * 3)
    df = _frame([edited], [7.0], [6.0])
    assert float(_run(df).iloc[0]) != pytest.approx(6.0)


@pytest.mark.parametrize("my_proj,dk_points", [
    (7.0, 0.0),        # no baseline to scale against
    (7.0, np.nan),     # column absent for this row
    (np.nan, 7.0),     # My Proj blank -> mean falls back, nothing was edited
    (70.0, 7.0),       # 10x: a mistyped cell, not an edit
    (0.1, 7.0),        # blanked toward zero
])
def test_degenerate_rows_keep_the_original_std(my_proj, dk_points):
    df = _frame([my_proj], [dk_points], [6.0])
    assert float(_run(df).iloc[0]) == pytest.approx(6.0)


def test_zero_std_stays_zero():
    df = _frame([6.0], [7.0], [0.0])
    assert float(_run(df).iloc[0]) == pytest.approx(0.0)
