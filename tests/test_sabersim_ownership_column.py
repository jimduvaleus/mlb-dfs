"""Ownership must come from SaberSim's `My Own`, never `Adj Own`.

`Adj Own` is a rating, not a projection of the field: across six 2026-09
exports `My Own` summed to exactly 200% for pitchers / ~800% for hitters (the
2 + 8 DK roster slots), while `Adj Own` hitters summed to 858-944% with chalk
stretched well past `My Own` (54.8% vs 37.7%). Every downstream consumer
(candidate sampling, field generation, the dupe model) assumes a slate that
sums to the roster size, so reading `Adj Own` silently over-chalks the field.

The two columns correlate ~0.99, so nothing downstream would look wrong --
hence a test pinned to the column itself.
"""
import numpy as np
import pandas as pd

from src.api.external_pool import parse_player_projections, parse_sabersim_projections


def _write_export(tmp_path):
    df = pd.DataFrame({
        "DFS ID": [101, 102, 201],
        "Name": ["Hitter A", "Hitter B", "Pitcher C"],
        "Pos": ["OF", "1B", "SP"],
        "Order": [1, 2, np.nan],
        "Team": ["SEA", "SEA", "LAA"],
        "Status": ["Confirmed", "Confirmed", ""],
        "Salary": [5000, 4000, 8000],
        "My Proj": [9.0, 8.0, 16.0],
        "dk_points": [9.0, 8.0, 16.0],
        "dk_std": [7.0, 6.5, 9.0],
        "fd_points": [11.0, 10.0, 30.0],
        "fd_std": [8.0, 7.5, 13.0],
        # Deliberately far apart so reading the wrong column can't pass.
        "My Own": [20.0, 10.0, 30.0],
        "Adj Own": [45.0, 25.0, 55.0],
    })
    path = tmp_path / "MLB_2026-09-26-710pm_DK_Main.csv"
    df.to_csv(path, index=False)
    return path


def test_projections_source_reads_my_own_as_fraction(tmp_path):
    out = parse_sabersim_projections(_write_export(tmp_path)).set_index("player_id")
    assert out.loc[[101, 102, 201], "ownership"].tolist() == [0.20, 0.10, 0.30]


def test_external_pool_reads_my_own_as_percentage_points(tmp_path):
    out = parse_player_projections(_write_export(tmp_path)).set_index("player_id")
    assert out.loc[[101, 102, 201], "ownership"].tolist() == [20.0, 10.0, 30.0]


def test_both_paths_agree_up_to_scale(tmp_path):
    """The two modes consume the same export and must not disagree on a
    player's projected ownership."""
    path = _write_export(tmp_path)
    src = parse_sabersim_projections(path).set_index("player_id")["ownership"]
    ext = parse_player_projections(path).set_index("player_id")["ownership"]
    np.testing.assert_allclose(src * 100.0, ext.loc[src.index])
