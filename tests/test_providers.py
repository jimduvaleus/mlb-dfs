import pytest
import pandas as pd
from src.providers.static_csv import StaticCSVProvider
from src.providers.base import ProjectionProvider
import os

# Define the path for the sample CSV for testing
SAMPLE_CSV_PATH = "data/raw/sample_projections.csv"

@pytest.fixture(scope="module")
def setup_sample_csv():
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAMPLE_CSV_PATH), exist_ok=True)
    
    # Create a dummy CSV file for testing
    data = {
        "player_id": ["player1", "player2", "player3"],
        "mu": [10.0, 15.5, 8.2],
        "sigma": [2.0, 3.1, 1.5]
    }
    df = pd.DataFrame(data)
    df.to_csv(SAMPLE_CSV_PATH, index=False)
    
    yield  # This allows tests to run
    
    # Clean up the dummy CSV file after tests are done
    if os.path.exists(SAMPLE_CSV_PATH):
        os.remove(SAMPLE_CSV_PATH)

def test_static_csv_provider_init(setup_sample_csv):
    provider = StaticCSVProvider(SAMPLE_CSV_PATH)
    assert isinstance(provider, ProjectionProvider)
    assert not provider.projections.empty
    assert "player1" in provider.projections.index
    assert "mu" in provider.projections.columns
    assert "sigma" in provider.projections.columns

def test_static_csv_provider_file_not_found():
    with pytest.raises(FileNotFoundError, match="Projection CSV file not found at non_existent_file.csv"):
        StaticCSVProvider("non_existent_file.csv")

def test_static_csv_provider_missing_columns():
    # Create a CSV with missing columns
    bad_csv_path = "data/raw/bad_projections.csv"
    os.makedirs(os.path.dirname(bad_csv_path), exist_ok=True)
    pd.DataFrame({"player_id": ["p1"], "mu": [10.0]}).to_csv(bad_csv_path, index=False)
    
    with pytest.raises(ValueError, match="CSV must contain 'player_id', 'mu', and 'sigma' columns."):
        StaticCSVProvider(bad_csv_path)
    
    os.remove(bad_csv_path)

def test_static_csv_provider_get_projections(setup_sample_csv):
    provider = StaticCSVProvider(SAMPLE_CSV_PATH)
    mu, sigma = provider.get_projections("player1")
    assert mu == pytest.approx(10.0)
    assert sigma == pytest.approx(2.0)
    
    mu, sigma = provider.get_projections("player2")
    assert mu == pytest.approx(15.5)
    assert sigma == pytest.approx(3.1)

def test_static_csv_provider_get_projections_non_existent_player(setup_sample_csv):
    provider = StaticCSVProvider(SAMPLE_CSV_PATH)
    with pytest.raises(ValueError, match="Player ID 'playerX' not found in projections."):
        provider.get_projections("playerX")
