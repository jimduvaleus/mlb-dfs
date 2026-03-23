import pytest
import pandas as pd
from src.ingestion.dk_slate import DraftKingsSlateIngestor, Player
from src.utils.scoring import calculate_batter_points, calculate_pitcher_points
from src.ingestion.retrosheet_parser import RetrosheetParser

# Define a fixture for the sample CSV path
@pytest.fixture
def sample_dk_classic_csv(tmp_path):
    csv_content = """
ID,Name,Roster Position,Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame,"Name + ID",Tier,Player and position,Starter?,PPR
10001,Shohei Ohtani,P/DH,P,10000,"LAD @ SD 03/20/2026 09:40PM ET",LAD,18.5,"Shohei Ohtani (10001)",,
10002,Freddie Freeman,1B,1B,9000,"LAD @ SD 03/20/2026 09:40PM ET",LAD,12.3,"Freddie Freeman (10002)",,
10003,Mookie Betts,2B/OF,2B,9500,"LAD @ SD 03/20/2026 09:40PM ET",LAD,15.1,"Mookie Betts (10003)",,
10004,Manny Machado,3B,3B,8500,"LAD @ SD 03/20/2026 09:40PM ET",SD,11.8,"Manny Machado (10004)",,
10005,Fernando Tatis Jr.,OF,OF,8800,"LAD @ SD 03/20/2026 09:40PM ET",SD,13.2,"Fernando Tatis Jr. (10005)",,
10006,Yu Darvish,P,P,8000,"LAD @ SD 03/20/2026 09:40PM ET",SD,16.0,"Yu Darvish (10006)",,
"""
    file_path = tmp_path / "sample_dk_classic.csv"
    file_path.write_text(csv_content)
    return str(file_path)

@pytest.fixture
def sample_dk_classic_csv_invalid_salary(tmp_path):
    csv_content = """
ID,Name,Roster Position,Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame,"Name + ID",Tier,Player and position,Starter?,PPR
10001,Shohei Ohtani,P/DH,P,10000,"LAD @ SD 03/20/2026 09:40PM ET",LAD,18.5,"Shohei Ohtani (10001)",,
10008,Missing Salary,,C,INVALID,"LAD @ SD 03/20/2026 09:40PM ET",LAD,0.0,"Missing Salary (10008)",,
"""
    file_path = tmp_path / "sample_dk_classic_invalid_salary.csv"
    file_path.write_text(csv_content)
    return str(file_path)

@pytest.fixture
def sample_dk_classic_csv_invalid_position(tmp_path):
    csv_content = """
ID,Name,Roster Position,Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame,"Name + ID",Tier,Player and position,Starter?,PPR
10001,Shohei Ohtani,P/DH,P,10000,"LAD @ SD 03/20/2026 09:40PM ET",LAD,18.5,"Shohei Ohtani (10001)",,
10007,Invalid Player,P,INVALID_POS,6000,"LAD @ SD 03/20/2026 09:40PM ET",SD,0.0,"Invalid Player (10007)",,
"""
    file_path = tmp_path / "sample_dk_classic_invalid_position.csv"
    file_path.write_text(csv_content)
    return str(file_path)


def test_dk_slate_ingestor_loads_data_frame_correctly(sample_dk_classic_csv):
    ingestor = DraftKingsSlateIngestor(sample_dk_classic_csv)
    df = ingestor.get_slate_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 6
    assert "player_id" in df.columns
    assert "name" in df.columns
    assert "position" in df.columns
    assert "salary" in df.columns
    assert "team" in df.columns
    assert "roster_position" in df.columns

    # Verify a specific player's data
    ohtani = df[df["name"] == "Shohei Ohtani"].iloc[0]
    assert ohtani["player_id"] == 10001
    assert ohtani["position"] == "P"
    assert ohtani["salary"] == 10000.0
    assert ohtani["team"] == "LAD"
    assert ohtani["roster_position"] == "P/DH"


def test_dk_slate_ingestor_returns_player_objects_correctly(sample_dk_classic_csv):
    ingestor = DraftKingsSlateIngestor(sample_dk_classic_csv)
    players = ingestor.get_players()

    assert isinstance(players, list)
    assert len(players) == 6
    assert all(isinstance(p, Player) for p in players)

    ohtani = next(p for p in players if p.name == "Shohei Ohtani")
    assert ohtani.player_id == 10001
    assert ohtani.position == "P"
    assert ohtani.salary == 10000.0
    assert ohtani.team == "LAD"
    assert ohtani.roster_position == "P/DH"


def test_dk_slate_ingestor_handles_invalid_salary(sample_dk_classic_csv_invalid_salary):
    with pytest.raises(ValueError, match="Missing or invalid salaries found in CSV."):
        DraftKingsSlateIngestor(sample_dk_classic_csv_invalid_salary)


def test_dk_slate_ingestor_handles_invalid_position(sample_dk_classic_csv_invalid_position):
    with pytest.raises(ValueError, match="Unexpected position strings found in CSV."):
        DraftKingsSlateIngestor(sample_dk_classic_csv_invalid_position)

# --- New tests for scoring.py and retrosheet_parser.py ---

def test_calculate_batter_points():
    # Single: 3, Double: 5, Triple: 8, HR: 10, RBI: 2, Run: 2, Walk: 2, HBP: 2, SB: 5
    # Example: 1B, 1D, 1T, 1HR, 2RBI, 2R, 1BB, 1HBP, 1SB
    points = calculate_batter_points(single=1, double=1, triple=1, hr=1, rbi=2, run=2, walk=1, hbp=1, sb=1)
    expected_points = (3 * 1) + (5 * 1) + (8 * 1) + (10 * 1) + (2 * 2) + (2 * 2) + (2 * 1) + (2 * 1) + (5 * 1)
    assert points == expected_points # 3 + 5 + 8 + 10 + 4 + 4 + 2 + 2 + 5 = 43

def test_calculate_pitcher_points():
    # Win: 4, ER: -2, SO: 2, IP: 2.25, H: -0.6, BB: -0.6, HB: -0.6, CG: 2.5, CGS: 2.5, NH: 5
    # Example: 1W, 1ER, 5SO, 6IP, 4H, 1BB, 0HB, 0CG, 0CGS, 0NH
    points = calculate_pitcher_points(win=1, er=1, so=5, ip=6, h=4, bb=1, hb=0, cg=0, cgs=0, nh=0)
    expected_points = (4 * 1) + (-2 * 1) + (2 * 5) + (2.25 * 6) + (-0.6 * 4) + (-0.6 * 1) + (0 * -0.6) + (2.5 * 0) + (2.5 * 0) + (5 * 0)
    assert points == expected_points # 4 - 2 + 10 + 13.5 - 2.4 - 0.6 = 22.5

def test_retrosheet_parser_process_batting_stats():
    data = {
        'player_id': [1, 2],
        'game_id': ['GID1', 'GID1'],
        'H': [2, 1], 'D': [1, 0], 'T': [0, 0], 'HR': [1, 0],
        'RBI': [2, 0], 'R': [1, 1], 'BB': [1, 0], 'HBP': [0, 0], 'SB': [1, 0],
    }
    df = pd.DataFrame(data)
    processed_df = RetrosheetParser.process_batting_stats(df)

    # Player 1: 1B (2-1-0-1=0), 1D, 0T, 1HR, 2RBI, 1R, 1BB, 0HBP, 1SB
    # Points: (0*3) + (1*5) + (0*8) + (1*10) + (2*2) + (1*2) + (1*2) + (0*2) + (1*5) = 0 + 5 + 0 + 10 + 4 + 2 + 2 + 0 + 5 = 28
    assert processed_df.loc[0, 'dk_points'] == 28.0

    # Player 2: 1B (1-0-0-0=1), 0D, 0T, 0HR, 0RBI, 1R, 0BB, 0HBP, 0SB
    # Points: (1*3) + (0*5) + (0*8) + (0*10) + (0*2) + (1*2) + (0*2) + (0*2) + (0*5) = 3 + 0 + 0 + 0 + 0 + 2 + 0 + 0 + 0 = 5
    assert processed_df.loc[1, 'dk_points'] == 5.0

def test_retrosheet_parser_process_pitching_stats():
    data = {
        'player_id': [1, 2],
        'game_id': ['GID1', 'GID1'],
        'W': [1, 0], 'IP': ["7.0", "5.1"], 'ER': [1, 2], 'K': [8, 3],
        'H': [5, 7], 'BB': [1, 2], 'HB': [0, 1]
    }
    df = pd.DataFrame(data)
    processed_df = RetrosheetParser.process_pitching_stats(df)

    # Player 1: 1W, 1ER, 8K, 7IP, 5H, 1BB, 0HB
    # Points: (1*4) + (1*-2) + (8*2) + (7*2.25) + (5*-0.6) + (1*-0.6) + (0*-0.6) = 4 - 2 + 16 + 15.75 - 3 - 0.6 = 30.15
    assert processed_df.loc[0, 'dk_points'] == 30.15

    # Player 2: 0W, 2ER, 3K, 5.1IP (5 + 1/3 = 5.333), 7H, 2BB, 1HB
    # Points: (0*4) + (2*-2) + (3*2) + (5.333333*2.25) + (7*-0.6) + (2*-0.6) + (1*-0.6) = 0 - 4 + 6 + 12 - 4.2 - 1.2 - 0.6 = 8
    # Note: Using approximate value for 5.333333*2.25 = 12
    # Recalculating precisely:
    # 5.1 IP -> 5 + 1/3 innings = 5.333333...
    # (5 + 1/3) * 2.25 = (16/3) * (9/4) = (4*3) = 12
    expected_p2_points = (0*4) + (2*-2) + (3*2) + ((16/3)*2.25) + (7*-0.6) + (2*-0.6) + (1*-0.6)
    assert processed_df.loc[1, 'dk_points'] == expected_p2_points
