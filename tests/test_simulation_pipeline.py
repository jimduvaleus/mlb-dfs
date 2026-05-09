import pandas as pd
import numpy as np
import os
import subprocess
from src.models.copula import EmpiricalCopula
from src.simulation.engine import SimulationEngine
from src.simulation.results import SimulationResults

def run_test():
    # Step 1: Build the copula (using the script)
    # This will generate dummy historical_logs.parquet and empirical_copula.parquet
    print("Building copula...")
    subprocess.run(["python", "scripts/build_copula.py"], check=True)
    
    # Step 2: Initialize Copula
    print("Initializing EmpiricalCopula...")
    copula = EmpiricalCopula("data/processed/empirical_copula.parquet")
    
    # Step 3: Create a mock slate of players (2 teams, 1 game)
    # Team A batters 1-9 vs Team B pitcher (slot 10)
    # Team B batters 1-9 vs Team A pitcher (slot 10)
    print("Creating mock slate...")
    players_data = []
    # Game: Team A vs Team B
    for slot in range(1, 10):
        players_data.append({'player_id': 100 + slot, 'team': 'NYY', 'opponent': 'BOS', 'slot': slot, 'mean': 8.0, 'std_dev': 4.0})
        players_data.append({'player_id': 200 + slot, 'team': 'BOS', 'opponent': 'NYY', 'slot': slot, 'mean': 7.0, 'std_dev': 3.5})
    
    # Add pitchers
    players_data.append({'player_id': 110, 'team': 'NYY', 'opponent': 'BOS', 'slot': 10, 'mean': 15.0, 'std_dev': 8.0})
    players_data.append({'player_id': 210, 'team': 'BOS', 'opponent': 'NYY', 'slot': 10, 'mean': 14.0, 'std_dev': 7.5})
    
    players_df = pd.DataFrame(players_data)
    
    # Step 4: Run Simulation Engine
    print("Running simulation...")
    engine = SimulationEngine(copula, players_df)
    n_sims = 1000
    results = engine.simulate(n_sims)
    
    # Step 5: Verify results
    print(f"Simulation completed with {results.n_sims} sims and {results.n_players} players.")
    
    # Check shape
    assert results.results_matrix.shape == (n_sims, len(players_df))
    
    # Check stats
    stats = results.get_player_stats()
    print("\nPlayer Stats Summary (first 5 players):")
    print(stats.head())
    
    # Save results
    output_path = "data/processed/test_sim_results.parquet"
    results.save_to_parquet(output_path)
    print(f"Results saved to {output_path}")
    
    # Load and check if saved correctly
    loaded_df = pd.read_parquet(output_path)
    print(f"Loaded {len(loaded_df)} simulation rows from parquet.")
    assert len(loaded_df) == n_sims * len(players_df)
    
    print("\nSimulation pipeline test PASSED!")

def test_ownership_consistency():
    """
    Verify that compute_heuristic_ownership produces consistent results
    whether called with the pipeline's column set (includes game_start_time,
    uses 'slot') or the eval script's column set (no game_start_time, uses
    'lineup_slot').

    This catches the class of silent mismatch where a column available to the
    pipeline is absent from the eval player pool, causing the function to take
    a different code path and produce different ownership estimates.
    """
    from src.optimization.ownership import compute_heuristic_ownership

    positions = ["OF", "OF", "OF", "1B", "SS", "2B", "3B", "C", "SS"]
    records = []
    for slot in range(1, 10):
        for team, opp in [("NYY", "BOS"), ("BOS", "NYY")]:
            records.append({
                "player_id": abs(hash(f"{team}{slot}")) % 100000,
                "name": f"{team}_B{slot}",
                "position": positions[slot - 1],
                "eligible_positions": [positions[slot - 1]],
                "team": team,
                "opponent": opp,
                "salary": 4000 + slot * 200,
                "mean": 6.0 + slot * 0.3,
                "game": f"NYY@BOS 05/09/2026 07:10PM ET",
                "game_start_time": "2026-05-09T19:10:00",
                "slot": slot,
                "lineup_slot": slot,
            })
    for team, opp in [("NYY", "BOS"), ("BOS", "NYY")]:
        records.append({
            "player_id": abs(hash(f"{team}P")) % 100000,
            "name": f"{team}_P",
            "position": "P",
            "eligible_positions": ["P"],
            "team": team,
            "opponent": opp,
            "salary": 8000,
            "mean": 20.0,
            "game": "NYY@BOS 05/09/2026 07:10PM ET",
            "game_start_time": "2026-05-09T19:10:00",
            "slot": 10,
            "lineup_slot": 10,
        })

    team_totals = {"NYY": 4.8, "BOS": 4.2}
    df_full = pd.DataFrame(records)

    # Pipeline column set: game_start_time + slot (no lineup_slot)
    df_pipeline = df_full.drop(columns=["lineup_slot"])
    own_pipeline = compute_heuristic_ownership(df_pipeline, team_totals)

    # Eval script column set: lineup_slot + no game_start_time
    df_eval = df_full.drop(columns=["game_start_time", "slot"])
    own_eval = compute_heuristic_ownership(df_eval, team_totals)

    assert np.allclose(own_pipeline.sum(), own_eval.sum(), rtol=1e-6), (
        f"Ownership sums diverge: pipeline={own_pipeline.sum():.4f} "
        f"eval={own_eval.sum():.4f}"
    )
    max_diff = float(np.abs(own_pipeline - own_eval).max())
    assert max_diff < 0.05, (
        f"Max per-player ownership divergence {max_diff:.4f} between pipeline "
        f"and eval column sets — a column is being silently dropped or parsed "
        f"differently between the two calling contexts."
    )

    for own, label in [(own_pipeline, "pipeline"), (own_eval, "eval")]:
        assert (own >= 0).all(), f"Negative ownership in {label} result"
        assert (own <= 3.0).all(), f"Unreasonably large ownership in {label} result"

    print("Ownership consistency test PASSED.")


if __name__ == "__main__":
    run_test()
    test_ownership_consistency()
