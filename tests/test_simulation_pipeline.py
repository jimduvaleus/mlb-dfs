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

if __name__ == "__main__":
    run_test()
