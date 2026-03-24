import pandas as pd
import numpy as np
import os
import argparse

def build_copula(input_path="data/processed/historical_logs.parquet", output_path="data/processed/empirical_copula.parquet"):
    """
    Builds an empirical copula from historical logs.
    
    Requirements:
    1. Read historical_logs.parquet.
    2. Map to G x 10 matrix (9 batters + 1 opposing pitcher).
    3. Pitcher quantiles calculated globally.
    4. Save as empirical_copula.parquet.
    """
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Creating sample data for demonstration.")
        # Create dummy data for 1000 games
        n_games = 1000
        data = []
        for g in range(n_games):
            for team in [0, 1]:
                # 9 batters
                for slot in range(1, 10):
                    data.append({
                        'game_id': f"game_{g}",
                        'team_id': f"team_{team}",
                        'slot': slot,
                        'dk_points': np.random.normal(8, 4) if slot < 5 else np.random.normal(5, 2)
                    })
                # 1 opposing pitcher
                data.append({
                    'game_id': f"game_{g}",
                    'team_id': f"team_{team}",
                    'slot': 10,
                    'dk_points': np.random.normal(15, 8)
                })
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        df.to_parquet(input_path, index=False)
        print(f"Created sample historical logs at {input_path}")
    else:
        df = pd.read_parquet(input_path)

    print("Calculating quantiles...")
    # Calculate quantiles for each batter slot independently
    df_with_quantiles = df.copy()
    for slot in range(1, 10):
        mask = df_with_quantiles['slot'] == slot
        if mask.any():
            df_with_quantiles.loc[mask, 'quantile'] = df_with_quantiles.loc[mask, 'dk_points'].rank(pct=True, method='average')
        
    # Global pitcher quantile (slot 10)
    mask_p = df_with_quantiles['slot'] == 10
    if mask_p.any():
        df_with_quantiles.loc[mask_p, 'quantile'] = df_with_quantiles.loc[mask_p, 'dk_points'].rank(pct=True, method='average')
    
    # Pivot to G x 10 matrix
    # The requirement is G x 10, where G is the number of historical game-team observations.
    copula_matrix = df_with_quantiles.pivot(index=['game_id', 'team_id'], columns='slot', values='quantile')
    
    # Drop rows with any missing values to ensure full 10-player correlations
    initial_len = len(copula_matrix)
    copula_matrix = copula_matrix.dropna()
    final_len = len(copula_matrix)
    
    if initial_len > final_len:
        print(f"Dropped {initial_len - final_len} incomplete game-team observations.")

    # Ensure columns are ordered 1..10
    copula_matrix = copula_matrix.reindex(columns=range(1, 11))
    
    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    copula_matrix.to_parquet(output_path)
    print(f"Saved empirical copula with shape {copula_matrix.shape} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build empirical copula from historical data.")
    parser.add_argument("--input", default="data/processed/historical_logs.parquet", help="Path to historical logs")
    parser.add_argument("--output", default="data/processed/empirical_copula.parquet", help="Path to save copula matrix")
    args = parser.parse_args()
    
    build_copula(args.input, args.output)
