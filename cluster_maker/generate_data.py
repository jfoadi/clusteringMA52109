import pandas as pd
from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data

def generate_simulated_data(num_points, cluster_stds) -> None:
    """
    Generate a simulated dataset with clusters and save to 'simulated_input.csv'.
    This file can be used as input for the demo clustering analysis.
    """
    # Create 5 feature specifications
    column_specs = [
        {"name": f"Feature_{i}", "reps": [0.0, 5.0, -5.0]} for i in range(5)
    ]

    seed_df = define_dataframe_structure(column_specs)

    # Simulate n data points
    simulated_df = simulate_data(
        seed_df=seed_df, 
        n_points=num_points, 
        cluster_std=cluster_stds, 
        random_state=42
    )

    # Save the file the demo needs
    simulated_df.to_csv("simulated_input.csv", index=False)
    print("simulated_input.csv created successfully.")
