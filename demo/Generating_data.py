# demo/generate_data.py
import os
import pandas as pd
from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data

# Define the cluster centers
CENTRE_SPECS = [
    {"name": "Feature_1", "reps": [0.0, 5.0, -5.0]},
    {"name": "Feature_2", "reps": [0.0, -5.0, 5.0]},
    {"name": "Feature_3", "reps": [0.0, 2.0, -2.0]},
    {"name": "Feature_4", "reps": [0.0, -2.0, 2.0]},
    {"name": "Feature_5", "reps": [0.0, 1.0, -1.0]},
    {"name": "Feature_6", "reps": [0.0, -1.0, 1.0]}
]

# 1. Define the structure
seed_df = define_dataframe_structure(CENTRE_SPECS)

# 2. Simulate the data (1000 points around 3 centers)
final_df = simulate_data(
    seed_df, 
    n_points=1000, 
    cluster_std=1.5, # Assuming you fixed the string/float error here
    random_state=42
)

# 3. Save the data to the expected file
OUTPUT_PATH = "my_data.csv"
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Successfully generated and saved data to {OUTPUT_PATH}")