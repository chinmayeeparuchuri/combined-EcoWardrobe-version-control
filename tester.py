import pandas as pd

fabric_dye_path = "/Users/chinmayee/Documents/ecowardrobe-project/data/processed/fabric_dye_training_data.csv"

df = pd.read_csv(fabric_dye_path, low_memory=False)

# Drop the problematic column
if "Natural Dyes vs Synthetic Blending Sensitivity" in df.columns:
    df = df.drop(columns=["Natural Dyes vs Synthetic Blending Sensitivity"])
