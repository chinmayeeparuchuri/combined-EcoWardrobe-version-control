import pandas as pd
from thefuzz import process

# Load dye names from training data
df = pd.read_csv("data/processed/fabric_dye_training_data.csv")
known_dyes = df["Dye Name"].dropna().unique()

def match_dye_to_known_dye(input_dye, threshold=80):
    if not input_dye:
        return None

    match, score = process.extractOne(input_dye.lower(), [d.lower() for d in known_dyes])
    if score >= threshold:
        # Return original-case dye from known_dyes
        for dye in known_dyes:
            if dye.lower() == match:
                return dye
    return None
