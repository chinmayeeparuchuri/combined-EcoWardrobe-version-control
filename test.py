import pandas as pd
from fuzzywuzzy import process
from backend.utils.color_to_dye import get_dye_from_color
from backend.utils.model import load_sustainability_model, predict_sustainability_score
from backend.utils.user_input_parser import extract_fabrics

# Load training data
df = pd.read_csv("data/processed/fabric_dye_training_data.csv", low_memory=False)

# Inputs
fabric_blend = "Cotton (24%) / Nylon  (76%)"
color = "Blue"
fabric_weights = extract_fabrics(fabric_blend)
dye_candidates = get_dye_from_color(color)
dye_candidates = dye_candidates if isinstance(dye_candidates, list) else [dye_candidates]

print("🧵 Parsed fabric blend:", fabric_weights)
print("🎨 Dye candidates:", dye_candidates)

# Available training entries
fabric_options = df['Fabric Name'].dropna().unique()
dye_options = df['Dye Name'].dropna().unique()

def get_closest_match(name, options, threshold=80):
    match, score = process.extractOne(name, options)
    return match if score >= threshold else None

# ✅ Load the model AND feature columns
model, feature_cols = load_sustainability_model()

# Score accumulator
scores = []

for dye in dye_candidates:
    best_dye = get_closest_match(dye, dye_options)
    if not best_dye:
        continue

    for fabric, weight in fabric_weights.items():
        best_fabric = get_closest_match(fabric, fabric_options)
        if not best_fabric:
            continue

        # Filter the matching row
        match_row = df[(df['Fabric Name'] == best_fabric) & (df['Dye Name'] == best_dye)]
        if match_row.empty:
            continue

        # Extract features and predict
        features = match_row[feature_cols].iloc[0]
        score = model.predict([features])[0]
        scores.append(score * weight)

        print(f"✅ Matched {fabric} + {dye} → score = {score:.2f} (weighted {score * weight:.2f})")

# Final blended score
if scores:
    final_score = sum(scores)
    print(f"\n🌱 Final sustainability score (weighted): {final_score:.2f}")
else:
    print("❌ No valid fabric-dye matches found.")
