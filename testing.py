import pandas as pd
from fuzzywuzzy import process
from backend.utils.color_to_dye import get_dye_from_color
from backend.utils.model import load_sustainability_model, predict_sustainability_score
from backend.utils.model import explain_sustainability_score

df = pd.read_csv("/Users/chinmayee/Documents/ecowardrobe-project/data/processed/fabric_dye_training_data.csv", low_memory=False)

fabric_name = "Linen"
color = "yellow"
dye_name = get_dye_from_color(color)
dye_candidates = dye_name if isinstance(dye_name, list) else [dye_name]

def get_closest_match(name, options, threshold=80):
    match, score = process.extractOne(name, options)
    return match if score >= threshold else None

fabric_options = df['Fabric Name'].dropna().unique()
dye_options = df['Dye Name'].dropna().unique()
best_fabric = get_closest_match(fabric_name, fabric_options)

print("🎯 Best fabric match:", best_fabric)
print("🎯 Dye candidates:", dye_candidates)

for dye in dye_candidates:
    best_dye = get_closest_match(dye, dye_options)
    print(f"Trying with dye: {dye} → closest match: {best_dye}")
    match = df[
        (df['Fabric Name'] == best_fabric) &
        (df['Dye Name'] == best_dye)
    ]
    print("Match found?", not match.empty)
    
model, feature_cols = load_sustainability_model()


print("\n🎯 Predicting for color: yellow")
result = predict_sustainability_score("Linen", "yellow", model)
print("🟡 Sustainability Score:", result)

print("\n🎯 Predicting for color: blue")
result = predict_sustainability_score("Linen", "blue", model)
print("🔵 Sustainability Score:", result)

print("\n📊 SHAP Explanation (yellow)")
explanation = explain_sustainability_score("Linen", "yellow", model)
print(explanation)

print("\n📊 SHAP Explanation (blue)")
explanation = explain_sustainability_score("Linen", "blue", model)
print(explanation)
