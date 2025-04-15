import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import shap
import matplotlib.pyplot as plt
import joblib
import os
from fuzzywuzzy import process

# ✅ FIXED import for get_dye_from_color (absolute import)
from backend.utils.color_to_dye import get_dye_from_color

# ------------------------------
# Utility for fuzzy match
# ------------------------------

def get_closest_match(name, options, threshold=80):
    match, score = process.extractOne(name, options)
    return match if score >= threshold else None

# ------------------------------
# Model Training & Explainability
# ------------------------------

def load_data(path='/Users/chinmayee/Documents/ecowardrobe-project/data/processed/fabric_dye_training_data.csv'):
    return pd.read_csv(path, low_memory=False)

def prepare_features(df):
    feature_cols = [
        "Water Usage", "CO2 Emissions", "Biodegradability_x", "Recyclability Score",
        "Water Consumption", "Energy Consumption", "Toxicity Level", "Biodegradability_y", "Eco-Toxicity"
    ]
    X = df[feature_cols].copy()
    y = df["Final Sustainability Score (0–10)"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X.columns = [c.replace(" ", "_") for c in X.columns]

    return X, y

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.03,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1"
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    print(f"✅ Evaluation Metrics:")
    print(f"   - MAE  : {mae:.3f}")
    print(f"   - RMSE : {rmse:.3f}")
    print(f"   - R²   : {r2:.3f}")
    print(f"   - MAPE : {mape * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sustainability_model.pkl")

    return model, X_train

def explain_model(model, X_train):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("models/shap_summary.png", bbox_inches="tight")
    print("📊 SHAP summary plot saved at models/shap_summary.png")

# ------------------------------
# API Utility Functions
# ------------------------------

def load_sustainability_model():
    model = joblib.load("/Users/chinmayee/Documents/ecowardrobe-project/models/sustainability_model.pkl")

    feature_cols = [
        "Water Usage",
        "CO2 Emissions",
        "Biodegradability_x",
        "Recyclability Score",
        "Water Consumption",
        "Energy Consumption",
        "Toxicity Level",
        "Biodegradability_y",
        "Eco-Toxicity"
    ]

    return model, feature_cols

def predict_sustainability_score(fabric, color, dye=None, model=None):
    import pandas as pd

    df = pd.read_csv("data/processed/fabric_dye_training_data.csv")
    df['Fabric Name'] = df['Fabric Name'].astype(str).str.strip().str.lower()
    df['Dye Name'] = df['Dye Name'].astype(str).str.strip().str.lower()

    fabric_options = df['Fabric Name'].dropna().unique()
    dye_options = df['Dye Name'].dropna().unique()

    def get_valid_dyes():
        if dye:
            matched_dye = get_closest_match(dye.lower(), dye_options)
            if matched_dye:
                return [matched_dye], matched_dye, "input dye"
        if color:
            dyes = get_dye_from_color(color)
            if not isinstance(dyes, list):
                dyes = [dyes]
            dye_matches = [get_closest_match(d.lower(), dye_options) for d in dyes if d]
            dye_matches = [d for d in dye_matches if d]
            if dye_matches:
                return dye_matches, dye_matches[0], "color-based mapping"
        fallback_dye = dye_options[0] if len(dye_options) > 0 else None
        return [fallback_dye], fallback_dye, "fallback dye"

    dye_candidates, actual_dye_used, source_info = get_valid_dyes()

    def extract_features(match):
        X = match[[
            "Water Usage", "CO2 Emissions", "Biodegradability_x", "Recyclability Score",
            "Water Consumption", "Energy Consumption", "Toxicity Level", "Biodegradability_y", "Eco-Toxicity"
        ]].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X.columns = [c.replace(" ", "_") for c in X.columns]
        return X

    def predict_with_fallback(best_fab, dye_candidates, weight=1):
        for dye_candidate in dye_candidates:
            match = df[(df["Fabric Name"] == best_fab) & (df["Dye Name"] == dye_candidate)]
            if not match.empty:
                X = extract_features(match)
                if X.sum().sum() > 0:
                    return model.predict(X)[0] * weight, weight

        match = df[df["Fabric Name"] == best_fab]
        if not match.empty:
            X = extract_features(match)
            if X.sum().sum() > 0:
                return model.predict(X)[0] * weight, weight

        for dye_candidate in dye_candidates:
            match = df[df["Dye Name"] == dye_candidate]
            if not match.empty:
                X = extract_features(match)
                if X.sum().sum() > 0:
                    return model.predict(X)[0] * weight, weight

        for _, row in df.iterrows():
            X = extract_features(pd.DataFrame([row]))
            if X.sum().sum() > 0:
                return model.predict(X)[0] * weight, weight

        return 0, 0

    if isinstance(fabric, dict):  # blended fabrics
        total_score = 0
        total_weight = 0

        for fab, weight in fabric.items():
            best_fab = get_closest_match(fab.lower(), fabric_options)
            if not best_fab:
                continue

            score, w = predict_with_fallback(best_fab, dye_candidates, weight)
            total_score += score
            total_weight += w

        if total_weight == 0:
            return {
                "score": 0,
                "error": "No valid fallback prediction possible",
                "dye_used": actual_dye_used,
                "source": source_info
            }

        return {
            "score": round(total_score / total_weight, 2),
            "dye_used": actual_dye_used,
            "source": source_info
        }

    else:  # single fabric
        best_fab = get_closest_match(fabric.lower(), fabric_options)
        if not best_fab:
            return {
                "score": 0,
                "error": "Fabric not found",
                "dye_used": actual_dye_used,
                "source": source_info
            }

        score, _ = predict_with_fallback(best_fab, dye_candidates)
        return {
            "score": round(score, 2),
            "dye_used": actual_dye_used,
            "source": source_info
        }


def explain_sustainability_score(fabric, color, dye=None, model=None):
    import pandas as pd
    import shap

    df = pd.read_csv("data/processed/fabric_dye_training_data.csv")
    df['Fabric Name'] = df['Fabric Name'].astype(str).str.strip().str.lower()
    df['Dye Name'] = df['Dye Name'].astype(str).str.strip().str.lower()

    fabric_options = df['Fabric Name'].dropna().unique()
    dye_options = df['Dye Name'].dropna().unique()

    def get_valid_dyes():
        if dye:
            matched_dye = get_closest_match(dye.lower(), dye_options)
            if matched_dye:
                return [matched_dye], matched_dye, "input dye"
        if color:
            dyes = get_dye_from_color(color)
            if not isinstance(dyes, list):
                dyes = [dyes]
            dye_matches = [get_closest_match(d.lower(), dye_options) for d in dyes if d]
            dye_matches = [d for d in dye_matches if d]
            if dye_matches:
                return dye_matches, dye_matches[0], "color-based mapping"
        fallback_dye = dye_options[0] if len(dye_options) > 0 else None
        return [fallback_dye], fallback_dye, "fallback dye"

    dye_candidates, actual_dye_used, source_info = get_valid_dyes()

    def extract_features(match):
        X = match[[
            "Water Usage", "CO2 Emissions", "Biodegradability_x", "Recyclability Score",
            "Water Consumption", "Energy Consumption", "Toxicity Level", "Biodegradability_y", "Eco-Toxicity"
        ]].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X.columns = [c.replace(" ", "_") for c in X.columns]
        return X

    def explain_with_fallback(best_fab, dye_candidates, weight=1):
        # Try exact match first
        for dye_candidate in dye_candidates:
            match = df[(df["Fabric Name"] == best_fab) & (df["Dye Name"] == dye_candidate)]
            if not match.empty:
                X = extract_features(match)
                if X.sum().sum() > 0:
                    return X, weight

        # Fabric-only fallback
        match = df[df["Fabric Name"] == best_fab]
        if not match.empty:
            X = extract_features(match)
            if X.sum().sum() > 0:
                return X, weight

        # Dye-only fallback
        for dye_candidate in dye_candidates:
            match = df[df["Dye Name"] == dye_candidate]
            if not match.empty:
                X = extract_features(match)
                if X.sum().sum() > 0:
                    return X, weight

        # Global fallback — find any non-zero row
        for _, row in df.iterrows():
            X = extract_features(pd.DataFrame([row]))
            if X.sum().sum() > 0:
                return X, weight

        return None, 0

    if isinstance(fabric, dict):  # blended fabrics
        explanations = []
        total_weight = 0

        for fab, weight in fabric.items():
            best_fab = get_closest_match(fab.lower(), fabric_options)
            if not best_fab:
                continue

            X, w = explain_with_fallback(best_fab, dye_candidates, weight)
            if X is not None:
                explainer = shap.Explainer(model)
                shap_values = explainer(X)
                shap_df = pd.DataFrame({
                    "feature": X.columns,
                    "contribution": shap_values.values[0],
                    "weight": w
                })
                explanations.append(shap_df)
                total_weight += w

        if explanations:
            combined = pd.concat(explanations)
            weighted = combined.groupby("feature").apply(
                lambda g: (g["contribution"] * g["weight"]).sum() / total_weight
            ).reset_index(name="contribution")
            return {
                "dye_used": actual_dye_used,
                "source": source_info,
                "explanation": weighted.sort_values("contribution", key=abs, ascending=False).to_dict(orient="records")
            }

        return {
            "error": "No valid fallback explanation possible",
            "dye_used": actual_dye_used,
            "source": source_info
        }

    else:  # single fabric
        best_fab = get_closest_match(fabric.lower(), fabric_options)
        if not best_fab:
            return {
                "error": "Fabric not found",
                "dye_used": actual_dye_used,
                "source": source_info
            }

        X, _ = explain_with_fallback(best_fab, dye_candidates)
        if X is not None:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap_df = pd.DataFrame({
                "feature": X.columns,
                "contribution": shap_values.values[0]
            })
            return {
                "dye_used": actual_dye_used,
                "source": source_info,
                "explanation": shap_df.sort_values("contribution", key=abs, ascending=False).to_dict(orient="records")
            }

        return {
            "error": f"No valid explanation available for {fabric} with any fallback",
            "dye_used": actual_dye_used,
            "source": source_info
        }
