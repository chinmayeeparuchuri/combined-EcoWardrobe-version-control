# backend/utils/ethical_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process

# ------------------------------
# Core logic — untouched
# ------------------------------

def compute_ethics_score(row):
    score = 0
    score += 2 if row['Is Fair Wages'] else 0
    score += 2 if row['No Child Labor'] else 0
    score += 2 if row['Is Worker Safe'] else 0
    score += 2 if row['Is Worker Satisfied'] else 0
    score += 2 if row['Certifications Received'] >= 2 else 1 if row['Certifications Received'] == 1 else 0
    return score

def train_ethics_model():
    df = pd.read_csv('/Users/chinmayee/Documents/ecowardrobe-project/data/processed/clean_manufacturers.csv')
    df['Ethical Score'] = df.apply(compute_ethics_score, axis=1)

    features = [
        'Is Fair Wages', 'No Child Labor', 'Is Worker Safe',
        'Is Worker Satisfied', 'Certifications Received',
        'Years Since Establishment'
    ]
    X = df[features]
    y = df['Ethical Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print("✅ Ethical Model - Evaluation Metrics:")
    print(f"   - MAE  : {mae:.3f}")
    print(f"   - RMSE : {rmse:.3f}")
    print(f"   - R²   : {r2:.3f}")
    print(f"   - MAPE : {mape:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/ethical_score_model.pkl')

    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title('Feature Importance - Ethical Score Model')
    plt.tight_layout()
    plt.savefig('models/ethical_feature_importance.png')
    print("📊 Feature importance saved at models/ethical_feature_importance.png")

# ------------------------------
# API Utility Functions
# ------------------------------



def load_ethical_model():
    return joblib.load('/Users/chinmayee/Documents/ecowardrobe-project/models/ethical_score_model.pkl')

def fuzzy_match_manufacturer(input_name, choices, threshold=85):
    if not input_name:
        return None
    match, score = process.extractOne(input_name, choices)
    return match if score >= threshold else None


def predict_ethics_score(manufacturer_name, model):
    if not manufacturer_name:
        return {"score": None, "error": "Manufacturer name not provided."}

    df = pd.read_csv('/Users/chinmayee/Documents/ecowardrobe-project/data/processed/clean_manufacturers.csv')
    all_names = df['Manufacturer Name'].tolist()
    matched_name = fuzzy_match_manufacturer(manufacturer_name, all_names)

    if matched_name is None:
        return {"score": None, "error": "Manufacturer not found (even with fuzzy match)"}

    match = df[df['Manufacturer Name'] == matched_name]
    features = [
        'Is Fair Wages', 'No Child Labor', 'Is Worker Safe',
        'Is Worker Satisfied', 'Certifications Received',
        'Years Since Establishment'
    ]
    X = match[features]
    score = model.predict(X)[0]
    return {"score": round(score, 2), "matched_name": matched_name}

def explain_ethics_score(manufacturer_name, model):
    if not manufacturer_name:
        return {"score": None, "error": "Manufacturer name not provided."}

    df = pd.read_csv('/Users/chinmayee/Documents/ecowardrobe-project/data/processed/clean_manufacturers.csv')
    all_names = df['Manufacturer Name'].tolist()
    matched_name = fuzzy_match_manufacturer(manufacturer_name, all_names)

    if matched_name is None:
        return {"score": None, "error": "Manufacturer not found (even with fuzzy match)"}

    match = df[df['Manufacturer Name'] == matched_name]
    features = [
        'Is Fair Wages', 'No Child Labor', 'Is Worker Safe',
        'Is Worker Satisfied', 'Certifications Received',
        'Years Since Establishment'
    ]
    X = match[features]

    importances = model.feature_importances_
    values = X.iloc[0].values
    explanation = [
        f"{feat}: Value={val}, Importance={round(imp, 3)}"
        for feat, val, imp in zip(features, values, importances)
    ]
    explanation.sort(key=lambda x: float(x.split("Importance=")[-1]), reverse=True)
    return explanation
