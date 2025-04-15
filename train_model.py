# train_model.py

from backend.utils.model import load_data, prepare_features, train_model, explain_model

if __name__ == "__main__":
    df = load_data()                         # Load from default path
    X, y = prepare_features(df)              # Select input/output features
    model, X_train = train_model(X, y)       # Train model and return
    explain_model(model, X_train)            # Generate SHAP plot
