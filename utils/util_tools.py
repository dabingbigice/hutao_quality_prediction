import joblib


def load_artifacts(model_path, scaler_path):
    with open(model_path, 'rb') as f_model, open(scaler_path, 'rb') as f_scaler:
        model = joblib.load(f_model)
        scaler = joblib.load(f_scaler)
    return model, scaler