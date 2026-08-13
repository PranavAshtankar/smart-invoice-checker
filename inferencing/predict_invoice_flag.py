import joblib
import pandas as pd


MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flag prediction model.
    """

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load trained scaler.
    """

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)

    return scaler


def predict_invoice_flag(input_data):
    """
    Predict suspicious invoice flag.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted invoice flag
    """

    model = load_model()

    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)

    sample_data = {
        "invoice_quantity":       [120, 50, 200, 75, 90, 60, 150, 30, 110, 95],

    "invoice_dollars":        [18500, 9000, 30000, 12000, 15000, 8000, 22000, 4000, 17000, 14000],

    "Freight":                [300, 120, 600, 200, 250, 100, 500, 80, 320, 270],

    "total_item_quantity":    [118, 50, 198, 75, 92, 60, 150, 28, 110, 93],

    "total_item_dollars":     [18490, 8995, 29950, 12000, 15100, 8000, 22000, 3900, 17000, 13950]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)