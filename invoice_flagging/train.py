from data_preprocessing import (
    load_invoice_data,
    split_data,
    scale_features,
    apply_labels
)

from model_evaluation import (
    train_random_forest,
    evaluate_classifier
)

import joblib
from pathlib import Path


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # Load data
    df = load_invoice_data()

    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    model_dir = Path("/Users/PRANAV/Desktop/ML project/invoice_flagging/models/")
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        model_dir / 'scaler.pkl'
    )

    model = train_random_forest(
        X_train_scaled,
        y_train
    )

    evaluate_classifier(
        model,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    joblib.dump(
        model,
        model_dir / 'predict_flag_invoice.pkl'
    )
    print(f"Best model saved: {model_dir / 'predict_flag_invoice.pkl'}")


if __name__ == "__main__":
    main()