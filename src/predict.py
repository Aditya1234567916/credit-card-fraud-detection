import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/fraud_detection_model.pkl"

DEFAULT_THRESHOLD = 0.70


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(
    transaction,
    threshold=DEFAULT_THRESHOLD
):

    # --------------------------------------------------------
    # Check for missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in transaction
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame(
        [
            [
                transaction[feature]
                for feature in FEATURE_NAMES
            ]
        ],
        columns=FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Get fraud probability
    # --------------------------------------------------------

    fraud_probability = float(
        model.predict_proba(data)[0][1]
    )


    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    prediction = int(
        fraud_probability >= threshold
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "prediction":
            "Fraud"
            if prediction == 1
            else "Legitimate",

        "fraud_probability":
            fraud_probability,

        "threshold":
            threshold

    }


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 60)


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        "data/creditcard.csv"
    )


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates()


    # --------------------------------------------------------
    # Select first transaction
    # --------------------------------------------------------

    transaction = (
        df.iloc[0]
        .drop("Class")
        .to_dict()
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    result = predict_transaction(
        transaction
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPrediction Result")
    print("-" * 40)

    print(
        "Prediction        :",
        result["prediction"]
    )

    print(
        "Fraud Probability :",
        round(
            result["fraud_probability"],
            4
        )
    )

    print(
        "Threshold         :",
        result["threshold"]
    )


    print("\n")
    print("=" * 60)
    print("PREDICTION COMPLETED")
    print("=" * 60)