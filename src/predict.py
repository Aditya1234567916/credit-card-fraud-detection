import joblib
import pandas as pd

# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "models/fraud_detection_model.pkl"

# ==============================
# FEATURES USED DURING TRAINING
# ==============================

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load(MODEL_PATH)


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_transaction(transaction, threshold=0.5):

    # Check for missing features
    missing_features = [
        feature for feature in FEATURE_NAMES
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Create DataFrame in the exact feature order
    data = pd.DataFrame(
        [[transaction[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    # Get fraud probability
    fraud_probability = float(
        model.predict_proba(data)[0][1]
    )

    # Apply threshold
    prediction = int(
        fraud_probability >= threshold
    )

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": fraud_probability,
        "threshold": threshold
    }


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    df = pd.read_csv("data/creditcard.csv")

    # Take first transaction
    transaction = (
        df.iloc[0]
        .drop("Class")
        .to_dict()
    )

    result = predict_transaction(transaction)

    print("\nPrediction Result")
    print("========================")
    print("Prediction        :", result["prediction"])
    print(
        "Fraud Probability :",
        round(result["fraud_probability"], 4)
    )
    print("Threshold         :", result["threshold"])