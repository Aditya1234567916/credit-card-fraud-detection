import os

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"

MODEL_PATH = "models/fraud_detection_model.pkl"

RESULTS_PATH = "results"

RANDOM_STATE = 42


os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)


# Remove duplicates exactly as during training
df = df.drop_duplicates()


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# ============================================================
# 4. SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

y_probability = model.predict_proba(
    X_test
)[:, 1]


# Default threshold = 0.50

y_pred = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


tn, fp, fn, tp = cm.ravel()


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


print("\nConfusion matrix:")
print(cm)


# ============================================================
# 8. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Legitimate",
        "Fraud"
    ],
    yticklabels=[
        "Legitimate",
        "Fraud"
    ]
)

plt.title(
    "Fraud Detection Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


rf_model = model.named_steps[
    "classifier"
]


feature_importance = (
    rf_model.feature_importances_
)


importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": feature_importance
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 15 features:")

print(
    importance_df.head(15)
)


# ============================================================
# 11. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (
    importance_df
    .head(15)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Random Forest Features"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


results = []


for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    results.append(
        {
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )


threshold_df = pd.DataFrame(
    results
)


print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    f"{RESULTS_PATH}/threshold_analysis.csv",
    index=False
)


# ============================================================
# 14. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/precision_recall_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved in:")
print("results/")