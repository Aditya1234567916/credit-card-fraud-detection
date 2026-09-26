# ============================================================
# CREDIT CARD FRAUD DETECTION
# MODEL EVALUATION
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"

MODEL_PATH = "models/fraud_detection_model.pkl"

RESULTS_DIR = "results"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# STEP 2: CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

print("\n")
print("=" * 80)
print("STEP 1 - LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# STEP 4: REMOVE DUPLICATES
# ============================================================

print("\n")
print("=" * 80)
print("STEP 2 - REMOVING DUPLICATES")
print("=" * 80)

df = df.drop_duplicates()

df = df.reset_index(drop=True)

print("\nDataset shape after duplicate removal:")
print(df.shape)


# ============================================================
# STEP 5: FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# ============================================================
# STEP 6: SAME TRAIN-TEST SPLIT
# ============================================================

print("\n")
print("=" * 80)
print("STEP 3 - CREATING TEST DATA")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:")
print(X_train.shape[0])

print("\nTesting samples:")
print(X_test.shape[0])


# ============================================================
# STEP 7: LOAD SAVED MODEL
# ============================================================

print("\n")
print("=" * 80)
print("STEP 4 - LOADING SAVED MODEL")
print("=" * 80)

model = joblib.load(
    MODEL_PATH
)

print("\nModel loaded successfully.")

print("Model:")
print(model)


# ============================================================
# STEP 8: GENERATE PREDICTIONS
# ============================================================

print("\n")
print("=" * 80)
print("STEP 5 - GENERATING PREDICTIONS")
print("=" * 80)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# Default threshold

DEFAULT_THRESHOLD = 0.50


y_pred = (
    y_probability >= DEFAULT_THRESHOLD
).astype(int)


# ============================================================
# STEP 9: CALCULATE FINAL METRICS
# ============================================================

print("\n")
print("=" * 80)
print("STEP 6 - MODEL PERFORMANCE")
print("=" * 80)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\nAccuracy  :", round(accuracy, 4))

print("Precision :", round(precision, 4))

print("Recall    :", round(recall, 4))

print("F1 Score  :", round(f1, 4))

print("ROC-AUC   :", round(roc_auc, 4))

print("PR-AUC    :", round(pr_auc, 4))


# ============================================================
# STEP 10: CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 80)
print("STEP 7 - CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# STEP 11: SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.colorbar()

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved."
)


# ============================================================
# STEP 12: FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 8 - FEATURE IMPORTANCE")
print("=" * 80)


# The saved model is an imblearn Pipeline.
# The Random Forest classifier is inside:
#
# model.named_steps["classifier"]
#

classifier = model.named_steps[
    "classifier"
]


feature_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        classifier.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 15 important features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# STEP 13: SAVE FEATURE IMPORTANCE
# ============================================================

plt.figure(
    figsize=(10, 7)
)

top_features = feature_importance.head(15)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Feature Importances - Random Forest"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "feature_importance.png"
    ),
    dpi=300
)

plt.close()


print(
    "\nFeature importance chart saved."
)


# ============================================================
# STEP 14: THRESHOLD ANALYSIS
# ============================================================

print("\n")
print("=" * 80)
print("STEP 9 - THRESHOLD ANALYSIS")
print("=" * 80)


threshold_results = []


for threshold in [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]:

    threshold_predictions = (
        y_probability >= threshold
    ).astype(int)


    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )


    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )


    threshold_f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )


    threshold_results.append({

        "Threshold":
            threshold,

        "Precision":
            threshold_precision,

        "Recall":
            threshold_recall,

        "F1 Score":
            threshold_f1

    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\nThreshold analysis:")

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# STEP 15: FIND HIGHEST F1 THRESHOLD
# ============================================================

best_threshold_row = threshold_df.loc[
    threshold_df["F1 Score"].idxmax()
]


best_threshold = best_threshold_row[
    "Threshold"
]


best_threshold_f1 = best_threshold_row[
    "F1 Score"
]


print("\n")
print("Highest F1 among tested thresholds:")

print(
    "Threshold:",
    best_threshold
)

print(
    "F1 Score:",
    round(
        best_threshold_f1,
        4
    )
)


# ============================================================
# STEP 16: SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "threshold_analysis.csv"
    ),
    index=False
)


print(
    "\nThreshold analysis saved."
)


# ============================================================
# STEP 17: PRECISION-RECALL CURVE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 10 - PRECISION RECALL CURVE")
print("=" * 80)


precision_values, recall_values, thresholds = (
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
    "Precision-Recall Curve - Random Forest"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.close()


print(
    "\nPrecision-Recall curve saved."
)


# ============================================================
# STEP 18: FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)


print("\nFinal model:")
print("Random Forest")


print("\nDefault threshold:")
print(DEFAULT_THRESHOLD)


print("\nAccuracy:")
print(round(accuracy, 4))


print("\nPrecision:")
print(round(precision, 4))


print("\nRecall:")
print(round(recall, 4))


print("\nF1 Score:")
print(round(f1, 4))


print("\nROC-AUC:")
print(round(roc_auc, 4))


print("\nPR-AUC:")
print(round(pr_auc, 4))


print("\nHighest tested threshold by F1:")
print(best_threshold)


print("\nResults saved in:")
print(RESULTS_DIR)


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)