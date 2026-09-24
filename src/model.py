import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_detection_model.pkl"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate transactions...")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"Rows before removing duplicates: {before}")
print(f"Rows after removing duplicates : {after}")
print(f"Duplicates removed             : {before - after}")


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\nSeparating features and target...")

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution before SMOTE:")

print(
    y.value_counts()
)


print("\nClass percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


print("\nTraining class distribution:")

print(
    y_train.value_counts()
)


print("\nTesting class distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 7. CREATE MACHINE LEARNING PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("CREATING ML PIPELINE")
print("=" * 70)


pipeline = Pipeline(
    steps=[

        # --------------------------------------------------
        # STEP 1: FEATURE SCALING
        # --------------------------------------------------

        (
            "scaler",

            StandardScaler()
        ),


        # --------------------------------------------------
        # STEP 2: HANDLE CLASS IMBALANCE
        # --------------------------------------------------

        (
            "smote",

            SMOTE(
                random_state=RANDOM_STATE,
                sampling_strategy=0.5
            )
        ),


        # --------------------------------------------------
        # STEP 3: RANDOM FOREST
        # --------------------------------------------------

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=300,

                max_depth=15,

                min_samples_split=5,

                min_samples_leaf=2,

                max_features="sqrt",

                random_state=RANDOM_STATE,

                n_jobs=-1
            )
        )
    ]
)


print("\nPipeline created successfully.")


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print("\nTraining started...")

pipeline.fit(
    X_train,
    y_train
)

print("\nTraining completed successfully.")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


print("Predictions generated.")


# ============================================================
# 10. CALCULATE METRICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


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


print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n")
print(cm)


print(
    "\n"
    "                 Predicted\n"
    "               Legit  Fraud\n"
    "Actual Legit    TN     FP\n"
    "       Fraud    FN     TP\n"
)


# ============================================================
# 12. CLASSIFICATION REPORT
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
# 13. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL_PATH
)


print(
    f"\nModel saved successfully:"
)

print(
    MODEL_PATH
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print(
    f"""
Dataset size       : {len(df)}
Training samples   : {len(X_train)}
Testing samples    : {len(X_test)}

Accuracy           : {accuracy:.4f}
Precision          : {precision:.4f}
Recall             : {recall:.4f}
F1 Score           : {f1:.4f}
ROC-AUC            : {roc_auc:.4f}
PR-AUC             : {pr_auc:.4f}
"""
)