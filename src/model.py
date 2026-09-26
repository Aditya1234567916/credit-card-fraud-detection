# ============================================================
# CREDIT CARD FRAUD DETECTION
# MODEL TRAINING AND COMPARISON
# ============================================================

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
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

from xgboost import XGBClassifier


# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"

MODEL_PATH = "models/fraud_detection_model.pkl"

COMPARISON_PATH = "results/model_comparison.csv"

RANDOM_STATE = 42

TEST_SIZE = 0.20

SMOTE_RATIO = 0.5


# ============================================================
# STEP 3: CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)

os.makedirs("results", exist_ok=True)


# ============================================================
# STEP 4: LOAD DATASET
# ============================================================

print("\n")
print("=" * 80)
print("STEP 1 - LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 5: BASIC DATA INFORMATION
# ============================================================

print("\n")
print("=" * 80)
print("STEP 2 - BASIC DATA INFORMATION")
print("=" * 80)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# ============================================================
# STEP 6: REMOVE DUPLICATES
# ============================================================

print("\n")
print("=" * 80)
print("STEP 3 - REMOVING DUPLICATES")
print("=" * 80)

before_duplicates = len(df)

df = df.drop_duplicates()

df = df.reset_index(drop=True)

after_duplicates = len(df)

removed_duplicates = before_duplicates - after_duplicates

print("\nRows before duplicate removal:")
print(before_duplicates)

print("\nRows after duplicate removal:")
print(after_duplicates)

print("\nDuplicates removed:")
print(removed_duplicates)


# ============================================================
# STEP 7: CHECK MISSING VALUES AGAIN
# ============================================================

print("\n")
print("=" * 80)
print("STEP 4 - CHECKING MISSING VALUES")
print("=" * 80)

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("\nTotal missing values:", total_missing)

if total_missing == 0:
    print("No missing values found.")
else:
    print("\nColumns containing missing values:")
    print(missing_values[missing_values > 0])


# ============================================================
# STEP 8: CHECK TARGET DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("STEP 5 - CHECKING CLASS DISTRIBUTION")
print("=" * 80)

class_counts = df["Class"].value_counts()

print("\nClass counts:")
print(class_counts)

print("\nClass percentages:")

class_percentages = df["Class"].value_counts(normalize=True) * 100

print(class_percentages)


legitimate_count = int(class_counts.get(0, 0))

fraud_count = int(class_counts.get(1, 0))


print("\nLegitimate transactions:", legitimate_count)

print("Fraudulent transactions:", fraud_count)

print(
    "Fraud percentage:",
    round((fraud_count / len(df)) * 100, 4),
    "%"
)


# ============================================================
# STEP 9: SEPARATE FEATURES AND TARGET
# ============================================================

print("\n")
print("=" * 80)
print("STEP 6 - SEPARATING FEATURES AND TARGET")
print("=" * 80)

X = df.drop("Class", axis=1)

y = df["Class"]


print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nNumber of features:")
print(X.shape[1])


# ============================================================
# IMPORTANT
# ============================================================
#
# The model uses:
#
# Time
# V1 ... V28
# Amount
#
# Class is the target.
#
# Amount_Log is NOT used because it was only created
# during EDA in train.py.
#
# ============================================================


# ============================================================
# STEP 10: TRAIN-TEST SPLIT
# ============================================================

print("\n")
print("=" * 80)
print("STEP 7 - TRAIN TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


print("\nTraining class distribution:")

print(y_train.value_counts())


print("\nTesting class distribution:")

print(y_test.value_counts())


# ============================================================
# STEP 11: EXPLAIN WHY STRATIFICATION IS USED
# ============================================================

print("\n")
print("=" * 80)
print("STEP 8 - STRATIFICATION CHECK")
print("=" * 80)

train_fraud_percentage = y_train.mean() * 100

test_fraud_percentage = y_test.mean() * 100

print(
    "\nFraud percentage in training data:",
    round(train_fraud_percentage, 4),
    "%"
)

print(
    "Fraud percentage in testing data:",
    round(test_fraud_percentage, 4),
    "%"
)


# ============================================================
# STEP 12: DEFINE LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 9 - DEFINING LOGISTIC REGRESSION")
print("=" * 80)

logistic_pipeline = Pipeline(
    steps=[
        
        # ----------------------------------------------------
        # Feature scaling
        # ----------------------------------------------------
        
        (
            "scaler",
            StandardScaler()
        ),

        # ----------------------------------------------------
        # Handle class imbalance
        # ----------------------------------------------------

        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE
            )
        ),

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# STEP 13: DEFINE RANDOM FOREST PIPELINE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 10 - DEFINING RANDOM FOREST")
print("=" * 80)

random_forest_pipeline = Pipeline(
    steps=[

        # ----------------------------------------------------
        # Handle class imbalance
        # ----------------------------------------------------

        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE
            )
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

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


# ============================================================
# STEP 14: DEFINE XGBOOST PIPELINE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 11 - DEFINING XGBOOST")
print("=" * 80)

xgboost_pipeline = Pipeline(
    steps=[

        # ----------------------------------------------------
        # Handle class imbalance
        # ----------------------------------------------------

        (
            "smote",
            SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE
            )
        ),

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# STEP 15: STORE MODELS
# ============================================================

models = {

    "Logistic Regression":
        logistic_pipeline,

    "Random Forest":
        random_forest_pipeline,

    "XGBoost":
        xgboost_pipeline

}


# ============================================================
# STEP 16: TRAIN MODELS
# ============================================================

print("\n")
print("=" * 80)
print("STEP 12 - MODEL TRAINING")
print("=" * 80)


results = []

trained_models = {}


for model_name, model in models.items():

    print("\n")
    print("#" * 80)

    print("TRAINING MODEL:")
    print(model_name)

    print("#" * 80)

    start_time = time.time()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    end_time = time.time()


    training_time = end_time - start_time


    # ========================================================
    # STEP 17: CALCULATE METRICS
    # ========================================================

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


    # ========================================================
    # STEP 18: DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("-" * 60)

    print("MODEL PERFORMANCE")

    print("-" * 60)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"ROC-AUC        : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : {pr_auc:.4f}"
    )

    print(
        f"Training Time  : {training_time:.2f} seconds"
    )


    # ========================================================
    # STEP 19: CONFUSION MATRIX
    # ========================================================

    print("\n")
    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)


    # ========================================================
    # STEP 20: CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # ========================================================
    # STEP 21: SAVE RESULTS
    # ========================================================

    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc,

        "Training Time":
            training_time

    })


    # Save trained model in memory

    trained_models[
        model_name
    ] = model


# ============================================================
# STEP 22: CREATE MODEL COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 80)
print("STEP 13 - MODEL COMPARISON")
print("=" * 80)


results_df = pd.DataFrame(
    results
)


print("\nAll model results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# STEP 23: SORT MODELS BY F1 SCORE
# ============================================================

results_sorted = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n")
print("Models sorted by F1 Score:")

print(
    results_sorted.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# STEP 24: SELECT MODEL
# ============================================================

best_model_name = results_sorted.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n")
print("=" * 80)
print("STEP 14 - SELECTED MODEL")
print("=" * 80)

print(
    "\nSelected model:",
    best_model_name
)

print(
    "\nSelection criterion:"
)

print(
    "Highest F1 Score among the tested models."
)


# ============================================================
# STEP 25: DISPLAY SELECTED MODEL METRICS
# ============================================================

selected_result = results_sorted.iloc[0]


print("\n")
print("Selected model performance:")

print(
    f"Accuracy  : {selected_result['Accuracy']:.4f}"
)

print(
    f"Precision : {selected_result['Precision']:.4f}"
)

print(
    f"Recall    : {selected_result['Recall']:.4f}"
)

print(
    f"F1 Score  : {selected_result['F1 Score']:.4f}"
)

print(
    f"ROC-AUC   : {selected_result['ROC-AUC']:.4f}"
)

print(
    f"PR-AUC    : {selected_result['PR-AUC']:.4f}"
)


# ============================================================
# STEP 26: SAVE BEST MODEL
# ============================================================

print("\n")
print("=" * 80)
print("STEP 15 - SAVING BEST MODEL")
print("=" * 80)


joblib.dump(
    best_model,
    MODEL_PATH
)


print("\nBest model saved successfully.")

print(
    "Location:",
    MODEL_PATH
)


# ============================================================
# STEP 27: SAVE MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("STEP 16 - SAVING MODEL COMPARISON")
print("=" * 80)


results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print("\nModel comparison saved successfully.")

print(
    "Location:",
    COMPARISON_PATH
)


# ============================================================
# STEP 28: FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL TRAINING SUMMARY")
print("=" * 80)


print("\nDataset:")
print(df.shape)


print("\nFeatures:")
print(X.shape[1])


print("\nTraining samples:")
print(X_train.shape[0])


print("\nTesting samples:")
print(X_test.shape[0])


print("\nFraud transactions:")
print(fraud_count)


print("\nModels tested:")

for model_name in models.keys():

    print(
        "-",
        model_name
    )


print("\nSelected model:")

print(
    best_model_name
)


print("\nSaved model:")

print(
    MODEL_PATH
)


print("\nComparison file:")

print(
    COMPARISON_PATH
)


print("\n")
print("=" * 80)
print("CREDIT CARD FRAUD DETECTION TRAINING COMPLETED")
print("=" * 80)