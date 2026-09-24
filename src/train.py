import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/creditcard.csv"
RESULTS_PATH = "results"


# Create results folder if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("\nTotal missing values:", total_missing)

if total_missing == 0:
    print("No missing values found.")
else:
    print("\nColumns containing missing values:")
    print(
        missing_values[
            missing_values > 0
        ]
    )


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate rows...")

original_size = len(df)

df = df.drop_duplicates()

new_size = len(df)

removed = original_size - new_size

print("Original rows:", original_size)
print("Rows after removing duplicates:", new_size)
print("Duplicates removed:", removed)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

class_counts = df["Class"].value_counts()

print("\nTransaction counts:")
print(class_counts)

class_percentages = (
    df["Class"]
    .value_counts(normalize=True)
    * 100
)

print("\nTransaction percentages:")
print(class_percentages)


# ============================================================
# 8. FRAUD / LEGITIMATE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Class"
)

plt.title(
    "Legitimate vs Fraudulent Transactions"
)

plt.xlabel(
    "Transaction Class"
)

plt.ylabel(
    "Number of Transactions"
)

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"]
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TRANSACTION AMOUNT ANALYSIS")
print("=" * 60)

print("\nOverall amount statistics:")

print(
    df["Amount"].describe()
)


# ============================================================
# 10. AMOUNT BY CLASS
# ============================================================

amount_by_class = (
    df.groupby("Class")["Amount"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
)

print("\nAmount statistics by transaction class:")

print(amount_by_class)


# ============================================================
# 11. TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title(
    "Transaction Amount Distribution"
)

plt.xlabel(
    "Transaction Amount"
)

plt.ylabel(
    "Density"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/amount_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. LOG TRANSFORMED AMOUNT
# ============================================================

df["Amount_Log"] = np.log1p(
    df["Amount"]
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Amount_Log",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title(
    "Log-Transformed Transaction Amount Distribution"
)

plt.xlabel(
    "Log(1 + Amount)"
)

plt.ylabel(
    "Density"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/log_amount_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation = (
    df.corr(numeric_only=True)["Class"]
    .sort_values(
        ascending=False
    )
)

print("\nFeatures most positively correlated with fraud:")

print(
    correlation.head(10)
)

print("\nFeatures most negatively correlated with fraud:")

print(
    correlation.tail(10)
)


# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(16, 12))

correlation_matrix = df.corr(
    numeric_only=True
)

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_PATH}/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)

print("\nShape after duplicate removal:")

print(df.shape)

print("\nClass distribution after duplicate removal:")

print(
    df["Class"].value_counts()
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("EDA AND DATA CLEANING COMPLETED")
print("=" * 60)

print("\nCharts saved inside:")
print(
    f"{RESULTS_PATH}/"
)