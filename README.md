# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using Python, Scikit-learn, SMOTE, Random Forest, and FastAPI.

## Project Overview

Credit card fraud detection is a binary classification problem where the goal is to identify whether a transaction is:

- `0` → Legitimate
- `1` → Fraudulent

The main challenge in this project is the highly imbalanced dataset, where fraudulent transactions represent only a very small percentage of all transactions.

To handle this class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training data. A Random Forest classifier was then trained to classify transactions as legitimate or fraudulent.

The trained machine learning pipeline was saved using Joblib and exposed through a FastAPI REST API for prediction.

---

## Dataset

The dataset contains the following features:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `Class` column is the target variable:

```text
0 → Legitimate
1 → Fraud