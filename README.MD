# Credit Risk Modeling & Loan Default Prediction

A machine learning pipeline for predicting the probability of loan default (Probability of Default, or PD) using the **Give Me Some Credit** dataset. The project covers end-to-end credit risk modeling — from exploratory analysis and class imbalance handling to multi-model comparison, threshold optimization, and SHAP-based explainability.

---

## Overview

Lenders need to assess the likelihood that a borrower will experience serious financial delinquency within two years. This notebook builds and evaluates three classifiers — Logistic Regression, Random Forest, and XGBoost — and selects the best-performing model to drive loan approval decisions. Rejected loans are accompanied by human-readable explanations powered by SHAP values.

---

## Dataset

**Source:** [Give Me Some Credit – Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)  
**File:** `cs-training.csv`

| Feature | Description |
|---|---|
| `SeriousDlqin2yrs` | Target: 1 = serious delinquency within 2 years, 0 = no delinquency |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card and personal line usage ratio |
| `age` | Age of the borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days past due in the last 2 years |
| `DebtRatio` | Monthly debt payments / monthly income |
| `MonthlyIncome` | Monthly gross income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days past due |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days past due in the last 2 years |
| `NumberOfDependents` | Number of dependents |

> **Note:** Update the `file_path` variable in Cell 2 to point to your local copy of `cs-training.csv`.

---

## Notebook Workflow

### 1. Exploratory Data Analysis
- Class distribution plot (highlighting class imbalance)
- Correlation heatmap across all features
- Distribution plot of `DebtRatio`

### 2. Data Preprocessing
- Missing value imputation using column medians
- One-hot encoding of categorical variables
- Feature/target split and an 80/20 stratified train-test split
- Standard scaling via `StandardScaler`

### 3. Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied to the training set to address the skewed class distribution

### 4. Model Training
Three classifiers are trained and compared:

| Model | Key Settings |
|---|---|
| Logistic Regression | Pipeline with StandardScaler, SMOTE-balanced, max_iter=5000 |
| Random Forest | 200 estimators, max_depth=12 |
| XGBoost | 200 estimators, max_depth=4, lr=0.05, class-weight balanced via `scale_pos_weight` |

### 5. Model Evaluation
- Classification report (precision, recall, F1 per class)
- ROC-AUC score
- ROC curve visualization
- Side-by-side model comparison table

### 6. Threshold Optimization
The default 0.5 classification threshold is replaced by an **optimal threshold** that maximizes the F1-score on the test set, making the loan rejection criterion more sensitive to the minority (default) class.

### 7. Loan Decision Table
A final output table is generated for all borrowers with columns:
- `PD` — Predicted probability of default
- `LGD` — Loss Given Default (proxied by `DebtRatio`)
- `EAD` — Exposure at Default (proxied by `MonthlyIncome`)
- `Loan_Decision` — 1 = approved, 0 = rejected

### 8. Explainability (SHAP)
- SHAP summary plot showing global feature importance for the XGBoost model
- Per-loan rejection explanations: for each rejected borrower, the top 3 features driving the decision are printed with their quantitative SHAP contributions

Example output:
```
Loan rejected (PD = 0.814 ≥ threshold 0.770). Top factors contributing to rejection:
NumberOfTime30-59DaysPastDueNotWorse increased PD by 0.922,
RevolvingUtilizationOfUnsecuredLines increased PD by 0.874,
NumberOfOpenCreditLinesAndLoans increased PD by 0.264
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
```

---

## Usage

1. Clone this repository and download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data).
2. Update the `file_path` variable in Cell 2 to your local path.
3. Open the notebook:
   ```bash
   jupyter notebook hackk__1_.ipynb
   ```
4. Run all cells sequentially (`Kernel > Restart & Run All`).

---

## Results Summary

The XGBoost model was selected as the final model based on AUC and F1 performance. Threshold tuning further improved recall on the minority (default) class. SHAP values provide transparent, per-applicant explanations for every loan rejection.

---

## Project Structure

```
.
├── hackk__1_.ipynb       # Main notebook
├── cs-training.csv       # Dataset (download separately from Kaggle)
└── README.md
```
