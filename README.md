Two end-to-end machine learning projects demonstrating practical data science skills—from data cleaning and feature engineering to model development, evaluation, and communication of results.

The repository contains:
- `LOAN_ELIGIBITY_PREDICTICTION.ipynb` and `Loan_Prediction.csv` — a supervised **classification** project for loan approval prediction.
- `Traffic Prediction.ipynb` and `Traffic Prediction Dataset.zip` — a **regression / forecasting** project for traffic volume prediction.
---

## Introduction

- **Loan Eligibility:** Scalable, explainable screening can help lenders reduce manual workload, improve consistency, and enhance fairness audits.
- **Traffic Prediction:** Accurate short-term forecasts support congestion management, route planning, and operational decision-making.

---

## Methodology

- Built **reproducible notebooks** that load, clean, and validate tabular datasets.
- Engineered **informative features** (categorical encodings, temporal breakout for traffic, ratio/interaction features).
- Implemented **baseline and improved models** with rigorous evaluation and clear metrics.
- Produced **visualizations** for EDA, feature impact, model diagnostics, and error analysis.
- Documented assumptions, limitations, and **next steps** for productionization and fairness.

---

## Project 1 — Loan Eligibility Prediction (Classification)

### Objective
Predict whether a loan application should be approved, given applicant and loan attributes.

### Typical Workflow
1. **Data Understanding & Cleaning**
   - Load `Loan_Prediction.csv`, verify schema, handle missing values/outliers.
   - Sanity checks: label balance, leakage risks, train/test split strategy.

2. **Feature Engineering**
   - Categorical encodings (One-Hot/Ordinal), numeric scaling where appropriate.
   - Construct ratio features (e.g., income-to-loan), binning options for stability tests.

3. **Modeling**
   - Baselines: Logistic Regression, Decision Tree.
   - Stronger models: Random Forest, Gradient Boosting (e.g., XGBoost/LightGBM), Support Vector Machines.
   - Class-imbalance handling if needed (class weights / resampling).

4. **Evaluation**
   - **Primary:** ROC-AUC, F1, Precision/Recall, PR-AUC, Accuracy.
   - **Secondary:** Confusion matrix, calibration curves, feature importance.
   - **Fairness checks (recommended):** group metrics, threshold analysis.

5. **Explainability**
   - Global feature importance (model-dependent).
   - Local explanations with SHAP/LIME for selected cases.

> **Deliverables:** clean notebook with reproducible pipeline, metrics tables, plots, and interpretation notes.

---

## Project 2 — Traffic Volume Prediction (Regression / Short-Term Forecasting)

### Objective
Estimate near-term traffic volumes for a roadway/segment to support planning and operations.

### Typical Workflow
1. **Data Understanding & Cleaning**
   - Unzip and load **Traffic Prediction Dataset**, check timestamp formats, remove anomalies.
   - Handle missing intervals and unify time granularity.

2. **Feature Engineering**
   - Time features: hour, day-of-week, month, holiday/weekend flags.
   - Rolling/statistical features: lagged volumes (t-1, t-2, …), moving averages.
   - (Optional) Weather/Events if available.

3. **Modeling**
   - Baselines: Linear Regression (regularized), Decision Tree Regressor.
   - Tree ensembles: RandomForestRegressor, Gradient Boosting (e.g., XGBoost/LightGBM).
   - Time-series splits (`TimeSeriesSplit`) to avoid look-ahead bias.
   - (Optional) Classical TS (ARIMA/Prophet) or deep learning (LSTM/Temporal CNN) for further improvement.

4. **Evaluation**
   - **Primary:** MAE, RMSE, MAPE; R² for context.
   - Temporal cross-validation; error analysis by hour/day/season.
   - Residual diagnostics and drift checks.

5. **Visualization**
   - Actual vs Predicted plots, residual distributions, error by time-of-day.

> **Deliverables:** notebook with a consistent modeling framework, CV metrics, and interpretable plots for stakeholders.

---

## Methods & Techniques

- **Data Prep:** Missing value imputation, outlier handling, categorical encoding (OHE/Ordinal), scaling (Standard/MinMax), train/validation/test protocols.
- **Feature Engineering:** Ratios & interactions (loan), temporal/lag features (traffic), target-aware validation designs.
- **Model Families:** Linear/logistic models, decision trees, random forests, gradient boosting, SVM; time-series baselines and (optional) advanced models.
- **Model Selection:** Grid/random search, cross-validation (including time-aware splits), early stopping (for boosting).
- **Evaluation & Reporting:** Task-appropriate metrics, confusion matrices, ROC/PR curves, residual/error breakdowns, calibration.
- **Explainability:** Feature importance, SHAP/LIME exemplars, threshold sensitivity, fairness probes.

---

## Tools & Stack

- **Language:** Python 3.x
- **Core:** Jupyter Notebook, Pandas, NumPy
- **ML:** scikit-learn (LogReg, Trees, Ensembles, SVM), (optional) XGBoost/LightGBM
- **TS (optional):** statsmodels/Prophet; torch/keras for deep baselines
- **Viz:** Matplotlib, Seaborn
- **Utilities:** joblib (model persistence), pyjanitor (optional), category_encoders (optional)

---
