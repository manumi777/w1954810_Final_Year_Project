# Financial Fraud Detection System


> **Author:** Manumi Amarasekara | **Student ID:** w1954810 / 20221050  
>
---

##  Project Overview

This project develops, evaluates, and deploys an end-to-end machine learning pipeline for detecting fraudulent transactions in financial mobile-money data. It addresses the core challenges of severe class imbalance, feature engineering, model explainability, and real-time operational monitoring.

The pipeline is built on the **PaySim** synthetic financial dataset (6.3M transactions) and achieves a **PR-AUC of 0.932** and **ROC-AUC of 0.995** using a Random Forest classifier with domain-informed feature engineering, SMOTENC resampling, and SHAP explainability.

A **Streamlit dashboard** provides real-time fraud monitoring, and a **GitHub Actions workflow** sends automated high-alert email notifications when transactions exceed the fraud risk threshold.




## Dataset

**PaySim** — Synthetic Financial Mobile Money Transactions  
- **Source:** [Kaggle — Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset)  
- **Full size:** 6,362,620 transactions  
- **Filtered size:** 2,770,409 transactions (TRANSFER + CASH_OUT only)  
- **Fraud rate:** ~0.30% (8,213 fraud cases)  
- **Time span:** 30 days (744 hourly steps)




##  Pipeline Summary

| Stage | Description | Tools |
|---|---|---|
| Data Loading | Chunked CSV reads, type filtering, memory downcasting | pandas |
| Data Integrity | Null check, duplicate check, class distribution | pandas |
| EDA | Distribution plots, temporal analysis, correlation heatmaps, outlier detection | matplotlib, seaborn |
| Feature Engineering | Balance discrepancy features, binary risk flags, time-of-day signal | pandas |
| Network Features | Account-level graph features extracted from nameOrig/nameDest | pandas groupby |
| Resampling | SMOTENC with binary column handling inside ImbPipeline | imbalanced-learn |
| Model Training | Logistic Regression, Random Forest, Gradient Boosting | scikit-learn |
| Evaluation | ROC-AUC, PR-AUC, F1, threshold optimisation, cost-sensitive analysis | scikit-learn |
| Explainability | SHAP TreeExplainer — global bar chart and beeswarm plots | shap |
| Temporal Validation | Chronological split and concept drift simulation | scikit-learn |
| Dashboard | Real-time fraud monitoring with custom HTML/CSS | Streamlit, Plotly |
| Alerting | Automated email on high-risk transactions | GitHub Actions |

---

## 📈 Results Summary

| Model | ROC-AUC | PR-AUC | Optimal F1 | Total Cost (units) |
|---|---|---|---|---|
| Logistic Regression | ~0.975 | ~0.760 | ~0.742 | ~1,050 |
| Gradient Boosting | ~0.991 | ~0.908 | ~0.856 | ~760 |
| **Random Forest** | **~0.995** | **~0.932** | **~0.871** | **~640** |

- **Optimal threshold:** 0.27 (maximises F1-score)
- **Fraud catch rate at optimal threshold:** 99.4% (1,633 of 1,643 fraud cases)
- **PR-AUC gain from feature engineering:** +13.7 percentage points over raw features
- **PR-AUC gain from network features:** +3.9 percentage points



6DATA007W Final Year Project at the University of Westminster. All rights reserved © Manumi Amarasekara, 2025.
