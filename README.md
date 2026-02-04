# Credit Card Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent credit card transactions on highly imbalanced data. The project includes preprocessing, baseline and ensemble models, hyperparameter tuning, and threshold tuning to optimize recall–precision trade-offs using ROC-AUC.

## Features
- Data preprocessing and train–test split
- Baseline Logistic Regression
- Random Forest and XGBoost models
- Hyperparameter tuning with Grid Search
- Threshold tuning for cost-sensitive decisions
- Clean, modular pipeline structure

## Evaluation
The project prioritizes recall due to the high cost of missed fraud, using ROC-AUC as the primary ranking metric and threshold tuning to control operational trade-offs.

