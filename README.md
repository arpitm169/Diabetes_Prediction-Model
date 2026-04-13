# Type 2 Diabetes Prediction using Stacking Ensemble

## Overview

This project focuses on predicting Type 2 Diabetes using a robust machine learning pipeline that combines multiple models through stacking. The system improves predictive performance by leveraging feature engineering, class balancing, hyperparameter tuning, and threshold optimization.

The model is trained on the Pima Indians Diabetes dataset and aims to achieve strong recall and F1-score while maintaining overall accuracy.

---

## Features

- Data preprocessing and missing value handling
- Feature engineering with interaction-based features
- Feature scaling using StandardScaler
- Handling class imbalance using SMOTE
- Hyperparameter tuning using RandomizedSearchCV
- Multiple base models:
  - Support Vector Machine (SVM)
  - XGBoost Classifier
  - Extra Trees Classifier
- Stacking ensemble model
- Threshold tuning for performance optimization
- Evaluation using multiple metrics:
  - Accuracy
  - Recall
  - Precision
  - F1 Score
  - ROC-AUC
- Model and scaler saving using joblib
- Visualization:
  - Confusion matrices
  - ROC curves

---

## Dataset

The dataset used is the Pima Indians Diabetes dataset, which contains medical predictor variables such as:

- Glucose
- Blood Pressure
- BMI
- Insulin
- Age
- Pregnancies

Target variable:
- Outcome (0 = No Diabetes, 1 = Diabetes)

---

## Project Workflow

1. Load dataset
2. Replace invalid zero values with median values
3. Perform feature engineering
4. Split dataset into training and testing sets
5. Apply feature scaling
6. Handle class imbalance using SMOTE
7. Train and tune individual models
8. Build stacking ensemble
9. Optimize classification threshold
10. Evaluate final model
11. Save trained model and results
12. Generate plots for analysis

---

