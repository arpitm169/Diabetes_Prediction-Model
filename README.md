Type 2 Diabetes Detection using Machine Learning (SVM)
Project Overview

This project implements a machine learning–based system for the early detection of Type 2 Diabetes using clinical and physiological data. A Support Vector Machine (SVM) classifier is trained on the PIMA Indians Diabetes Dataset to learn patterns associated with diabetes and predict the condition for unseen patient data.

Although the dataset does not explicitly label diabetes type, it is widely treated as Type 2 Diabetes in academic and research contexts due to the adult patient population and the presence of insulin-resistance–related features such as BMI, glucose level, and insulin.

Objectives

To preprocess and clean medical data

To train an accurate machine learning model for Type 2 Diabetes detection

To evaluate the model using standard performance metrics

To enable prediction for custom (unseen) patient inputs

To persist the trained model for future deployment

Dataset Description

Dataset Name: PIMA Indians Diabetes Dataset

Features

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target Variable

Outcome = 1: Type 2 Diabetes Present

Outcome = 0: No Type 2 Diabetes

Data Preprocessing

Medically invalid zero values in the following attributes are treated as missing values:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

Missing values are handled using median imputation

Feature scaling is applied using StandardScaler to normalize the input features

Machine Learning Model

Algorithm: Support Vector Machine (SVM)

Kernel Functions: Linear and Radial Basis Function (RBF)

Hyperparameter Optimization: GridSearchCV

Train–Test Split: 80% training and 20% testing with stratification

Model Evaluation

The trained model is evaluated using the following metrics:

Accuracy

ROC–AUC Score

Confusion Matrix

Precision, Recall, and F1-score

Receiver Operating Characteristic (ROC) Curve

Typical performance achieved on this dataset:

Accuracy in the range of 75% to 82%

ROC–AUC in the range of 0.78 to 0.85

Custom Input Prediction

The system supports prediction for new patient data by:

Accepting user-provided clinical inputs

Applying the same feature scaling used during training

Predicting the presence or absence of Type 2 Diabetes along with a probability score

This enables real-world applicability and future system deployment.

Model Persistence

The following files are saved after training:

svm_type2_diabetes_model.joblib: Trained SVM model

scaler_type2_diabetes.joblib: Feature scaler

These files allow the model to be reused without retraining.

Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

