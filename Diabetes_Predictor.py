"""
TYPE 2 DIABETES DETECTION USING SUPPORT VECTOR MACHINE (SVM)


Dataset: PIMA Indians Diabetes Dataset

"""


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# 2. LOAD DATASET

df = pd.read_csv(r"C:\Users\HP-PC\Desktop\python\pima\diabetes.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 3. DATA CLEANING
zero_as_missing_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_as_missing_cols] = df[zero_as_missing_cols].replace(0, np.nan)

print("\nMissing values after replacing 0 with NaN:")
print(df.isnull().sum())

for col in zero_as_missing_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 4. FEATURES AND TARGET
# Outcome:
# 0 → No Type 2 Diabetes
# 1 → Type 2 Diabetes Present
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5. TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. SVM MODEL + HYPERPARAMETER TUNING
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm_model = SVC(probability=True, random_state=42)

grid = GridSearchCV(
    svm_model,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_

print("\nBest SVM Parameters:", grid.best_params_)

# 8. MODEL EVALUATION
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. CONFUSION MATRIX PLOT
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Type 2 Diabetes")
plt.colorbar()

classes = ["No Type 2 Diabetes", "Type 2 Diabetes"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# 10. ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Type 2 Diabetes")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 11. SAVE MODEL & SCALER
joblib.dump(best_model, "svm_type2_diabetes_model.joblib")
joblib.dump(scaler, "scaler_type2_diabetes.joblib")

print("\nModel and scaler saved successfully.")

# 12. USER INPUT PREDICTION
def predict_type2_diabetes(user_input, model, scaler):
    """
    Predict Type 2 Diabetes for custom user input
    """
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        return "Type 2 Diabetes Detected", probability
    else:
        return "No Type 2 Diabetes", probability


print("\n--- TYPE 2 DIABETES PREDICTION (USER INPUT) ---")

pregnancies = float(input("Enter number of pregnancies: "))
glucose = float(input("Enter glucose level: "))
blood_pressure = float(input("Enter blood pressure: "))
skin_thickness = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
diabetes_pedigree = float(input("Enter diabetes pedigree function: "))
age = float(input("Enter age: "))

user_data = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]

result, prob = predict_type2_diabetes(user_data, best_model, scaler)

print("\nPrediction Result:", result)
print(f"Probability of Type 2 Diabetes: {prob * 100:.2f}%")

