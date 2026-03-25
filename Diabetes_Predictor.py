



# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier



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

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv(r"C:\Users\HP-PC\Downloads\pima\diabetes.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ===============================
# 3. DATA CLEANING
# ===============================
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

# ===============================
# 3.5 FEATURE ENGINEERING
# ===============================
df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
df["Age_BMI"] = df["Age"] * df["BMI"]
df["Glucose_Age"] = df["Glucose"] * df["Age"]
df["BMI_Age_Ratio"] = df["BMI"] / df["Age"]
df["Glucose_Insulin"] = df["Glucose"] * df["Insulin"]
df["BMI_Squared"] = df["BMI"] ** 2
df["Age_Squared"] = df["Age"] ** 2
df["Preg_Age"] = df["Pregnancies"] * df["Age"]
df["Glucose_per_Age"] = df["Glucose"] / df["Age"]
df["BMI_Insulin"] = df["BMI"] * df["Insulin"]
df["BP_BMI"] = df["BloodPressure"] * df["BMI"]
df["Age_Group"] = (df["Age"] > 50).astype(int)


# ===============================
# 4. FEATURES AND TARGET
# ===============================
# Outcome:
# 0 → No Type 2 Diabetes
# 1 → Type 2 Diabetes Present
X = df.drop("Outcome", axis=1)
y = df["Outcome"]



# ===============================
# 5. TRAIN–TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# 6. FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 6.5 HANDLE CLASS IMBALANCE (SMOTE)
# ===============================
smote = SMOTE(sampling_strategy=1.0, random_state=42)




# ===============================
# 7. SVM MODEL + HYPERPARAMETER TUNING
# ===============================

param_grid = {
    "C": [10, 50, 100, 200, 500],
    "gamma": ["scale", 0.001, 0.01, 0.1],
    "kernel": ["rbf"]
}

svm_model = SVC(
    probability=True,
    class_weight="balanced",
    random_state=42
)


from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid = GridSearchCV(
    svm_model,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)



grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_

print("\nBest SVM Parameters:", grid.best_params_)


# ===============================
# 7B. XGBOOST WITH GRIDSEARCH 🔥
# ===============================

# Apply SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

param_grid_xgb = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    ),
    param_grid_xgb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train FIRST
grid_xgb.fit(X_train_smote, y_train_smote)

# Then get best model
best_xgb = grid_xgb.best_estimator_

print("\nBest XGBoost Params:", grid_xgb.best_params_)

# THEN predict
y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

# Metrics
print("\n--- XGBOOST RESULTS ---")
print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGB F1:", f1_score(y_test, y_pred_xgb))
print("XGB ROC AUC:", roc_auc_score(y_test, y_proba_xgb))
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack_model = StackingClassifier(
    estimators=[
        ("svm", best_model),
        ("xgb", best_xgb)
    ],
    final_estimator=LogisticRegression()
)

stack_model.fit(X_train_scaled, y_train)

y_pred_stack = stack_model.predict(X_test_scaled)

# ===============================
# 7C. STACKING RESULTS 🔥
# ===============================

y_proba_stack = stack_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- STACKING RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("F1:", f1_score(y_test, y_pred_stack))
print("ROC AUC:", roc_auc_score(y_test, y_proba_stack))
# ===============================
# 8. MODEL EVALUATION
# ===============================
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ===============================
# 9. CONFUSION MATRIX PLOT
# ===============================
def plot_cm(cm, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    classes = ["No Type 2 Diabetes", "Type 2 Diabetes"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12, fontweight="bold"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
cm_svm = confusion_matrix(y_test, y_pred)
plot_cm(cm_svm, "SVM Confusion Matrix")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plot_cm(cm_xgb, "XGBoost Confusion Matrix")
cm_stack = confusion_matrix(y_test, y_pred_stack)
plot_cm(cm_stack, "Stacking Confusion Matrix")




# ===============================
# 10. ROC CURVE
# ===============================
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

# ===============================
# 11. SAVE MODEL & SCALER
# ===============================
joblib.dump(best_model, "svm_type2_diabetes_model.joblib")
joblib.dump(scaler, "scaler_type2_diabetes.joblib")
print("\n--- MODEL COMPARISON ---")
print("SVM F1:", f1)
print("XGBoost F1:", f1_score(y_test, y_pred_xgb))


print("\nModel and scaler saved successfully.")




