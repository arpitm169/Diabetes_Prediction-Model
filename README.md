# Type 2 Diabetes Detection Using Machine Learning (SVM)

## Project Overview
This project presents a machine learning–based system for the early detection of Type 2 Diabetes using clinical and physiological data. The system employs a Support Vector Machine (SVM) classifier trained on the PIMA Indians Diabetes Dataset to identify patterns associated with diabetes and predict the condition for unseen patient data.

Although the dataset does not explicitly distinguish between diabetes types, it is widely treated as Type 2 Diabetes in academic and research contexts due to the adult patient population and insulin-resistance–related features.

---

## Objectives
- Preprocess and clean clinical healthcare data  
- Train an accurate machine learning model for Type 2 Diabetes detection  
- Evaluate model performance using standard classification metrics  
- Enable prediction for custom patient inputs  
- Save the trained model for future reuse  

---

## Dataset Description
**Dataset Name:** PIMA Indians Diabetes Dataset  

The dataset consists of diagnostic measurements collected from adult female patients and is widely used for diabetes prediction research.

---

## Dataset Features

### Input Features

| Feature Name | Description |
|-------------|-------------|
| Pregnancies | Number of times the patient has been pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body Mass Index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | Genetic influence on diabetes |
| Age | Age of the patient in years |

---

### Target Variable

| Value | Meaning |
|------|--------|
| Outcome = 1 | Type 2 Diabetes Present |
| Outcome = 0 | No Type 2 Diabetes |

---

## Data Preprocessing
- Replaced medically invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with missing values  
- Handled missing values using median imputation  
- Applied feature scaling using StandardScaler  

---

## Machine Learning Model
- **Algorithm:** Support Vector Machine (SVM)  
- **Kernel Functions:** Linear and Radial Basis Function (RBF)  
- **Hyperparameter Optimization:** GridSearchCV  
- **Train–Test Split:** 80% training and 20% testing with stratification  

---

## Model Evaluation
The model performance is evaluated using the following metrics:
- Accuracy  
- ROC–AUC Score  
- Confusion Matrix  
- Precision, Recall, and F1-score  
- Receiver Operating Characteristic (ROC) Curve  

Typical performance on this dataset:
- Accuracy between 75% and 82%  
- ROC–AUC between 0.78 and 0.85  

---

## Custom Input Prediction
After training, the model supports prediction for new patient data by:
1. Accepting user-provided clinical inputs  
2. Applying the same feature scaling used during training  
3. Predicting the presence or absence of Type 2 Diabetes along with a probability score  

---

## Model Persistence
The following files are saved after training:
- `svm_type2_diabetes_model.joblib` – Trained SVM model  
- `scaler_type2_diabetes.joblib` – Feature scaler  

These files allow reuse of the trained model without retraining.

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

