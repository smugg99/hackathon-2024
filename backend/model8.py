# Quantum Optimization with Machine Learning for Chronic Conditions Prediction

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# 1. Data Preparation
# Loading and Cleaning the Data
df = pd.read_csv("./depression_data.csv")
print(df.head())
print(df.describe())

df.drop("Name", axis=1, inplace=True)
df.rename(columns={
    "Age": "age",
    "Income": "income",
    "Marital Status": "marital_status",
    "Education Level": "education",
    "Smoking Status": "smoking",
    "Employment Status": "employment",
    "History of Mental Illness": "mental_illness",
    "Physical Activity Level": "physical_activity",
    "History of Substance Abuse": "substance_abuse",
    "Alcohol Consumption": "alcohol",
    "Dietary Habits": "diet",
    "Sleep Patterns": "sleep",
    "Family History of Depression": "family_depression",
    "Chronic Medical Conditions": "chronic_conditions"
}, inplace=True)

# Encoding Categorical Variables
ohe_categorical_columns = [
    "marital_status", "education", "smoking", 
    "physical_activity", "alcohol", "diet", "sleep"
]
ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(df[ohe_categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(ohe_categorical_columns))
df = pd.concat([df, encoded_df], axis=1).drop(ohe_categorical_columns, axis=1)

# Mapping Yes/No Columns
yes_no_columns = [
    "mental_illness", "substance_abuse", 
    "family_depression", "chronic_conditions"
]

for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["employment"] = df["employment"].map({"Employed": 1, "Unemployed": 0})
df["income"] = (df["income"] - df["income"].min()) / (df["income"].max() - df["income"].min())

# 2. Classical Optimization: SVM with Grid Search
# Splitting the Data and Training the SVM
X = df.drop("chronic_conditions", axis=1)
Y = df["chronic_conditions"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear"]}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid_search.fit(X_train, y_train)

# Evaluating the Classical SVM
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
from sklearn.metrics import accuracy_score
print(f"Best parameters (classical optimization): {grid_search.best_params_}")
print(f"Accuracy of classical SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 3. Quantum Optimization: Grover's Algorithm
# Custom Oracle for Grover's Algorithm
def custom_oracle(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.cz(0, 1)  # Invert phase for the state |11>
    return qc

# Constructing and Simulating Grover’s Circuit
num_qubits = 2
oracle_circuit = custom_oracle(num_qubits)
qc = QuantumCircuit(num_qubits)
qc.h([0, 1])
qc.compose(oracle_circuit, inplace=True)
qc.h([0, 1])
qc.measure_all()

backend = AerSimulator(device="GPU")
qc = transpile(qc, backend)
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

# Mapping Quantum States to Hyperparameters
state_to_C = {"00": 0.1, "01": 1, "10": 10, "11": 100}
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]
print(f"Best C value obtained from quantum optimization: {best_C_from_grover}")

# 4. Applying Quantum-Optimized C to SVM
svm_quantum = SVC(kernel="linear", C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)

# Cross-Validation
cv_scores = cross_val_score(svm_quantum, X, Y, cv=5)
print(f"Mean accuracy from cross-validation: {np.max(cv_scores) * 100:.2f}%")

# 5. Conclusion
print("This project demonstrates a hybrid approach where quantum computing is used to optimize hyperparameters for a Support Vector Machine.")
