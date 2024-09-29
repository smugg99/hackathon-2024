from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit, transpile
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from cuml.model_selection import GridSearchCV
import pandas as pd


df = pd.read_csv("./depression_data.csv")
print(df.head())
print(df.describe())

df.drop("Name", axis=1, inplace=True)

df.rename(
    columns={
        "Name": "name",
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
        "Chronic Medical Conditions": "chronic_conditions",
    },
    inplace=True,
)

ohe_categorical_columns = [
    "marital_status",
    "education",
    "smoking",
    "physical_activity",
    "alcohol",
    "diet",
    "sleep",
]

ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(df[ohe_categorical_columns])
encoded_df = pd.DataFrame(
    encoded_data, columns=ohe.get_feature_names_out(ohe_categorical_columns)
)
df = pd.concat([df, encoded_df], axis=1).drop(ohe_categorical_columns, axis=1)

yes_no_columns = [
    "mental_illness",
    "substance_abuse",
    "family_depression",
    "chronic_conditions",
]

for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["employment"] = df["employment"].map({"Employed": 1, "Unemployed": 0})

df["income"] = (df["income"] - df["income"].min()) / (
    df["income"].max() - df["income"].min()
)

print(df.info())

X = df[[col for col in df.columns if col != "chronic_conditions"]]
Y = df["chronic_conditions"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear"]}
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    refit=True,
    verbose=1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Display results for classical SVM
print(f"Best parameters (classical optimization): {grid_search.best_params_}")
print(f"Accuracy of classical SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")


def custom_oracle(num_qubits):
    # Create a quantum circuit for the oracle
    qc = QuantumCircuit(num_qubits)

    # Example oracle: phase inversion for a specific state
    # Assume we are looking for the state |11> on two qubits
    qc.cz(0, 1)  # Invert phase for the state |11|

    return qc


num_qubits = 2
oracle_circuit = custom_oracle(num_qubits)

# Creating Grover's circuit
qc = QuantumCircuit(num_qubits)
qc.h([0, 1])  # Initialize qubits in superposition state
qc.compose(oracle_circuit, inplace=True)  # Add the oracle
qc.h([0, 1])  # Apply Hadamard gates again

# Adding measurements to the circuit
qc.measure_all()  # Add measurement to all qubits

# Running the simulation using AerSimulator
backend = AerSimulator(device="GPU")  # Using AerSimulator
qc = transpile(qc, backend)  # Transpile the circuit
job = backend.run(qc, shots=1024)  # Run the simulation
result = job.result()
counts = result.get_counts()

state_to_C = {"00": 0.1, "01": 1, "10": 10, "11": 100}

# Identify the state with the highest measurement count
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]

print(f"Best C value obtained from quantum optimization: {best_C_from_grover}")

# 3. Applying the optimal C value to the classical SVM model
svm_quantum = SVC(kernel="linear", C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)

# Applying cross-validation
cv_scores = cross_val_score(svm_quantum, X, Y, cv=5)  # 5-fold cross-validation

print(f"Cross-validation results: {cv_scores}")
print(f"Mean accuracy from cross-validation: {np.max(cv_scores) * 100:.2f}%")
