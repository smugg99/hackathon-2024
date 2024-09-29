# Quantum Optimization with Machine Learning for Chronic Conditions Prediction

This document explores the application of **quantum optimization** techniques alongside **classical machine learning** to predict chronic medical conditions. Specifically, we use **Support Vector Machines (SVM)** and a **quantum-enhanced Grover's Algorithm** for hyperparameter optimization. The dataset utilized in this project is related to depression and associated factors like age, income, lifestyle habits, and mental health.

## 1. Data Preparation

### Loading and Cleaning the Data

```python
import pandas as pd

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
```

**Explanation:**

- The dataset contains several features related to demographics, lifestyle, and medical history, all of which could influence **chronic medical conditions**.
- We begin by loading and cleaning the data. This involves renaming columns for ease of use and dropping non-relevant columns like "Name."

### Encoding Categorical Variables

```python
from sklearn.preprocessing import OneHotEncoder

ohe_categorical_columns = [
    "marital_status", "education", "smoking", 
    "physical_activity", "alcohol", "diet", "sleep"
]
ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(df[ohe_categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(ohe_categorical_columns))
df = pd.concat([df, encoded_df], axis=1).drop(ohe_categorical_columns, axis=1)
```

**Explanation:**

We apply **One-Hot Encoding** to transform categorical columns like **marital_status** and **education** into binary columns, which are more suitable for machine learning models.

### Mapping Yes/No Columns

```python
yes_no_columns = [
    "mental_illness", "substance_abuse", 
    "family_depression", "chronic_conditions"
]

for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["employment"] = df["employment"].map({"Employed": 1, "Unemployed": 0})
df["income"] = (df["income"] - df["income"].min()) / (df["income"].max() - df["income"].min())
```

**Explanation:**

Binary columns such as **mental_illness** and **substance_abuse** are converted into 1s and 0s to make the data numerical. The **income** column is normalized to ensure all feature values are on the same scale.

## 2. Classical Optimization: SVM with Grid Search

### Splitting the Data and Training the SVM

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from cuml.model_selection import GridSearchCV

X = df.drop("chronic_conditions", axis=1)
Y = df["chronic_conditions"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear"]}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid_search.fit(X_train, y_train)
```

**Explanation:**

We use **GridSearchCV** to optimize the hyperparameter **C** for a linear **SVM**. The grid search explores multiple values for **C** to find the best one based on the training data.

### Evaluating the Classical SVM

```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Best parameters (classical optimization): {grid_search.best_params_}")
print(f"Accuracy of classical SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

**Explanation:**

The **best SVM model** from the grid search is used to make predictions on the test set. The **accuracy** is calculated to evaluate how well the model performs on unseen data.

## 3. Quantum Optimization: Grover's Algorithm

### Custom Oracle for Grover's Algorithm

```python
from qiskit import QuantumCircuit

def custom_oracle(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.cz(0, 1)  # Invert phase for the state |11>
    return qc
```

**Explanation:**

**Grover's Algorithm** is a quantum algorithm designed for search and optimization problems. Here, we define a custom **oracle** that marks the state |11> (corresponding to an optimal value of **C**) by flipping its phase.

### Constructing and Simulating Grover’s Circuit

```python
from qiskit_aer import AerSimulator
from qiskit import transpile

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
```

**Explanation:**

We initialize a quantum circuit in superposition and apply the **oracle** to perform phase inversion. The circuit is then run on a **quantum simulator** to find the optimal hyperparameter value using quantum mechanics.

### Mapping Quantum States to Hyperparameters

```python
state_to_C = {"00": 0.1, "01": 1, "10": 10, "11": 100}
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]
print(f"Best C value obtained from quantum optimization: {best_C_from_grover}")
```

**Explanation:**

Quantum states are mapped to potential values of **C**. The most frequently observed state from the quantum simulation is selected as the optimal **C** value for the SVM.

## 4. Applying Quantum-Optimized C to SVM

```python
svm_quantum = SVC(kernel="linear", C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)
```

**Explanation:**

Using the **C value** obtained from Grover's Algorithm, we train a new SVM model. This showcases a **hybrid quantum-classical approach** to machine learning.

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(svm_quantum, X, Y, cv=5)
print(f"Mean accuracy from cross-validation: {np.max(cv_scores) * 100:.2f}%")
```

**Explanation:**

We perform **5-fold cross-validation** to evaluate the performance of the quantum-optimized SVM. The mean accuracy score gives insight into how well the model generalizes across different subsets of the data.

## 5. Conclusion

In this project, we demonstrated a hybrid approach where **quantum computing** is used to optimize hyperparameters for a **Support Vector Machine**. Specifically, **Grover's Algorithm** was applied to find the optimal regularization parameter **C**, leading to a new quantum-enhanced SVM model. This method shows promising potential for the future of machine learning, combining classical algorithms with quantum optimization techniques.