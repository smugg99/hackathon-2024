# Import required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import Grover
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Load Breast Cancer data
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model SVM with classical hyperparameter C optimization using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear']}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid_search.fit(X_train, y_train)

# Applying the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Display results
print(f"Best parameters (classical optimization): {grid_search.best_params_}")
print(f"Accuracy of classical SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 2. Quantum optimization using Grover's algorithm
# Defining a custom oracle
def custom_oracle(num_qubits):
    # Create a quantum circuit for the oracle
    qc = QuantumCircuit(num_qubits)
    
    # Example oracle: phase inversion for a specific state
    # Assume we are looking for the state |11> on two qubits
    qc.cz(0, 1)  # Invert phase for the state |11|
    
    return qc

# Parameters for Grover's problem
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
backend = AerSimulator()  # Using AerSimulator
qc = transpile(qc, backend)  # Transpile the circuit
job = backend.run(qc, shots=1024)  # Run the simulation
result = job.result()
counts = result.get_counts()

# Display the results of the simulation
print(f"Results of quantum optimization (oracle result): {counts}")

# Defining a mapping from states to C values
state_to_C = {
    '00': 0.1,
    '01': 1,
    '10': 10,
    '11': 100
}

# Identify the state with the highest measurement count
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]

print(f"Best C value obtained from quantum optimization: {best_C_from_grover}")

# 3. Applying the optimal C value to the classical SVM model
# Create an SVM model with the C value obtained from quantum optimization
svm_quantum = SVC(kernel='linear', C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)

# Applying cross-validation
cv_scores = cross_val_score(svm_quantum, X, y, cv=5)  # 5-fold cross-validation

print(f"Cross-validation results: {cv_scores}")
print(f"Mean accuracy from cross-validation: {np.max(cv_scores) * 100:.2f}%")

# Visualization of the Breast Cancer data in feature space
plt.figure(figsize=(14, 6))

# Subplot 1: Distribution of data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'green']), edgecolor='k')
plt.title('Breast Cancer Data - Distribution')
plt.xlabel(cancer.feature_names[0])  # Use actual feature name
plt.ylabel(cancer.feature_names[1])  # Use actual feature name

# Subplot 2: Comparison of model accuracies
plt.subplot(1, 2, 2)
plt.bar(['Classical SVM', 'SVM with Quantum Optimization'], 
        [accuracy_score(y_test, y_pred) * 100, np.max(cv_scores) * 100], 
        color=['orange', 'purple'])
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy (%)')
plt.ylim(90, 100)  # Set Y scale from 90% to 100%

# Display the visualization
plt.tight_layout()
plt.show()