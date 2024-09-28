# Importujemy wymagane biblioteki
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import qiskit
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit_algorithms import Grover
import numpy as np

# 1. Klasyczny model SVM z klasyczną optymalizacją

# Załaduj dane Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model SVM z klasyczną optymalizacją hiperparametru C za pomocą GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear']}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid_search.fit(X_train, y_train)

# Zastosowanie najlepszego modelu
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Wyświetlenie wyników
print(f"Najlepsze parametry (klasyczna optymalizacja): {grid_search.best_params_}")
print(f"Dokładność klasycznego SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 2. Optymalizacja kwantowa za pomocą algorytmu Grovera

# Definiujemy niestandardowe orakulum
def custom_oracle(num_qubits):
    # Tworzymy obwód kwantowy dla orakulum
    qc = QuantumCircuit(num_qubits)
    
    # Przykładowe orakulum: odwrócenie fazy dla konkretnego stanu
    # Tu załóżmy, że szukamy stanu |11> na dwóch kubitach
    qc.cz(0, 1)  # Odwróć fazę dla stanu |11>
    
    return qc

# Parametry do problemu Grovera
num_qubits = 2
oracle_circuit = custom_oracle(num_qubits)

# Dodajemy obwód do algorytmu Grovera
grover = Grover(oracle_circuit)

# Tworzymy obwód Grovera
qc = QuantumCircuit(num_qubits)
qc.h([0, 1])  # Inicjalizacja kubitów w stanie superpozycji
qc.compose(oracle_circuit, inplace=True)  # Dodajemy orakulum
qc.h([0, 1])  # Zastosowanie bramek Hadamarda ponownie

# Uruchomienie symulacji
simulator = AerSimulator()
sim_result = simulator.run(qc).result()

# Wyświetlenie wyników symulacji
print(f"Wyniki optymalizacji kwantowej (wynik orakulum): {sim_result}")

# Przykład: na podstawie wyników Grovera załóżmy, że optymalna wartość C to 10
best_C_from_grover = 10

# 3. Zastosowanie optymalnej wartości C do klasycznego modelu SVM

# Tworzymy model SVM z wartością C uzyskaną z optymalizacji kwantowej
svm_quantum = SVC(kernel='linear', C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)

# Dokonaj predykcji i oceń dokładność modelu
y_pred_quantum = svm_quantum.predict(X_test)
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)

# Wyświetlenie wyników
print(f"Dokładność SVM z optymalizacją kwantową: {accuracy_quantum * 100:.2f}%")