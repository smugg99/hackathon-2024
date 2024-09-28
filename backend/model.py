# Importujemy wymagane biblioteki
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import Grover
import numpy as np
from sklearn.model_selection import cross_val_score

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

# Tworzymy obwód Grovera
qc = QuantumCircuit(num_qubits)
qc.h([0, 1])  # Inicjalizacja kubitów w stanie superpozycji
qc.compose(oracle_circuit, inplace=True)  # Dodajemy orakulum
qc.h([0, 1])  # Zastosowanie bramek Hadamarda ponownie

# Dodajemy pomiary do obwodu
qc.measure_all()  # Dodajemy pomiar do wszystkich kubitów

# Uruchomienie symulacji z użyciem AerSimulator
backend = AerSimulator()  # Używamy AerSimulator
qc = transpile(qc, backend)  # Transpilacja obwodu
job = backend.run(qc, shots=1024)  # Uruchomienie symulacji
result = job.result()
counts = result.get_counts()

# Wyświetlenie wyników symulacji
print(f"Wyniki optymalizacji kwantowej (wynik orakulum): {counts}")

# Zdefiniowanie mapowania stanów do wartości C
state_to_C = {
    '00': 0.1,
    '01': 1,
    '10': 10,
    '11': 100
}

# Zidentyfikowanie stanu o najwyższej liczbie pomiarów
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]

print(f"Najlepsza wartość C uzyskana z optymalizacji kwantowej: {best_C_from_grover}")

# 3. Zastosowanie optymalnej wartości C do klasycznego modelu SVM
# Tworzymy model SVM z wartością C uzyskaną z optymalizacji kwantowej
svm_quantum = SVC(kernel='linear', C=best_C_from_grover)
svm_quantum.fit(X_train, y_train)

# Zastosowanie walidacji krzyżowej
cv_scores = cross_val_score(svm_quantum, X, y, cv=5)  # 5-fold cross-validation

# Dokonaj predykcji i oceń dokładność modelu
y_pred_quantum = svm_quantum.predict(X_test)
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)

print(f"Wyniki walidacji krzyżowej: {cv_scores}")
print(f"Średnia dokładność z walidacji krzyżowej: {np.mean(cv_scores) * 100:.2f}%")