import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit_algorithms import Grover
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib.colors import ListedColormap

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
df = pd.read_csv("./depression_data.csv")
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

# One-hot encoding
ohe_categorical_columns = [
    "marital_status",
    "education",
    "smoking",
    "physical_activity",
    "alcohol",
    "diet",
    "sleep",
]
print(df.head())

ohe = LabelEncoder()
for column in ohe_categorical_columns:
    print(ohe.fit_transform(df[column]))
    df[column] = ohe.fit_transform(df[column])

# Mapping yes/no columns
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

# Standardizing numerical columns using torch
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("chronic_conditions", axis=1))

X = torch.FloatTensor(X_scaled).to(device)  # Transfer to CUDA if available
Y = (
    torch.FloatTensor(df["chronic_conditions"].values).view(-1, 1).to(device)
)  # Transfer to CUDA if available

# Split the data into training and testing using random_split
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_data, test_data = random_split(TensorDataset(X, Y), [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define SVM-like model using PyTorch
class SVMTorch(nn.Module):
    def __init__(self, input_size, C=1.0):
        super(SVMTorch, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.C = C

    def forward(self, x):
        return self.fc(x)

    def svm_loss(self, outputs, labels):
        hinge_loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        reg_loss = 0.5 * torch.norm(self.fc.weight) ** 2
        return self.C * hinge_loss + reg_loss


# Train SVM model
def train_svm(model, optimizer, criterion, train_loader, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = model.svm_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}"
            )


# Quantum optimization section using Grover's algorithm
def custom_oracle(num_qubits):
    # Create a quantum circuit for the oracle
    qc = QuantumCircuit(num_qubits)
    # Example oracle: phase inversion for a specific state
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
backend = AerSimulator()  # Using AerSimulator
qc = transpile(qc, backend)  # Transpile the circuit
job = backend.run(qc, shots=1024)  # Run the simulation
result = job.result()
counts = result.get_counts()

state_to_C = {"00": 0.1, "01": 1, "10": 10, "11": 100}
optimal_state = max(counts, key=counts.get)
best_C_from_grover = state_to_C[optimal_state]

print(f"Best C value obtained from quantum optimization: {best_C_from_grover}")

# First run SVM model with default C=1.0
model_classical = SVMTorch(input_size=X.shape[1], C=1.0).to(
    device
)  # Transfer model to CUDA if available
optimizer_classical = optim.SGD(model_classical.parameters(), lr=0.001)
train_svm(model_classical, optimizer_classical, model_classical.svm_loss, train_loader)

# Second run SVM model with quantum-optimized C
model_quantum = SVMTorch(input_size=X.shape[1], C=best_C_from_grover).to(
    device
)  # Transfer model to CUDA if available
optimizer_quantum = optim.SGD(model_quantum.parameters(), lr=0.001)
train_svm(model_quantum, optimizer_quantum, model_quantum.svm_loss, train_loader)


# Evaluation function for the models
def evaluate_svm(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = torch.sign(outputs)  # SVM outputs hinge scores
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


# Evaluate the classical and quantum-optimized models
accuracy_classical = evaluate_svm(model_classical, test_loader)
accuracy_quantum = evaluate_svm(model_quantum, test_loader)

print(f"Classical SVM Accuracy: {accuracy_classical * 100:.2f}%")
print(f"Quantum-Optimized SVM Accuracy: {accuracy_quantum * 100:.2f}%")

# Visualization of model accuracies
plt.figure(figsize=(8, 6))
plt.bar(
    ["Classical SVM", "Quantum-Optimized SVM"],
    [accuracy_classical * 100, accuracy_quantum * 100],
    color=["orange", "purple"],
)
plt.title("Comparison of Classical vs Quantum-Optimized SVM")
plt.ylabel("Accuracy (%)")
plt.show()
