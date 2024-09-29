import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

# Load and preprocess data
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
ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(df[ohe_categorical_columns])
encoded_df = pd.DataFrame(
    encoded_data, columns=ohe.get_feature_names_out(ohe_categorical_columns)
)
df = pd.concat([df, encoded_df], axis=1).drop(ohe_categorical_columns, axis=1)

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

print(df.info())

X = df[[col for col in df.columns if col != "chronic_conditions"]].values
Y = df["chronic_conditions"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Quantum optimization section
def custom_oracle(num_qubits):
    qc = QuantumCircuit(num_qubits)
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

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors and move to device
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)


# SVM model definition
class SVM(nn.Module):
    def __init__(self, input_size, C):
        super(SVM, self).__init__()
        self.C = C
        self.w = nn.Parameter(torch.zeros(input_size).to(device))  # Weight vector
        self.b = nn.Parameter(torch.zeros(1).to(device))  # Bias term

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def hinge_loss(self, outputs, labels):
        # Hinge loss calculation
        return self.C * torch.mean(torch.clamp(1 - labels * outputs, min=0))


# Instantiate the SVM model with C from Grover's optimization
model = SVM(input_size=X.shape[1], C=best_C_from_grover).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = model.hinge_loss(outputs.view(-1), y_train_tensor.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_label = (y_pred.view(-1) > 0).float()  # Convert to binary predictions

accuracy = accuracy_score(y_test, y_pred_label.cpu().numpy())
print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")
