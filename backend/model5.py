import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Define the SVM model
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


# Hyperparameters
input_dim = 2  # Example input dimension
learning_rate = 0.01
num_epochs = 100

# Create the model, loss function, and optimizer
model = SVM(input_dim).to(device)
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Dummy data (replace with your dataset)
X_train = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]).to(device)
y_train = torch.tensor([1, 1, -1, -1], dtype=torch.float32).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs.view(-1), y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing (replace with your test data)
model.eval()
X_test = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).to(device)
outputs = model(X_test)
predictions = torch.sign(outputs).view(-1)
print(f"Predictions: {predictions}")
