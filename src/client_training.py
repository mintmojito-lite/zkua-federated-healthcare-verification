import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# 1. Generate synthetic healthcare-like dataset
def create_dataset():
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )


# 2. Simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.layers(x)


# 3. Train the model locally (simulating a single hospital)
def local_train(model, X_train, y_train, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model


# 4. Evaluate accuracy
def evaluate(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item() * 100
        print(f"\nLocal Model Accuracy: {accuracy:.2f}%")
        return accuracy


# --- main pipeline ---
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_dataset()
    model = SimpleNet()
    model = local_train(model, X_train, y_train, epochs=5)
    evaluate(model, X_test, y_test)
