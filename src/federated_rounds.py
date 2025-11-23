import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# -----------------------------
#  Model
# -----------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------
# Non-IID Dataset Generator
# -----------------------------
def create_non_iid_dataset(seed, bias=0.2):
    np.random.seed(seed)

    # Control class imbalance for non-IID effects
    n_samples = 2000
    class_ratio = [bias, 1 - bias]   # bias toward class 0 or class 1

    y = np.random.choice([0, 1], size=n_samples, p=class_ratio)
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


# -----------------------------
# Local training
# -----------------------------
def train_local(model, X_train, y_train, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    return model


# -----------------------------
# Weighted FedAvg Aggregation
# -----------------------------
def fedavg(global_model, client_models, client_sizes):
    new_state = global_model.state_dict()

    total_samples = sum(client_sizes)

    for key in new_state.keys():
        new_state[key] = torch.sum(
            torch.stack([
                client_models[i].state_dict()[key] * (client_sizes[i] / total_samples)
                for i in range(len(client_models))
            ]),
            dim=0,
        )

    global_model.load_state_dict(new_state)
    return global_model


# -----------------------------
# Accuracy Evaluation
# -----------------------------
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        out = model(X_test)
        pred = torch.argmax(out, 1)
        acc = (pred == y_test).float().mean().item() * 100
    return acc


# -----------------------------
# MAIN: Multi-Round FL
# -----------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 3
    ROUNDS = 5

    # Each client gets different bias → non-IID
    biases = [0.7, 0.5, 0.3]  # Very imbalanced to simulate heterogeneity
    datasets = [
        create_non_iid_dataset(seed=100 + i, bias=biases[i])
        for i in range(NUM_CLIENTS)
    ]

    client_sizes = [len(datasets[i][0]) for i in range(NUM_CLIENTS)]
    global_model = SimpleNet()

    print("\n=== MULTI-ROUND FEDERATED LEARNING STARTED ===\n")

    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- ROUND {rnd} ---")
        client_models = []

        # Local training at each hospital
        for i in range(NUM_CLIENTS):
            model = SimpleNet()
            model.load_state_dict(global_model.state_dict())  # start from global weights

            X_train, y_train, X_test, y_test = datasets[i]
            trained = train_local(model, X_train, y_train)
            acc = evaluate(trained, X_test, y_test)
            print(f" Client {i+1} Local Acc: {acc:.2f}%")

            client_models.append(trained)

        # Aggregation
        global_model = fedavg(global_model, client_models, client_sizes)

        # Evaluate global model after aggregation
        print("\n Global Model Accuracies:")
        for i in range(NUM_CLIENTS):
            _, _, X_test, y_test = datasets[i]
            acc = evaluate(global_model, X_test, y_test)
            print(f"  Global on Client {i+1}: {acc:.2f}%")

    print("\n=== FEDERATED LEARNING COMPLETED ===\n")
