import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Simple Neural Network
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
# 2. Create DISTINCT datasets for each hospital
# -----------------------------
def create_client_dataset(seed):
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=12,
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
# 3. Local training on each hospital
# -----------------------------
def train_local(model, X_train, y_train, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    return model


# -----------------------------
# 4. FedAvg aggregation
# -----------------------------
def fedavg(global_model, client_models):
    global_dict = global_model.state_dict()

    # Average parameters across clients
    for key in global_dict.keys():
        global_dict[key] = torch.mean(
            torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))]),
            dim=0,
        )

    global_model.load_state_dict(global_dict)
    return global_model


# -----------------------------
# 5. Evaluate model
# -----------------------------
def evaluate(model, X_test, y_test, name="Model"):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        pred = torch.argmax(output, 1)
        acc = (pred == y_test).float().mean().item() * 100
    print(f"{name} Accuracy: {acc:.2f}%")
    return acc


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 3
    client_datasets = []
    client_models = []

    # Step A: generate distinct datasets
    for i in range(NUM_CLIENTS):
        client_datasets.append(create_client_dataset(seed=100 + i))

    # Step B: initialize global model
    global_model = SimpleNet()

    # Step C: each client trains locally
    for i in range(NUM_CLIENTS):
        print(f"\n--- Training Client {i+1} ---")
        model = SimpleNet()
        X_train, y_train, X_test, y_test = client_datasets[i]
        trained = train_local(model, X_train, y_train)
        evaluate(trained, X_test, y_test, name=f"Client {i+1} Local")
        client_models.append(trained)

    # Step D: server aggregates using FedAvg
    print("\n=== Performing FedAvg Aggregation ===")
    global_model = fedavg(global_model, client_models)

    # Step E: evaluate global model on EACH client's test set
    print("\n=== Evaluating Global Model on Client Test Sets ===")
    for i in range(NUM_CLIENTS):
        _, _, X_test, y_test = client_datasets[i]
        evaluate(global_model, X_test, y_test, name=f"Global on Client {i+1}")
