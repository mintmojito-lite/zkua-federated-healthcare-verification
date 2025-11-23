import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# ============================================================
# 1) BIGGER MODEL FOR STRONGER BASELINE
# ============================================================
class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# 2) NON-IID DATASET (REDUCED IMBALANCE)
# ============================================================
def create_non_iid_dataset(seed, bias):
    np.random.seed(seed)

    n_samples = 2500
    class_ratio = [bias, 1 - bias]

    y = np.random.choice([0, 1], size=n_samples, p=class_ratio)

    X, _ = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=0,
        n_classes=2,
        random_state=seed
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
        torch.tensor(y_test, dtype=torch.long)
    )


# ============================================================
# 3) LOCAL TRAINING FOR EACH CLIENT
# ============================================================
def train_local(model, X_train, y_train, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    return model


# ============================================================
# 4) WEIGHTED FEDAVG (REAL FEDERATED AVERAGING)
# ============================================================
def fedavg(global_model, client_models, client_sizes):
    total = sum(client_sizes)
    new_state = global_model.state_dict()

    for key in new_state.keys():
        new_state[key] = torch.sum(
            torch.stack([
                client_models[i].state_dict()[key] * (client_sizes[i] / total)
                for i in range(len(client_models))
            ]),
            dim=0
        )

    global_model.load_state_dict(new_state)
    return global_model


# ============================================================
# 5) ACCURACY EVALUATION
# ============================================================
def evaluate(model, X_test, y_test):
    with torch.no_grad():
        out = model(X_test)
        pred = torch.argmax(out, 1)
        acc = (pred == y_test).float().mean().item() * 100
    return acc


# ============================================================
# 6) MULTI-ROUND FEDERATED TRAINING
# ============================================================
if __name__ == "__main__":

    NUM_CLIENTS = 3
    ROUNDS = 15   # more rounds for better convergence
    LOCAL_EPOCHS = 5

    biases = [0.60, 0.50, 0.40]   # reduced imbalance

    # Load non-IID datasets
    datasets = [
        create_non_iid_dataset(seed=100 + i, bias=biases[i])
        for i in range(NUM_CLIENTS)
    ]

    # Client dataset sizes
    sizes = [len(datasets[i][0]) for i in range(NUM_CLIENTS)]

    global_model = BiggerNet()

    print("\n=== ADVANCED MULTI-ROUND FEDERATED LEARNING ===\n")

    # For graphing (next step)
    global_acc_history = []
    client_acc_history = [[] for _ in range(NUM_CLIENTS)]

    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- ROUND {rnd} ---")
        client_models = []

        # Local Training
        for i in range(NUM_CLIENTS):
            model = BiggerNet()
            model.load_state_dict(global_model.state_dict())

            X_train, y_train, X_test, y_test = datasets[i]

            trained = train_local(model, X_train, y_train, epochs=LOCAL_EPOCHS)
            acc = evaluate(trained, X_test, y_test)

            print(f" Client {i+1} Local Acc: {acc:.2f}%")

            client_acc_history[i].append(acc)
            client_models.append(trained)

        # FedAvg Aggregation
        global_model = fedavg(global_model, client_models, sizes)

        # Global Accuracy (per client)
        print("\n Global Model Accuracies:")
        round_global_acc = 0

        for i in range(NUM_CLIENTS):
            _, _, X_test, y_test = datasets[i]
            acc = evaluate(global_model, X_test, y_test)
            print(f"  Global on Client {i+1}: {acc:.2f}%")

            round_global_acc += acc

        avg_global_acc = round_global_acc / NUM_CLIENTS
        global_acc_history.append(avg_global_acc)

    print("\n=== TRAINING COMPLETED ===\n")

    # Save accuracy logs for plotting (next step)
    np.save("global_acc.npy", np.array(global_acc_history))
    np.save("client_acc.npy", np.array(client_acc_history))

    print("Accuracy logs saved for plotting.")
