import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import jwt
from jwt import InvalidSignatureError, ExpiredSignatureError
from cryptography.hazmat.primitives import serialization
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------
# Utility: Load public key & verify credential
# ----------------------------------------------------------
def load_public_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def verify_vc(vc_token, client_id):
    public_key_path = f"keys/{client_id}/public.pem"

    try:
        public_key = load_public_key(public_key_path)

        payload = jwt.decode(
            vc_token,
            public_key,
            algorithms=["RS256"]
        )

        if payload.get("role") != "authorized_fl_client":
            return False, "Invalid role"

        if payload.get("issuer") != "HealthAuthority":
            return False, "Invalid issuer"

        return True, payload

    except ExpiredSignatureError:
        return False, "Credential expired"

    except InvalidSignatureError:
        return False, "Invalid signature"

    except Exception as e:
        return False, str(e)


# ----------------------------------------------------------
# Model
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------
def create_dataset(seed, bias):
    np.random.seed(seed)
    n_samples = 2000
    class_ratio = [bias, 1 - bias]
    y = np.random.choice([0,1], size=n_samples, p=class_ratio)

    X, _ = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
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


# ----------------------------------------------------------
# Local training
# ----------------------------------------------------------
def train_local(model, X_train, y_train, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    return model


# ----------------------------------------------------------
# FedAvg
# ----------------------------------------------------------
def fedavg(global_model, client_models, sizes):
    total = sum(sizes)
    new_state = global_model.state_dict()

    for key in new_state.keys():
        new_state[key] = torch.sum(
            torch.stack([
                client_models[i].state_dict()[key] * (sizes[i] / total)
                for i in range(len(client_models))
            ]),
            dim=0
        )

    global_model.load_state_dict(new_state)
    return global_model


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------
def evaluate(model, X_test, y_test):
    with torch.no_grad():
        out = model(X_test)
        pred = torch.argmax(out, 1)
        return (pred == y_test).float().mean().item() * 100


# ----------------------------------------------------------
# MAIN — Secure Federated Learning
# ----------------------------------------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 3
    ROUNDS = 5
    biases = [0.6, 0.5, 0.4]

    # Load datasets
    datasets = [
        create_dataset(100+i, bias=biases[i])
        for i in range(NUM_CLIENTS)
    ]

    sizes = [len(datasets[i][0]) for i in range(NUM_CLIENTS)]

    # Load credentials
    vc_tokens = []
    for i in range(NUM_CLIENTS):
        with open(f"credentials/client{i+1}_vc.jwt") as f:
            vc_tokens.append(f.read().strip())

    global_model = BiggerNet()

    print("\n=== SECURE FEDERATED LEARNING STARTED ===\n")

    for rnd in range(1, ROUNDS+1):
        print(f"\n--- Round {rnd} ---")
        client_models = []

        for i in range(NUM_CLIENTS):
            print(f"\n[Client {i+1}] Verifying credential...")

            ok, info = verify_vc(vc_tokens[i], f"client{i+1}")

            if not ok:
                print("  Verification failed:", info)
                print(" Skipping this client.")
                continue

            print(" ✔ Credential valid!")

            # Local training
            model = BiggerNet()
            model.load_state_dict(global_model.state_dict())

            X_train, y_train, X_test, y_test = datasets[i]
            trained = train_local(model, X_train, y_train)
            acc = evaluate(trained, X_test, y_test)

            print(f" Local Accuracy: {acc:.2f}%")

            client_models.append(trained)

        # Aggregate only verified clients
        global_model = fedavg(global_model, client_models, sizes)

        # Evaluate global model
        print("\n Global Accuracies:")
        for i in range(NUM_CLIENTS):
            _, _, X_test, y_test = datasets[i]
            acc = evaluate(global_model, X_test, y_test)
            print(f"  Client {i+1}: {acc:.2f}%")

    print("\n=== SECURE FL COMPLETED ===")
