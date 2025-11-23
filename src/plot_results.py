import numpy as np
import matplotlib.pyplot as plt
import os

# Load saved accuracy logs
global_acc = np.load("global_acc.npy")
client_acc = np.load("client_acc.npy", allow_pickle=True)

# Create folder for graphs
os.makedirs("graphs", exist_ok=True)

# -----------------------------
# 1. GLOBAL ACCURACY VS ROUNDS
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(global_acc, marker='o', linewidth=2)
plt.title("Global Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Global Accuracy (%)")
plt.grid(True)
plt.savefig("graphs/global_accuracy.png")
plt.show()


# -----------------------------
# 2. CLIENT ACCURACIES VS ROUNDS
# -----------------------------
plt.figure(figsize=(8, 5))
for i in range(client_acc.shape[0]):
    plt.plot(client_acc[i], marker='o', linewidth=2, label=f"Client {i+1}")

plt.title("Client Local Accuracies vs Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.savefig("graphs/client_accuracy.png")
plt.show()


# -----------------------------
# 3. NON-IID DRIFT VISUALIZATION
# -----------------------------
plt.figure(figsize=(8, 5))
end_acc = [client_acc[i][-1] for i in range(client_acc.shape[0])]
plt.bar([f"Client {i+1}" for i in range(len(end_acc))], end_acc)
plt.title("Final Local Accuracies (Non-IID Divergence)")
plt.ylabel("Accuracy (%)")
plt.savefig("graphs/non_iid_divergence.png")
plt.show()

print("\nGraphs saved in /graphs folder!")
