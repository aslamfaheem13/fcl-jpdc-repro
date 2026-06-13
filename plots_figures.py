import matplotlib.pyplot as plt

# =========================
# FIGURE 1: Accuracy vs Forgetting
# =========================
methods = ["FULL_FEDAVG", "SHARED_ADAPTER", "TASK_ADAPTER", "TRUE_FEDAVG"]

accuracy = [0.1049, 0.1026, 0.1043, 0.3048]
forgetting = [0.0, 0.0021, 0.0, 0.2842]

plt.figure(figsize=(6,5))

for i, m in enumerate(methods):
    plt.scatter(forgetting[i], accuracy[i])
    plt.text(forgetting[i]+0.005, accuracy[i], m)

plt.xlabel("Forgetting")
plt.ylabel("Final Accuracy")
plt.title("Accuracy vs Forgetting Trade-off (CIFAR-100)")
plt.grid()

plt.savefig("figure1_accuracy_vs_forgetting.png", dpi=300)
plt.close()


# =========================
# FIGURE 2: Replay size
# =========================
rpc = [5, 10, 20, 40]
acc = [0.4450, 0.4693, 0.4868, 0.5024]

plt.figure(figsize=(6,5))
plt.plot(rpc, acc, marker='o')

plt.xlabel("Replay per class")
plt.ylabel("Final Accuracy")
plt.title("Effect of Replay Memory Size")

plt.grid()
plt.savefig("figure2_replay.png", dpi=300)
plt.close()


# =========================
# FIGURE 3: Heterogeneity
# =========================
alpha = [0.03, 0.1, 0.3]
acc = [0.3363, 0.5024, 0.6947]

plt.figure(figsize=(6,5))
plt.plot(alpha, acc, marker='o')

plt.xlabel("Dirichlet Alpha")
plt.ylabel("Final Accuracy")
plt.title("Impact of Client Heterogeneity")

plt.grid()
plt.savefig("figure3_heterogeneity.png", dpi=300)
plt.close()

print("All figures saved successfully!")