import pandas as pd
import matplotlib.pyplot as plt

# ====== STEP 1: Use your exact CSV file names ======

robust_file = "myresults/AIDS_GIN_subgraph_ratio0.05_triggerratio0.2_robustlr0.001_1.csv"
sanitize_file = "myresults/AIDS_GIN_subgraph_ratio0.05_triggerratio0.2_sanitize0.2_1.csv"
reg_file = "myresults/AIDS_GIN_subgraph_ratio0.05_triggerratio0.2_reglr0.001_1.csv"

# ====== STEP 2: Read CSV Files ======

robust = pd.read_csv(robust_file)
sanitize = pd.read_csv(sanitize_file)
reg = pd.read_csv(reg_file)

# ====== STEP 3: Plot Clean Accuracy ======

plt.figure()
plt.plot(robust["epoch"], robust["clean_test_acc"], label="Robust Training")
plt.plot(sanitize["epoch"], sanitize["clean_test_acc"], label="Data Sanitization")
plt.plot(reg["epoch"], reg["clean_test_acc"], label="Graph Regularization")
plt.xlabel("Epoch")
plt.ylabel("Clean Test Accuracy")
plt.title("Clean Accuracy Comparison Across Defenses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("clean_accuracy_comparison.png")

# ====== STEP 4: Plot Backdoor Accuracy (ASR) ======

plt.figure()
plt.plot(robust["epoch"], robust["backdoor_test_acc"], label="Robust Training")
plt.plot(sanitize["epoch"], sanitize["backdoor_test_acc"], label="Data Sanitization")
plt.plot(reg["epoch"], reg["backdoor_test_acc"], label="Graph Regularization")
plt.xlabel("Epoch")
plt.ylabel("Backdoor Accuracy (ASR)")
plt.title("ASR Comparison Across Defenses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("backdoor_accuracy_comparison.png")

print("🎉 Visualization Completed! 2 Images saved:")
print(" - clean_accuracy_comparison.png")
print(" - backdoor_accuracy_comparison.png")
