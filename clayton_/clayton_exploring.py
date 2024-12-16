import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON file
with open("out/text/arxiv/lora_clamped_results.json") as f:
    lora_data = json.load(f)

# Function to calculate the Gini coefficient
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Ensure non-negative values
    array += 1e-10  # Avoid division by zero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    return (np.sum((2 * index - array.size - 1) * array)) / (array.size * np.sum(array))

# Extract and plot metrics
def plot_metrics(metrics, dataset_name):
    steps = metrics["Step"]
    training_loss = metrics["Training Loss"]
    validation_loss = metrics["Validation Loss"]
    validation_accuracy = metrics["Validation Accuracy"]

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Training and Validation Loss
    ax[0].plot(steps, training_loss, label="Training Loss", marker="o", color="blue")
    ax[0].plot(steps, validation_loss, label="Validation Loss", marker="o", color="orange")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Plot Validation Accuracy
    ax[1].plot(steps, validation_accuracy, label="Validation Accuracy", marker="o", color="green")
    ax[1].set_title("Validation Accuracy Curve")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    # Adjust layout and show plots
    fig.suptitle(f"Metrics for {dataset_name}")
    plt.tight_layout()
    plt.show()

# Extract Gini coefficients for SVD Diagonal Entries
def extract_svd_gini(svd_entries):
    gini_coeffs = {"query": [], "key": [], "value": []}

    for layer_name, layer_data in svd_entries.items():
        for entry_type, svd_values in layer_data.items():
            gini_score = gini(np.array(svd_values))
            gini_coeffs[entry_type].append(gini_score)

    return gini_coeffs

# Plot Gini coefficients
def plot_gini_coefficients(gini_coeffs, dataset_name):
    plt.figure(figsize=(10, 6))

    # Plot for each type
    for entry_type, coeffs in gini_coeffs.items():
        plt.plot(range(len(coeffs)), coeffs, label=f"{entry_type.capitalize()} Gini Coefficients")

    # Add labels and legend
    plt.title(f"Gini Coefficients for {dataset_name}")
    plt.xlabel("Layer Index")
    plt.ylabel("Gini Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()

# Process each dataset in lora_clamped_results.json
for dataset_key, dataset_data in lora_data.items():
    # Extract metrics
    metrics = dataset_data["Metrics"]
    plot_metrics(metrics, dataset_key)

    # Extract and plot Gini coefficients
    svd_entries = dataset_data["SVD Diagonal Entries"]
    gini_coeffs = extract_svd_gini(svd_entries)
    plot_gini_coefficients(gini_coeffs, dataset_key)
