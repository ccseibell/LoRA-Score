# %% [markdown]
# # Importing Libraries and Loading Data

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Load Arxiv Dataset Results from JSON File

# %%
with open("out/base_metric_results.json") as f:
    training_data = json.load(f)

# %% [markdown]
# # Gini Coefficient Calculation Function

# %%
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Ensure non-negative values
    array += 1e-10  # Avoid division by zero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    return (np.sum((2 * index - array.size - 1) * array)) / (array.size * np.sum(array))

# %% [markdown]
# # Extract Gini Coefficients for LoRA Layers in Arxiv Dataset

# %%
def extract_gini_coefficients(dataset_data):
    gini_coeffs_a = []
    gini_coeffs_b = []
    layer_types_a = []
    layer_types_b = []

    for key, value in dataset_data.items():
        svd_entries = value.get("SVD Diagonal Entries", {})
        for layer_name, svd_values in svd_entries.items():
            gini_score = gini(np.array(svd_values))
            if "lora_A" in layer_name:
                gini_coeffs_a.append(gini_score)
                if "query" in layer_name:
                    layer_types_a.append("query")
                elif "key" in layer_name:
                    layer_types_a.append("key")
                elif "value" in layer_name:
                    layer_types_a.append("value")
            elif "lora_B" in layer_name:
                gini_coeffs_b.append(gini_score)
                if "query" in layer_name:
                    layer_types_b.append("query")
                elif "key" in layer_name:
                    layer_types_b.append("key")
                elif "value" in layer_name:
                    layer_types_b.append("value")

    return gini_coeffs_a, gini_coeffs_b, layer_types_a, layer_types_b

# %% [markdown]
# # Map Layer Types to Colors

# %%
color_map = {"query": "purple", "key": "green", "value": "turquoise"}

# %% [markdown]
# # Plot Gini Coefficients

# %%
def plot_gini_coefficients(gini_coeffs_a, gini_coeffs_b, layer_types_a, layer_types_b, dataset_name):
    plt.figure(figsize=(12, 6))

    # Colors for dots
    dot_colors_a = [color_map[layer] for layer in layer_types_a]
    dot_colors_b = [color_map[layer] for layer in layer_types_b]

    # Plot LoRA_A (red line) with color-coded dots
    plt.plot(range(len(gini_coeffs_a)), gini_coeffs_a, color="red", linestyle="--", label="LoRA_A (Red)")
    for i, color in enumerate(dot_colors_a):
        plt.scatter(i, gini_coeffs_a[i], color=color, edgecolor="black", s=50)

    # Plot LoRA_B (blue line) with color-coded dots
    plt.plot(range(len(gini_coeffs_b)), gini_coeffs_b, color="blue", linestyle="--", label="LoRA_B (Blue)")
    for i, color in enumerate(dot_colors_b):
        plt.scatter(i, gini_coeffs_b[i], color=color, edgecolor="black", s=50)

    # Add average lines
    average_score_a = np.mean(gini_coeffs_a) if gini_coeffs_a else 0
    average_score_b = np.mean(gini_coeffs_b) if gini_coeffs_b else 0
    plt.axhline(y=average_score_a, color="darkred", linestyle="--", label=f"Average Gini (LoRA_A: {average_score_a:.3f})")
    plt.axhline(y=average_score_b, color="darkblue", linestyle="--", label=f"Average Gini (LoRA_B: {average_score_b:.3f})")

    # Add legend for colors
    plt.scatter([], [], color="purple", edgecolor="black", s=50, label="Query Layers")
    plt.scatter([], [], color="green", edgecolor="black", s=50, label="Key Layers")
    plt.scatter([], [], color="turquoise", edgecolor="black", s=50, label="Value Layers")

    # Labels, legend, and grid
    plt.xlabel("Layer Index")
    plt.ylabel("Gini Coefficient")
    plt.title(f"{dataset_name}: Gini Coefficients of Singular Values Across Layers")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% [markdown]
# # Process and Plot for Arxiv Dataset

# %%
if "arxiv" in training_data:
    arxiv_data = training_data["arxiv"]
    gini_coeffs_a, gini_coeffs_b, layer_types_a, layer_types_b = extract_gini_coefficients(arxiv_data)
    plot_gini_coefficients(gini_coeffs_a, gini_coeffs_b, layer_types_a, layer_types_b, "Arxiv Dataset")
else:
    print("No Arxiv dataset found in the provided JSON file.")
