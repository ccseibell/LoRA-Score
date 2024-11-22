import argparse
from matplotlib import pyplot as plt

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT with optional LoRA on various datasets.")

    # Dataset and preprocessing
    parser.add_argument(
        "--dataset",
        choices=["mnist", "stanford-dogs", "oxford-pet", "FGVC-Aircraft", "caltech-101", "food101", "flowers-102"],
        required = True,
        help="Dataset to use: 'mnist' or 'oxford' (default: oxford)",
    )

    # Classes selection
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="List of classes to train on (e.g., --classes 1 7 29)."
    )

    # LoRA configuration
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Apply LoRA to the model."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="Rank parameter for LoRA (default: 64)."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="Alpha parameter for LoRA (default: 8)."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout parameter for LoRA (default: 0.1)."
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size (default: 8)."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=50,
        help="Per-device batch size (default: 8)."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 2)."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum training steps (default: 500)."
    )

    return parser.parse_args()

def plot_metrics(data):
    """Given the dictionary of the results -> Plot the results"""
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Plot training and validation loss
    axes[0].plot(data["Step"], data["Training Loss"], label="Training Loss", marker="o", color="blue")
    axes[0].plot(data["Step"], data["Validation Loss"], label="Validation Loss", marker="o", color="orange")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot validation accuracy
    axes[1].plot(data["Step"], data["Validation Accuracy"], label="Validation Accuracy", marker="o", color="green")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    return plt.show()

def gather_metrics(trainer):
    """Given a tranformer trainer, return dictionary with training and validation loss 
       as well as the accuracy."""

    log_history = trainer.state.log_history

    # Initialize lists for training and validation metrics
    steps = []
    training_loss = []
    validation_loss = []
    validation_accuracy = []

    # Extract relevant metrics
    for log in log_history:
        if 'loss' in log and 'step' in log:  # Training logs
            steps.append(log['step'])
            training_loss.append(log['loss'])
        elif 'eval_loss' in log and 'step' in log:  # Validation logs
            validation_loss.append(log['eval_loss'])
            validation_accuracy.append(log['eval_accuracy'])

    # Create a DataFrame
    data = {
        "Step": steps,
        "Training Loss": training_loss,
        "Validation Loss": validation_loss,
        "Validation Accuracy": validation_accuracy,
        "Best Results": {
            "Training Loss": min(training_loss),
            "Validation Loss": min(validation_loss),
            "Validation Accuracy": max(validation_accuracy),
        }
    }

    return data