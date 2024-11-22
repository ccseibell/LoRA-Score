# Importing libraries
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
import warnings
import os

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import pandas as pd
import json

# Importing the arg parser
from utils import parse_args, gather_metrics

# Set warnings to ignore to keep output clean
warnings.filterwarnings('ignore')

# Parse the arguments
args = parse_args()

# Dataset selection
if args.dataset == "mnist":
    dataset = load_dataset("mnist")
    num_classes = 10
    label_column_name = 'label'
    image_col_name = "image"

elif args.dataset == "oxford-pet":
    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    num_classes = 37
    label_column_name = 'label_breed'
    image_col_name = "image"

elif args.dataset == "stanford-dogs":
    dataset = load_dataset("amaye15/stanford-dogs")
    num_classes = 120
    label_column_name = 'label'
    image_col_name = "pixel_values"

else:
    raise ValueError("Currently not supported -> You can add them now")

# Creating val/train split
dataset = dataset["train"].train_test_split(test_size=0.15, shuffle=True, seed=1)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Preprocessing for the labels
label_encoder = LabelEncoder()

def label_preprocessing(dataset):
    # Fit the encoder on the string labels and transform them to integer labels
    label_encoder.fit(dataset[label_column_name])
    encoded_labels = label_encoder.transform(dataset[label_column_name])

    # Add the encoded labels as a new column in the dataset
    return dataset.add_column('label', encoded_labels)

# Apply preprocessing
train_dataset = label_preprocessing(train_dataset)
val_dataset = label_preprocessing(val_dataset)

# Filter classes if specified
if args.classes:
    selected_classes = args.classes

    def filter_classes(batch):
        return batch['label'] in selected_classes

    train_dataset = train_dataset.filter(filter_classes)
    val_dataset = val_dataset.filter(filter_classes)

    # Update num_classes to reflect the number of selected classes
    num_classes = len(selected_classes)
else:
    selected_classes = [i for i in range(num_classes)]

# Preprocessing dataset to be compatible with ViT
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# Combined function to resize, convert to RGB, and then to tensor
def preprocess_images(batch):
    batch['pixel_values'] = [transform(image.convert("RGB")) for image in batch[image_col_name]]
    if image_col_name!='pixel_values':
        del batch[image_col_name]
    return batch

# Apply resizing
train_dataset = train_dataset.map(preprocess_images, batched=True)
val_dataset = val_dataset.map(preprocess_images, batched=True)

if args.use_lora:
    layers = ["query", "key", "value"]
    target_modules = [f"vit.encoder.layer.{i}.attention.attention.{layer}" for i in range(0, 12) for layer in layers]


    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        use_rslora=True,
    )

# Load model and tokenizer
model_name = "google/vit-base-patch16-224" 
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels = num_classes, ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Apply LoRA to the model
if args.use_lora:
    model = get_peft_model(model, lora_config)

# Move model to GPU
model = model.to("cuda")

# Define accuracy metric
accuracy = evaluate.load("accuracy")

# Define the compute_metrics function to calculate accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

run_name = "ViT"
training_args = TrainingArguments(
    output_dir=f"results/{run_name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=args.max_steps,
    logging_steps=10,
    eval_steps=10,
    save_steps=10,
    save_total_limit=1,
    evaluation_strategy="steps"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

######################
### Results Logger ###
######################

if args.use_lora:
    pass # TODO: Figure out how to store LoRA weights aggregated

else:
    metrics = gather_metrics(trainer)

    with open("out/base_metric_results.json") as f:
        curr_results = json.load(f)

    run_name = "-".join([str(i) for i in sorted(selected_classes)])
    curr_results[args.dataset][run_name] = metrics

    with open("out/base_metric_results.json", "w") as f:
        json.dump(curr_results, f)
