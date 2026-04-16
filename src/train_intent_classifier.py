# train_intent_classifier.py
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv("intent_data.csv")
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Label mapping
unique_labels = df['label'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.map(lambda x: {"labels": label2id[x["label"]]}, remove_columns=["label"])
eval_dataset = eval_dataset.map(lambda x: {"labels": label2id[x["label"]]}, remove_columns=["label"])

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Training args
training_args = TrainingArguments(
    output_dir="./intent_model_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")

# Export to ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification
from pathlib import Path

ort_model = ORTModelForSequenceClassification.from_pretrained("./intent_model", export=True)
ort_model.save_pretrained("./intent_model_onnx")

print("Model exported to ONNX at ./intent_model_onnx")