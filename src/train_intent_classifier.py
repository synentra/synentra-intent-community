#!/usr/bin/env python3
"""
Train an intent classifier using a DistilBERT model.
The script requires a model path (any valid DistilBERT checkpoint) and a CSV
dataset with 'text' and 'label' columns.
The --model_type flag allows distinguishing between "community" and "pro"
versions while always using the DistilBERT architecture.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Optional ONNX export
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
except ImportError:
    ORTModelForSequenceClassification = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an intent classifier using DistilBERT."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="DistilBERT model name or local path (e.g., 'distilbert-base-uncased').",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CSV file with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["community", "pro"],
        default="community",
        help="Type of the DistilBERT model (community or pro).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./intent_model",
        help="Directory where the final model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for training logs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per‑device training batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Per‑device evaluation batch size.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum token length for tokenization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the trained model to ONNX format after training.",
    )
    parser.add_argument(
        "--onnx_output_dir",
        type=str,
        default="./intent_model_onnx",
        help="Directory to save the ONNX model (only used if --export_onnx is set).",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1.0",
        help="Model version string (e.g., '1.0', '2.1.3'). Saved in metadata/config.",
    )
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def prepare_datasets(data_path: str, tokenizer, max_length: int, seed: int):
    """Load CSV, split, tokenize, and create label mappings."""
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Create label mappings
    unique_labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(unique_labels)

    # Stratified split
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    eval_dataset = Dataset.from_pandas(eval_df[["text", "label"]])

    # Single map for tokenization + label conversion
    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [label2id[l] for l in examples["label"]]
        return tokenized

    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["text", "label"])
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=["text", "label"])

    return train_dataset, eval_dataset, num_labels, label2id, id2label


																					 
	   
																	
																					 
	   
																		
							  

								  
						
																											 
						   
																						  
						   
																							 
		 
																								   

				
								
								
								   
								   
																 
												 
	 

									
												 

													  
									   
										

															 


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    logger.info(f"Loading DistilBERT tokenizer and model from: {args.model_path}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model version: {args.version}")

    # Use only DistilBERT classes – no AutoModel, no other architectures.
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
    # DistilBERT always has a pad token; no need to set it.

    # Prepare data
    train_dataset, eval_dataset, num_labels, label2id, id2label = prepare_datasets(
        args.data_path, tokenizer, args.max_length, args.seed
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model.config.version = args.version
    model.config.model_type_meta = args.model_type  # also store the type for provenance

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_dir=args.logging_dir,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=args.seed,
        report_to="none",  # disable wandb/mlflow unless explicitly wanted
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")

    metadata = {
        "version": args.version,
        "model_type": args.model_type,
        "max_length": args.max_length,
        "num_labels": num_labels,
        "trained_at": datetime.now().isoformat(),
        "dataset_size": len(train_dataset) + len(eval_dataset)
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    # Optional ONNX export
    if args.export_onnx:
        if ORTModelForSequenceClassification is None:
            logger.error(
                "ONNX export requested but 'optimum[onnxruntime]' is not installed. "
                "Install it with: pip install optimum[onnxruntime]"
            )
        else:
            logger.info("Exporting model to ONNX...")
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                output_dir, export=True
            )
            ort_output_dir = Path(args.onnx_output_dir)
            ort_output_dir.mkdir(parents=True, exist_ok=True)
            ort_model.save_pretrained(ort_output_dir)
            logger.info(f"ONNX model saved to {ort_output_dir}")
    else:
        logger.info("Skipping ONNX export (use --export_onnx to enable).")


if __name__ == "__main__":
    main()