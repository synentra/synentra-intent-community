# Vectra Intent Classification Model (VICM)

This repository contains code and resources for training, evaluating, and deploying an intent classification model. The system is designed to classify user intents from natural language input, enabling downstream policy enforcement and decision-making in the Vectra ecosystem.

## Overview

The project provides:

-   A pipeline for generating synthetic and curated training data\
-   Model training and evaluation scripts\
-   Export to optimized runtime formats (e.g., ONNX)\
-   Integration-ready inference components

Typical use cases include:

-   AI agent action classification\
-   API intent detection\
-   Security and policy enforcement triggers

### Getting Started

### Prerequisites

-   Python 3.8 or higher\
-   pip or compatible package manager\
-   (Optional) GPU support for faster training

Install dependencies:

``` sh
pip install -r requirements.txt
```

## Installation

Clone the repository:

``` sh
git clone <repository-url>
cd <repository-directory>
```

## Usage

### 1. Generate Training Data

``` sh
python src/generate_training_data.py
```

### 2. Train the Model

``` sh
python src/train_intent_classifier.py
```

## Model Deployment

-   ONNX: `src/intent_model_onnx/`

## License

Apache 2.0
