# Sentiment Analysis Mini-Challenge (NPR)

## Overview
This project is part of the **Natural Language Processing - Mini-Challenge 2: Sentiment Analysis** at FHNW. The goal is to develop and evaluate models to classify the sentiment of financial news headlines (Financial Phrasebank dataset).

The project explores:
- **Transformer-based models**: Fine-tuning BERT/ModernBERT for sentiment classification.
- **Weak Labeling**: Generating synthetic labels using sentence embeddings and similarity measures.
- **Semi-Supervised Learning**: Training models on a combination of hard (manual) and weak (synthetic) labels.
- **Evaluation**: Analyzing performance across different training data sizes and annotator agreement levels.

## Project Structure

```
├── data/               # Dataset directory (ignored by git)
├── models/             # Trained models (ignored by git)
├── notebooks/          # Jupyter notebooks for analysis and experiments
├── src/                # Source code for data loading, training, and utilities
├── verify_*.py         # Verification scripts for different pipeline stages
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key (for Bonus Task):**
    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - Open `.env` and add your OpenAI API key:
      ```
      OPENAI_API_KEY=sk-...
      ```
    - This key is required for the LLM-based bonus tasks.

## Usage

### Verification Scripts
The project includes several verification scripts to ensure the pipeline components are working correctly:

- `python verify_data.py`: Verifies data loading and splitting.
- `python verify_baseline.py`: Verifies the baseline model training pipeline.
- `python verify_weak_labeling.py`: Verifies the weak labeling generation.
- `python verify_semi_supervised.py`: Verifies the semi-supervised training loop.
- `python verify_bonus.py`: Verifies bonus tasks (LLM/UMAP).

### Notebooks
Detailed analysis and experiments are conducted in the `notebooks/` directory.

## Data
The project uses the [Financial Phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset. The data is automatically downloaded and processed by the `src/data_loader.py` module.

## Authors
- Stefan Binkert
