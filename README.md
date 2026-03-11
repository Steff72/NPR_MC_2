# Sentiment Analysis Mini-Challenge (NPR)

## Overview
This repository contains a compact sentiment analysis workflow for the FHNW NLP mini-challenge. The main experiment flow lives in `npr_mc_2.ipynb`, and the reusable Hugging Face classifier wrapper lives in `models.py`.

## Project Structure

```
├── npr_mc_2.ipynb      # End-to-end notebook for data loading, training, weak labeling, and bonus analysis
├── models.py           # Reusable transformer classifier wrapper
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

Training runs write model artifacts to a runtime-created `models/` directory.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Optional, for the OpenAI bonus section, ensure `OPENAI_API_KEY` is already set in your environment or notebook session.

## Usage

Open `npr_mc_2.ipynb` and run the notebook top to bottom. It:

- downloads the Financial Phrasebank dataset from Hugging Face
- creates train/validation/test splits
- trains baseline transformer models
- generates weak labels with sentence embeddings and k-NN
- evaluates the semi-supervised setup
- optionally runs the OpenAI bonus section when `OPENAI_API_KEY` is available

## Data
The dataset is loaded directly inside `npr_mc_2.ipynb` via `datasets.load_dataset(...)` using [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).

## Author
Stefan Binkert
