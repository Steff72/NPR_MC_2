# Sentiment Analysis Mini-Challenge (NPR)

## Overview
This repository is a notebook-centered sentiment-classification project for the FHNW NLP mini-challenge. The full workflow lives in `npr_mc_2.ipynb`; `models.py` only contains the small reusable classifier wrapper and metric helpers that the notebook calls.

The notebook covers:

- agreement-aware train/validation/test splitting
- baseline transformer comparison and epoch calibration
- embedding comparison and weak-label generation
- semi-supervised retraining and learning curves
- agreement-level comparison
- an optional OpenAI bonus section that skips cleanly when the client or API key is unavailable

## Repository Layout

```text
├── npr_mc_2.ipynb   # Main end-to-end analysis notebook
├── models.py        # Small Hugging Face classifier wrapper and metrics
├── requirements.txt # Python dependencies
├── pics/            # Static image used by the report
├── _quarto.yml      # Quarto config for rendering
└── README.md        # Project documentation
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in Jupyter or VS Code and select the `.venv` interpreter/kernel.
4. Optional: set `OPENAI_API_KEY` if you want to run the bonus LLM section.

## Running The Notebook

Run `npr_mc_2.ipynb` from top to bottom.

The notebook:

- loads Financial PhraseBank agreement subsets
- falls back to a local `FinancialPhraseBank-v1.0` directory if one is already available
- otherwise loads the same data from [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)
- trains the baseline transformer models directly from notebook cells
- writes model outputs and intermediate trainer state to a runtime-generated `models/` directory
- keeps local plotting and numba caches inside the project folder for easier cleanup

## Notes

- The workflow is intentionally notebook-first rather than packaged as a training framework.
- The optional OpenAI section is guarded and will be skipped automatically if the client import fails or `OPENAI_API_KEY` is not set.
