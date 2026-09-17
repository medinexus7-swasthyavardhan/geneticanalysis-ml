# Genetic Analysis ML Prototype

## Overview

A machine learning prototype that analyzes selected synthetic SNP/genotype
features and predicts six phenotype-associated scores:

- Height
- Strength
- Power
- Endurance
- Auditory
- Neuromotor

The project provides trained machine learning models and a FastAPI interface
for prediction.

> **Important:** This is a research/demo prototype trained on synthetic data.
> The predictions should not be interpreted as deterministic predictions of
> real-world human abilities or suitability for a profession or field.

## Machine Learning

Each trait has a separate Linear Regression model trained using trait-specific
SNP features.

### Models

| Trait | Model |
|---|---|
| Height | Linear Regression |
| Strength | Linear Regression |
| Power | Linear Regression |
| Endurance | Linear Regression |
| Auditory | Linear Regression |
| Neuromotor | Linear Regression |

The models are stored as `.pkl` files.

## API

The project uses FastAPI to expose the trained models.

### Start the API

Install dependencies:

```bash
pip install -r requirements.txt