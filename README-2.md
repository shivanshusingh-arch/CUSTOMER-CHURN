# Customer Churn Prediction 🏦

A machine learning project that predicts whether a bank customer will churn (close their account) based on demographic and account-related features.

## Overview

This project builds a binary classifier using an Artificial Neural Network (ANN) built with TensorFlow / Keras to predict customer churn on the popular Churn Modelling dataset (10,000 customer records).

The accompanying notebook walks through:
- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- One-hot encoding of categorical variables (Geography, Gender)
- Feature scaling with StandardScaler
- Building and training a fully-connected ANN
- Evaluating the model with accuracy on a held-out test set

## Dataset

| File | Description |
|------|-------------|
| Churn_Modelling.csv | 10,000 rows x 14 columns. Target column: Exited (1 = churned, 0 = retained). |

Features include: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

The original dataset is from the Kaggle Churn Modelling dataset. Credit goes to the original author.

## Tech Stack

- Python 3.x
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- TensorFlow / Keras

## Notebook

Open the notebook to follow the full analysis:

```bash
jupyter notebook "customer churn.ipynb"
```

The notebook is self-contained. It reads Churn_Modelling.csv from a local path; update the pd.read_csv(...) line to point at the file's location on your machine if needed.

## Model

A simple fully-connected feed-forward neural network:

- Input layer: 12 features (after dropping non-predictive columns and one-hot encoding)
- Hidden layer 1: 22 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation
- Loss: Binary cross-entropy
- Optimizer: Adam
- Epochs: 50 (with 20% validation split)

## How to Run

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn tensorflow jupyter
   ```
4. Launch Jupyter and open the notebook

## Results

The model achieves reasonable accuracy on the test set — see the final cell of the notebook for the exact score on your run.