# Network Anomaly Detection using Machine Learning

A supervised machine learning pipeline for detecting network intrusions and anomalies using the *UNSW-NB15* dataset. The project trains and evaluates five classifiers, combines them via a stacking ensemble, and provides full model explainability through SHAP and LIME.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Code Files](#code-files)
- [Pipeline](#pipeline)
- [Models](#models)
- [Explainability](#explainability)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

This project detects network anomalies (normal vs. attack traffic) through a full end-to-end ML pipeline covering preprocessing, training, evaluation, stacking ensemble, and explainability.

- *Binary classification:* Normal (0) vs. Attack (1)
- *191 features* after encoding
- *5-fold stratified cross-validation* for all models
- *SHAP + LIME* for global and local model interpretability

---

## Dataset

*UNSW-NB15* — A network intrusion detection dataset created by the Australian Centre for Cyber Security (ACCS).

Download the dataset from the [official UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place both CSV files in the project root before running any scripts:


UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv


| Column | Description |
|--------|-------------|
| label | Target — 0 = Normal, 1 = Attack |
| attack_cat | Attack category (dropped during preprocessing) |
| id | Row identifier (dropped during preprocessing) |

---

## Code Files

| File | Description |
|------|-------------|
| Preprocess.py | Step 1 — Cleans, encodes, and aligns train/test features |
| train_models.py | Step 2 — Trains 5 classifiers with 5-fold CV |
| test_models.py | Step 3 — Evaluates all models on the held-out test set |
| stacking.py | Step 4 — Builds a stacking ensemble with a meta-learner |
| explainability.py | Step 5 — SHAP and LIME explanations for XGBoost |
| run_all.py | Master script — runs all 5 steps in sequence |

---

## Pipeline


Preprocess.py  →  train_models.py  →  test_models.py  →  stacking.py  →  explainability.py
   (Step 1)           (Step 2)           (Step 3)          (Step 4)          (Step 5)


### Step 1 — Preprocessing (Preprocess.py)
- Loads training and test CSVs
- Handles missing values (median for numeric, mode for categorical)
- One-hot encodes categorical columns
- Aligns test features to match training features exactly
- Saves processed data and feature metadata to outputs/

### Step 2 — Training (train_models.py)
- Runs 5-fold stratified cross-validation for each model
- Trains a final model on the full training set
- Saves all models and scalers to outputs/

### Step 3 — Testing (test_models.py)
- Loads all saved models and evaluates on the held-out test set
- Reports confusion matrix, accuracy, precision, recall, F1, and ROC-AUC per model
- Saves a combined ROC curve plot and comparison CSV to results/

### Step 4 — Stacking Ensemble (stacking.py)
- Uses RF, SVM (calibrated), and XGBoost as base learners
- Generates out-of-fold (OOF) predictions via 5-fold CV
- Trains a Logistic Regression meta-learner on OOF predictions
- Produces ROC curve, SHAP summary, and LIME explanation for the meta-learner

### Step 5 — Explainability (explainability.py)
- Applies SHAP KernelExplainer to the XGBoost model
- Generates global (beeswarm, bar, dependence) and local (force plot) SHAP visualizations
- Generates a LIME local explanation for an attack-class sample

---

## Models

| Model | Details |
|-------|---------|
| Linear SVM | LinearSVC with StandardScaler |
| Naive Bayes | GaussianNB; no scaling required |
| XGBoost | 200 estimators, depth 6, histogram tree method |
| Random Forest | 300 estimators, max depth 30 |
| Logistic Regression | LogisticRegression with StandardScaler |
| *Stacking Ensemble* | RF + SVM + XGBoost → Logistic Regression meta-learner |

---

## Explainability

### SHAP — XGBoost model
| Output | Description |
|--------|-------------|
| shap_summary_beeswarm.png | Feature impact distribution across 100 test samples |
| shap_feature_importance_bar.png | Mean absolute SHAP values per feature |
| shap_dependence_<feature>.png | Dependence plot for the top feature |
| shap_force_plot_attack.png | Local explanation for a single attack sample |

### LIME — XGBoost model
| Output | Description |
|--------|-------------|
| lime_explanation_attack.png | Top-10 feature contributions for an attack sample |

### SHAP + LIME — Stacking meta-learner
| Output | Description |
|--------|-------------|
| shap_summary_meta.png | Base model contribution to meta-learner decisions |
| lime_meta_instance_0.png | Local LIME explanation for the meta-learner |

---

## Installation

bash
# Clone the repository
git clone https://github.com/Vaishnavi0301/Network-Anomaly-Detection-using-ML.git
cd Network-Anomaly-Detection-using-ML

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap lime matplotlib joblib


> Python 3.8+ recommended.

---

## Usage

### Run the full pipeline at once
bash
python run_all.py


### Run individual steps
bash
python Preprocess.py        # Step 1
python train_models.py      # Step 2
python test_models.py       # Step 3
python stacking.py          # Step 4
python explainability.py    # Step 5


> Steps must be run in order — each script depends on outputs from the previous one.
