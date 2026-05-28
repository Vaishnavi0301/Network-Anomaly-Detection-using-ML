# -*- coding: utf-8 -*-

# ============================================================
# STEP 1: Preprocess UNSW-NB15 training data (191 FEATURES)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_file = os.path.join(base_dir, "UNSW_NB15_training-set.csv")

if not os.path.exists(train_file):
    raise FileNotFoundError(f"[ERROR] Dataset not found at: {train_file}")

print(f"[INFO] Loading training dataset...")
df = pd.read_csv(train_file)
print(f"[OK] Training dataset loaded! Shape: {df.shape}")

target_column = "label"
X = df.drop(columns=[target_column, "attack_cat", "id"], errors="ignore")
y = df[target_column]

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("[INFO] Handling missing values...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if np.issubdtype(X[col].dtype, np.number):
            X[col].fillna(X[col].median(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0], inplace=True)

# ============================================================
# ENCODE CATEGORICAL COLUMNS
# ============================================================

print("[INFO] Encoding categorical columns...")
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print(f"[INFO] Features after encoding: {X.shape[1]}")

# ============================================================
# NO LOW-VARIANCE REMOVAL - KEEP ALL FEATURES
# ============================================================

print("[INFO] Keeping all features (NO low-variance removal)...")
print(f"[OK] Total features: {X.shape[1]}")

# Store training features for test set matching
training_features = X.columns.tolist()

print("[INFO] Saving preprocessed data...")
X.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
pd.DataFrame(y).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
print("[OK] Training data saved!(without scaling)")


# ============================================================
# NO FEATURE SELECTION - USE ALL 191 FEATURES
# ============================================================

print("[INFO] Using all available features (NO FEATURE SELECTION)...")

selected_features = training_features
print(f"[OK] Total features to use: {len(selected_features)}")

# Print selected features
print(f"[INFO] Selected feature names: {selected_features}")

joblib.dump(training_features, os.path.join(output_dir, "training_features.pkl"))


# ============================================================
# STEP 2: PREPROCESS TEST DATA (MATCH FEATURES EXACTLY)
# ============================================================

test_file = os.path.join(base_dir, "UNSW_NB15_testing-set.csv")

if not os.path.exists(test_file):
    print("\n[WARNING] Test dataset not found. Skipping test preprocessing.")
else:
    print("\n[INFO] Loading test dataset...")
    df_test = pd.read_csv(test_file)
    print(f"[OK] Test dataset loaded! Shape: {df_test.shape}")

    X_test = df_test.drop(columns=["label", "attack_cat", "id"], errors="ignore")
    y_test = df_test["label"].astype(int)

    # Handle missing values
    print("[INFO] Handling missing values in test set...")
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            if np.issubdtype(X_test[col].dtype, np.number):
                X_test[col].fillna(X_test[col].median(), inplace=True)
            else:
                X_test[col].fillna(X_test[col].mode()[0], inplace=True)

    # Encode categorical features
    print("[INFO] Encoding categorical columns in test set...")
    if len(cat_cols) > 0:
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # Match test features with training features
    print("[INFO] Matching test features with training features...")
    
    # Add missing columns with 0 values
    for col in training_features:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Remove extra columns not in training
    extra_cols = [col for col in X_test.columns if col not in training_features]
    if extra_cols:
        X_test = X_test.drop(columns=extra_cols)
    
    # Reorder columns to match training set
    X_test = X_test[training_features]
    
    print(f"[OK] Test features matched! Shape: {X_test.shape}")

    # Scale features (NO feature selection - use all features)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print("[OK] Test data preprocessing complete and saved!")

# ============================================================
# SAVE FEATURE NAMES FOR SHAP
# ============================================================

print("[INFO] Saving feature names...")
with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
    for feature in selected_features:
        f.write(feature + "\n")
print("[OK] Feature names saved!")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Training data shape: {X.shape}")
try:
    print(f"Test data shape: {X_test.shape}")
except:
    print(f"Test data: Not processed")
print(f"Selected features: {len(selected_features)}")
print(f"Training features available: {len(training_features)}")
print(f"\nNext step: python train_models.py")
print("="*60)