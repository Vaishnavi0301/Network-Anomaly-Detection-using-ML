"""
Master Script: Run all steps in sequence
"""

import subprocess
import sys

steps = [
    ("Preprocess.py", "DATA PREPROCESSING"),
    ("train_models.py", "TRAIN SVM, NAIVE BAYES, XGBOOST, RANDOM FOREST & LOGISTIC REGRESSION"),
    ("test_models.py", "TEST MODELS SVM, NAIVE BAYES, XGBOOST, RANDOM FOREST & LOGISTIC REGRESSION"),
    ("stacking.py", "ENSEMBLE LEARNING"),
    ("explainability.py", "MODEL EXPLAINABILITY"),
]

print("\n" + "="*60)
print("NETWORK ANAMOLY DETECTION - SUPERVISED LEARNING")
print("Models: SVM, Naive Bayes, XGBOOST, RF & LR")
print("="*60)

for i, (script, description) in enumerate(steps, 1):
    print(f"\n[STEP {i}] {description}")
    print("="*60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed!")
        sys.exit(1)

print("\n" + "="*60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*60)
