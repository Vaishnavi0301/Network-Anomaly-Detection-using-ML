import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir=os.path.join(base_dir, "results")
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


print("="*60)
print("STEP 4: TESTING MODELS (191 FEATURES)")
print("="*60)

# Load Testing data
print("\n[INFO] Loading preprocessed test data...")
X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(output_dir, "y_test.csv")).iloc[:, 0]
X_test_np = X_test.to_numpy()
print(f"[OK] Test data loaded: {X_test.shape}")
# ============================================================

# Load Models & Scalers
print("\n[INFO] Loading models...")
svm = joblib.load(os.path.join(output_dir, "svm_fast_model.pkl"))
svm_scaler = joblib.load(os.path.join(output_dir, "svm_scaler.pkl"))
nb = joblib.load(os.path.join(output_dir, "naive_bayes_fast_model.pkl"))
xgb = joblib.load(os.path.join(output_dir, "xgboost_model.pkl"))  
lr=joblib.load(os.path.join(output_dir, "logistic_regression_model.pkl"))
rf=joblib.load(os.path.join(output_dir, "random_forest_model.pkl"))
lr_scaler = joblib.load(os.path.join(output_dir, "logistic_regression_scaler.pkl"))
print("[OK] All models loaded!")
# ============================================================

X_test_svm = svm_scaler.transform(X_test_np)
X_test_lr = lr_scaler.transform(X_test_np)

# Evaluation Function
def evaluate(model, X, y, name):
    print("\n" + "="*60)
    print(f"{name} - TEST RESULTS")
    print("="*60)
    
    y_pred = model.predict(X)
    
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    print("\nMetrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Attack"]))
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
# ============================================================

# Test Models
print("\n[TESTING MODELS ON TEST SET (191 FEATURES)]")
test_results = {}
test_results["Linear SVM"] = evaluate(svm, X_test_svm, y_test, "Linear SVM")#scalar
test_results["Naive Bayes"] = evaluate(nb, X_test_np, y_test, "Naive Bayes")
test_results["XGBoost"] = evaluate(xgb, X_test_np, y_test, "XGBoost")
test_results["Random Forest"] = evaluate(rf, X_test_np, y_test, "Random Forest")
test_results["Logistic Regression"] = evaluate(lr, X_test_lr, y_test, "Logistic Regression")#scalar
# ============================================================

# Test Comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON - TEST SET")
print("=" * 60)

comparison_df = pd.DataFrame(test_results).T
print("\n" + comparison_df.to_string())

comparison_path = os.path.join(result_dir, "test_comparison.csv")
comparison_df.to_csv(comparison_path)
print(f"\n[OK] Saved: {comparison_path}")
# ============================================================

# Identify Best Model (by F1)
best_model = max(test_results.items(), key=lambda x: x[1]["F1-Score"])
print("\n" + "=" * 60)
print(f"BEST MODEL ON TEST SET: {best_model[0]}")
print(f"F1-Score: {best_model[1]['F1-Score']:.4f}")
print("=" * 60)
# ============================================================

# ROC Curve (All Models)
print("\n" + "=" * 60)
print("GENERATING ROC CURVES")
print("=" * 60)

models_for_roc = {
    "Linear SVM": (svm, X_test_svm),
    "Naive Bayes": (nb, X_test_np),
    "XGBoost": (xgb, X_test_np),
    "Random Forest": (rf, X_test_np),
    "Logistic Regression": (lr, X_test_lr)
}

plt.figure(figsize=(10, 8))
roc_scores = {}

for name, (model, X) in models_for_roc.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    roc_scores[name] = auc

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set (191 Features)")
plt.legend()
plt.grid(alpha=0.3)

roc_path = os.path.join(result_dir, "roc_curve_test.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] ROC curve saved: {roc_path}")

# Save ROC-AUC scores
roc_df = pd.DataFrame(roc_scores.items(), columns=["Model", "ROC-AUC"])
roc_df.to_csv(os.path.join(result_dir, "roc_auc_scores_test.csv"), index=False)
# ============================================================

# Final Summary
print("\n" + "=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)

print(f"Test Samples: {X_test.shape[0]}")
print(f"Features Used: {X_test.shape[1]}")

print("\nF1-Scores:")
for model, scores in test_results.items():
    print(f" {model:20s}: {scores['F1-Score']:.4f}")

print("\nROC-AUC Scores:")
for model, auc in sorted(roc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f" {model:20s}: {auc:.4f}")

print("\n[OK] STEP-4 TESTING COMPLETED SUCCESSFULLY")
print("=" * 60)

