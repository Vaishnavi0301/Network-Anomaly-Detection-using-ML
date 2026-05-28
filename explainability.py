# -*- coding: utf-8 -*-

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "outputs")
result_dir = os.path.join(base_dir, "results")

os.makedirs(result_dir, exist_ok=True)

print("=" * 60)
print("STEP 5: MODEL EXPLAINABILITY (XGBOOST + SHAP + LIME)")
print("=" * 60)

# ============================================================
# Load TEST Data
# ============================================================

print("\n[INFO] Loading test data...")
X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(output_dir, "y_test.csv")).iloc[:, 0]

# ---- Ensure numeric (CRITICAL) ----
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

feature_names = X_test.columns.tolist()
X_test_np = X_test.to_numpy()

print(f"[OK] Test data loaded: {X_test_np.shape}")

# ============================================================
# Load Trained XGBoost Model
# ============================================================

print("\n[INFO] Loading trained XGBoost model...")
xgb_model = joblib.load(os.path.join(output_dir, "xgboost_model.pkl"))
print("[OK] Model loaded")

# ============================================================
# SHAP EXPLAINABILITY (KernelExplainer — SAFE)
# ============================================================

print("\n[INFO] Initializing SHAP KernelExplainer...")

# Background sample (small for stability)
background = shap.sample(X_test_np, 100, random_state=42)

# Probability wrapper (1D output)
def xgb_predict(X):
    return xgb_model.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(
    lambda X: xgb_predict(X),
    background
)

print("[OK] SHAP explainer initialized")

# Use small subset for global plots
X_shap = X_test_np[:100]

print("[INFO] Computing SHAP values (this may take a minute)...")
shap_values = explainer.shap_values(X_shap)
print("[OK] SHAP values computed")

# ============================================================
# 1. SHAP Summary Plot (Beeswarm)
# ============================================================

print("[INFO] Generating SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "shap_summary_beeswarm.png"), dpi=300)
plt.close()

# ============================================================
# 2. SHAP Feature Importance (Bar Plot)
# ============================================================

print("[INFO] Generating SHAP bar plot...")
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "shap_feature_importance_bar.png"), dpi=300)
plt.close()

# ============================================================
# 3. SHAP Dependence Plot (Top Feature)
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap)
top_feature = feature_names[top_feature_idx]

print(f"[INFO] Generating dependence plot for: {top_feature}")

shap.dependence_plot(
    top_feature_idx,
    shap_values,
    X_shap,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, f"shap_dependence_{top_feature}.png"),
    dpi=300
)
plt.close()

# ============================================================
# 4. SHAP Local Explanation (Force Plot – Attack Sample)
# ============================================================

print("[INFO] Generating SHAP force plot...")

attack_indices = y_test[y_test == 1].index
sample_idx = attack_indices[0]

sample_np = X_test_np[sample_idx:sample_idx + 1]

shap_single = explainer.shap_values(sample_np)

shap.force_plot(
    explainer.expected_value,
    shap_single,
    sample_np,
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.savefig(os.path.join(result_dir, "shap_force_plot_attack.png"), dpi=300)
plt.close()

# ============================================================
# 5. LIME Local Explanation (TEST DATA)
# ============================================================

print("\n[INFO] Generating LIME explanation...")

class_names = ["Normal", "Attack"]

lime_explainer = LimeTabularExplainer(
    training_data=X_test_np,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    discretize_continuous=True
)

def xgb_predict_proba(data):
    return xgb_model.predict_proba(data)

lime_exp = lime_explainer.explain_instance(
    data_row=sample_np[0],
    predict_fn=xgb_predict_proba,
    num_features=10
)

fig = lime_exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig(os.path.join(result_dir, "lime_explanation_attack.png"), dpi=300)
plt.close(fig)

print("[OK] LIME explanation generated")

# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 60)
print("EXPLAINABILITY COMPLETED SUCCESSFULLY")
print("Generated files:")
print(" - shap_summary_beeswarm.png")
print(" - shap_feature_importance_bar.png")
print(" - shap_dependence_<feature>.png")
print(" - shap_force_plot_attack.png")
print(" - lime_explanation_attack.png")
print("=" * 60)
