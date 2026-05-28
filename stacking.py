import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


if __name__=="__main__":
# Base directory and input paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    class_names = ["benign", "attack"]

    # Load datasets
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv")).iloc[:, 0]
    X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))

    # Scale data once (instead of inside pipelines)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('svm', CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)),
        ('xgb', XGBClassifier( eval_metric='logloss', random_state=42, n_jobs=-1))
    ]

    # Prepare arrays for stacking
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Out-of-fold predictions for meta-learner training
    oof_train = np.zeros((X_train.shape[0], len(base_models)))
    # Test predictions from each fold
    oof_test = np.zeros((X_test.shape[0], len(base_models), n_folds))

    # Generate OOF predictions
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\n[INFO] Training fold {fold_idx + 1}/{n_folds} ...")

        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for model_idx, (name, model) in enumerate(base_models):
            print(f"Training base model: {name}")
            model.fit(X_tr, y_tr)

            # OOF validation predictions
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_train[val_idx, model_idx] = val_preds

            # Test predictions for this fold
            test_preds = model.predict_proba(X_test_scaled)[:, 1]
            oof_test[:, model_idx, fold_idx] = test_preds

    # Average test predictions across folds
    oof_test_mean = oof_test.mean(axis=2)

    print("\n[INFO] Evaluating individual base models on OOF predictions...")
    base_model_metrics = []

    for i, (name, _) in enumerate(base_models):
        oof_preds_binary = (oof_train[:, i] >= 0.5).astype(int)
        cm = confusion_matrix(y_train, oof_preds_binary)
        acc = accuracy_score(y_train, oof_preds_binary)
        prec = precision_score(y_train, oof_preds_binary)
        rec = recall_score(y_train, oof_preds_binary)
        f1 = f1_score(y_train, oof_preds_binary)

        base_model_metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })

        print(f"\n{name.upper()} OOF Performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        
    # Train meta-learner on OOF predictions (corrected)
    print("\n[INFO] Training meta-learner (Logistic Regression)...")
    meta_learner = LogisticRegression(max_iter=5000, random_state=42)
    meta_learner.fit(oof_train, y_train)

    # Predict on test set using meta-learner
    final_test_preds_proba = meta_learner.predict_proba(oof_test_mean)[:, 1]
    final_test_preds = (final_test_preds_proba >= 0.5).astype(int)
    
    # ---------------- Evaluation on training (OOF) ----------------
    print("\n[INFO] Evaluating meta-learner on training (OOF) predictions...")
    oof_proba = meta_learner.predict_proba(oof_train)[:, 1]
    oof_preds = (oof_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_train, oof_preds)
    acc = accuracy_score(y_train, oof_preds)
    prec = precision_score(y_train, oof_preds)
    rec = recall_score(y_train, oof_preds)
    f1 = f1_score(y_train, oof_preds)

    # ---------- ROC & AUC ----------
    fpr, tpr, _ = roc_curve(y_train, oof_proba)
    auc_oof = roc_auc_score(y_train, oof_proba)

    print("\nConfusion Matrix:\n", cm)
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC (OOF): {auc_oof:.4f}")

    # ROC curve plot -> PNG
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Meta-learner (AUC = {auc_oof:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Meta-learner (OOF)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    roc_oof_path = os.path.join(results_dir, "roc_meta_oof.png")
    plt.savefig(roc_oof_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] ROC curve for meta-learner (OOF) saved: {roc_oof_path}")

    print("\n[INFO] Computing SHAP values for the meta-learner (stacked ensemble)...")
    # The meta-learner takes OOF predictions from base models as its input
    X_meta = pd.DataFrame(oof_train, columns=[name for name, _ in base_models])
    X_meta_test = pd.DataFrame(oof_test_mean, columns=[name for name, _ in base_models])

    # Initialize SHAP explainer for the logistic regression meta-model
    explainer = shap.Explainer(meta_learner, X_meta)
    shap_values = explainer(X_meta)

    # Make sure results_dir exists (you already create it later; keep that too)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    #SHAP summary (beeswarm) -> PNG
    shap.summary_plot(shap_values, X_meta, show=False)
    shap_summary_path = os.path.join(results_dir, "shap_summary_meta.png")
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP summary plot saved: {shap_summary_path}")

    meta_feature_names = [name for name, _ in base_models]

    explainer_meta = LimeTabularExplainer(
        training_data=X_meta.values,
        feature_names=meta_feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,
    )

    i = 0
    x_meta_instance = X_meta.iloc[i].values

    exp_meta = explainer_meta.explain_instance(
        data_row=x_meta_instance,
        predict_fn=meta_learner.predict_proba,
        num_features=3,
        num_samples=5000
    )
    fig_meta = exp_meta.as_pyplot_figure()
    plt.tight_layout()
    lime_meta_path = os.path.join(results_dir, f"lime_meta_instance_{i}.png")
    fig_meta.savefig(lime_meta_path, dpi=300, bbox_inches="tight")
    plt.close(fig_meta)
    print(f"[OK] LIME meta-learner plot saved: {lime_meta_path}")

    # Save results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({
        "Predicted_Label": final_test_preds,
        "Predicted_Probability": final_test_preds_proba
    }).to_csv(os.path.join(results_dir, "stacking_predictions.csv"), index=False)

    print("\n✅ Stacking complete! Predictions saved to 'results/stacking_predictions.csv'")
    