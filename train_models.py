import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "outputs")

print("="*60)
print("STEP 2: TRAINING MODELS (191 FEATURES)")
print("="*60)

print("\n[INFO] Loading preprocessed training data...")
X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv")).iloc[:, 0]

print(f"[OK] Training data loaded: {X_train.shape}")
print(f"[OK] Class distribution:\n{y_train.value_counts()}")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Train Linear SVM
svm_fold_accuracies = []
print("\n[INFO] 5-Fold CV for Linear SVM")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr = X_train.iloc[train_idx].to_numpy()
    X_val = X_train.iloc[val_idx].to_numpy()
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)   # fit ONLY on train fold
    X_val = scaler.transform(X_val)

    svm_model = LinearSVC(
    C=1.0,
    max_iter=10000,
    dual=False,
    random_state=42
    )
    svm_model.fit(X_tr, y_tr)
    y_val_pred = svm_model.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    svm_fold_accuracies.append(acc)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")
svm_mean = sum(svm_fold_accuracies) / len(svm_fold_accuracies)
print(f"\n[OK] Linear SVM CV Accuracy: {svm_mean:.4f}")

#final model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.to_numpy())
svm_model = LinearSVC(
   C=1.0,
    max_iter=10000,
    dual=False,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)
joblib.dump(svm_model, os.path.join(output_dir, "svm_fast_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "svm_scaler.pkl"))
# ============================================================

# Train Naive Bayes
print("\n" + "="*60)
print("TRAINING NAIVE BAYES (191 FEATURES)")
print("="*60)

nb_fold_accuracies = []
print("\n[INFO] 5-Fold CV for Naive Bayes")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):

    X_tr = X_train.iloc[train_idx].to_numpy()
    X_val = X_train.iloc[val_idx].to_numpy()
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    nb_model = GaussianNB()
    nb_model.fit(X_tr, y_tr)
    y_val_pred = nb_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    nb_fold_accuracies.append(acc)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")
nb_mean = sum(nb_fold_accuracies) / len(nb_fold_accuracies)
print(f"\n[OK] Naive Bayes CV Accuracy: {nb_mean:.4f}")

#final model training
nb_model = GaussianNB()
nb_model.fit(X_train.to_numpy(), y_train)
joblib.dump(nb_model, os.path.join(output_dir, "naive_bayes_fast_model.pkl"))
# ============================================================

# Train XGBoost
print("\n" + "="*60)
print("TRAINING XGBOOST (191 FEATURES)")
print("="*60)

xgb_fold_accuracies = []
print("\n[INFO] 5-Fold CV for XGBoost")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    xgb_model.fit(X_tr, y_tr)
    y_val_pred = (xgb_model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_val_pred)
    xgb_fold_accuracies.append(acc)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")
xgb_mean = sum(xgb_fold_accuracies) / len(xgb_fold_accuracies)
print(f"\n[OK] XGBoost CV Accuracy: {xgb_mean:.4f}")

#final model training
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, os.path.join(output_dir, "xgboost_model.pkl"))
# ============================================================

# Train Random Forest Classifier
print("\n" + "="*60)
print("TRAINING RANDOM FOREST (191 FEATURES)")
print("="*60)

rf_fold_accuracies = []
print("\n[INFO] 5-Fold CV for Random Forest")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    y_val_pred = rf_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    rf_fold_accuracies.append(acc)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")
rf_mean = sum(rf_fold_accuracies) / len(rf_fold_accuracies)
print(f"\n[OK] Random Forest CV Accuracy: {rf_mean:.4f}")

#final deployment
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(output_dir, "random_forest_model.pkl"))
# ============================================================

# Train Logistic regression model
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION (191 FEATURES)")
print("="*60)

lr_fold_accuracies = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr = X_train.iloc[train_idx].to_numpy()
    X_val = X_train.iloc[val_idx].to_numpy()
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_tr, y_tr)
    y_val_pred = lr_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    lr_fold_accuracies.append(acc)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")
lr_mean = sum(lr_fold_accuracies)/len(lr_fold_accuracies)
print(f"\n[OK] Logistic Regression CV Accuracy: {lr_mean:.4f}")

#final model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.to_numpy())
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, os.path.join(output_dir, "logistic_regression_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "logistic_regression_scaler.pkl"))

# ============================================================

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

print(f"\n5-Fold Cross-Validation Accuracies (191 Features):")
print(f"  Linear SVM:          {svm_mean:.4f} ({svm_mean*100:.2f}%)")
print(f"  Naive Bayes:         {nb_mean:.4f} ({nb_mean*100:.2f}%)")
print(f"  XGBoost:             {xgb_mean:.4f} ({xgb_mean*100:.2f}%)")
print(f"  Random Forest:       {rf_mean:.4f} ({rf_mean*100:.2f}%)")
print(f"  Logistic Regression: {lr_mean:.4f} ({lr_mean*100:.2f}%)")  # make sure you compute lr_mean

# Determine the best model based on CV accuracy
model_names = ['Linear SVM', 'Naive Bayes', 'XGBoost', 'Random Forest', 'Logistic Regression']
cv_accuracies = [svm_mean, nb_mean, xgb_mean, rf_mean, lr_mean]
best_model_idx = cv_accuracies.index(max(cv_accuracies))

print(f"\nBest Model Based on CV Accuracy: {model_names[best_model_idx]}")
print(f"\n[OK] All models saved to: {output_dir}")
print(f"[OK] Number of features used: {X_train.shape[1]}")
print(f"\nNext step: python test_models.py")
print("="*60)
