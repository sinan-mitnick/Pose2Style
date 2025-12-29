import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# CONFIG
# ------------------------
FEATURES_PATH = "features/all_features.csv"
MODEL_DIR = "model"
PLOT_DIR = "model_plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

STATIC_DIM = 233
DELTA_DIM = 233
TOTAL_DIM = STATIC_DIM + DELTA_DIM   # 466


# ------------------------
# LOAD STATIC FEATURES
# ------------------------
print("Loading existing 233-D features...")
df = pd.read_csv(FEATURES_PATH)

X_static = df.drop("label", axis=1).astype(np.float32).values
y = df["label"].values

print("Found:", X_static.shape[0], "samples")
print("Feature dim:", X_static.shape[1])
print("Classes:", Counter(y))

if X_static.shape[1] != STATIC_DIM:
    raise ValueError(f"Expected 233 dims but got {X_static.shape[1]}")


# ------------------------
# BUILD DELTA FEATURES
# ------------------------
print("\nBuilding Δ-motion features...")

X_delta = np.zeros_like(X_static)

df["label_shifted"] = df["label"].shift(1)

for i in range(1, len(df)):
    if df.loc[i, "label"] == df.loc[i - 1, "label"]:
        X_delta[i] = X_static[i] - X_static[i - 1]

X_full = np.concatenate([X_static, X_delta], axis=1)
print("Δ-motion done → new shape:", X_full.shape)


# ------------------------
# ENCODING
# ------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)


# ------------------------
# TRAIN / TEST SPLIT
# ------------------------
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X_full, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

# scale static & delta
scaler_static = StandardScaler()
scaler_delta  = StandardScaler()

X_train_static = scaler_static.fit_transform(X_train_raw[:, :STATIC_DIM])
X_test_static  = scaler_static.transform(X_test_raw[:, :STATIC_DIM])

X_train_delta  = scaler_delta.fit_transform(X_train_raw[:, STATIC_DIM:])
X_test_delta   = scaler_delta.transform(X_test_raw[:, STATIC_DIM:])


# ------------------------
# TRAIN STATIC MODEL
# ------------------------
print("\nTraining STATIC model (233-D)...")

weights_static = compute_sample_weight("balanced", y_train_raw)

model_static = XGBClassifier(
    n_estimators=700,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.2,
    reg_alpha=4,
    reg_lambda=20,
    objective="multi:softprob",
    eval_metric=["mlogloss", "merror"],
    n_jobs=-1,
)

model_static.fit(
    X_train_static,
    y_train_raw,
    sample_weight=weights_static,
    eval_set=[(X_train_static, y_train_raw), (X_test_static, y_test)],
    verbose=False
)


# ------------------------
# TRAIN MOTION MODEL
# ------------------------
print("\nTraining MOTION model (233-D Δ)...")

weights_delta = compute_sample_weight("balanced", y_train_raw)

model_motion = XGBClassifier(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    objective="multi:softprob",
    eval_metric=["mlogloss", "merror"],
    n_jobs=-1,
)

model_motion.fit(
    X_train_delta,
    y_train_raw,
    sample_weight=weights_delta,
    eval_set=[(X_train_delta, y_train_raw), (X_test_delta, y_test)],
    verbose=False
)


# ------------------------
# STACKED FINAL MODEL
# ------------------------
print("\nTraining COMBINED stacked model (best accuracy)...")

stack_train = np.hstack([
    model_static.predict_proba(X_train_static),
    model_motion.predict_proba(X_train_delta)
])

stack_test = np.hstack([
    model_static.predict_proba(X_test_static),
    model_motion.predict_proba(X_test_delta)
])

model_stack = XGBClassifier(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    n_jobs=-1,
)

model_stack.fit(stack_train, y_train_raw)


# ------------------------
# EVALUATE STACKED MODEL
# ------------------------
y_pred = model_stack.predict(stack_test)

acc = accuracy_score(y_test, y_pred)
print("\nFINAL ACCURACY =", acc)
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ------------------------
# SAVE MODELS
# ------------------------
joblib.dump(model_static, os.path.join(MODEL_DIR, "dance_model.pkl"))
joblib.dump(model_motion, os.path.join(MODEL_DIR, "dance_model_motion.pkl"))
joblib.dump(model_stack,  os.path.join(MODEL_DIR, "dance_model_combined.pkl"))

joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(scaler_static, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(scaler_delta,  os.path.join(MODEL_DIR, "scaler_delta.pkl"))

print("\n✔ Saved all models & scalers.")


# ---------------------------------------------------------------
# 📊 FIXED TRAINING CURVES (aligned lengths)
# ---------------------------------------------------------------

print("\nGenerating final training graphs (accuracy + loss)...")

evals_static = model_static.evals_result()
evals_motion = model_motion.evals_result()

# Extract curves
static_loss_train = np.array(evals_static["validation_0"]["mlogloss"])
static_loss_val   = np.array(evals_static["validation_1"]["mlogloss"])

motion_loss_train = np.array(evals_motion["validation_0"]["mlogloss"])
motion_loss_val   = np.array(evals_motion["validation_1"]["mlogloss"])

static_acc_train = 1 - np.array(evals_static["validation_0"]["merror"])
static_acc_val   = 1 - np.array(evals_static["validation_1"]["merror"])

motion_acc_train = 1 - np.array(evals_motion["validation_0"]["merror"])
motion_acc_val   = 1 - np.array(evals_motion["validation_1"]["merror"])

# ALIGN lengths
min_len = min(len(static_loss_train), len(motion_loss_train))

static_loss_train = static_loss_train[:min_len]
static_loss_val   = static_loss_val[:min_len]
motion_loss_train = motion_loss_train[:min_len]
motion_loss_val   = motion_loss_val[:min_len]

static_acc_train = static_acc_train[:min_len]
static_acc_val   = static_acc_val[:min_len]
motion_acc_train = motion_acc_train[:min_len]
motion_acc_val   = motion_acc_val[:min_len]

# AVERAGE
train_loss = (static_loss_train + motion_loss_train) / 2
val_loss   = (static_loss_val   + motion_loss_val) / 2

train_acc  = (static_acc_train  + motion_acc_train) / 2
val_acc    = (static_acc_val    + motion_acc_val) / 2


# -----------------------------
# ACCURACY CURVE (FINAL)
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.savefig(f"{PLOT_DIR}/training_accuracy.png")
plt.close()


# -----------------------------
# LOSS CURVE (FINAL)
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(f"{PLOT_DIR}/training_loss.png")
plt.close()

print("✔ Final graphs saved in:", PLOT_DIR)


# ---------------------------------------------------------------
# 📌 CONFUSION MATRIX → model/confusion_matrix.png
# ---------------------------------------------------------------

print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

print("Confusion matrix saved in:", MODEL_DIR)
print("\nTraining complete.")
