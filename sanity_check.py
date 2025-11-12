# sanity_check.py
import os, glob, joblib, numpy as np, pandas as pd, cv2
from collections import Counter
from train_model import extract_features_from_landmarks, save_landmarks, KEYPOINTS_DIR, landmarks_visibility_ok

MODEL_DIR = "model"
DATASET = "dataset"
model = joblib.load(os.path.join(MODEL_DIR, "dance_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def predict_image(img_path, label):
    base = os.path.splitext(os.path.basename(img_path))[0]
    csv_path = os.path.join(KEYPOINTS_DIR, label, f"{base}.csv")
    if os.path.exists(csv_path):
        lms = pd.read_csv(csv_path).values
    else:
        lms = save_landmarks(img_path, label, KEYPOINTS_DIR)
        if lms is None:
            return None
    if not landmarks_visibility_ok(lms, 0.5, 18):
        return None
    feat = extract_features_from_landmarks(lms)
    X = scaler.transform([feat])
    idx = model.predict(X)[0]
    return le.inverse_transform([idx])[0]

def sample_images(folder, k=5):
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        paths.extend(glob.glob(os.path.join(DATASET, folder, ext)))
    return paths[:k]

labels = [d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d))]
print("Found classes:", labels)

results = {}
for lab in labels:
    paths = sample_images(lab, k=5)
    preds = []
    for p in paths:
        pred = predict_image(p, lab)
        if pred: preds.append(pred)
    results[lab] = Counter(preds)

print("\nPer-class predictions on sample images:")
for lab, cnt in results.items():
    total = sum(cnt.values())
    print(f"  {lab:15s} -> {dict(cnt)}  (n={total})")
