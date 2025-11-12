import os, cv2, numpy as np, joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_DIR = "model"
TEST_ROOT = "test_videos"  # folder layout: test_videos/<class>/*.mp4
OUT_DIR = "eval_out"
os.makedirs(OUT_DIR, exist_ok=True)

clf = joblib.load(os.path.join(MODEL_DIR, "dance_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
expected_dim = int(getattr(scaler, "n_features_in_", None) or scaler.mean_.shape[0])

# ---- copy of your extractor (pose + hands), simplified to import from app if you prefer
def extract_features_from_frame(frame, pose_estimator, hand_estimator):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_estimator.process(img_rgb)
    hand_results = hand_estimator.process(img_rgb)

    if not pose_results.pose_landmarks:
        return None

    pose_lms = pose_results.pose_landmarks.landmark
    pose_coords = np.array([(lm.x, lm.y) for lm in pose_lms], dtype=np.float32)
    hip_center = (pose_coords[23] + pose_coords[24]) / 2.0
    pose_coords -= hip_center
    torso_size = np.linalg.norm(pose_coords[11] - pose_coords[23]) + 1e-6
    pose_coords /= torso_size
    pose_vector = pose_coords.flatten()

    def angle_between(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b; bc = c - b
        denom = (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        cos_v = np.dot(ba, bc)/denom
        return np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))

    def pt(i): return (pose_coords[i][0], pose_coords[i][1])
    try:
        angles = [
            angle_between(pt(11), pt(13), pt(15)),
            angle_between(pt(12), pt(14), pt(16)),
            angle_between(pt(23), pt(25), pt(27)),
            angle_between(pt(24), pt(26), pt(28)),
            angle_between(pt(11), pt(23), pt(25)),
            angle_between(pt(12), pt(24), pt(26)),
            angle_between(pt(13), pt(11), pt(23)),
            angle_between(pt(14), pt(12), pt(24)),
        ]
        angle_vector = np.array(angles, dtype=np.float32)
    except Exception:
        angle_vector = np.zeros(8, dtype=np.float32)

    left_hand_vector = np.zeros(42, dtype=np.float32)
    right_hand_vector = np.zeros(42, dtype=np.float32)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
            wrist = coords[0]
            coords -= wrist
            palm_size = np.linalg.norm(coords[0] - coords[9]) + 1e-6
            coords /= palm_size
            flat = coords.flatten()
            if handedness.classification[0].label == "Left":
                left_hand_vector = flat
            else:
                right_hand_vector = flat

    return np.concatenate([pose_vector, angle_vector, left_hand_vector, right_hand_vector])

def predict_video_label(video_path, stride=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    preds = []
    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % stride == 0:
                feats = extract_features_from_frame(frame, pose, hands)
                if feats is None or feats.shape[0] != expected_dim:
                    idx += 1; continue
                scaled = scaler.transform([feats])
                preds.append(clf.predict(scaled)[0])
            idx += 1
    cap.release()
    if not preds: return None
    # majority vote
    from statistics import mode
    try:
        p = mode(preds)
    except:
        p = preds[-1]
    return p

# --- Walk test set and gather y_true, y_pred
y_true, y_pred = [], []
classes = sorted([d.name for d in Path(TEST_ROOT).iterdir() if d.is_dir()])
for cls in classes:
    for vid in sorted((Path(TEST_ROOT)/cls).glob("*.*")):
        pred = predict_video_label(str(vid))
        if pred is None: continue
        y_pred.append(pred)
        y_true.append(label_encoder.transform([cls])[0])

# ---- Plot CM
cm = confusion_matrix(y_true, y_pred, labels=range(len(label_encoder.classes_)))
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Video-level)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ---- Optional: normalized CM + classification report
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(7,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Normalized Confusion Matrix (Video-level)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_normalized.png"))
plt.close()

report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=3)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print(report)
print(f"Saved CM images to {OUT_DIR}")
