import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# --- Config ---
DATA_ROOT           = "dataset"
FEATURES_DIR        = "features"
FEATURES_PATH       = os.path.join(FEATURES_DIR, "all_features.csv")

STATIC_DIM          = 233  # 99 pose + 8 angles + 63 LH + 63 RH

os.makedirs(FEATURES_DIR, exist_ok=True)

# --- MediaPipe ---
mp_pose   = mp.solutions.pose
mp_hands  = mp.solutions.hands

# --- Worker globals ---
_worker_pose  = None
_worker_hands = None


def init_worker():
    global _worker_pose, _worker_hands
    _worker_pose  = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    _worker_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def extract_233d_features(img_path):
    """Same 233-D static descriptor you trained on (pose+vis+angles+hands)."""
    global _worker_pose, _worker_hands

    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_results = _worker_pose.process(img_rgb)
    hand_results = _worker_hands.process(img_rgb)

    if not pose_results.pose_landmarks:
        return None

    # Pose: 33 x (x, y, visibility)
    pose_lms = pose_results.pose_landmarks.landmark
    pose_arr = np.array([[lm.x, lm.y, lm.visibility] for lm in pose_lms], dtype=np.float32)

    coords = pose_arr[:, :2]
    hip = (coords[23] + coords[24]) / 2.0
    coords -= hip
    torso = np.linalg.norm(coords[11] - coords[23]) + 1e-6
    coords /= torso
    pose_arr[:, :2] = coords

    pose_vec = pose_arr.flatten()  # 99

    # Angles (8) using normalized x,y
    def ang(a, b, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_v = np.dot(ba, bc) / denom
        return np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))

    def pt(i):
        return coords[i]

    try:
        angles = np.array([
            ang(pt(11), pt(13), pt(15)),
            ang(pt(12), pt(14), pt(16)),
            ang(pt(23), pt(25), pt(27)),
            ang(pt(24), pt(26), pt(28)),
            ang(pt(11), pt(23), pt(25)),
            ang(pt(12), pt(24), pt(26)),
            ang(pt(13), pt(11), pt(23)),
            ang(pt(14), pt(12), pt(24)),
        ], dtype=np.float32)
    except Exception:
        angles = np.zeros(8, dtype=np.float32)

    # Hands: 21 x (x, y, presence)
    LH = np.zeros(63, dtype=np.float32)
    RH = np.zeros(63, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for h_lms, handed in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            coords_h = np.array([[lm.x, lm.y] for lm in h_lms.landmark], dtype=np.float32)
            coords_h -= coords_h[0]
            palm = np.linalg.norm(coords_h[0] - coords_h[9]) + 1e-6
            coords_h /= palm

            presence = np.ones((21, 1), dtype=np.float32)
            full = np.concatenate([coords_h, presence], axis=1).flatten()  # 63

            if handed.classification[0].label == "Left":
                LH = full
            else:
                RH = full

    feat = np.concatenate([pose_vec, angles, LH, RH])
    if feat.shape[0] != STATIC_DIM:
        return None
    return feat


def extract_and_label(task):
    img_path, label = task
    feat = extract_233d_features(img_path)
    if feat is None:
        return None
    return np.append(feat, label)


def main():
    # 1) Load existing features CSV
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ {FEATURES_PATH} not found. Run your original 233-D feature build at least once.")
        return

    df_existing = pd.read_csv(FEATURES_PATH)
    if df_existing.shape[1] - 1 != STATIC_DIM:
        print(f"❌ Existing CSV has {df_existing.shape[1]-1} features, expected {STATIC_DIM}.")
        return

    existing_labels = set(df_existing["label"].unique())
    print("Existing labels in features.csv:", existing_labels)

    # 2) Find new class folders in dataset
    data_root = Path(DATA_ROOT)
    tasks = []
    new_labels = []

    for cls_dir in sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
        label = cls_dir.name
        if label in existing_labels:
            continue  # already in CSV → skip

        new_labels.append(label)
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                tasks.append((str(img_path), label))

    if not tasks:
        print("No NEW labels found. Nothing to add.")
        return

    print("New labels detected:", new_labels)
    print(f"Total new images to process: {len(tasks)}")

    # 3) Extract features only for new classes
    with Pool(processes=cpu_count(), initializer=init_worker) as pool:
        rows = [r for r in pool.map(extract_and_label, tasks) if r is not None]

    if not rows:
        print("⚠ No features extracted for new classes.")
        return

    df_new = pd.DataFrame(rows)
    df_new.columns = [f"f{i}" for i in range(STATIC_DIM)] + ["label"]

    # 4) Append to existing CSV
    df_all = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
    df_all.to_csv(FEATURES_PATH, index=False)

    print("Appended features for new labels:", new_labels)
    print("New dataset size:", df_all.shape[0])


if __name__ == "__main__":
    main()
