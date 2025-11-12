import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count # <--- NEW IMPORT

# --- Configuration ---
DATA_ROOT = "dataset"
MODEL_DIR = "model"
FEATURES_DIR = "features"
LANDMARK_IMAGES_DIR = "dataset_landmarks"  # New directory for annotated images
FEATURES_PATH = os.path.join(FEATURES_DIR, "all_features.csv")
FORCE_REBUILD = False
 # flip to True after you add new images

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(LANDMARK_IMAGES_DIR, exist_ok=True)

# --- MediaPipe initializations (static images) ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# IMPORTANT: We will manage these within the multiprocessing pool for safety/efficiency.
# The global 'pose' and 'hands' objects are REMOVED here.

EXPECTED_DIM = 158 # 66 pose (x,y) + 8 angles + 42 LH + 42 RH

# Global variables for the worker process to hold MediaPipe instances
_worker_pose = None
_worker_hands = None

def init_worker():
    """Initializes MediaPipe objects once per worker process."""
    global _worker_pose, _worker_hands
    _worker_pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    _worker_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# --- Feature Extraction (Refactored to use worker globals) ---
def extract_all_features_worker(img_path, output_landmark_path=None):
    global _worker_pose, _worker_hands
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use the worker-local MediaPipe objects
    pose_results = _worker_pose.process(img_rgb)
    hand_results = _worker_hands.process(img_rgb)

    if not pose_results.pose_landmarks:
        return None
    
    # --- Visualization (runs only on the first process that successfully extracts features) ---
    if output_landmark_path:
        annotated_image = img.copy() 
        mp_drawing.draw_landmarks(
            annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Ensure the directory exists before writing
        Path(output_landmark_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_landmark_path, annotated_image)

    pose_lms = pose_results.pose_landmarks.landmark

    # --- Assemble Feature Vector (Normalization logic remains the same) ---
    pose_coords = np.array([(lm.x, lm.y) for lm in pose_lms], dtype=np.float32)
    hip_center = (pose_coords[23] + pose_coords[24]) / 2.0
    pose_coords -= hip_center
    torso_size = np.linalg.norm(pose_coords[11] - pose_coords[23]) + 1e-6
    pose_coords /= torso_size
    pose_vector = pose_coords.flatten()

    def angle_between(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_v = np.dot(ba, bc) / denom
        return np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))

    def pt(i): 
        return (pose_coords[i][0], pose_coords[i][1])

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

    left_hand_vector = np.zeros(21 * 2, dtype=np.float32)
    right_hand_vector = np.zeros(21 * 2, dtype=np.float32)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
            wrist = hand_coords[0]
            hand_coords -= wrist
            palm_size = np.linalg.norm(hand_coords[0] - hand_coords[9]) + 1e-6
            hand_coords /= palm_size
            flat = hand_coords.flatten()
            
            # MediaPipe hands are mirrored for static image mode, so 'Left' is the person's right hand.
            # We assume the original logic handles this correctly:
            if handedness.classification[0].label == "Left":
                # Person's left hand (MediaPipe calls it Left for images)
                left_hand_vector = flat
            else:
                # Person's right hand (MediaPipe calls it Right for images)
                right_hand_vector = flat
                
    feats = np.concatenate([pose_vector, angle_vector, left_hand_vector, right_hand_vector])
    
    if feats.shape[0] != EXPECTED_DIM:
        return None
        
    return feats

# New worker function for the multiprocessing pool
def extract_and_package_features(task_tuple):
    """
    Input: (img_path, output_landmark_path, label)
    Output: np.array of [features, label_string] or None
    """
    img_path, output_landmark_path, label = task_tuple
    try:
        feats = extract_all_features_worker(img_path, output_landmark_path=output_landmark_path)
        if feats is not None:
            return np.append(feats, label)
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def dataset_snapshot(root: Path):
    """Return (classes, files, counts, newest_mtime) for quick change detection."""
    img_exts = {".jpg", ".jpeg", ".png"}
    # Use list comprehension for efficient snapshot
    classes = [d for d in root.iterdir() if d.is_dir()]
    files = []
    counts = {}
    newest = 0
    for c in classes:
        imgs = [p for p in c.rglob("*") if p.suffix.lower() in img_exts]
        files.extend(imgs)
        counts[c.name] = len(imgs)
        for p in imgs:
            try:
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
            except Exception:
                pass
    return classes, files, counts, newest

# --- Refactored build_or_load_features for Parallelism ---
def build_or_load_features(data_root, cache_path, landmark_output_root):
    root = Path(data_root)
    cache = Path(cache_path)
    classes, files, counts, newest_img_mtime = dataset_snapshot(root)
    need_rebuild = FORCE_REBUILD or (not cache.exists())
    
    # --- 1. Cache Check Logic (Unchanged) ---
    if cache.exists() and not FORCE_REBUILD:
        try:
            df = pd.read_csv(cache)
            cached_counts = df["label"].value_counts().to_dict()
            cache_mtime = cache.stat().st_mtime
            counts_match = all(cached_counts.get(k, -1) == v for k, v in counts.items())
            cols_match = (df.shape[1] - 1) == EXPECTED_DIM
            fresh_enough = newest_img_mtime <= cache_mtime
            
            if counts_match and cols_match and fresh_enough:
                print(f"Loading cached features from {cache}")
                X = df.drop("label", axis=1).astype(np.float32).values
                y = df["label"].values
                return X, y
            else:
                need_rebuild = True
                reason = []
                if not counts_match: reason.append("class counts changed")
                if not cols_match: reason.append("feature dim changed")
                if not fresh_enough: reason.append("newer images detected")
                print(f"Rebuilding features because: {', '.join(reason)}")
        except Exception as e:
            print(f"Cache read failed ({e}), rebuilding features.")
            need_rebuild = True

    # --- 2. Feature Extraction (Parallelized) ---
    if need_rebuild:
        print(f"Extracting Pose + Hand features from dataset: {root}")
        if os.path.exists(landmark_output_root):
            shutil.rmtree(landmark_output_root)
        
        # Prepare all tasks for the pool
        tasks = []
        for c in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            label = c.name
            
            for p in c.rglob("*"):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                
                relative_path = os.path.relpath(p, c)
                output_landmark_path = os.path.join(landmark_output_root, label, relative_path)
                
                # Task: (img_path, output_path, label)
                tasks.append((str(p), output_landmark_path, label))

        if not tasks:
            print("No images found in the dataset folder.")
            return np.empty((0, EXPECTED_DIM), dtype=np.float32), np.array([])
        
        # Run in parallel
        num_cores = cpu_count()
        print(f"Starting parallel extraction of {len(tasks)} images using {num_cores} CPU cores...")
        
        rows = []
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            # Use map to apply the worker function to all tasks
            results = pool.map(extract_and_package_features, tasks)
            
            # Filter out None results (skipped images)
            rows = [r for r in results if r is not None]

        total_kept = len(rows)
        total_skipped = len(tasks) - total_kept
        
        print(f"All features extracted (total kept {total_kept}, skipped {total_skipped})")

        if not rows:
            print("No features extracted successfully. Check image files and MediaPipe detection.")
            return np.empty((0, EXPECTED_DIM), dtype=np.float32), np.array([])
            
        # 3. Save Features (Unchanged)
        df = pd.DataFrame(rows)
        num_features = df.shape[1] - 1
        df.columns = [f"f{i}" for i in range(num_features)] + ["label"]
        df.to_csv(cache, index=False)
        print(f"Features cached to {cache}")
        
        X = df.drop("label", axis=1).astype(np.float32).values
        y = df["label"].values
        return X, y
    
    # Should only happen if need_rebuild was False but df was None, which is handled above
    return np.empty((0, EXPECTED_DIM), dtype=np.float32), np.array([])

# --- Plotting Functions (Unchanged) ---
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_classification_report(y_true, y_pred, target_names, save_path):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        df_report.loc[target_names, [metric]].plot(kind='bar', ax=axes[i], rot=45)
        axes[i].set_title(f'{metric.capitalize()} by Class')
        axes[i].set_xlabel('Dance Class')
        axes[i].set_ylabel(metric.capitalize() + ' Score')
        axes[i].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Main Logic (Unchanged) ---
def main():
    X, y = build_or_load_features(DATA_ROOT, FEATURES_PATH, LANDMARK_IMAGES_DIR)
    
    if X is None or len(X) == 0:
        print("No data extracted. Check your dataset folder and image paths.")
        return

    print("\nClass distribution:", Counter(y))
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature matrix shape after scaling: {X_scaled.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    
    print("\nTraining XGBoost model...")
    # Consider adding 'tree_method='gpu_hist'' here if you have a powerful NVIDIA GPU
    clf = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="mlogloss"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # --- New: Plot and save confusion matrix and classification report ---
    plot_confusion_matrix(y_test, y_pred, le.classes_, os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_classification_report(y_test, y_pred, le.classes_, os.path.join(MODEL_DIR, "classification_report_chart.png"))

    joblib.dump(clf, os.path.join(MODEL_DIR, "dance_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\n✅ Saved new model artifacts and plots to {MODEL_DIR}/")

if __name__ == "__main__":
    main()
    # No need to manually close pose/hands here since they are closed implicitly 
    # when the worker processes in the Pool are terminated.