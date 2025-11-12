import os, glob, csv
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm

DATASET_DIR = "dataset"
FEATURES_DIR = "features"
os.makedirs(FEATURES_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

# We’ll store for each image: [x1,y1, ..., x33,y33] normalized (66 numbers) + label
def pose_from_image(img_bgr):
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        h, w = img_bgr.shape[:2]
        coords = []
        for lm in res.pose_landmarks.landmark:
            x = lm.x  # normalized [0..1]
            y = lm.y
            coords.extend([x, y])
        return coords  # length 66

def main():
    rows = []
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR,d))])
    print("Classes:", classes)
    for label in classes:
        img_paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            img_paths += glob.glob(os.path.join(DATASET_DIR, label, ext))
        for p in tqdm(img_paths, desc=f"Extract {label}"):
            img = cv2.imread(p)
            if img is None: continue
            feat = pose_from_image(img)
            if feat is not None:
                rows.append(feat + [label])

    out_csv = os.path.join(FEATURES_DIR, "pose_features.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(66)] + ["label"]
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved:", out_csv, "rows:", len(rows))

if __name__ == "__main__":
    main()
