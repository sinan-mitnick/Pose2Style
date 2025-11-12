import os
import random
import pickle
import cv2
import mediapipe as mp
import numpy as np

# --- Load model ---
model = pickle.load(open("model/dance_model.pkl", "rb"))

# --- Class mapping (index → dance style) ---
class_names = [
    "Bharatanatyam",
    "Kathak",
    "Kathakali",
    "Kuchipudi",
    "Manipuri",
    "Mohiniyattam",
    "Odissi",
    "Sattriya"
]

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_pose_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            # include x, y, z
            landmarks.extend([lm.x, lm.y, lm.z])
        # Trim/pad to 74 features
        features = landmarks[:74]
        if len(features) < 74:
            features += [0.0] * (74 - len(features))
        return np.array(features).reshape(1, -1)
    else:
        return None

# --- Dataset path ---
dataset_path = "dataset"

# --- 1. Check dataset balance ---
print("Checking dataset balance...\n")
for dance_class in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, dance_class)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.endswith((".jpg", ".png"))])
        print(f"{dance_class}: {count} images")

# --- 2. Test predictions ---
print("\nTesting predictions...\n")
for dance_class in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, dance_class)
    if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".png"))]
        if len(images) > 0:
            sample_files = random.sample(images, min(2, len(images)))
            for img_file in sample_files:
                img_path = os.path.join(class_path, img_file)
                features = extract_pose_features(img_path)
                if features is not None:
                    pred = model.predict(features)
                    probs = model.predict_proba(features)[0]
                    predicted_label = class_names[int(pred[0])]

                    print(f"True: {dance_class}, Predicted: {predicted_label} ({pred[0]})")
                    for idx, prob in enumerate(probs):
                        print(f"   {class_names[idx]}: {prob:.3f}")
                else:
                    print(f"True: {dance_class}, Pose not detected in {img_file}")
