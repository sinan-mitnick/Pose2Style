import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
import joblib
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statistics import mode

SAFE_DIRS = {"dataset", "dataset_landmarks", "features"}
def safe_delete(path):
    """Prevent deleting important folders."""
    base = os.path.basename(path.rstrip("/"))
    if base in SAFE_DIRS:
        print(f"⚠ BLOCKED DELETE: {base} is protected.")
        return
    # Use glob to allow safe_delete on specific files inside safe directories if needed, but for rmtree, stick to the safe check
    shutil.rmtree(path, ignore_errors=True)

# ---------- DANCE DESCRIPTIONS ----------
DANCE_DESCRIPTIONS = {
    "Kathak": "Kathak is one of the ten major forms of Indian classical dance. Originating from the nomadic bards of ancient northern India, its name is derived from the Sanskrit word 'katha' meaning 'story'. The dance form is characterized by intricate footwork, rhythmic patterns, and expressive storytelling through gestures and facial expressions.",
    "Bharatanatyam": "Bharatanatyam is a major South Indian dance form that originated in the Hindu temples of Tamil Nadu. It is known for its fixed upper torso, bent legs and knees (Aramandi), spectacular footwork, and a sophisticated vocabulary of sign language based on gestures of hands, eyes, and face muscles.",
    "Ballet": "Ballet is a classical Western dance technique that originated in the Italian Renaissance courts of the 15th century. It has since become a widespread, highly technical concert dance. Ballet is characterized by its graceful, flowing movements, precise and acrobatic techniques, and ethereal qualities.",
    "Salsa": "Salsa is a lively Latin social dance with Afro-Cuban roots. It features quick weight transfers, hip motion, partner connection, and turn patterns danced to syncopated rhythms.",
    "Garba": "Garba is a Gujarati folk dance performed in circles, traditionally around a lamp or image of the goddess. It uses claps, spins, and coordinated footwork with rhythmic hand movements.",
    "Bhangra": "Energetic Punjabi folk dance with shoulder and arm movements, high knee lifts, kicks, and rhythmic hops performed to dhol-driven beats.",
    "Hip Hop": "Hip Hop dance includes popping, locking, breakdance and grooves, rooted in funk and urban culture, featuring grooves, isolations, and freestyle expression.",
    "Belly Dance": "Middle Eastern torso-focused expressive dance style emphasizing fluid torso movements, hip isolations, and rhythmic articulation performed to traditional or fusion music.",
    "Breakdance": "Breakdance (B-boying/B-girling) features toprock, footwork, power moves and freezes, originating from the Bronx, New York.",
    "Ghoomar": "Traditional Rajasthani folk dance with circular spins, deep squats, and rhythmic steps, often performed by women in swirling robes.",
    "Odissi": "Classical Indian dance from Odisha with Tribhangi posture and Chowka stance, featuring fluid movements, beautiful sculptural poses, and expressive storytelling.",
    "Tango": "Tango is a partner dance that originated in the 1880s along the border of Argentina and Uruguay. It is characterized by a close embrace, staccato movements, intricate leg play (ganchos), and intense emotional connection between partners.",
    "Flamenco": "Flamenco is a highly expressive Spanish art form from Andalusia. It involves percussive footwork (zapateado), hand clapping (palmas), and fluid, dramatic arm movements, often performed with intense emotion.",
    "Samba": "Samba is a lively, rhythmic dance of Brazilian origin with African roots. It is recognized by its rapid footwork, characteristic 'bouncing' action created by bending and straightening the knees, and energetic hip movements.",
    "Waltz": "Waltz is a smooth, progressive ballroom dance performed in 3/4 time. Originating in Europe, it is distinguished by its graceful 'rise and fall' action, sweeping turns, and elegant closed-partner posture."
}


MODEL_DIR = "model"
UPLOAD_DIR = "uploads"
TMP_DIR = "tmp"
FRAMES_DIR = os.path.join(TMP_DIR, "frames")
ANNOTATED_FRAMES_DIR = os.path.join(TMP_DIR, "annotated_frames")
PLOTS_DIR = os.path.join(TMP_DIR, "plots")

app = Flask(__name__, static_folder="static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


MODEL_PATH = os.path.join(MODEL_DIR, "dance_model.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

clf = label_encoder = scaler = None
expected_dim = None

try:
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENC_PATH)
    scaler = joblib.load(SCALER_PATH)

  
    label_encoder.classes_ = np.array([c.title() for c in label_encoder.classes_])

    # Safely get expected_dim, assuming Code 1's calculation leads to 234 if the docstring is wrong (33*3 + 8 + 63 + 63 = 234)
    expected_dim = int(getattr(scaler, "n_features_in_", None) or scaler.mean_.shape[0])

    print("✅ Model loaded. Feature dim =", expected_dim)

except Exception as e:
    print("Could not load model. Train again.", e)



mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_frame(frame, pose_estimator, hand_estimator):
    """Extract 234-D feature vector (assuming 33*3 + 8 + 63 + 63)."""

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_estimator.process(img_rgb)
    hand_results = hand_estimator.process(img_rgb)

    annotated = frame.copy()
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated, pose_results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS)

    if hand_results.multi_hand_landmarks:
        for h in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, h, mp_hands.HAND_CONNECTIONS)

    if not pose_results.pose_landmarks:
        return None, annotated

    # Pose 33×(x,y,visibility)
    pose_lms = pose_results.pose_landmarks.landmark
    pose_arr = np.array([[lm.x, lm.y, lm.visibility] for lm in pose_lms], np.float32)

    coords = pose_arr[:, :2]
    hip = (coords[23] + coords[24]) / 2
    coords -= hip
    torso = np.linalg.norm(coords[11] - coords[23]) + 1e-6
    coords /= torso
    pose_arr[:, :2] = coords

    pose_vec = pose_arr.flatten()

    # Angles (8)
    def ang(a, b, c):
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1)))

    def pt(i): return coords[i]

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
        ], np.float32)
    except:
        angles = np.zeros(8, np.float32)

    # Hands (21×3 each for x, y, presence)
    LH = np.zeros(63, np.float32)
    RH = np.zeros(63, np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for h, handed in zip(hand_results.multi_hand_landmarks,
                             hand_results.multi_handedness):

            coords_h = np.array([[lm.x, lm.y] for lm in h.landmark], np.float32)
            coords_h -= coords_h[0]
            palm = np.linalg.norm(coords_h[0] - coords_h[9]) + 1e-6
            coords_h /= palm
            presence = np.ones((21, 1), np.float32)
            full = np.concatenate([coords_h, presence], 1).flatten()

            if handed.classification[0].label == "Left":
                LH = full
            else:
                RH = full

    feat = np.concatenate([pose_vec, angles, LH, RH])
    return feat, annotated


# ---------- VIDEO PREDICTION (Updated Stride for Speed) ----------
def predict_video(video_path, start_time, end_time, stride=20): # Increased stride from 6 to 20
    if not all([clf is not None, label_encoder is not None, scaler is not None, expected_dim]):
        return "Model not loaded.", None, None, None, None, None

    # CLEAN TMP SAFE
    for d in [FRAMES_DIR, ANNOTATED_FRAMES_DIR, PLOTS_DIR]:
        safe_delete(d)
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Video error", None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_f = int(start_time * fps)
    end_f = int(end_time * fps) if end_time > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    preds = []
    probs_list = []
    frame_indices = [] # Added to track frame indices for the timeline graph
    best = {"idx": -1, "conf": -1, "raw": None, "ann": None}

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        idx = start_f
        while idx < end_f:
            ret, f = cap.read()
            if not ret:
                break

            if idx % stride == 0:
                feat, ann = extract_features_from_frame(f, pose, hands)

                # Check for dim mismatch with feature vector (should be 234 if the model is correct)
                if feat is not None and feat.shape[0] == expected_dim: 
                    feat_scaled = scaler.transform([feat])
                    prob = clf.predict_proba(feat_scaled)[0]
                    pred = int(np.argmax(prob))

                    preds.append(pred)
                    probs_list.append(prob)
                    frame_indices.append(idx) # Save the frame index

                    if prob[pred] > best["conf"]:
                        best["conf"] = prob[pred]
                        best["idx"] = idx
                        best["raw"] = f.copy()
                        best["ann"] = ann.copy()

            idx += 1

    cap.release()

    if not preds:
        return "No pose", None, None, None, None, None

    # Final prediction: mode of per-frame predictions
    try:
        final_idx = mode(preds)
    except:
        final_idx = preds[-1]

    label = label_encoder.inverse_transform([final_idx])[0]

    # Save best frames
    raw_path = os.path.join(FRAMES_DIR, "best.jpg")
    ann_path = os.path.join(ANNOTATED_FRAMES_DIR, "best_ann.jpg")
    cv2.imwrite(raw_path, best["raw"])
    cv2.imwrite(ann_path, best["ann"])

    # ---------- CONFIDENCE CHART (Exact Code 2 Logic) ----------
    avg_probs = np.mean(probs_list, axis=0) * 100
    all_classes = label_encoder.classes_

    plot_df = pd.DataFrame({
        'Dance Class': all_classes,
        'Average Probability (%)': avg_probs
    })

    plt.figure(figsize=(10, 7))
    sns.barplot(
        x='Dance Class',
        y='Average Probability (%)',
        data=plot_df,
        palette='Spectral',
        edgecolor='black'
    )
    plt.title(f'Average Model Confidence per Class (Final Prediction: {label})', fontsize=16)
    plt.ylabel('Average Prediction Probability (%)', fontsize=12)
    plt.xlabel('Dance Class', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')

    for index, row in plot_df.iterrows():
        plt.text(
            index,
            row['Average Probability (%)'] + 1,
            f"{row['Average Probability (%)']:.1f}%",
            color='black',
            ha="center"
        )

    plt.tight_layout()

    chart_name = "conf.png"
    chart_path = os.path.join(PLOTS_DIR, chart_name)
    plt.savefig(chart_path)
    plt.close()

    # ---------- TIMELINE GRAPH (Updated X-axis to Frame Index) ----------
    probs_list = np.array(probs_list)
    timeline_path = os.path.join(PLOTS_DIR, "timeline.png")
    timeline_name = "timeline.png"

    final_conf = probs_list[:, final_idx] * 100

    # Optional: Second best class for comparison
    second_conf = None
    second_label = None
    if avg_probs.size > 1:
        sorted_indices = np.argsort(avg_probs)
        if len(sorted_indices) >= 2:
            second_idx = int(sorted_indices[-2])
            second_label = all_classes[second_idx]
            second_conf = probs_list[:, second_idx] * 100

    plt.figure(figsize=(12, 5))
    
    # Plot final prediction confidence
    plt.plot(frame_indices, final_conf, label=str(label), linewidth=2) 
    
    # Plot second best confidence if available
    if second_conf is not None:
        plt.plot(frame_indices, second_conf, linestyle='--', label=str(second_label), linewidth=1.4) 

    plt.title("Prediction confidence over Frame Index")
    plt.xlabel(f"Frame Index (Frames sampled every {stride})")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 100)
    plt.legend(loc="upper right", ncol=1, fontsize=8)
    plt.tight_layout()
    plt.savefig(timeline_path)
    plt.close()

    return label, chart_name, raw_path, ann_path, timeline_name, None


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

#index.html sends the uploaded video to the backend through https post request
@app.route("/predict", methods=["POST"])
def predict():
    # flask receives the video with the below 3 line code     
    f = request.files.get("video")
    start = float(request.form.get("start", 0))
    end = float(request.form.get("end", 0))

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)

    label, chart, raw, ann, timeline_name, _ = predict_video(path, start, end)

    desc = DANCE_DESCRIPTIONS.get(label, "No description available for this dance form.")

    os.remove(path)
        # backend sends the result to result.html 
    return render_template(
        "result.html",
        predicted_label=label,
        description=desc,
        plot_name=chart,
        best_raw_frame=raw,
        best_annotated_frame=ann,
        timeline_name=timeline_name,
        pose_overlay_name=None 
    )


@app.route("/<path:filename>")
def serve_tmp(filename):
    """Serve files from tmp/* so result.html can access frames, plots, overlays."""
    
    # Using a more robust serving for TMP_DIR content:
    full_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(full_path):
        return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
    
    # The original fallback route:
    base = os.path.dirname(filename)
    name = os.path.basename(filename)
    return send_from_directory(base if base else ".", name) 
    
if __name__ == "__main__":
    print("🚀 Flask running at: http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)