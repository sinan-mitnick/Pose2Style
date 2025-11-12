import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
import joblib
from statistics import mode
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

DANCE_DESCRIPTIONS = {
    "Kathak": "Kathak is one of the ten major forms of Indian classical dance. Originating from the nomadic bards of ancient northern India, its name is derived from the Sanskrit word 'katha' meaning 'story'. The dance form is characterized by intricate footwork, rhythmic patterns, and expressive storytelling through gestures and facial expressions.",
    "Bharatanatyam": "Bharatanatyam is a major genre of Indian classical dance that originated in the Hindu temples of Tamil Nadu. It is known for its fixed upper torso, bent legs and knees (Aramandi), spectacular footwork, and a sophisticated vocabulary of sign language based on gestures of hands, eyes, and face muscles.",
    "Ballet": "Ballet is a formalized dance form that originated in the Italian Renaissance courts of the 15th century. It has since become a widespread, highly technical concert dance. Ballet is characterized by its graceful, flowing movements, precise and acrobatic techniques, and ethereal qualities.",
    "Salsa": "Salsa is a lively Latin social dance with Afro-Cuban roots. It features quick weight transfers, hip motion, partner connection, and turn patterns danced to syncopated rhythms.",
    "Garba": "Garba is a folk dance from Gujarat, India, traditionally performed in a circle around a lamp or image of the goddess. It uses claps, spins, and coordinated footwork with rhythmic hand movements.",
    "Bhangra": "Bhangra is an energetic folk dance from Punjab, known for powerful shoulder movements, high knee lifts, kicks, and rhythmic hops performed to dhol-driven beats.",
    "Hip Hop": "Hip hop dance is a street style rooted in funk and urban culture, featuring grooves, isolations, popping, locking, breaking, and freestyle expression.",
    "Belly Dance": "Belly dance is a Middle Eastern expressive dance style emphasizing fluid torso movements, hip isolations, and rhythmic articulation performed to traditional or fusion music."
}

# --- Configuration ---
MODEL_DIR = "model"
UPLOAD_DIR = "uploads"
TMP_DIR = "tmp"
FRAMES_DIR = os.path.join(TMP_DIR, "frames")
ANNOTATED_FRAMES_DIR = os.path.join(TMP_DIR, "annotated_frames")
PLOTS_DIR = os.path.join(TMP_DIR, "plots")
PLOTS_DIR = os.path.abspath(PLOTS_DIR)
print("PLOTS_DIR =", PLOTS_DIR)

# Flask setup
app = Flask(__name__, static_folder='static')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# --- MediaPipe ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Load Artifacts ---
MODEL_PATH = os.path.join(MODEL_DIR, "dance_model.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

clf, label_encoder, scaler, expected_dim = None, None, None, None
try:
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENC_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_dim = int(getattr(scaler, "n_features_in_", None) or scaler.mean_.shape[0])
    # Make labels title-case for display (assumes training labels were lowercase)
    label_encoder.classes_ = np.array([c.title() for c in label_encoder.classes_])
    print("✅ All model artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}. Please run train_model.py first.")

# --- Feature Extraction ---
def extract_features_from_frame(frame, pose_estimator, hand_estimator, mp_pose_mod, mp_hands_mod):
    """Returns (feature_vector or None, annotated_frame)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_estimator.process(img_rgb)
    hand_results = hand_estimator.process(img_rgb)

    annotated_frame = frame.copy()

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame, pose_results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS
        )

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, hand_landmarks, mp_hands_mod.HAND_CONNECTIONS
            )

    if not pose_results.pose_landmarks:
        return None, annotated_frame

    pose_lms = pose_results.pose_landmarks.landmark
    pose_coords = np.array([(lm.x, lm.y) for lm in pose_lms], dtype=np.float32)

    # Normalize by hip center and torso size
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
        for hand_landmarks, handedness in zip(
            hand_results.multi_hand_landmarks, hand_results.multi_handedness
        ):
            coords = np.array(
                [(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32
            )
            wrist = coords[0]
            coords -= wrist
            palm_size = np.linalg.norm(coords[0] - coords[9]) + 1e-6
            coords /= palm_size
            flat = coords.flatten()
            hand_type = handedness.classification[0].label
            if hand_type == "Left":
                left_hand_vector = flat
            else:
                right_hand_vector = flat

    feats = np.concatenate(
        [pose_vector, angle_vector, left_hand_vector, right_hand_vector]
    )
    return feats, annotated_frame

# --- Prediction Logic ---
def predict_video(video_path, start_time, end_time, stride=5):
    if not all([clf is not None, label_encoder is not None, scaler is not None, expected_dim]):
        return "Model not loaded.", None, None, None, None, None

    # Clean temp dirs
    for d in [FRAMES_DIR, ANNOTATED_FRAMES_DIR, PLOTS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error opening video.", None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_idx = start_frame

    preds_raw = []
    probs_list = []
    frame_meta = []  # list of (frame_idx, probs) for valid frames

    skipped_dim = 0
    detected_frames = 0

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            if current_frame_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % stride == 0:
                # Save raw frame
                raw_frame_path = os.path.join(
                    FRAMES_DIR, f"frame_{current_frame_idx:06d}.jpg"
                )
                cv2.imwrite(raw_frame_path, frame)

                # Extract features + annotated frame
                features, annotated_frame = extract_features_from_frame(
                    frame, pose, hands, mp_pose, mp_hands
                )

                if annotated_frame is not None:
                    annotated_frame_path = os.path.join(
                        ANNOTATED_FRAMES_DIR,
                        f"frame_{current_frame_idx:06d}_annotated.jpg",
                    )
                    cv2.imwrite(annotated_frame_path, annotated_frame)

                if features is not None and features.shape[0] == expected_dim:
                    scaled = scaler.transform([features])
                    probs = clf.predict_proba(scaled)[0]
                    pred = np.argmax(probs)

                    preds_raw.append(pred)
                    probs_list.append(probs)
                    frame_meta.append((current_frame_idx, probs))

                    detected_frames += 1
                elif features is not None:
                    skipped_dim += 1

            current_frame_idx += 1

    cap.release()

    if not preds_raw:
        print("Prediction failed: No valid frames.")
        return "Prediction failed: No pose detected in the trimmed segment.", None, None, None, None, None

    # Final prediction: mode of per-frame predictions
    try:
        final_pred_idx = mode(preds_raw)
    except:
        final_pred_idx = preds_raw[-1]

    label = label_encoder.inverse_transform([final_pred_idx])[0]

    # --- Average probabilities bar chart ---
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

    plot_name = "average_confidence_chart.png"
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    plt.savefig(plot_path)
    plt.close()

    # --- Pick best frame for explanation ---
    best_raw_frame_url = None
    best_annotated_frame_url = None

    if frame_meta:
        # frame where model is most confident for final predicted class
        best_frame_idx, _ = max(
            frame_meta,
            key=lambda item: item[1][final_pred_idx]
        )

        raw_frame_path = os.path.join(
            FRAMES_DIR, f"frame_{best_frame_idx:06d}.jpg"
        )
        ann_frame_path = os.path.join(
            ANNOTATED_FRAMES_DIR,
            f"frame_{best_frame_idx:06d}_annotated.jpg",
        )

        if os.path.exists(raw_frame_path) and os.path.exists(ann_frame_path):
            best_raw_frame_url = raw_frame_path.replace("\\", "/")
            best_annotated_frame_url = ann_frame_path.replace("\\", "/")

    # --- Pose overlay video from annotated frames ---
    pose_overlay_name = None
    annotated_files = sorted(
        glob.glob(os.path.join(ANNOTATED_FRAMES_DIR, "*_annotated.jpg"))
    )
    if annotated_files:
        first = cv2.imread(annotated_files[0])
        if first is not None:
            h, w, _ = first.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            pose_overlay_name = "pose_overlay.mp4"
            pose_overlay_path = os.path.join(PLOTS_DIR, pose_overlay_name)

            # fps/stride so playback speed roughly matches sampled frames
            out = cv2.VideoWriter(
                pose_overlay_path, fourcc, max(1, int(fps / stride) or 1), (w, h)
            )
            for fpath in annotated_files:
                frame = cv2.imread(fpath)
                if frame is not None:
                    out.write(frame)
            out.release()
        else:
            pose_overlay_name = None

    # --- Confidence timeline over the clip ---
    timeline_name = None
    if frame_meta:
        timeline_name = "confidence_timeline.png"
        timeline_path = os.path.join(PLOTS_DIR, timeline_name)

        times = [idx / fps for idx, _ in frame_meta]
        final_conf = [probs[final_pred_idx] * 100 for _, probs in frame_meta]

        # optional: second best class for comparison
        second_conf = None
        second_label = None
        if avg_probs.size > 1:
            sorted_indices = np.argsort(avg_probs)
            if len(sorted_indices) >= 2:
                second_idx = int(sorted_indices[-2])
                second_label = all_classes[second_idx]
                second_conf = [probs[second_idx] * 100 for _, probs in frame_meta]

        plt.figure(figsize=(8, 4))
        plt.plot(times, final_conf, label=str(label), linewidth=2)
        if second_conf is not None:
            plt.plot(times, second_conf, linestyle='--', label=str(second_label), linewidth=1.4)
        plt.xlabel("Time (s)")
        plt.ylabel("Confidence (%)")
        plt.ylim(0, 100)
        plt.title("Prediction confidence over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(timeline_path)
        plt.close()

    print(f"Frames with detections: {detected_frames}, skipped (dim mismatch): {skipped_dim}")
    return (
        label,
        plot_name,
        best_raw_frame_url,
        best_annotated_frame_url,
        pose_overlay_name,
        timeline_name,
    )

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("video")
    start_str = request.form.get("start", "0.0")
    end_str = request.form.get("end", "0.0")

    if not f or f.filename == "":
        return "No file uploaded", 400

    try:
        start_time = float(start_str)
        end_time = float(end_str)
    except ValueError:
        return "Invalid trim times provided.", 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    (
        label,
        plot_name,
        best_raw_frame,
        best_annotated_frame,
        pose_overlay_name,
        timeline_name,
    ) = predict_video(save_path, start_time, end_time)

    description = DANCE_DESCRIPTIONS.get(
        str(label),
        "No description available for this dance form."
    )

    if os.path.exists(save_path):
        os.remove(save_path)

    return render_template(
        "result.html",
        predicted_label=label,
        description=description,
        plot_name=plot_name,
        best_raw_frame=best_raw_frame,
        best_annotated_frame=best_annotated_frame,
        pose_overlay_name=pose_overlay_name,
        timeline_name=timeline_name,
    )

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve files from tmp/* so result.html can access frames, plots, overlays."""
    if filename.startswith(TMP_DIR + '/'):
        dir_name = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        return send_from_directory(dir_name, base_name)
    return "File not found.", 404

@app.route('/plots/<path:filename>')
def serve_plots(filename):
    abs_dir = os.path.abspath(PLOTS_DIR)
    abs_file = os.path.join(abs_dir, filename)
    print(">>> trying to serve", abs_file)
    if not os.path.exists(abs_file):
        print("!!! file missing")
        return "not found", 404
    return send_from_directory(abs_dir, filename, mimetype='video/mp4', as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
