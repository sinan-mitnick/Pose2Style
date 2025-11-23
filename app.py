import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
import joblib
from statistics import mode
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# --- NEW IMPORTS FOR S3 AND TEMPORARY FILE HANDLING ---
import io
import uuid
import boto3 
# -----------------------------------------------------

DANCE_DESCRIPTIONS = {
    "Kathak": "Kathak is one of the ten major forms of Indian classical dance. Originating from the nomadic bards of ancient northern India, its name is derived from the Sanskrit word 'katha' meaning 'story'. The dance form is characterized by intricate footwork, rhythmic patterns, and expressive storytelling through gestures and facial expressions.",
    "Bharatanatyam": "Bharatanatyam is a major genre of Indian classical dance that originated in the Hindu temples of Tamil Nadu. It is known for its fixed upper torso, bent legs and knees (Aramandi), spectacular footwork, and a sophisticated vocabulary of sign language based on gestures of hands, eyes, and face muscles.",
    "Ballet": "Ballet is a formalized dance form that originated in the Italian Renaissance courts of the 15th century. It has since become a widespread, highly technical concert dance. Ballet is characterized by its graceful, flowing movements, precise and acrobatic techniques, and ethereal qualities.",
    "Salsa": "Salsa is a lively Latin social dance with Afro-Cuban roots. It features quick weight transfers, hip motion, partner connection, and turn patterns danced to syncopated rhythms.",
    "Garba": "Garba is a folk dance from Gujarat, India, traditionally performed in a circle around a lamp or image of the goddess. It uses claps, spins, and coordinated footwork with rhythmic hand movements.",
    "Bhangra": "Bhangra is an energetic folk dance from Punjab, known for powerful shoulder movements, high knee lifts, kicks, and rhythmic hops performed to dhol-driven beats.",
    "Hip Hop": "Hip hop dance is a street style rooted in funk and urban culture, featuring grooves, isolations, popping, locking, breaking, and freestyle expression.",
    "Belly Dance": "Belly dance is a Middle Eastern expressive dance style emphasizing fluid torso movements, hip isolations, and rhythmic articulation performed to traditional or fusion music.",
    "Breakdance": "Break Dance (B-boying/B-girling) is an athletic street dance originating from the Bronx, New York. It features four main elements: toprock (upright movements), downrock (footwork on the floor), power moves (acrobatics like headspins and flares), and freezes (stylish poses).",
    "Ghoomar": "Ghoomar is a traditional folk dance of the Bhil tribe from Rajasthan, India, often performed by women in swirling robes. It involves graceful circular movements, deep squats, and rhythmic steps, accelerating as the dance progresses.",
    "Odissi": "Odissi is one of the eight classical dance forms of India, originating from the temples of Odisha. It is characterized by the Tribhangi (three-bend posture) and Chowka (square stance), featuring fluid movements, beautiful sculptural poses, and expressive storytelling (Abhinaya).",
    "Flamenco": "Flamenco is an expressive Spanish art form that combines guitar, singing, handclaps, and percussive footwork with proud, dramatic body posture.",
    "Waltz": "Waltz is a smooth ballroom dance in 3/4 time characterized by rise-and-fall motion and graceful, flowing rotary movements.",
    "Samba": "Samba is a lively Brazilian dance characterized by fast footwork, hip action, and rhythmic bounce, typically performed to upbeat samba music.",
    "Tango": "Tango is a dramatic partner dance originating from Argentina, marked by sharp movements, close embrace, and expressive leg actions."
}

# --- Configuration (MODIFIED FOR S3) ---
MODEL_DIR = "model"
# We define TMP_DIR for local download/processing (Ephemeral FS)
TMP_DIR = "tmp" 

# --- AWS S3 CONFIGURATION (Credentials loaded from Vercel/Render Env Vars) ---
S3_BUCKET = os.environ.get("S3_BUCKET_NAME") 
# IMPORTANT: CHANGE THIS TO YOUR BUCKET'S REGION (e.g., 'us-east-1')
S3_REGION = 'ap-south-1' 

# Flask setup (Removed static_folder='static' for Vercel)
app = Flask(__name__)

# Initialize Boto3 S3 client
s3_client = None
if S3_BUCKET:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=S3_REGION
    )
    print("✅ AWS S3 client initialized.")
else:
    print("⚠️ S3_BUCKET_NAME not set. S3 features disabled. Local file storage will fail.")

# DELETED ALL os.makedirs calls for UPLOAD_DIR, TMP_DIR, PLOTS_DIR, etc.

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
    # Feature dimension must match train_model.py (230)
    expected_dim = int(getattr(scaler, "n_features_in_", None) or scaler.mean_.shape[0])

    # Make labels display in Title Case
    label_encoder.classes_ = np.array([c.title() for c in label_encoder.classes_])

    print("✅ All model artifacts loaded successfully.")
    print("Expected feature dimension:", expected_dim)
except Exception as e:
    print(f"❌ Error loading artifacts: {e}. Please run train_model.py first.")


# --- Image descriptor helper (MATCHES train_model.py, 230 dims total) ---
def extract_image_features_for_descriptor(img_bgr: np.ndarray) -> np.ndarray:
    """
    Uses WEAK image features:
        HSV histogram = 4x4x4 = 64 dims
        Edge histogram = 8 bins = 8 dims
    (Total image features = 72)
    """
    img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    # HSV histogram (coarse)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [4, 4, 4],  # 64 dims
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    # Edge orientations (8 bins)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mask = mag > (0.1 * mag.max() + 1e-6)
    valid = ang[mask]

    if valid.size == 0:
        edge_hist = np.zeros(8, dtype=np.float32)
    else:
        edge_hist, _ = np.histogram(valid, bins=8, range=(0, 180), density=True)
        edge_hist = edge_hist.astype(np.float32)

    return np.concatenate([hist, edge_hist]).astype(np.float32)


# --- Feature extraction for a single frame (MUST MATCH train_model.py layout) ---
def extract_features_from_frame(frame, pose_estimator, hand_estimator, mp_pose_mod, mp_hands_mod):
    """
    Returns (feature_vector or None, annotated_frame).
    Must produce EXACT 230 dims:
        Pose vector:        66
        Angles:             8
        Left hand:          42
        Right hand:         42
        Image histogram:    64
        Edge histogram:     8
        -----------------------
        TOTAL =            230 dims
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_estimator.process(img_rgb)
    hand_results = hand_estimator.process(img_rgb)

    annotated_frame = frame.copy()

    # Draw landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame, pose_results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS
        )
    if hand_results.multi_hand_landmarks:
        for hl in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, hl, mp_hands_mod.HAND_CONNECTIONS
            )

    # MUST have pose
    if not pose_results.pose_landmarks:
        return None, annotated_frame

    # Pose vector ----------
    pose_lms = pose_results.pose_landmarks.landmark
    pose_coords = np.array([(lm.x, lm.y) for lm in pose_lms], dtype=np.float32)
    hip = (pose_coords[23] + pose_coords[24]) / 2
    pose_coords -= hip
    torso = np.linalg.norm(pose_coords[11] - pose_coords[23]) + 1e-6
    pose_coords /= torso
    pose_vec = pose_coords.flatten()  # 66

    # Angles ----------
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosv = np.dot(ba, bc) / denom
        return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

    def pt(i):
        return (pose_coords[i][0], pose_coords[i][1])

    try:
        angs = np.array([
            angle(pt(11), pt(13), pt(15)),
            angle(pt(12), pt(14), pt(16)),
            angle(pt(23), pt(25), pt(27)),
            angle(pt(24), pt(26), pt(28)),
            angle(pt(11), pt(23), pt(25)),
            angle(pt(12), pt(24), pt(26)),
            angle(pt(13), pt(11), pt(23)),
            angle(pt(14), pt(12), pt(24)),
        ], dtype=np.float32)
    except Exception:
        angs = np.zeros(8, dtype=np.float32)

    # Hands ----------
    left_hand = np.zeros(42, dtype=np.float32)
    right_hand = np.zeros(42, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hl, hd in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            coords = np.array([(lm.x, lm.y) for lm in hl.landmark], dtype=np.float32)
            wrist = coords[0]
            coords -= wrist
            size = np.linalg.norm(coords[0] - coords[9]) + 1e-6
            coords /= size
            flat = coords.flatten()
            if hd.classification[0].label == "Left":
                left_hand = flat
            else:
                right_hand = flat

    pose_hand_feats = np.concatenate([pose_vec, angs, left_hand, right_hand])  # 158

    # Image features ----------
    img_feats = extract_image_features_for_descriptor(frame)  # 72

    final = np.concatenate([pose_hand_feats, img_feats]).astype(np.float32)

    if expected_dim is not None and final.shape[0] != expected_dim:
        print("❌ Feature dim mismatch:", final.shape[0], "!=", expected_dim)
        return None, annotated_frame

    return final, annotated_frame


# --- Prediction Logic (S3 integrated) ---
def predict_video(video_path, start_time, end_time, stride=12):
    """
    Predicts dance style from a video file path.
    The file path is now a temporary local file downloaded from S3.
    """
    if not all([clf is not None, label_encoder is not None, scaler is not None, expected_dim]):
        return "Model not loaded.", None, None, None, None, None

    # DELETED LOCAL DIR CLEANUP (shutil.rmtree)

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
    frame_meta = []

    skipped_dim = 0
    detected_frames = 0

    best_frame_data = {
        'raw_frame': None,
        'annotated_frame': None,
        'index': -1,
        'confidence': -1.0
    }

    # Turbo MediaPipe settings
    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4
    ) as hands:
        while True:
            if current_frame_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % stride == 0:
                features, annotated_frame = extract_features_from_frame(
                    frame, pose, hands, mp_pose, mp_hands
                )

                if features is not None and features.shape[0] == expected_dim:
                    scaled = scaler.transform([features])
                    probs = clf.predict_proba(scaled)[0]
                    pred = np.argmax(probs)

                    preds_raw.append(pred)
                    probs_list.append(probs)
                    frame_meta.append((current_frame_idx, probs))
                    detected_frames += 1

                    current_confidence = probs[pred]
                    if current_confidence > best_frame_data['confidence']:
                        best_frame_data['confidence'] = current_confidence
                        best_frame_data['index'] = current_frame_idx
                        best_frame_data['raw_frame'] = frame.copy()
                        best_frame_data['annotated_frame'] = annotated_frame.copy()

                elif features is not None:
                    skipped_dim += 1

            current_frame_idx += 1

    cap.release()

    if not preds_raw:
        print("Prediction failed: No valid frames with matching feature dimension.")
        return "Prediction failed: No pose detected in the trimmed segment.", None, None, None, None, None

    try:
        final_pred_idx = mode(preds_raw)
    except Exception:
        final_pred_idx = preds_raw[-1]

    label = label_encoder.inverse_transform([final_pred_idx])[0]

    best_raw_frame_url = None
    best_annotated_frame_url = None
    plot_name = None # Will hold the S3 URL
    timeline_name = None # Will hold the S3 URL
    pose_overlay_name = None 
    label_slug = label.lower().replace(" ", "-") # Define slug once

    # --- S3 Upload for Best Frame Images ---
    if best_frame_data['index'] != -1 and s3_client:
        best_frame_idx = best_frame_data['index']

        # 1. Save Raw Frame to S3
        raw_key = f"tmp/{label_slug}/{uuid.uuid4()}_frame_raw.jpg"
        _, raw_img_encoded = cv2.imencode('.jpg', best_frame_data['raw_frame'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=raw_key,
            Body=raw_img_encoded.tobytes(),
            ContentType='image/jpeg'
        )
        best_raw_frame_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{raw_key}"


        # 2. Save Annotated Frame to S3
        ann_key = f"tmp/{label_slug}/{uuid.uuid4()}_frame_annotated.jpg"
        _, ann_img_encoded = cv2.imencode('.jpg', best_frame_data['annotated_frame'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=ann_key,
            Body=ann_img_encoded.tobytes(),
            ContentType='image/jpeg'
        )
        best_annotated_frame_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{ann_key}"


    # --- Average probabilities bar chart (S3 Upload) ---
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
    
    # Upload Bar Chart to S3
    if s3_client:
        plot_key = f"plots/{label_slug}/{uuid.uuid4()}_avg_confidence_chart.png"
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=plot_key,
            Body=buf.read(),
            ContentType='image/png'
        )
        plot_name = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{plot_key}"
    else:
        plt.close() # Still close the plot if S3 failed


    # --- Timeline plot (S3 Upload) ---
    if frame_meta:
        times = [idx / fps for idx, _ in frame_meta]
        main_conf = [p[final_pred_idx] * 100 for _, p in frame_meta]

        # second-best class confidence
        second_conf = []
        second_label = None

        for _, p in frame_meta:
            rank = np.argsort(p)[::-1]
            if len(rank) > 1:
                second_idx = rank[1]
                second_conf.append(p[second_idx] * 100)
                if second_label is None:
                    second_label = label_encoder.inverse_transform([second_idx])[0]
            else:
                second_conf.append(0.0)

        plt.figure(figsize=(10, 5))
        plt.plot(times, main_conf, label=f"Predicted: {label}", linewidth=2, color="orange")
        if second_label is not None:
            plt.plot(times, second_conf, label=f"Second best: {second_label}", linewidth=2, color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("Confidence (%)")
        plt.ylim(0, 100)
        plt.title("Prediction Confidence Over Time")
        plt.legend()
        plt.tight_layout()
        
        # Upload Timeline Chart to S3
        if s3_client:
            timeline_key = f"plots/{label_slug}/{uuid.uuid4()}_confidence_timeline.png"
            timeline_buf = io.BytesIO()
            plt.savefig(timeline_buf, format='png')
            timeline_buf.seek(0)
            plt.close()
            
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=timeline_key,
                Body=timeline_buf.read(),
                ContentType='image/png'
            )
            timeline_name = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{timeline_key}"
        else:
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

    # Vercel's Serverless function is constrained by the 60-second timeout.
    # Video processing must happen fast!
    
    try:
        start_time = float(start_str)
        end_time = float(end_str)
    except ValueError:
        return "Invalid trim times provided.", 400

    # Initialize variables for cleanup
    local_temp_path = None
    temp_video_key = None
    video_path_for_opencv = None
    
    if not s3_client:
        return "Error: S3 Client not initialized. Deployment setup incomplete.", 500

    # 1. Upload Video to S3 (Temporary Storage)
    try:
        # Save the file object directly to S3
        temp_video_key = f"uploads/{secure_filename(f.filename)}_{uuid.uuid4()}.mp4"
        f.seek(0) # Reset file pointer for S3 upload
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=temp_video_key,
            Body=f.read(),
            ContentType=f.content_type or 'video/mp4'
        )
    except Exception as e:
        return f"Error uploading video to S3: {e}", 500
    
    # 2. Download from S3 to a local temp file for OpenCV
    # OpenCV's VideoCapture needs a filesystem path, which is volatile on Vercel.
    local_temp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_{secure_filename(f.filename)}")
    
    # MUST re-create TMP_DIR here, as the ephemeral FS clears it.
    try:
        os.makedirs(TMP_DIR, exist_ok=True)
        s3_client.download_file(S3_BUCKET, temp_video_key, local_temp_path)
        video_path_for_opencv = local_temp_path
    except Exception as e:
        return f"Error downloading video from S3 for processing: {e}", 500


    # 3. Run Prediction using the local temporary file
    (
        label,
        plot_name,
        best_raw_frame,
        best_annotated_frame,
        pose_overlay_name,
        timeline_name,
    ) = predict_video(video_path_for_opencv, start_time, end_time)

    # 4. Cleanup: Delete the local temp video file and the S3 object
    try:
        if local_temp_path and os.path.exists(local_temp_path):
            os.remove(local_temp_path)
            print("Local temp video file deleted.")
        
        if s3_client and temp_video_key:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=temp_video_key)
            print("S3 temp video file deleted.")
            
    except Exception as e:
        print(f"Cleanup failed: {e}")

    description = DANCE_DESCRIPTIONS.get(
        str(label),
        "No description available for this dance form."
    )

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


# --- DELETED LOCAL FILE SERVING ROUTES ---
# The /serve_tmp and /serve_plots routes are removed as images/plots are served directly from S3 URLs.

# Vercel does not use this block to run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)