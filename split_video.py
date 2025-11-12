import os, sys, cv2

# Usage:
# python split_video.py dataset/Bharatanatyam/myclip.mp4 dataset/Bharatanatyam --every_n_frames 5
# This will save frames in the target folder.

def extract_frames(video_path, out_dir, every_n_frames=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open:", video_path); return
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n_frames == 0:
            out = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(out, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python split_video.py <video_path> <out_dir> [--every_n_frames 5]")
        sys.exit(1)
    video_path, out_dir = sys.argv[1], sys.argv[2]
    every = 5
    if len(sys.argv) >= 5 and sys.argv[3] == "--every_n_frames":
        every = int(sys.argv[4])
    extract_frames(video_path, out_dir, every)
