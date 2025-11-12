import os, shutil

# ---- Configure your class names and tokens that might appear in filenames/paths
CLASS_TOKENS = {
    "Bharatanatyam": ["bharatanatyam", "bharatnatyam", "bharat-anatyam"],
    "Kathak":        ["kathak"],
    "Kathakali":     ["kathakali"],
    "Kuchipudi":     ["kuchipudi"],
    "Manipuri":      ["manipuri"],
    "Mohiniyattam":  ["mohiniyattam", "mohiniattam"],
    "Odissi":        ["odissi", "orissi"],
    "Sattriya":      ["sattriya"],
    "HipHop":        ["hiphop", "hip-hop", "hip_hop"],
    "Ballet":        ["ballet"]
}

# ---- Where to look for source images (add folders if you have more)
SOURCE_ROOTS = [
    ".",  # search the whole project tree (includes human-activity-recognition-dancing*)
]

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def classify_path(path_lower: str):
    for cls, tokens in CLASS_TOKENS.items():
        for t in tokens:
            if t in path_lower:
                return cls
    return None

def main():
    dst_root = "dataset"
    os.makedirs(dst_root, exist_ok=True)
    for cls in CLASS_TOKENS.keys():
        os.makedirs(os.path.join(dst_root, cls), exist_ok=True)

    copied = 0
    skipped = 0
    seen = set()

    for root in SOURCE_ROOTS:
        for dirpath, _, filenames in os.walk(root):
            # ignore target dataset itself to avoid re-copying
            if os.path.abspath(dirpath).startswith(os.path.abspath(dst_root)):
                continue
            for f in filenames:
                if not f.lower().endswith(IMG_EXT):
                    continue
                src = os.path.join(dirpath, f)
                # avoid duplicates
                key = os.path.abspath(src)
                if key in seen:
                    continue
                seen.add(key)

                cls = classify_path(src.lower())
                if not cls:
                    skipped += 1
                    continue

                # ensure unique output filename
                base, ext = os.path.splitext(os.path.basename(src))
                out_dir = os.path.join(dst_root, cls)
                out = os.path.join(out_dir, base + ext.lower())
                i = 1
                while os.path.exists(out):
                    out = os.path.join(out_dir, f"{base}_{i}{ext.lower()}")
                    i += 1
                try:
                    shutil.copy2(src, out)
                    copied += 1
                    if copied % 50 == 0:
                        print(f"[{copied}] {cls} ← {src}")
                except Exception as e:
                    print("skip copy:", src, e)

    print(f"\nDone. Copied: {copied}, Skipped (no class match): {skipped}")
    print("Classes populated under ./dataset")

if __name__ == "__main__":
    main()
