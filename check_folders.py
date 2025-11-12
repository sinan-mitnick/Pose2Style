import shutil, os

# Correct path (no extra "dataset" level)
src_root = r"C:\dance-recognition\dataset_tmp\train"
dst_root = r"C:\dance-recognition\dataset"

os.makedirs(dst_root, exist_ok=True)

for cls in os.listdir(src_root):
    src = os.path.join(src_root, cls)
    if os.path.isdir(src):
        dst = os.path.join(dst_root, cls.capitalize())
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {cls} → {dst} ({len(os.listdir(src))} images)")

print("\n✅ Dataset prepared under ./dataset/")
