# get_dataset.py
import os, shutil, zipfile, glob, kagglehub

# 8 Indian dance styles (images)
path = kagglehub.dataset_download("akash14/dance-form-dataset")
print("Cache:", path)

os.makedirs("dataset_tmp", exist_ok=True)
for z in glob.glob(os.path.join(path, "*.zip")):
    with zipfile.ZipFile(z) as f:
        f.extractall("dataset_tmp")

MAP = {
    "bharatanatyam":"Bharatanatyam","kathak":"Kathak","kathakali":"Kathakali",
    "kuchipudi":"Kuchipudi","manipuri":"Manipuri","mohiniyattam":"Mohiniyattam",
    "odissi":"Odissi","sattriya":"Sattriya"
}

os.makedirs("dataset", exist_ok=True)
for v in MAP.values():
    os.makedirs(os.path.join("dataset", v), exist_ok=True)

count = 0
for root, _, files in os.walk("dataset_tmp"):
    for f in files:
        if not f.lower().endswith((".jpg",".jpeg",".png",".bmp")): continue
        src = os.path.join(root, f).lower()
        for k, v in MAP.items():
            if k in src:
                # unique filename
                dst_dir = os.path.join("dataset", v)
                name, ext = os.path.splitext(os.path.basename(src))
                out = os.path.join(dst_dir, name + ext)
                i = 1
                while os.path.exists(out):
                    out = os.path.join(dst_dir, f"{name}_{i}{ext}"); i += 1
                shutil.copy2(os.path.join(root, os.path.basename(src)), out)
                count += 1
                break
print("Copied images:", count)
