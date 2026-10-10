"""
FASE 2 (RESOLUSI STANDAR 640) - Script 01: Environment & Dataset Check
========================================================================
Memeriksa kesiapan lingkungan (GPU/CUDA, RAM, library) dan integritas dataset
tea_yolo sebelum pelatihan YOLO26n resolusi standar (640x640).
"""

import os
import sys
from pathlib import Path

# Deteksi otomatis Colab vs PC Lokal
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

print("=" * 70)
print("  FASE 2 (RES 640) - Script 01: Pemeriksaan Lingkungan & Dataset")
print("=" * 70)
print(f"Platform: {'Google Colab (Linux)' if IS_COLAB else 'PC Lokal (Windows)'}")
print(f"Base Dir: {BASE_DIR}")

# 1. Cek PyTorch & CUDA
try:
    import torch
    print(f"\n[1] PyTorch Info:")
    print(f"    - Version : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"    - CUDA Ada: {cuda_avail}")
    if cuda_avail:
        print(f"    - Device  : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    - VRAM    : {total_mem:.2f} GB")
    else:
        print("    - [INFO] Menggunakan CPU (Pelatihan resolusi 640 disarankan menggunakan GPU Colab).")
except ImportError:
    print("[ERROR] PyTorch belum terinstall!")
    sys.exit(1)

# 2. Cek Ultralytics
try:
    import ultralytics
    print(f"\n[2] Ultralytics Info:")
    print(f"    - Version : {ultralytics.__version__}")
except ImportError:
    print("[ERROR] Ultralytics belum terinstall! Jalankan: pip install ultralytics")
    sys.exit(1)

# 3. Cek Pretrained Model (yolo26n.pt)
model_candidates = [
    BASE_DIR / "yolo26n.pt",
    Path("yolo26n.pt"),
]
found_model = None
for c in model_candidates:
    if c.exists():
        found_model = c
        break

print(f"\n[3] Pretrained Weight:")
if found_model:
    print(f"    - [OK] Ditemukan: {found_model} ({found_model.stat().st_size / (1024**2):.2f} MB)")
else:
    print("    - [INFO] yolo26n.pt belum ada lokal. Ultralytics akan mendownload otomatis saat train.")

# 4. Cek Dataset tea_yolo
dataset_dir = BASE_DIR / "tea_yolo"
data_yaml = dataset_dir / "data.yaml"

print(f"\n[4] Dataset Status ({dataset_dir}):")
if not dataset_dir.exists():
    print(f"    - [ERROR] Direktori {dataset_dir} tidak ditemukan!")
    sys.exit(1)

if not data_yaml.exists():
    print(f"    - [ERROR] File {data_yaml} tidak ditemukan!")
    sys.exit(1)

print(f"    - [OK] data.yaml ditemukan di {data_yaml}")

for split in ["train", "val", "test"]:
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    n_img = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    print(f"    - Split '{split:<5}': {n_img:4d} citra, {n_lbl:4d} label")

print("\n" + "=" * 70)
print("  [SELESAI] Lingkungan dan dataset siap untuk pelatihan resolusi 640!")
print("=" * 70)
