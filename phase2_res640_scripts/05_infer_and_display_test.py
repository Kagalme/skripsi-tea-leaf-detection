"""
FASE 2 (RESOLUSI STANDAR 640) - Script 05: Inferensi & Tampilan Citra Test Set
================================================================================
Menjalankan inferensi model YOLO26n (best.pt) pada gambar test set dengan
resolusi standar 640x640, menyimpan gambar hasil deteksi (bounding box + label +
confidence), serta menyusun galeri visualisasi komposit (grid 3x3).
"""

import os
import sys
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Deteksi platform
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

PROJECT_DIR = BASE_DIR / "phase2_runs"
RUN_NAME    = "yolo26n_baseline_640"
BEST_MODEL  = PROJECT_DIR / RUN_NAME / "weights/best.pt"
TEST_IMG_DIR= BASE_DIR / "tea_yolo/images/test"
OUTPUT_DIR  = PROJECT_DIR / RUN_NAME / "test_predictions"

print("=" * 70)
print("  FASE 2 - INFERENSI & GALERI TEST SET (RESOLUSI STANDAR 640x640)")
print("=" * 70)
print(f"Model Path    : {BEST_MODEL}")
print(f"Test Images   : {TEST_IMG_DIR}")
print(f"Output Folder : {OUTPUT_DIR}")

if not BEST_MODEL.exists():
    print(f"\n[ERROR] File model {BEST_MODEL} tidak ditemukan!")
    print("Pastikan pelatihan script 02_train_baseline_640.py sudah selesai.")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Model
print("\n[1/3] Memuat model YOLO26n Baseline...")
model = YOLO(str(BEST_MODEL))

# 2. Ambil daftar citra
test_image_files = sorted(
    glob.glob(str(TEST_IMG_DIR / "*.jpg"))
    + glob.glob(str(TEST_IMG_DIR / "*.jpeg"))
    + glob.glob(str(TEST_IMG_DIR / "*.png"))
)

if not test_image_files:
    print(f"[ERROR] Tidak ada berkas gambar ditemukan di {TEST_IMG_DIR}!")
    sys.exit(1)

print(f"Ditemukan {len(test_image_files)} citra test set.")

# 3. Jalankan Inferensi (imgsz=640)
print("\n[2/3] Menjalankan inferensi pada seluruh citra test (imgsz=640) ...")
results = model.predict(
    source      = test_image_files,
    conf        = 0.25,
    iou         = 0.5,
    imgsz       = 640,              # Resolusi standar
    save        = True,             # Simpan gambar ber-bounding box
    project     = str(OUTPUT_DIR.parent),
    name        = OUTPUT_DIR.name,
    exist_ok    = True,
    verbose     = False,
)

# Hitung statistik deteksi
class_counts = {}
for r in results:
    if r.boxes is not None:
        for c in r.boxes.cls:
            cid = int(c.item())
            cname = model.names.get(cid, f"Class {cid}")
            class_counts[cname] = class_counts.get(cname, 0) + 1

print("\n--- Statistik Deteksi Objek pada Test Set ---")
for cname, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  • {cname:<25}: {cnt:4d} deteksi")
print("---------------------------------------------")

# 4. Buat Galeri Grid 3x3 Komposit
print("\n[3/3] Menyusun galeri ringkasan 9 gambar (grid 3x3) ...")
saved_pred_images = sorted(
    glob.glob(str(OUTPUT_DIR / "*.jpg"))
    + glob.glob(str(OUTPUT_DIR / "*.jpeg"))
    + glob.glob(str(OUTPUT_DIR / "*.png"))
)

if saved_pred_images:
    selected = saved_pred_images[:9]
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle(
        "Hasil Deteksi Daun Teh - YOLO26n Baseline (Resolusi 640x640, Conf >= 0.25)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            fpath = selected[i]
            img = cv2.imread(fpath)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(Path(fpath).name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    gallery_path = OUTPUT_DIR / "gallery_test_predictions.png"
    plt.savefig(str(gallery_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUKSES] Galeri visual komposit tersimpan di:")
    print(f"         {gallery_path}")

print(f"\n[SELESAI] Seluruh file citra ber-bounding box tersimpan di:")
print(f"          {OUTPUT_DIR}")
print("=" * 70)
