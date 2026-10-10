"""
FASE 2 - Script 07: Inference & Visualisasi Hasil Test Set (Baseline YOLO26n)
=============================================================================
Menjalankan inferensi model baseline (best.pt) pada gambar test set,
menyimpan hasil gambar ber-bounding box, dan membuat galeri visualisasi.

Dapat dijalankan di:
  1. PC Lokal (Windows): python phase2_scripts/07_infer_and_display_test.py
  2. Google Colab: python phase2_scripts/07_infer_and_display_test.py
"""

import os
import sys
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==================== Deteksi Environment & Path ====================
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    BASE_DIR = Path("/content")
    DATA_YAML = BASE_DIR / "tea_yolo/data.yaml"
    TEST_IMG_DIR = BASE_DIR / "tea_yolo/images/test"
    MODEL_CANDIDATES = [
        BASE_DIR / "phase2_runs/yolo26n_baseline_50ep/weights/best.pt",
        BASE_DIR / "drive/MyDrive/SKRIPSI/phase2_runs/yolo26n_baseline_50ep/weights/best.pt",
        BASE_DIR / "yolo26n.pt",
    ]
    OUTPUT_DIR = BASE_DIR / "phase2_runs/yolo26n_baseline_50ep/test_predictions"
else:
    BASE_DIR = Path("D:/SKRIPSI")
    DATA_YAML = BASE_DIR / "tea_yolo/data.yaml"
    TEST_IMG_DIR = BASE_DIR / "tea_yolo/images/test"
    MODEL_CANDIDATES = [
        BASE_DIR / "phase2_runs/yolo26n_baseline_50ep/weights/best.pt",
        Path("phase2_runs/yolo26n_baseline_50ep/weights/best.pt"),
    ]
    OUTPUT_DIR = BASE_DIR / "phase2_runs/yolo26n_baseline_50ep/test_predictions"

# Cari path bobot model terbaik yang tersedia
MODEL_PATH = None
for candidate in MODEL_CANDIDATES:
    if candidate.exists():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    # Coba cari fallback best.pt
    found = list(BASE_DIR.glob("**/best.pt"))
    if found:
        MODEL_PATH = found[0]
    else:
        print("[ERROR] Model best.pt tidak ditemukan!")
        print(f"Dicari di: {[str(c) for c in MODEL_CANDIDATES]}")
        sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  FASE 2 - VISUALISASI HASIL INFERENSI TEST SET (YOLO26n BASELINE)")
print("=" * 70)
print(f"Model Path    : {MODEL_PATH}")
print(f"Test Images   : {TEST_IMG_DIR}")
print(f"Output Folder : {OUTPUT_DIR}")
print("=" * 70)

# ==================== Load Model ====================
print("\n[1/3] Memuat model YOLO26n Baseline...")
model = YOLO(str(MODEL_PATH))

# ==================== Jalankan Prediksi ====================
print("\n[2/3] Menjalankan inferensi pada gambar test set...")
test_image_files = sorted(
    glob.glob(str(TEST_IMG_DIR / "*.jpg"))
    + glob.glob(str(TEST_IMG_DIR / "*.jpeg"))
    + glob.glob(str(TEST_IMG_DIR / "*.png"))
)

if not test_image_files:
    print(f"[ERROR] Tidak ada gambar ditemukan di {TEST_IMG_DIR}!")
    sys.exit(1)

print(f"Ditemukan {len(test_image_files)} gambar test.")

# Jalankan predict Ultralytics dengan save=True
results = model.predict(
    source=test_image_files,
    conf=0.25,
    iou=0.5,
    imgsz=416,
    save=True,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    verbose=False,
)

# Hitung statistik deteksi per kelas
class_counts = {}
for r in results:
    if r.boxes is not None:
        for c in r.boxes.cls:
            c_id = int(c.item())
            c_name = model.names.get(c_id, f"Class {c_id}")
            class_counts[c_name] = class_counts.get(c_name, 0) + 1

print("\n--- Statistik Objek Terdeteksi di Test Set ---")
for c_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {c_name:<25}: {count} deteksi")
print("----------------------------------------------")

# ==================== Buat Galeri Grid (3x3 = 9 Gambar) ====================
print("\n[3/3] Membuat galeri gambar ringkasan (grid 3x3)...")
saved_pred_images = sorted(
    glob.glob(str(OUTPUT_DIR / "*.jpg"))
    + glob.glob(str(OUTPUT_DIR / "*.jpeg"))
    + glob.glob(str(OUTPUT_DIR / "*.png"))
)

if saved_pred_images:
    sample_to_display = saved_pred_images[:9]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Sample Hasil Deteksi Test Set - YOLO26n Baseline (Conf >= 0.25)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for i, ax in enumerate(axes.flat):
        if i < len(sample_to_display):
            img_path = sample_to_display[i]
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(Path(img_path).name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    gallery_path = OUTPUT_DIR / "gallery_test_predictions.png"
    plt.savefig(str(gallery_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUKSES] Galeri disimpan di: {gallery_path}")

print(f"\n[SELESAI] Seluruh hasil gambar dengan bounding box tersimpan di:")
print(f"          -> {OUTPUT_DIR}")
print("=" * 70)
