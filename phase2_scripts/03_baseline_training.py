"""
FASE 2 - Script 03: Baseline Training (50 Epochs, CPU-Optimized)
==================================================================
Training YOLO26n sebagai baseline model dengan konfigurasi standard.

ATURAN (sesuai kesepakatan penelitian):
- Tidak ada custom loss / Focal Loss
- Tidak ada Coordinate Attention / CBAM
- Tidak ada oversampling
- Augmentation: default Ultralytics recipe
- Optimizer: default (SGD dengan cosine LR)
- 50 epochs (dikurangi dari 100 karena CPU-only)

Hasil ini menjadi angka baseline yang akan dibandingkan dengan
FASE 3: YOLO26n + Coordinate Attention.
"""

import os
import time
from ultralytics import YOLO

print("=" * 70)
print("  FASE 2 - Baseline Training YOLO26n (50 Epochs, CPU-Optimized)")
print("=" * 70)

DATA_YAML        = "D:/SKRIPSI/tea_yolo/data.yaml"
MODEL_PRETRAINED = "D:/SKRIPSI/yolo26n.pt"
PROJECT_DIR      = "D:/SKRIPSI/phase2_runs"
RUN_NAME         = "yolo26n_baseline_50ep"

print(f"\nKonfigurasi Training:")
print(f"  data      : {DATA_YAML}")
print(f"  model     : {MODEL_PRETRAINED}")
print(f"  output    : {PROJECT_DIR}/{RUN_NAME}")
print(f"  epochs    : 50")
print(f"  imgsz     : 416  (dikurangi dari 640 karena CPU)")
print(f"  batch     : 4    (dikurangi dari 16 karena CPU)")
print(f"  workers   : 0    (Windows multi-process aman)")
print(f"  device    : cpu")
print(f"  optimizer : SGD (default)")
print(f"  augment   : Mosaic ON (default)")
print(f"  patience  : 15   (early stopping)")

print(f"\n[WARNING] Training CPU bisa memakan waktu 10-30 jam.")
print(f"          Jangan matikan komputer selama proses berjalan.")
print(f"          Gunakan laptop terhubung charger.\n")

t0 = time.time()

model = YOLO(MODEL_PRETRAINED)

# ====== KONFIGURASI BASELINE (tidak boleh diubah tanpa alasan penelitian) ======
# Catatan: Semua augmentation dan optimizer dibiarkan DEFAULT Ultralytics.
# Ini termasuk: Mosaic=1.0, RandAugment, RandomErasing=0.4, FlipLR=0.5
# Optimizer: 'auto' (Ultralytics memilih SGD/AdamW berdasarkan model)
# AMP (Automatic Mixed Precision): aktif secara default
results = model.train(
    data        = DATA_YAML,
    epochs      = 50,
    imgsz       = 416,
    batch       = 4,
    workers     = 0,
    device      = "cpu",
    project     = PROJECT_DIR,
    name        = RUN_NAME,
    exist_ok    = False,        # Gagal jika run sudah ada (cegah overwrite)
    
    # Optimizer: biarkan 'auto' sesuai default Ultralytics recipe
    # (tidak di-override agar murni baseline)
    
    # Augmentation: semua DEFAULT (tidak diubah)
    # close_mosaic=10 (default): mosaic dimatikan 10 epoch terakhir
    
    # Training control
    patience    = 15,           # Early stopping setelah 15 epoch tanpa improvement
    save_period = 10,           # Simpan checkpoint setiap 10 epoch
    seed        = 42,           # Reproducibility
    verbose     = True,
    plots       = True,         # Generate training plots (loss curves, PR curve, dll)
)

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"  Baseline training SELESAI!")
print(f"  Total waktu: {elapsed/3600:.2f} jam ({elapsed/60:.1f} menit)")
print(f"  Hasil disimpan di: {PROJECT_DIR}/{RUN_NAME}")
print(f"{'='*70}")

# Print ringkasan metrics
print(f"\nMetrics ringkasan:")
print(f"  mAP50  : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"  Recall  : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
