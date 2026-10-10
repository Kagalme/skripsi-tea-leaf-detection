"""
FASE 3 - Script 01: Training YOLO26n + Coordinate Attention (CA)
==================================================================
Melatih YOLO26n yang telah diinjeksi modul Coordinate Attention (CA)
di 3 titik backbone (P3, P4, P5) menggunakan:
  - Pretrained weights: yolo26n.pt (COCO) — di-load ke backbone yang sama,
    kecuali 3 layer CA baru (diinisialisasi acak dengan seed=42).
  - Hyperparameter: IDENTIK dengan baseline (50 epoch, imgsz=640, seed=42).
  - Tidak ada perubahan lain (augmentation, optimizer, loss = default).
  - Device: GPU CUDA (Google Colab T4).

Prinsip fair comparison:
  Satu-satunya perbedaan antara Baseline dan CA adalah penambahan 3 CA blocks.
  Semua hyperparameter lain di-lock sama persis agar perbandingan valid.

Cara menjalankan:
  python phase3_scripts/01_train_ca.py

  Atau dari Google Colab (sudah ada di notebook Training_Phase3_CA_Colab.ipynb):
  !python phase3_scripts/01_train_ca.py --device cuda
"""

import os
import sys
import time
import argparse

# ─────────────────────────────────────── 0. PATH SETUP ─────────────────────
# Tambahkan root project ke sys.path agar ca_module bisa diimport
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# ─────────────────────────────────────── 1. REGISTRASI CA ──────────────────
# HARUS dilakukan SEBELUM 'from ultralytics import YOLO'
from phase3_scripts.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO   # noqa: E402 — import setelah registrasi

# ─────────────────────────────────────── 2. ARGPARSE ───────────────────────
parser = argparse.ArgumentParser(description="FASE 3: Training YOLO26n + CA")
parser.add_argument("--device",    default="cuda",  help="cuda | cpu | 0,1 (GPU id)")
parser.add_argument("--epochs",    type=int, default=50)
parser.add_argument("--imgsz",     type=int, default=640)
parser.add_argument("--batch",     type=int, default=16)
parser.add_argument("--workers",   type=int, default=8,  help="DataLoader workers; pakai 0 di Windows")
parser.add_argument("--run_name",  default="yolo26n_ca_50ep", help="Nama direktori output")
args = parser.parse_args()

# ─────────────────────────────────────── 3. PATH KONFIGURASI ───────────────
# Auto-detect path untuk Colab vs Lokal Windows
if os.path.exists("/content/tea_yolo"):
    DATA_YAML    = "/content/tea_yolo/data.yaml"
    PRETRAINED   = "/content/yolo26n.pt"
    MODEL_YAML   = "/content/phase3_scripts/yolo26n_ca.yaml"
    RUNS_DIR     = "/content/phase3_runs"
else:
    DATA_YAML    = os.path.join(PROJECT_DIR, "tea_yolo", "data.yaml")
    PRETRAINED   = os.path.join(PROJECT_DIR, "yolo26n.pt")
    MODEL_YAML   = os.path.join(SCRIPT_DIR, "yolo26n_ca.yaml")
    RUNS_DIR     = os.path.join(PROJECT_DIR, "phase3_runs")

os.makedirs(RUNS_DIR, exist_ok=True)

# ─────────────────────────────────────── 4. PRINT HEADER ───────────────────
print("=" * 70)
print("  FASE 3: Training YOLO26n + Coordinate Attention (CA)")
print("  Hou et al., CVPR 2021 — https://arxiv.org/abs/2103.02907")
print("=" * 70)
print(f"\n  Model YAML   : {MODEL_YAML}")
print(f"  Pretrained   : {PRETRAINED}")
print(f"  Dataset      : {DATA_YAML}")
print(f"  Output Dir   : {RUNS_DIR}/{args.run_name}")
print(f"\n  === HYPERPARAMETER (IDENTIK BASELINE) ===")
print(f"  Epochs       : {args.epochs}")
print(f"  Image size   : {args.imgsz}x{args.imgsz}")
print(f"  Batch size   : {args.batch}")
print(f"  Device       : {args.device}")
print(f"  Seed         : 42  (reproducible)")
print(f"  Optimizer    : auto (SGD/AdamW — default Ultralytics)")
print(f"  Augmentation : Default recipe (Mosaic, RandAugment, FlipLR)")
print(f"  close_mosaic : 10  (default — mosaic OFF 10 epoch terakhir)")
print(f"\n  === MODIFIKASI (SATU-SATUNYA PERBEDAAN DARI BASELINE) ===")
print(f"  CA blocks    : 3x CoordAttYOLO di backbone P3, P4, P5")
print(f"  Reduction    : 32 (bottleneck ratio)")
print(f"  Extra params : ~3 x 2x(c//32 * c) ≈ +0.1% parameter overhead")
print(f"\n{'=' * 70}\n")

# ─────────────────────────────────────── 5. BUILD MODEL ────────────────────
print("[1/4] Membangun model YOLO26n + CA dari YAML...")
model = YOLO(MODEL_YAML)

# ─────────────────────────────────────── 6. LOAD PRETRAINED WEIGHTS ────────
print("[2/4] Memuat pretrained weights yolo26n.pt (COCO)...")
print("      Layer CA baru (CoordAttYOLO) diinisialisasi random (seed=42).")
# Ultralytics .train() dengan pretrained=PRETRAINED akan secara otomatis
# memuat semua layer yang namanya cocok dan melewati layer yang tidak ada.

# ─────────────────────────────────────── 7. TRAINING ───────────────────────
print("[3/4] Memulai training...\n")
t0 = time.time()

results = model.train(
    data           = DATA_YAML,
    pretrained     = PRETRAINED,   # Muat backbone weights dari yolo26n.pt
    epochs         = args.epochs,
    imgsz          = args.imgsz,
    batch          = args.batch,
    workers        = args.workers,
    device         = args.device,
    project        = RUNS_DIR,
    name           = args.run_name,
    exist_ok       = False,         # Gagal jika run sudah ada (cegah overwrite)

    # ── HYPERPARAMETER IDENTIK BASELINE ──────────────────────────────────
    patience       = 15,            # Early stopping 15 epoch tanpa improvement
    save_period    = 10,            # Checkpoint setiap 10 epoch
    seed           = 42,            # Reproducible
    verbose        = True,
    plots          = True,

    # ── Catatan: semua augmentation & optimizer dibiarkan DEFAULT ─────────
    # Ini menjamin fair comparison: satu-satunya variabel = modul CA
)

elapsed = time.time() - t0

# ─────────────────────────────────────── 8. RINGKASAN ──────────────────────
print(f"\n{'=' * 70}")
print(f"  [4/4] Training SELESAI!")
print(f"  Total waktu  : {elapsed/3600:.2f} jam ({elapsed/60:.1f} menit)")
print(f"  Output dir   : {RUNS_DIR}/{args.run_name}")
print(f"  Best weights : {RUNS_DIR}/{args.run_name}/weights/best.pt")
print(f"{'=' * 70}")

print(f"\n  Metrics akhir training (val set):")
rd = results.results_dict
print(f"    mAP@0.5        : {rd.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"    mAP@0.5:0.95   : {rd.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"    Precision (val): {rd.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"    Recall    (val): {rd.get('metrics/recall(B)', 'N/A'):.4f}")

print(f"\n  >> Lanjut ke: python phase3_scripts/02_evaluate_testset.py")
