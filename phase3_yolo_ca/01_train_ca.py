"""
FASE 3 (DIPERBARUI) - Script 01: Training YOLO26n + Coordinate Attention (CA)
=============================================================================
Melatih arsitektur ASLI YOLO26n (C3k2 + SPPF + C2PSA + Detect end2end=True, reg_max=1)
yang telah disisipi 3 blok Coordinate Attention (CA) di P3, P4, dan P5.

Perbaikan Kritis yang Diterapkan:
1. Menggunakan `build_yolo26n_ca_pretrained()` untuk memetakan (remapping) 100%
   bobot backbone & neck dari `yolo26n.pt` (24 layer) ke `yolo26n_ca.yaml` (27 layer).
   Tanpa langkah ini, pergeseran indeks layer (+1, +2, +3) membuat bobot layer 5..23
   gagal ditransfer.
2. Hyperparameter dikunci pada resolusi 640x640, 50 epoch, seed=42 agar perbandingan
   terhadap YOLO26n Baseline 640x640 bersifat 100% apple-to-apple.
"""

import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Registrasi modul CA sebelum import YOLO
from phase3_yolo_ca.ca_module import register_ca_module, build_yolo26n_ca_pretrained
register_ca_module()

from ultralytics import YOLO  # noqa: E402

parser = argparse.ArgumentParser(description="FASE 3 (v2): Training YOLO26n + CA")
parser.add_argument("--device", default="cuda", help="cuda | cpu | 0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--run_name", default="yolo26n_ca_50ep")
args = parser.parse_args()

# Deteksi path Google Colab vs Lokal
if os.path.exists("/content/tea_yolo"):
    DATA_YAML = "/content/tea_yolo/data.yaml"
    BASE_PT = "/content/yolo26n.pt" if os.path.exists("/content/yolo26n.pt") else os.path.join(PROJECT_DIR, "yolo26n.pt")
    MODEL_YAML = os.path.join(SCRIPT_DIR, "yolo26n_ca.yaml")
    RUNS_DIR = "/content/phase3_runs"

    try:
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            y_text = f.read()
        if "D:/SKRIPSI/tea_yolo" in y_text or "D:\\" in y_text:
            y_text = (
                y_text.replace("D:/SKRIPSI/tea_yolo", "/content/tea_yolo")
                .replace("D:\\SKRIPSI\\tea_yolo", "/content/tea_yolo")
            )
            with open(DATA_YAML, "w", encoding="utf-8") as f:
                f.write(y_text)
            print("[INFO] Path di data.yaml disesuaikan ke /content/tea_yolo")
    except Exception as e:
        print(f"[WARNING] Gagal menyesuaikan path data.yaml: {e}")
else:
    DATA_YAML = os.path.join(PROJECT_DIR, "tea_yolo", "data.yaml")
    BASE_PT = os.path.join(PROJECT_DIR, "yolo26n.pt")
    MODEL_YAML = os.path.join(SCRIPT_DIR, "yolo26n_ca.yaml")
    RUNS_DIR = os.path.join(PROJECT_DIR, "phase3_runs")

os.makedirs(RUNS_DIR, exist_ok=True)
REMAPPED_PT = os.path.join(RUNS_DIR, "yolo26n_ca_coco_init.pt")

print("=" * 72)
print("  FASE 3 (v2): Training YOLO26n + Coordinate Attention (CA)")
print("  Arsitektur : YOLO26n Asli (C3k2 + SPPF + C2PSA + end2end=True, reg_max=1)")
print("  Modifikasi : +3x CoordAttYOLO pada P3 (128ch), P4 (128ch), P5 (256ch)")
print("=" * 72)
print(f"  Model YAML   : {MODEL_YAML}")
print(f"  Base Weights : {BASE_PT}")
print(f"  Remapped Pt  : {REMAPPED_PT}")
print(f"  Dataset YAML : {DATA_YAML}")
print(f"  Output Dir   : {RUNS_DIR}/{args.run_name}")
print(f"  Resolusi     : {args.imgsz}x{args.imgsz} | Epochs: {args.epochs} | Batch: {args.batch}")
print("=" * 72)

# 1. Buat checkpoint pretrained hasil remapping layer 0..23 -> 0..26
print("\n[1/4] Memetakan bobot pretrained yolo26n.pt ke struktur layer YOLO26n+CA...")
build_yolo26n_ca_pretrained(
    base_pt_path=BASE_PT,
    ca_yaml_path=MODEL_YAML,
    out_pt_path=REMAPPED_PT,
    seed=42,
)

# 2. Bangun model dari YAML dan muat bobot hasil remapping
print("\n[2/4] Membangun model YOLO26n+CA dari YAML...")
model = YOLO(MODEL_YAML)

print("\n[3/4] Memulai pelatihan YOLO26n+CA...")
t0 = time.time()

results = model.train(
    data=DATA_YAML,
    pretrained=REMAPPED_PT,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    device=args.device,
    project=RUNS_DIR,
    name=args.run_name,
    exist_ok=True,
    patience=15,
    save_period=10,
    seed=42,
    verbose=True,
    plots=True,
)

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print("  [4/4] Training YOLO26n + CA SELESAI!")
print(f"  Total waktu  : {elapsed/3600:.2f} jam ({elapsed/60:.1f} menit)")
print(f"  Best weights : {RUNS_DIR}/{args.run_name}/weights/best.pt")
print(f"{'=' * 72}")

rd = results.results_dict
print("\n  Metrik Akhir Validation Set:")
print(f"    mAP@0.5        : {rd.get('metrics/mAP50(B)', 0.0):.4f}")
print(f"    mAP@0.5:0.95   : {rd.get('metrics/mAP50-95(B)', 0.0):.4f}")
print(f"    Precision (val): {rd.get('metrics/precision(B)', 0.0):.4f}")
print(f"    Recall    (val): {rd.get('metrics/recall(B)', 0.0):.4f}")
print("\n  >> Lanjut ke: python phase3_yolo_ca/02_evaluate_testset.py")

