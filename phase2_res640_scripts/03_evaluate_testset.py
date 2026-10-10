"""
FASE 2 (RESOLUSI STANDAR 640) - Script 03: Evaluasi Test Set
==============================================================
Menjalankan evaluasi komprehensif model terbaik (best.pt) pada Test set
dengan resolusi standar 640x640, menghasilkan metrik per-kelas, kurva PR,
dan confusion matrix.
"""

import os
import sys
import json
import csv
from pathlib import Path
import torch
from ultralytics import YOLO

# Deteksi platform
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

PROJECT_DIR = BASE_DIR / "phase2_runs"
RUN_NAME    = "yolo26n_baseline_640"
DATA_YAML   = BASE_DIR / "tea_yolo/data.yaml"
BEST_MODEL  = PROJECT_DIR / RUN_NAME / "weights/best.pt"
EVAL_OUTPUT = PROJECT_DIR / RUN_NAME / "test_eval"

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",         # 1
    "Gray Blight",          # 2
    "Helopeltis",           # 3
    "Red Spider",           # 4
    "Green Mirid Bug",      # 5
    "Healty Leaf",          # 6
]

print("=" * 70)
print("  FASE 2 - EVALUASI FINAL TEST SET (RESOLUSI STANDAR 640x640)")
print("=" * 70)
print(f"Model Path  : {BEST_MODEL}")
print(f"Dataset YAML: {DATA_YAML}")
print(f"Output Eval : {EVAL_OUTPUT}")

if not BEST_MODEL.exists():
    print(f"\n[ERROR] File {BEST_MODEL} tidak ditemukan!")
    print("Pastikan pelatihan script 02_train_baseline_640.py sudah selesai.")
    sys.exit(1)

# Pilih device
device = 0 if torch.cuda.is_available() else "cpu"

print(f"\n[1/3] Memuat model {BEST_MODEL.name} ...")
model = YOLO(str(BEST_MODEL))

print(f"[2/3] Menjalankan evaluasi pada Test Set (imgsz=640, device={device}) ...")
metrics = model.val(
    data        = str(DATA_YAML),
    split       = "test",           # Menggunakan split test murni
    imgsz       = 640,              # Resolusi standar 640
    batch       = 16 if device == 0 else 4,
    workers     = 4 if IS_COLAB else 0,
    device      = device,
    project     = str(EVAL_OUTPUT),
    name        = "test_results",
    save_json   = True,
    plots       = True,             # Generate confusion matrix & curves
    verbose     = True,
)

print(f"\n[3/3] Mengolah dan mencatat metrik evaluasi ...")
mp      = float(metrics.box.mp)
mr      = float(metrics.box.mr)
map50   = float(metrics.box.map50)
map5095 = float(metrics.box.map)
f1_all  = 2 * mp * mr / (mp + mr + 1e-9)

print("\n" + "=" * 70)
print("  RINGKASAN METRIK EVALUASI KESELURUHAN (TEST SET - 640x640)")
print("=" * 70)
print(f"  Precision    : {mp:.4f}  ({mp*100:.2f}%)")
print(f"  Recall       : {mr:.4f}  ({mr*100:.2f}%)")
print(f"  F1-Score     : {f1_all:.4f}  ({f1_all*100:.2f}%)")
print(f"  mAP@0.5      : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95 : {map5095:.4f}  ({map5095*100:.2f}%)")

print("\n" + "-" * 70)
print(f"{'Kelas Penyakit / Hama':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'mAP50':>10}")
print("-" * 70)

per_class_records = []
if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        idx_int = int(class_idx)
        cname = CLASS_NAMES[idx_int] if idx_int < len(CLASS_NAMES) else f"class_{idx_int}"
        p = float(metrics.box.p[i]) if metrics.box.p is not None else 0.0
        r = float(metrics.box.r[i]) if metrics.box.r is not None else 0.0
        ap = float(metrics.box.ap50[i]) if metrics.box.ap50 is not None else 0.0
        f1 = 2 * p * r / (p + r + 1e-9)
        
        per_class_records.append({
            "class": cname,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap,
        })
        print(f"{cname:<25} {p*100:>9.2f}% {r*100:>9.2f}% {f1*100:>7.2f}% {ap*100:>9.2f}%")

print("-" * 70)

# Simpan CSV per-class
out_dir = EVAL_OUTPUT / "test_results"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "per_class_metrics.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "map50"])
    writer.writeheader()
    for row in per_class_records:
        writer.writerow(row)
    writer.writerow({
        "class": "MEAN",
        "precision": mp,
        "recall": mr,
        "f1": f1_all,
        "map50": map50,
    })

# Simpan JSON summary
json_path = out_dir / "summary_metrics.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "overall": {
            "precision": mp,
            "recall": mr,
            "f1": f1_all,
            "map50": map50,
            "map50_95": map5095,
        },
        "per_class": per_class_records
    }, f, indent=2)

print(f"\n[SUKSES] Laporan tersimpan:")
print(f"         • CSV  : {csv_path}")
print(f"         • JSON : {json_path}")
print(f"         • Plot : {out_dir}/confusion_matrix.png")
print("=" * 70)
