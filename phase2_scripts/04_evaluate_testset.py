"""
FASE 2 - Script 04: Evaluasi Baseline pada Test Set
=====================================================
Menjalankan evaluasi final model terbaik (best.pt) pada Test set,
kemudian menghasilkan laporan lengkap dengan metrics per-kelas.

Jalankan SETELAH script 03_baseline_training.py selesai.
"""

import os
import json
import csv
from pathlib import Path
from ultralytics import YOLO

print("=" * 70)
print("  FASE 2 - Evaluasi Final pada Test Set")
print("=" * 70)

# ============ Konfigurasi ============
DATA_YAML    = "D:/SKRIPSI/tea_yolo/data.yaml"
PROJECT_DIR  = "D:/SKRIPSI/phase2_runs"
RUN_NAME     = "yolo26n_baseline_50ep"
BEST_MODEL   = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
EVAL_OUTPUT  = f"{PROJECT_DIR}/{RUN_NAME}/test_eval"

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",          # 1
    "Gray Blight",           # 2
    "Helopeltis",            # 3
    "Red Spider",            # 4
    "Green Mirid Bug",       # 5
    "Healty Leaf",           # 6
]

print(f"\nModel : {BEST_MODEL}")
print(f"Data  : {DATA_YAML}")
print(f"Output: {EVAL_OUTPUT}")

if not os.path.exists(BEST_MODEL):
    print(f"\nERROR: best.pt tidak ditemukan di {BEST_MODEL}")
    print("Pastikan training (script 03) sudah selesai!")
    exit(1)

# ============ Load model terbaik ============
print(f"\nLoading best.pt...")
model = YOLO(BEST_MODEL)

# ============ Evaluasi di Test set ============
print(f"\nMenjalankan evaluasi pada Test set...")
metrics = model.val(
    data     = DATA_YAML,
    split    = "test",         # Gunakan test split, bukan val
    imgsz    = 416,
    batch    = 4,
    workers  = 0,
    device   = "cpu",
    project  = EVAL_OUTPUT,
    name     = "test_results",
    save_json= True,           # Simpan hasil dalam JSON
    plots    = True,           # Confusion matrix, PR curve, dll
    verbose  = True,
)

# ============ Ekstrak dan cetak metrics ============
print(f"\n{'='*70}")
print(f"  HASIL EVALUASI TEST SET - YOLO26n BASELINE")
print(f"{'='*70}")

# Overall metrics
mp  = metrics.box.mp        # Mean Precision
mr  = metrics.box.mr        # Mean Recall  
map50   = metrics.box.map50     # mAP@0.5
map5095 = metrics.box.map       # mAP@0.5:0.95

# F1 overall (harmonic mean P & R)
f1_overall = 2 * mp * mr / (mp + mr + 1e-9)

print(f"\n[OVERALL METRICS]")
print(f"  Precision  : {mp:.4f}  ({mp*100:.2f}%)")
print(f"  Recall     : {mr:.4f}  ({mr*100:.2f}%)")
print(f"  F1-Score   : {f1_overall:.4f}  ({f1_overall*100:.2f}%)")
print(f"  mAP@0.5    : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95: {map5095:.4f}  ({map5095*100:.2f}%)")

# Per-class metrics
print(f"\n[PER-CLASS METRICS]")
print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'mAP50':>10}")
print("-" * 65)

per_class_data = []
if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        cname = CLASS_NAMES[int(class_idx)] if int(class_idx) < len(CLASS_NAMES) else f"class_{class_idx}"
        
        # Extract per-class values
        p  = float(metrics.box.p[i])  if metrics.box.p  is not None else 0
        r  = float(metrics.box.r[i])  if metrics.box.r  is not None else 0
        ap = float(metrics.box.ap50[i]) if metrics.box.ap50 is not None else 0
        f1 = 2 * p * r / (p + r + 1e-9)
        
        per_class_data.append({
            "class": cname,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap,
        })
        print(f"  {cname:<23} {p:>10.4f} {r:>10.4f} {f1:>8.4f} {ap:>10.4f}")
else:
    print("  (Per-class data tidak tersedia)")

print("-" * 65)
print(f"  {'MEAN':<23} {mp:>10.4f} {mr:>10.4f} {f1_overall:>8.4f} {map50:>10.4f}")

# ============ Simpan ke CSV ============
os.makedirs(EVAL_OUTPUT + "/test_results", exist_ok=True)
csv_path = f"{EVAL_OUTPUT}/test_results/per_class_metrics.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "map50"])
    writer.writeheader()
    for row in per_class_data:
        writer.writerow(row)
    writer.writerow({"class": "MEAN", "precision": mp, "recall": mr, "f1": f1_overall, "map50": map50})
print(f"\n  Metrics per-kelas disimpan ke: {csv_path}")

# ============ Simpan summary JSON ============
summary = {
    "model": "YOLO26n Baseline",
    "epochs": 50,
    "imgsz": 416,
    "split": "test",
    "overall": {
        "precision": float(mp),
        "recall": float(mr),
        "f1": float(f1_overall),
        "mAP50": float(map50),
        "mAP50_95": float(map5095),
    },
    "per_class": per_class_data
}

json_path = f"{EVAL_OUTPUT}/test_results/summary_metrics.json"
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Summary JSON disimpan ke   : {json_path}")

print(f"\n{'='*70}")
print(f"  Evaluasi selesai! Lanjut ke script 05_visual_analysis.py")
print(f"{'='*70}")
