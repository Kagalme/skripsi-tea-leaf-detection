"""
FASE 3 - Script 02: Evaluasi YOLO26n + CA pada Test Set
=========================================================
Mengevaluasi model terbaik (best.pt) hasil training Fase 3
pada Test Set (529 gambar, tidak pernah dilihat saat training).

Output:
  - per_class_metrics.csv       : precision, recall, F1, mAP50 per kelas
  - summary_metrics.json        : ringkasan overall + per-kelas
  - confusion_matrix.png        : confusion matrix
  - BoxPR_curve.png, dll.       : kurva evaluasi YOLO

Jalankan SETELAH script 01_train_ca.py selesai.
"""

import os
import sys
import json
import csv
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_scripts.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO  # noqa

# ─────────────────────────────────────── ARGPARSE ──────────────────────────
parser = argparse.ArgumentParser(description="FASE 3: Evaluasi Test Set")
parser.add_argument("--device",   default="cuda",            help="cuda | cpu")
parser.add_argument("--imgsz",    type=int, default=640)
parser.add_argument("--batch",    type=int, default=16)
parser.add_argument("--workers",  type=int, default=8)
parser.add_argument("--run_name", default="yolo26n_ca_50ep")
args = parser.parse_args()

# ─────────────────────────────────────── PATH ──────────────────────────────
if os.path.exists("/content/tea_yolo"):
    DATA_YAML   = "/content/tea_yolo/data.yaml"
    RUNS_DIR    = "/content/phase3_runs"
else:
    DATA_YAML   = os.path.join(PROJECT_DIR, "tea_yolo", "data.yaml")
    RUNS_DIR    = os.path.join(PROJECT_DIR, "phase3_runs")

RUN_DIR    = os.path.join(RUNS_DIR, args.run_name)
BEST_MODEL = os.path.join(RUN_DIR, "weights", "best.pt")
EVAL_DIR   = os.path.join(RUN_DIR, "test_eval")

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",          # 1
    "Gray Blight",           # 2
    "Helopeltis",            # 3
    "Red Spider",            # 4
    "Green Mirid Bug",       # 5
    "Healty Leaf",           # 6
]

# ─────────────────────────────────────── HEADER ─────────────────────────────
print("=" * 70)
print("  FASE 3 - Evaluasi Final YOLO26n + CA pada Test Set")
print("=" * 70)
print(f"\n  Model  : {BEST_MODEL}")
print(f"  Data   : {DATA_YAML}")
print(f"  Output : {EVAL_DIR}")

if not os.path.exists(BEST_MODEL):
    print(f"\n[ERROR] best.pt tidak ditemukan di:\n  {BEST_MODEL}")
    print("  Pastikan script 01_train_ca.py sudah selesai!")
    sys.exit(1)

# ─────────────────────────────────────── EVALUASI ──────────────────────────
print("\n[1/3] Loading model CA (best.pt)...")
model = YOLO(BEST_MODEL)

print("[2/3] Evaluasi pada Test split...")
metrics = model.val(
    data     = DATA_YAML,
    split    = "test",
    imgsz    = args.imgsz,
    batch    = args.batch,
    workers  = args.workers,
    device   = args.device,
    project  = EVAL_DIR,
    name     = "test_results",
    save_json= True,
    plots    = True,
    verbose  = True,
)

# ─────────────────────────────────────── EKSTRAK METRIK ────────────────────
mp      = float(metrics.box.mp)
mr      = float(metrics.box.mr)
map50   = float(metrics.box.map50)
map5095 = float(metrics.box.map)
f1_overall = 2 * mp * mr / (mp + mr + 1e-9)

print(f"\n{'=' * 70}")
print(f"  HASIL EVALUASI TEST SET — YOLO26n + Coordinate Attention")
print(f"{'=' * 70}")
print(f"\n  [OVERALL]")
print(f"  Precision      : {mp:.4f}  ({mp*100:.2f}%)")
print(f"  Recall         : {mr:.4f}  ({mr*100:.2f}%)")
print(f"  F1-Score       : {f1_overall:.4f}  ({f1_overall*100:.2f}%)")
print(f"  mAP@0.5        : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95   : {map5095:.4f}  ({map5095*100:.2f}%)")

print(f"\n  [PER KELAS]")
print(f"  {'Kelas':<25} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8}")
print("  " + "-" * 60)

per_class_data = []
if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        cname = CLASS_NAMES[int(class_idx)] if int(class_idx) < len(CLASS_NAMES) else f"class_{class_idx}"
        p  = float(metrics.box.p[i])
        r  = float(metrics.box.r[i])
        ap = float(metrics.box.ap50[i])
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class_data.append({"class": cname, "precision": p, "recall": r, "f1": f1, "map50": ap})
        print(f"  {cname:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {ap:>8.4f}")

print("  " + "-" * 60)
print(f"  {'MEAN':<25} {mp:>8.4f} {mr:>8.4f} {f1_overall:>8.4f} {map50:>8.4f}")

# ─────────────────────────────────────── SIMPAN HASIL ──────────────────────
print(f"\n[3/3] Menyimpan hasil evaluasi...")

results_dir = os.path.join(EVAL_DIR, "test_results")
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "per_class_metrics.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "map50"])
    writer.writeheader()
    for row in per_class_data:
        writer.writerow(row)
    writer.writerow({"class": "MEAN", "precision": mp, "recall": mr, "f1": f1_overall, "map50": map50})
print(f"  → {csv_path}")

summary = {
    "model": "YOLO26n + Coordinate Attention",
    "epochs": 50,
    "imgsz": args.imgsz,
    "split": "test",
    "ca_blocks": 3,
    "ca_positions": ["P3 (backbone stage 3)", "P4 (backbone stage 4)", "P5 (backbone stage 5)"],
    "ca_reduction": 32,
    "overall": {
        "precision": mp, "recall": mr, "f1": f1_overall,
        "mAP50": map50, "mAP50_95": map5095,
    },
    "per_class": per_class_data,
    "baseline_reference": {
        "model": "YOLO26n Baseline (Fase 2)",
        "mAP50": 0.9469,
        "mAP50_95": 0.9070,
        "precision": 0.9387,
        "recall": 0.8794,
        "f1": 0.9081,
    }
}

json_path = os.path.join(results_dir, "summary_metrics.json")
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  → {json_path}")

print(f"\n{'=' * 70}")
print(f"  Evaluasi selesai! Lanjut ke:")
print(f"  python phase3_scripts/03_visual_analysis.py")
print(f"{'=' * 70}")
