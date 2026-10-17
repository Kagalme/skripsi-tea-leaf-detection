"""
FASE 3 (DIPERBARUI) - Script 02: Evaluasi YOLO26n + CA pada Test Set (640x640)
==============================================================================
Mengevaluasi model terbaik (best.pt) hasil training Fase 3 pada Held-Out Test Set
(529 gambar) dengan resolusi 640x640.

Perbaikan:
- Menyimpan baik key `"mAP50"` maupun `"map50"` (serta `"mAP50_95"` dan `"map50_95"`)
  di dalam `summary_metrics.json` untuk mencegah ketidakcocokan kapitalisasi key JSON
  ketika dibandingkan dengan berbagai versi baseline Fase 2.
- Menyertakan referensi baseline YOLO26n resolusi 640x640 (bukan 416x416) agar
  perbandingan adil (apple-to-apple).
"""

import os
import sys
import json
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_yolo_ca.ca_module import register_ca_module, find_best_weights
register_ca_module()

from ultralytics import YOLO  # noqa: E402

parser = argparse.ArgumentParser(description="FASE 3 (v2): Evaluasi Test Set")
parser.add_argument("--device", default="cuda", help="cuda | cpu")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--run_name", default="yolo26n_ca_50ep")
args = parser.parse_args()

if os.path.exists("/content/tea_yolo"):
    DATA_YAML = "/content/tea_yolo/data.yaml"
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
    except Exception:
        pass
else:
    DATA_YAML = os.path.join(PROJECT_DIR, "tea_yolo", "data.yaml")
    RUNS_DIR = os.path.join(PROJECT_DIR, "phase3_runs")

RUN_DIR = os.path.join(RUNS_DIR, args.run_name)
BEST_MODEL = os.path.join(RUN_DIR, "weights", "best.pt")
if not os.path.exists(BEST_MODEL):
    BEST_MODEL = find_best_weights(args.run_name)
EVAL_DIR = os.path.join(RUN_DIR, "test_eval")

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",         # 1
    "Gray Blight",          # 2
    "Helopeltis",           # 3
    "Red Spider",           # 4
    "Green Mirid Bug",      # 5
    "Healty Leaf",          # 6
]

print("=" * 72)
print("  FASE 3 (v2) - Evaluasi Final YOLO26n + CA pada Test Set (640x640)")
print("=" * 72)
print(f"  Model  : {BEST_MODEL}")
print(f"  Data   : {DATA_YAML}")
print(f"  Output : {EVAL_DIR}")

if not os.path.exists(BEST_MODEL):
    print(f"\n[ERROR] best.pt tidak ditemukan di: {BEST_MODEL}")
    print("  Pastikan script 01_train_ca.py sudah selesai dijalankan!")
    sys.exit(1)

print("\n[1/3] Memuat model YOLO26n + CA (best.pt)...")
model = YOLO(BEST_MODEL)

print("[2/3] Mengevaluasi pada Held-Out Test Split (529 gambar, 640x640)...")
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    device=args.device,
    project=EVAL_DIR,
    name="test_results",
    exist_ok=True,
    save_json=True,
    plots=True,
    verbose=True,
)

mp = float(metrics.box.mp)
mr = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)
f1_overall = 2 * mp * mr / (mp + mr + 1e-9)

print(f"\n{'=' * 72}")
print("  HASIL EVALUASI TEST SET — YOLO26n + Coordinate Attention (640x640)")
print(f"{'=' * 72}")
print(f"  Precision      : {mp:.4f}  ({mp*100:.2f}%)")
print(f"  Recall         : {mr:.4f}  ({mr*100:.2f}%)")
print(f"  F1-Score       : {f1_overall:.4f}  ({f1_overall*100:.2f}%)")
print(f"  mAP@0.5        : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95   : {map5095:.4f}  ({map5095*100:.2f}%)")

print(f"\n  {'Kelas':<25} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8} {'mAP50-95':>9}")
print("  " + "-" * 68)

per_class_data = []
if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        c_idx = int(class_idx)
        cname = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"class_{c_idx}"
        p = float(metrics.box.p[i])
        r = float(metrics.box.r[i])
        ap50 = float(metrics.box.ap50[i])
        ap95 = float(metrics.box.ap[i]) if hasattr(metrics.box, "ap") else ap50 * 0.95
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class_data.append({
            "class_id": c_idx,
            "class": cname,
            "class_name": cname,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap50,
            "mAP50": ap50,
            "map50_95": ap95,
            "mAP50_95": ap95,
        })
        print(f"  {cname:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {ap50:>8.4f} {ap95:>9.4f}")

print("  " + "-" * 68)
print(f"  {'MEAN':<25} {mp:>8.4f} {mr:>8.4f} {f1_overall:>8.4f} {map50:>8.4f} {map5095:>9.4f}")

print("\n[3/3] Menyimpan laporan CSV & JSON...")
results_dir = os.path.join(EVAL_DIR, "test_results")
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "per_class_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "map50", "map50_95"])
    writer.writeheader()
    for row in per_class_data:
        writer.writerow({
            "class": row["class"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "map50": row["map50"],
            "map50_95": row["map50_95"],
        })
    writer.writerow({
        "class": "MEAN",
        "precision": mp,
        "recall": mr,
        "f1": f1_overall,
        "map50": map50,
        "map50_95": map5095,
    })

summary = {
    "model": "YOLO26n + Coordinate Attention (v2 Authentic Architecture)",
    "architecture": "YOLO26n (C3k2 + SPPF + C2PSA + 3x CoordAttYOLO, end2end=True, reg_max=1)",
    "epochs": 50,
    "imgsz": args.imgsz,
    "split": "test",
    "ca_blocks": 3,
    "ca_positions": [
        "P3/8 (layer 5, 128ch)",
        "P4/16 (layer 8, 128ch)",
        "P5/32 (layer 13, 256ch)",
    ],
    "ca_reduction": 32,
    "overall": {
        "precision": mp,
        "recall": mr,
        "f1": f1_overall,
        "mAP50": map50,
        "map50": map50,
        "mAP50_95": map5095,
        "map50_95": map5095,
    },
    "per_class": per_class_data,
    "baseline_640_reference": {
        "model": "YOLO26n Baseline (640x640, 50 Epochs)",
        "mAP50": 0.9476,
        "mAP50_95": 0.9041,
        "precision": 0.9234,
        "recall": 0.8936,
        "f1": 0.9083,
    },
}

json_path = os.path.join(results_dir, "summary_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"  -> CSV  : {csv_path}")
print(f"  -> JSON : {json_path}")
print(f"\n{'=' * 72}")
print("  >> Lanjut ke: python phase3_yolo_ca/03_visual_analysis.py")
print(f"{'=' * 72}")

