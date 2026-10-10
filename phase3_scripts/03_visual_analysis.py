"""
FASE 3 - Script 03: Visual Analysis YOLO26n + CA
==================================================
Menghasilkan visualisasi kesalahan deteksi (TP, FP, WC, FN)
untuk model YOLO26n + CA pada test set.

Output:
  visual_analysis/
    ├── true_positive/     (maks 10 contoh)
    ├── false_positive/    (maks 10 contoh)
    ├── wrong_class/       (maks 10 contoh)
    ├── false_negative/    (maks 10 contoh)
    ├── samples/
    └── visual_stats.json

Jalankan SETELAH script 02_evaluate_testset.py selesai.
"""

import os
import sys
import cv2
import json
import random
import argparse
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_scripts.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO  # noqa

parser = argparse.ArgumentParser(description="FASE 3: Visual Analysis")
parser.add_argument("--device",   default="cuda")
parser.add_argument("--run_name", default="yolo26n_ca_50ep")
parser.add_argument("--conf",     type=float, default=0.25)
parser.add_argument("--iou_thr",  type=float, default=0.5)
parser.add_argument("--max_samples", type=int, default=10)
args = parser.parse_args()

# ─────────────────────────────────────── PATH ──────────────────────────────
if os.path.exists("/content/tea_yolo"):
    RUNS_DIR    = "/content/phase3_runs"
    TEST_IMG    = "/content/tea_yolo/images/test"
    TEST_LBL    = "/content/tea_yolo/labels/test"
else:
    RUNS_DIR    = os.path.join(PROJECT_DIR, "phase3_runs")
    TEST_IMG    = os.path.join(PROJECT_DIR, "tea_yolo", "images", "test")
    TEST_LBL    = os.path.join(PROJECT_DIR, "tea_yolo", "labels", "test")

RUN_DIR    = os.path.join(RUNS_DIR, args.run_name)
BEST_MODEL = os.path.join(RUN_DIR, "weights", "best.pt")
OUT_DIR    = os.path.join(RUN_DIR, "visual_analysis")

CLASS_NAMES = [
    "Tea algal leaf spot", "Brown Blight", "Gray Blight",
    "Helopeltis", "Red Spider", "Green Mirid Bug", "Healty Leaf",
]
CLASS_COLORS = [
    (0, 165, 255), (0, 0, 255), (128, 128, 128),
    (255, 0, 255), (0, 0, 128), (0, 255, 0), (255, 255, 0),
]

print("=" * 70)
print("  FASE 3 - Visual Analysis YOLO26n + CA (TP, FP, WC, FN)")
print("=" * 70)

if not os.path.exists(BEST_MODEL):
    print(f"[ERROR] {BEST_MODEL} tidak ditemukan.")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)
for sub in ["true_positive", "false_positive", "wrong_class", "false_negative", "samples"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

model = YOLO(BEST_MODEL)

# ─────────────────────────────────────── UTILS ─────────────────────────────
def xywhn_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)]

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def draw_box(img, box, label, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1-th-4), (x1+tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

# ─────────────────────────────────────── ANALISIS ──────────────────────────
test_images = sorted(
    list(Path(TEST_IMG).glob("*.jpg")) +
    list(Path(TEST_IMG).glob("*.jpeg")) +
    list(Path(TEST_IMG).glob("*.png"))
)
random.seed(42)
random.shuffle(test_images)

stats = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}
saved = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}

print(f"\nAnalisis {len(test_images)} gambar test (model: YOLO26n + CA)...")

for idx, img_path in enumerate(test_images):
    img = cv2.imread(str(img_path))
    if img is None: continue
    h, w = img.shape[:2]

    # Ground truth
    lbl_path = Path(TEST_LBL) / (img_path.stem + ".txt")
    gt_boxes, gt_cls = [], []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                gt_cls.append(int(parts[0]))
                gt_boxes.append(xywhn_to_xyxy([float(x) for x in parts[1:5]], w, h))

    # Prediction
    preds = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
    pred_boxes, pred_cls, pred_confs = [], [], []
    if preds.boxes is not None and len(preds.boxes) > 0:
        for box_obj in preds.boxes:
            pred_boxes.append([int(v) for v in box_obj.xyxy[0].tolist()])
            pred_cls.append(int(box_obj.cls[0]))
            pred_confs.append(float(box_obj.conf[0]))

    # Matching
    gt_matched = [False] * len(gt_boxes)
    img_cats = []

    for pb, pc, pconf in zip(pred_boxes, pred_cls, pred_confs):
        best_iou, best_gi = 0, -1
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
            if gt_matched[gi]: continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_gi = v, gi

        if best_iou >= args.iou_thr and best_gi >= 0:
            gt_matched[best_gi] = True
            gc = gt_cls[best_gi]; gb = gt_boxes[best_gi]
            if pc == gc:
                stats["TP"] += 1; img_cats.append(("TP", pb, gb, pc, gc, pconf))
            else:
                stats["WC"] += 1; img_cats.append(("WC", pb, gb, pc, gc, pconf))
        else:
            stats["FP"] += 1; img_cats.append(("FP", pb, None, pc, None, pconf))

    for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
        if not gt_matched[gi]:
            stats["FN"] += 1; img_cats.append(("FN", None, gb, None, gc, 0))

    # Simpan sampel
    for cat, pb, gb, pc, gc, conf in img_cats:
        if saved[cat] < args.max_samples:
            vis = img.copy()
            cname_pred = CLASS_NAMES[pc] if pc is not None else ""
            cname_gt   = CLASS_NAMES[gc] if gc is not None else ""
            col_pred   = CLASS_COLORS[pc] if pc is not None else (0,0,255)
            if cat == "TP":
                draw_box(vis, pb, f"TP:{cname_pred}({conf:.2f})", (0,255,0))
            elif cat == "FP":
                draw_box(vis, pb, f"FP:{cname_pred}({conf:.2f})", (0,0,255))
            elif cat == "WC":
                draw_box(vis, pb, f"PRED:{cname_pred}({conf:.2f})", (0,165,255))
                draw_box(vis, gb, f"GT:{cname_gt}", (255,0,0), thickness=1)
                cv2.line(vis, (pb[0],pb[1]), (gb[0],gb[1]), (0,0,255), 2)
            elif cat == "FN":
                draw_box(vis, gb, f"FN:{cname_gt}(missed)", (255,0,255))
            out = os.path.join(OUT_DIR, cat.lower(), f"{cat}_{img_path.stem}_{saved[cat]}.jpg")
            cv2.imwrite(out, vis)
            saved[cat] += 1

    if (idx+1) % 50 == 0:
        print(f"  [{idx+1}/{len(test_images)}] TP:{stats['TP']} FP:{stats['FP']} WC:{stats['WC']} FN:{stats['FN']}")

# ─────────────────────────────────────── SUMMARY ───────────────────────────
total = sum(stats.values())
prec  = stats['TP'] / (stats['TP'] + stats['FP'] + stats['WC'] + 1e-9)
rec   = stats['TP'] / (stats['TP'] + stats['FN'] + stats['WC'] + 1e-9)
f1    = 2 * prec * rec / (prec + rec + 1e-9)

print(f"\n{'=' * 70}")
print(f"  VISUAL ANALYSIS — YOLO26n + CA")
print(f"{'=' * 70}")
print(f"  True Positive  (TP) : {stats['TP']:4d}  ({stats['TP']/total*100:.1f}%)")
print(f"  False Positive (FP) : {stats['FP']:4d}  ({stats['FP']/total*100:.1f}%)")
print(f"  Wrong Class    (WC) : {stats['WC']:4d}  ({stats['WC']/total*100:.1f}%)")
print(f"  False Negative (FN) : {stats['FN']:4d}  ({stats['FN']/total*100:.1f}%)")
print(f"  Total               : {total}")
print(f"\n  Visual Precision : {prec:.4f}")
print(f"  Visual Recall    : {rec:.4f}")
print(f"  Visual F1        : {f1:.4f}")

with open(os.path.join(OUT_DIR, "visual_stats.json"), 'w') as f:
    json.dump({"stats": stats, "precision": prec, "recall": rec, "f1": f1}, f, indent=2)

print(f"\n  Sampel tersimpan di: {OUT_DIR}")
print(f"\n  >> Lanjut ke: python phase3_scripts/04_compare_baseline_vs_ca.py")
