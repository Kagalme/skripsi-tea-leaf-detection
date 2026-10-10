"""
FASE 2 (RESOLUSI STANDAR 640) - Script 04: Visual Analysis (TP, FP, WC, FN)
=============================================================================
Analisis visual mendalam per-gambar pada Test set:
- True Positive (TP): Deteksi benar, kelas benar (Kotak Hijau)
- False Positive (FP): Deteksi ada, tidak ada di Ground Truth (Kotak Merah)
- Wrong Classification (WC): Deteksi benar, kelas salah (Kotak Kuning-Oranye)
- False Negative (FN): Ground Truth ada, terlewat oleh model (Kotak Magenta)
"""

import os
import sys
import cv2
import json
import random
from pathlib import Path
from ultralytics import YOLO

# Deteksi platform
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

PROJECT_DIR = BASE_DIR / "phase2_runs"
RUN_NAME    = "yolo26n_baseline_640"
BEST_MODEL  = PROJECT_DIR / RUN_NAME / "weights/best.pt"
TEST_IMG_DIR= BASE_DIR / "tea_yolo/images/test"
TEST_LBL_DIR= BASE_DIR / "tea_yolo/labels/test"
OUTPUT_DIR  = PROJECT_DIR / RUN_NAME / "visual_analysis"

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",         # 1
    "Gray Blight",          # 2
    "Helopeltis",           # 3
    "Red Spider",           # 4
    "Green Mirid Bug",      # 5
    "Healty Leaf",          # 6
]

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
MAX_SAMPLES = 10

print("=" * 70)
print("  FASE 2 - VISUAL ANALYSIS KATEGORISAL (TP, FP, WC, FN) [640x640]")
print("=" * 70)
print(f"Model Path  : {BEST_MODEL}")
print(f"Test Images : {TEST_IMG_DIR}")
print(f"Output Dir  : {OUTPUT_DIR}")

if not BEST_MODEL.exists():
    print(f"[ERROR] {BEST_MODEL} tidak ditemukan! Pastikan pelatihan selesai.")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for cat in ["true_positive", "false_positive", "wrong_class", "false_negative"]:
    (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

model = YOLO(str(BEST_MODEL))

def xywhn_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_box(img, box, label, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img

test_images = sorted(
    list(TEST_IMG_DIR.glob("*.jpg"))
    + list(TEST_IMG_DIR.glob("*.jpeg"))
    + list(TEST_IMG_DIR.glob("*.png"))
)
random.seed(42)
random.shuffle(test_images)

stats = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}
samples_saved = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}

print(f"\nMenganalisis {len(test_images)} citra test set...")

for idx, img_path in enumerate(test_images):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # Baca Ground Truth
    lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
    gt_boxes, gt_classes = [], []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    gt_boxes.append(xywhn_to_xyxy(coords, w, h))
                    gt_classes.append(cls_id)

    # Jalankan Prediksi YOLO (imgsz=640)
    preds = model.predict(str(img_path), imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes, pred_classes, pred_confs = [], [], []
    if preds.boxes is not None and len(preds.boxes) > 0:
        for b in preds.boxes:
            pred_boxes.append([int(x) for x in b.xyxy[0].tolist()])
            pred_classes.append(int(b.cls[0]))
            pred_confs.append(float(b.conf[0]))

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    img_events = []

    # Match Prediksi -> Ground Truth
    for pi, (pb, pc, pconf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
        best_iou = 0.0
        best_gi = -1
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_matched[gi]:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= IOU_THRESHOLD and best_gi >= 0:
            gt_matched[best_gi] = True
            pred_matched[pi] = True
            gc = gt_classes[best_gi]
            gb = gt_boxes[best_gi]
            if pc == gc:
                stats["TP"] += 1
                img_events.append(("TP", pb, gb, pc, gc, pconf))
            else:
                stats["WC"] += 1
                img_events.append(("WC", pb, gb, pc, gc, pconf))
        else:
            stats["FP"] += 1
            img_events.append(("FP", pb, None, pc, None, pconf))

    # Ground Truth yang terlewat (False Negative)
    for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
        if not gt_matched[gi]:
            stats["FN"] += 1
            img_events.append(("FN", None, gb, None, gc, 0.0))

    # Simpan contoh gambar per kategori
    for cat, pb, gb, pc, gc, conf in img_events:
        if samples_saved[cat] < MAX_SAMPLES:
            vis = img.copy()
            if cat == "TP":
                draw_box(vis, pb, f"TP: {CLASS_NAMES[pc]} ({conf*100:.1f}%)", (0, 220, 0))
            elif cat == "FP":
                draw_box(vis, pb, f"FP: {CLASS_NAMES[pc]} ({conf*100:.1f}%)", (0, 0, 240))
            elif cat == "WC":
                draw_box(vis, pb, f"PRED: {CLASS_NAMES[pc]} ({conf*100:.1f}%)", (0, 165, 255))
                draw_box(vis, gb, f"GT: {CLASS_NAMES[gc]}", (255, 100, 0), thickness=1)
                cv2.line(vis, (pb[0], pb[1]), (gb[0], gb[1]), (0, 0, 255), 2)
            elif cat == "FN":
                draw_box(vis, gb, f"FN: {CLASS_NAMES[gc]} (Missed)", (255, 0, 255))

            folder_map = {
                "TP": "true_positive",
                "FP": "false_positive",
                "WC": "wrong_class",
                "FN": "false_negative",
            }
            out_file = OUTPUT_DIR / folder_map[cat] / f"{cat}_{img_path.stem}_{samples_saved[cat]+1}.jpg"
            cv2.imwrite(str(out_file), vis)
            samples_saved[cat] += 1

    if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
        print(f"  [{idx+1:3d}/{len(test_images)}] TP: {stats['TP']:3d} | FP: {stats['FP']:2d} | WC: {stats['WC']:2d} | FN: {stats['FN']:2d}")

total = sum(stats.values())
p_der = stats["TP"] / (stats["TP"] + stats["FP"] + stats["WC"] + 1e-9)
r_der = stats["TP"] / (stats["TP"] + stats["FN"] + stats["WC"] + 1e-9)
f1_der = 2 * p_der * r_der / (p_der + r_der + 1e-9)

print("\n" + "=" * 70)
print("  RINGKASAN ANALISIS VISUAL (TEST SET)")
print("=" * 70)
print(f"  True Positive  (TP) : {stats['TP']:4d}  ({stats['TP']/total*100:.1f}%) - Deteksi benar, kelas tepat")
print(f"  False Positive (FP) : {stats['FP']:4d}  ({stats['FP']/total*100:.1f}%) - Deteksi ada, tidak ada di GT")
print(f"  Wrong Class    (WC) : {stats['WC']:4d}  ({stats['WC']/total*100:.1f}%) - Deteksi benar, kelas tertukar")
print(f"  False Negative (FN) : {stats['FN']:4d}  ({stats['FN']/total*100:.1f}%) - GT ada, terlewat oleh model")
print(f"  Total Event         : {total:4d}")
print(f"\n  Derived Precision   : {p_der:.4f} ({p_der*100:.2f}%)")
print(f"  Derived Recall      : {r_der:.4f} ({r_der*100:.2f}%)")
print(f"  Derived F1-Score    : {f1_der:.4f} ({f1_der*100:.2f}%)")
print(f"\n  Sampel citra tersimpan di: {OUTPUT_DIR}")
print("=" * 70)

with open(OUTPUT_DIR / "visual_stats.json", "w", encoding="utf-8") as f:
    json.dump({
        "stats": stats,
        "derived_precision": p_der,
        "derived_recall": r_der,
        "derived_f1": f1_der
    }, f, indent=2)
