"""
FASE 2 - Script 05: Visual Analysis
=====================================
Menghasilkan visualisasi prediksi model pada test set:
- True Positive (TP): Deteksi benar, kelas benar
- False Positive (FP): Deteksi ada, tapi tidak ada ground truth
- Wrong Classification (WC): Deteksi benar, tapi kelas salah
- False Negative (FN): Ground truth ada, tapi tidak terdeteksi

Jalankan SETELAH script 04_evaluate_testset.py selesai.
"""

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

print("=" * 70)
print("  FASE 2 - Visual Analysis (TP, FP, WC, FN)")
print("=" * 70)

# ============ Konfigurasi ============
PROJECT_DIR = "D:/SKRIPSI/phase2_runs"
RUN_NAME    = "yolo26n_baseline_50ep"
BEST_MODEL  = f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
TEST_IMG_DIR= "D:/SKRIPSI/tea_yolo/images/test"
TEST_LBL_DIR= "D:/SKRIPSI/tea_yolo/labels/test"
OUTPUT_DIR  = f"{PROJECT_DIR}/{RUN_NAME}/visual_analysis"

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",          # 1
    "Gray Blight",           # 2
    "Helopeltis",            # 3
    "Red Spider",            # 4
    "Green Mirid Bug",       # 5
    "Healty Leaf",           # 6
]

# Warna per kelas (BGR)
CLASS_COLORS = [
    (0, 165, 255),    # Orange - algal
    (0, 0, 255),      # Red - brown blight
    (128, 128, 128),  # Gray - gray blight
    (255, 0, 255),    # Magenta - helopeltis
    (0, 0, 128),      # Dark red - red spider
    (0, 255, 0),      # Green - green mirid
    (255, 255, 0),    # Cyan - healthy
]

IOU_THRESHOLD = 0.5   # IoU threshold untuk match
CONF_THRESHOLD = 0.25 # Confidence threshold prediksi

os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ["true_positive", "false_positive", "wrong_class", "false_negative", "samples"]:
    os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)

print(f"\nModel     : {BEST_MODEL}")
print(f"Test imgs : {TEST_IMG_DIR}")
print(f"Output    : {OUTPUT_DIR}")

model = YOLO(BEST_MODEL)

def xywhn_to_xyxy(box, w, h):
    """Convert YOLO normalized xywh to pixel xyxy."""
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_box(img, box, label, color, thickness=2):
    """Draw bounding box dengan label."""
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1-label_size[1]-4), (x1+label_size[0], y1), color, -1)
    cv2.putText(img, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

# ============ Analisis per gambar ============
test_images = list(Path(TEST_IMG_DIR).glob("*.jpg")) + list(Path(TEST_IMG_DIR).glob("*.jpeg")) + list(Path(TEST_IMG_DIR).glob("*.png"))
random.seed(42)
random.shuffle(test_images)

stats = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}
samples_saved = {"TP": 0, "FP": 0, "WC": 0, "FN": 0}
MAX_SAMPLES = 10  # Simpan max 10 contoh per kategori

print(f"\nAnalisis {len(test_images)} gambar test...")

for idx, img_path in enumerate(test_images):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # Load ground truth labels
    lbl_path = Path(TEST_LBL_DIR) / (img_path.stem + ".txt")
    gt_boxes = []
    gt_classes = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    box = xywhn_to_xyxy([float(x) for x in parts[1:5]], w, h)
                    gt_boxes.append(box)
                    gt_classes.append(cls)
    
    # Run prediction
    preds = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = []
    pred_classes = []
    pred_confs = []
    if preds.boxes is not None and len(preds.boxes) > 0:
        for box_obj in preds.boxes:
            pred_boxes.append([int(x) for x in box_obj.xyxy[0].tolist()])
            pred_classes.append(int(box_obj.cls[0]))
            pred_confs.append(float(box_obj.conf[0]))
    
    # Match predictions to ground truth using IoU
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    
    img_categories = []  # (category, pred_box, gt_box, pred_cls, gt_cls, conf)
    
    for pi, (pb, pc, pconf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
        best_iou = 0
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
                # True Positive
                stats["TP"] += 1
                img_categories.append(("TP", pb, gb, pc, gc, pconf))
            else:
                # Wrong Classification
                stats["WC"] += 1
                img_categories.append(("WC", pb, gb, pc, gc, pconf))
        else:
            # False Positive
            stats["FP"] += 1
            img_categories.append(("FP", pb, None, pc, None, pconf))
    
    # False Negatives (unmatched GT)
    for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
        if not gt_matched[gi]:
            stats["FN"] += 1
            img_categories.append(("FN", None, gb, None, gc, 0))
    
    # Simpan contoh gambar
    for cat, pb, gb, pc, gc, conf in img_categories:
        if samples_saved[cat] < MAX_SAMPLES:
            vis = img.copy()
            
            if cat == "TP":
                draw_box(vis, pb, f"TP:{CLASS_NAMES[pc]}({conf:.2f})", (0,255,0))
            elif cat == "FP":
                draw_box(vis, pb, f"FP:{CLASS_NAMES[pc]}({conf:.2f})", (0,0,255))
            elif cat == "WC":
                draw_box(vis, pb, f"PRED:{CLASS_NAMES[pc]}({conf:.2f})", (0,165,255))
                draw_box(vis, gb, f"GT:{CLASS_NAMES[gc]}", (255,0,0), thickness=1)
                # Garis merah penanda salah kelas
                cv2.line(vis, (pb[0],pb[1]), (gb[0],gb[1]), (0,0,255), 2)
            elif cat == "FN":
                draw_box(vis, gb, f"FN:{CLASS_NAMES[gc]}(missed)", (255,0,255))
            
            out_path = f"{OUTPUT_DIR}/{cat.lower()}/{cat}_{img_path.stem}_{samples_saved[cat]}.jpg"
            cv2.imwrite(out_path, vis)
            samples_saved[cat] += 1
    
    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/{len(test_images)}] TP:{stats['TP']} FP:{stats['FP']} WC:{stats['WC']} FN:{stats['FN']}")

# ============ Print summary ============
total = sum(stats.values())
print(f"\n{'='*70}")
print(f"  VISUAL ANALYSIS SUMMARY")
print(f"{'='*70}")
print(f"  True Positive  (TP) : {stats['TP']:4d}  ({stats['TP']/total*100:.1f}%) - deteksi benar, kelas benar")
print(f"  False Positive (FP) : {stats['FP']:4d}  ({stats['FP']/total*100:.1f}%) - deteksi ada, tidak ada di GT")
print(f"  Wrong Class    (WC) : {stats['WC']:4d}  ({stats['WC']/total*100:.1f}%) - deteksi benar, kelas salah")
print(f"  False Negative (FN) : {stats['FN']:4d}  ({stats['FN']/total*100:.1f}%) - GT ada, tidak terdeteksi")
print(f"  Total               : {total:4d}")

precision = stats['TP'] / (stats['TP'] + stats['FP'] + stats['WC'] + 1e-9)
recall    = stats['TP'] / (stats['TP'] + stats['FN'] + stats['WC'] + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)
print(f"\n  Derived Precision: {precision:.4f}")
print(f"  Derived Recall   : {recall:.4f}")
print(f"  Derived F1       : {f1:.4f}")

print(f"\n  Contoh gambar disimpan di: {OUTPUT_DIR}/")
print(f"{'='*70}")

# Save stats JSON
with open(f"{OUTPUT_DIR}/visual_stats.json", 'w') as f:
    json.dump({"stats": stats, "precision": precision, "recall": recall, "f1": f1}, f, indent=2)
