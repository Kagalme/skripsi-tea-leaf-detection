"""
FASE 3 (DIPERBARUI) - Script 05: Inferensi End-to-End YOLO26n+CA + Edge Contour
===============================================================================
Menjalankan deteksi penyakit daun teh dengan YOLO26n+CA diikuti ekstraksi
kontur tepi daun (Canny + Morfologi Closing) dan penyaringan daun tidak utuh.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_yolo_ca_finetune.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO  # noqa: E402
from phase3_yolo_ca_finetune.edge_contour import (
    EdgeContourPostProcessor,
    EdgeContourConfig,
    yolo_result_to_detections,
    CLASS_NAMES,
)

parser = argparse.ArgumentParser(description="Inferensi YOLO26n+CA + Edge Contour (Fase 3 v2)")
parser.add_argument("--source", required=True, help="Path gambar tunggal atau folder gambar")
parser.add_argument("--weights", default=None, help="Path ke best.pt CA")
parser.add_argument("--device", default="cuda", help="cuda | cpu")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--iou", type=float, default=0.45)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--canny_low", type=int, default=30)
parser.add_argument("--canny_high", type=int, default=100)
parser.add_argument("--morph_close", type=int, default=7)
parser.add_argument("--border_margin", type=int, default=5)
parser.add_argument("--min_solidity", type=float, default=0.45)
parser.add_argument("--fallback", action="store_true", default=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save_dir", default=None)
parser.add_argument("--show_suppressed", action="store_true")
parser.add_argument("--show_bbox", action="store_true")
parser.add_argument("--max_images", type=int, default=0)
parser.add_argument("--save_stats", action="store_true")
args = parser.parse_args()


def find_weights():
    candidates = [
        args.weights,
        os.path.join(PROJECT_DIR, "phase3_runs", "yolo26n_ca_finetune", "weights", "best.pt"),
        "/content/phase3_runs/yolo26n_ca_50ep/weights/best.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


weights = find_weights()
if weights is None:
    print("[ERROR] best.pt model YOLO26n+CA tidak ditemukan. Jalankan 01_train_ca.py terlebih dahulu.")
    sys.exit(1)

source = Path(args.source)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
if source.is_file() and source.suffix.lower() in IMG_EXTS:
    image_paths = [source]
elif source.is_dir():
    image_paths = sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS])
else:
    print(f"[ERROR] Source '{source}' tidak ditemukan.")
    sys.exit(1)

if args.max_images > 0:
    image_paths = image_paths[: args.max_images]

if args.save:
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif os.path.exists("/content"):
        save_dir = Path("/content/phase3_runs/contour_inference")
    else:
        save_dir = Path(PROJECT_DIR) / "phase3_runs" / "contour_inference"
    save_dir.mkdir(parents=True, exist_ok=True)
else:
    save_dir = None

print("=" * 72)
print("  FASE 3 (v2) — Inferensi YOLO26n+CA + Edge Contour Post-Processing")
print("=" * 72)
print(f"  Model   : {weights}")
print(f"  Gambar  : {len(image_paths)} citra dari '{source}'")
if save_dir:
    print(f"  Save Dir: {save_dir}")

model = YOLO(weights)
cfg = EdgeContourConfig(
    canny_low=args.canny_low,
    canny_high=args.canny_high,
    morph_close_ksize=args.morph_close,
    border_margin=args.border_margin,
    min_solidity=args.min_solidity,
    fallback_to_bbox=args.fallback,
)
proc = EdgeContourPostProcessor(config=cfg)

global_stats = {
    "total_images": len(image_paths),
    "total_detections": 0,
    "complete_leaves": 0,
    "suppressed_leaves": 0,
    "per_class_complete": {c: 0 for c in CLASS_NAMES},
    "per_class_suppressed": {c: 0 for c in CLASS_NAMES},
    "processing_time_s": [],
}

t_total_start = time.time()
for idx, img_path in enumerate(image_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    t_start = time.time()
    yolo_preds = model.predict(
        str(img_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]
    detections = yolo_result_to_detections(yolo_preds)
    contour_results = proc.process_detections(img, detections)
    summary = proc.summarize(contour_results)

    global_stats["total_detections"] += summary["total_detections"]
    global_stats["complete_leaves"] += summary["complete_leaves"]
    global_stats["suppressed_leaves"] += summary["suppressed_leaves"]
    for res in contour_results:
        cname = res.detection.class_name
        if res.is_complete:
            global_stats["per_class_complete"][cname] = global_stats["per_class_complete"].get(cname, 0) + 1
        else:
            global_stats["per_class_suppressed"][cname] = global_stats["per_class_suppressed"].get(cname, 0) + 1

    t_elapsed = time.time() - t_start
    global_stats["processing_time_s"].append(t_elapsed)

    vis = proc.draw_results(
        img,
        contour_results,
        show_suppressed=args.show_suppressed,
        show_bbox=args.show_bbox,
    )

    if save_dir is not None:
        cv2.imwrite(str(save_dir / img_path.name), vis)

    if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
        print(
            f"  [{idx+1:4d}/{len(image_paths)}] {img_path.name:<32} "
            f"det={summary['total_detections']:2d} utuh={summary['complete_leaves']:2d} "
            f"buang={summary['suppressed_leaves']:2d} ({t_elapsed*1000:.0f}ms)"
        )

t_total = time.time() - t_total_start
avg_ms = float(np.mean(global_stats["processing_time_s"]) * 1000) if global_stats["processing_time_s"] else 0.0
avg_fps = len(image_paths) / (t_total + 1e-6)

if args.save_stats:
    stats_out = (save_dir if save_dir else Path(PROJECT_DIR) / "phase3_runs") / "contour_stats.json"
    global_stats["processing_time_s"] = avg_ms
    global_stats["avg_fps"] = avg_fps
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)
    print(f"\n  Statistik disimpan di: {stats_out}")

print("=" * 72)

