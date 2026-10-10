"""
FASE 3 - Script 05: Inferensi End-to-End YOLO26n+CA + Edge Contour
====================================================================
Pipeline lengkap untuk inferensi pada gambar baru:
  1. YOLO26n + CA mendeteksi penyakit (bounding box + kelas)
  2. Edge Contour Postprocessor:
     a. Crop setiap ROI
     b. Edge detection (Canny + Morfologi)
     c. Temukan kontur daun terbesar
     d. Filter daun tidak utuh (kontur menyentuh tepi gambar)
  3. Output: Polygon kontur mengikuti bentuk daun + label penyakit

PENGGUNAAN:
  # Satu gambar:
  python phase3_scripts/05_infer_with_contour.py --source gambar.jpg

  # Folder gambar:
  python phase3_scripts/05_infer_with_contour.py --source folder/

  # Test set:
  python phase3_scripts/05_infer_with_contour.py --source tea_yolo/images/test

  # Simpan output:
  python phase3_scripts/05_infer_with_contour.py --source folder/ --save

  # Tampilkan deteksi yang dibuang (debug):
  python phase3_scripts/05_infer_with_contour.py --source folder/ --show_suppressed
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# ─── Registrasi CA (HARUS sebelum YOLO import) ───────────────────────────
from phase3_scripts.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO  # noqa
from phase3_scripts.edge_contour import (
    EdgeContourPostProcessor, EdgeContourConfig,
    yolo_result_to_detections, CLASS_NAMES
)

# ─────────────────────────────────────── ARGPARSE ──────────────────────────
parser = argparse.ArgumentParser(
    description="Inferensi YOLO26n+CA dengan Edge Contour Postprocessing",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--source",    required=True,
                    help="Path gambar tunggal, folder gambar, atau 'tea_yolo/images/test'")
parser.add_argument("--weights",   default=None,
                    help="Path ke best.pt CA (auto-detect jika tidak diisi)")
parser.add_argument("--device",    default="cuda",   help="cuda | cpu")
parser.add_argument("--conf",      type=float, default=0.25)
parser.add_argument("--iou",       type=float, default=0.45, help="NMS IoU threshold")
parser.add_argument("--imgsz",     type=int, default=640)

# Edge Contour parameters
parser.add_argument("--canny_low",    type=int,   default=30,   help="Canny low threshold")
parser.add_argument("--canny_high",   type=int,   default=100,  help="Canny high threshold")
parser.add_argument("--morph_close",  type=int,   default=7,    help="Morphological closing kernel size")
parser.add_argument("--border_margin",type=int,   default=5,    help="Border margin untuk cek kelengkapan")
parser.add_argument("--min_solidity", type=float, default=0.45, help="Min solidity kontur")
parser.add_argument("--fallback",     action="store_true", default=True,
                    help="Fallback ke bbox jika kontur tidak ditemukan")

# Output
parser.add_argument("--save",             action="store_true", help="Simpan gambar hasil")
parser.add_argument("--save_dir",         default=None,        help="Direktori output (auto jika kosong)")
parser.add_argument("--show",             action="store_true", help="Tampilkan gambar (butuh display)")
parser.add_argument("--show_suppressed",  action="store_true", help="Gambar juga deteksi yang dibuang")
parser.add_argument("--show_bbox",        action="store_true", help="Gambar bbox asli YOLO (abu-abu)")
parser.add_argument("--max_images",       type=int, default=0, help="Batas jumlah gambar (0=semua)")
parser.add_argument("--save_stats",       action="store_true", help="Simpan statistik ke JSON")
args = parser.parse_args()

# ─────────────────────────────────────── AUTO-DETECT WEIGHTS ───────────────
def find_weights():
    candidates = [
        args.weights,
        os.path.join(PROJECT_DIR, "phase3_runs", "yolo26n_ca_50ep", "weights", "best.pt"),
        "/content/phase3_runs/yolo26n_ca_50ep/weights/best.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

weights = find_weights()
if weights is None:
    print("[ERROR] Tidak bisa menemukan best.pt CA model.")
    print("  Jalankan dulu: python phase3_scripts/01_train_ca.py")
    print("  Atau tentukan path dengan: --weights path/to/best.pt")
    sys.exit(1)

# ─────────────────────────────────────── COLLECT IMAGES ────────────────────
source = Path(args.source)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if source.is_file() and source.suffix.lower() in IMG_EXTS:
    image_paths = [source]
elif source.is_dir():
    image_paths = sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS])
else:
    print(f"[ERROR] source '{source}' tidak ditemukan atau bukan gambar/folder.")
    sys.exit(1)

if args.max_images > 0:
    image_paths = image_paths[:args.max_images]

print("=" * 70)
print("  FASE 3 — Inferensi: YOLO26n+CA + Edge Contour Postprocessing")
print("=" * 70)
print(f"\n  Model     : {weights}")
print(f"  Gambar    : {len(image_paths)} file dari '{source}'")
print(f"  Device    : {args.device}")
print(f"  Conf      : {args.conf}")
print(f"\n  Edge Contour Config:")
print(f"    Canny        : {args.canny_low}/{args.canny_high}")
print(f"    Morph Close  : {args.morph_close}×{args.morph_close} kernel")
print(f"    Border Margin: {args.border_margin}px")
print(f"    Min Solidity : {args.min_solidity}")
print(f"    Fallback BBox: {args.fallback}")
print()

# ─────────────────────────────────────── SETUP OUTPUT DIR ──────────────────
if args.save:
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(PROJECT_DIR) / "phase3_runs" / "contour_inference"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {save_dir}\n")

# ─────────────────────────────────────── LOAD MODEL ────────────────────────
print("[1/3] Loading YOLO26n+CA model...")
model = YOLO(weights)

# ─────────────────────────────────────── SETUP PROCESSOR ───────────────────
print("[2/3] Setup Edge Contour Postprocessor...")
cfg = EdgeContourConfig(
    canny_low        = args.canny_low,
    canny_high       = args.canny_high,
    morph_close_ksize= args.morph_close,
    border_margin    = args.border_margin,
    min_solidity     = args.min_solidity,
    fallback_to_bbox = args.fallback,
)
proc = EdgeContourPostProcessor(config=cfg)

# ─────────────────────────────────────── INFERENSI ─────────────────────────
print(f"[3/3] Inferensi {len(image_paths)} gambar...\n")

# Statistik global
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
        print(f"  [SKIP] {img_path.name} — gagal dimuat")
        continue

    t_start = time.time()

    # 1. YOLO deteksi
    yolo_preds = model.predict(
        str(img_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False
    )[0]

    # 2. Konversi ke Detection
    detections = yolo_result_to_detections(yolo_preds)

    # 3. Edge Contour Post-processing
    contour_results = proc.process_detections(img, detections)

    # 4. Statistik
    summary = proc.summarize(contour_results)
    global_stats["total_detections"]  += summary["total_detections"]
    global_stats["complete_leaves"]   += summary["complete_leaves"]
    global_stats["suppressed_leaves"] += summary["suppressed_leaves"]
    for res in contour_results:
        cname = res.detection.class_name
        if res.is_complete:
            global_stats["per_class_complete"][cname] = \
                global_stats["per_class_complete"].get(cname, 0) + 1
        else:
            global_stats["per_class_suppressed"][cname] = \
                global_stats["per_class_suppressed"].get(cname, 0) + 1

    t_elapsed = time.time() - t_start
    global_stats["processing_time_s"].append(t_elapsed)

    # 5. Visualisasi
    vis = proc.draw_results(
        img.copy(),
        contour_results,
        show_suppressed=args.show_suppressed,
        show_bbox=args.show_bbox,
    )

    # 6. Simpan / tampilkan
    if args.save:
        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), vis)

    if args.show:
        cv2.imshow("YOLO26n+CA + Edge Contour", vis)
        key = cv2.waitKey(0 if source.is_file() else 1)
        if key == ord('q'):
            break

    # Progress
    if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths) or source.is_file():
        print(f"  [{idx+1:4d}/{len(image_paths)}] {img_path.name:<35} "
              f"det={summary['total_detections']:2d} "
              f"utuh={summary['complete_leaves']:2d} "
              f"buang={summary['suppressed_leaves']:2d} "
              f"({t_elapsed*1000:.0f}ms)")

if args.show:
    cv2.destroyAllWindows()

# ─────────────────────────────────────── RINGKASAN ─────────────────────────
t_total = time.time() - t_total_start
n_imgs  = len(image_paths)
n_det   = global_stats["total_detections"]
n_comp  = global_stats["complete_leaves"]
n_supp  = global_stats["suppressed_leaves"]
avg_fps = n_imgs / (t_total + 1e-6)
avg_ms  = np.mean(global_stats["processing_time_s"]) * 1000

print(f"\n{'=' * 70}")
print(f"  RINGKASAN INFERENSI — YOLO26n+CA + Edge Contour")
print(f"{'=' * 70}")
print(f"  Gambar diproses   : {n_imgs}")
print(f"  Total deteksi     : {n_det}")
print(f"  Daun UTUH         : {n_comp}  ({n_comp/max(n_det,1)*100:.1f}%)")
print(f"  Daun DIBUANG      : {n_supp} ({n_supp/max(n_det,1)*100:.1f}%)")
print(f"\n  Throughput        : {avg_fps:.2f} FPS | {avg_ms:.0f} ms/gambar")

print(f"\n  Per-Kelas (Utuh / Dibuang):")
print(f"  {'Kelas':<25} {'Utuh':>6} {'Dibuang':>8}")
print("  " + "-" * 40)
for cname in CLASS_NAMES:
    c = global_stats["per_class_complete"].get(cname, 0)
    s = global_stats["per_class_suppressed"].get(cname, 0)
    print(f"  {cname:<25} {c:>6} {s:>8}")

if args.save_stats:
    stats_path = (save_dir if args.save else Path(PROJECT_DIR) / "phase3_runs") / "contour_stats.json"
    global_stats["processing_time_s"] = avg_ms  # ringkas
    global_stats["avg_fps"]           = avg_fps
    with open(stats_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    print(f"\n  Statistik disimpan ke: {stats_path}")

if args.save:
    print(f"\n  Gambar output: {save_dir}")

print(f"\n{'=' * 70}")
