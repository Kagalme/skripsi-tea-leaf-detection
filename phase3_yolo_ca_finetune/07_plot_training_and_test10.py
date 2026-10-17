"""
FASE 3 (DIPERBARUI) - Script 07: Visualisasi Grafik Training & 10 Hasil Testing
===============================================================================
Menghasilkan:
1. Grafik proses training resolusi tinggi dari `results.csv` (Box Loss, Cls Loss,
   Precision, Recall, mAP@0.5, dan mAP@0.5:0.95 per epoch) -> `training_curves_hd.png`
2. Grid 10 Hasil Testing (`10_test_predictions_grid.png`) yang menampilkan 10 sampel
   citra Test Set lintas kelas lengkap dengan label Ground Truth (GT), prediksi
   YOLO26n+CA, confidence score, dan polygon kontur daun.
"""

import os
import sys
import csv
import glob
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_yolo_ca_finetune.ca_module import register_ca_module, find_best_weights
register_ca_module()

from ultralytics import YOLO  # noqa: E402
from phase3_yolo_ca_finetune.edge_contour import (
    EdgeContourPostProcessor,
    EdgeContourConfig,
    yolo_result_to_detections,
    CLASS_NAMES,
)

parser = argparse.ArgumentParser(description="FASE 3 (v2): Plot Grafik Training & 10 Hasil Testing")
parser.add_argument("--run_name", default="yolo26n_ca_finetune")
parser.add_argument("--device", default="cuda")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--num_samples", type=int, default=10)
args = parser.parse_args()

if os.path.exists("/content/tea_yolo"):
    RUNS_DIR = Path("/content/phase3_runs")
    TEST_IMG_DIR = Path("/content/tea_yolo/images/test")
    TEST_LBL_DIR = Path("/content/tea_yolo/labels/test")
else:
    RUNS_DIR = Path(PROJECT_DIR) / "phase3_runs"
    TEST_IMG_DIR = Path(PROJECT_DIR) / "tea_yolo" / "images" / "test"
    TEST_LBL_DIR = Path(PROJECT_DIR) / "tea_yolo" / "labels" / "test"

RUN_DIR = RUNS_DIR / args.run_name
RUN_DIR.mkdir(parents=True, exist_ok=True)
BEST_WEIGHTS = Path(find_best_weights(args.run_name))
RESULTS_CSV = BEST_WEIGHTS.parent.parent / "results.csv" if BEST_WEIGHTS.exists() else (RUN_DIR / "results.csv")
OUT_CURVES = RUN_DIR / "training_curves_hd.png"
OUT_GRID10 = RUN_DIR / "10_test_predictions_grid.png"


def plot_training_curves():
    """Membaca results.csv dan mem-plot kurva proses training 6-panel yang rapi."""
    if not RESULTS_CSV.exists():
        print(f"[WARNING] {RESULTS_CSV} tidak ditemukan, melewati pembuatan grafik kustom.")
        return None

    epochs, box_loss, cls_loss, val_box, val_cls = [], [], [], [], []
    prec, rec, map50, map50_95 = [], [], [], []

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items() if k}
            try:
                epochs.append(int(float(clean.get("epoch", len(epochs) + 1))))
                box_loss.append(float(clean.get("train/box_loss", 0.0)))
                cls_loss.append(float(clean.get("train/cls_loss", 0.0)))
                val_box.append(float(clean.get("val/box_loss", 0.0)))
                val_cls.append(float(clean.get("val/cls_loss", 0.0)))
                prec.append(float(clean.get("metrics/precision(B)", 0.0)))
                rec.append(float(clean.get("metrics/recall(B)", 0.0)))
                map50.append(float(clean.get("metrics/mAP50(B)", 0.0)))
                map50_95.append(float(clean.get("metrics/mAP50-95(B)", 0.0)))
            except ValueError:
                continue

    if not epochs:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Grafik Proses Training YOLO26n + Coordinate Attention ({len(epochs)} Epoch, 640×640)",
        fontsize=15,
        fontweight="bold",
    )

    # 1. Box Loss
    axes[0, 0].plot(epochs, box_loss, label="Train Box Loss", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, val_box, label="Val Box Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("1. Bounding Box Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Classification Loss
    axes[0, 1].plot(epochs, cls_loss, label="Train Cls Loss", color="#d62728", lw=2)
    axes[0, 1].plot(epochs, val_cls, label="Val Cls Loss", color="#9467bd", lw=2, linestyle="--")
    axes[0, 1].set_title("2. Classification Loss", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Precision & Recall
    axes[0, 2].plot(epochs, prec, label="Precision", color="#2ca02c", lw=2)
    axes[0, 2].plot(epochs, rec, label="Recall", color="#e377c2", lw=2)
    axes[0, 2].set_title("3. Validation Precision & Recall", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Score")
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # 4. mAP@0.50
    best_map50_idx = int(np.argmax(map50))
    axes[1, 0].plot(epochs, map50, label="mAP@0.50", color="#17becf", lw=2.5)
    axes[1, 0].scatter(
        [epochs[best_map50_idx]],
        [map50[best_map50_idx]],
        color="red",
        zorder=5,
        label=f"Peak: {map50[best_map50_idx]*100:.2f}% (Ep {epochs[best_map50_idx]})",
    )
    axes[1, 0].set_title("4. Kurva Konvergensi mAP@0.50", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP@0.50")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 5. mAP@0.50:0.95
    best_map95_idx = int(np.argmax(map50_95))
    axes[1, 1].plot(epochs, map50_95, label="mAP@0.50:0.95", color="#bcbd22", lw=2.5)
    axes[1, 1].scatter(
        [epochs[best_map95_idx]],
        [map50_95[best_map95_idx]],
        color="red",
        zorder=5,
        label=f"Peak: {map50_95[best_map95_idx]*100:.2f}% (Ep {epochs[best_map95_idx]})",
    )
    axes[1, 1].set_title("5. Kurva Konvergensi mAP@0.50:0.95", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP@0.50:0.95")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # 6. F1-Score per Epoch
    f1_curve = [2 * p * r / (p + r + 1e-9) for p, r in zip(prec, rec)]
    best_f1_idx = int(np.argmax(f1_curve))
    axes[1, 2].plot(epochs, f1_curve, label="Validation F1-Score", color="#8c564b", lw=2.5)
    axes[1, 2].scatter(
        [epochs[best_f1_idx]],
        [f1_curve[best_f1_idx]],
        color="red",
        zorder=5,
        label=f"Peak: {f1_curve[best_f1_idx]*100:.2f}% (Ep {epochs[best_f1_idx]})",
    )
    axes[1, 2].set_title("6. Kurva F1-Score", fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("F1-Score")
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(str(OUT_CURVES), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafik proses training disimpan di: {OUT_CURVES}")
    return str(OUT_CURVES)


def generate_10_test_predictions():
    """Memilih 10 citra Test Set secara proporsional lintas 7 kelas dan mem-plot hasil deteksi."""
    if not BEST_WEIGHTS.exists():
        print(f"[ERROR] Bobot {BEST_WEIGHTS} tidak ditemukan!")
        return None

    all_test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))
    if not all_test_imgs:
        print(f"[ERROR] Tidak ada gambar test di {TEST_IMG_DIR}")
        return None

    # Kelompokkan berdasarkan kelas ground truth agar ke-7 kelas terwakili di dalam 10 sampel
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for img_p in all_test_imgs:
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        if lbl_p.exists():
            first_line = lbl_p.read_text().strip().splitlines()
            if first_line:
                cid = int(first_line[0].split()[0])
                by_class.setdefault(cid, []).append(img_p)

    random.seed(42)
    selected = []
    # Ambil 1 dari tiap kelas (7 gambar), lalu tambah 3 gambar lagi dari kelas berbeda
    for cid in range(len(CLASS_NAMES)):
        if by_class.get(cid):
            selected.append((cid, by_class[cid][0]))
    for cid in range(len(CLASS_NAMES)):
        if len(selected) >= args.num_samples:
            break
        if len(by_class.get(cid, [])) > 1:
            selected.append((cid, by_class[cid][1]))

    selected = selected[: args.num_samples]

    model = YOLO(str(BEST_WEIGHTS))
    proc = EdgeContourPostProcessor(EdgeContourConfig())

    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(22, 9.5))
    fig.suptitle(
        f"10 Hasil Pengujian Model YOLO26n + Coordinate Attention + Edge Contour pada Test Set",
        fontsize=15,
        fontweight="bold",
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (gt_cid, img_p) in enumerate(selected):
        img_bgr = cv2.imread(str(img_p))
        res = model.predict(str(img_p), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        detections = yolo_result_to_detections(res)
        contour_res = proc.process_detections(img_bgr, detections)
        vis_bgr = proc.draw_results(img_bgr, contour_res, show_suppressed=True, show_bbox=True)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        gt_name = CLASS_NAMES[gt_cid] if gt_cid < len(CLASS_NAMES) else f"Class {gt_cid}"
        if detections:
            top_d = detections[0]
            pred_str = f"Pred: {top_d.class_name} ({top_d.confidence*100:.1f}%)"
            is_correct = (top_d.class_id == gt_cid)
            title_color = "darkgreen" if is_correct else "crimson"
        else:
            pred_str = "Pred: Tidak Terdeteksi"
            title_color = "crimson"

        ax = axes_flat[idx]
        ax.imshow(vis_rgb)
        ax.set_title(f"[{idx+1}] GT: {gt_name}\n{pred_str}", fontsize=9.5, fontweight="bold", color=title_color)
        ax.axis("off")

    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(str(OUT_GRID10), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grid 10 hasil testing disimpan di: {OUT_GRID10}")
    return str(OUT_GRID10)


if __name__ == "__main__":
    print("=" * 72)
    print("  FASE 3 (v2) - Pembuatan Grafik Training & Grid 10 Hasil Testing")
    print("=" * 72)
    plot_training_curves()
    generate_10_test_predictions()
    print("=" * 72)

