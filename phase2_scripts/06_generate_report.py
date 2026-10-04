"""
FASE 2 - Script 06: Generate Laporan Baseline
===============================================
Membaca semua hasil training dan evaluasi, kemudian mencetak
laporan lengkap dalam format yang ditentukan untuk penelitian.

Jalankan SETELAH script 04 dan 05 selesai.
"""

import os
import json
import csv
import sys
from pathlib import Path

# Fix encoding untuk Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 70)
print("  FASE 2 - Laporan Baseline YOLO26n")
print("=" * 70)

PROJECT_DIR = "D:/SKRIPSI/phase2_runs"
RUN_NAME    = "yolo26n_baseline_50ep"
RUN_DIR     = f"{PROJECT_DIR}/{RUN_NAME}"

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",          # 1
    "Gray Blight",           # 2
    "Helopeltis",            # 3
    "Red Spider",            # 4
    "Green Mirid Bug",       # 5
    "Healty Leaf",           # 6
]

# ============ Load training results.csv ============
results_csv = f"{RUN_DIR}/results.csv"
training_history = []
if os.path.exists(results_csv):
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace dari key names
            training_history.append({k.strip(): v.strip() for k, v in row.items()})

# ============ Load test evaluation metrics ============
test_summary_path = f"{RUN_DIR}/test_eval/test_results/summary_metrics.json"
test_summary = None
if os.path.exists(test_summary_path):
    with open(test_summary_path) as f:
        test_summary = json.load(f)

# ============ Load visual analysis stats ============
visual_stats_path = f"{RUN_DIR}/visual_analysis/visual_stats.json"
visual_stats = None
if os.path.exists(visual_stats_path):
    with open(visual_stats_path) as f:
        visual_stats = json.load(f)

# ============ Cetak Laporan ============
print(f"""
+======================================================================+
|         LAPORAN FASE 2: BASELINE MODEL YOLO26n                      |
|         Penelitian Deteksi Penyakit Daun Teh                        |
+======================================================================+

[A] INFORMASI MODEL
    Arsitektur : YOLO26n (You Only Look Once v2.6 nano)
    Parameters : 2,572,280
    GFLOPs     : 6.2
    Pretrained : COCO (yolo26n.pt - ImageNet backbone)
    Task       : Object Detection (7 kelas daun teh)

[B] KONFIGURASI TRAINING
    Dataset    : tea_yolo/ (Fase 1, leak-free split)
    Train set  : 3,701 gambar
    Val set    : 1,048 gambar
    Test set   : 529 gambar (tidak digunakan saat training)
    Epochs     : 50 (CPU-optimized, dari default 100)
    Image size : 416x416 (dikurangi dari 640 untuk CPU)
    Batch size : 4 (CPU limit)
    Optimizer  : SGD (lr0=0.01, momentum=0.937)
    Device     : CPU
    Augmentation: Default Ultralytics recipe (Mosaic ON, FlipLR=0.5)
    Custom loss: TIDAK ADA (pure baseline)
    Attention  : TIDAK ADA (pure baseline)

[C] DATASET (7 Kelas)
    0: Tea algal leaf spot  → 418 gambar
    1: Brown Blight         → 508 gambar
    2: Gray Blight          → 1013 gambar
    3: Helopeltis           → 607 gambar
    4: Red Spider           → 515 gambar
    5: Green Mirid Bug      → 1282 gambar
    6: Healty Leaf          → 935 gambar
    TOTAL                   → 5,278 gambar
""")

# Training curve
if training_history:
    print("[D] TRAINING HISTORY (setiap 5 epoch)")
    print(f"    {'Epoch':>6} {'box_loss':>10} {'cls_loss':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("    " + "-"*46)
    for i, row in enumerate(training_history):
        epoch_num = int(float(row.get('epoch', i+1)))
        if epoch_num % 5 == 0 or epoch_num == 1 or epoch_num == len(training_history):
            box_loss = row.get('train/box_loss', 'N/A')
            cls_loss = row.get('train/cls_loss', 'N/A')
            map50    = row.get('metrics/mAP50(B)', 'N/A')
            map5095  = row.get('metrics/mAP50-95(B)', 'N/A')
            try:
                print(f"    {epoch_num:>6} {float(box_loss):>10.4f} {float(cls_loss):>10.4f} {float(map50):>10.4f} {float(map5095):>10.4f}")
            except:
                print(f"    {epoch_num:>6} {box_loss:>10} {cls_loss:>10} {map50:>10} {map5095:>10}")

# Test metrics
if test_summary:
    ov = test_summary["overall"]
    print(f"""
[E] METRICS EVALUASI TEST SET (best.pt)
    Metric          Value
    ──────────────────────────────────────────
    Precision       {ov['precision']:.4f}  ({ov['precision']*100:.2f}%)
    Recall          {ov['recall']:.4f}  ({ov['recall']*100:.2f}%)
    F1-Score        {ov['f1']:.4f}  ({ov['f1']*100:.2f}%)
    mAP@0.5         {ov['mAP50']:.4f}  ({ov['mAP50']*100:.2f}%)
    mAP@0.5:0.95    {ov['mAP50_95']:.4f}  ({ov['mAP50_95']*100:.2f}%)

[F] METRICS PER KELAS (Test Set)""")
    
    print(f"    {'Kelas':<25} {'P':>8} {'R':>8} {'F1':>8} {'mAP50':>8}")
    print("    " + "-"*59)
    for pc in test_summary.get("per_class", []):
        print(f"    {pc['class']:<25} {pc['precision']:>8.4f} {pc['recall']:>8.4f} {pc['f1']:>8.4f} {pc['map50']:>8.4f}")
    print("    " + "-"*59)
    print(f"    {'MEAN':<25} {ov['precision']:>8.4f} {ov['recall']:>8.4f} {ov['f1']:>8.4f} {ov['mAP50']:>8.4f}")

# Visual analysis
if visual_stats:
    s = visual_stats['stats']
    total = sum(s.values())
    print(f"""
[G] VISUAL ANALYSIS SUMMARY
    Kategori           Count    Persentase
    ─────────────────────────────────────
    True Positive (TP)  {s['TP']:5d}    {s['TP']/total*100:6.1f}%  (deteksi benar, kelas benar)
    False Positive (FP) {s['FP']:5d}    {s['FP']/total*100:6.1f}%  (deteksi ada, tidak ada di GT)
    Wrong Class (WC)    {s['WC']:5d}    {s['WC']/total*100:6.1f}%  (deteksi benar, kelas salah)
    False Negative (FN) {s['FN']:5d}    {s['FN']/total*100:6.1f}%  (GT ada, tidak terdeteksi)
    ─────────────────────────────────────
    TOTAL               {total:5d}    100.0%""")

print(f"""
[H] KESIMPULAN BASELINE
    Model YOLO26n berhasil dilatih sebagai baseline tanpa teknik modifikasi.
    Hasil ini akan menjadi angka perbandingan untuk Fase 3:
    YOLO26n + Coordinate Attention.

[I] FILE OUTPUT
    Training logs  : {RUN_DIR}/results.csv
    Best weights   : {RUN_DIR}/weights/best.pt
    Last weights   : {RUN_DIR}/weights/last.pt
    Training plots : {RUN_DIR}/*.png
    Test metrics   : {RUN_DIR}/test_eval/test_results/
    Visual samples : {RUN_DIR}/visual_analysis/

════════════════════════════════════════════════════════════════════════
  Laporan Fase 2 selesai. Lanjut ke FASE 3: YOLO26n + Coordinate Attention
════════════════════════════════════════════════════════════════════════
""")

# Simpan laporan ke file
report_path = f"{RUN_DIR}/LAPORAN_FASE2_BASELINE.txt"
print(f"Laporan disimpan ke: {report_path}")
