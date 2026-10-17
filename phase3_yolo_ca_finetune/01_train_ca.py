"""
FASE 3 (FINE-TUNING EDITION) - Script 01: Two-Stage Training + Precision Fine-Tuning YOLO26n + CA
=================================================================================================
Strategi Pelatihan 2-Tahap (Two-Stage Precision-Pushing Pipeline):

- **TAHAP 1 (Stage-1 Base Training — 50 Epoch)**:
  Melatih YOLO26n + 3x Coordinate Attention (CA) dari bobot `yolo26n.pt` hasil remapping
  selama 50 epoch pada resolusi 640x640 untuk membangun fitur deteksi dasar yang kuat.
  *(Catatan: Jika `best.pt` dari Tahap 1 sudah tersedia di Colab/Drive, skrip otomatis
  melompati Tahap 1 dan langsung masuk ke Tahap 2 Fine-Tuning kecuali diberi flag `--from_scratch`)*.

- **TAHAP 2 (Stage-2 Precision Fine-Tuning — 30 Epoch)**:
  Melakukan *Fine-Tuning* khusus untuk **mendongkrak Precision (menekan False Positive &
  Wrong Classification antar penyakit daun teh)** menggunakan:
  1. **Bobot Klasifikasi Diperkuat (`cls=1.15`, `box=8.5`)**: Memberikan penalti >2x lebih tegas
     terhadap kesalahan klasifikasi kelas penyakit (`Tea algal leaf spot`, `Brown Blight`,
     `Gray Blight`, `Helopeltis`, `Red Spider`) dan meningkatkan ketepatan kotak batas.
  2. **Cosine Annealing Low Learning Rate (`optimizer='AdamW'`, `lr0=0.0005`, `lrf=0.01`, `cos_lr=True`)**:
     Menstabilkan konvergensi bobot Coordinate Attention tanpa merusak fitur yang sudah dipelajari.
  3. **Whole-Leaf Clean Fine-Tuning (`mosaic=0.0`, `close_mosaic=30`, `mixup=0.0`)**:
     Mematikan augmentasi Mosaic agar model belajar dari morfologi daun utuh (tanpa potongan
     artifisial), ditambah `erasing=0.15` untuk melatih ketahanan terhadap oklusi ranting kecil.
  4. **Stem Freezing (`freeze=4`)**: Membekukan 4 layer awal (stem konvolusi dasar) agar fokus
     pembaruan gradien terpusat pada blok P3/P4/P5, 3x `CoordAttYOLO`, Neck, dan Head `Detect`.
"""

import os
import sys
import time
import shutil
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from phase3_yolo_ca_finetune.ca_module import (
    register_ca_module,
    build_yolo26n_ca_pretrained,
    prepare_finetune_checkpoint,
    find_stage1_weights,
)
register_ca_module()

from ultralytics import YOLO  # noqa: E402

parser = argparse.ArgumentParser(description="FASE 3: Two-Stage Training + Precision Fine-Tuning YOLO26n+CA")
parser.add_argument("--device", default="cuda", help="cuda | cpu | 0")
parser.add_argument("--stage1_epochs", type=int, default=50, help="Jumlah epoch Stage-1 (Base Training)")
parser.add_argument("--finetune_epochs", type=int, default=30, help="Jumlah epoch Stage-2 (Precision Fine-Tuning)")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--stage1_name", default="yolo26n_ca_50ep")
parser.add_argument("--run_name", default="yolo26n_ca_finetune")
parser.add_argument("--stage1_weights", default=None, help="Path eksplisit ke best.pt Stage-1 (opsional)")
parser.add_argument("--from_scratch", action="store_true", help="Paksa jalankan Stage-1 dari awal meskipun best.pt sudah ada")
parser.add_argument("--freeze", type=int, default=4, help="Jumlah layer awal backbone yang di-freeze saat Fine-Tuning")
parser.add_argument("--lr0", type=float, default=0.0005, help="Initial learning rate saat Fine-Tuning")
parser.add_argument("--cls_gain", type=float, default=1.15, help="Classification loss gain untuk mendorong Precision")
parser.add_argument("--box_gain", type=float, default=8.5, help="Box loss gain untuk ketepatan lokalisasi")
args = parser.parse_args()

# Deteksi path Google Colab vs Lokal
if os.path.exists("/content/tea_yolo"):
    DATA_YAML = "/content/tea_yolo/data.yaml"
    BASE_PT = "/content/yolo26n.pt" if os.path.exists("/content/yolo26n.pt") else os.path.join(PROJECT_DIR, "yolo26n.pt")
    MODEL_YAML = os.path.join(SCRIPT_DIR, "yolo26n_ca.yaml")
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
            print("[INFO] Path di data.yaml disesuaikan ke /content/tea_yolo")
    except Exception as e:
        print(f"[WARNING] Gagal menyesuaikan path data.yaml: {e}")
else:
    DATA_YAML = os.path.join(PROJECT_DIR, "tea_yolo", "data.yaml")
    BASE_PT = os.path.join(PROJECT_DIR, "yolo26n.pt")
    MODEL_YAML = os.path.join(SCRIPT_DIR, "yolo26n_ca.yaml")
    RUNS_DIR = os.path.join(PROJECT_DIR, "phase3_runs")

os.makedirs(RUNS_DIR, exist_ok=True)
REMAPPED_PT = os.path.join(RUNS_DIR, "yolo26n_ca_coco_init.pt")
FINETUNE_INIT_PT = os.path.join(RUNS_DIR, "yolo26n_ca_stage1_for_finetune.pt")

print("=" * 76)
print("  FASE 3 (FINE-TUNING EDITION): YOLO26n + Coordinate Attention (CA)")
print("  Target Utama : Mendongkrak Precision & Menekan False Positive / Wrong Class")
print("=" * 76)

# Cek apakah bobot Stage-1 (50 epoch) sudah tersedia
stage1_best = args.stage1_weights if (args.stage1_weights and os.path.exists(args.stage1_weights)) else find_stage1_weights(args.stage1_name)

if not stage1_best or not os.path.exists(stage1_best) or args.from_scratch:
    print(f"\n[TAHAP 1/2] Menjalankan Base Training YOLO26n+CA ({args.stage1_epochs} Epoch)...")
    build_yolo26n_ca_pretrained(
        base_pt_path=BASE_PT,
        ca_yaml_path=MODEL_YAML,
        out_pt_path=REMAPPED_PT,
        seed=42,
    )
    model_s1 = YOLO(MODEL_YAML)
    t_s1 = time.time()
    res_s1 = model_s1.train(
        data=DATA_YAML,
        pretrained=REMAPPED_PT,
        epochs=args.stage1_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=RUNS_DIR,
        name=args.stage1_name,
        exist_ok=True,
        patience=15,
        save_period=10,
        seed=42,
        verbose=True,
        plots=True,
    )
    s1_target_dir = os.path.join(RUNS_DIR, args.stage1_name)
    s1_actual_dir = str(getattr(res_s1, "save_dir", s1_target_dir))
    if os.path.abspath(s1_actual_dir) != os.path.abspath(s1_target_dir) and os.path.exists(s1_actual_dir):
        shutil.copytree(s1_actual_dir, s1_target_dir, dirs_exist_ok=True)
    stage1_best = os.path.join(s1_target_dir, "weights", "best.pt")
    print(f"[TAHAP 1 SELESAI] Waktu: {(time.time() - t_s1)/60:.1f} menit | Bobot: {stage1_best}")
else:
    print(f"\n[TAHAP 1 DITEMUKAN] Menggunakan bobot Stage-1 yang sudah ada:")
    print(f"                    {stage1_best}")
    print("                    (Langsung melanjutkan ke Tahap 2: Precision Fine-Tuning!)")

# ── TAHAP 2: PRECISION FINE-TUNING ────────────────────────────────────────────
print("\n" + "=" * 76)
print(f"  [TAHAP 2/2] Memulai PRECISION FINE-TUNING ({args.finetune_epochs} Epoch)")
print(f"  - Optimizer & LR : AdamW | lr0={args.lr0} -> lrf=0.01 (Cosine Annealing)")
print(f"  - Loss Gains     : cls={args.cls_gain} (Penajaman Klasifikasi) | box={args.box_gain}")
print(f"  - Augmentasi     : mosaic=0.0 (100% Daun Utuh) | mixup=0.0 | erasing=0.15")
print(f"  - Layer Freeze   : freeze={args.freeze} (Stem dasar dikunci, CA + Neck + Head di-finetune)")
print("=" * 76)

prepare_finetune_checkpoint(
    stage1_pt_path=stage1_best,
    ca_yaml_path=MODEL_YAML,
    out_pt_path=FINETUNE_INIT_PT,
)

model_ft = YOLO(MODEL_YAML)
t_ft = time.time()

results_ft = model_ft.train(
    data=DATA_YAML,
    pretrained=FINETUNE_INIT_PT,
    epochs=args.finetune_epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    device=args.device,
    project=RUNS_DIR,
    name=args.run_name,
    exist_ok=True,
    optimizer="AdamW",
    lr0=args.lr0,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=2.0,
    weight_decay=0.0008,
    box=args.box_gain,
    cls=args.cls_gain,
    mosaic=0.0,
    close_mosaic=args.finetune_epochs,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.15,
    hsv_h=0.012,
    hsv_s=0.55,
    hsv_v=0.35,
    scale=0.35,
    fliplr=0.5,
    freeze=args.freeze,
    patience=15,
    save_period=10,
    seed=42,
    verbose=True,
    plots=True,
)

elapsed_ft = time.time() - t_ft
target_run_dir = os.path.join(RUNS_DIR, args.run_name)
actual_save_dir = str(getattr(results_ft, "save_dir", target_run_dir))
if os.path.abspath(actual_save_dir) != os.path.abspath(target_run_dir) and os.path.exists(actual_save_dir):
    shutil.copytree(actual_save_dir, target_run_dir, dirs_exist_ok=True)
    print(f"[INFO] Menyinkronkan output fine-tuning dari {actual_save_dir} -> {target_run_dir}")

# Salin juga ke yolo26n_ca_50ep/weights/best_finetuned.pt sebagai cadangan
ft_best_pt = os.path.join(target_run_dir, "weights", "best.pt")

print(f"\n{'=' * 76}")
print("  [SELESAI] PRECISION FINE-TUNING YOLO26n + CA BERHASIL!")
print(f"  Total waktu Fine-Tune : {elapsed_ft/60:.1f} menit")
print(f"  Best Fine-Tuned Model : {ft_best_pt}")
print(f"{'=' * 76}")

rd = results_ft.results_dict
print("\n  Metrik Akhir Validation Set (Setelah Precision Fine-Tuning):")
print(f"    Precision (val): {rd.get('metrics/precision(B)', 0.0):.4f}")
print(f"    Recall    (val): {rd.get('metrics/recall(B)', 0.0):.4f}")
print(f"    mAP@0.5        : {rd.get('metrics/mAP50(B)', 0.0):.4f}")
print(f"    mAP@0.5:0.95   : {rd.get('metrics/mAP50-95(B)', 0.0):.4f}")
print("\n  >> Lanjut ke: python phase3_yolo_ca_finetune/02_evaluate_testset.py")

