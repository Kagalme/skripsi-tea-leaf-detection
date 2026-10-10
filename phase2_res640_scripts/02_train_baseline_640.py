"""
FASE 2 (RESOLUSI STANDAR 640) - Script 02: Baseline Training YOLO26n
======================================================================
Melatih model YOLO26n resmi dengan resolusi standar (640x640),
jadwal Cosine Learning Rate, augmentasi rotasi domain daun teh,
dan evaluasi checkpoint otomatis.

Hasil disimpan di: phase2_runs/yolo26n_baseline_640/
"""

import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

# ==================== 1. Deteksi Lingkungan ====================
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

PROJECT_DIR = BASE_DIR / "phase2_runs"
RUN_NAME    = "yolo26n_baseline_640"
DATA_YAML   = BASE_DIR / "tea_yolo/data.yaml"

# Cari pretrained yolo26n.pt
PRETRAINED_CANDIDATES = [
    BASE_DIR / "yolo26n.pt",
    Path("yolo26n.pt"),
]
MODEL_WEIGHTS = "yolo26n.pt"
for cand in PRETRAINED_CANDIDATES:
    if cand.exists():
        MODEL_WEIGHTS = str(cand)
        break

print("=" * 70)
print("  FASE 2 - PELATIHAN BASELINE YOLO26n (RESOLUSI STANDAR 640x640)")
print("=" * 70)
print(f"Platform       : {'Google Colab (GPU)' if IS_COLAB else 'PC Lokal'}")
print(f"Dataset YAML   : {DATA_YAML}")
print(f"Pretrained     : {MODEL_WEIGHTS}")
print(f"Output Folder  : {PROJECT_DIR}/{RUN_NAME}")

if not DATA_YAML.exists():
    print(f"[ERROR] {DATA_YAML} tidak ditemukan!")
    sys.exit(1)

# ==================== 2. Konfigurasi Device & Batch ====================
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    DEVICE = 0
    BATCH_SIZE = 16
    WORKERS = 4 if IS_COLAB else 0
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Akselerasi GPU : [OK] {gpu_name} (Batch: {BATCH_SIZE})")
else:
    DEVICE = "cpu"
    BATCH_SIZE = 4
    WORKERS = 0
    print(f"[PERINGATAN] GPU tidak terdeteksi. Berjalan di CPU (Batch: {BATCH_SIZE}).")
    print("             Disarankan menjalankan script ini di Google Colab GPU.")

# ==================== 3. Sinkronisasi data.yaml ====================
# Memastikan path di dalam data.yaml cocok dengan direktori aktif
try:
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        content = f.read()
    
    target_path_str = f"path: {str(BASE_DIR / 'tea_yolo').replace(chr(92), '/')}"
    lines = content.splitlines()
    updated_lines = []
    has_path = False
    for line in lines:
        if line.strip().startswith("path:"):
            updated_lines.append(target_path_str)
            has_path = True
        else:
            updated_lines.append(line)
    if not has_path:
        updated_lines.insert(0, target_path_str)
    
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")
    print(f"Sinkronisasi data.yaml: [OK] -> {target_path_str}")
except Exception as e:
    print(f"[INFO] Lewati update data.yaml: {e}")

# ==================== 4. Load Model & Mulai Training ====================
print("\n[Memuat Arsitektur YOLO26n]")
model = YOLO(MODEL_WEIGHTS)

EPOCHS = 80
PATIENCE = 20
IMGSZ = 640

print("\nParameter Pelatihan yang Dioptimasi:")
print(f"  • Resolusi Gambar (imgsz) : {IMGSZ} (Standar detail lesi daun)")
print(f"  • Total Epoch             : {EPOCHS}")
print(f"  • Early Stopping Patience : {PATIENCE}")
print(f"  • Batch Size              : {BATCH_SIZE}")
print(f"  • Learning Rate Scheduler : Cosine Annealing (cos_lr=True)")
print(f"  • Augmentasi Rotasi Daun  : degrees=15.0, flipud=0.5, fliplr=0.5")
print(f"  • Device                  : {DEVICE}")
print("-" * 70)

t_start = time.time()

results = model.train(
    data        = str(DATA_YAML),
    epochs      = EPOCHS,
    patience    = PATIENCE,
    batch       = BATCH_SIZE,
    imgsz       = IMGSZ,            # 640x640 resolusi standar
    device      = DEVICE,
    workers     = WORKERS,
    project     = str(PROJECT_DIR),
    name        = RUN_NAME,
    exist_ok    = True,
    
    # Optimasi & Scheduler
    optimizer   = "auto",
    cos_lr      = True,             # Cosine learning rate decay
    lr0         = 0.01,
    lrf         = 0.01,
    warmup_epochs = 3.0,
    
    # Augmentasi Citra Daun Teh
    degrees     = 15.0,             # Rotasi acak daun
    flipud      = 0.5,              # Pembalikan vertikal
    fliplr      = 0.5,              # Pembalikan horizontal
    mosaic      = 1.0,
    close_mosaic= 10,               # Nonaktifkan mosaic 10 epoch terakhir
    
    # Kontrol & Logging
    save        = True,
    save_period = 10,
    plots       = True,
    verbose     = True,
    seed        = 42,
    deterministic = True,
)

t_elapsed = time.time() - t_start
hours = int(t_elapsed // 3600)
minutes = int((t_elapsed % 3600) // 60)

print("\n" + "=" * 70)
print(f"  PELATIHAN SELESAI DALAM {hours} JAM {minutes} MENIT!")
print(f"  Bobot Model Terbaik : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
print(f"  Grafik & Log        : {PROJECT_DIR}/{RUN_NAME}/")
print("=" * 70)
