"""
FASE 2 - Script 02: Smoke Test
================================
Training YOLO26n 3 epoch untuk memverifikasi pipeline berjalan lancar.
Jika 3 epoch sukses, lanjut ke baseline training.
"""

import os
import time
from ultralytics import YOLO

print("=" * 60)
print("  FASE 2 - Smoke Test (3 epochs)")
print("=" * 60)

DATA_YAML   = "D:/SKRIPSI/tea_yolo/data.yaml"
MODEL_PRETRAINED = "D:/SKRIPSI/yolo26n.pt"  # downloaded dari env_check
PROJECT_DIR  = "D:/SKRIPSI/phase2_runs"
RUN_NAME     = "smoke_test_3ep"

print(f"\nConfig:")
print(f"  data      : {DATA_YAML}")
print(f"  model     : {MODEL_PRETRAINED}")
print(f"  project   : {PROJECT_DIR}/{RUN_NAME}")
print(f"  epochs    : 3")
print(f"  imgsz     : 416  (dikurangi untuk CPU)")
print(f"  batch     : 4    (dikurangi untuk CPU)")
print(f"  workers   : 0    (aman untuk Windows)")
print(f"  device    : cpu")

print(f"\nMulai training...")
t0 = time.time()

model = YOLO(MODEL_PRETRAINED)

results = model.train(
    data       = DATA_YAML,
    epochs     = 3,
    imgsz      = 416,
    batch      = 4,
    workers    = 0,        # 0 = main thread, penting di Windows
    device     = "cpu",
    project    = PROJECT_DIR,
    name       = RUN_NAME,
    exist_ok   = True,
    # Augmentation: pakai default (tidak diubah sesuai aturan)
    # Optimizer: default (SGD)
    verbose    = True,
)

elapsed = time.time() - t0
print(f"\nSmoke test selesai dalam {elapsed/60:.1f} menit")
print(f"  Estimasi waktu 1 epoch   : {elapsed/3/60:.1f} menit")
print(f"  Estimasi waktu 50 epochs : {elapsed/3*50/3600:.1f} jam")
print(f"\nHasil disimpan di: {PROJECT_DIR}/{RUN_NAME}")
print("\nSmoke test SUKSES - siap lanjut ke baseline training!")
