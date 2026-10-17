"""
FASE 3 (FINE-TUNING EDITION) - Skrip Pengemas Otomatis phase3_yolo_ca_finetune.zip
==================================================================================
Menghasilkan:
1. Menyelesaikan sinkronisasi skrip 05, 06, 07 di `D:\\SKRIPSI\\phase3_yolo_ca_finetune\\`
2. Mengemas `D:\\SKRIPSI\\phase3_yolo_ca_finetune.zip`
3. Membuat Notebook Colab `D:\\SKRIPSI\\Training_Phase3_YOLO26_CA_Finetune_Colab.ipynb`

Jalankan dari D:\\SKRIPSI:
    python 05_siapkan_phase3_finetune_colab.py
"""

import os
import json
import zipfile
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "phase3_yolo_ca")
TARGET_DIR = os.path.join(SCRIPT_DIR, "phase3_yolo_ca_finetune")
OUTPUT_ZIP = os.path.join(SCRIPT_DIR, "phase3_yolo_ca_finetune.zip")
OUTPUT_NB = os.path.join(SCRIPT_DIR, "Training_Phase3_YOLO26_CA_Finetune_Colab.ipynb")

os.makedirs(TARGET_DIR, exist_ok=True)

# 1. Salin & sesuaikan skrip 05, 06, 07 dari phase3_yolo_ca ke phase3_yolo_ca_finetune
for fname in ["05_infer_with_contour.py", "06_test_external_image.py", "07_plot_training_and_test10.py"]:
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(TARGET_DIR, fname)
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("from phase3_yolo_ca.", "from phase3_yolo_ca_finetune.")
    content = content.replace('"yolo26n_ca_50ep"', '"yolo26n_ca_finetune"')
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(content)

# 2. Kemas ke dalam phase3_yolo_ca_finetune.zip
EXCLUDE_DIRS = {"__pycache__", ".git"}
EXCLUDE_EXTS = {".pyc", ".pyo"}

print("=" * 68)
print("  FASE 3 (FINE-TUNING EDITION) - Pengemas phase3_yolo_ca_finetune.zip")
print("=" * 68)

file_list = []
for root, dirs, files in os.walk(TARGET_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        if Path(f).suffix in EXCLUDE_EXTS:
            continue
        full_path = os.path.join(root, f)
        arcname = os.path.relpath(full_path, SCRIPT_DIR).replace("\\", "/")
        file_list.append((full_path, arcname))

print(f"\nMenambahkan {len(file_list)} berkas ke dalam {os.path.basename(OUTPUT_ZIP)}:\n")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for full_path, arcname in file_list:
        zf.write(full_path, arcname)
        print(f"  + {arcname}")

# 3. Buat Notebook Colab khusus Fine-Tuning
src_nb_path = os.path.join(SCRIPT_DIR, "Training_Phase3_YOLO26_CA_Colab.ipynb")
if os.path.exists(src_nb_path):
    with open(src_nb_path, "r", encoding="utf-8") as f:
        nb_txt = f.read()
    nb_txt = nb_txt.replace("phase3_yolo_ca", "phase3_yolo_ca_finetune")
    nb_txt = nb_txt.replace("yolo26n_ca_50ep", "yolo26n_ca_finetune")
    nb_txt = nb_txt.replace("--epochs 50", "--stage1_epochs 50 \\\n    --finetune_epochs 30")
    with open(OUTPUT_NB, "w", encoding="utf-8") as f:
        f.write(nb_txt)

size_kb = os.path.getsize(OUTPUT_ZIP) / 1024
print(f"\n{'=' * 68}")
print(f"  [SUKSES] phase3_yolo_ca_finetune.zip ({size_kb:.1f} KB)")
print(f"  Lokasi ZIP      : {OUTPUT_ZIP}")
print(f"  Lokasi Notebook : {OUTPUT_NB}")
print(f"{'=' * 68}")

