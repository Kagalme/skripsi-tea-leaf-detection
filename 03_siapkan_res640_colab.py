"""
FASE 2 (RESOLUSI STANDAR 640) - Skrip Pengemas Otomatis untuk Google Colab
============================================================================
Menghasilkan: D:\\SKRIPSI\\phase2_res640_scripts.zip
Berisi       : Seluruh script pelatihan, evaluasi, dan visualisasi resolusi 640.

Jalankan dari D:\\SKRIPSI:
    python 03_siapkan_res640_colab.py
"""

import os
import zipfile
from pathlib import Path

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR   = os.path.join(SCRIPT_DIR, "phase2_res640_scripts")
OUTPUT_ZIP   = os.path.join(SCRIPT_DIR, "phase2_res640_scripts.zip")

EXCLUDE_DIRS  = {"__pycache__", ".git"}
EXCLUDE_EXTS  = {".pyc", ".pyo"}

print("=" * 60)
print("  PENGEMAS OTOMATIS: phase2_res640_scripts.zip")
print("=" * 70)

if not os.path.exists(TARGET_DIR):
    print(f"[ERROR] Folder target tidak ditemukan di:\n  {TARGET_DIR}")
    exit(1)

file_list = []
for root, dirs, files in os.walk(TARGET_DIR):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        if Path(f).suffix in EXCLUDE_EXTS:
            continue
        full_path = os.path.join(root, f)
        arcname = os.path.relpath(full_path, SCRIPT_DIR).replace('\\', '/')
        file_list.append((full_path, arcname))

print(f"\nMenambahkan {len(file_list)} berkas ke ZIP...\n")
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for full_path, arcname in file_list:
        zf.write(full_path, arcname)
        print(f"  + {arcname}")

size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print(f"\n{'=' * 60}")
print(f"  Berhasil! phase2_res640_scripts.zip ({size_mb:.2f} MB)")
print(f"  Lokasi: {OUTPUT_ZIP}")
print(f"{'=' * 60}")
