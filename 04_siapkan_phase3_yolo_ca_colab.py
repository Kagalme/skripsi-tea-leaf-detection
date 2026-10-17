"""
FASE 3 (DIPERBARUI) - Skrip Pengemas Otomatis phase3_yolo_ca.zip
================================================================
Menghasilkan: D:\\SKRIPSI\\phase3_yolo_ca.zip
Berisi       : Seluruh kode Fase 3 baru (phase3_yolo_ca/) yang telah diperbaiki.

Jalankan dari D:\\SKRIPSI:
    python 04_siapkan_phase3_yolo_ca_colab.py
"""

import os
import zipfile
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "phase3_yolo_ca")
OUTPUT_ZIP = os.path.join(SCRIPT_DIR, "phase3_yolo_ca.zip")

EXCLUDE_DIRS = {"__pycache__", ".git"}
EXCLUDE_EXTS = {".pyc", ".pyo"}

print("=" * 65)
print("  FASE 3 (v2) - Pengemas Otomatis phase3_yolo_ca.zip")
print("=" * 65)

if not os.path.exists(TARGET_DIR):
    print(f"[ERROR] Folder tidak ditemukan: {TARGET_DIR}")
    raise SystemExit(1)

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

size_kb = os.path.getsize(OUTPUT_ZIP) / 1024
print(f"\n{'=' * 65}")
print(f"  [SUKSES] phase3_yolo_ca.zip ({size_kb:.1f} KB)")
print(f"  Lokasi : {OUTPUT_ZIP}")
print(f"{'=' * 65}")

