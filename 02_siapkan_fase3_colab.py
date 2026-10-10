"""
FASE 3 - Skrip Pengemas Otomatis untuk Google Colab
=====================================================
Menghasilkan: D:\\SKRIPSI\\phase3_scripts.zip
Berisi       : Seluruh kode Fase 3 (ca_module, YAML, training scripts)

Jalankan dari D:\\SKRIPSI:
    python 02_siapkan_fase3_colab.py
"""

import os
import zipfile
from pathlib import Path

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR   = os.path.join(SCRIPT_DIR, "phase3_scripts")
OUTPUT_ZIP   = os.path.join(SCRIPT_DIR, "phase3_scripts.zip")

EXCLUDE_DIRS  = {"__pycache__", ".git"}
EXCLUDE_EXTS  = {".pyc", ".pyo"}

print("=" * 60)
print("  FASE 3 - Pengemas Otomatis phase3_scripts.zip")
print("=" * 60)

if not os.path.exists(PHASE3_DIR):
    print(f"[ERROR] Folder phase3_scripts tidak ditemukan di:\n  {PHASE3_DIR}")
    exit(1)

file_list = []
for root, dirs, files in os.walk(PHASE3_DIR):
    # Skip excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for f in files:
        if Path(f).suffix in EXCLUDE_EXTS:
            continue
        full_path = os.path.join(root, f)
        # Arsip relatif terhadap SCRIPT_DIR dengan pemisah '/' standar Linux
        arcname = os.path.relpath(full_path, SCRIPT_DIR).replace('\\', '/')
        file_list.append((full_path, arcname))

print(f"\nMenambahkan {len(file_list)} berkas ke ZIP...\n")
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for full_path, arcname in file_list:
        zf.write(full_path, arcname)
        print(f"  + {arcname}")

size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print(f"\n{'=' * 60}")
print(f"  Berhasil! phase3_scripts.zip ({size_mb:.2f} MB)")
print(f"  Lokasi: {OUTPUT_ZIP}")
print(f"{'=' * 60}")
print(f"""
LANGKAH SELANJUTNYA:
  1. Upload ke Google Drive > SKRIPSI/:
       - phase3_scripts.zip     (dibuat tadi)
       - tea_yolo.zip           (sudah ada)
       - yolo26n.pt             (sudah ada)
       - Training_Phase3_CA_Colab.ipynb (di D:\\SKRIPSI)

  2. Buka Training_Phase3_CA_Colab.ipynb di Google Colab.
  3. Jalankan cell satu per satu (Langkah 1 s.d. 7).
  4. Total waktu di GPU T4: ~10–15 menit.
""")
