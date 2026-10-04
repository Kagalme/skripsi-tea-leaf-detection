"""
FASE 2: MODEL SELECTION - PERSIAPAN BERKAS UNTUK GOOGLE COLAB
=============================================================
Skrip otomatis untuk mengompres dataset tea_yolo/ dan modul model_selection/
menjadi file .ZIP yang siap diunggah ke Google Drive / Google Colab.

Output:
1. D:/SKRIPSI/tea_yolo.zip          (~1.1 GB, dataset gambar & label)
2. D:/SKRIPSI/model_selection.zip    (~50 KB, kode arsitektur & pipeline)
"""

import os
import sys
import zipfile
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "tea_yolo")
SCRIPTS_DIR = os.path.join(BASE_DIR, "model_selection")

ZIP_DATASET = os.path.join(BASE_DIR, "tea_yolo.zip")
ZIP_SCRIPTS = os.path.join(BASE_DIR, "model_selection.zip")


def zip_directory(source_dir, output_zip, filter_ext=None):
    """Mengompres seluruh isi direktori ke file ZIP dengan progress tracker."""
    print(f"\n[KOMPRESI] Memulai pengemasan: {os.path.basename(source_dir)} -> {os.path.basename(output_zip)}")
    t0 = time.time()

    all_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if filter_ext and not f.lower().endswith(filter_ext):
                continue
            all_files.append(os.path.join(root, f))

    total = len(all_files)
    print(f"  Total berkas yang akan dikompres: {total:,} file")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for idx, file_path in enumerate(all_files, 1):
            arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
            zipf.write(file_path, arcname)
            if idx % 500 == 0 or idx == total:
                pct = (idx / total) * 100
                print(f"  Progress: [{idx:,}/{total:,}] ({pct:5.1f}%)", end="\r", flush=True)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n  [SELESAI] File ZIP tersimpan: {output_zip} ({size_mb:.2f} MB, waktu: {elapsed:.1f}s)")


def main():
    print("=" * 75)
    print("  FASE 2: PENGEMASAN BERKAS UNTUK GOOGLE COLAB (GPU T4)")
    print("=" * 75)

    # 1. Kompres model_selection (kode)
    if os.path.exists(SCRIPTS_DIR):
        zip_directory(SCRIPTS_DIR, ZIP_SCRIPTS)
    else:
        print(f"[ERROR] Folder {SCRIPTS_DIR} tidak ditemukan!")

    # 2. Kompres tea_yolo (dataset)
    if os.path.exists(DATASET_DIR):
        zip_directory(DATASET_DIR, ZIP_DATASET)
    else:
        print(f"[ERROR] Folder {DATASET_DIR} tidak ditemukan!")

    print("\n" + "=" * 75)
    print("  RINGKASAN BERKAS SIAP UNGGAH KE GOOGLE DRIVE:")
    print("=" * 75)
    print(f"  1. {ZIP_DATASET}")
    print(f"  2. {ZIP_SCRIPTS}")
    print("\n  Langkah selanjutnya:")
    print("  1. Buka Google Drive Anda (drive.google.com)")
    print("  2. Buat folder baru bernama: 'SKRIPSI'")
    print("  3. Unggah kedua file .zip di atas ke dalam folder 'SKRIPSI' di Google Drive.")
    print("  4. Buka Google Colab dan jalankan notebook 'Training_Model_Selection_Colab.ipynb'!")
    print("=" * 75)


if __name__ == "__main__":
    main()
