import json
from pathlib import Path
from PIL import Image
import numpy as np

OUTPUT_DIR = Path(r'D:\SKRIPSI\phase1_output')
with open(OUTPUT_DIR / 'audit_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print('=== DAFTAR NEAR-DUPLICATES DI AUDIT REPORT ===')
cross_class_pairs = []
same_class_pairs = []

for group in report['near_duplicates']:
    phash = group['phash']
    files = group['files']
    classes = set(Path(fl).parent.name for fl in files)
    if len(classes) > 1:
        cross_class_pairs.append((phash, files, classes))
    else:
        same_class_pairs.append((phash, files, classes))

print(f"Total kelompok near-duplicate: {len(report['near_duplicates'])}")
print(f"Kelompok beda kelas (cross-class): {len(cross_class_pairs)}")
print(f"Kelompok kelas sama (intra-class): {len(same_class_pairs)}")

print('\n--- DETAIL INVESTIGASI CROSS-CLASS PAIRS ---')
for idx, (phash, files, classes) in enumerate(cross_class_pairs, 1):
    print(f"\n[Kasus {idx}] pHash: {phash} (Classes: {', '.join(classes)})")
    imgs = []
    for fl in files:
        im = Image.open(fl)
        imgs.append((fl, im))
        print(f"  File: {Path(fl).parent.name}/{Path(fl).name} - Size: {im.size} - Mode: {im.mode}")
    
    # Periksa kesamaan piksel antar pasangan
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            fl1, im1 = imgs[i]
            fl2, im2 = imgs[j]
            name1 = f"{Path(fl1).parent.name[:5]}.._{Path(fl1).name}"
            name2 = f"{Path(fl2).parent.name[:5]}.._{Path(fl2).name}"
            if im1.size != im2.size:
                print(f"    -> Ukuran berbeda ({im1.size} vs {im2.size}). Ini BUKAN duplikat identik, melainkan pHash collision (false positive hash) karena siluet/komposisi umum.")
            else:
                arr1 = np.array(im1.convert('RGB'), dtype=np.float32)
                arr2 = np.array(im2.convert('RGB'), dtype=np.float32)
                diff = np.abs(arr1 - arr2)
                mae = np.mean(diff)
                max_diff = np.max(diff)
                diff_pct_large = np.mean(diff > 25) * 100
                print(f"    -> Ukuran sama ({im1.size}). MAE={mae:.2f}, MaxDiff={max_diff:.1f}, Piksel beda > 25: {diff_pct_large:.2f}%")
                if mae < 10.0:
                    print(f"    [PERINGATAN KRITIS] Kedua gambar ini secara visual SANGAT IDENTIK tetapi berada di kelas berbeda!")
                    print(f"       File 1: {fl1}")
                    print(f"       File 2: {fl2}")

print('\n--- DETAIL INTRA-CLASS NEAR DUPLICATES ---')
for idx, (phash, files, classes) in enumerate(same_class_pairs, 1):
    cls = list(classes)[0]
    print(f"[Intra-Class {idx}] Kelas: {cls} ({len(files)} files)")
    for fl in files:
        print(f"   {Path(fl).name}")
