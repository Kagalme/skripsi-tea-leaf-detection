import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import sys

DATASET_ROOT = Path(r'D:\SKRIPSI\teaLeafBD')
OUTPUT_DIR   = Path(r'D:\SKRIPSI\phase1_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
    '1. Tea algal leaf spot': 0,
    '2. Brown Blight':        1,
    '3. Gray Blight':         2,
    '4. Helopeltis':          3,
    '5. Red spider':          4,
    '6. Green mirid bug':     5,
    '7. Healthy leaf':        6,
}

try:
    from PIL import Image
    import numpy as np
    print('[OK] PIL dan numpy tersedia')
except ImportError:
    print('[ERROR] pip install pillow numpy')
    sys.exit(1)

try:
    import imagehash
    USE_PHASH = True
    print('[OK] imagehash tersedia')
except ImportError:
    USE_PHASH = False
    print('[WARN] imagehash tidak tersedia - hanya MD5')

def md5_of_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

print('\n' + '='*60)
print('FASE 1 - DATASET AUDIT')
print('='*60)

results = {
    'total_images': 0,
    'per_class': {},
    'resolutions': defaultdict(int),
    'corrupted': [],
    'exact_duplicates': [],
    'near_duplicates': [],
    'unexpected_format': [],
}

md5_registry   = defaultdict(list)
phash_registry = defaultdict(list)
all_images = []
res_list = []

for folder_name, class_id in CLASS_MAPPING.items():
    folder_path = DATASET_ROOT / folder_name
    if not folder_path.exists():
        print(f'[WARN] Folder tidak ditemukan: {folder_path}')
        results['per_class'][folder_name] = {'count': 0, 'class_id': class_id}
        continue

    files = list(folder_path.glob('*'))
    image_files = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
    non_image   = [f for f in files if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]

    results['per_class'][folder_name] = {
        'class_id': class_id,
        'count': len(image_files),
        'non_image_files': [str(f) for f in non_image],
    }
    results['total_images'] += len(image_files)
    print(f'\n[{class_id}] {folder_name} - {len(image_files)} gambar')

    for img_path in image_files:
        if img_path.suffix.lower() not in ('.jpg', '.jpeg'):
            results['unexpected_format'].append(str(img_path))

        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
        except Exception as e:
            results['corrupted'].append({'path': str(img_path), 'error': str(e)})
            print(f'  [CORRUPT] {img_path.name}: {e}')
            continue

        res_key = f'{w}x{h}'
        results['resolutions'][res_key] += 1
        res_list.append((w, h))

        md5 = md5_of_file(img_path)
        md5_registry[md5].append(str(img_path))

        ph = None
        if USE_PHASH:
            ph = str(imagehash.phash(img))
            phash_registry[ph].append(str(img_path))

        all_images.append({
            'path': str(img_path),
            'class': folder_name,
            'class_id': class_id,
            'width': w,
            'height': h,
            'md5': md5,
            'phash': ph,
        })

print('\n' + '='*60)
print('ANALISIS DUPLIKAT')
print('='*60)

exact_dup_groups = {k: v for k, v in md5_registry.items() if len(v) > 1}
for md5, paths in exact_dup_groups.items():
    results['exact_duplicates'].append({'md5': md5, 'files': paths})

print(f'Exact duplicates (MD5): {len(exact_dup_groups)} grup')
for grp in results['exact_duplicates'][:10]:
    print(f"  MD5={grp['md5'][:12]}... -> {len(grp['files'])} file")
    for p in grp['files']:
        print(f'    {p}')

if USE_PHASH:
    phash_dup_groups = {k: v for k, v in phash_registry.items() if len(v) > 1}
    for ph, paths in phash_dup_groups.items():
        results['near_duplicates'].append({'phash': ph, 'files': paths})
    print(f'\nNear-duplicates (pHash): {len(phash_dup_groups)} grup')
    for grp in results['near_duplicates'][:10]:
        print(f"  pHash={grp['phash']} -> {len(grp['files'])} file")
        for p in grp['files']:
            print(f'    {p}')

print('\n' + '='*60)
print('ANALISIS RESOLUSI')
print('='*60)

if res_list:
    widths  = [r[0] for r in res_list]
    heights = [r[1] for r in res_list]
    print(f'Resolusi minimum : {min(widths)} x {min(heights)}')
    print(f'Resolusi maksimum: {max(widths)} x {max(heights)}')
    print(f'Rata-rata lebar  : {sum(widths)/len(widths):.1f}')
    print(f'Rata-rata tinggi : {sum(heights)/len(heights):.1f}')
    print(f'\nDistribusi resolusi unik (top 15):')
    sorted_res = sorted(results['resolutions'].items(), key=lambda x: -x[1])
    for res, count in sorted_res[:15]:
        print(f'  {res:>20s} : {count:>5d} gambar')

print('\n' + '='*60)
print('RINGKASAN AUDIT')
print('='*60)
print(f"Total gambar valid   : {results['total_images']}")
print(f"File rusak (corrupt) : {len(results['corrupted'])}")
print(f"Exact duplicates     : {len(results['exact_duplicates'])} grup")
print(f"Near-duplicates      : {len(results['near_duplicates'])} grup")
print(f"Format non-JPG       : {len(results['unexpected_format'])}")

print('\nJumlah per kelas:')
for folder_name, info in results['per_class'].items():
    print(f"  [{info['class_id']}] {folder_name:35s} : {info['count']:>5d}")

results['resolutions'] = dict(results['resolutions'])

output_json = OUTPUT_DIR / 'audit_report.json'
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

all_images_json = OUTPUT_DIR / 'all_images_metadata.json'
with open(all_images_json, 'w', encoding='utf-8') as f:
    json.dump(all_images, f, indent=2, ensure_ascii=False)

print(f'\n[SAVED] {output_json}')
print(f'[SAVED] {all_images_json}')
print('\nAudit selesai.')
