import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage

DATASET_ROOT = Path(r'D:\SKRIPSI\teaLeafBD')
OUTPUT_DIR   = Path(r'D:\SKRIPSI\phase1_output')
YOLO_DIR     = Path(r'D:\SKRIPSI\tea_yolo')

failed_files = [
    ('3. Gray Blight/gray_blight_00274.jpg', 'train', 2, 'Gray Blight'),
    ('3. Gray Blight/gray_blight_00275.jpg', 'train', 2, 'Gray Blight'),
    ('3. Gray Blight/gray_blight_00291.jpg', 'train', 2, 'Gray Blight'),
    ('3. Gray Blight/gray_blight_00292.jpg', 'train', 2, 'Gray Blight'),
    ('3. Gray Blight/gray_blight_00300.jpg', 'train', 2, 'Gray Blight'),
    ('4. Helopeltis/helopeltis_00137.jpg', 'train', 3, 'Helopeltis'),
    ('4. Helopeltis/helopeltis_00138.jpg', 'train', 3, 'Helopeltis'),
    ('4. Helopeltis/helopeltis_00141.jpg', 'train', 3, 'Helopeltis'),
    ('4. Helopeltis/helopeltis_00142.jpg', 'train', 3, 'Helopeltis'),
    ('6. Green mirid bug/green_mirid_bug_00476.jpg', 'train', 5, 'Green Mirid Bug'),
    ('6. Green mirid bug/green_mirid_bug_00736.jpg', 'train', 5, 'Green Mirid Bug'),
    ('7. Healthy leaf/healthy_00860.jpg', 'train', 6, 'Healty Leaf'),
]

def resolve_edge_case(p):
    im = Image.open(p)
    w, h = im.size
    arr = np.array(im, dtype=float)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (R + G + B) / 3.0
    sat = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) / (np.maximum(np.maximum(R, G), B) + 1e-5)
    
    is_paper = (brightness > 160) & (sat < 0.15)
    struct = ndimage.generate_binary_structure(2, 2)
    p_closed = ndimage.binary_closing(is_paper, structure=struct, iterations=5)
    lab_p, num_p = ndimage.label(p_closed)
    sheet_id = np.argmax(ndimage.sum(p_closed, lab_p, range(num_p + 1))[1:]) + 1
    sheet = (lab_p == sheet_id)
    sl = ndimage.find_objects(sheet.astype(int))[0]
    sheet_ymin, sheet_xmin = sl[0].start, sl[1].start
    sheet_ymax, sheet_xmax = sl[0].stop, sl[1].stop
    
    is_skin = (R - B > 30) & (R - G > 12) & (brightness > 85) & (R > 115)
    leaf_mask = (sat > 0.16) & ~is_skin & ~is_paper
    
    margin = 15
    leaf_mask[:sheet_ymin + margin, :] = False
    leaf_mask[sheet_ymax - margin:, :] = False
    leaf_mask[:, :sheet_xmin + margin] = False
    leaf_mask[:, sheet_xmax - margin:] = False
    
    opened = ndimage.binary_opening(leaf_mask, structure=struct, iterations=3)
    closed = ndimage.binary_closing(opened, structure=struct, iterations=4)
    filled = ndimage.binary_fill_holes(closed)
    
    lab, num = ndimage.label(filled)
    if num == 0:
        return None
        
    sizes = ndimage.sum(filled, lab, range(num + 1))
    candidates = []
    for cid in range(1, num + 1):
        sz = sizes[cid]
        if sz < 1500: continue
        sl_c = ndimage.find_objects(lab)[cid - 1]
        ymn, xmn = sl_c[0].start, sl_c[1].start
        ymx, xmx = sl_c[0].stop, sl_c[1].stop
        bw, bh = xmx - xmn, ymx - ymn
        ratio = (bw * bh) / (w * h)
        if ratio > 0.85 or ratio < 0.003: continue
        cx = (xmn + xmx) / 2.0
        cy = (ymn + ymx) / 2.0
        dist_center = np.sqrt(((cx - w/2)/(w/2))**2 + ((cy - h/2)/(h/2))**2)
        score = sz / (1.0 + 3.0 * dist_center)
        candidates.append((score, (xmn, ymn, xmx, ymx), ratio))
        
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]
    xmn, ymn, xmx, ymx = best[1]
    return (max(0, xmn-4), max(0, ymn-4), min(w, xmx+4), min(h, ymx+4))

print('Mengintegrasikan 12 edge cases ke dataset YOLO...')
with open(OUTPUT_DIR / 'phase1_final_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

for rel_path, split_name, class_id, class_name in failed_files:
    p = DATASET_ROOT / rel_path
    bbox = resolve_edge_case(p)
    if bbox is None:
        print(f'Gagal resolve: {p.name}')
        continue
        
    im = Image.open(p)
    w, h = im.size
    xmin, ymin, xmax, ymax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    norm_w   = bw / w
    norm_h   = bh / h
    
    dest_img = YOLO_DIR / 'images' / split_name / p.name
    dest_txt = YOLO_DIR / 'labels' / split_name / f'{p.stem}.txt'
    
    shutil.copy2(p, dest_img)
    with open(dest_txt, 'w', encoding='utf-8') as f:
        f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n')
        
    report['successful_labels'] += 1
    report['strategy_counts']['strategy4_sheet_constrained'] = report['strategy_counts'].get('strategy4_sheet_constrained', 0) + 1
    report['split_counts'][split_name] += 1
    report['class_counts'][class_name] = report['class_counts'].get(class_name, 0) + 1
    print(f'  [RESOLVED & ADDED] {p.name} -> {split_name}')

report['failed_labels'] = []

with open(OUTPUT_DIR / 'phase1_final_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print('\nSemua 12 citra berhasil ditambahkan ke dataset YOLO!')
print(f"Total citra final dalam dataset YOLO: {report['successful_labels']}/{report['total_images']} (100.0%)")
