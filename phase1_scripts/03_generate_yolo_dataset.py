import os
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# KONFIGURASI PATH & MAPPING KELAS
# ==============================================================================
DATASET_ROOT = Path(r'D:\SKRIPSI\teaLeafBD')
OUTPUT_DIR   = Path(r'D:\SKRIPSI\phase1_output')
YOLO_DIR     = Path(r'D:\SKRIPSI\tea_yolo')

CLASS_MAPPING = {
    '1. Tea algal leaf spot': (0, 'Tea algal leaf spot'),
    '2. Brown Blight':        (1, 'Brown Blight'),
    '3. Gray Blight':         (2, 'Gray Blight'),
    '4. Helopeltis':          (3, 'Helopeltis'),
    '5. Red spider':          (4, 'Red Spider'),
    '6. Green mirid bug':     (5, 'Green Mirid Bug'),
    '7. Healthy leaf':        (6, 'Healty Leaf'),  # Nama 'Healty Leaf' dipertahankan
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==============================================================================
# ALGORITMA BOUNDING-BOX PER CITRA
# ==============================================================================
def detect_leaf_bbox(im):
    w, h = im.size
    arr = np.array(im, dtype=float)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (R + G + B) / 3.0
    
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    sat = (maxc - minc) / (maxc + 1e-5)
    
    struct = ndimage.generate_binary_structure(2, 2)
    
    # STRATEGI 1: Leaf sebagai pulau (island) di dalam kertas putih
    is_paper = (brightness > 155) & (sat < 0.22)
    paper_closed = ndimage.binary_closing(is_paper, structure=struct, iterations=3)
    paper_filled = ndimage.binary_fill_holes(paper_closed)
    leaf_cand1 = paper_filled & ~paper_closed
    
    lab1, num1 = ndimage.label(leaf_cand1)
    if num1 > 0:
        sizes1 = ndimage.sum(leaf_cand1, lab1, range(num1 + 1))
        max_idx = np.argmax(sizes1[1:]) + 1
        if sizes1[max_idx] > 2000:
            sl = ndimage.find_objects(lab1)[max_idx - 1]
            ymin, xmin = sl[0].start, sl[1].start
            ymax, xmax = sl[0].stop, sl[1].stop
            bw, bh = xmax - xmin, ymax - ymin
            ratio = (bw * bh) / (w * h)
            if 0.005 <= ratio <= 0.85:
                xmin = max(0, xmin - 4)
                ymin = max(0, ymin - 4)
                xmax = min(w, xmax + 4)
                ymax = min(h, ymax + 4)
                return (xmin, ymin, xmax, ymax), 'strategy1_island', ratio
                
    # STRATEGI 2: Organic Chromaticity & Saturation (untuk kasus bayangan / pencahayaan rendah / tepi kertas)
    is_skin = (R - B > 35) & (R - G > 15) & (brightness > 85) & (R > 120)
    leaf_organic = (sat > 0.065) & ~is_skin
    
    leaf_organic[:8, :] = False
    leaf_organic[-8:, :] = False
    leaf_organic[:, :8] = False
    leaf_organic[:, -8:] = False
    
    leaf_closed = ndimage.binary_closing(leaf_organic, structure=struct, iterations=4)
    leaf_filled = ndimage.binary_fill_holes(leaf_closed)
    
    lab2, num2 = ndimage.label(leaf_filled)
    if num2 > 0:
        sizes2 = ndimage.sum(leaf_filled, lab2, range(num2 + 1))
        candidates = []
        for cid in range(1, num2 + 1):
            sz = sizes2[cid]
            if sz < 2000:
                continue
            sl = ndimage.find_objects(lab2)[cid - 1]
            ymn, xmn = sl[0].start, sl[1].start
            ymx, xmx = sl[0].stop, sl[1].stop
            
            bw, bh = xmx - xmn, ymx - ymn
            ratio = (bw * bh) / (w * h)
            if ratio > 0.88 or ratio < 0.003:
                continue
                
            cx = (xmn + xmx) / 2.0
            cy = (ymn + ymx) / 2.0
            dist_center = np.sqrt(((cx - w/2)/(w/2))**2 + ((cy - h/2)/(h/2))**2)
            
            touch_margin = (xmn <= 10) or (xmx >= w - 10) or (ymn <= 10) or (ymx >= h - 10)
            penalty = 2.5 if touch_margin else 1.0
            
            score = sz / (1.0 + 3.0 * dist_center) / penalty
            candidates.append((score, (xmn, ymn, xmx, ymx), ratio))
            
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0]
            xmn, ymn, xmx, ymx = best[1]
            xmn = max(0, xmn - 4)
            ymn = max(0, ymn - 4)
            xmx = min(w, xmx + 4)
            ymx = min(h, ymx + 4)
            return (xmn, ymn, xmx, ymx), 'strategy2_saturation', best[2]
            
    # STRATEGI 3: Fallback Adaptif Contrast
    gray = arr.mean(axis=2)
    fg_contrast = (gray < np.percentile(gray, 40)) & ~is_skin
    fg_contrast[:12, :] = False
    fg_contrast[-12:, :] = False
    fg_contrast[:, :12] = False
    fg_contrast[:, -12:] = False
    fg_c_closed = ndimage.binary_closing(fg_contrast, structure=struct, iterations=4)
    lab3, num3 = ndimage.label(fg_c_closed)
    if num3 > 0:
        sizes3 = ndimage.sum(fg_c_closed, lab3, range(num3 + 1))
        best_id = np.argmax(sizes3[1:]) + 1
        sl = ndimage.find_objects(lab3)[best_id - 1]
        ymn, xmn = sl[0].start, sl[1].start
        ymx, xmx = sl[0].stop, sl[1].stop
        ratio = ((xmx - xmn) * (ymx - ymn)) / (w * h)
        if 0.01 <= ratio <= 0.85:
            return (max(0, xmn-4), max(0, ymn-4), min(w, xmx+4), min(h, ymx+4)), 'strategy3_fallback', ratio

    return None, 'failed', 0.0

def process_single_image(args):
    img_path_str, split_name, class_id, class_name, yolo_dir_str, should_save_preview = args
    img_path = Path(img_path_str)
    yolo_dir = Path(yolo_dir_str)
    
    try:
        im = Image.open(img_path).convert('RGB')
        w, h = im.size
        bbox, strategy, area_ratio = detect_leaf_bbox(im)
        
        if bbox is None:
            return {
                'success': False,
                'path': img_path_str,
                'class': class_name,
                'class_id': class_id,
                'split': split_name,
                'strategy': 'failed',
                'preview': None
            }
            
        xmin, ymin, xmax, ymax = bbox
        bw = xmax - xmin
        bh = ymax - ymin
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        norm_w   = bw / w
        norm_h   = bh / h
        
        # Simpan citra
        dest_img_path = yolo_dir / 'images' / split_name / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # Simpan label txt
        dest_txt_path = yolo_dir / 'labels' / split_name / f'{img_path.stem}.txt'
        with open(dest_txt_path, 'w', encoding='utf-8') as f:
            f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n')
            
        preview_path = None
        if should_save_preview:
            preview_im = im.copy()
            draw = ImageDraw.Draw(preview_im)
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=5)
            text_str = f'[{class_id}] {class_name} ({norm_w*100:.1f}%x{norm_h*100:.1f}%)'
            draw.rectangle([xmin, max(0, ymin - 35), xmin + len(text_str)*13, ymin], fill='red')
            draw.text((xmin + 5, max(0, ymin - 30)), text_str, fill='white')
            
            preview_out = yolo_dir / 'previews' / f'preview_cls{class_id}_{img_path.stem}.jpg'
            preview_im.resize((600, int(600 * h / w))).save(preview_out)
            preview_path = str(preview_out)
            
        return {
            'success': True,
            'path': img_path_str,
            'class': class_name,
            'class_id': class_id,
            'split': split_name,
            'strategy': strategy,
            'bbox': (xmin, ymin, xmax, ymax),
            'norm_bbox': (x_center, y_center, norm_w, norm_h),
            'preview': preview_path
        }
    except Exception as e:
        return {
            'success': False,
            'path': img_path_str,
            'class': class_name,
            'class_id': class_id,
            'split': split_name,
            'strategy': f'error_{e}',
            'preview': None
        }

# ==============================================================================
# GROUPED STRATIFIED SPLIT
# ==============================================================================
def create_leak_free_split():
    with open(OUTPUT_DIR / 'audit_report.json', 'r', encoding='utf-8') as f:
        audit = json.load(f)

    parent_map = {}
    def find(item):
        parent_map.setdefault(item, item)
        if parent_map[item] != item:
            parent_map[item] = find(parent_map[item])
        return parent_map[item]

    def union(item1, item2):
        root1 = find(item1)
        root2 = find(item2)
        if root1 != root2:
            parent_map[root1] = root2

    for grp in audit['exact_duplicates']:
        files = grp['files']
        for i in range(1, len(files)):
            union(files[0], files[i])

    for grp in audit['near_duplicates']:
        files = grp['files']
        for i in range(1, len(files)):
            union(files[0], files[i])

    all_files_by_class = defaultdict(list)
    for folder_name, (cid, cname) in CLASS_MAPPING.items():
        fld_path = DATASET_ROOT / folder_name
        for p in sorted(fld_path.glob('*.jpg')):
            all_files_by_class[folder_name].append(str(p))

    split_assignment = {}
    split_stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}

    for folder_name, (cid, cname) in CLASS_MAPPING.items():
        files = all_files_by_class[folder_name]
        clusters_dict = defaultdict(list)
        for fl in files:
            root_id = find(fl)
            clusters_dict[root_id].append(fl)

        clusters = list(clusters_dict.values())
        rng = random.Random(RANDOM_SEED + cid)
        rng.shuffle(clusters)

        n_total = len(files)
        target_test = int(round(0.10 * n_total))
        target_val  = int(round(0.20 * n_total))
        target_train = n_total - target_test - target_val

        test_cnt, val_cnt, train_cnt = 0, 0, 0

        for cl in clusters:
            sz = len(cl)
            if test_cnt + sz <= target_test or (test_cnt < target_test and abs(test_cnt + sz - target_test) <= abs(val_cnt + sz - target_val)):
                for fl in cl:
                    split_assignment[fl] = 'test'
                    split_stats['test'][cname] += 1
                test_cnt += sz
            elif val_cnt + sz <= target_val:
                for fl in cl:
                    split_assignment[fl] = 'val'
                    split_stats['val'][cname] += 1
                val_cnt += sz
            else:
                for fl in cl:
                    split_assignment[fl] = 'train'
                    split_stats['train'][cname] += 1
                train_cnt += sz

    return split_assignment, split_stats

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print('='*70)
    print('FASE 1: GENERASI DATASET YOLO26 OBJECT DETECTION')
    print('='*70)

    for sub in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test', 'previews']:
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)

    split_assignment, split_stats = create_leak_free_split()
    total_files = len(split_assignment)
    print(f'Total dataset: {total_files} citra')

    print('\nDistribusi Stratified Split Bebas Leakage:')
    for s_name in ['train', 'val', 'test']:
        total_s = sum(split_stats[s_name].values())
        print(f"  {s_name.upper():5s}: {total_s:>5d} ({total_s/total_files*100:.1f}%)")
        for c_name, cnt in sorted(split_stats[s_name].items()):
            print(f"      - {c_name:25s}: {cnt:>4d}")

    # Pilih 3 sampel acak per kelas untuk dibuatkan visual preview
    preview_targets = set()
    for folder_name, (cid, cname) in CLASS_MAPPING.items():
        cls_files = [fl for fl, sp in split_assignment.items() if Path(fl).parent.name == folder_name]
        chosen = random.sample(cls_files, min(3, len(cls_files)))
        for ch in chosen:
            preview_targets.add(ch)

    # Siapkan argumen proses
    tasks = []
    for fl, split_name in split_assignment.items():
        folder_name = Path(fl).parent.name
        class_id, class_name = CLASS_MAPPING[folder_name]
        should_save = fl in preview_targets
        tasks.append((fl, split_name, class_id, class_name, str(YOLO_DIR), should_save))

    num_workers = min(10, os.cpu_count() or 4)
    print(f'\nMemproses {total_files} citra secara paralel menggunakan {num_workers} worker CPU...')

    summary = {
        'total_images': total_files,
        'successful_labels': 0,
        'failed_labels': [],
        'strategy_counts': defaultdict(int),
        'split_counts': defaultdict(int),
        'class_counts': defaultdict(int),
        'previews': []
    }

    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            done += 1
            if res['success']:
                summary['successful_labels'] += 1
                summary['strategy_counts'][res['strategy']] += 1
                summary['split_counts'][res['split']] += 1
                summary['class_counts'][res['class']] += 1
                if res['preview']:
                    summary['previews'].append(res['preview'])
            else:
                summary['failed_labels'].append(res)
                print(f"  [GAGAL] {res['path']}")

            if done % 500 == 0 or done == total_files:
                pct = done / total_files * 100
                print(f"  Progress: {done:>5d}/{total_files} ({pct:5.1f}%) | Sukses: {summary['successful_labels']} | Gagal: {len(summary['failed_labels'])}")

    # Buat data.yaml
    data_yaml_content = f"""# Dataset YOLO Tanaman Teh (Fase 1: Dataset Preparation)
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 7

names:
  0: Tea algal leaf spot
  1: Brown Blight
  2: Gray Blight
  3: Helopeltis
  4: Red Spider
  5: Green Mirid Bug
  6: Healty Leaf
"""
    yaml_file = YOLO_DIR / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    summary['strategy_counts'] = dict(summary['strategy_counts'])
    summary['split_counts'] = dict(summary['split_counts'])
    summary['class_counts'] = dict(summary['class_counts'])

    report_path = OUTPUT_DIR / 'phase1_final_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n' + '='*70)
    print('RINGKASAN AKHIR FASE 1:')
    print('='*70)
    print(f"Total citra diproses     : {summary['total_images']}")
    print(f"Anotasi berhasil (valid) : {summary['successful_labels']} ({summary['successful_labels']/summary['total_images']*100:.2f}%)")
    print(f"Anotasi gagal            : {len(summary['failed_labels'])}")
    print("\nStrategi Bounding Box yang digunakan:")
    for strat, cnt in summary['strategy_counts'].items():
        print(f"  - {strat:25s}: {cnt:>5d} ({cnt/summary['total_images']*100:.1f}%)")
    print("\nDistribusi Per Split:")
    for sp, cnt in summary['split_counts'].items():
        print(f"  - {sp.upper():5s}: {cnt:>5d} ({cnt/summary['total_images']*100:.1f}%)")
    print(f"\n[OK] data.yaml tersimpan di : {yaml_file}")
    print(f"[OK] Laporan lengkap di    : {report_path}")
    print(f"[OK] Preview visual di     : {YOLO_DIR / 'previews'}")

if __name__ == '__main__':
    main()
