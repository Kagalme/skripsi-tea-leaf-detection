import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
import random

random.seed(42)

def detect_leaf_bbox(im):
    w, h = im.size
    arr = np.array(im, dtype=float)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (R + G + B) / 3.0
    
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    sat = (maxc - minc) / (maxc + 1e-5)
    
    # 1. White paper: high brightness and low saturation (achromatic)
    is_paper = (brightness > 155) & (sat < 0.22)
    struct = ndimage.generate_binary_structure(2, 2)
    paper_closed = ndimage.binary_closing(is_paper, structure=struct, iterations=3)
    
    # Strategy 1: Leaf as enclosed island in paper
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
                # Add small padding 4px
                xmin = max(0, xmin - 4)
                ymin = max(0, ymin - 4)
                xmax = min(w, xmax + 4)
                ymax = min(h, ymax + 4)
                return (xmin, ymin, xmax, ymax), 'strategy1_island', ratio
                
    # Strategy 2: Saturation & Organic Chromaticity (handles shadows, open borders, dim paper)
    # The tea leaf has organic chromaticity (sat > 0.065), whereas paper and shadow on paper are achromatic (sat < 0.05).
    # Also exclude skin tone if present (fingers holding leaf)
    is_skin = (R - B > 35) & (R - G > 15) & (brightness > 85) & (R > 120)
    leaf_organic = (sat > 0.065) & ~is_skin
    
    # Exclude extreme borders (outer margin 8px) to disconnect outer tea bushes/hands
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
            
            # Penalize components touching the outer margins
            touch_margin = (xmn <= 10) or (xmx >= w - 10) or (ymn <= 10) or (ymx >= h - 10)
            penalty = 2.5 if touch_margin else 1.0
            
            score = sz / (1.0 + 3.0 * dist_center) / penalty
            candidates.append((score, (xmn, ymn, xmx, ymx), ratio))
            
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0]
            xmn, ymn, xmx, ymx = best[1]
            # Add padding 4px
            xmn = max(0, xmn - 4)
            ymn = max(0, ymn - 4)
            xmx = min(w, xmx + 4)
            ymx = min(h, ymx + 4)
            return (xmn, ymn, xmx, ymx), 'strategy2_saturation', best[2]
            
    return None, 'failed', 0

# Benchmark across 140 random images
root = Path(r'D:\SKRIPSI\teaLeafBD')
classes = sorted([d for d in root.iterdir() if d.is_dir()])
results = {'strat1': 0, 'strat2': 0, 'failed': 0}
tested = 0

print('Memulai benchmark 140 citra (20 citra x 7 kelas)...')
for c in classes:
    imgs = list(c.glob('*.jpg'))
    samples = random.sample(imgs, 20)
    for p in samples:
        tested += 1
        im = Image.open(p)
        bbox, strat, ratio = detect_leaf_bbox(im)
        if strat == 'strategy1_island':
            results['strat1'] += 1
        elif strat == 'strategy2_saturation':
            results['strat2'] += 1
        else:
            results['failed'] += 1
            print(f'  [FAILED] {c.name}/{p.name}')

print(f'\nTotal Diuji: {tested}')
print(f"Strategy 1 (Paper Island): {results['strat1']} ({results['strat1']/tested*100:.1f}%)")
print(f"Strategy 2 (Organic Saturation): {results['strat2']} ({results['strat2']/tested*100:.1f}%)")
print(f"Failed: {results['failed']} ({results['failed']/tested*100:.1f}%)")
print(f"Total Success: {results['strat1'] + results['strat2']}/{tested} ({(results['strat1'] + results['strat2'])/tested*100:.1f}%)")
