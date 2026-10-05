"""
FASE 2: MODEL SELECTION EXPERIMENT - VISUAL ANALYSIS MODULE
============================================================
Script analisis visual kualitatif pada Test Set:
Mengekstraksi dan memvisualisasikan contoh:
1. True Positive (TP) - Deteksi tepat kelas & kotak akurat (IoU >= 0.5)
2. False Positive (FP) - Deteksi salah sasaran / background dianggap objek
3. False Negative (FN) - Daun berpenyakit terlewat (missed detection)
4. Misclassification - Penyakit tertukar (khususnya Helopeltis vs Green Mirid Bug)
5. Localization Error - Kelas benar namun kotak meleset (IoU < 0.5)
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_selection.config import (
    BASE_DIR,
    RUNS_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE,
    get_model_dir
)
from model_selection.dataset import get_dataloader
from model_selection.models import build_detector, MODEL_REGISTRY


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_bounding_box(image_pil, box, label_text, color="green", width=3):
    """Menggambar bounding box dan teks keterangan pada gambar PIL."""
    draw = ImageDraw.Draw(image_pil)
    draw.rectangle(box, outline=color, width=width)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    # Text background
    text_bbox = draw.textbbox((box[0], max(0, box[1] - 15)), label_text, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((box[0] + 2, max(0, box[1] - 15)), label_text, fill="white", font=font)
    return image_pil


def run_visual_analysis(model_type, weights_path=None, imgsz=DEFAULT_IMGSZ, device="cpu", max_samples=4):
    """
    Menghasilkan grid visualisasi kesalahan deteksi pada test set.
    """
    spec = MODEL_REGISTRY[model_type]
    model_dir = get_model_dir(model_type)
    save_dir = os.path.join(model_dir, "visual_analysis")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"  FASE 2: MODEL SELECTION - VISUAL ERROR ANALYSIS: {spec['name']}")
    print(f"  Direktori Output: {save_dir}")
    print("=" * 70)

    # Muat model (Torchvision vs Ultralytics)
    is_torchvision = model_type in ["retinanet", "ssd", "faster_rcnn"]

    if is_torchvision:
        model, _ = build_detector(model_type, num_classes=NUM_CLASSES, pretrained=False, imgsz=imgsz)
        if weights_path is None:
            weights_path = os.path.join(model_dir, "best_checkpoint.pth")
        
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[INFO] Bobot loaded dari: {weights_path}")
        else:
            print(f"[WARNING] Bobot tidak ditemukan di {weights_path}, analisis visual ditangguhkan.")
            return

        model.to(device)
        model.eval()
    else:
        from ultralytics import YOLO, RTDETR
        if weights_path is None:
            candidate_weights = [
                os.path.join(model_dir, "weights", "best.pt"),
                os.path.join(model_dir, "best.pt"),
                os.path.join(BASE_DIR, "phase2_runs", "yolo26n_baseline_50ep", "weights", "best.pt") if model_type == "yolo26n" else None
            ]
            parent_dir = os.path.dirname(model_dir)
            base_name = os.path.basename(model_dir)
            if os.path.exists(parent_dir):
                matching_dirs = sorted(
                    [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if d.startswith(base_name)],
                    key=lambda p: os.path.getmtime(p),
                    reverse=True
                )
                for md in matching_dirs:
                    candidate_weights.append(os.path.join(md, "weights", "best.pt"))
                    candidate_weights.append(os.path.join(md, "best.pt"))

            weights_path = next((w for w in candidate_weights if w and os.path.exists(w)), None)

        if weights_path and os.path.exists(weights_path):
            model_cls = RTDETR if model_type == "rtdetr" else YOLO
            model = model_cls(weights_path)
            print(f"[INFO] Bobot Ultralytics loaded dari: {weights_path}")
        else:
            print(f"[WARNING] Bobot {model_type} tidak ditemukan di {weights_path}, analisis visual ditangguhkan.")
            return

    test_loader = get_dataloader(split="test", imgsz=imgsz, batch_size=1,
                                 augment=False, background_offset=spec["background_offset"], workers=0)

    categories = {
        "True_Positive": [],
        "False_Negative": [],
        "Misclassification": [],
        "Localization_Error": []
    }

    with torch.no_grad():
        for images, targets in test_loader:
            img_name = targets[0]["img_name"]
            orig_img_path = os.path.join(BASE_DIR, "tea_yolo", "images", "test", img_name)
            if not os.path.exists(orig_img_path):
                continue

            # Dapatkan dimensi gambar asli untuk unifikasi koordinat
            with Image.open(orig_img_path) as tmp_img:
                w_orig, h_orig = tmp_img.size
            scale_x = w_orig / imgsz
            scale_y = h_orig / imgsz

            # Ground Truth dikonversi ke koordinat gambar asli
            raw_gb = targets[0]["boxes"].cpu().numpy()
            gl = (targets[0]["labels"].cpu().numpy() - spec["background_offset"])

            gb = []
            for b in raw_gb:
                gb.append([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y])
            gb = np.array(gb) if len(gb) > 0 else np.zeros((0, 4))

            # Prediksi Model
            if is_torchvision:
                images_dev = [img.to(device) for img in images]
                outputs = model(images_dev)
                raw_pb = outputs[0]["boxes"].cpu().numpy()
                ps = outputs[0]["scores"].cpu().numpy()
                pl = (outputs[0]["labels"].cpu().numpy() - spec["background_offset"])

                # Skalakan koordinat deteksi torchvision ke dimensi gambar asli
                pb = []
                for b in raw_pb:
                    pb.append([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y])
                pb = np.array(pb) if len(pb) > 0 else np.zeros((0, 4))
            else:
                results = model.predict(orig_img_path, conf=0.25, imgsz=imgsz, device=device, verbose=False)
                res = results[0]
                pb = res.boxes.xyxy.cpu().numpy()
                ps = res.boxes.conf.cpu().numpy()
                pl = res.boxes.cls.cpu().numpy().astype(int)

            # Filter deteksi dengan confidence >= 0.25
            high_conf = ps >= 0.25
            pb = pb[high_conf]
            ps = ps[high_conf]
            pl = pl[high_conf]

            if len(gb) > 0 and len(pb) == 0:
                # Missed detection (FN)
                if len(categories["False_Negative"]) < max_samples:
                    categories["False_Negative"].append((orig_img_path, gb[0], gl[0], None, None, None))
            elif len(gb) > 0 and len(pb) > 0:
                iou = compute_iou(gb[0], pb[0])
                gt_cls = gl[0]
                pred_cls = pl[0]
                pred_score = ps[0]

                if iou >= 0.5 and gt_cls == pred_cls:
                    if len(categories["True_Positive"]) < max_samples:
                        categories["True_Positive"].append((orig_img_path, gb[0], gt_cls, pb[0], pred_cls, pred_score))
                elif iou >= 0.5 and gt_cls != pred_cls:
                    if len(categories["Misclassification"]) < max_samples:
                        categories["Misclassification"].append((orig_img_path, gb[0], gt_cls, pb[0], pred_cls, pred_score))
                elif 0.1 <= iou < 0.5 and gt_cls == pred_cls:
                    if len(categories["Localization_Error"]) < max_samples:
                        categories["Localization_Error"].append((orig_img_path, gb[0], gt_cls, pb[0], pred_cls, pred_score))

            # Hentikan jika semua kategori sudah terkumpul
            if all(len(v) >= max_samples for v in categories.values()):
                break

        # Render dan simpan visual grid
        for cat_name, samples in categories.items():
            if len(samples) == 0:
                continue
            fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5))
            if len(samples) == 1:
                axes = [axes]
            
            for ax, (img_path, gbox, gcls, pbox, pcls, pscore) in zip(axes, samples):
                img = Image.open(img_path).convert("RGB")
                img = draw_bounding_box(img, gbox, f"GT: {CLASS_NAMES[gcls]}", color="green")

                if pbox is not None:
                    color = "blue" if pcls == gcls else "red"
                    img = draw_bounding_box(img, pbox, f"Pred: {CLASS_NAMES[pcls]} ({pscore:.2f})", color=color)

                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"{os.path.basename(img_path)}", fontsize=10)

            plt.suptitle(f"{spec['name']} - {cat_name.replace('_', ' ')}", fontsize=14, weight="bold")
            plt.tight_layout()
            out_file = os.path.join(save_dir, f"{cat_name.lower()}.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            print(f"[SAVED] Visualisasi {cat_name} disimpan di: {out_file}")

    print(f"\n[DONE] Analisis visual selesai! Seluruh visualisasi berada di: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="FASE 2: Visual Prediction Analyzer")
    parser.add_argument("--model", type=str, required=True, choices=["retinanet", "ssd", "faster_rcnn", "rtdetr", "yolo26n"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device target ('cpu' atau 'cuda')")
    args = parser.parse_args()

    run_visual_analysis(model_type=args.model, weights_path=args.weights, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
