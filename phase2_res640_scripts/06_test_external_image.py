"""
FASE 2 (RESOLUSI STANDAR 640) - Script 06: Testing Gambar Eksternal / Foto Baru
================================================================================
Menguji model YOLO26n Baseline pada citra daun teh mandiri
(berkas lokal baru, foto kamera HP, atau URL langsung dari web).
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Deteksi platform
IS_COLAB = os.path.exists("/content")
BASE_DIR = Path("/content") if IS_COLAB else Path("D:/SKRIPSI")

PROJECT_DIR = BASE_DIR / "phase2_runs"
RUN_NAME    = "yolo26n_baseline_640"
DEFAULT_WEIGHTS = PROJECT_DIR / RUN_NAME / "weights/best.pt"

def download_image(url: str, save_path: str) -> str:
    print(f"[INFO] Mengunduh gambar dari URL: {url} ...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp, open(save_path, 'wb') as f:
        f.write(resp.read())
    print(f"[INFO] Berhasil diunduh ke: {save_path}")
    return save_path

def test_single_image(
    image_source: str,
    weights_path: str = None,
    is_url: bool = False,
    conf_thresh: float = 0.25,
    output_dir: str = "external_test_results"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if weights_path is None:
        weights_path = str(DEFAULT_WEIGHTS)
        if not os.path.exists(weights_path):
            # Fallback ke baseline 50ep sebelumnya jika yang 640 belum selesai
            alt = BASE_DIR / "phase2_runs/yolo26n_baseline_50ep/weights/best.pt"
            if alt.exists():
                weights_path = str(alt)

    print("=" * 70)
    print("  FASE 2 - PENGUJIAN CITRA EKSTERNAL (YOLO26n BASELINE 640x640)")
    print("=" * 70)
    print(f"Model Weights : {weights_path}")
    print(f"Input Source  : {image_source}")
    print(f"Confidence Th : {conf_thresh}")

    if not os.path.exists(weights_path):
        print(f"[ERROR] Bobot model tidak ditemukan di {weights_path}!")
        return

    # Ambil berkas gambar
    if is_url:
        img_path = str(out_dir / "downloaded_leaf.jpg")
        try:
            download_image(image_source, img_path)
        except Exception as e:
            print(f"[ERROR] Gagal download gambar: {e}")
            return
    else:
        img_path = image_source
        if not os.path.exists(img_path):
            print(f"[ERROR] Berkas {img_path} tidak ditemukan!")
            return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Format gambar tidak dapat dibaca: {img_path}")
        return

    # Muat model & Prediksi
    model = YOLO(weights_path)
    res = model.predict(img_path, conf=conf_thresh, imgsz=640, verbose=False)[0]

    # Plot hasil deteksi
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Buat figure komparasi berdampingan
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Citra Asli", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    n_det = len(res.boxes) if res.boxes is not None else 0
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f"Deteksi YOLO26n Baseline ({n_det} Objek)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    save_result_path = out_dir / f"result_{Path(img_path).name}"
    plt.savefig(str(save_result_path), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[HASIL DETEKSI] Ditemukan {n_det} objek:")
    if res.boxes is not None and len(res.boxes) > 0:
        for i, box in enumerate(res.boxes):
            cid = int(box.cls[0])
            cname = model.names.get(cid, f"Class {cid}")
            conf = float(box.conf[0])
            print(f"  {i+1}. {cname:<25} - Confidence: {conf*100:.2f}%")
    else:
        print("  (Tidak ada penyakit/hama terdeteksi pada ambang batas conf ini)")

    print(f"\n[SUKSES] Gambar komparasi tersimpan di: {save_result_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gambar eksternal YOLO26n Baseline")
    parser.add_argument("--image", type=str, help="Path gambar lokal")
    parser.add_argument("--url", type=str, help="URL gambar dari web")
    parser.add_argument("--weights", type=str, default=None, help="Path bobot best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="external_test_results", help="Folder output")
    args = parser.parse_args()

    if args.url:
        test_single_image(args.url, weights_path=args.weights, is_url=True, conf_thresh=args.conf, output_dir=args.output)
    elif args.image:
        test_single_image(args.image, weights_path=args.weights, is_url=False, conf_thresh=args.conf, output_dir=args.output)
    else:
        # Default test pada sample pertama jika ada
        default_sample = BASE_DIR / "tea_yolo/images/test"
        sample_files = list(default_sample.glob("*.jpg")) if default_sample.exists() else []
        if sample_files:
            test_single_image(str(sample_files[0]), weights_path=args.weights, is_url=False, conf_thresh=args.conf, output_dir=args.output)
        else:
            print("Gunakan argumen --image <path> atau --url <link>")
