"""
FASE 3 - Script 06: Testing Gambar Eksternal / Google (URL & Upload)
===================================================================
Script mandiri untuk menguji model YOLO26n+CA dengan gambar baru dari
Google (via URL) atau gambar upload lokal.

FITUR:
  1. Download otomatis dari URL (Google Images direct link / URL apapun).
  2. Mendukung file lokal yang di-upload ke Colab / PC.
  3. Membandingkan 4 visualisasi berdampingan (4-panel):
     - Panel 1: Gambar Asli
     - Panel 2: Deteksi YOLO26n+CA (Bounding Box Standar)
     - Panel 3: Edge Contour (Polygon Mengikuti Bentuk Daun)
     - Panel 4: Canny Edge Map (Peta Tepi Daun)
  4. Analisis status kelengkapan daun (Utuh vs Terpotong tepi).

PENGGUNAAN:
  # Via URL:
  python phase3_scripts/06_test_external_image.py \
      --url "https://example.com/daun_teh.jpg" \
      --weights phase3_runs/yolo26n_ca_50ep/weights/best.pt

  # Via Gambar Lokal:
  python phase3_scripts/06_test_external_image.py \
      --image "path/to/gambar.jpg" \
      --weights phase3_runs/yolo26n_ca_50ep/weights/best.pt
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Registrasi CA
from phase3_scripts.ca_module import register_ca_module
register_ca_module()

from ultralytics import YOLO  # noqa
from phase3_scripts.edge_contour import (
    EdgeContourPostProcessor, EdgeContourConfig,
    yolo_result_to_detections, CLASS_NAMES, CLASS_COLORS
)


def download_image(url: str, save_path: str) -> str:
    """Download gambar dari URL dengan User-Agent agar tidak diblokir web."""
    print(f"[INFO] Mengunduh gambar dari URL: {url} ...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"[INFO] Gambar berhasil diunduh ke: {save_path}")
    return save_path


def test_external_image(
    image_source: str,
    weights_path: str,
    is_url: bool = False,
    conf_thresh: float = 0.25,
    output_dir: str = "test_google_results"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dapatkan file gambar
    if is_url:
        temp_img_path = str(out_dir / "downloaded_image.jpg")
        try:
            img_path = download_image(image_source, temp_img_path)
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh gambar: {e}")
            return None
    else:
        img_path = image_source
        if not os.path.exists(img_path):
            print(f"[ERROR] Berkas gambar '{img_path}' tidak ditemukan.")
            return None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Format gambar tidak didukung atau berkas rusak: {img_path}")
        return None

    H, W = img_bgr.shape[:2]
    print(f"\n[INFO] Dimensi gambar: {W}×{H} piksel")

    # 2. Muat model YOLO26n+CA
    print(f"[INFO] Memuat bobot model: {weights_path}")
    model = YOLO(weights_path)

    # 3. Inferensi YOLO
    print("[INFO] Menjalankan deteksi penyakit (YOLO26n + CA)...")
    results = model.predict(img_bgr, conf=conf_thresh, device="cpu" if not cv2.cuda.getCudaEnabledDeviceCount() else "cuda", verbose=False)[0]
    detections = yolo_result_to_detections(results)
    print(f"[INFO] Terdeteksi {len(detections)} kandidat penyakit daun teh.")

    # 4. Edge Contour Post-processing
    print("[INFO] Mengekstrak kontur tepi daun & memeriksa kelengkapan...")
    config = EdgeContourConfig(
        canny_low=30,
        canny_high=100,
        morph_close_ksize=7,
        border_margin=5,
        min_solidity=0.45
    )
    processor = EdgeContourPostProcessor(config=config)
    contour_results = processor.process_detections(img_bgr, detections)

    # 5. Visualisasi Bounding Box Biasa
    img_bbox = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_abs(W, H)
        color = CLASS_COLORS.get(det.class_name, (0, 255, 0))
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(img_bbox, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_bbox, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 6. Visualisasi Edge Contour
    img_contour = processor.visualize(
        img_bgr,
        contour_results,
        draw_contours=True,
        draw_orig_bbox=False,
        show_suppressed=True
    )

    # 7. Edge Mask keseluruhan (Canny) untuk visualisasi perbandingan
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 8. Buat Plot 4-Panel
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Asli
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Gambar Asli (Input Google)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: YOLO26n+CA Bounding Box
    axes[0, 1].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2. YOLO26n+CA Bounding Box ({len(detections)} deteksi)", fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: Edge Contour (Mengikuti Daun & Filter Kelengkapan)
    valid_count = sum(1 for c in contour_results if not c.is_suppressed)
    supp_count = len(contour_results) - valid_count
    axes[1, 0].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f"3. Edge Contour Daun (Utuh: {valid_count} | Terpotong/Suppressed: {supp_count})",
        fontsize=13, fontweight='bold'
    )
    axes[1, 0].axis('off')

    # Panel 4: Canny Edge Map
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title("4. Canny Edge Map (Deteksi Garis Tepi)", fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    comparison_save = str(out_dir / "hasil_testing_lengkap.png")
    plt.savefig(comparison_save, dpi=150, bbox_inches='tight')
    plt.close()

    # Cetak rangkuman ke terminal
    print("\n" + "=" * 65)
    print("  HASIL ANALISIS DETEKSI")
    print("=" * 65)
    for i, res in enumerate(contour_results):
        status = "✅ UTUH (DITERIMA)" if not res.is_suppressed else f"❌ SUPPRESSED ({res.suppress_reason})"
        print(f"[{i+1}] Kelas: {res.detection.class_name:<20} Conf: {res.detection.confidence:.3f}")
        print(f"    Status Daun : {status}")
        if res.contour_found:
            print(f"    Luas Kontur : {res.contour_area:.1f} px | Solidity: {res.solidity:.2f}")
    print("=" * 65)
    print(f"\n[SUKSES] Hasil visualisasi disimpan di: {comparison_save}")

    return comparison_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gambar eksternal/Google dengan YOLO26n+CA + Edge Contour")
    parser.add_argument("--url", default=None, help="URL langsung gambar daun dari Google / web")
    parser.add_argument("--image", default=None, help="Path gambar lokal yang di-upload")
    parser.add_argument("--weights", default=None, help="Path ke weights best.pt (default: auto-detect)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", default="test_google_results", help="Folder output")
    args = parser.parse_args()

    # Auto detect weights
    weights = args.weights
    if not weights:
        candidates = [
            os.path.join(PROJECT_DIR, "phase3_runs", "yolo26n_ca_50ep", "weights", "best.pt"),
            "/content/phase3_runs/yolo26n_ca_50ep/weights/best.pt",
            os.path.join(PROJECT_DIR, "best.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                weights = c
                break

    if not weights:
        print("[ERROR] File best.pt tidak ditemukan. Tentukan dengan argumen --weights <path>")
        sys.exit(1)

    if args.url:
        test_external_image(args.url, weights, is_url=True, conf_thresh=args.conf, output_dir=args.out)
    elif args.image:
        test_external_image(args.image, weights, is_url=False, conf_thresh=args.conf, output_dir=args.out)
    else:
        print("[ERROR] Mohon masukkan argumen --url <URL_GAMBAR> atau --image <PATH_GAMBAR>")
