"""
FASE 3 - Edge Contour Postprocessing Module
============================================
Modul post-processing yang mengubah bounding box YOLO menjadi
POLYGON KONTUR DAUN menggunakan edge detection (Canny + Morfologi),
sekaligus memfilter daun yang TIDAK UTUH (terpotong batas gambar/ROI).

ALUR KERJA:
  1. YOLO26n+CA mendeteksi penyakit → bounding box (cx, cy, w, h)
  2. Crop ROI dari bounding box
  3. Preprocessing ROI: Gaussian blur → CLAHE → Canny edge
  4. Morfologi: closing (tutup celah) → dilasi tipis
  5. Temukan kontur terbesar → periksa kelengkapan
  6. Jika kontur UTUH: jadikan polygon output
     Jika kontur TIDAK UTUH: suppress deteksi (hapus)

DEFINISI "DAUN UTUH" (parameter yang bisa disesuaikan):
  - Kontur tidak menyentuh tepi ROI (margin ≥ BORDER_MARGIN px)
  - Luas kontur ≥ MIN_CONTOUR_RATIO × luas ROI
  - Solidity (luas/luas convex hull) ≥ MIN_SOLIDITY (mencegah kontur terlalu terpotong)

TANTANGAN DAUN TIDAK SEMPURNA:
  Daun yang sakit/rusak tetap memiliki tepi yang kontinyu (walau tidak bulat).
  Solusinya: tidak menggunakan threshold bentuk (circularity/aspect ratio),
  HANYA menggunakan kelengkapan batas — apakah kontur tertutup tanpa
  menyentuh tepi gambar/ROI.

Referensi tambahan:
  - Canny, J. (1986). A computational approach to edge detection. IEEE TPAMI.
  - Rother, C. et al. (2004). GrabCut — untuk segmentasi interaktif.

Cara penggunaan:
  >>> from phase3_scripts.edge_contour import EdgeContourPostProcessor
  >>> proc = EdgeContourPostProcessor()
  >>> result = proc.process_image(img_bgr, detections)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Parameter (semua bisa di-override saat instansiasi)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeContourConfig:
    """
    Konfigurasi lengkap untuk edge contour post-processing.

    Setiap parameter dilengkapi penjelasan agar mudah di-tune
    ketika akurasi kurang memuaskan pada dataset tertentu.
    """

    # ── Preprocessing ─────────────────────────────────────────────────────
    gaussian_ksize: int = 5
    """Ukuran kernel Gaussian blur sebelum Canny.
    Lebih besar → noise berkurang, tepi lebih halus, tapi detail hilang.
    Disarankan: 3–7 (harus ganjil)."""

    clahe_clip: float = 2.0
    """Clip limit CLAHE (Contrast Limited Adaptive Histogram Equalization).
    CLAHE meningkatkan kontras lokal sehingga tepi daun lebih tegas
    bahkan di area yang gelap/overexposed. Range: 1.0–4.0."""

    clahe_grid: int = 8
    """Ukuran grid CLAHE. Lebih kecil → kontras lebih lokal."""

    # ── Canny Edge Detection ───────────────────────────────────────────────
    canny_low: int = 30
    """Threshold bawah Canny.
    Piksel dengan gradient < canny_low → bukan tepi.
    Lebih rendah → lebih banyak tepi terdeteksi (lebih sensitif)."""

    canny_high: int = 100
    """Threshold atas Canny.
    Piksel dengan gradient > canny_high → pasti tepi.
    Antara low–high: tepi hanya jika terhubung ke tepi kuat."""

    # ── Morfologi (menutup celah pada tepi daun yang rusak/sakit) ──────────
    morph_close_ksize: int = 7
    """Ukuran kernel morphological closing.
    Closing = dilasi diikuti erosi: menutup celah kecil pada kontur daun.
    Lebih besar → celah lebih besar bisa ditutup, tapi bentuk bisa berubah.
    Kritis untuk daun sakit yang tepinya tidak sempurna."""

    morph_dilate_ksize: int = 3
    """Ukuran kernel dilasi setelah closing untuk menebalkan kontur."""

    morph_iterations: int = 2
    """Jumlah iterasi morphological operations."""

    # ── Kelengkapan Kontur (filter daun tidak utuh) ────────────────────────
    border_margin: int = 5
    """Margin batas ROI (pixel).
    Kontur yang menyentuh area ≤ border_margin dari tepi ROI dianggap
    daun terpotong → dibuang.
    Lebih kecil → lebih ketat (lebih banyak dibuang)."""

    min_contour_ratio: float = 0.05
    """Luas minimum kontur relatif terhadap luas ROI.
    Kontur dengan luas < min_contour_ratio × ROI dianggap noise → diabaikan.
    Default 5% artinya kontur harus mengisi minimal 5% area deteksi."""

    min_solidity: float = 0.45
    """Minimum solidity = luas_kontur / luas_convex_hull.
    Solidity rendah = kontur sangat concave/terpotong.
    Default 0.45 cukup longgar untuk daun rusak/tidak sempurna.
    Jangan terlalu tinggi (>0.8) karena daun sakit bisa sangat irregular."""

    max_contour_points: int = 100
    """Jumlah maksimum titik polygon kontur setelah approxPolyDP.
    Lebih sedikit → kontur lebih smooth tapi kurang presisi."""

    approx_epsilon_ratio: float = 0.005
    """Epsilon untuk approxPolyDP sebagai rasio dari panjang arc kontur.
    Lebih besar → kontur lebih disederhanakan (smooth)."""

    # ── Fallback ──────────────────────────────────────────────────────────
    fallback_to_bbox: bool = True
    """Jika kontur tidak ditemukan, kembali ke bounding box asli YOLO.
    True = tetap tampilkan deteksi dengan box asli (lebih aman untuk skripsi).
    False = buang deteksi jika kontur tidak ditemukan."""

    draw_contour: bool = True
    """Gambar polygon kontur (True) atau bounding box asli (False) pada output."""


# ─────────────────────────────────────────────────────────────────────────────
# Struktur Data Hasil Deteksi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Representasi satu deteksi YOLO."""
    x1: int; y1: int; x2: int; y2: int    # Bounding box (pixel absolute)
    class_id: int
    confidence: float
    class_name: str = ""


@dataclass
class ContourResult:
    """Hasil post-processing satu deteksi."""
    detection: Detection
    is_complete: bool               # True = daun utuh, False = terpotong
    contour: Optional[np.ndarray]   # Polygon kontur (dalam koordinat gambar absolut)
    reason: str = ""                # Alasan jika dibuang
    contour_area: float = 0.0
    solidity: float = 0.0
    fallback_used: bool = False     # True jika kembali ke bbox karena kontur gagal


# ─────────────────────────────────────────────────────────────────────────────
# Processor Utama
# ─────────────────────────────────────────────────────────────────────────────

class EdgeContourPostProcessor:
    """
    Post-processor utama yang mengubah bounding box YOLO menjadi
    polygon kontur daun dan memfilter daun yang tidak utuh.

    Penggunaan umum:
        proc = EdgeContourPostProcessor()
        results = proc.process_detections(img_bgr, detections)
        vis = proc.draw_results(img_bgr.copy(), results)
    """

    # Warna per kelas (BGR)
    CLASS_COLORS = [
        (0, 165, 255),   # 0: Tea algal leaf spot  — Orange
        (0, 0, 255),     # 1: Brown Blight          — Red
        (128, 128, 128), # 2: Gray Blight            — Gray
        (255, 0, 255),   # 3: Helopeltis             — Magenta
        (0, 0, 128),     # 4: Red Spider             — Dark Red
        (0, 255, 0),     # 5: Green Mirid Bug        — Green
        (255, 255, 0),   # 6: Healty Leaf            — Cyan
    ]

    def __init__(self, config: Optional[EdgeContourConfig] = None):
        self.cfg = config or EdgeContourConfig()

        # Inisialisasi CLAHE sekali (efisien)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid)
        )

        # Kernel morfologi
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_close_ksize, self.cfg.morph_close_ksize)
        )
        self._kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_dilate_ksize, self.cfg.morph_dilate_ksize)
        )

    # ── Private: Cari kontur daun dalam ROI ──────────────────────────────

    def _find_leaf_contour(self, roi_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, float, float]:
        """
        Cari kontur daun terbesar di dalam ROI.

        Returns:
            (contour_xy | None, reason_str, contour_area, solidity)
            contour_xy: array [N, 1, 2] dalam koordinat ROI
        """
        H, W = roi_bgr.shape[:2]

        # 1. Konversi ke grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE — tingkatkan kontras adaptif (membantu daun gelap/kusam)
        gray = self._clahe.apply(gray)

        # 3. Gaussian Blur — kurangi noise sebelum edge detection
        blurred = cv2.GaussianBlur(gray, (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize), 0)

        # 4. Canny Edge Detection
        edges = cv2.Canny(blurred, self.cfg.canny_low, self.cfg.canny_high)

        # 5. Morphological Closing — tutup celah pada tepi daun sakit/rusak
        #    Closing = dilasi → erosi: celah kecil pada kontur tertutup
        closed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            self._kernel_close,
            iterations=self.cfg.morph_iterations
        )

        # 6. Dilasi tipis — tebalkan kontur agar lebih mudah ditemukan
        dilated = cv2.dilate(
            closed, self._kernel_dilate,
            iterations=1
        )

        # 7. Temukan semua kontur eksternal
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, "Tidak ada kontur ditemukan setelah edge detection", 0.0, 0.0

        # 8. Pilih kontur terbesar (asumsi: daun adalah objek dominan di ROI)
        roi_area    = H * W
        min_area_px = self.cfg.min_contour_ratio * roi_area

        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        if not valid_contours:
            return None, f"Semua kontur terlalu kecil (threshold: {min_area_px:.0f}px²)", 0.0, 0.0

        largest = max(valid_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)

        # 9. Hitung solidity
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / (hull_area + 1e-6)

        # 10. Sederhanakan kontur (approxPolyDP) agar tidak terlalu banyak titik
        epsilon  = self.cfg.approx_epsilon_ratio * cv2.arcLength(largest, closed=True)
        approx   = cv2.approxPolyDP(largest, epsilon, closed=True)

        return approx, "OK", float(contour_area), float(solidity)

    # ── Private: Cek kelengkapan kontur ──────────────────────────────────

    def _is_contour_complete(
        self, contour: np.ndarray, roi_h: int, roi_w: int, solidity: float
    ) -> Tuple[bool, str]:
        """
        Periksa apakah kontur mewakili daun yang UTUH (tidak terpotong).

        Kriteria TIDAK UTUH:
          1. Ada titik kontur yang berada dalam margin border_margin dari tepi ROI.
          2. Solidity terlalu rendah (kontur sangat terpotong/sobek).

        Daun sakit yang tidak sempurna TETAP bisa lolos selama:
          - Tidak menyentuh tepi ROI/gambar (artinya YOLO mendeteksi seluruh daun)
          - Solidity ≥ min_solidity (kontur cukup solid meski irregular)
        """
        margin = self.cfg.border_margin
        pts    = contour.reshape(-1, 2)  # [N, 2]

        # Cek apakah ada titik yang menyentuh tepi ROI
        touches_left   = np.any(pts[:, 0] <= margin)
        touches_right  = np.any(pts[:, 0] >= roi_w - margin)
        touches_top    = np.any(pts[:, 1] <= margin)
        touches_bottom = np.any(pts[:, 1] >= roi_h - margin)

        if touches_left or touches_right or touches_top or touches_bottom:
            sides = []
            if touches_left:   sides.append("kiri")
            if touches_right:  sides.append("kanan")
            if touches_top:    sides.append("atas")
            if touches_bottom: sides.append("bawah")
            return False, f"Kontur menyentuh tepi ROI: {', '.join(sides)} → daun tidak utuh"

        # Cek solidity
        if solidity < self.cfg.min_solidity:
            return False, (f"Solidity {solidity:.3f} < min_solidity {self.cfg.min_solidity:.3f} "
                           f"→ kontur terlalu terpotong")

        return True, "Daun utuh"

    # ── Public: Proses semua deteksi dalam satu gambar ────────────────────

    def process_detections(
        self, img_bgr: np.ndarray, detections: List[Detection]
    ) -> List[ContourResult]:
        """
        Proses daftar deteksi YOLO untuk satu gambar.

        Args:
            img_bgr    : Gambar BGR lengkap (numpy array H×W×3).
            detections : Daftar objek Detection dari output YOLO.

        Returns:
            List[ContourResult]: Hasil per-deteksi. Filter yang dibuang
            memiliki is_complete=False.
        """
        H_img, W_img = img_bgr.shape[:2]
        results = []

        for det in detections:
            # Clamp box ke dalam batas gambar
            x1 = max(0, det.x1); y1 = max(0, det.y1)
            x2 = min(W_img, det.x2); y2 = min(H_img, det.y2)

            if x2 <= x1 or y2 <= y1:
                results.append(ContourResult(
                    detection=det, is_complete=False, contour=None,
                    reason="Bounding box tidak valid (zero area)"
                ))
                continue

            # Crop ROI
            roi = img_bgr[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]

            # Cari kontur daun di dalam ROI
            contour_roi, reason, c_area, solidity = self._find_leaf_contour(roi)

            if contour_roi is None:
                # Kontur tidak ditemukan
                if self.cfg.fallback_to_bbox:
                    # Fallback ke bounding box sebagai polygon persegi panjang
                    bbox_poly = np.array([
                        [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                    ], dtype=np.int32)
                    results.append(ContourResult(
                        detection=det, is_complete=True, contour=bbox_poly,
                        reason=f"Fallback bbox: {reason}",
                        contour_area=float((x2-x1)*(y2-y1)),
                        solidity=1.0, fallback_used=True
                    ))
                else:
                    results.append(ContourResult(
                        detection=det, is_complete=False, contour=None,
                        reason=reason
                    ))
                continue

            # Cek kelengkapan kontur
            is_complete, complete_reason = self._is_contour_complete(
                contour_roi, roi_h, roi_w, solidity
            )

            if not is_complete:
                results.append(ContourResult(
                    detection=det, is_complete=False, contour=None,
                    reason=complete_reason,
                    contour_area=c_area, solidity=solidity
                ))
                continue

            # Transformasikan kontur dari koordinat ROI ke koordinat gambar absolut
            contour_abs = contour_roi.copy()
            contour_abs[:, :, 0] += x1   # offset X
            contour_abs[:, :, 1] += y1   # offset Y

            results.append(ContourResult(
                detection=det, is_complete=True,
                contour=contour_abs,
                reason=complete_reason,
                contour_area=c_area, solidity=solidity
            ))

        return results

    # ── Public: Visualisasi ────────────────────────────────────────────────

    def draw_results(
        self,
        img_bgr: np.ndarray,
        results: List[ContourResult],
        show_suppressed: bool = False,
        show_bbox: bool = False,
        alpha_fill: float = 0.15,
    ) -> np.ndarray:
        """
        Gambar hasil post-processing pada gambar.

        Args:
            img_bgr         : Gambar BGR (akan dimodifikasi in-place).
            results         : Output dari process_detections().
            show_suppressed : Jika True, gambar juga deteksi yang dibuang
                              (dengan warna merah transparan + label "SUPPRESSED").
            show_bbox       : Jika True, juga gambar bounding box asli YOLO (abu-abu tipis).
            alpha_fill      : Opacity fill polygon kontur (0=transparan, 1=solid).

        Returns:
            img_bgr yang sudah digambari.
        """
        overlay = img_bgr.copy()

        for res in results:
            det    = res.detection
            color  = self.CLASS_COLORS[det.class_id % len(self.CLASS_COLORS)]
            label  = f"{det.class_name} {det.confidence:.2f}"

            # Gambar bounding box asli (opsional, abu-abu tipis)
            if show_bbox:
                cv2.rectangle(img_bgr,
                              (det.x1, det.y1), (det.x2, det.y2),
                              (180, 180, 180), 1)

            if res.is_complete and res.contour is not None:
                # ── Deteksi UTUH: gambar polygon kontur ──────────────────
                if self.cfg.draw_contour:
                    # Fill transparan
                    cv2.fillPoly(overlay, [res.contour], color)
                    # Garis kontur solid
                    cv2.drawContours(img_bgr, [res.contour], -1, color, 2)
                else:
                    # Gambar bounding box berwarna
                    cv2.rectangle(img_bgr,
                                  (det.x1, det.y1), (det.x2, det.y2), color, 2)

                # Label
                lx, ly = res.contour[:, 0, 0].min(), res.contour[:, 0, 1].min()
                _put_label(img_bgr, label, lx, ly, color)

                # Info debug (opsional)
                if show_suppressed:  # gunakan flag ini untuk debug info juga
                    debug = f"sol={res.solidity:.2f}"
                    _put_label(img_bgr, debug, lx, ly + 18, color, scale=0.35)

            elif show_suppressed:
                # ── Deteksi DIBUANG: gambar dengan warna merah ───────────
                cv2.rectangle(overlay,
                              (det.x1, det.y1), (det.x2, det.y2),
                              (0, 0, 255), -1)
                cv2.rectangle(img_bgr,
                              (det.x1, det.y1), (det.x2, det.y2),
                              (0, 0, 255), 2)
                _put_label(img_bgr, f"SUPPRESSED: {res.reason[:40]}", det.x1, det.y1, (0, 0, 255))

        # Blend overlay (fill transparan)
        cv2.addWeighted(overlay, alpha_fill, img_bgr, 1 - alpha_fill, 0, img_bgr)
        return img_bgr

    # ── Public: Statistik ─────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[ContourResult]) -> dict:
        """Ringkasan statistik hasil post-processing."""
        total      = len(results)
        complete   = sum(1 for r in results if r.is_complete)
        suppressed = total - complete
        fallback   = sum(1 for r in results if r.fallback_used)

        return {
            "total_detections"    : total,
            "complete_leaves"     : complete,
            "suppressed_leaves"   : suppressed,
            "suppression_rate"    : suppressed / max(total, 1),
            "fallback_used"       : fallback,
            "mean_solidity"       : float(np.mean([r.solidity for r in results if r.is_complete]) or 0),
            "suppression_reasons" : [r.reason for r in results if not r.is_complete],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Render Label Text
# ─────────────────────────────────────────────────────────────────────────────

def _put_label(img, text, x, y, color, scale=0.5, thickness=1):
    """Render label teks dengan background solid agar mudah dibaca."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_top = max(y - th - baseline - 2, 0)
    cv2.rectangle(img, (x, y_top), (x + tw + 2, y), color, -1)
    cv2.putText(img, text, (x + 1, max(y - baseline, th)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Konversi output Ultralytics YOLO ke List[Detection]
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Tea algal leaf spot", "Brown Blight", "Gray Blight",
    "Helopeltis", "Red Spider", "Green Mirid Bug", "Healty Leaf",
]

def yolo_result_to_detections(yolo_result) -> List[Detection]:
    """
    Konversi output ultralytics model.predict() ke List[Detection].

    Args:
        yolo_result: Elemen hasil model.predict(img)[0]

    Returns:
        List[Detection] yang sudah diurutkan by confidence (tertinggi dulu)
    """
    detections = []
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return detections

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
        detections.append(Detection(x1, y1, x2, y2, class_id, confidence, class_name))

    # Urutkan by confidence tertinggi
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Unit Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 65)
    print("  Unit Test: EdgeContourPostProcessor")
    print("=" * 65)

    proc = EdgeContourPostProcessor()

    # Test 1: Buat gambar sintetis dengan 'daun' hijau di tengah
    print("\n[TEST 1] Daun utuh (ellipse di tengah ROI)...")
    img_test = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(img_test, (320, 240), (180, 120), 0, 0, 360, (30, 130, 30), -1)
    cv2.ellipse(img_test, (320, 240), (180, 120), 0, 0, 360, (10, 80, 10), 3)

    det_ok = Detection(140, 120, 500, 360, class_id=6, confidence=0.91, class_name="Healty Leaf")
    results = proc.process_detections(img_test, [det_ok])
    status  = "UTUH ✓" if results[0].is_complete else f"DIBUANG: {results[0].reason}"
    print(f"  → {status} | solidity={results[0].solidity:.3f}")

    # Test 2: Daun terpotong (ellipse menyentuh tepi kiri)
    print("\n[TEST 2] Daun terpotong (menyentuh tepi kiri)...")
    img_test2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(img_test2, (50, 240), (180, 120), 0, 0, 360, (30, 130, 30), -1)
    cv2.ellipse(img_test2, (50, 240), (180, 120), 0, 0, 360, (10, 80, 10), 3)

    det_cut = Detection(0, 120, 230, 360, class_id=6, confidence=0.85, class_name="Healty Leaf")
    results2 = proc.process_detections(img_test2, [det_cut])
    status2  = "UTUH ✓" if results2[0].is_complete else f"DIBUANG ✓: {results2[0].reason}"
    print(f"  → {status2}")

    # Test 3: Daun dengan tepi tidak beraturan (sakit)
    print("\n[TEST 3] Daun tidak beraturan (simulasi sakit)...")
    img_test3 = np.zeros((480, 640, 3), dtype=np.uint8)
    pts_irreg = np.array([
        [200, 150], [280, 120], [380, 160], [420, 240],
        [390, 330], [300, 360], [200, 340], [160, 260]
    ], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img_test3, [pts_irreg], (30, 120, 30))
    cv2.polylines(img_test3, [pts_irreg], True, (10, 70, 10), 3)
    # Tambahkan noise untuk simulasi lesi
    noise_mask = np.random.rand(*img_test3.shape[:2]) < 0.05
    img_test3[noise_mask] = [0, 0, 80]  # bercak gelap

    det_irreg = Detection(150, 110, 430, 370, class_id=1, confidence=0.78, class_name="Brown Blight")
    results3 = proc.process_detections(img_test3, [det_irreg])
    status3  = "UTUH ✓" if results3[0].is_complete else f"DIBUANG: {results3[0].reason}"
    print(f"  → {status3} | solidity={results3[0].solidity:.3f}")

    print(f"\n{'=' * 65}")
    print(f"  Konfigurasi default:")
    cfg = EdgeContourConfig()
    print(f"    canny_low/high   : {cfg.canny_low}/{cfg.canny_high}")
    print(f"    morph_close_ksize: {cfg.morph_close_ksize} (untuk menutup celah daun sakit)")
    print(f"    border_margin    : {cfg.border_margin}px")
    print(f"    min_solidity     : {cfg.min_solidity}")
    print(f"    min_contour_ratio: {cfg.min_contour_ratio}")
    print(f"    fallback_to_bbox : {cfg.fallback_to_bbox}")
    print(f"{'=' * 65}")
