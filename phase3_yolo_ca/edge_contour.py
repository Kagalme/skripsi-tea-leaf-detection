"""
FASE 3 (DIPERBARUI v2.5) - Multi-Leaf Detection (Sakit + Sehat) + Foreground DoF & Twig Filter
==============================================================================================
Menggabungkan keunggulan Multi-Deteksi dari v2.3 (mendeteksi 6-10+ daun sekaligus, baik daun
berpenyakit maupun daun sehat) dengan Filter Foreground Depth-of-Field (DoF) yang terkalibrasi:

Mengapa pada v2.4 sebelumnya hanya terdeteksi 1 objek?
1. Pada v2.4, penghapusan `MORPH_CLOSE` pada masker daun membuat helai daun yang memiliki
   banyak bercak terpecah menjadi serpihan kecil, dan perbandingan `focus_score >= 0.48 * max_focus`
   terdistorsi oleh kontras ekstrem lubang hitam pada 1 serpihan daun sehingga daun-daun lain
   ikut terbuang.
2. Pada v2.5 ini, pipeline segmentasi Multi-Daun v2.3 dikembalikan 100% utuh, dan filter DoF
   diganti menjadi **Normalized Edge Acutance + Foreground Illumination & ExG Purity** yang
   HANYA membuang ranting berlumut serta daun latar belakang yang gelap/blur tanpa membuang
   daun-daun utama di bidang fokus depan!
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta Kelas & Warna di Level Modul
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Tea algal leaf spot",  # 0
    "Brown Blight",         # 1
    "Gray Blight",          # 2
    "Helopeltis",           # 3
    "Red Spider",           # 4
    "Green Mirid Bug",      # 5
    "Healty Leaf",          # 6
]

# Daftar warna BGR berdasarkan class_id (0..6)
CLASS_COLORS_LIST = [
    (0, 165, 255),   # 0: Tea algal leaf spot  — Orange
    (0, 0, 255),     # 1: Brown Blight          — Red
    (128, 128, 128), # 2: Gray Blight           — Gray
    (255, 0, 255),   # 3: Helopeltis            — Magenta
    (0, 0, 128),     # 4: Red Spider            — Dark Red
    (0, 255, 0),     # 5: Green Mirid Bug       — Green
    (255, 255, 0),   # 6: Healty Leaf           — Cyan
]

# Dict warna BGR yang mendukung lookup via class_name maupun class_id (.get() aman!)
CLASS_COLORS = {name: CLASS_COLORS_LIST[i] for i, name in enumerate(CLASS_NAMES)}
for _i, _c in enumerate(CLASS_COLORS_LIST):
    CLASS_COLORS[_i] = _c
CLASS_COLORS["Healthy Leaf"] = CLASS_COLORS_LIST[6]


@dataclass
class EdgeContourConfig:
    """Konfigurasi parameter untuk Edge Contour, Twig Filter & Depth-of-Field (DoF)."""
    gaussian_ksize: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    canny_low: int = 30
    canny_high: int = 100
    morph_close_ksize: int = 7
    morph_dilate_ksize: int = 3
    morph_iterations: int = 2
    border_margin: int = 6
    min_contour_ratio: float = 0.015
    min_solidity: float = 0.50
    max_aspect_ratio: float = 4.2       # Menolak kontur memanjang seperti ranting (L/W > 4.2)
    filter_twigs: bool = True           # Aktifkan filter pemisah daun vs ranting
    filter_dof_blur: bool = True        # Aktifkan filter Depth-of-Field (tolak ranting/background blur)
    only_diseased_in_field: bool = False  # False = Multi-deteksi SEMUA daun fokus depan (Sakit + Sehat)
    max_contour_points: int = 100
    approx_epsilon_ratio: float = 0.004
    fallback_to_bbox: bool = False
    draw_contour: bool = True


@dataclass
class Detection:
    """Representasi satu deteksi YOLO dalam koordinat piksel absolut."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    class_name: str = ""

    def bbox_abs(self, img_w: Optional[int] = None, img_h: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Mengembalikan tuple (x1, y1, x2, y2) yang sudah di-clamp ke ukuran gambar."""
        if img_w is not None and img_h is not None:
            return (
                max(0, min(int(self.x1), img_w)),
                max(0, min(int(self.y1), img_h)),
                max(0, min(int(self.x2), img_w)),
                max(0, min(int(self.y2), img_h)),
            )
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class ContourResult:
    """Hasil post-processing satu deteksi."""
    detection: Detection
    is_complete: bool
    contour: Optional[np.ndarray]
    reason: str = ""
    contour_area: float = 0.0
    solidity: float = 0.0
    fallback_used: bool = False

    @property
    def is_suppressed(self) -> bool:
        """True jika daun terpotong / tidak utuh (kompatibilitas script eksternal)."""
        return not self.is_complete

    @property
    def suppress_reason(self) -> str:
        """Alasan penolakan kontur (kompatibilitas script eksternal)."""
        return self.reason

    @property
    def contour_found(self) -> bool:
        """True jika kontur ditemukan (kompatibilitas script eksternal)."""
        return self.contour is not None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pemisah Spektral Daun vs Ranting Kayu (Midrib Bridging + Hole Filling Utuh)
# ─────────────────────────────────────────────────────────────────────────────

def is_white_paper_background(img_bgr: np.ndarray) -> bool:
    """Mendeteksi apakah gambar berasal dari kertas putih studio (teaLeafBD) atau kebun."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    border_mask = np.zeros((H, W), dtype=bool)
    b_h = max(4, int(H * 0.08))
    b_w = max(4, int(W * 0.08))
    border_mask[:b_h, :] = True
    border_mask[-b_h:, :] = True
    border_mask[:, :b_w] = True
    border_mask[:, -b_w:] = True

    s_border = hsv[:, :, 1][border_mask]
    v_border = hsv[:, :, 2][border_mask]
    paper_pixels = (v_border > 150) & (s_border < 58)
    return float(np.mean(paper_pixels)) > 0.35


def _fill_small_internal_holes(binary_mask: np.ndarray, max_hole_ratio: float = 0.045) -> np.ndarray:
    """
    Mengisi lubang-lubang bercak penyakit coklat/hitam di bagian dalam lamina daun
    TANPA mengisi celah latar belakang / ranting di luar daun.
    """
    H, W = binary_mask.shape[:2]
    max_hole_px = int(H * W * max_hole_ratio)
    inv = cv2.bitwise_not(binary_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    out = binary_mask.copy()
    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl]
        touches_border = (x <= 1) or (y <= 1) or (x + bw >= W - 1) or (y + bh >= H - 1)
        if not touches_border and area <= max_hole_px:
            out[labels == lbl] = 255
    return out


def compute_leaf_mask_no_twigs(img_bgr: np.ndarray) -> np.ndarray:
    """
    Menghasilkan binary mask (0 atau 255) yang berisi seluruh helai daun teh
    (baik daun hijau sehat, pucuk muda kuning-hijau, maupun daun berpenyakit/perunggu),
    menyambung tulang daun tengah (midrib), serta membuang ranting kayu coklat/abu-abu.
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    b_f = img_bgr[:, :, 0].astype(np.float32)
    g_f = img_bgr[:, :, 1].astype(np.float32)
    r_f = img_bgr[:, :, 2].astype(np.float32)

    exg = 2.0 * g_f - r_f - b_f

    if is_white_paper_background(img_bgr):
        not_paper = ~((v_ch > 160) & (s_ch < 50))
        not_skin = ~((r_f - b_f > 35) & (r_f - g_f > 15) & (v_ch > 90))
        raw_mask = ((not_paper & not_skin) | (exg > 15)).astype(np.uint8) * 255
        raw_mask = _fill_small_internal_holes(raw_mask, max_hole_ratio=0.08)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Mode Kebun / Lapangan (Sama seperti v2.3 agar seluruh daun depan utuh 100%):
    is_woody_twig = (
        (h_ch < 13)
        | (h_ch > 150)
        | (r_f > g_f + 20.0)
        | ((r_f > g_f + 4.0) & (g_f - b_f < 32.0))
        | (exg < 8.0)
    )
    is_gray_twig = (s_ch < 35) & (exg < 22.0)
    is_dark_shadow = (v_ch < 48)

    is_leaf_lamina = (
        (h_ch >= 13)
        & (h_ch <= 90)
        & (s_ch >= 35)
        & (v_ch >= 48)
        & (exg >= 10.0)
        & (g_f > b_f + 14.0)
        & (~is_woody_twig)
        & (~is_gray_twig)
        & (~is_dark_shadow)
    )

    mask = (is_leaf_lamina.astype(np.uint8)) * 255

    # Sambungkan garis tulang daun tengah (midrib) agar daun tidak terbelah dua
    close_k = max(7, int(min(H, W) * 0.013) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        iterations=1,
    )

    # Isi seluruh lubang bercak penyakit coklat/hitam di bagian dalam helai daun
    mask = _fill_small_internal_holes(mask, max_hole_ratio=0.045)

    # Putuskan jembatan tangkai/ranting tipis dengan Morphological Opening
    open_k = max(5, int(min(H, W) * 0.012) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        iterations=1,
    )
    mask = _fill_small_internal_holes(mask, max_hole_ratio=0.045)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. Filter Post-Segmentasi: Tolak Ranting Lumut & Daun Background Blur (DoF)
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_foreground_in_focus_leaf(
    img_bgr: np.ndarray,
    contour_abs: np.ndarray,
    filter_dof_blur: bool = True,
) -> Tuple[bool, str]:
    """
    Memeriksa apakah sebuah kandidat kontur hasil segmentasi adalah **Daun Foreground
    yang Fokus Tajam**, ataukah salah satu dari 2 jenis false positive berikut:
    1. **Kumpulan Ranting Berlumut / Celah Semak** (seperti di kiri-atas & tengah-atas):
       Memiliki nilai rata-rata Excess Green (`mean_exg`) rendah (< 28) dan rasio piksel
       abu-abu/coklat/gelap (`twig_or_dark_ratio`) tinggi (> 38%).
    2. **Daun Background di Luar Depth-of-Field (Blur / Gelap di Latar Belakang)**:
       Berada di bayangan dalam (`median_g < 102` atau `mean_v < 100`) atau memiliki
       ketajaman tepi ternormalisasi (*Normalized Edge Acutance*) yang rendah (< 0.21).
       PENTING: Normalized Edge Acutance membagi gradien 3x3 dengan rentang kontras lokal
       9x9 sehingga daun sehat yang mulus (`Healty Leaf`) maupun daun berbercak SAMA-SAMA
       lolos selama berada di bidang fokus kamera!
    """
    H, W = img_bgr.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(contour_abs)
    x1, y1, x2, y2 = max(0, bx), max(0, by), min(W, bx + bw), min(H, by + bh)
    if x2 <= x1 or y2 <= y1:
        return False, "Empty bbox"

    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    c_mask = np.zeros((ch, cw), dtype=np.uint8)
    cnt_local = contour_abs.copy()
    cnt_local[:, :, 0] -= x1
    cnt_local[:, :, 1] -= y1
    cv2.drawContours(c_mask, [cnt_local], -1, 255, -1)

    inside = c_mask > 0
    if int(np.count_nonzero(inside)) < 80:
        return False, "Area terlalu kecil"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_in = hsv[:, :, 0][inside].astype(np.float32)
    s_in = hsv[:, :, 1][inside].astype(np.float32)
    v_in = hsv[:, :, 2][inside].astype(np.float32)

    b_in = crop[:, :, 0][inside].astype(np.float32)
    g_in = crop[:, :, 1][inside].astype(np.float32)
    r_in = crop[:, :, 2][inside].astype(np.float32)

    exg_in = 2.0 * g_in - r_in - b_in
    gb_in = g_in - b_in

    mean_exg = float(np.mean(exg_in))
    median_g = float(np.median(g_in))
    mean_v = float(np.mean(v_in))

    # Proporsi piksel ranting kayu, lumut abu-abu, atau celah bayangan gelap di dalam kontur
    is_twig_or_dark = (
        (v_in < 78.0)
        | ((s_in < 42.0) & (exg_in < 24.0))
        | ((r_in > g_in + 4.0) & (gb_in < 30.0))
        | (exg_in < 14.0)
    )
    twig_dark_ratio = float(np.mean(is_twig_or_dark))

    # 1. Tolak kumpulan ranting berlumut / celah semak (seperti kotak di kiri-atas & tengah-atas)
    if mean_exg < 30.0 or twig_dark_ratio > 0.36:
        return False, f"Ranting/lumut (ExG={mean_exg:.1f}, twig_ratio={twig_dark_ratio:.2f})"

    if not filter_dof_blur:
        return True, "OK"

    # 2. Tolak daun latar belakang yang gelap di bawah bayangan semak (understory)
    if median_g < 104.0 or mean_v < 102.0:
        return False, f"Background gelap di luar DoF (medG={median_g:.1f}, meanV={mean_v:.1f})"

    # 3. Normalized Edge Acutance (mengukur ketajaman fokus optik tanpa bias terhadap lubang/bercak)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy) / 4.0  # Skala gradien 3x3 per piksel

    k_rng = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    local_range = cv2.dilate(gray, k_rng) - cv2.erode(gray, k_rng)

    # Acutance pada piksel-piksel tepi/urat yang memiliki perubahan intensitas nyata (local_range >= 18)
    edge_zones = inside & (local_range >= 18.0)
    if int(np.count_nonzero(edge_zones)) >= 20:
        acutance_vals = grad_mag[edge_zones] / (local_range[edge_zones] + 8.0)
        acutance_p80 = float(np.percentile(acutance_vals, 80))
    else:
        acutance_p80 = 0.25

    if acutance_p80 < 0.20:
        return False, f"Objek blur di luar DoF (acutance={acutance_p80:.2f})"

    return True, "OK"


def compute_leaf_canny_edges(
    img_bgr: np.ndarray, canny_low: int = 30, canny_high: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan Canny Edge Map khusus area daun fokus depan (bebas ranting & bebas blur).
    """
    leaf_mask = compute_leaf_mask_no_twigs(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    g_ch = img_bgr[:, :, 1].astype(np.float32)
    r_ch = img_bgr[:, :, 2].astype(np.float32)
    b_ch = img_bgr[:, :, 0].astype(np.float32)
    exg = 2.0 * g_ch - r_ch - b_ch

    # Hilangkan ranting berlumut & bayangan gelap latar belakang dari peta Canny Panel 4
    fg_clean = (
        (leaf_mask > 0)
        & (hsv[:, :, 2] >= 85)
        & (hsv[:, :, 1] >= 40)
        & (exg >= 20.0)
    ).astype(np.uint8) * 255
    fg_clean = cv2.morphologyEx(
        fg_clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    fg_clean = _fill_small_internal_holes(fg_clean, max_hole_ratio=0.025)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    raw_edges = cv2.Canny(blurred, canny_low, canny_high)

    mask_boundary = cv2.morphologyEx(
        fg_clean, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    edges_leaf_only = cv2.bitwise_and(raw_edges, fg_clean)
    edges_leaf_only = cv2.bitwise_or(edges_leaf_only, mask_boundary)
    return edges_leaf_only, fg_clean


def _contour_geometry_metrics(contour: np.ndarray) -> Tuple[float, float, float]:
    """Menghitung (area, solidity, aspect_ratio) dari sebuah kontur."""
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)

    rect = cv2.minAreaRect(contour)
    w_r, h_r = rect[1]
    if min(w_r, h_r) < 1e-3:
        aspect_ratio = 99.0
    else:
        aspect_ratio = float(max(w_r, h_r) / min(w_r, h_r))
    return area, solidity, aspect_ratio


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Deteksi Per Helai Daun (Daun Sakit & Daun Sehat di Bidang Fokus)
# ─────────────────────────────────────────────────────────────────────────────

def compute_leaf_disease_symptom_score(img_bgr: np.ndarray, contour_abs: np.ndarray) -> Tuple[bool, float, str]:
    """
    Menganalisis bagian dalam (interior) dari satu helai daun untuk mengecek apakah
    helai daun tersebut memiliki gejala penyakit (bercak nekrosis, lubang Helopeltis,
    Gray Blight, atau bronzing Red Spider) ATAU merupakan daun hijau sehat (`Healty Leaf`).
    """
    H, W = img_bgr.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(contour_abs)
    x1, y1, x2, y2 = max(0, bx), max(0, by), min(W, bx + bw), min(H, by + bh)
    if x2 <= x1 or y2 <= y1:
        return False, 0.0, "empty"

    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cnt_local = contour_abs.copy()
    cnt_local[:, :, 0] -= x1
    cnt_local[:, :, 1] -= y1
    cv2.drawContours(mask, [cnt_local], -1, 255, -1)

    erode_k = max(3, int(min(ch, cw) * 0.05) | 1)
    inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))
    n_inner = int(np.count_nonzero(inner_mask))
    if n_inner < 80:
        inner_mask = mask
        n_inner = int(np.count_nonzero(inner_mask))
    if n_inner == 0:
        return False, 0.0, "empty"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_in = hsv[:, :, 0][inner_mask > 0].astype(np.float32)
    s_in = hsv[:, :, 1][inner_mask > 0].astype(np.float32)
    v_in = hsv[:, :, 2][inner_mask > 0].astype(np.float32)

    b_in = crop[:, :, 0][inner_mask > 0].astype(np.float32)
    g_in = crop[:, :, 1][inner_mask > 0].astype(np.float32)
    r_in = crop[:, :, 2][inner_mask > 0].astype(np.float32)

    necrotic_spots = (
        (v_in < 82.0)
        | (r_in > g_in - 3.0)
        | (h_in < 20.0)
        | (h_in > 140.0)
        | ((s_in < 40.0) & (v_in < 150.0))
    )
    spot_ratio = float(np.mean(necrotic_spots))

    bronze_pixels = (h_in >= 13.0) & (h_in <= 27.0) & (r_in >= 0.82 * g_in)
    bronze_ratio = float(np.mean(bronze_pixels))

    exg_in = 2.0 * g_in - r_in - b_in
    low_exg_ratio = float(np.mean(exg_in < 26.0))

    symptom_score = max(spot_ratio * 3.5, bronze_ratio * 1.8, low_exg_ratio * 1.2)
    is_diseased = (spot_ratio >= 0.020) or (bronze_ratio >= 0.13) or (low_exg_ratio >= 0.15)
    detail = f"spot={spot_ratio*100:.1f}%, bronze={bronze_ratio*100:.1f}%"
    return is_diseased, float(symptom_score), detail


def _build_spot_inpainted_step_image(img_bgr: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """
    Menghasilkan citra daun yang sudah diratakan bercak penyakit kecil & tulang daunnya
    menggunakan Inpainting Klorofil Lokal + 2D Median Filter (persis seperti v2.3).
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, _, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g_f = img_bgr[:, :, 1].astype(np.float32)
    r_f = img_bgr[:, :, 2].astype(np.float32)

    is_spot_pixel = ((v_ch < 95) | (r_f > g_f - 4.0) | (h_ch < 21)) & (base_mask > 0)
    spot_u8 = (is_spot_pixel.astype(np.uint8)) * 255

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(spot_u8, connectivity=8)
    small_spots = np.zeros((H, W), dtype=np.uint8)
    max_spot_px = int(H * W * 0.025)
    for lbl in range(1, num_l):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_spot_px:
            small_spots[labels == lbl] = 255

    med_large_k = max(15, int(min(H, W) * 0.032) | 1)
    bg_leaf = cv2.medianBlur(img_bgr, med_large_k)
    clean_img = img_bgr.copy()
    spot_dil = cv2.dilate(small_spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    clean_img[spot_dil > 0] = bg_leaf[spot_dil > 0]

    med_k = max(11, int(min(H, W) * 0.022) | 1)
    step_img = cv2.medianBlur(clean_img, med_k)
    step_img = cv2.bilateralFilter(step_img, 7, 35, 35)
    return step_img


def segment_field_leaves(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.014,
    max_area_ratio: float = 0.45,
    min_solidity: float = 0.50,
    max_aspect_ratio: float = 4.2,
    filter_dof_blur: bool = True,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
    """
    Memisahkan BANYAK helai daun sekaligus (`multi-deteksi`, baik daun sakit maupun sehat)
    menggunakan algoritma v2.3 yang terbukti berhasil memisahkan ~10+ kandidat daun,
    lalu menyaring hasilnya dengan `is_valid_foreground_in_focus_leaf` agar HANYA
    ranting berlumut & daun background blur/gelap yang dibuang!
    """
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)
    min_dim = min(H, W)
    base_mask = compute_leaf_mask_no_twigs(img_bgr)

    # 1. Buat citra step-edge yang bersih dari bercak kecil & tulang daun tipis
    step_img = _build_spot_inpainted_step_image(img_bgr, base_mask)
    lab = cv2.cvtColor(step_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    edges_l = cv2.Canny(l_ch, 24, 70)
    edges_a = cv2.Canny(a_ch, 16, 48)
    edges_b = cv2.Canny(b_ch, 16, 48)
    raw_inter_edges = cv2.bitwise_or(edges_l, cv2.bitwise_or(edges_a, edges_b))

    dil_e = cv2.dilate(raw_inter_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num_e, lbl_e, stats_e, _ = cv2.connectedComponentsWithStats(dil_e, connectivity=8)
    interleaf_barrier = np.zeros((H, W), dtype=np.uint8)
    min_span = int(min_dim * 0.075)
    for idx_e in range(1, num_e):
        bw_e = stats_e[idx_e, cv2.CC_STAT_WIDTH]
        bh_e = stats_e[idx_e, cv2.CC_STAT_HEIGHT]
        if max(bw_e, bh_e) >= min_span:
            interleaf_barrier[lbl_e == idx_e] = 255

    hsv_step = cv2.cvtColor(step_img, cv2.COLOR_BGR2HSV)
    dark_gap = hsv_step[:, :, 2] < 65
    interleaf_barrier[dark_gap] = 255

    # 2. Potong base_mask dengan interleaf_barrier untuk mencari pusat tiap helai daun
    cut_mask = base_mask.copy()
    cut_mask[interleaf_barrier > 0] = 0
    cut_mask = _fill_small_internal_holes(cut_mask, max_hole_ratio=0.025)
    k_open = max(5, int(min_dim * 0.010) | 1)
    cut_mask = cv2.morphologyEx(
        cut_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)),
        iterations=1,
    )

    # 3. Distance Transform untuk mencari titik pusat (seed) semua helai daun
    dist = cv2.distanceTransform(cut_mask, cv2.DIST_L2, 5)
    sigma = max(3.0, min_dim * 0.014)
    k_blur = int(sigma * 4) | 1
    dist_smooth = cv2.GaussianBlur(dist, (k_blur, k_blur), sigma)

    nms_radius = max(10, int(min_dim * 0.065))
    k_nms = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_radius * 2 + 1, nms_radius * 2 + 1))
    min_peak_dist = max(8.0, min_dim * 0.020)
    local_max = (
        (dist_smooth == cv2.dilate(dist_smooth, k_nms))
        & (dist_smooth >= min_peak_dist)
        & (cut_mask > 0)
    )

    peak_ys, peak_xs = np.where(local_max)
    peaks = sorted(
        [(float(dist[py, px]), int(px), int(py)) for py, px in zip(peak_ys, peak_xs)],
        key=lambda t: t[0],
        reverse=True,
    )

    # 4. Seeded Watershed (persis seperti v2.3)
    markers = np.zeros((H, W), dtype=np.int32)
    markers[base_mask == 0] = 1
    seed_info = []
    for idx_s, (r_peak, px, py) in enumerate(peaks, start=2):
        too_close = False
        for _, prev_r, prev_x, prev_y in seed_info:
            if np.hypot(px - prev_x, py - prev_y) < max(12.0, 0.75 * min(r_peak, prev_r)):
                too_close = True
                break
        if too_close:
            continue
        seed_info.append((idx_s, r_peak, px, py))
        rad_s = max(4, int(0.32 * r_peak))
        cv2.circle(markers, (px, py), rad_s, idx_s, -1)

    markers[base_mask == 0] = 1
    cv2.watershed(step_img, markers)

    raw_candidates = []
    yy, xx = np.ogrid[:H, :W]

    for inst_id, r_peak, px, py in seed_info:
        inst_mask = ((markers == inst_id) & (base_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(inst_mask) < int(img_area * min_area_ratio):
            continue

        reach_rx = max(28.0, 1.85 * r_peak)
        reach_ry = max(36.0, 2.55 * r_peak)
        in_reach = (((xx - px) / reach_rx) ** 2 + ((yy - py) / reach_ry) ** 2) <= 1.0
        inst_mask[~in_reach] = 0

        _, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(inst_mask, connectivity=8)
        target_cc = cc_labels[py, px]
        if target_cc == 0:
            if len(cc_stats) <= 1:
                continue
            target_cc = int(np.argmax(cc_stats[1:, cv2.CC_STAT_AREA])) + 1
        inst_mask = (cc_labels == target_cc).astype(np.uint8) * 255

        k_smooth = max(5, int(min_dim * 0.012) | 1)
        inst_mask = cv2.morphologyEx(
            inst_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_smooth, k_smooth)),
            iterations=1,
        )
        inst_mask = _fill_small_internal_holes(inst_mask, max_hole_ratio=0.04)

        cnts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area, solidity, aspect_ratio = _contour_geometry_metrics(c)
        area_ratio = area / img_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if aspect_ratio > max_aspect_ratio or solidity < min_solidity:
            continue

        eps = 0.004 * cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, eps, closed=True)

        # Seleksi Foreground DoF & Penolak Ranting Lumut tanpa mengorbankan multi-deteksi daun!
        is_valid_fg, _ = is_valid_foreground_in_focus_leaf(
            img_bgr, approx, filter_dof_blur=filter_dof_blur
        )
        if not is_valid_fg:
            continue

        bx, by, bw, bh = cv2.boundingRect(approx)
        bbox = (bx, by, bx + bw, by + bh)

        cx, cy = bx + bw / 2.0, by + bh / 2.0
        dist_center = np.hypot((cx - W / 2.0) / (W / 2.0), (cy - H / 2.0) / (H / 2.0))
        priority = area * solidity / (1.0 + 0.4 * dist_center)
        raw_candidates.append((priority, approx, bbox, solidity, inst_mask))

    raw_candidates.sort(key=lambda item: item[0], reverse=True)
    kept: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    kept_masks: List[np.ndarray] = []

    for _, approx, bbox, solidity, m_cand in raw_candidates:
        overlap = False
        cand_px = float(np.count_nonzero(m_cand))
        for m_prev in kept_masks:
            inter = float(np.count_nonzero((m_cand > 0) & (m_prev > 0)))
            min_a = min(cand_px, float(np.count_nonzero(m_prev)))
            if inter / max(min_a, 1.0) > 0.40:
                overlap = True
                break
        if not overlap:
            kept.append((approx, bbox, solidity))
            kept_masks.append(m_cand)

    return kept


def classify_leaf_instance_with_yolo(
    model,
    img_bgr: np.ndarray,
    contour_abs: np.ndarray,
    bbox: Tuple[int, int, int, int],
    imgsz: int = 640,
    device: str = "cuda",
) -> Tuple[int, float, str]:
    """
    Mengekstrak 1 helai daun menggunakan maskernya, meletakkannya di atas
    Kanvas Kertas Putih Virtual (menyerupai distribusi dataset teaLeafBD),
    dan mengklasifikasikan kelasnya (baik kelas Penyakit 0..5 maupun `Healty Leaf` 6).
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(6, int(max(x2 - x1, y2 - y1) * 0.04))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)

    crop = img_bgr[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    mask_crop = np.zeros((ch, cw), dtype=np.uint8)
    cnt_shifted = contour_abs.copy()
    cnt_shifted[:, :, 0] -= cx1
    cnt_shifted[:, :, 1] -= cy1
    cv2.drawContours(mask_crop, [cnt_shifted], -1, 255, -1)
    mask_crop = cv2.dilate(
        mask_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
    )

    paper_color = np.array([236, 238, 238], dtype=np.uint8)
    isolated_leaf = np.full_like(crop, paper_color)
    isolated_leaf[mask_crop > 0] = crop[mask_crop > 0]

    side = int(max(ch, cw) * 1.36)
    canvas = np.full((side, side, 3), paper_color, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    canvas[y_off : y_off + ch, x_off : x_off + cw] = isolated_leaf

    has_symptom, symptom_score, _ = compute_leaf_disease_symptom_score(img_bgr, contour_abs)

    res = model.predict(canvas, conf=0.05, imgsz=imgsz, device=device, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        if not has_symptom:
            healthy_idxs = [i for i, c in enumerate(clss) if c == 6]
            if healthy_idxs:
                best_h = max(healthy_idxs, key=lambda i: confs[i])
                return 6, max(float(confs[best_h]), 0.80), CLASS_NAMES[6]
            if symptom_score < 0.050:
                return 6, 0.84, CLASS_NAMES[6]

        disease_idxs = [i for i, c in enumerate(clss) if c != 6]
        if has_symptom and disease_idxs:
            best_i = max(disease_idxs, key=lambda i: confs[i])
        else:
            best_i = int(np.argmax(confs))

        cid = int(clss[best_i])
        conf = float(confs[best_i])
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        return cid, conf, cname

    if has_symptom:
        return 1, 0.65, CLASS_NAMES[1]
    return 6, 0.82, CLASS_NAMES[6]


class EdgeContourPostProcessor:
    """
    Post-processor yang mengubah bounding box YOLO menjadi polygon kontur daun,
    menolak objek ranting kayu & objek blur di luar Depth-of-Field, serta
    memfilter daun yang terpotong batas gambar/ROI.
    """

    CLASS_COLORS = CLASS_COLORS_LIST

    def __init__(self, config: Optional[EdgeContourConfig] = None):
        self.cfg = config or EdgeContourConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid),
        )
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_close_ksize, self.cfg.morph_close_ksize),
        )
        self._kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.cfg.morph_dilate_ksize, self.cfg.morph_dilate_ksize),
        )

    def _find_leaf_contour(self, roi_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, float, float]:
        H, W = roi_bgr.shape[:2]
        roi_area = float(H * W)

        if self.cfg.filter_twigs:
            leaf_mask = compute_leaf_mask_no_twigs(roi_bgr)
            green_ratio = float(np.count_nonzero(leaf_mask)) / max(roi_area, 1.0)
            if green_ratio < 0.08:
                return None, f"Bukan daun / didominasi ranting (rasio klorofil {green_ratio*100:.1f}%)", 0.0, 0.0

            closed_mask = cv2.morphologyEx(
                leaf_mask,
                cv2.MORPH_CLOSE,
                self._kernel_close,
                iterations=self.cfg.morph_iterations,
            )
            closed_mask = _fill_small_internal_holes(closed_mask, max_hole_ratio=0.04)
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = []

        if not contours:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)
            ksize = self.cfg.gaussian_ksize if self.cfg.gaussian_ksize % 2 == 1 else self.cfg.gaussian_ksize + 1
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            edges = cv2.Canny(blurred, self.cfg.canny_low, self.cfg.canny_high)
            if self.cfg.filter_twigs:
                edges = cv2.bitwise_and(edges, cv2.dilate(leaf_mask, self._kernel_dilate, iterations=2))

            closed = cv2.morphologyEx(
                edges,
                cv2.MORPH_CLOSE,
                self._kernel_close,
                iterations=self.cfg.morph_iterations,
            )
            dilated = cv2.dilate(closed, self._kernel_dilate, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, "Tidak ada kontur daun ditemukan setelah filter ranting", 0.0, 0.0

        min_area_px = self.cfg.min_contour_ratio * roi_area
        valid_contours = []
        for c in contours:
            c_area, _, c_ar = _contour_geometry_metrics(c)
            if c_area < min_area_px:
                continue
            if self.cfg.filter_twigs and c_ar > self.cfg.max_aspect_ratio:
                continue
            valid_contours.append(c)

        if not valid_contours:
            return None, "Objek ditolak (berbentuk ranting memanjang atau terlalu kecil)", 0.0, 0.0

        largest = max(valid_contours, key=cv2.contourArea)
        contour_area, solidity, _ = _contour_geometry_metrics(largest)

        epsilon = self.cfg.approx_epsilon_ratio * cv2.arcLength(largest, closed=True)
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)

        return approx, "OK", float(contour_area), float(solidity)

    def _is_contour_complete(
        self, contour: np.ndarray, roi_h: int, roi_w: int, solidity: float
    ) -> Tuple[bool, str]:
        margin = self.cfg.border_margin
        pts = contour.reshape(-1, 2)

        touches_left = np.any(pts[:, 0] <= margin)
        touches_right = np.any(pts[:, 0] >= roi_w - margin)
        touches_top = np.any(pts[:, 1] <= margin)
        touches_bottom = np.any(pts[:, 1] >= roi_h - margin)

        if touches_left or touches_right or touches_top or touches_bottom:
            sides = []
            if touches_left:
                sides.append("kiri")
            if touches_right:
                sides.append("kanan")
            if touches_top:
                sides.append("atas")
            if touches_bottom:
                sides.append("bawah")
            return False, f"Kontur menyentuh tepi ({', '.join(sides)}) -> daun terpotong"

        if solidity < self.cfg.min_solidity:
            return False, f"Solidity {solidity:.2f} < {self.cfg.min_solidity:.2f} -> kontur tidak utuh"

        return True, "Daun utuh"

    def process_detections(self, img_bgr: np.ndarray, detections: List[Detection]) -> List[ContourResult]:
        H_img, W_img = img_bgr.shape[:2]
        results = []

        for det in detections:
            x1 = max(0, det.x1)
            y1 = max(0, det.y1)
            x2 = min(W_img, det.x2)
            y2 = min(H_img, det.y2)

            if x2 <= x1 or y2 <= y1:
                results.append(
                    ContourResult(
                        detection=det,
                        is_complete=False,
                        contour=None,
                        reason="Bounding box tidak valid (zero area)",
                    )
                )
                continue

            roi = img_bgr[y1:y2, x1:x2]
            contour_roi, reason, c_area, solidity = self._find_leaf_contour(roi)

            if contour_roi is None:
                if self.cfg.fallback_to_bbox:
                    bbox_poly = np.array(
                        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32
                    )
                    results.append(
                        ContourResult(
                            detection=det,
                            is_complete=True,
                            contour=bbox_poly,
                            reason=f"Fallback bbox: {reason}",
                            contour_area=float((x2 - x1) * (y2 - y1)),
                            solidity=1.0,
                            fallback_used=True,
                        )
                    )
                else:
                    results.append(
                        ContourResult(
                            detection=det, is_complete=False, contour=None, reason=reason
                        )
                    )
                continue

            contour_abs = contour_roi.copy()
            contour_abs[:, :, 0] += x1
            contour_abs[:, :, 1] += y1

            is_complete, complete_reason = self._is_contour_complete(
                contour_abs, H_img, W_img, solidity
            )

            results.append(
                ContourResult(
                    detection=det,
                    is_complete=is_complete,
                    contour=contour_abs,
                    reason=complete_reason,
                    contour_area=c_area,
                    solidity=solidity,
                )
            )

        return results

    def process_field_image_with_model(
        self,
        model,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.25,
        imgsz: int = 640,
        device: str = "cuda",
    ) -> Tuple[List[Detection], List[ContourResult], bool]:
        """
        Pipeline Multi-Deteksi untuk gambar eksternal / lapangan:
        Memisahkan banyak helai daun sekaligus (seperti v2.3) dan menolak ranting berlumut
        serta daun latar belakang yang gelap/blur (di luar Depth-of-Field).
        """
        H, W = img_bgr.shape[:2]
        img_area = float(H * W)

        yolo_res = model.predict(img_bgr, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        raw_dets = yolo_result_to_detections(yolo_res)

        needs_field_seg = False
        if not is_white_paper_background(img_bgr):
            if len(raw_dets) <= 1:
                needs_field_seg = True
            else:
                for d in raw_dets:
                    box_area = float((d.x2 - d.x1) * (d.y2 - d.y1))
                    if box_area / img_area > 0.50:
                        needs_field_seg = True
                        break

        if not needs_field_seg:
            c_results = self.process_detections(img_bgr, raw_dets)
            if any(r.contour is not None for r in c_results):
                return raw_dets, c_results, False

        leaf_instances = segment_field_leaves(
            img_bgr,
            min_area_ratio=0.014,
            max_area_ratio=0.45,
            min_solidity=self.cfg.min_solidity,
            max_aspect_ratio=self.cfg.max_aspect_ratio,
            filter_dof_blur=self.cfg.filter_dof_blur,
        )

        all_dets: List[Detection] = []
        all_contours: List[ContourResult] = []
        diseased_dets: List[Detection] = []
        diseased_contours: List[ContourResult] = []

        for cnt_abs, (x1, y1, x2, y2), sol in leaf_instances:
            has_symptom, _, _ = compute_leaf_disease_symptom_score(img_bgr, cnt_abs)
            cid, conf, cname = classify_leaf_instance_with_yolo(
                model, img_bgr, cnt_abs, (x1, y1, x2, y2), imgsz=imgsz, device=device
            )

            det = Detection(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                class_id=cid,
                confidence=conf,
                class_name=cname,
            )
            c_area = float(cv2.contourArea(cnt_abs))
            is_comp, comp_reason = self._is_contour_complete(cnt_abs, H, W, sol)

            c_res = ContourResult(
                detection=det,
                is_complete=is_comp,
                contour=cnt_abs,
                reason=comp_reason,
                contour_area=c_area,
                solidity=sol,
            )
            all_dets.append(det)
            all_contours.append(c_res)

            if cid != 6 and has_symptom:
                diseased_dets.append(det)
                diseased_contours.append(c_res)

        if self.cfg.only_diseased_in_field and len(diseased_dets) > 0:
            return diseased_dets, diseased_contours, True

        return all_dets, all_contours, True

    def draw_results(
        self,
        img_bgr: np.ndarray,
        results: List[ContourResult],
        show_suppressed: bool = False,
        show_bbox: bool = False,
        alpha_fill: float = 0.22,
    ) -> np.ndarray:
        out_img = img_bgr.copy()
        overlay = out_img.copy()

        if show_suppressed:
            for res in results:
                if not res.is_complete:
                    det = res.detection
                    supp_color = (0, 110, 255)
                    if res.contour is not None:
                        _draw_dashed_contour(out_img, res.contour, supp_color, thickness=2)
                        lx, ly = int(res.contour[:, 0, 0].min()), int(res.contour[:, 0, 1].min())
                    else:
                        cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), supp_color, 1)
                        lx, ly = det.x1, det.y1
                    short_txt = f"[TEPI] {det.class_name} {det.confidence:.2f}"
                    _put_label(out_img, short_txt, max(4, lx), max(18, ly + 15), supp_color, scale=0.44)

        for res in results:
            if not res.is_complete:
                continue
            det = res.detection
            color = self.CLASS_COLORS[det.class_id % len(self.CLASS_COLORS)]
            label = f"{det.class_name} {det.confidence:.2f}"

            if show_bbox:
                cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), (200, 200, 200), 1)

            if res.contour is not None and self.cfg.draw_contour:
                cv2.fillPoly(overlay, [res.contour], color)
                cv2.drawContours(out_img, [res.contour], -1, color, 3)
                lx, ly = int(res.contour[:, 0, 0].min()), int(res.contour[:, 0, 1].min())
            else:
                cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                lx, ly = det.x1, det.y1

            _put_label(out_img, label, max(4, lx), max(18, ly), color, scale=0.50)

        cv2.addWeighted(overlay, alpha_fill, out_img, 1 - alpha_fill, 0, out_img)
        return out_img

    def visualize(
        self,
        img_bgr: np.ndarray,
        results: List[ContourResult],
        draw_contours: bool = True,
        draw_orig_bbox: bool = False,
        show_suppressed: bool = True,
        alpha_fill: float = 0.22,
    ) -> np.ndarray:
        """Alias kompatibel untuk script yang memanggil processor.visualize(...)."""
        prev_draw = self.cfg.draw_contour
        self.cfg.draw_contour = draw_contours
        vis = self.draw_results(
            img_bgr,
            results,
            show_suppressed=show_suppressed,
            show_bbox=draw_orig_bbox,
            alpha_fill=alpha_fill,
        )
        self.cfg.draw_contour = prev_draw
        return vis

    @staticmethod
    def summarize(results: List[ContourResult]) -> dict:
        total = len(results)
        complete = sum(1 for r in results if r.is_complete)
        suppressed = total - complete
        fallback = sum(1 for r in results if r.fallback_used)
        solidities = [r.solidity for r in results if r.is_complete]

        return {
            "total_detections": total,
            "complete_leaves": complete,
            "suppressed_leaves": suppressed,
            "suppression_rate": suppressed / max(total, 1),
            "fallback_used": fallback,
            "mean_solidity": float(np.mean(solidities)) if solidities else 0.0,
            "suppression_reasons": [r.reason for r in results if not r.is_complete],
        }


def _draw_dashed_contour(img: np.ndarray, contour: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
    pts = contour.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        if i % 2 == 0:
            p1 = tuple(int(v) for v in pts[i])
            p2 = tuple(int(v) for v in pts[(i + 1) % n])
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _put_label(img, text, x, y, color, scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_top = max(y - th - baseline - 4, 0)
    x_right = min(x + tw + 4, img.shape[1] - 1)
    cv2.rectangle(img, (x, y_top), (x_right, y), color, -1)
    cv2.putText(
        img,
        text,
        (x + 2, max(y - baseline - 1, th)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def yolo_result_to_detections(yolo_result) -> List[Detection]:
    detections = []
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return detections

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        confidence = float(box.cls[0]) if not hasattr(box, "conf") else float(box.conf[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
        detections.append(Detection(x1, y1, x2, y2, class_id, confidence, class_name))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


if __name__ == "__main__":
    print("=" * 65)
    print("  Unit Test: EdgeContourPostProcessor + Multi-Leaf DoF (v2.5)")
    print("=" * 65)
    proc = EdgeContourPostProcessor()

    img_test = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.ellipse(img_test, (320, 240), (150, 95), 0, 0, 360, (35, 145, 45), -1)
    cv2.line(img_test, (50, 50), (580, 70), (30, 70, 120), 8)

    det_leaf = Detection(150, 130, 490, 350, class_id=6, confidence=0.94, class_name="Healty Leaf")
    det_twig = Detection(40, 40, 590, 80, class_id=1, confidence=0.60, class_name="Brown Blight")

    res_list = proc.process_detections(img_test, [det_leaf, det_twig])
    print(f"  [TEST 1 - Daun Utuh]     is_complete={res_list[0].is_complete} | reason='{res_list[0].reason}'")
    print(f"  [TEST 2 - Ranting Kayu]  is_complete={res_list[1].is_complete} | reason='{res_list[1].reason}'")
    ok = res_list[0].is_complete and (not res_list[1].is_complete)
    print(f"  Status Unit Test: {'LULUS 100%' if ok else 'PERIKSA KEMBALI'}")
    print("=" * 65)
