"""
FASE 3 (DIPERBARUI v2.1) - Edge Contour & Twig-Filter Postprocessing Module
===========================================================================
Mengubah bounding box YOLO26n+CA menjadi POLYGON KONTUR DAUN menggunakan:
1. Filter Spektral Klorofil & Eliminasi Ranting (Excess Green ExG + HSV Twig Gating
   + Internal Lesion Hole-Filling) agar ranting kayu coklat/abu-abu tidak ikut
   terdeteksi sebagai daun maupun bercak penyakit.
2. Multi-Leaf Field Instance Segmentation (Distance Transform + Shadow-Cut Watershed)
   untuk memisahkan helai-helai daun pada foto semak kebun (in-situ) dan mengatasi
   Domain Shift (1 kotak penuh) melalui inferensi Kanvas Kertas Virtual per helai daun.
3. Canny Edge Detection khusus area daun (bebas ranting) + pemeriksaan kelengkapan
   kontur daun (Utuh vs Terpotong batas citra).
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
    """Konfigurasi parameter untuk Edge Contour & Twig Filter Post-Processing."""
    gaussian_ksize: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    canny_low: int = 30
    canny_high: int = 100
    morph_close_ksize: int = 7
    morph_dilate_ksize: int = 3
    morph_iterations: int = 2
    border_margin: int = 6
    min_contour_ratio: float = 0.04
    min_solidity: float = 0.50
    max_aspect_ratio: float = 4.2   # Menolak kontur memanjang seperti ranting (L/W > 4.2)
    filter_twigs: bool = True       # Aktifkan filter pemisah daun vs ranting
    max_contour_points: int = 100
    approx_epsilon_ratio: float = 0.004
    fallback_to_bbox: bool = False  # Jangan fallback ke kotak penuh jika kontur adalah ranting
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
# Fungsi Inti Pemisah Daun vs Ranting (Spectral Chlorophyll + Twig Suppression)
# ─────────────────────────────────────────────────────────────────────────────

def is_white_paper_background(img_bgr: np.ndarray) -> bool:
    """
    Mendeteksi apakah gambar berasal dari studio/kertas putih seperti dataset teaLeafBD
    atau foto kebun lapangan (in-situ).
    """
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


def _fill_small_internal_holes(binary_mask: np.ndarray, max_hole_ratio: float = 0.025) -> np.ndarray:
    """
    Mengisi lubang-lubang kecil di bagian dalam lamina daun (seperti bercak coklat
    Brown Blight / Helopeltis atau lubang gigitan serangga) TANPA mengisi celah
    latar belakang / ranting di luar daun.
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
    Menghasilkan binary mask (0 atau 255) yang HANYA berisi area helai daun teh
    dan secara aktif membuang:
    - Ranting kayu coklat / kemerahan (R >= G, Hue < 20 atau > 155, ExG rendah)
    - Ranting kering / lumut abu-abu (Saturasi rendah & ExG rendah)
    - Bayangan gelap antar-daun (Value rendah)
    Namun tetap mempertahankan bercak penyakit di dalam helai daun melalui
    Internal Hole Filling.
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    b_f = img_bgr[:, :, 0].astype(np.float32)
    g_f = img_bgr[:, :, 1].astype(np.float32)
    r_f = img_bgr[:, :, 2].astype(np.float32)

    # Excess Green Index (ExG = 2G - R - B)
    exg = 2.0 * g_f - r_f - b_f

    if is_white_paper_background(img_bgr):
        # Mode Lab / Kertas Putih (teaLeafBD): semua objek non-kertas adalah daun,
        # tetapi tetap buang tangkai/ranting tipis dengan morphological opening
        not_paper = ~((v_ch > 160) & (s_ch < 50))
        not_skin = ~((r_f - b_f > 35) & (r_f - g_f > 15) & (v_ch > 90))
        raw_mask = ((not_paper & not_skin) | (exg > 15)).astype(np.uint8) * 255
        raw_mask = _fill_small_internal_holes(raw_mask, max_hole_ratio=0.08)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # Mode Lapangan / Kebun (In-Situ):
    # 1. Deteksi eksplisit piksel ranting kayu (coklat, kemerahan, atau abu-abu kering)
    is_brown_twig = (
        (h_ch < 19) | (h_ch > 150) | (r_f > g_f + 4.0) | ((exg < 14.0) & (r_f >= g_f - 6.0))
    )
    is_gray_twig = (s_ch < 42) & (exg < 22.0)
    is_dark_shadow = (v_ch < 48)

    # 2. Deteksi klorofil lamina daun (hijau hingga kuning-hijau pucuk teh)
    is_green_lamina = (
        (h_ch >= 19)
        & (h_ch <= 88)
        & (s_ch >= 40)
        & (v_ch >= 48)
        & (exg > 14.0)
        & (g_f > b_f + 10.0)
        & (~is_brown_twig)
        & (~is_gray_twig)
        & (~is_dark_shadow)
    )

    mask = (is_green_lamina.astype(np.uint8)) * 255

    # 3. Isi bercak penyakit coklat / lubang serangga yang berada DI DALAM helai daun
    mask = _fill_small_internal_holes(mask, max_hole_ratio=0.022)

    # 4. Putuskan jembatan ranting / tangkai tipis dengan Morphological Opening
    open_k = max(5, int(min(H, W) * 0.012) | 1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 5. Isi kembali lubang internal kecil setelah opening
    mask = _fill_small_internal_holes(mask, max_hole_ratio=0.022)
    return mask


def compute_leaf_canny_edges(
    img_bgr: np.ndarray, canny_low: int = 30, canny_high: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan Canny Edge Map yang HANYA menampilkan garis tepi daun & bercak
    di dalam daun, sedangkan seluruh ranting kayu di latar belakang dieliminasi.
    Returns:
        edges_leaf_only: Peta tepi Canny bebas ranting (H, W) uint8
        leaf_mask      : Masker biner area daun bebas ranting (H, W) uint8
    """
    leaf_mask = compute_leaf_mask_no_twigs(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    raw_edges = cv2.Canny(blurred, canny_low, canny_high)

    # Tambahkan kontur batas masker daun agar siluet helai daun tampak tegas
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(leaf_mask, k_dil, iterations=1)
    mask_boundary = cv2.morphologyEx(leaf_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    edges_leaf_only = cv2.bitwise_and(raw_edges, mask_dilated)
    edges_leaf_only = cv2.bitwise_or(edges_leaf_only, mask_boundary)
    return edges_leaf_only, leaf_mask


def _contour_geometry_metrics(contour: np.ndarray) -> Tuple[float, float, float]:
    """
    Menghitung (area, solidity, aspect_ratio) dari sebuah kontur.
    Ranting kayu memiliki aspect_ratio sangat besar (> 4.5) atau solidity rendah (< 0.45).
    """
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


def segment_field_leaves(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.020,
    max_area_ratio: float = 0.65,
    min_solidity: float = 0.52,
    max_aspect_ratio: float = 4.2,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
    """
    Memisahkan helai-helai daun individual dari citra semak kebun (in-situ) menggunakan:
    1. Masker Klorofil Bebas-Ranting (`compute_leaf_mask_no_twigs`)
    2. Pemotongan Celah Bayangan & Gradien Antar-Daun (Inter-Leaf Crease Cut)
    3. Euclidean Distance Transform + Local Maxima Seeded Watershed
    4. Seleksi Geometri Helai Daun (menolak objek tipis memanjang seperti ranting)

    Returns:
        List of (contour_abs, (x1, y1, x2, y2), solidity) diurutkan dari daun paling
        dominan/tengah ke tepi.
    """
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)
    base_mask = compute_leaf_mask_no_twigs(img_bgr)

    # Potong celah bayangan gelap / tepi batas antar daun yang saling menempel
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 45, 45)
    strong_edges = cv2.Canny(smooth, 40, 110)

    # Hanya potong pada garis tepi yang sekaligus gelap (celah bayangan antar daun),
    # bukan bercak di tengah daun yang sudah di-fill
    inter_leaf_crease = ((strong_edges > 0) & (v_ch < 105)) | (v_ch < 58)
    cut_mask = base_mask.copy()
    cut_mask[inter_leaf_crease] = 0

    k_open = max(5, int(min(H, W) * 0.012) | 1)
    cut_mask = cv2.morphologyEx(
        cut_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)),
        iterations=1,
    )
    cut_mask = _fill_small_internal_holes(cut_mask, max_hole_ratio=0.018)

    # Distance Transform untuk menemukan pusat masing-masing helai daun
    dist = cv2.distanceTransform(cut_mask, cv2.DIST_L2, 5)
    sigma = max(3.0, min(H, W) * 0.018)
    k_blur = int(sigma * 4) | 1
    dist_smooth = cv2.GaussianBlur(dist, (k_blur, k_blur), sigma)

    # Cari local maxima (puncak tengah tiap helai daun) dengan non-maximum suppression
    nms_radius = max(15, int(min(H, W) * 0.12))
    k_nms = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_radius * 2 + 1, nms_radius * 2 + 1))
    local_max = (dist_smooth == cv2.dilate(dist_smooth, k_nms)) & (
        dist_smooth >= max(10.0, min(H, W) * 0.032)
    ) & (cut_mask > 0)

    num_seeds, seed_labels = cv2.connectedComponents(local_max.astype(np.uint8))

    # Siapkan marker untuk Watershed:
    # 1 = Background (bukan daun / ranting), 2..N+1 = Kandidat helai daun
    markers = np.zeros((H, W), dtype=np.int32)
    markers[base_mask == 0] = 1
    for s_id in range(1, num_seeds):
        # Perluas sedikit titik puncak sebagai core marker
        seed_mask = (seed_labels == s_id).astype(np.uint8) * 255
        seed_dil = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        markers[(seed_dil > 0) & (base_mask > 0)] = s_id + 1

    # Jalankan Watershed pada citra terfilter bilateral
    img_ws = cv2.bilateralFilter(img_bgr, 7, 40, 40)
    cv2.watershed(img_ws, markers)

    leaf_instances = []
    k_close_inst = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(7, int(min(H, W) * 0.018) | 1),) * 2
    )

    for s_id in range(1, num_seeds):
        inst_id = s_id + 1
        inst_mask = ((markers == inst_id) & (base_mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(inst_mask) < int(img_area * min_area_ratio):
            continue

        # Haluskan bentuk helai daun & isi lubang bercak internalnya
        inst_mask = cv2.morphologyEx(inst_mask, cv2.MORPH_CLOSE, k_close_inst, iterations=1)
        inst_mask = _fill_small_internal_holes(inst_mask, max_hole_ratio=0.03)

        cnts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area, solidity, aspect_ratio = _contour_geometry_metrics(c)
        area_ratio = area / img_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if aspect_ratio > max_aspect_ratio:
            # Kontur terlalu tipis/memanjang -> ranting
            continue
        if solidity < min_solidity:
            # Jika solidity agak rendah akibat tepi bergerigi, cek apakah convex hull-nya berbentuk daun
            hull = cv2.convexHull(c)
            _, _, hull_ar = _contour_geometry_metrics(hull)
            if solidity < 0.42 or hull_ar > max_aspect_ratio:
                continue

        eps = 0.004 * cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, eps, closed=True)
        bx, by, bw, bh = cv2.boundingRect(approx)
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        # Skor prioritas: utamakan daun yang dekat ke pusat foto & berukuran wajar
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist_center = np.hypot((cx - W / 2.0) / (W / 2.0), (cy - H / 2.0) / (H / 2.0))
        priority = area * solidity / (1.0 + 0.8 * dist_center)
        leaf_instances.append((priority, approx, (x1, y1, x2, y2), solidity))

    leaf_instances.sort(key=lambda item: item[0], reverse=True)
    return [(cnt, bbox, sol) for _, cnt, bbox, sol in leaf_instances]


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
    lalu menjalankan YOLO26n+CA untuk memprediksi kelas penyakit daun tersebut
    tanpa gangguan ranting/semak di latar belakang.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(6, int(max(x2 - x1, y2 - y1) * 0.04))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)

    crop = img_bgr[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    # Buat masker kontur pada koordinat crop, dilasi sedikit agar tepi daun alami tetap utuh
    mask_crop = np.zeros((ch, cw), dtype=np.uint8)
    cnt_shifted = contour_abs.copy()
    cnt_shifted[:, :, 0] -= cx1
    cnt_shifted[:, :, 1] -= cy1
    cv2.drawContours(mask_crop, [cnt_shifted], -1, 255, -1)
    mask_crop = cv2.dilate(
        mask_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
    )

    # Kanvas kertas putih keabu-abuan seperti latar teaLeafBD (BGR: 236, 238, 238)
    paper_color = np.array([236, 238, 238], dtype=np.uint8)
    isolated_leaf = np.full_like(crop, paper_color)
    isolated_leaf[mask_crop > 0] = crop[mask_crop > 0]

    # Tempatkan di tengah kanvas persegi dengan margin 18%
    side = int(max(ch, cw) * 1.36)
    canvas = np.full((side, side, 3), paper_color, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    canvas[y_off : y_off + ch, x_off : x_off + cw] = isolated_leaf

    res = model.predict(canvas, conf=0.05, imgsz=imgsz, device=device, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        # Ambil prediksi dengan confidence tertinggi
        confs = res.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
        cid = int(res.boxes.cls[best_i])
        conf = float(confs[best_i])
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        return cid, conf, cname

    # Fallback jika confidence sangat rendah: cek rasio bercak coklat di dalam daun
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    inside = mask_crop > 0
    if np.count_nonzero(inside) > 0:
        h_in = hsv_crop[:, :, 0][inside]
        s_in = hsv_crop[:, :, 1][inside]
        lesion_ratio = float(np.mean(((h_in < 20) | (h_in > 150)) & (s_in > 45)))
        if lesion_ratio > 0.06:
            return 1, 0.55, CLASS_NAMES[1]  # Brown Blight
    return 6, 0.60, CLASS_NAMES[6]          # Healty Leaf


class EdgeContourPostProcessor:
    """
    Post-processor yang mengubah bounding box YOLO menjadi polygon kontur daun,
    menolak objek ranting kayu, dan memfilter daun yang terpotong batas gambar/ROI.
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

        # 1. Gunakan masker daun bebas-ranting terlebih dahulu jika filter_twigs aktif
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
            closed_mask = _fill_small_internal_holes(closed_mask, max_hole_ratio=0.05)
            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours = []

        # 2. Fallback ke Canny ter-masking jika belum ada kontur
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
            c_area, c_sol, c_ar = _contour_geometry_metrics(c)
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
            roi_h, roi_w = roi.shape[:2]
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

            # Periksa apakah kontur menyentuh batas luar gambar (frame kamera)
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
        Pipeline cerdas untuk gambar eksternal / lapangan:
        1. Jalankan prediksi langsung YOLO26n+CA.
        2. Evaluasi apakah terjadi Domain Shift (misal YOLO memprediksi 1 kotak penuh > 65%
           area gambar karena latar semak/ranting bukan kertas putih, atau tidak ada kotak
           daun valid).
        3. Jika terjadi Domain Shift pada foto semak lapangan, aktifkan segmentasi helai
           daun bebas-ranting (`segment_field_leaves`) lalu klasifikasikan tiap helai daun
           menggunakan YOLO26n+CA pada Kanvas Kertas Virtual.

        Returns:
            (detections, contour_results, used_field_segmentation)
        """
        H, W = img_bgr.shape[:2]
        img_area = float(H * W)

        yolo_res = model.predict(img_bgr, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        raw_dets = yolo_result_to_detections(yolo_res)

        # Cek apakah hasil langsung YOLO valid atau justru terkena Domain Shift (kotak raksasa / ranting)
        needs_field_seg = False
        if not is_white_paper_background(img_bgr):
            if len(raw_dets) == 0:
                needs_field_seg = True
            else:
                for d in raw_dets:
                    box_area = float((d.x2 - d.x1) * (d.y2 - d.y1))
                    if box_area / img_area > 0.65:
                        # Kotak membungkus hampir seluruh layar (gejala Domain Shift latar semak)
                        needs_field_seg = True
                        break

        if not needs_field_seg:
            c_results = self.process_detections(img_bgr, raw_dets)
            valid_any = any(r.contour is not None for r in c_results)
            if valid_any:
                return raw_dets, c_results, False

        # Aktifkan Multi-Leaf Field Instance Segmentation (Bebas Ranting)
        leaf_instances = segment_field_leaves(
            img_bgr,
            min_area_ratio=0.020,
            max_area_ratio=0.65,
            min_solidity=self.cfg.min_solidity,
            max_aspect_ratio=self.cfg.max_aspect_ratio,
        )

        field_dets: List[Detection] = []
        field_contours: List[ContourResult] = []

        for cnt_abs, (x1, y1, x2, y2), sol in leaf_instances:
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

            field_dets.append(det)
            field_contours.append(
                ContourResult(
                    detection=det,
                    is_complete=is_comp,
                    contour=cnt_abs,
                    reason=comp_reason,
                    contour_area=c_area,
                    solidity=sol,
                )
            )

        return field_dets, field_contours, True

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

        # 1. Gambar daun yang TERPOTONG / SUPPRESSED terlebih dahulu (dengan garis putus-putus tanpa blok warna penuh)
        if show_suppressed:
            for res in results:
                if not res.is_complete:
                    det = res.detection
                    supp_color = (0, 69, 255)  # Oranye-Merah
                    if res.contour is not None:
                        _draw_dashed_contour(out_img, res.contour, supp_color, thickness=2)
                        lx, ly = int(res.contour[:, 0, 0].min()), int(res.contour[:, 0, 1].min())
                    else:
                        cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), supp_color, 1)
                        lx, ly = det.x1, det.y1
                    short_txt = f"[TERPOTONG] {det.class_name}"
                    _put_label(out_img, short_txt, max(4, lx), max(18, ly + 15), supp_color, scale=0.45)

        # 2. Gambar daun yang UTUH (DITERIMA) dengan poligon kontur penuh yang jelas
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

            _put_label(out_img, label, max(4, lx), max(18, ly), color, scale=0.52)

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
    """Menggambar garis kontur putus-putus untuk menandai daun yang terpotong batas gambar."""
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
        confidence = float(box.conf[0])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
        detections.append(Detection(x1, y1, x2, y2, class_id, confidence, class_name))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


if __name__ == "__main__":
    print("=" * 65)
    print("  Unit Test: EdgeContourPostProcessor + Twig Filter (Fase 3 v2.1)")
    print("=" * 65)
    proc = EdgeContourPostProcessor()

    # Buat citra sintetis: latar putih + 1 daun hijau utuh + 1 ranting coklat memanjang
    img_test = np.full((480, 640, 3), 240, dtype=np.uint8)
    # Gambar daun hijau elips
    cv2.ellipse(img_test, (320, 240), (150, 95), 0, 0, 360, (35, 145, 45), -1)
    # Gambar ranting coklat tipis di bagian atas
    cv2.line(img_test, (50, 50), (580, 70), (30, 70, 120), 8)

    det_leaf = Detection(150, 130, 490, 350, class_id=6, confidence=0.94, class_name="Healty Leaf")
    det_twig = Detection(40, 40, 590, 80, class_id=1, confidence=0.60, class_name="Brown Blight")

    res_list = proc.process_detections(img_test, [det_leaf, det_twig])
    print(f"  [TEST 1 - Daun Utuh]     is_complete={res_list[0].is_complete} | reason='{res_list[0].reason}'")
    print(f"  [TEST 2 - Ranting Kayu]  is_complete={res_list[1].is_complete} | reason='{res_list[1].reason}'")
    ok = res_list[0].is_complete and (not res_list[1].is_complete)
    print(f"  Status Unit Test: {'LULUS 100%' if ok else 'PERIKSA KEMBALI'}")
    print("=" * 65)
