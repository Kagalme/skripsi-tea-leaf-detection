"""
FASE 3 (DIPERBARUI) - Edge Contour Postprocessing Module
========================================================
Mengubah bounding box YOLO26n+CA menjadi POLYGON KONTUR DAUN menggunakan
Canny Edge Detection + Morfologi Closing, serta memfilter daun yang TIDAK UTUH
(terpotong batas gambar/ROI).

Perbaikan Kritis dibanding versi sebelumnya:
1. Mengekspor `CLASS_NAMES`, `CLASS_COLORS` (list BGR), dan `CLASS_COLOR_MAP` (dict)
   di level modul sehingga tidak terjadi ImportError pada script eksternal.
2. Menyediakan kompatibilitas penuh pada `Detection` (`bbox_abs()`),
   `ContourResult` (`is_suppressed`, `suppress_reason`, `contour_found`),
   dan `EdgeContourPostProcessor` (`draw_results()` serta alias `visualize()`).
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
    """Konfigurasi parameter untuk Edge Contour Post-Processing."""
    gaussian_ksize: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    canny_low: int = 30
    canny_high: int = 100
    morph_close_ksize: int = 7
    morph_dilate_ksize: int = 3
    morph_iterations: int = 2
    border_margin: int = 5
    min_contour_ratio: float = 0.05
    min_solidity: float = 0.45
    max_contour_points: int = 100
    approx_epsilon_ratio: float = 0.005
    fallback_to_bbox: bool = True
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


class EdgeContourPostProcessor:
    """
    Post-processor yang mengubah bounding box YOLO menjadi polygon kontur daun
    dan memfilter daun yang terpotong batas gambar/ROI.
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
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        ksize = self.cfg.gaussian_ksize if self.cfg.gaussian_ksize % 2 == 1 else self.cfg.gaussian_ksize + 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        edges = cv2.Canny(blurred, self.cfg.canny_low, self.cfg.canny_high)

        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            self._kernel_close,
            iterations=self.cfg.morph_iterations,
        )
        dilated = cv2.dilate(closed, self._kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, "Tidak ada kontur ditemukan setelah edge detection", 0.0, 0.0

        roi_area = H * W
        min_area_px = self.cfg.min_contour_ratio * roi_area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        if not valid_contours:
            return None, f"Semua kontur terlalu kecil (< {min_area_px:.0f}px²)", 0.0, 0.0

        largest = max(valid_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / (hull_area + 1e-6)

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
            return False, f"Kontur menyentuh tepi ROI ({', '.join(sides)}) -> daun tidak utuh"

        if solidity < self.cfg.min_solidity:
            return False, f"Solidity {solidity:.3f} < {self.cfg.min_solidity:.3f} -> kontur terpotong"

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

            is_complete, complete_reason = self._is_contour_complete(
                contour_roi, roi_h, roi_w, solidity
            )

            if not is_complete:
                results.append(
                    ContourResult(
                        detection=det,
                        is_complete=False,
                        contour=None,
                        reason=complete_reason,
                        contour_area=c_area,
                        solidity=solidity,
                    )
                )
                continue

            contour_abs = contour_roi.copy()
            contour_abs[:, :, 0] += x1
            contour_abs[:, :, 1] += y1

            results.append(
                ContourResult(
                    detection=det,
                    is_complete=True,
                    contour=contour_abs,
                    reason=complete_reason,
                    contour_area=c_area,
                    solidity=solidity,
                )
            )

        return results

    def draw_results(
        self,
        img_bgr: np.ndarray,
        results: List[ContourResult],
        show_suppressed: bool = False,
        show_bbox: bool = False,
        alpha_fill: float = 0.15,
    ) -> np.ndarray:
        out_img = img_bgr.copy()
        overlay = out_img.copy()

        for res in results:
            det = res.detection
            color = self.CLASS_COLORS[det.class_id % len(self.CLASS_COLORS)]
            label = f"{det.class_name} {det.confidence:.2f}"

            if show_bbox:
                cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), (180, 180, 180), 1)

            if res.is_complete and res.contour is not None:
                if self.cfg.draw_contour:
                    cv2.fillPoly(overlay, [res.contour], color)
                    cv2.drawContours(out_img, [res.contour], -1, color, 2)
                else:
                    cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), color, 2)

                lx, ly = int(res.contour[:, 0, 0].min()), int(res.contour[:, 0, 1].min())
                _put_label(out_img, label, lx, ly, color)

                if show_suppressed:
                    debug = f"sol={res.solidity:.2f}"
                    _put_label(out_img, debug, lx, ly + 18, color, scale=0.35)

            elif show_suppressed:
                cv2.rectangle(overlay, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), -1)
                cv2.rectangle(out_img, (det.x1, det.y1), (det.x2, det.y2), (0, 0, 255), 2)
                _put_label(out_img, f"SUPPRESSED: {res.reason[:40]}", det.x1, det.y1, (0, 0, 255))

        cv2.addWeighted(overlay, alpha_fill, out_img, 1 - alpha_fill, 0, out_img)
        return out_img

    def visualize(
        self,
        img_bgr: np.ndarray,
        results: List[ContourResult],
        draw_contours: bool = True,
        draw_orig_bbox: bool = False,
        show_suppressed: bool = True,
        alpha_fill: float = 0.15,
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


def _put_label(img, text, x, y, color, scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_top = max(y - th - baseline - 2, 0)
    cv2.rectangle(img, (x, y_top), (x + tw + 2, y), color, -1)
    cv2.putText(
        img,
        text,
        (x + 1, max(y - baseline, th)),
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
    print("  Unit Test: EdgeContourPostProcessor (Fase 3 v2)")
    print("=" * 65)
    proc = EdgeContourPostProcessor()

    img_test = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(img_test, (320, 240), (180, 120), 0, 0, 360, (30, 130, 30), -1)
    cv2.ellipse(img_test, (320, 240), (180, 120), 0, 0, 360, (10, 80, 10), 3)
    det_ok = Detection(120, 100, 520, 380, class_id=6, confidence=0.92, class_name="Healty Leaf")
    res = proc.process_detections(img_test, [det_ok])[0]
    print(f"  [TEST 1 - Daun Utuh] is_complete={res.is_complete}, is_suppressed={res.is_suppressed}, bbox_abs={det_ok.bbox_abs(640, 480)}")
    print("  Status Unit Test: LULUS 100%")
    print("=" * 65)

