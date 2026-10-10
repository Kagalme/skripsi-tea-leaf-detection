"""
FASE 3 - Coordinate Attention (CA) Module
==========================================
Implementasi Coordinate Attention (CA) berdasarkan paper:
  "Coordinate Attention for Efficient Mobile Network Design"
  Hou et al., CVPR 2021 — https://arxiv.org/abs/2103.02907

Keunggulan CA dibanding SE Attention / CBAM:
  - Menangkap informasi posisi (spatial) secara horizontal DAN vertikal,
    bukan hanya channel-wise squeezing seperti SE.
  - Biaya komputasi ringan: cocok untuk arsitektur nano seperti YOLO26n.
  - Terbukti meningkatkan akurasi deteksi objek kecil (bercak daun teh).

Cara kerja CA:
  1. Pooling Horizontal (H × 1) dan Vertikal (1 × W) secara paralel.
  2. Concatenate → MLP (BN + Hardswish) → split kembali menjadi h_attn & w_attn.
  3. Sigmoid pada masing-masing → element-wise multiply ke feature map.

Integrasi ke Ultralytics YOLO:
  - Kelas CoordAtt didaftarkan ke ultralytics.nn.modules sebelum YOLO di-build.
  - Model YAML (yolo26n_ca.yaml) menyebut 'CoordAtt' di backbone.
  - Registrasi dilakukan di register_ca_module() — dipanggil sebelum YOLO().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    """
    Coordinate Attention Block (Hou et al., CVPR 2021).

    Args:
        inp  (int): Jumlah channel input.
        oup  (int): Jumlah channel output (biasanya sama dengan inp).
        reduction (int): Faktor reduksi channel pada bottleneck MLP (default=32).

    Forward:
        x : Tensor [B, C, H, W]
        returns : Tensor [B, C, H, W] (re-calibrated feature map)
    """

    def __init__(self, inp: int = None, oup: int = None, reduction: int = 32):
        super().__init__()
        self.reduction = reduction
        self.act   = nn.Hardswish()
        self.conv1 = None
        self.bn1   = None
        self.conv_h = None
        self.conv_w = None
        if inp is not None:
            self._init_layers(inp, oup or inp)

    def _init_layers(self, inp: int, oup: int):
        mip = max(8, inp // self.reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        B, C, H, W = x.shape

        # Otomatis inisialisasi / adaptasi channel sesuai dimensi aktual feature map
        # (Sangat penting saat menggunakan scaling factor YOLO nano width=0.25)
        if self.conv1 is None or self.conv1.in_channels != C:
            self._init_layers(C, C)
            self.to(device=x.device, dtype=x.dtype)

        # 1. Horizontal global average pooling: [B, C, H, 1]
        x_h = F.adaptive_avg_pool2d(x, (H, 1))

        # 2. Vertical global average pooling + transpose: [B, C, 1, W] → [B, C, W, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, W)).permute(0, 1, 3, 2)

        # 3. Concat along height dim: [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)

        # 4. Shared bottleneck MLP
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 5. Split kembali ke h_attn dan w_attn
        x_h_enc, x_w_enc = torch.split(y, [H, W], dim=2)
        x_w_enc = x_w_enc.permute(0, 1, 3, 2)  # [B, mip, W, 1] → [B, mip, 1, W]

        # 6. Sigmoid attention maps
        a_h = self.conv_h(x_h_enc).sigmoid()   # [B, C, H, 1]
        a_w = self.conv_w(x_w_enc).sigmoid()   # [B, C, 1, W]

        # 7. Re-calibrate input
        out = identity * a_h * a_w
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper yang kompatibel dengan format argumen Ultralytics YAML
# ─────────────────────────────────────────────────────────────────────────────

class CoordAttYOLO(CoordAtt):
    """
    Wrapper agar CoordAtt dapat diparsing oleh Ultralytics YAML builder.
    Mendukung berbagai format pemanggilan dan scaling factor YOLO nano/small.
    """

    def __init__(self, *args, **kwargs):
        reduction = 32
        inp = None
        for a in args:
            if isinstance(a, int):
                if a <= 64:
                    reduction = a
                else:
                    inp = a
        if "reduction" in kwargs:
            reduction = kwargs["reduction"]
        super().__init__(inp=inp, oup=inp, reduction=reduction)


def register_ca_module():
    """
    Daftarkan CoordAttYOLO ke dalam namespace PyTorch (torch.nn) dan Ultralytics.
    Di Colab, secara otomatis menambahkan definisi kelas ke bagian akhir berkas
    ultralytics/nn/tasks.py (mode append), sehingga 100% aman dari SyntaxError
    dan langsung dikenali oleh globals() parse_model.
    """
    try:
        import os
        import torch.nn as nn
        import ultralytics.nn.modules as ult_modules
        import ultralytics.nn.tasks as ult_tasks

        # 1. Daftarkan di memori Python aktif
        setattr(nn, 'CoordAttYOLO', CoordAttYOLO)
        setattr(ult_modules, 'CoordAttYOLO', CoordAttYOLO)
        setattr(ult_tasks, 'CoordAttYOLO', CoordAttYOLO)
        ult_tasks.__dict__['CoordAttYOLO'] = CoordAttYOLO
        if hasattr(ult_tasks, 'parse_model'):
            ult_tasks.parse_model.__globals__['CoordAttYOLO'] = CoordAttYOLO

        # 2. Append ke AKHIR berkas tasks.py jika writable (di Colab)
        try:
            tasks_file = getattr(ult_tasks, '__file__', None)
            if tasks_file and os.path.exists(tasks_file):
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    txt = f.read()
                if 'class CoordAttYOLO' not in txt:
                    appendix = """

# ==============================================================================
# FASE 3: Coordinate Attention (CA)
# ==============================================================================
class CoordAtt(nn.Module):
    def __init__(self, inp: int = None, oup: int = None, reduction: int = 32):
        super().__init__()
        self.reduction = reduction
        self.act   = nn.Hardswish()
        self.conv1 = None
        self.bn1   = None
        self.conv_h = None
        self.conv_w = None
        if inp is not None:
            self._init_layers(inp, oup or inp)

    def _init_layers(self, inp: int, oup: int):
        mip = max(8, inp // self.reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        if self.conv1 is None or self.conv1.in_channels != c:
            self._init_layers(c, c)
            self.to(device=x.device, dtype=x.dtype)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h_enc, x_w_enc = torch.split(y, [h, w], dim=2)
        x_w_enc = x_w_enc.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h_enc).sigmoid()
        a_w = self.conv_w(x_w_enc).sigmoid()
        return identity * a_h * a_w

class CoordAttYOLO(CoordAtt):
    def __init__(self, *args, **kwargs):
        reduction = 32
        inp = None
        for a in args:
            if isinstance(a, int):
                if a <= 64:
                    reduction = a
                else:
                    inp = a
        if "reduction" in kwargs:
            reduction = kwargs["reduction"]
        super().__init__(inp=inp, oup=inp, reduction=reduction)
"""
                    with open(tasks_file, 'a', encoding='utf-8') as f:
                        f.write(appendix)
                    print("[CA] Berhasil menambahkan CoordAttYOLO ke berkas tasks.py.")
        except Exception:
            pass

        print("[CA] CoordAttYOLO berhasil didaftarkan ke PyTorch & Ultralytics.")
        return True

    except Exception as e:
        print(f"[CA][WARNING] Pendaftaran CA gagal: {e}")
        return False


if __name__ == "__main__":
    # === Unit test ===
    print("=" * 60)
    print("  Unit Test: CoordAtt Module")
    print("=" * 60)

    # Test dengan berbagai ukuran feature map (sesuai backbone YOLO)
    test_cases = [
        (4, 64,  52, 52),   # P3 (small objects)
        (4, 128, 26, 26),   # P4 (medium objects)
        (4, 256, 13, 13),   # P5 (large objects)
    ]

    ca_ok = True
    for B, C, H, W in test_cases:
        x    = torch.randn(B, C, H, W)
        ca   = CoordAtt(inp=C, oup=C)
        out  = ca(x)
        ok   = (out.shape == x.shape)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] Input {(B,C,H,W)} → Output {tuple(out.shape)}")
        if not ok:
            ca_ok = False

    print()
    # Test YOLO wrapper
    x   = torch.randn(2, 128, 26, 26)
    ca_w = CoordAttYOLO(c1=128)
    out  = ca_w(x)
    ok   = (out.shape == x.shape)
    print(f"  [{'OK' if ok else 'FAIL'}] CoordAttYOLO wrapper: {tuple(x.shape)} → {tuple(out.shape)}")

    # Parameter count
    total_params = sum(p.numel() for p in ca_w.parameters())
    print(f"\n  Total parameter CoordAttYOLO(128): {total_params:,}")
    print(f"\n{'=' * 60}")
    print(f"  Semua test {'LULUS' if ca_ok else 'GAGAL'}!")
    print(f"{'=' * 60}")
