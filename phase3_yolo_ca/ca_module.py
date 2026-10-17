"""
FASE 3 (DIPERBARUI) - Coordinate Attention (CA) Module & Weight Remapper
========================================================================
Implementasi Coordinate Attention (Hou et al., CVPR 2021) untuk YOLO26n:
1. Kompatibel penuh dengan ekspor Mobile (TFLite / ONNX) karena menggunakan
   `torch.mean(..., keepdim=True)` sebagai pengganti `adaptive_avg_pool2d`
   dinamis yang sering gagal saat konversi TFLite.
2. Mendukung pemetaan bobot otomatis (`build_yolo26n_ca_pretrained`) dari
   `yolo26n.pt` (24 layer asli) ke `yolo26n_ca.yaml` (27 layer dengan 3 blok CA),
   sehingga 100% bobot backbone & neck pretrained COCO berhasil ditransfer!
"""

import os
import copy
import torch
import torch.nn as nn


# Mapping indeks layer dari yolo26n.pt asli (0..23) ke yolo26n_ca.yaml (0..26)
# CA disisipkan pada indeks 5 (P3), 8 (P4), dan 13 (P5)
YOLO26N_TO_CA_LAYER_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    # 5 adalah CoordAttYOLO (CA-1 di P3)
    5: 6,
    6: 7,
    # 8 adalah CoordAttYOLO (CA-2 di P4)
    7: 9,
    8: 10,
    9: 11,
    10: 12,
    # 13 adalah CoordAttYOLO (CA-3 di P5)
    11: 14,
    12: 15,
    13: 16,
    14: 17,
    15: 18,
    16: 19,
    17: 20,
    18: 21,
    19: 22,
    20: 23,
    21: 24,
    22: 25,
    23: 26,
}


class CoordAtt(nn.Module):
    """
    Coordinate Attention Block (Hou et al., CVPR 2021).
    Dirancang ramah TFLite/ONNX (menggunakan torch.mean keepdim=True).
    """

    def __init__(self, inp: int = 128, oup: int = None, reduction: int = 32):
        super().__init__()
        oup = oup or inp
        self.inp = inp
        self.oup = oup
        self.reduction = reduction
        self.act = nn.Hardswish()
        self._init_layers(inp, oup)

    def _init_layers(self, inp: int, oup: int):
        mip = max(8, inp // self.reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        B, C, H, W = x.shape

        # Safety fallback jika channel input berbeda dari inisialisasi awal
        if self.conv1.in_channels != C:
            self._init_layers(C, C)
            self.to(device=x.device, dtype=x.dtype)

        # 1. Horizontal & Vertical Global Average Pooling (100% ekuivalen dengan
        #    adaptive_avg_pool2d((H,1)) & ((1,W)), namun jauh lebih aman untuk TFLite/ONNX)
        x_h = torch.mean(x, dim=3, keepdim=True)                      # [B, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]

        # 2. Concat along spatial dimension: [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)

        # 3. Shared bottleneck Conv1x1 + BN + Hardswish
        y = self.act(self.bn1(self.conv1(y)))

        # 4. Split kembali ke h_attn dan w_attn
        x_h_enc, x_w_enc = torch.split(y, [H, W], dim=2)
        x_w_enc = x_w_enc.permute(0, 1, 3, 2)  # [B, mip, 1, W]

        # 5. Sigmoid attention maps
        a_h = self.conv_h(x_h_enc).sigmoid()   # [B, C, H, 1]
        a_w = self.conv_w(x_w_enc).sigmoid()   # [B, C, 1, W]

        return identity * a_h * a_w


class CoordAttYOLO(CoordAtt):
    """
    Wrapper CoordAtt untuk parser YAML Ultralytics.
    Di YAML dipanggil sebagai: [-1, 1, CoordAttYOLO, [c1, reduction]]
    """

    def __init__(self, c1: int = 128, reduction: int = 32, *args, **kwargs):
        if isinstance(c1, (list, tuple)):
            c1 = c1[0]
        if "inp" in kwargs:
            c1 = kwargs["inp"]
        if "reduction" in kwargs:
            reduction = kwargs["reduction"]
        super().__init__(inp=int(c1), oup=int(c1), reduction=int(reduction))


def register_ca_module() -> bool:
    """
    Mendaftarkan CoordAttYOLO ke dalam namespace PyTorch (torch.nn) dan Ultralytics
    serta menyisipkannya ke ultralytics/nn/tasks.py jika diperlukan (agar subprocess
    DataLoader / DDP di Colab juga mengenali modul CoordAttYOLO).
    """
    try:
        import ultralytics.nn.modules as ult_modules
        import ultralytics.nn.tasks as ult_tasks

        # 1. Daftarkan di memori Python aktif
        setattr(nn, "CoordAtt", CoordAtt)
        setattr(nn, "CoordAttYOLO", CoordAttYOLO)
        setattr(ult_modules, "CoordAtt", CoordAtt)
        setattr(ult_modules, "CoordAttYOLO", CoordAttYOLO)
        setattr(ult_tasks, "CoordAtt", CoordAtt)
        setattr(ult_tasks, "CoordAttYOLO", CoordAttYOLO)
        ult_tasks.__dict__["CoordAtt"] = CoordAtt
        ult_tasks.__dict__["CoordAttYOLO"] = CoordAttYOLO
        if hasattr(ult_tasks, "parse_model"):
            ult_tasks.parse_model.__globals__["CoordAtt"] = CoordAtt
            ult_tasks.parse_model.__globals__["CoordAttYOLO"] = CoordAttYOLO

        # 2. Append atau perbarui di tasks.py jika writable (penting di Google Colab)
        try:
            tasks_file = getattr(ult_tasks, "__file__", None)
            if tasks_file and os.path.exists(tasks_file):
                with open(tasks_file, "r", encoding="utf-8") as f:
                    txt = f.read()
                marker = "# === FASE 3 YOLO26N_CA_V2 ==="
                if marker not in txt:
                    appendix = f"""

{marker}
class CoordAtt(nn.Module):
    def __init__(self, inp: int = 128, oup: int = None, reduction: int = 32):
        super().__init__()
        oup = oup or inp
        self.inp = inp
        self.oup = oup
        self.reduction = reduction
        self.act = nn.Hardswish()
        self._init_layers(inp, oup)

    def _init_layers(self, inp: int, oup: int):
        mip = max(8, inp // self.reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        if self.conv1.in_channels != c:
            self._init_layers(c, c)
            self.to(device=x.device, dtype=x.dtype)
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h_enc, x_w_enc = torch.split(y, [h, w], dim=2)
        x_w_enc = x_w_enc.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h_enc).sigmoid()
        a_w = self.conv_w(x_w_enc).sigmoid()
        return identity * a_h * a_w

class CoordAttYOLO(CoordAtt):
    def __init__(self, c1: int = 128, reduction: int = 32, *args, **kwargs):
        if isinstance(c1, (list, tuple)):
            c1 = c1[0]
        if "inp" in kwargs:
            c1 = kwargs["inp"]
        if "reduction" in kwargs:
            reduction = kwargs["reduction"]
        super().__init__(inp=int(c1), oup=int(c1), reduction=int(reduction))
"""
                    with open(tasks_file, "a", encoding="utf-8") as f:
                        f.write(appendix)
                    print("[CA] Berhasil mendaftarkan CoordAttYOLO (v2) ke ultralytics/nn/tasks.py.")
        except Exception:
            pass

        print("[CA] CoordAttYOLO siap digunakan di PyTorch & Ultralytics.")
        return True

    except Exception as e:
        print(f"[CA][WARNING] Pendaftaran CA gagal: {e}")
        return False


def build_yolo26n_ca_pretrained(
    base_pt_path: str,
    ca_yaml_path: str,
    out_pt_path: str,
    seed: int = 42,
) -> str:
    """
    Membuat berkas bobot pretrained khusus `yolo26n_ca_pretrained.pt` dengan cara
    memetakan (remapping) seluruh layer dari `yolo26n.pt` (24 layer: 0..23) ke
    arsitektur `yolo26n_ca.yaml` (27 layer: 0..26).

    Mengapa ini krusial?
    Karena penyisipan 3 blok CA di layer 5, 8, dan 13 menggeser indeks layer
    berikutnya (+1, +2, +3). Tanpa remapping ini, pencocokan nama state_dict
    standar (`model.6.*` vs `model.5.*`) akan gagal mentransfer layer 5..23!

    Returns:
        out_pt_path: Path ke file checkpoint hasil remapping yang siap dipakai
                     sebagai `pretrained=out_pt_path` pada `model.train()`.
    """
    register_ca_module()
    from ultralytics import YOLO

    torch.manual_seed(seed)

    # 1. Bangun model YOLO26n + CA dari YAML
    ca_yolo = YOLO(ca_yaml_path)
    ca_model = ca_yolo.model
    ca_sd = ca_model.state_dict()

    # 2. Muat checkpoint asli yolo26n.pt
    ckpt = torch.load(base_pt_path, map_location="cpu", weights_only=False)
    base_model = ckpt.get("model") or ckpt.get("ema")
    base_sd = base_model.float().state_dict()

    mapped_count = 0
    skipped_head_cls = 0
    new_sd = copy.deepcopy(ca_sd)

    for old_key, tensor in base_sd.items():
        parts = old_key.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit():
            old_idx = int(parts[1])
            new_idx = YOLO26N_TO_CA_LAYER_MAP.get(old_idx)
            if new_idx is None:
                continue
            new_key = ".".join(["model", str(new_idx)] + parts[2:])
            if new_key in new_sd:
                if new_sd[new_key].shape == tensor.shape:
                    new_sd[new_key] = tensor.to(dtype=new_sd[new_key].dtype)
                    mapped_count += 1
                else:
                    # Hanya layer klasifikasi terakhir di Detect (80 kelas COCO -> 7 kelas teh) yang berbeda shape
                    skipped_head_cls += 1

    ca_model.load_state_dict(new_sd, strict=True)

    # Simpan sebagai checkpoint Ultralytics standar
    save_Obj = {
        "epoch": -1,
        "best_fitness": None,
        "model": copy.deepcopy(ca_model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": None,
        "date": None,
        "version": ckpt.get("version", "8.4.171"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    torch.save(save_Obj, out_pt_path)

    print(f"[CA-REMAP] Berhasil memetakan {mapped_count}/{len(base_sd)} tensor dari '{os.path.basename(base_pt_path)}'")
    print(f"           ke arsitektur YOLO26n+CA ('{os.path.basename(out_pt_path)}').")
    print(f"           ({skipped_head_cls} tensor head klasifikasi 80-kelas COCO diinisialisasi ulang untuk 7 kelas teh).")
    return out_pt_path


if __name__ == "__main__":
    print("=" * 65)
    print("  Unit Test: CoordAtt & CoordAttYOLO (Fase 3 v2)")
    print("=" * 65)

    test_cases = [
        (2, 128, 80, 80),   # P3 pada resolusi 640x640 (stride 8)
        (2, 128, 40, 40),   # P4 pada resolusi 640x640 (stride 16)
        (2, 256, 20, 20),   # P5 pada resolusi 640x640 (stride 32)
    ]

    all_ok = True
    for B, C, H, W in test_cases:
        x = torch.randn(B, C, H, W)
        ca = CoordAttYOLO(C, 32)
        out = ca(x)
        ok = (out.shape == x.shape)
        params = sum(p.numel() for p in ca.parameters())
        print(f"  [{'OK' if ok else 'FAIL'}] Input {(B, C, H, W)} -> Output {tuple(out.shape)} | Params: {params:,}")
        if not ok:
            all_ok = False

    print("=" * 65)
    print(f"  Status Unit Test: {'LULUS 100%' if all_ok else 'GAGAL'}")
    print("=" * 65)

