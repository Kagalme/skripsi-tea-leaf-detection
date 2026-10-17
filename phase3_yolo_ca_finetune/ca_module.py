"""
FASE 3 (FINE-TUNING EDITION) - Coordinate Attention (CA) Module, Weight Remapper & Fine-Tune Loader
===================================================================================================
Implementasi Coordinate Attention (Hou et al., CVPR 2021) untuk YOLO26n dengan dukungan:
1. `build_yolo26n_ca_pretrained()` : Memetakan 100% bobot `yolo26n.pt` (24 layer) ke `yolo26n_ca.yaml` (27 layer).
2. `prepare_finetune_checkpoint()` : Menyiapkan bobot Stage-1 (`best.pt` 50 epoch) sebagai inisialisasi
   bersih untuk Stage-2 Precision Fine-Tuning (mereset counter epoch & optimizer state sambil
   mempertahankan 100% bobot terlatih YOLO26n+CA).
3. `find_best_weights()` & `find_stage1_weights()` : Pencarian otomatis lintas direktori Colab & Drive.
"""

import os
import copy
import glob
import torch
import torch.nn as nn


# Mapping indeks layer dari yolo26n.pt asli (0..23) ke yolo26n_ca.yaml (0..26)
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

        if self.conv1.in_channels != C:
            self._init_layers(C, C)
            self.to(device=x.device, dtype=x.dtype)

        x_h = torch.mean(x, dim=3, keepdim=True)                      # [B, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h_enc, x_w_enc = torch.split(y, [H, W], dim=2)
        x_w_enc = x_w_enc.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h_enc).sigmoid()
        a_w = self.conv_w(x_w_enc).sigmoid()

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
    serta menyisipkannya ke ultralytics/nn/tasks.py jika diperlukan.
    """
    try:
        import ultralytics.nn.modules as ult_modules
        import ultralytics.nn.tasks as ult_tasks

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
    Membuat berkas bobot pretrained khusus `yolo26n_ca_coco_init.pt` dengan cara
    memetakan (remapping) seluruh layer dari `yolo26n.pt` (24 layer: 0..23) ke
    arsitektur `yolo26n_ca.yaml` (27 layer: 0..26).
    """
    register_ca_module()
    from ultralytics import YOLO

    torch.manual_seed(seed)

    if not os.path.exists(base_pt_path):
        candidates = [
            "/content/yolo26n.pt",
            "/content/skripsi-tea-leaf-detection/yolo26n.pt",
            "/content/drive/MyDrive/SKRIPSI/yolo26n.pt",
            "yolo26n.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                base_pt_path = c
                break
        else:
            print("[CA-REMAP] Mengunduh/memuat yolo26n.pt otomatis via Ultralytics...")
            _tmp = YOLO("yolo26n.pt")
            if os.path.exists("yolo26n.pt"):
                base_pt_path = os.path.abspath("yolo26n.pt")

    ca_yolo = YOLO(ca_yaml_path)
    ca_model = ca_yolo.model
    ca_sd = ca_model.state_dict()

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
                    skipped_head_cls += 1

    ca_model.load_state_dict(new_sd, strict=True)

    orig_train_args = ckpt.get("train_args")
    if not isinstance(orig_train_args, dict):
        orig_train_args = {}

    save_Obj = {
        "epoch": -1,
        "best_fitness": None,
        "model": copy.deepcopy(ca_model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": orig_train_args,
        "date": None,
        "version": ckpt.get("version", "8.4.178"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    torch.save(save_Obj, out_pt_path)

    print(f"[CA-REMAP] Berhasil memetakan {mapped_count}/{len(base_sd)} tensor dari '{os.path.basename(base_pt_path)}'")
    print(f"           ke arsitektur YOLO26n+CA ('{os.path.basename(out_pt_path)}').")
    return out_pt_path


def prepare_finetune_checkpoint(
    stage1_pt_path: str,
    ca_yaml_path: str,
    out_pt_path: str,
) -> str:
    """
    Menyiapkan checkpoint Stage-1 (`best.pt` hasil 50 epoch) menjadi checkpoint bersih
    untuk Stage-2 Precision Fine-Tuning:
    - Menyalin 100% bobot terlatih (27 layer termasuk 3 blok CoordAttYOLO & Detect head 7 kelas).
    - Mereset `epoch=-1`, `optimizer=None`, `ema=None`, dan `best_fitness=None` agar
      Ultralytics menjalankan jadwal Cosine Annealing LR baru dengan mulus.
    """
    register_ca_module()
    from ultralytics import YOLO

    ca_yolo = YOLO(ca_yaml_path)
    ca_model = ca_yolo.model
    ca_sd = ca_model.state_dict()

    ckpt = torch.load(stage1_pt_path, map_location="cpu", weights_only=False)
    src_model = ckpt.get("ema") or ckpt.get("model")
    src_sd = src_model.float().state_dict()

    matched = 0
    new_sd = copy.deepcopy(ca_sd)
    for k, v in src_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.to(dtype=new_sd[k].dtype)
            matched += 1

    ca_model.load_state_dict(new_sd, strict=True)

    orig_train_args = ckpt.get("train_args")
    if not isinstance(orig_train_args, dict):
        orig_train_args = {}

    save_obj = {
        "epoch": -1,
        "best_fitness": None,
        "model": copy.deepcopy(ca_model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": orig_train_args,
        "date": None,
        "version": ckpt.get("version", "8.4.178"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    torch.save(save_obj, out_pt_path)
    print(f"[FINETUNE-INIT] Berhasil memuat {matched}/{len(new_sd)} tensor dari Stage-1 '{stage1_pt_path}'")
    print(f"                ke checkpoint siap Fine-Tune: '{out_pt_path}'")
    return out_pt_path


def find_stage1_weights(stage1_name: str = "yolo26n_ca_50ep") -> str:
    """Mencari bobot Stage-1 (`yolo26n_ca_50ep/weights/best.pt`) di Colab, Drive, atau lokal."""
    candidates = [
        f"/content/phase3_runs/{stage1_name}/weights/best.pt",
        f"/content/runs/detect/{stage1_name}/weights/best.pt",
        f"/content/runs/detect/content/phase3_runs/{stage1_name}/weights/best.pt",
        f"/content/drive/MyDrive/SKRIPSI/phase3_runs/{stage1_name}/weights/best.pt",
        f"/content/drive/MyDrive/SKRIPSI_Backup/{stage1_name}/weights/best.pt",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase3_runs", stage1_name, "weights", "best.pt")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    search_patterns = [
        f"/content/phase3_runs/*{stage1_name}*/**/best.pt",
        f"/content/runs/**/*{stage1_name}*/**/best.pt",
        "/content/drive/MyDrive/**/yolo26n_ca_50ep/**/best.pt",
        "/content/phase3_runs/**/weights/best.pt",
    ]
    for pattern in search_patterns:
        matches = [
            m for m in glob.glob(pattern, recursive=True)
            if "finetune" not in m.lower()
        ]
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return ""


def find_best_weights(run_name: str = "yolo26n_ca_finetune") -> str:
    """
    Mencari file `best.pt` hasil Fine-Tuning (`yolo26n_ca_finetune`) terlebih dahulu,
    dan otomatis fallback ke `yolo26n_ca_50ep` jika ditemukan.
    """
    exact_candidates = [
        f"/content/phase3_runs/{run_name}/weights/best.pt",
        f"/content/runs/detect/{run_name}/weights/best.pt",
        f"/content/runs/detect/content/phase3_runs/{run_name}/weights/best.pt",
        f"/content/drive/MyDrive/SKRIPSI/phase3_runs/{run_name}/weights/best.pt",
        f"/content/phase3_runs/yolo26n_ca_finetune/weights/best.pt",
        f"/content/phase3_runs/yolo26n_ca_50ep/weights/best.pt",
    ]
    for path in exact_candidates:
        if os.path.exists(path):
            return path

    search_patterns = [
        "/content/phase3_runs/**/weights/best.pt",
        "/content/runs/**/weights/best.pt",
        "/content/drive/MyDrive/SKRIPSI/**/weights/best.pt",
    ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]

    return exact_candidates[0]


if __name__ == "__main__":
    print("=" * 65)
    print("  Unit Test: CoordAtt & CoordAttYOLO (Fase 3 Fine-Tuning Edition)")
    print("=" * 65)
    for B, C, H, W in [(2, 128, 80, 80), (2, 128, 40, 40), (2, 256, 20, 20)]:
        x = torch.randn(B, C, H, W)
        ca = CoordAttYOLO(C, 32)
        out = ca(x)
        assert out.shape == x.shape
        print(f"  [OK] Input {(B, C, H, W)} -> Output {tuple(out.shape)}")
    print("=" * 65)

