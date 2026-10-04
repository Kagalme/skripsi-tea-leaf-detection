"""
FASE 2: MODEL SELECTION EXPERIMENT - CONFIGURATION MODULE
==========================================================
Konfigurasi terpusat untuk eksperimen perbandingan baseline object detector:
1. RetinaNet (ResNet-50-FPN)
2. SSD (SSD300-VGG16 / SSDLite320-MobileNetV3)
3. Faster R-CNN (ResNet-50-FPN)
4. RT-DETR (RT-DETR-L / ResNet backbone)
5. YOLO26n (Baseline Anchor-free)

Protokol:
- Dataset identik: tea_yolo/
- Split identik: Train (3701), Val (1048), Test (529)
- 7 Kelas identik (catatan: nama kelas asli 'Healty Leaf' tetap dipertahankan)
- Random Seed identik: 42
- Evaluasi adil: Test set HANYA dievaluasi setelah training selesai
"""

import os
import torch
import numpy as np
import random

# ==========================================
# 1. PATH CONFIGURATION (AUTO-DETECT COLAB VS LOKAL)
# ==========================================
if os.path.exists("/content/tea_yolo"):
    # Lingkungan Google Colab
    BASE_DIR = "/content"
    DATASET_DIR = "/content/tea_yolo"
    DATA_YAML_PATH = "/content/tea_yolo/data.yaml"
    RUNS_DIR = "/content/phase2_runs/model_selection"
else:
    # Lingkungan Komputer Lokal (Windows / Linux)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "tea_yolo")
    DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
    RUNS_DIR = os.path.join(BASE_DIR, "phase2_runs", "model_selection")

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. DATASET DEFINITIONS & EXACT CLASS NAMES
# ==========================================
# Catatan penting: Nama kelas asli 'Healty Leaf' TIDAK diubah.
CLASS_NAMES = [
    "Tea algal leaf spot",   # Class 0
    "Brown Blight",          # Class 1
    "Gray Blight",           # Class 2
    "Helopeltis",            # Class 3
    "Red Spider",            # Class 4
    "Green Mirid Bug",       # Class 5
    "Healty Leaf"            # Class 6 (Original spelling preserved)
]

NUM_CLASSES = len(CLASS_NAMES)  # 7 classes

# Distribusi data per split (hasil split bebas kebocoran / Grouped Stratified Split)
SPLIT_COUNTS = {
    "train": 3701,
    "val": 1048,
    "test": 529,
    "total": 5278
}

# ==========================================
# 3. BASELINE HYPERPARAMETERS (FAIR BENCHMARK)
# ==========================================
DEFAULT_SEED = 42
DEFAULT_IMGSZ = 640          # Resolusi 640x640 dipertahankan sesuai instruksi dosen pembimbing
DEFAULT_BATCH_SIZE = 4       # Disesuaikan untuk keterbatasan memori / CPU
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
# Gunakan 2 workers di Google Colab / Linux untuk prefetch gambar paralel, 0 di Windows lokal
DEFAULT_WORKERS = 2 if (os.path.exists("/content/tea_yolo") or os.name != "nt") else 0

# Model Candidates
CANDIDATE_MODELS = [
    "retinanet",
    "ssd",
    "faster_rcnn",
    "rtdetr",
    "yolo26n"
]

def get_model_dir(model_type):
    """
    Mengembalikan path direktori khusus untuk setiap model secara terisolasi.
    Contoh: phase2_runs/model_selection/retinanet/
    """
    d = os.path.join(RUNS_DIR, model_type.lower())
    os.makedirs(d, exist_ok=True)
    return d

def set_seed(seed=DEFAULT_SEED):
    """Memastikan reproducibility eksperimen di seluruh pustaka acak."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
