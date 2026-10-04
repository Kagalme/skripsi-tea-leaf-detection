"""
FASE 2: MODEL SELECTION EXPERIMENT PACKAGE
===========================================
Modul terpadu perbandingan baseline object detector:
1. RetinaNet (Lin et al., ICCV 2017)
2. SSD (Liu et al., ECCV 2016)
3. Faster R-CNN (Ren et al., NeurIPS 2015)
4. RT-DETR (Lv et al., CVPR 2023)
5. YOLO26n Baseline
"""

from model_selection.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_SEED
)
from model_selection.models import (
    build_detector,
    MODEL_REGISTRY,
    count_parameters
)
