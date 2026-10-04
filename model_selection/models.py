"""
FASE 2: MODEL SELECTION EXPERIMENT - MODELS BUILDER MODULE
===========================================================
Factory module untuk membangun 4 baseline object detector + YOLO26n:
1. RetinaNet (Lin et al., ICCV 2017) -> ResNet-50-FPN, Focal Loss
2. SSD (Liu et al., ECCV 2016) -> SSD300-VGG16 / Multi-scale Conv
3. Faster R-CNN (Ren et al., NeurIPS 2015) -> Two-stage, ResNet-50-FPN, RPN, RoIAlign
4. RT-DETR (Lv et al., CVPR 2023) -> Transformer-based, Hybrid Encoder (AIFI + CCFM), End-to-end No NMS
5. YOLO26n (Ultralytics) -> Anchor-free single-stage baseline

Semua model dikonfigurasi murni baseline tanpa modifikasi arsitektur:
- Tanpa Coordinate Attention
- Tanpa CBAM
- Tanpa Edge Detection (Canny/Sobel)
- Tanpa custom loss tambahan
"""

import os
import torch
import torch.nn as nn
import torchvision.models.detection as d_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

from model_selection.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_IMGSZ
)

MODEL_REGISTRY = {
    "retinanet": {
        "name": "RetinaNet-ResNet50-FPN",
        "paper": "Lin et al., ICCV 2017 (Focal Loss for Dense Object Detection)",
        "paradigm": "One-stage Dense Anchor-based",
        "backbone": "ResNet-50",
        "neck": "FPN (Feature Pyramid Network, P3-P7)",
        "head": "Classification Subnet (Focal Loss) + Box Regression Subnet (Smooth L1)",
        "nms_needed": True,
        "background_offset": 0,  # 7 foreground classes (0..6)
        "num_classes_model": NUM_CLASSES,
        "pretrained_weights": "RetinaNet_ResNet50_FPN_Weights.DEFAULT (COCO)"
    },
    "ssd": {
        "name": "SSD300-VGG16",
        "paper": "Liu et al., ECCV 2016 (SSD: Single Shot MultiBox Detector)",
        "paradigm": "One-stage Multi-scale Anchor-based",
        "backbone": "Truncated VGG-16",
        "neck": "Multi-scale feature conv cascade (6 scales)",
        "head": "MultiBox Head (Softmax + Smooth L1) + Hard Negative Mining (3:1)",
        "nms_needed": True,
        "background_offset": 1,  # 8 classes: 0=background, 1..7=classes
        "num_classes_model": NUM_CLASSES + 1,
        "pretrained_weights": "SSD300_VGG16_Weights.DEFAULT (COCO)"
    },
    "faster_rcnn": {
        "name": "Faster R-CNN-ResNet50-FPN",
        "paper": "Ren et al., NeurIPS 2015 (Towards Real-Time Object Detection with RPN)",
        "paradigm": "Two-stage Proposal-based",
        "backbone": "ResNet-50",
        "neck": "FPN (Feature Pyramid Network)",
        "head": "RPN (Proposal Generation) + RoIAlign (7x7) + Fast R-CNN Head (Two-layer MLP)",
        "nms_needed": True,
        "background_offset": 1,  # 8 classes: 0=background, 1..7=classes
        "num_classes_model": NUM_CLASSES + 1,
        "pretrained_weights": "FasterRCNN_ResNet50_FPN_Weights.DEFAULT (COCO)"
    },
    "rtdetr": {
        "name": "RT-DETR-L",
        "paper": "Lv et al., CVPR 2023 (DETRs Beat YOLOs on Real-time Object Detection)",
        "paradigm": "Transformer-based End-to-End (No NMS)",
        "backbone": "ResNet-50 / HGNetv2",
        "neck": "Efficient Hybrid Encoder (AIFI + CCFM)",
        "head": "Transformer Decoder + Dynamic Query Selection + Hungarian Bipartite Matching",
        "nms_needed": False,
        "background_offset": 0,
        "num_classes_model": NUM_CLASSES,
        "pretrained_weights": "rtdetr-l.pt (COCO)"
    },
    "yolo26n": {
        "name": "YOLO26n",
        "paper": "Ultralytics Baseline (Anchor-free CNN Detector)",
        "paradigm": "One-stage Anchor-free",
        "backbone": "Modified CSPDarknet",
        "neck": "PANet / C2f",
        "head": "Decoupled Anchor-free Head (Task-Aligned Assigner)",
        "nms_needed": True,
        "background_offset": 0,
        "num_classes_model": NUM_CLASSES,
        "pretrained_weights": "yolo26n.pt (COCO)"
    }
}


def count_parameters(model):
    """Menghitung total parameter dan trainable parameter model."""
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        return total_params, trainable_params
    return 0, 0


def build_detector(model_type, num_classes=NUM_CLASSES, pretrained=True, imgsz=DEFAULT_IMGSZ):
    """
    Factory function untuk menginstansiasi model object detection.
    
    Args:
        model_type (str): 'retinanet', 'ssd', 'faster_rcnn', 'rtdetr', atau 'yolo26n'
        num_classes (int): Jumlah kelas target (default = 7)
        pretrained (bool): Menggunakan pretrained COCO weights
        imgsz (int): Target resolusi gambar
        
    Returns:
        model: Instance model terkonfigurasi untuk 7 kelas dataset teh
        meta: Dictionary metadata spesifikasi arsitektur
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' tidak dikenal. Pilih dari: {list(MODEL_REGISTRY.keys())}")

    meta = MODEL_REGISTRY[model_type].copy()

    if model_type == "retinanet":
        # 1. RETINANET (Lin et al., ICCV 2017)
        # Backbone ResNet-50 + FPN. Kunci resolusi input ke imgsz (416)
        weights = d_models.RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = d_models.retinanet_resnet50_fpn(
            weights=weights,
            min_size=imgsz,
            max_size=imgsz
        )

        # Modifikasi Classification Head
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.head.classification_head.conv[0].in_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

    elif model_type == "ssd":
        # 2. SSD300 (Liu et al., ECCV 2016)
        # Truncated VGG-16 + Multi-scale Conv layers.
        # num_classes = 7 + 1 background = 8 kelas
        weights = d_models.SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model = d_models.ssd300_vgg16(weights=weights)

        # Modifikasi MultiBox Classification Head
        in_channels = [layer.in_channels for layer in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1  # Sertakan background
        )

    elif model_type == "faster_rcnn":
        # 3. FASTER R-CNN (Ren et al., NeurIPS 2015)
        # Two-stage: ResNet-50-FPN + RPN + RoIAlign + Fast R-CNN Predictor.
        # Kunci resolusi ke imgsz (416) dan rasionalkan proposal RPN untuk dataset single-leaf di CPU
        weights = d_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = d_models.fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=imgsz,
            max_size=imgsz,
            rpn_pre_nms_top_n_train=1000,
            rpn_post_nms_top_n_train=500,
            rpn_pre_nms_top_n_test=500,
            rpn_post_nms_top_n_test=300,
            box_batch_size_per_image=256
        )

        # Modifikasi FastRCNNPredictor pada RoI Head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features,
            num_classes=num_classes + 1  # Sertakan background
        )

    elif model_type == "rtdetr":
        # 4. RT-DETR (Lv et al., CVPR 2023)
        # Transformer-based End-to-end Detector via Ultralytics API
        from ultralytics import RTDETR
        weights_name = "rtdetr-l.pt" if pretrained else "rtdetr-l.yaml"
        model = RTDETR(weights_name)

    elif model_type == "yolo26n":
        # 5. YOLO26n Baseline
        from ultralytics import YOLO
        candidates = [
            os.path.join(os.path.dirname(__file__), "yolo26n.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo26n.pt"),
            "/content/yolo26n.pt",
            "yolo26n.pt"
        ]
        weights_path = "yolo26n.pt"
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
        model = YOLO(weights_path)

    # Catat parameter counts ke metadata
    tot_p, train_p = count_parameters(model)
    meta["total_parameters"] = tot_p
    meta["trainable_parameters"] = train_p

    return model, meta


def print_model_specifications():
    """Mencetak ringkasan spesifikasi seluruh model kandidat model selection."""
    print("=" * 85)
    print("  FASE 2: MODEL SELECTION - DAFTAR SPESIFIKASI ARSITEKTUR BASELINE")
    print("=" * 85)
    for key, spec in MODEL_REGISTRY.items():
        print(f"\n[{key.upper()}] - {spec['name']}")
        print(f"  Landasan Paper   : {spec['paper']}")
        print(f"  Paradigma        : {spec['paradigm']}")
        print(f"  Backbone         : {spec['backbone']}")
        print(f"  Neck / Piramida  : {spec['neck']}")
        print(f"  Detection Head   : {spec['head']}")
        print(f"  NMS Dependent    : {spec['nms_needed']}")
        print(f"  Pretrained COCO  : {spec['pretrained_weights']}")
        print(f"  Background Class : {'Yes (class 0)' if spec['background_offset']==1 else 'No (Sigmoid/Set matching)'}")
    print("\n" + "=" * 85)


if __name__ == "__main__":
    print_model_specifications()
