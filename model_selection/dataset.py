"""
FASE 2: MODEL SELECTION EXPERIMENT - DATASET LOADER MODULE
===========================================================
Modul PyTorch Dataset untuk membaca dataset tea_yolo format:
- Bounding box konversi: Normalized [x_center, y_center, w, h] -> Absolute [x1, y1, x2, y2]
- Fair Augmentation: Standard Horizontal Flip (p=0.5), Resize, Normalization ImageNet
- Dukungan background class offset:
  * background_offset = 1 untuk Faster R-CNN & SSD (0 = background, 1..7 = kelas daun teh)
  * background_offset = 0 untuk RetinaNet (0..6 = foreground classes)
- Collate function untuk DataLoader PyTorch detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import random

from model_selection.config import (
    DATASET_DIR,
    CLASS_NAMES,
    DEFAULT_IMGSZ
)

class TeaLeafDetectionDataset(Dataset):
    """
    Dataset loader terstandarisasi untuk deteksi daun teh.
    Membaca langsung dari struktur tea_yolo/ (train, val, test).
    """
    def __init__(self, split="train", imgsz=DEFAULT_IMGSZ, augment=False, background_offset=0):
        super().__init__()
        assert split in ["train", "val", "test"], f"Split tidak valid: {split}"
        self.split = split
        self.imgsz = imgsz
        self.augment = augment
        self.background_offset = background_offset

        self.images_dir = os.path.join(DATASET_DIR, "images", split)
        self.labels_dir = os.path.join(DATASET_DIR, "labels", split)

        # Kumpulkan semua file gambar yang valid
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        # 1. Buka Gambar
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # 2. Baca Anotasi Bounding Box (Format YOLO: class_id xc yc w h ter-normalisasi)
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])

                        # Konversi ke koordinat absolut gambar asli [x1, y1, x2, y2]
                        x1 = max(0.0, (xc - bw / 2.0) * orig_w)
                        y1 = max(0.0, (yc - bh / 2.0) * orig_h)
                        x2 = min(float(orig_w), (xc + bw / 2.0) * orig_w)
                        y2 = min(float(orig_h), (yc + bh / 2.0) * orig_h)

                        # Validasi dimensi kotak agar tidak degenerate
                        if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_id + self.background_offset)

        # Konversi ke Tensor PyTorch
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # 3. Standard & Fair Augmentation (Horizontal Flip pada training)
        if self.augment and random.random() < 0.5:
            image = F.hflip(image)
            if len(boxes) > 0:
                # Balik koordinat horizontal [x1, y1, x2, y2]
                new_x1 = orig_w - boxes[:, 2]
                new_x2 = orig_w - boxes[:, 0]
                boxes[:, 0] = new_x1
                boxes[:, 2] = new_x2

        # 4. Resize ke target image size (imgsz, imgsz)
        scale_x = self.imgsz / orig_w
        scale_y = self.imgsz / orig_h
        image = F.resize(image, [self.imgsz, self.imgsz])

        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * scale_x
            boxes[:, 1] = boxes[:, 1] * scale_y
            boxes[:, 2] = boxes[:, 2] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_y

            # Clamp koordinat ke batas gambar
            boxes[:, 0].clamp_(min=0, max=self.imgsz)
            boxes[:, 1].clamp_(min=0, max=self.imgsz)
            boxes[:, 2].clamp_(min=0, max=self.imgsz)
            boxes[:, 3].clamp_(min=0, max=self.imgsz)

        # 5. Normalisasi Tensor ImageNet
        image_tensor = F.to_tensor(image)  # [0.0, 1.0]
        # Torchvision detection models memiliki internal Normalizer, namun jika diperlukan:
        # image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Hitung area untuk COCO metric evaluation
        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "img_name": img_name
        }

        return image_tensor, target

def collate_fn(batch):
    """
    Collate function untuk mengakomodasi jumlah bounding box yang bervariasi per gambar.
    """
    return tuple(zip(*batch))

def get_dataloader(split="train", imgsz=DEFAULT_IMGSZ, batch_size=4, augment=None, background_offset=0, shuffle=None, workers=0):
    """
    Helper untuk menginstansiasi DataLoader PyTorch.
    """
    if augment is None:
        augment = (split == "train")
    if shuffle is None:
        shuffle = (split == "train")

    dataset = TeaLeafDetectionDataset(
        split=split,
        imgsz=imgsz,
        augment=augment,
        background_offset=background_offset
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    return loader
