"""
FASE 2: MODEL SELECTION EXPERIMENT - UNIFIED TRAINING MODULE
=============================================================
Script training baseline untuk semua arsitektur object detector:
1. RetinaNet (ResNet-50-FPN)
2. SSD (SSD300-VGG16)
3. Faster R-CNN (ResNet-50-FPN)
4. RT-DETR (RT-DETR-L)
5. YOLO26n

ATURAN RISET & METODOLOGI:
- Dataset identik: tea_yolo/ (train=3,701, val=1,048)
- Test set (529 gambar) DIISOLASI TOTAL dan TIDAK PERNAH disentuh saat training!
- Seed tetap: 42 (Reproducible)
- Optimizer: SGD momentum 0.9, weight decay 0.0005, Cosine Annealing LR
- Augmentasi setara: Standard Horizontal Flip (p=0.5), ImageNet normalization
- Checkpoint disimpan: 'best_checkpoint.pth' (berdasarkan val mAP) dan 'last_checkpoint.pth'
"""

import os
import sys
import time
import argparse
import math
import json
import multiprocessing
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optimasi PyTorch CPU Multi-threading
try:
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(min(num_cpus, 8))
    torch.set_num_interop_threads(2)
except Exception:
    pass

# Tambahkan direktori root proyek ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_selection.config import (
    BASE_DIR,
    DATA_YAML_PATH,
    RUNS_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_SEED,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    DEFAULT_DEVICE,
    set_seed,
    get_model_dir
)
from model_selection.dataset import get_dataloader
from model_selection.models import build_detector, MODEL_REGISTRY, count_parameters


def calculate_iou_matrix(boxes1, boxes2):
    """Menghitung matriks IoU antara dua set bounding box [N, 4] dan [M, 4]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)))
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def evaluate_validation_map50(model, val_loader, device, background_offset=0, iou_thresh=0.5):
    """
    Evaluasi mAP@0.50 pada validation set untuk seleksi best checkpoint.
    """
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    total_val_batches = len(val_loader)
    with torch.no_grad():
        for val_idx, (images, targets) in enumerate(val_loader, 1):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, out in enumerate(outputs):
                pred_b = out["boxes"].cpu()
                pred_s = out["scores"].cpu()
                pred_l = out["labels"].cpu() - background_offset
                
                gt_b = targets[i]["boxes"].cpu()
                gt_l = targets[i]["labels"].cpu() - background_offset

                all_pred_boxes.append(pred_b)
                all_pred_scores.append(pred_s)
                all_pred_labels.append(pred_l)
                all_gt_boxes.append(gt_b)
                all_gt_labels.append(gt_l)

            if val_idx == 1 or val_idx % 50 == 0 or val_idx == total_val_batches:
                print(f"    [Val Progress] Batch [{val_idx:03d}/{total_val_batches}] ({(val_idx/total_val_batches)*100:5.1f}%)", flush=True)

    # Hitung AP@0.50 per kelas
    aps = []
    for c in range(NUM_CLASSES):
        tp_list = []
        fp_list = []
        scores_list = []
        total_gt = 0

        for img_idx in range(len(all_gt_boxes)):
            gt_mask = (all_gt_labels[img_idx] == c)
            gt_boxes_c = all_gt_boxes[img_idx][gt_mask]
            total_gt += len(gt_boxes_c)

            pred_mask = (all_pred_labels[img_idx] == c)
            pred_boxes_c = all_pred_boxes[img_idx][pred_mask]
            pred_scores_c = all_pred_scores[img_idx][pred_mask]

            if len(pred_boxes_c) == 0:
                continue

            ious = calculate_iou_matrix(pred_boxes_c, gt_boxes_c)
            detected_gt = set()

            # Urutkan prediksi berdasarkan confidence score menurun
            sorted_idx = torch.argsort(pred_scores_c, descending=True)
            for idx in sorted_idx:
                score = pred_scores_c[idx].item()
                scores_list.append(score)
                best_iou = 0.0
                best_gt_idx = -1
                if len(gt_boxes_c) > 0:
                    for g_idx in range(len(gt_boxes_c)):
                        if g_idx not in detected_gt and ious[idx, g_idx] > best_iou:
                            best_iou = ious[idx, g_idx].item()
                            best_gt_idx = g_idx

                if best_iou >= iou_thresh:
                    tp_list.append(1)
                    fp_list.append(0)
                    detected_gt.add(best_gt_idx)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if total_gt == 0:
            continue
        if len(tp_list) == 0:
            aps.append(0.0)
            continue

        tp_arr = np.array(tp_list)
        fp_arr = np.array(fp_list)
        scores_arr = np.array(scores_list)

        order = np.argsort(-scores_arr)
        tp_arr = tp_arr[order]
        fp_arr = fp_arr[order]

        acc_tp = np.cumsum(tp_arr)
        acc_fp = np.cumsum(fp_arr)

        rec = acc_tp / total_gt
        prec = acc_tp / (acc_tp + acc_fp + 1e-8)

        # 11-point interpolation or area under PR curve
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        i_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_idx + 1] - mrec[i_idx]) * mpre[i_idx + 1])
        aps.append(ap)

    val_map50 = float(np.mean(aps)) if len(aps) > 0 else 0.0
    return val_map50


def train_torchvision_model(model_type, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                            imgsz=DEFAULT_IMGSZ, lr=DEFAULT_LR, device="cpu", seed=DEFAULT_SEED):
    """
    Training pipeline untuk Torchvision detection models: RetinaNet, SSD, Faster R-CNN.
    """
    set_seed(seed)
    save_dir = get_model_dir(model_type)

    spec = MODEL_REGISTRY[model_type]
    bg_offset = spec["background_offset"]

    print(f"\n[INFO] Membangun arsitektur {spec['name']}...")
    model, meta = build_detector(model_type, num_classes=NUM_CLASSES, pretrained=True, imgsz=imgsz)
    model.to(device)

    # Siapkan DataLoader
    print(f"[INFO] Menyiapkan Training dan Validation Loader...")
    train_loader = get_dataloader(split="train", imgsz=imgsz, batch_size=batch_size,
                                  augment=True, background_offset=bg_offset, workers=DEFAULT_WORKERS)
    val_loader = get_dataloader(split="val", imgsz=imgsz, batch_size=batch_size,
                                augment=False, background_offset=bg_offset, workers=DEFAULT_WORKERS)

    # Optimizer dan Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=DEFAULT_MOMENTUM, weight_decay=DEFAULT_WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    print(f"[INFO] Konfigurasi Eksperimen {model_type.upper()}:")
    print(f"  Model Name       : {spec['name']}")
    print(f"  Total Parameters : {meta['total_parameters']:,}")
    print(f"  Image Size       : {imgsz}x{imgsz}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Epochs           : {epochs}")
    print(f"  Learning Rate    : {lr} (Cosine Annealing)")
    print(f"  Device           : {device}")
    print(f"  Checkpoint Path  : {save_dir}/best_checkpoint.pth\n")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_map50": [],
        "lr": [],
        "epoch_time_sec": []
    }

    best_val_map = 0.0
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)

        print(f"\n{'='*75}", flush=True)
        print(f"  >>> MEMULAI TRAINING: EPOCH [{epoch:02d}/{epochs:02d}] (Total {total_batches} Batch) <<<", flush=True)
        print(f"{'='*75}", flush=True)

        for batch_i, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                print(f"[WARNING] Loss tidak finite pada epoch {epoch} batch {batch_i}: {losses.item()}", flush=True)
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            running_loss += losses.item()
            batch_count += 1

            # Cetak indikator progres real-time setiap 25 batch (dan batch awal/akhir)
            if batch_i == 1 or batch_i % 25 == 0 or batch_i == total_batches:
                elapsed_b = time.time() - epoch_start
                avg_b_time = elapsed_b / batch_i
                eta_sec = avg_b_time * (total_batches - batch_i)
                curr_loss = losses.item()
                avg_loss_so_far = running_loss / batch_count
                pct = (batch_i / total_batches) * 100.0
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch [{batch_i:03d}/{total_batches}] ({pct:5.1f}%) | "
                      f"Loss Rata2: {avg_loss_so_far:.4f} (curr: {curr_loss:.4f}) | "
                      f"Speed: {avg_b_time:.2f}s/batch | ETA: {eta_sec/60:.1f}m", flush=True)

        avg_loss = running_loss / max(1, batch_count)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Validasi pada Validation Set
        print(f"\n  [Validasi] Mengevaluasi mAP@50 pada Validation Set (1.048 gambar)...", flush=True)
        val_map50 = evaluate_validation_map50(model, val_loader, device, background_offset=bg_offset)
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_map50"].append(val_map50)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        print(f"\n[RINGKASAN EPOCH {epoch:02d}/{epochs:02d}]", flush=True)
        print(f"  Loss Training : {avg_loss:.4f}", flush=True)
        print(f"  Val mAP@0.50  : {val_map50:.4f} ({val_map50*100:.2f}%)", flush=True)
        print(f"  Learning Rate : {current_lr:.6f}", flush=True)
        print(f"  Waktu Epoch   : {epoch_time:.1f}s ({epoch_time/60:.1f} menit)", flush=True)

        # Simpan checkpoint
        is_best = val_map50 > best_val_map
        if is_best:
            best_val_map = val_map50
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_map50": val_map50,
                "meta": meta
            }, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_map50": val_map50,
            "meta": meta
        }, os.path.join(save_dir, "last_checkpoint.pth"))

    total_time = time.time() - start_total_time
    print(f"\n[DONE] Training {model_type.upper()} selesai dalam {total_time/3600:.2f} jam ({total_time/60:.1f} menit).")
    print(f"[INFO] Best Val mAP@50: {best_val_map:.4f} tersimpan di {save_dir}/best_checkpoint.pth")

    # Simpan log riwayat training ke JSON dan CSV
    log_csv_path = os.path.join(save_dir, "training_history.csv")
    with open(log_csv_path, "w") as f:
        f.write("epoch,train_loss,val_map50,lr,epoch_time_sec\n")
        for ep, l, vm, clr, t in zip(history["epoch"], history["train_loss"], history["val_map50"], history["lr"], history["epoch_time_sec"]):
            f.write(f"{ep},{l:.6f},{vm:.6f},{clr:.8f},{t:.2f}\n")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="tab:blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{spec['name']} - Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_map50"], label="Val mAP@50", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.50")
    plt.title(f"{spec['name']} - Validation mAP@50")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=200)
    plt.close()


def train_ultralytics_model(model_type, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                            imgsz=DEFAULT_IMGSZ, device="cpu", seed=DEFAULT_SEED):
    """
    Training pipeline untuk model berbasis Ultralytics (RT-DETR dan YOLO26n).
    """
    set_seed(seed)
    save_dir = get_model_dir(model_type)
    save_project = os.path.dirname(save_dir)
    save_name = os.path.basename(save_dir)

    spec = MODEL_REGISTRY[model_type]
    print(f"\n[INFO] Membangun arsitektur {spec['name']}...")
    model, meta = build_detector(model_type, num_classes=NUM_CLASSES, pretrained=True, imgsz=imgsz)

    print(f"[INFO] Konfigurasi Eksperimen {model_type.upper()}:")
    print(f"  Model Name       : {spec['name']}")
    print(f"  Image Size       : {imgsz}x{imgsz}")
    print(f"  Batch Size       : {batch_size}")
    print(f"  Epochs           : {epochs}")
    print(f"  Device           : {device}")
    print(f"  Output Project   : {save_project}/{save_name}\n")

    # Eksekusi training Ultralytics murni baseline
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=DEFAULT_WORKERS,
        device=device,
        project=save_project,
        name=save_name,
        exist_ok=False,
        seed=seed,
        patience=15,
        plots=True,
        verbose=True
    )
    print(f"\n[DONE] Training {model_type.upper()} selesai! Output tersimpan di {save_project}/{save_name}")
    return results


def main():
    parser = argparse.ArgumentParser(description="FASE 2: Unified Baseline Training Module")
    parser.add_argument("--model", type=str, required=True, choices=["retinanet", "ssd", "faster_rcnn", "rtdetr", "yolo26n"],
                        help="Pilih model object detection baseline")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Jumlah epoch training (default: 50)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 4)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Resolusi gambar (default: 640)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device target ('cpu' atau 'cuda')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("  FASE 2: MODEL SELECTION - TRAINING PIPELINE")
    print("=" * 70)

    if args.model in ["retinanet", "ssd", "faster_rcnn"]:
        train_torchvision_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr=args.lr,
            device=args.device,
            seed=args.seed
        )
    elif args.model in ["rtdetr", "yolo26n"]:
        train_ultralytics_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
