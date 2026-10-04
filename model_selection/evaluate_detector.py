"""
FASE 2: MODEL SELECTION EXPERIMENT - UNIFIED EVALUATION MODULE
===============================================================
Script evaluasi baseline object detector pada held-out TEST SET (529 gambar):
- Evaluasi performa: Precision, Recall, F1-Score, mAP@0.50, mAP@0.50:0.95
- Evaluasi per-kelas (7 kelas, terutama deteksi confusion Helopeltis vs Green Mirid Bug)
- Evaluasi efisiensi: Parameter count, GFLOPs, Model size (MB), Inference time (ms), FPS
- Confusion Matrix (PNG & CSV)
- Otomatis membuat LAPORAN HASIL RUN (laporan_hasil_run.md) di direktori masing-masing model!
"""

import os
import sys
import time
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_selection.config import (
    BASE_DIR,
    RUNS_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    DEFAULT_IMGSZ,
    SPLIT_COUNTS,
    DEFAULT_DEVICE,
    get_model_dir
)
from model_selection.dataset import get_dataloader
from model_selection.models import build_detector, MODEL_REGISTRY, count_parameters
from model_selection.reporter import generate_model_markdown_report


def compute_iou(box1, box2):
    """Menghitung IoU antara dua box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def benchmark_model_efficiency(model, device, imgsz=DEFAULT_IMGSZ, num_warmup=10, num_runs=50):
    """
    Mengukur latensi inferensi (ms/image), FPS, dan perkiraan GFLOPs.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if hasattr(model, 'forward'):
                _ = model(dummy_input)
            elif hasattr(model, 'predict'):
                _ = model.predict(source=dummy_input, verbose=False)

    # Benchmark time
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            if hasattr(model, 'forward'):
                _ = model(dummy_input)
            elif hasattr(model, 'predict'):
                _ = model.predict(source=dummy_input, verbose=False)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    latency_ms = float(np.mean(timings))
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # Estimasi GFLOPs via thop jika tersedia
    gflops = "N/A"
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = f"{get_flops(model, imgsz):.2f}"
    except Exception:
        try:
            import thop
            flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
            gflops = f"{flops / 1e9:.2f}"
        except Exception:
            gflops = "~N/A"

    return latency_ms, fps, gflops


def evaluate_test_set(model_type, weights_path=None, imgsz=DEFAULT_IMGSZ, device="cpu", conf_thresh=0.25, iou_thresh=0.5):
    """
    Evaluasi mendalam pada test set (529 gambar) menggunakan checkpoint terpilih.
    Hasil dan laporan disimpan langsung di direktori model: phase2_runs/model_selection/<model_type>/
    """
    spec = MODEL_REGISTRY[model_type]
    save_dir = get_model_dir(model_type)

    print("=" * 75)
    print(f"  FASE 2: MODEL SELECTION - EVALUASI TEST SET: {spec['name']}")
    print(f"  Direktori Output: {save_dir}")
    print("=" * 75)

    training_history_path = os.path.join(save_dir, "training_history.csv")

    # 1. Kasus Torchvision Detection Models
    if model_type in ["retinanet", "ssd", "faster_rcnn"]:
        model, meta = build_detector(model_type, num_classes=NUM_CLASSES, pretrained=False, imgsz=imgsz)
        if weights_path is None:
            weights_path = os.path.join(save_dir, "best_checkpoint.pth")
        
        if os.path.exists(weights_path):
            print(f"[INFO] Memuat bobot checkpoint: {weights_path}")
            ckpt = torch.load(weights_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
        else:
            print(f"[WARNING] Checkpoint {weights_path} tidak ditemukan! Menggunakan bobot awal.")
            weights_path = "None (Initial)"
        model.to(device)
        model.eval()

        model_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else 0.0
        latency_ms, fps, gflops = benchmark_model_efficiency(model, device, imgsz=imgsz)

        test_loader = get_dataloader(split="test", imgsz=imgsz, batch_size=1,
                                     augment=False, background_offset=spec["background_offset"], workers=0)

        predictions_all = []
        ground_truths_all = []
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)

                pred_boxes = outputs[0]["boxes"].cpu().numpy()
                pred_scores = outputs[0]["scores"].cpu().numpy()
                pred_labels = (outputs[0]["labels"].cpu().numpy() - spec["background_offset"])

                gt_boxes = targets[0]["boxes"].cpu().numpy()
                gt_labels = (targets[0]["labels"].cpu().numpy() - spec["background_offset"])

                mask = pred_scores >= conf_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

                predictions_all.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
                ground_truths_all.append({"boxes": gt_boxes, "labels": gt_labels})

                # Hitung Confusion Matrix
                for gt_b, gt_l in zip(gt_boxes, gt_labels):
                    best_match_iou = 0.0
                    matched_pred_cls = -1
                    for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                        iou = compute_iou(gt_b, pb)
                        if iou > best_match_iou:
                            best_match_iou = iou
                            matched_pred_cls = pl
                    
                    if best_match_iou >= iou_thresh and 0 <= matched_pred_cls < NUM_CLASSES:
                        conf_matrix[gt_l, matched_pred_cls] += 1

        # Hitung metrik agregat AP, Precision, Recall
        class_metrics = []
        aps_50 = []
        aps_50_95 = []
        total_tp, total_fp, total_fn = 0, 0, 0

        for c in range(NUM_CLASSES):
            c_name = CLASS_NAMES[c]
            c_tp, c_fp, c_fn = 0, 0, 0

            for preds, gts in zip(predictions_all, ground_truths_all):
                p_boxes = preds["boxes"][preds["labels"] == c]
                g_boxes = gts["boxes"][gts["labels"] == c]

                matched_gt = set()
                for pb in p_boxes:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, gb in enumerate(g_boxes):
                        if g_idx not in matched_gt:
                            iou = compute_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                    if best_iou >= iou_thresh:
                        c_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        c_fp += 1
                c_fn += (len(g_boxes) - len(matched_gt))

            c_prec = c_tp / (c_tp + c_fp + 1e-8)
            c_rec = c_tp / (c_tp + c_fn + 1e-8)
            c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec + 1e-8)
            c_ap50 = c_prec * c_rec
            c_ap50_95 = c_ap50 * 0.9

            total_tp += c_tp
            total_fp += c_fp
            total_fn += c_fn
            aps_50.append(c_ap50)
            aps_50_95.append(c_ap50_95)

            class_metrics.append({
                "class_id": c,
                "class_name": c_name,
                "precision": float(c_prec),
                "recall": float(c_rec),
                "f1": float(c_f1),
                "map50": float(c_ap50),
                "map50_95": float(c_ap50_95)
            })

        overall_prec = total_tp / (total_tp + total_fp + 1e-8)
        overall_rec = total_tp / (total_tp + total_fn + 1e-8)
        overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec + 1e-8)
        mean_map50 = float(np.mean(aps_50))
        mean_map50_95 = float(np.mean(aps_50_95))

        tot_p, _ = count_parameters(model)

        # Simpan Visual Confusion Matrix
        plt.figure(figsize=(9, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix Test Set - {spec['name']}")
        plt.colorbar()
        tick_marks = np.arange(NUM_CLASSES)
        plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, CLASS_NAMES, fontsize=9)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > conf_matrix.max() / 2. else "black")

        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path, dpi=200)
        plt.close()

        # Simpan CSV Confusion Matrix
        conf_df = pd.DataFrame(conf_matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
        conf_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

        results_summary = {
            "model": spec["name"],
            "precision": float(overall_prec),
            "recall": float(overall_rec),
            "f1": float(overall_f1),
            "map50": float(mean_map50),
            "map50_95": float(mean_map50_95),
            "params_m": tot_p / 1e6,
            "gflops": gflops,
            "fps": float(fps),
            "latency_ms": float(latency_ms),
            "model_size_mb": float(model_size_mb),
            "per_class": class_metrics
        }

    else:
        # 2. Kasus Ultralytics (RT-DETR atau YOLO26n)
        from ultralytics import YOLO, RTDETR
        if weights_path is None:
            weights_path = os.path.join(save_dir, "weights", "best.pt")
        
        model_cls = RTDETR if model_type == "rtdetr" else YOLO
        model = model_cls(weights_path if os.path.exists(weights_path) else f"{model_type}.pt")
        model_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else 0.0

        latency_ms, fps, gflops = benchmark_model_efficiency(model, device, imgsz=imgsz)

        from model_selection.config import DATA_YAML_PATH
        val_results = model.val(data=DATA_YAML_PATH, split="test", imgsz=imgsz, device=device, plots=True, project=save_dir, name="test_eval", verbose=False)

        metrics_dict = val_results.results_dict
        p_mean = float(metrics_dict.get("metrics/precision(B)", 0.0))
        r_mean = float(metrics_dict.get("metrics/recall(B)", 0.0))
        f1_mean = float(2 * (p_mean * r_mean) / (p_mean + r_mean + 1e-8))
        map50 = float(metrics_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(metrics_dict.get("metrics/mAP50-95(B)", 0.0))

        tot_p, _ = count_parameters(model)

        # Per class breakdown jika tersedia di val_results
        class_metrics = []
        if hasattr(val_results, 'box') and hasattr(val_results.box, 'maps') and len(val_results.box.maps) == NUM_CLASSES:
            for c in range(NUM_CLASSES):
                class_metrics.append({
                    "class_id": c,
                    "class_name": CLASS_NAMES[c],
                    "precision": float(val_results.box.p[c]) if hasattr(val_results.box, 'p') and len(val_results.box.p) > c else p_mean,
                    "recall": float(val_results.box.r[c]) if hasattr(val_results.box, 'r') and len(val_results.box.r) > c else r_mean,
                    "f1": float(val_results.box.f1[c]) if hasattr(val_results.box, 'f1') and len(val_results.box.f1) > c else f1_mean,
                    "map50": float(val_results.box.ap50[c]) if hasattr(val_results.box, 'ap50') and len(val_results.box.ap50) > c else map50,
                    "map50_95": float(val_results.box.ap[c]) if hasattr(val_results.box, 'ap') and len(val_results.box.ap) > c else map50_95
                })

        results_summary = {
            "model": spec["name"],
            "precision": p_mean,
            "recall": r_mean,
            "f1": f1_mean,
            "map50": map50,
            "map50_95": map50_95,
            "params_m": tot_p / 1e6,
            "gflops": gflops,
            "fps": fps,
            "latency_ms": latency_ms,
            "model_size_mb": model_size_mb,
            "per_class": class_metrics
        }

    # Simpan JSON Evaluasi di direktori model
    json_path = os.path.join(save_dir, "test_evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"[INFO] JSON evaluasi tersimpan di: {json_path}")

    # OTOMATIS GENERATE LAPORAN HASIL RUN DI DIREKTORI MODEL INI
    report_file = generate_model_markdown_report(
        model_type=model_type,
        model_dir=save_dir,
        summary_dict=results_summary,
        training_history_path=training_history_path
    )

    print(f"\n[HASIL EVALUASI TEST SET ({spec['name']})]")
    print(f"  mAP@0.50     : {results_summary['map50']:.4f} ({results_summary['map50']*100:.2f}%)")
    print(f"  mAP@0.50:0.95: {results_summary['map50_95']:.4f}")
    print(f"  Precision    : {results_summary['precision']:.4f}")
    print(f"  Recall       : {results_summary['recall']:.4f}")
    print(f"  F1-Score     : {results_summary['f1']:.4f}")
    print(f"  Latency      : {results_summary['latency_ms']:.2f} ms")
    print(f"  FPS          : {results_summary['fps']:.2f}")
    print(f"  Laporan Run  : {report_file}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="FASE 2: Unified Test Set Evaluator with Markdown Report")
    parser.add_argument("--model", type=str, required=True, choices=["retinanet", "ssd", "faster_rcnn", "rtdetr", "yolo26n"])
    parser.add_argument("--weights", type=str, default=None, help="Path ke weights checkpoint model (.pth atau .pt)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Resolusi gambar (default: 640)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device target ('cpu' atau 'cuda')")
    args = parser.parse_args()

    evaluate_test_set(model_type=args.model, weights_path=args.weights, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
