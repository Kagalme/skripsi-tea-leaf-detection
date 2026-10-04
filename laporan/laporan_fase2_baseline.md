# LAPORAN FASE 2: BASELINE MODEL YOLO26n
## Penelitian Deteksi Penyakit Daun Teh

---

## A. Informasi Model

| Parameter | Nilai |
|-----------|-------|
| Arsitektur | YOLO26n (You Only Look Once v2.6 nano) |
| Parameters | 2,572,280 |
| GFLOPs | 6.2 |
| Pretrained | COCO (`yolo26n.pt`) |
| Framework | Ultralytics 8.4.171 |
| Task | Object Detection (7 kelas) |

---

## B. Konfigurasi Training

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| Dataset | `tea_yolo/` | Fase 1, leak-free split |
| Train set | 3,701 gambar | 70% |
| Val set | 1,048 gambar | 20% |
| Test set | 529 gambar | 10%, tidak dilihat saat training |
| Epochs | 50 | CPU-optimized (default: 100) |
| Image size | 416×416 | Dikurangi dari 640 untuk CPU |
| Batch size | 4 | CPU limit |
| Optimizer | auto (SGD) | Default Ultralytics recipe |
| Device | CPU | AMD64, 12 cores |
| Waktu training | ~6 jam | 7.7 menit/epoch |
| Augmentation | Default | Mosaic=1.0, RandAugment, Erasing=0.4, FlipLR=0.5 |
| Custom loss | **TIDAK ADA** | Pure baseline |
| Attention module | **TIDAK ADA** | Pure baseline |
| Seed | 42 | Reproducible |

---

## C. Dataset (7 Kelas)

| ID | Kelas | Total | Train | Val | Test |
|----|-------|-------|-------|-----|------|
| 0 | Tea algal leaf spot | 418 | 293 | 83 | 42 |
| 1 | Brown Blight | 508 | 355 | 102 | 51 |
| 2 | Gray Blight | 1013 | 711 | 201 | 101 |
| 3 | Helopeltis | 607 | 428 | 118 | 61 |
| 4 | Red Spider | 515 | 360 | 103 | 52 |
| 5 | Green Mirid Bug | 1282 | 899 | 255 | 128 |
| 6 | Healty Leaf | 935 | 655 | 186 | 94 |
| | **TOTAL** | **5,278** | **3,701** | **1,048** | **529** |

---

## D. Hasil Evaluasi — Test Set (best.pt)

> Test set = 529 gambar, **tidak pernah dilihat** selama training maupun validation.

### Overall Metrics

| Metric | Nilai | Persentase |
|--------|-------|-----------|
| **Precision** | **0.9387** | **93.87%** |
| **Recall** | **0.8794** | **87.94%** |
| **F1-Score** | **0.9081** | **90.81%** |
| **mAP@0.5** | **0.9469** | **94.69%** |
| **mAP@0.5:0.95** | **0.9070** | **90.70%** |

### Per-Class Metrics (Test Set)

| Kelas | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|----------|---------|--------------|
| Tea algal leaf spot | 0.9453 | 0.8234 | 0.8801 | 0.9400 | 0.900 |
| Brown Blight | 0.8786 | 0.8519 | 0.8651 | 0.8916 | 0.846 |
| Gray Blight | 0.9361 | 0.8416 | 0.8863 | 0.9373 | 0.892 |
| Helopeltis | 0.9805 | 0.8234 | 0.8951 | 0.9531 | 0.940 |
| Red Spider | 0.9147 | 0.8654 | 0.8893 | 0.9496 | 0.916 |
| Green Mirid Bug | 0.9372 | 0.9609 | 0.9489 | 0.9774 | 0.932 |
| Healty Leaf | 0.9782 | 0.9894 | **0.9838** | 0.9791 | 0.923 |
| **MEAN** | **0.9387** | **0.8794** | **0.9081** | **0.9469** | **0.907** |

### COCO Eval (faster-coco-eval)

| Metric | Nilai |
|--------|-------|
| AP @ IoU=0.50:0.95 (all) | **0.904** |
| AP @ IoU=0.50 (all) | **0.944** |
| AP @ IoU=0.75 (all) | **0.925** |
| AR @ maxDets=100 (all) | **0.958** |
| AR @ IoU=0.50 | 0.989 |
| AR @ IoU=0.75 | 0.973 |

---

## E. Perbandingan Val vs Test

| Metric | Val Set | Test Set | Δ |
|--------|---------|----------|---|
| Precision | 0.904 | 0.939 | +0.035 |
| Recall | 0.895 | 0.879 | -0.016 |
| mAP@0.5 | 0.942 | 0.947 | +0.005 |
| mAP@0.5:0.95 | 0.898 | 0.907 | +0.009 |

> ✅ **Tidak ada overfitting** — performa Test ≈ Val, bahkan sedikit lebih tinggi di beberapa metrik. Model generalize dengan baik.

---

## F. Analisis Kelas

### Kelas Terkuat
| Kelas | F1 | Keterangan |
|-------|----|-----------|
| Healty Leaf | **0.983** | Sampel terbanyak ke-2, fitur visual paling distinct |
| Green Mirid Bug | **0.949** | Sampel terbanyak, kerusakan khas mudah dikenali |

### Kelas Paling Lemah (perlu perhatian di Fase 3)
| Kelas | F1 | Masalah Utama | Analisis |
|-------|----|--------------|----------|
| **Brown Blight** | 0.865 | Recall rendah (0.852) | Sampel paling sedikit ke-2 (508), gejala mirip Gray Blight |
| **Tea algal leaf spot** | 0.880 | Recall terendah (0.823) | Sampel paling sedikit (418), bercak kecil sering miss |
| **Helopeltis** | 0.895 | Recall rendah (0.823) | Kerusakan bisa tampak mirip Brown Blight |

### Recall Rendah = Banyak False Negative
Kelas dengan recall < 0.88 menunjukkan model sering **gagal mendeteksi** (false negative), bukan salah mengklasifikasi. Ini mengindikasikan fitur spasial kelas tersebut kurang ditangkap — salah satu target Fase 3 (Coordinate Attention untuk meningkatkan perhatian spasial model).

---

## G. Smoke Test vs Final (Progres Training)

| Metrik | 3 Epoch (Smoke) | 50 Epoch (Final) | Improvement |
|--------|----------------|-----------------|-------------|
| mAP@0.5 | 0.796 | 0.947 | +0.151 (+19%) |
| mAP@0.5-0.95 | 0.743 | 0.907 | +0.164 (+22%) |
| Precision | 0.758 | 0.939 | +0.181 |
| Recall | 0.787 | 0.879 | +0.092 |

---

## H. Output Files

| File/Folder | Lokasi |
|-------------|--------|
| Training logs | `phase2_runs/yolo26n_baseline_50ep/results.csv` |
| Best weights | `phase2_runs/yolo26n_baseline_50ep/weights/best.pt` |
| Last weights | `phase2_runs/yolo26n_baseline_50ep/weights/last.pt` |
| Training plots | `phase2_runs/yolo26n_baseline_50ep/*.png` |
| Test metrics CSV | `phase2_runs/yolo26n_baseline_50ep/test_eval/test_results/per_class_metrics.csv` |
| Test metrics JSON | `phase2_runs/yolo26n_baseline_50ep/test_eval/test_results/summary_metrics.json` |
| Visual samples | `phase2_runs/yolo26n_baseline_50ep/visual_analysis/` |

---

## I. Kesimpulan Fase 2

**YOLO26n Baseline berhasil dilatih** dengan performa yang sangat baik:
- mAP@0.5 = **94.69%** pada test set
- F1 rata-rata = **90.81%**
- Tidak ada overfitting (val ≈ test)

**Angka baseline ini** akan menjadi tolok ukur untuk **Fase 3: YOLO26n + Coordinate Attention**.

### Target Improvement di Fase 3
Berdasarkan analisis kelas lemah:
1. Meningkatkan **Recall** kelas `Tea algal leaf spot`, `Brown Blight`, dan `Helopeltis` (< 0.86)
2. Coordinate Attention diharapkan membantu model fokus pada **fitur spasial lokal** yang khas untuk kelas-kelas tersebut

---

*Laporan dibuat otomatis oleh pipeline Fase 2*  
*Model: YOLO26n | Framework: Ultralytics 8.4.171 | Device: CPU*
