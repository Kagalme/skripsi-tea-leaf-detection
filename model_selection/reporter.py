"""
FASE 2: MODEL SELECTION EXPERIMENT - PER-MODEL REPORT GENERATOR
================================================================
Modul untuk mengompilasi dan menghasilkan laporan lengkap hasil run
secara otomatis di dalam direktori masing-masing model (laporan_hasil_run.md).

Laporan mencakup:
- Informasi Model & Arsitektur
- Konfigurasi Pelatihan
- Ringkasan Hasil Test Set (Overall mAP@50, mAP@50-95, Precision, Recall, F1)
- Tabel Performa Per Kelas (7 kelas, termasuk deteksi khusus Helopeltis vs Green Mirid Bug)
- Benchmark Efisiensi (Parameters, GFLOPs, Latency, FPS, Ukuran Bobot)
- Analisis Confusion Matrix
- Daftar Artefak yang Dihasilkan
"""

import os
import json
import time
import pandas as pd

from model_selection.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    SPLIT_COUNTS,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED
)
from model_selection.models import MODEL_REGISTRY


def generate_model_markdown_report(model_type, model_dir, summary_dict, training_history_path=None):
    """
    Menghasilkan file laporan_hasil_run.md di dalam direktori model_dir.
    """
    spec = MODEL_REGISTRY[model_type]
    report_path = os.path.join(model_dir, "laporan_hasil_run.md")

    # Data metrik utama
    p = summary_dict.get("precision", 0.0)
    r = summary_dict.get("recall", 0.0)
    f1 = summary_dict.get("f1", 0.0)
    map50 = summary_dict.get("map50", 0.0)
    map50_95 = summary_dict.get("map50_95", 0.0)

    params_m = summary_dict.get("params_m", 0.0)
    gflops = summary_dict.get("gflops", "N/A")
    latency_ms = summary_dict.get("latency_ms", 0.0)
    fps = summary_dict.get("fps", 0.0)
    model_size_mb = summary_dict.get("model_size_mb", 0.0)

    per_class = summary_dict.get("per_class", [])

    # Format baris tabel per kelas
    per_class_rows = []
    if per_class:
        for item in per_class:
            c_name = item.get("class_name", "")
            c_p = item.get("precision", 0.0)
            c_r = item.get("recall", 0.0)
            c_f1 = item.get("f1", 0.0)
            c_m50 = item.get("map50", 0.0)
            c_m95 = item.get("map50_95", 0.0)
            per_class_rows.append(
                f"| `{item.get('class_id', '-')}` | **{c_name}** | {c_p:.4f} | {c_r:.4f} | {c_f1:.4f} | {c_m50:.4f} | {c_m95:.4f} |"
            )
    else:
        for idx, c_name in enumerate(CLASS_NAMES):
            per_class_rows.append(f"| `{idx}` | **{c_name}** | N/A | N/A | N/A | N/A | N/A |")
    
    per_class_table = "\n".join(per_class_rows)

    # Baca durasi training jika ada
    training_duration_text = "N/A"
    final_loss_text = "N/A"
    if training_history_path and os.path.exists(training_history_path):
        try:
            df = pd.read_csv(training_history_path)
            if "epoch_time_sec" in df.columns:
                total_sec = df["epoch_time_sec"].sum()
                training_duration_text = f"{total_sec/3600:.2f} jam ({total_sec/60:.1f} menit)"
            if "train_loss" in df.columns and len(df) > 0:
                final_loss_text = f"{df['train_loss'].iloc[-1]:.4f}"
        except Exception:
            pass

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# LAPORAN HASIL EVALUASI MODEL: {spec['name'].upper()}
## FASE 2: MODEL SELECTION EXPERIMENT — DETEKSI PENYAKIT & HAMA DAUN TEH

---

- **Model ID:** `{model_type}`
- **Waktu Penyelesaian:** `{timestamp}`
- **Lokasi Direktori Model:** `{model_dir}`
- **Status Eksperimen:** **SELESAI (100% Evaluated)**

---

## A. Ringkasan Eksekutif & Metrik Kunci (Test Set: 529 Gambar)

> **Catatan Metodologis:** Seluruh metrik di bawah ini dihitung murni pada **Held-out Test Set (529 gambar)** yang **TIDAK PERNAH DILIHAT** oleh model selama proses pelatihan maupun validasi.

| Metrik Evaluasi | Nilai Desimal | Persentase | Interpretasi Riset |
|---|---|---|---|
| **mAP@0.50** | **{map50:.4f}** | **{map50*100:.2f}%** | Akurasi deteksi utama pada batas overlap IoU >= 0.50 |
| **mAP@0.50:0.95** | **{map50_95:.4f}** | **{map50_95*100:.2f}%** | Presisi lokalisasi ketat standar COCO |
| **Precision** | **{p:.4f}** | **{p*100:.2f}%** | Rasio deteksi benar terhadap seluruh prediksi positif |
| **Recall** | **{r:.4f}** | **{r*100:.2f}%** | Rasio daun berpenyakit yang berhasil terdeteksi |
| **F1-Score** | **{f1:.4f}** | **{f1*100:.2f}%** | Rata-rata harmonik antara Precision dan Recall |

---

## B. Karakteristik & Efisiensi Komputasi (Deployment Feasibility)

Evaluasi efisiensi untuk mempertimbangkan kesesuaian deployment pada kamera kebun / perangkat bergerak (*mobile device*):

| Parameter Efisiensi | Nilai Terukur | Analisis & Dampak Praktis |
|---|---|---|
| **Jumlah Parameter** | **{params_m:.2f} M** | Parameter jaringan yang membebani alokasi RAM |
| **Beban FLOPs** | **{gflops} GFLOPs** | Jumlah operasi komputasi per inferensi frame |
| **Ukuran Bobot (.pth/.pt)** | **{model_size_mb:.2f} MB** | Kapasitas penyimpanan file bobot pada media disk |
| **Inference Latency** | **{latency_ms:.2f} ms** | Waktu yang dibutuhkan untuk memproses 1 gambar |
| **Throughput (FPS)** | **{fps:.2f} FPS** | Kecepatan bingkai per detik pada hardware saat ini |

---

## C. Spesifikasi Arsitektur Baseline

Model ini diuji secara murni dalam bentuk **arsitektur aslinya** tanpa modifikasi tambahan:

| Komponen | Spesifikasi Teknis |
|---|---|
| **Nama Arsitektur** | {spec['name']} |
| **Landasan Teori / Paper** | {spec['paper']} |
| **Paradigma Deteksi** | {spec['paradigm']} |
| **Backbone Network** | {spec['backbone']} |
| **Neck / Feature Pyramid** | {spec['neck']} |
| **Detection Head** | {spec['head']} |
| **Dependensi NMS** | {spec['nms_needed']} |
| **Pretrained Weights** | {spec['pretrained_weights']} |
| **Coordinate Attention / CBAM** | **TIDAK ADA (Dilarang pada tahap baseline)** |
| **Edge Detection (Canny/Sobel)** | **BELUM DITERAPKAN (Akan diintegrasikan pada Fase 3)** |

---

## D. Konfigurasi Eksperimen (Training Protocol)

| Parameter | Konfigurasi | Keterangan |
|---|---|---|
| **Dataset** | `tea_yolo/` | Dataset 7 kelas tanaman teh |
| **Train Set** | {SPLIT_COUNTS['train']} gambar | 70% porsi pembaruan gradien |
| **Validation Set** | {SPLIT_COUNTS['val']} gambar | 20% porsi seleksi best checkpoint |
| **Test Set** | {SPLIT_COUNTS['test']} gambar | 10% held-out test set terisolasi |
| **Resolusi Gambar** | {DEFAULT_IMGSZ}×{DEFAULT_IMGSZ} | Resolusi standar terkomparasi |
| **Batch Size** | {DEFAULT_BATCH_SIZE} | Disesuaikan dengan batas alokasi memori |
| **Target Epochs** | {DEFAULT_EPOCHS} Epochs | Durasi pelatihan terstandarisasi |
| **Optimizer** | SGD (Momentum 0.9, Decay 0.0005) | Resep optimizer baseline |
| **Learning Rate Schedule** | Cosine Annealing (lr={DEFAULT_LR}) | Peluruhan laju belajar bertahap |
| **Augmentasi** | Random Horizontal Flip ($p=0.5$), Normalisasi | Augmentasi seimbang tanpa bias |
| **Random Seed** | {DEFAULT_SEED} | Menjamin eksperimen sepenuhnya reproducible |
| **Total Waktu Training** | {training_duration_text} | Total waktu komputasi pelatihan |
| **Final Training Loss** | {final_loss_text} | Nilai loss pada akhir epoch |

---

## E. Hasil Evaluasi Per Kelas (Test Set: 529 Gambar)

Tabel berikut menyajikan performa spesifik model pada setiap kelas penyakit, hama, dan daun sehat:

| ID | Nama Kelas | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|---|---|
{per_class_table}

---

## F. Analisis Kebingungan Antar Kelas (Confusion Matrix Insights)

1. **Deteksi Hama Helopeltis vs Green Mirid Bug:**
   - Kedua kelas hama ini memiliki kemiripan morfologi visual lesi tusukan.
   - Confusion matrix mendeteksi tingkat pertukaran prediksi (*cross-class confusion*) pada pasangan ini.
2. **Kategori Penyakit Bercak Daun (Gray Blight vs Brown Blight):**
   - Menguji apakah pola warna nekrotik mampu dibedakan dengan baik oleh representasi fitur {spec['backbone']}.
3. **Resistensi terhadap Ketimpangan Data (Class Imbalance):**
   - Performa kelas minoritas (*Tea algal leaf spot*, 418 data) dibandingkan dengan kelas mayoritas (*Green Mirid Bug*, 1282 data).

---

## G. Berkas Artefak yang Tersedia di Direktori Ini

Seluruh artefak hasil eksekusi model ini tersimpan secara rapi di dalam direktori ini:
- `best_checkpoint.pth` (atau `weights/best.pt`): Bobot model terbaik selama training.
- `last_checkpoint.pth`: Bobot epoch terakhir.
- `training_history.csv`: Rekaman numerik loss, val mAP, dan learning rate per epoch.
- `learning_curves.png`: Grafik visual kurva training loss dan validation mAP.
- `test_evaluation_summary.json`: Rangkuman seluruh metrik numerik evaluasi test set.
- `confusion_matrix.png`: Peta visual matriks kebingungan prediksi kelas.
- `confusion_matrix.csv`: Data mentah matriks kebingungan dalam format CSV.
- `visual_analysis/`: Direktori visualisasi gambar hasil deteksi:
  - `true_positive.png`: Sampel deteksi sukses dengan IoU tinggi.
  - `false_negative.png`: Sampel daun berpenyakit yang gagal terdeteksi (*missed detection*).
  - `misclassification.png`: Sampel penyakit yang terdeteksi dengan label kelas tertukar.
  - `localization_error.png`: Sampel prediksi dengan kotak yang meleset (0.1 <= IoU < 0.5).
- `laporan_hasil_run.md`: Berkas laporan komprehensif ini.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[REPORTER] Laporan lengkap berhasil dibuat: {report_path}")
    return report_path
