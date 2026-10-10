# PANDUAN PELAKSANAAN FASE 2: YOLO26n RESOLUSI STANDAR (640x640)

Panduan ini berisi instruksi lengkap untuk menjalankan pelatihan, evaluasi test set, dan visualisasi gambar prediksi menggunakan **model baseline YOLO26n resmi** yang telah disesuaikan ke **resolusi standar 640×640**.

---

## 📌 Ringkasan Penyesuaian Utama
| Komponen | Baseline Sebelumnya | Baseline Baru (Res 640) | Dampak Ilmiah |
| :--- | :---: | :---: | :--- |
| **Resolusi Citra (`imgsz`)** | `416` | **`640`** | Mempertahankan detail tekstur lesi kecil (*Tea algal*, *Helopeltis*) |
| **Epoch** | `50` | **`80`** | Memberi ruang model konvergen lebih optimal |
| **Patience** | `15` | **`20`** | Early stopping aman jika model sudah tidak membaik |
| **Learning Rate Scheduler** | Linear | **Cosine Annealing (`cos_lr=True`)** | Penurunan bobot yang lebih stabil dan halus |
| **Augmentasi Orientasi** | `degrees=0, flipud=0` | **`degrees=15.0, flipud=0.5`** | Menirukan variasi sudut daun nyata di perkebunan teh |
| **Batch Size** | `4` (CPU) | **`16`** (GPU Colab) | Menstabilkan statistik Batch Normalization |

---

## 🚀 Alur Eksekusi di Google Colab

### Langkah 1: Siapkan Lingkungan & Mount Google Drive
```python
from google.colab import drive
import os
drive.mount('/content/drive')

# Ekstrak dataset jika belum diekstrak
if not os.path.exists('/content/tea_yolo'):
    !unzip -q /content/drive/MyDrive/SKRIPSI/tea_yolo.zip -d /content/

# Salin script phase2_res640_scripts ke /content
!cp -r /content/drive/MyDrive/SKRIPSI/phase2_res640_scripts /content/
```

### Langkah 2: Periksa Lingkungan & GPU
```python
!python /content/phase2_res640_scripts/01_env_check.py
```

### Langkah 3: Jalankan Pelatihan Baseline 640
```python
!python /content/phase2_res640_scripts/02_train_baseline_640.py
```
*Proses ini memakan waktu sekitar 30–50 menit di GPU Tesla T4 Google Colab.*

### Langkah 4: Evaluasi pada Test Set
```python
!python /content/phase2_res640_scripts/03_evaluate_testset.py
```
*Menghasilkan tabel metrik per-kelas, `confusion_matrix.png`, serta kurva PR dan F1.*

### Langkah 5: Analisis Visual Kategorikal (TP, FP, WC, FN)
```python
!python /content/phase2_res640_scripts/04_visual_analysis.py
```
*Menghasilkan contoh gambar yang dikategorikan berdasarkan keberhasilan dan jenis kesalahan model.*

### Langkah 6: Inferensi & Tampilkan Galeri Gambar Test Set
```python
!python /content/phase2_res640_scripts/05_infer_and_display_test.py
```
*Menyimpan seluruh citra test set yang diberi bounding box dan menghasilkan `gallery_test_predictions.png` (grid 3×3).*

### Langkah 7: Tampilkan Gambar Langsung di Layar Colab
```python
from IPython.display import Image, display

# 1. Tampilkan Galeri Grid 3x3
display(Image('/content/phase2_runs/yolo26n_baseline_640/test_predictions/gallery_test_predictions.png', width=900))

# 2. Tampilkan Confusion Matrix
display(Image('/content/phase2_runs/yolo26n_baseline_640/test_eval/test_results/confusion_matrix.png', width=700))
```

---

## 💾 Menyimpan Hasil Kembali ke Google Drive
Agar bobot model dan grafik tidak hilang saat runtime Colab ditutup:
```python
!cp -r /content/phase2_runs/yolo26n_baseline_640 /content/drive/MyDrive/SKRIPSI/phase2_runs/
print("Semua hasil training & evaluasi berhasil dicadangkan ke Google Drive!")
```
