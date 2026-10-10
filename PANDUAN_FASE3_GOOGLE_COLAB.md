# PANDUAN LENGKAP: FASE 3 DI GOOGLE COLAB (GPU T4)
## YOLO26n + Coordinate Attention (CA) + Edge Detection Kontur Daun Utuh

Dokumen ini memandu Anda menjalankan **Fase 3** secara penuh di Google Colab:
1. **YOLO26n + Coordinate Attention (CA)** — melatih modul perhatian koordinat pada backbone P3, P4, P5 (50 epoch, GPU T4 ~10–12 menit).
2. **Evaluasi & Perbandingan Baseline vs CA** — menghasilkan tabel komparasi otomatis mAP@50, mAP@50-95, Precision, Recall, dan FPS.
3. **Edge Detection Kontur Daun (Canny + Morfologi)** — mengubah bounding box kotak standar menjadi poligon kontur yang mengikuti lekukan bentuk daun asli.
4. **Filter Daun Tidak Utuh (Incomplete Leaf Filter)** — mengeliminasi deteksi daun yang terpotong batas gambar/ROI dengan toleransi terhadap bentuk daun tidak sempurna/rusak penyakit.

---

## 📦 Berkas yang Perlu Ada di Google Drive (`Drive > SKRIPSI/`)

Pastikan folder **`SKRIPSI`** di Google Drive Anda memiliki berkas berikut:

| Nama Berkas | Sumber di Laptop | Keterangan |
|---|---|---|
| `phase3_scripts.zip` | `D:\SKRIPSI\phase3_scripts.zip` | Seluruh script Fase 3 (sudah dikemas otomatis) |
| `tea_yolo.zip` | `D:\SKRIPSI\tea_yolo.zip` | Dataset gambar & anotasi |
| `yolo26n.pt` | `D:\SKRIPSI\yolo26n.pt` | Pretrained weights model dasar |
| `Training_Phase3_CA_Colab.ipynb` | `D:\SKRIPSI\Training_Phase3_CA_Colab.ipynb` | Notebook Colab Fase 3 |

> **Catatan:** Jika Anda melakukan perubahan pada folder `phase3_scripts/`, kemas ulang dengan perintah PowerShell berikut:
> ```powershell
> Compress-Archive -Path D:\SKRIPSI\phase3_scripts -DestinationPath D:\SKRIPSI\phase3_scripts.zip -Force
> ```

---

## 🚀 Langkah Menjalankan di Google Colab

### 1. Buka Notebook di Colab
1. Masuk ke Google Drive $\to$ buka folder **`SKRIPSI`**.
2. Klik kanan pada **`Training_Phase3_CA_Colab.ipynb`** $\to$ **Buka dengan** $\to$ **Google Colaboratory**.

### 2. Aktifkan GPU T4
1. Klik menu **Runtime** di bilah atas $\to$ **Change runtime type**.
2. Pilih **T4 GPU** $\to$ klik **Save**.

### 3. Jalankan Seluruh Cell Berurutan
Cukup tekan tombol **Play ($\blacktriangleright$)** pada tiap cell:
- **Langkah 1–2:** Cek GPU dan Mount Google Drive.
- **Langkah 3–4:** Ekstrak dataset & kode ke NVMe disk lokal Colab.
- **Langkah 5–6:** Install dependency & jalankan unit test `ca_module.py` dan `edge_contour.py`.
- **Langkah 7 (Training):** Melatih model `yolo26n_ca_50ep` (~8–12 menit).
- **Langkah 8 (Evaluasi):** Evaluasi 529 gambar test set.
- **Langkah 9 (Visual Analisis):** Analisis kesalahan deteksi TP, FP, WC, FN.
- **Langkah 10 (Komparasi):** Membuat tabel perbandingan Baseline Fase 2 vs CA Fase 3.
- **Langkah 11 (Edge Contour Post-processing):** Menjalankan inferensi kontur bentuk daun dan penyaringan daun tidak utuh.
- **Langkah 12 (Simpan ke Drive):** Menyalin semua bobot terbaik (`best.pt`), grafik, dan laporan ke Google Drive.

---

## 🧠 Cara Kerja Edge Detection & Filter Daun Tidak Utuh

### 1. Mengapa Bounding Box Biasa Tidak Cukup?
Bounding box standar YOLO berbentuk kotak persegi panjang kaku ($x, y, w, h$). Pada daun teh yang posisinya miring atau melengkung, kotak persegi panjang menyertakan banyak latar belakang (daun lain/tanah/ranting).

### 2. Solusi: Pipeline Edge Contour (Pendekatan A)
```
YOLO26n + CA (Deteksi)
       │
       ▼
Bounding Box ROI
       │
       ▼
Preprocessing ROI (Gaussian Blur + CLAHE Enhancement)
       │
       ▼
Canny Edge Detection (Gradient Thresholding)
       │
       ▼
Morfologi Closing (Menutup celah pada tepi daun yang robek/berpenyakit)
       │
       ▼
Temukan Kontur Daun Terbesar
       │
       ├─── Kontur menyentuh batas ROI (Margin ≤ 5px)? ──► DAUN TIDAK UTUH (SUPPRESSED/DIBUANG)
       │
       └─── Kontur tertutup sempurna di dalam ROI? ─────► DAUN UTUH (POLYGON KONTUR DITAMPILKAN)
```

### 3. Mengatasi Daun yang Tidak Sempurna (Penyakit / Robek)
- **Tepi tidak rata / berlubang akibat bercak penyakit:**
  Modul menggunakan *Morphological Closing* dengan kernel $7\times 7$ yang secara otomatis menjembatani celah kecil pada tepi daun sehingga kontur tetap utuh dan tidak terputus.
- **Bukan filter bentuk kaku:**
  Sistem **tidak** menggunakan aturan bentuk kaku seperti kebulatan (*circularity*) atau rasio panjang/lebar tetap, karena daun teh alami memiliki variasi bentuk yang sangat dinamis.
- **Kriteria kelengkapan:**
  Filter difokuskan pada **apakah tepi kontur daun terpotong oleh bingkai foto / tepi bounding box**. Daun yang terpotong tepi kamera langsung dibuang dari hasil deteksi akhir.

---

## 📂 Struktur Output di Google Drive Setelah Selesai

```
Google Drive > SKRIPSI/
├── phase3_runs/
│   ├── yolo26n_ca_50ep/
│   │   ├── weights/
│   │   │   ├── best.pt              ← Bobot terbaik model YOLO26n + CA
│   │   │   └── last.pt
│   │   ├── results.png              ← Grafik kurva loss, mAP50, mAP50-95
│   │   ├── confusion_matrix.png
│   │   ├── test_eval/test_results/  ← Metrik evaluasi lengkap
│   │   └── visual_analysis/         ← Visualisasi TP, FP, WC, FN
│   │
│   └── contour_inference/           ← HASIL EDGE CONTOUR DAUN
│       ├── *_contour.jpg            ← Gambar dengan polygon kontur daun
│       └── contour_stats.json       ← Statistik daun utuh vs suppressed
│
└── laporan/
    └── fase3_ca_vs_baseline.md      ← LAPORAN KOMPARASI OTOMATIS
```

Semua berkas di atas siap diunduh kembali ke laptop untuk bahan bab hasil dan pembahasan skripsi Anda!
