# 🌿 FASE 3: YOLO26n + Coordinate Attention (CA) Pipeline Notebook (.ipynb)

Folder ini berisi versi **Notebook All-in-One Mandiri** yang dirancang khusus untuk dijalankan langsung di **Google Colab**.

## 📁 Berkas di dalam Folder
1. **`YOLO26n_CA_Pipeline.ipynb`**: Notebook Jupyter utama berisi alur lengkap:
   - **Cell 1**: Inisialisasi Environment, GPU Check, Mount Google Drive, dan Ekstraksi Dataset `tea_yolo.zip`.
   - **Cell 2**: Registrasi Modul `CoordAtt` & `CoordAttYOLO` ke PyTorch & Ultralytics langsung di memori.
   - **Cell 3**: Konfigurasi Arsitektur `yolo26n_ca.yaml` & Pemetaan Bobot Pretrained (*100% Remapping* dari `yolo26n.pt`).
   - **Cell 4**: Pelatihan Model (*Training* 50 Epoch atau *Fine-Tuning* Precision).
   - **Cell 5**: Evaluasi Lengkap pada Held-Out Test Set (529 Citra, 640×640) + Ringkasan Metrik Per-Kelas.
   - **Cell 6**: Visualisasi Kurva Konvergensi HD (Loss, Precision, Recall, mAP50, F1) & Confusion Matrix.
   - **Cell 7**: Engine Segmentasi Daun Kebun (v2.7): Multi-Daun, Eliminasi Ranting, Filter DoF Acutance, White-Twig Healing, dan 2-View TTA.
   - **Cell 8**: Uji Coba Interaktif (*Upload* Foto Daun Kebun dari Laptop/HP) dengan Tampilan 4-Panel.
   - **Cell 9**: Cadangkan Bobot Terbaik (`best.pt`) dan Laporan Otomatis ke Google Drive.
2. **`yolo26n_ca.yaml`**: Definisi YAML arsitektur asli YOLO26n dengan 3 blok `CoordAttYOLO` pada P3, P4, P5.

## 🚀 Keunggulan Versi Ini
- **Tanpa Kode Berulang (*DRY Principle*)**: Nama kelas, palet warna, dan fungsi utilitas didefinisikan satu kali secara berurutan dan digunakan bersama oleh seluruh cell.
- **Bebas Error Pickle / Namespace**: `CoordAttYOLO` didaftarkan langsung ke memori Python Colab dan modul Ultralytics, sehingga tidak akan memicu error `ModuleNotFoundError`.
- **Referensi Path Tetap Sama**: Otomatis mendeteksi file dataset `tea_yolo.zip` di Google Drive Anda (`/content/drive/MyDrive/SKRIPSI/tea_yolo.zip`).

