# PANDUAN LENGKAP: TRAINING MODEL SELECTION DI GOOGLE COLAB (GPU T4)
## Resolusi 640×640 (Sesuai Arahan Dosen) — Selesai dalam ~30–35 Menit!

Dokumen ini memandu Anda langkah demi langkah menjalankan eksperimen **Model Selection** di Google Colab menggunakan GPU T4 secara gratis, sehingga:
1. **Resolusi 640×640 tetap dipertahankan 100%** sesuai instruksi dosen pembimbing.
2. Waktu komputasi yang tadinya **20–50 jam di CPU** terpangkas menjadi hanya **~30–35 Menit** di GPU!
3. Laptop Anda tidak panas dan tidak perlu dipaksa menyala semalaman.
4. Laporan lengkap `laporan_hasil_run.md`, confusion matrix, visual analysis, dan checkpoint bobot terbaik otomatis tersimpan rapi di Google Drive Anda.

---

## TAHAP 1: Kemas Berkas di Komputer Lokal (Hanya 1 Menit)

Buka terminal di komputer Anda (`D:\SKRIPSI`) dan jalankan skrip pengemas otomatis:
```powershell
python 01_siapkan_berkas_colab.py
```
Skrip ini akan otomatis menghasilkan 2 berkas ZIP di folder `D:\SKRIPSI`:
1. `tea_yolo.zip` (Dataset gambar & anotasi 7 kelas)
2. `model_selection.zip` (Seluruh kode arsitektur model & pipeline evaluasi)

---

## TAHAP 2: Unggah ke Google Drive

1. Buka browser dan masuk ke **[Google Drive](https://drive.google.com)** menggunakan akun Google Anda.
2. Klik tombol **+ Baru (+ New)** $\to$ **Folder Baru (New Folder)**.
3. Beri nama folder: **`SKRIPSI`** (gunakan huruf kapital persis seperti ini).
4. Masuk ke dalam folder `SKRIPSI` tersebut, lalu unggah 3 berkas berikut:
   * `tea_yolo.zip`
   * `model_selection.zip`
   * `Training_Model_Selection_Colab.ipynb` (berkas notebook yang sudah saya siapkan di `D:\SKRIPSI\`)

---

## TAHAP 3: Buka Notebook di Google Colab

1. Di dalam Google Drive, klik kanan pada file **`Training_Model_Selection_Colab.ipynb`**.
2. Pilih **Buka dengan (Open with)** $\to$ **Google Colaboratory**.
   *(Jika belum terpasang, buka [colab.research.google.com](https://colab.research.google.com), klik tab **Upload**, dan pilih file `Training_Model_Selection_Colab.ipynb`).*

---

## TAHAP 4: Pastikan Akselerator GPU T4 Aktif (GRATIS)

Di jendela Google Colab:
1. Klik menu **Runtime** di bilah atas.
2. Pilih **Change runtime type (Ubah jenis runtime)**.
3. Pada bagian *Hardware accelerator*, pilih **T4 GPU**.
4. Klik tombol **Save (Simpan)**.

---

## TAHAP 5: Jalankan Cell Selangkah Demi Selangkah

Di dalam notebook `Training_Model_Selection_Colab.ipynb`, Anda cukup menekan tombol **Play ($\blacktriangleright$)** pada masing-masing kotak kode (*cell*):

1. **Cell Langkah 1 (`!nvidia-smi`):**  
   Memastikan GPU Tesla T4 (15–16 GB VRAM) telah terhubung.
2. **Cell Langkah 2 (`drive.mount('/content/drive')`):**  
   Menghubungkan akun Google Drive Anda. Klik *Connect to Google Drive* saat jendela pop-up konfirmasi muncul.
3. **Cell Langkah 3 (Ekstraksi Dataset ke SSD Colab):**  
   Mengekstrak `tea_yolo.zip` dan `model_selection.zip` langsung ke disk NVMe Colab. Selesai dalam ~10 detik.
4. **Cell Langkah 4 (Install Dependency):**  
   Memasang pustaka pendukung (`ultralytics`, `faster-coco-eval`, `thop`).

---

## TAHAP 6: Memulai Training 1 Model per Tahap

Setiap model memiliki kotak kode (*cell*) tersendiri, sehingga Anda bebas memilih model mana yang ingin Anda uji terlebih dahulu:

* **Untuk Menjalankan Faster R-CNN (Resolusi 640×640):**  
  Tekan tombol Play pada Cell **`Model 1: Faster R-CNN`**.  
  *Waktu komputasi:* **~30 s.d. 35 menit untuk 50 Epoch penuh!**
* **Untuk Menjalankan RetinaNet:**  
  Tekan tombol Play pada Cell **`Model 2: RetinaNet`**.
* **Untuk Menjalankan SSD:**  
  Tekan tombol Play pada Cell **`Model 3: SSD`**.
* **Untuk Menjalankan RT-DETR:**  
  Tekan tombol Play pada Cell **`Model 4: RT-DETR`**.
* **Untuk Menjalankan YOLO26n Baseline (Pembanding):**  
  Tekan tombol Play pada Cell **`Model 5: YOLO26n Baseline`**.  
  *Waktu komputasi:* **~5 s.d. 8 menit saja!**

---

## TAHAP 7: Mengambil Hasil & Laporan di Google Drive

Setelah cell model selesai dijalankan, skrip otomatis menyalin seluruh hasil ke folder Google Drive Anda:
`Google Drive > SKRIPSI > phase2_runs > <nama_model>/`

Di dalam folder tersebut sudah langsung tersedia:
* **`laporan_hasil_run.md`** $\to$ Laporan komprehensif metrik mAP@50, Precision, Recall, FPS, dan tabel per-kelas.
* **`best_checkpoint.pth`** $\to$ File bobot terbaik hasil training.
* **`learning_curves.png`** $\to$ Grafik kurva loss dan mAP.
* **`confusion_matrix.png`** $\to$ Peta kebingungan deteksi antar kelas.
* **`visual_analysis/`** $\to$ Contoh gambar deteksi benar dan salah pada test set.

Anda tinggal mengunduh folder tersebut ke laptop Anda untuk dimasukkan ke dalam naskah skripsi!
