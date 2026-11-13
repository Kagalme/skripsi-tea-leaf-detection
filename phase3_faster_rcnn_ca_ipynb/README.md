# 🌿 FASE 3: Faster R-CNN + Coordinate Attention (CA) Pipeline Notebook (.ipynb)

Folder ini berisi versi **Notebook All-in-One Mandiri** berbasis **Faster R-CNN-ResNet50-FPN + Coordinate Attention (CA) + Edge Contour** yang dirancang khusus untuk dijalankan langsung di **Google Colab** (sebagai pasangan komparasi langsung terhadap `phase3_yolo_ca_ipynb/YOLO26n_CA_Pipeline.ipynb`).

## 📁 Berkas di dalam Folder
1. **`FasterRCNN_CA_Pipeline.ipynb`**: Notebook Jupyter utama berisi alur lengkap:
   - **Bagian 1**: Inisialisasi Environment, GPU Check, Mount Google Drive, dan Ekstraksi Dataset `tea_yolo.zip`.
   - **Bagian 2**: Definisi Arsitektur `Faster R-CNN-ResNet50-FPN` + `BackboneFPNWithCA` (`CoordAtt` pada level FPN `P2, P3, P4, P5, pool`).
   - **Bagian 3**: Dataset Loader `TeaLeafFasterRCNNDataset`, Evaluator Metrik COCO/VOC (`mAP@0.50` & `mAP@0.50:0.95`), dan Inisialisasi Bobot Pretrained COCO (`faster_rcnn_ca_coco_init.pth`).
   - **Bagian 4**: Pelatihan Model (*Training* 50 Epoch dengan PyTorch AMP Mixed Precision atau *Fine-Tuning* Data Lapangan).
   - **Bagian 5**: Evaluasi Lengkap pada Held-Out Test Set (529 Citra, 640×640) + Tabel Metrik Per-Kelas.
   - **Bagian 6**: Visualisasi Kurva Konvergensi HD (Loss, Precision, Recall, mAP50, mAP50-95, F1) & Confusion Matrix + PR Curve.
   - **Bagian 7**: Engine Segmentasi Daun Kebun (`TeaLeafFieldDetector` v2.7): Multi-Daun, *Background Blur (Bokeh)*, Penolakan Ranting/Pakis/Tanah, *White-Twig Healing*, dan 2-View TTA berbasis Faster R-CNN + CA.
   - **Bagian 8**: Uji Coba Interaktif (*Upload* Foto Daun Kebun dari Laptop/HP) dengan Tampilan 4-Panel.
   - **Bagian 9**: Cadangkan Bobot Terbaik (`best.pth`) dan Hasil Evaluasi Otomatis ke Google Drive.
   - **Bagian 10–12**: Pra-Label Otomatis Data Lapangan, Penggabungan Dataset (`roboflow_lapangan.zip` / pra-label), *Fine-Tuning*, dan Evaluasi Perbandingan Kiri-Kanan (*Data Berlabel* vs *Prediksi Faster R-CNN + CA*).

## 🚀 Keunggulan Versi Ini
- **Komparasi Apple-to-Apple dengan YOLO26n + CA**: Menggunakan dataset `tea_yolo.zip`, pembagian train/val/test, ukuran input `640×640`, modul `CoordAtt`, dan engine segmentasi kebun `TeaLeafFieldDetector` yang identik.
- ** Cepat di GPU Colab (AMP Mixed Precision)**: Menggunakan `torch.amp.autocast('cuda')` + `GradScaler` sehingga pelatihan Faster R-CNN 3–4× lebih cepat di GPU T4 Colab.
- **Fleksibel terhadap Penambahan Kelas (`Ranting`)**: Fungsi `load_flexible_weights()` otomatis menyesuaikan bobot `FastRCNNPredictor` apabila kelas bertambah dari 7 menjadi 8 saat *fine-tuning* hasil koreksi Roboflow.

