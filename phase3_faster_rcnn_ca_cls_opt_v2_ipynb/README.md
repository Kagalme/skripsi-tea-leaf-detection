# 🌿 FASE 3 (ClsOpt v2 — Ultra High Accuracy): Faster R-CNN + Dual-Stage Coordinate Attention

Folder ini berisi versi pembanding lanjutan **`ClsOpt v2`** (`FasterRCNN_CA_ClsOpt_v2_Pipeline.ipynb`) yang diselaraskan secara *apple-to-apple* dengan `YOLO26n_CA_ClsOpt_v2_Pipeline.ipynb`, menambahkan **4 Peningkatan Akurasi Lanjutan** di atas `ClsOpt v1` (tanpa menghapus data asli pada dataset):

1. **Resolusi Tinggi `768×768` (`min_size=768, max_size=1024`, +44% Kepadatan Piksel)**: Mempertajam fitur spasial `RoIAlign` saat membedakan lesi mikroskopis *Helopeltis* vs *Green Mirid Bug* serta bercak *Tea algal leaf spot* vs *Brown Blight*.
2. **Dual-Stage Coordinate Attention (`FPN + RoIAlign`, `reduction=16` & `8`, `mip >= 16`)**: Memasang `5× CoordAtt` pada seluruh skala piramida FPN (`0, 1, 2, 3, pool`) **sekaligus** `1× CoordAtt` (`RoIPoolWithCA`, `reduction=8`, `mip=32`) langsung pada fitur `7×7 RoIAlign` sebelum masuk ke kepala klasifikasi `TwoMLPHead`.
3. **ModelEMA (`decay=0.999`) + Object Detection MixUp (`0.08`) + Elevated RoI Classification Gain (`1.30` / `1.40`)**: Menstabilkan bobot evaluasi seperti EMA internal YOLO dan meredam dampak *cross-class label noise* bawaan dataset publik tanpa menghapus sampel asli.
4. **Weighted Box Fusion (WBF) 2-View TTA pada Test Set & 5-View Multi-Scale/Orientation TTA pada Foto Kebun** (dengan *Background Blur* natural 25%: `ksize=15, sigma=2.15`).

