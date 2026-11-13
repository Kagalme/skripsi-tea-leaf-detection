# 🌿 FASE 3 (ClsOpt v2 — Ultra High Accuracy): YOLO26n + Dual-Stage Coordinate Attention

Folder ini berisi versi pembanding lanjutan **`ClsOpt v2`** (`YOLO26n_CA_ClsOpt_v2_Pipeline.ipynb`) yang menambahkan **4 Peningkatan Akurasi Lanjutan** di atas `ClsOpt v1`:

1. **Resolusi Tinggi `768×768` (+44% Kepadatan Piksel)**: Mempertajam pembacaan lesi mikroskopis tusukan *Helopeltis* vs *Green Mirid Bug* serta bercak *Tea algal leaf spot* vs *Brown Blight*.
2. **Dual-Stage Coordinate Attention (`6x CoordAtt`, `reduction=16`, `mip >= 16`)**: Memasang `CoordAtt` di keluaran **Backbone (`P3, P4, P5`)** sekaligus di keluaran **Neck PAN-FPN (`Head-P3, Head-P4, Head-P5`)** sebelum `Detect`, dengan 100% pemetaan bobot COCO (`24 -> 30 layer`).
3. **Anti-Label-Noise Regularization (`mixup=0.08` + `cls=1.30`)**: Meredam dampak *cross-class label noise* bawaan dataset publik tanpa menghapus data asli.
4. **Weighted Box Fusion (WBF) 2-View TTA pada Test Set & 5-View TTA pada Foto Kebun** (dengan *Background Blur* natural 25%: `ksize=15, sigma=2.15`).

