# 🌿 FASE 3 (Classification-Optimized): YOLO26n + Coordinate Attention (CA) Pipeline

Folder ini berisi notebook **`YOLO26n_CA_ClsOpt_Pipeline.ipynb`** yang mengimplementasikan **3 Strategi Optimasi Akurasi Klasifikasi Kondisi Daun** (tanpa menghapus sampel *cross-class* asli pada dataset):

1. **Strategi 1 — Targeted Minority Oversampling (`train` split only)**:
   - Menyeimbangkan kelas minoritas (`Tea algal leaf spot`, `Brown Blight`, `Helopeltis`, `Red Spider`) menjadi minimal 650 citra pada split `train`.
   - Split `val` (1.048 citra) dan `test` (529 citra) **100% tidak disentuh** (*zero data leakage*).
2. **Strategi 2 — Color-Preserving & Multi-Orientation Classification Tuning**:
   - `hsv_h=0.004` (mengunci warna asli bercak penyakit), `hsv_s=0.35`, `hsv_v=0.35`, `degrees=30.0`, `flipud=0.5`, `fliplr=0.5`, `cls=1.25`, dan `close_mosaic=15`.
3. **Strategi 3 — Pure Model-Driven 4-View TTA + Shadow CLAHE**:
   - Menghapus aturan warna kaku yang memblokir prediksi model, diganti dengan normalisasi bayangan lembut (`LAB-CLAHE`) dan konsensus **4-View TTA** (*Original, Flip-H, Flip-V, Rotate-90°*).

