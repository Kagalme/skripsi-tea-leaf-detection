# 🌿 FASE 3 (Classification-Optimized): Faster R-CNN + Coordinate Attention (CA) Pipeline

Folder ini berisi notebook **`FasterRCNN_CA_ClsOpt_Pipeline.ipynb`** yang mengimplementasikan **3 Strategi Optimasi Akurasi Klasifikasi Kondisi Daun** secara setara (*apple-to-apple*) dengan `YOLO26n_CA_ClsOpt_Pipeline.ipynb` (tanpa menghapus sampel *cross-class* asli pada dataset):

1. **Strategi 1 — Targeted Minority Oversampling (`train` split only)**:
   - Menyeimbangkan kelas minoritas (`Tea algal leaf spot`, `Brown Blight`, `Helopeltis`, `Red Spider`) menjadi minimal 650 citra pada split `train` (`seed=42`, sampel identik dengan YOLO26n).
   - Split `val` (1.048 citra) dan `test` (529 citra) **100% tidak disentuh** (*zero data leakage*).
2. **Strategi 2 — Color-Preserving & Multi-Orientation Classification Tuning**:
   - *ColorJitter* dengan `hue=0.004` (mengunci warna asli bercak penyakit), `brightness=0.30, contrast=0.25, saturation=0.35`, transformasi kotak tanpa kehilangan presisi (*Flip-H, Flip-V, Rotasi 90°*), serta bobot *RoI Classification Loss* `1.25x`.
3. **Strategi 3 — Pure Model-Driven 4-View TTA + Shadow CLAHE**:
   - Menghapus aturan warna kaku yang memblokir prediksi model, diganti dengan normalisasi bayangan lembut (`LAB-CLAHE`) dan konsensus **4-View TTA** (*Original, Flip-H, Flip-V, Rotate-90°*) menggunakan Faster R-CNN + CA.

