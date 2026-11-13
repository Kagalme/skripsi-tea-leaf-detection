# YOLO26n + Coordinate Attention (CA) + YOLO26n-seg — Revised Research Pipeline (v3 Audit-Compliant Edition)

Folder ini berisi versi revisi menyeluruh dari notebook penelitian deteksi penyakit daun teh (`YOLO26n_CA_ClsOpt_Pipeline.ipynb`) beserta berkas konfigurasi arsitektur pendukungnya:
- [YOLO26n_CA_ClsOpt_Pipeline.ipynb](YOLO26n_CA_ClsOpt_Pipeline.ipynb)
- [yolo26n_ca.yaml](yolo26n_ca.yaml)
- [yolo26n_seg_leaf.yaml](yolo26n_seg_leaf.yaml)

---

## 1. Perbaikan Temuan Audit Teknis & Metodologi (Poin A – H)

| Kode | Cell Terkait | Temuan Audit | Perbaikan yang Diterapkan |
| :---: | :--- | :--- | :--- |
| **A** | **Cell 2, 14, 16, 22, 34** | Kelas penyakit yang salah dikecualikan (`if cid == 3: continue`, padahal `3 = Helopeltis` dan `6 = Healty Leaf`). | Mendefinisikan konstanta eksplisit `HEALTHY_LEAF_CLASS_ID = 6` dan `DISEASE_LESION_CLASS_IDS = (0, 1, 2, 3, 4, 5)` di Cell 2. Kelas `3` (`Helopeltis`) **tidak pernah dikecualikan** dan selalu diproses sebagai penyakit/hama bercak pada `detect_lesions()` (Cell 16), pseudo-labeling (Cell 22), serta evaluasi pasangan Ground Truth (Cell 34). Kelas `6` (`Healty Leaf`) disaring dari pendeteksian bercak melalui `exclude_healthy_leaf_class=True`. |
| **B** | **Cell 2, 24, 28** | Pemanggilan `compute_md5(img_p)` tidak sesuai dengan definisi `compute_file_md5(filepath)` (`NameError`). | Mengubah seluruh pemanggilan di Cell 24 dan Cell 28 menjadi `compute_file_md5()` serta menyediakan alias `compute_md5 = compute_file_md5` di Cell 2. |
| **C** | **Cell 2, 24, 28, 30** | Penggunaan `yaml.dump(...)` tanpa `import yaml`. | Menambahkan `pyyaml` pada instalasi dependensi serta menambahkan `import yaml` secara eksplisit di Cell 2, Cell 24, Cell 28, dan Cell 30. |
| **D** | **Cell 16, 18, 22, 28, 30, 32, 34** | Urutan segmentasi bermasalah (memuat `yolo26n-seg.pt` umum sebelum model khusus daun teh dilatih di Cell 30; Cell 22 memakai `analyze_field_image()` sehingga bercak hilang bila segmentasi daun gagal). | 1. Cell 16 tidak lagi memuat `yolo26n-seg.pt` COCO umum sebagai *fallback*; bobot segmentasi hanya dimuat dari `LEAF_SEG_WEIGHTS` hasil training khusus daun teh di Cell 30 melalui `field_detector.load_trained_leaf_seg()`.<br>2. Cell 22 memanggil `detector.detect_lesions()` secara **independen** dari segmentasi daun untuk menghasilkan kandidat pseudo-label bercak.<br>3. Menegakkan urutan eksekusi agar Cell 28 & Cell 30 dijalankan terlebih dahulu sebelum pengujian sistem terintegrasi di Cell 18, Cell 32, dan Cell 34. |
| **E** | **Cell 22, 24** | Verifikasi manual belum sepenuhnya terjamin (status disetujui diberikan hanya karena file `.txt` valid tanpa bukti pemeriksaan manusia, dan gambar 0 prediksi dilewati). | 1. Gambar `field_train` tanpa prediksi (`0` bbox) tetap dimasukkan ke `01_candidates_for_review` (`priority_band = "ZERO_PRED_MUST_CHECK_MISSED_LESIONS"`) agar anotator memeriksa kemungkinan bercak yang terlewat (*false negative*).<br>2. Fungsi `save_and_verify_manually_corrected_annotations()` di Cell 22 dan filter di Cell 24 mewajibkan bukti eksplisit review manusia pada manifest (`annotator_id` terisi, `review_decision` bernilai `"APPROVED_POSITIVE"` atau `"APPROVED_NEGATIVE_NO_LESION"`, serta pencatatan `human_action` dari perbandingan hash MD5 draft vs hasil koreksi). |
| **F** | **Cell 2, 24, 28** | Pembacaan ekspor Roboflow mencari `train/labels/images/nama.txt` (seharusnya `train/labels/nama.txt`) dan berpotensi menduplikasi foto. | Menambahkan fungsi `find_matching_yolo_label_file(img_p)` yang mendukung struktur standar Roboflow (`train/labels/nama.txt`), YOLO standar (`labels/train/nama.txt`), dan satu folder, serta `match_image_to_field_manifest(img_p, field_manifest)` yang memetakan setiap foto ke nama kanonik `entry["filename"]` (`dst_stem`) di `field_500_split_manifest.json` tanpa membuat duplikat foto. |
| **G** | **Cell 28, 30** | `val_verified = (stats["val"] > 0)` belum membuktikan seluruh gambar dan seluruh instance daun target telah diperiksa manual. | Menggunakan daftar periksa verifikasi manual per gambar (`leaf_seg_manual_verification_checklist.csv`) yang mencatat `annotator_id`, `all_target_leaves_annotated=True`, `includes_hard_or_overlapping_cases`, dan `review_decision="APPROVED_COMPLETE_POLYGONS"`. Status `val_verified` dan `test_verified` hanya bernilai `True` apabila **100% citra** pada split `val` dan `test` di `field_500_split_manifest.json` hadir dan telah disetujui manual (`missing_or_unverified == 0`), sehingga kasus sulit/bertumpuk tidak dapat diabaikan. |
| **H** | **Cell 34** | Ground Truth pasangan daun–bercak dibentuk otomatis dari overlap (`best_lcr >= 0.30`) yang rawan keliru pada daun bertumpuk. | Menambahkan manifest verifikasi manual pasangan Ground Truth `(GT_Leaf, GT_Lesion)` pada test set (`gt_leaf_lesion_pairs_test_verified.csv`) yang menandai kasus daun bertumpuk (`is_overlapping_leaf_case=True`). Perhitungan akurasi asosiasi (*Pairwise Precision, Recall, F1*) di Cell 34 wajib menggunakan `verified_gt_leaf_id` yang telah disetujui manual (`manual_verified=True` dan `annotator_id` terisi). |

---

## 2. Urutan Eksekusi Notebook yang Direkomendasikan

1. **Tahap 1 — Persiapan Dataset, Arsitektur CA, & Training Detektor Bercak**: Jalankan **Cell 2 → Cell 4 → Cell 6 → Cell 8 → Cell 10 → Cell 12 → Cell 14**.
2. **Tahap 2 — Inisialisasi Detektor, Pseudo-Labeling Independen, & Fine-Tuning Terverifikasi**: Jalankan **Cell 16 → Cell 22 → Cell 24 → Cell 26**.
3. **Tahap 3 — Persiapan Dataset Segmentasi Daun & Training `YOLO26n-seg`**: Jalankan **Cell 28 → Cell 30** (memuat bobot `LEAF_SEG_WEIGHTS` ke `field_detector`).
4. **Tahap 4 — Pengujian & Evaluasi Sistem Terintegrasi + Backup**: Jalankan **Cell 18 → Cell 32 → Cell 34 → Cell 20**.
