# SKRIPSI — Laporan Penelitian
## Deteksi Penyakit, Hama, dan Kondisi Kesehatan Daun Teh
### Menggunakan YOLO26n Baseline vs YOLO26n + Coordinate Attention

---

## Daftar Laporan

| Fase | File | Status | Highlight |
|------|------|--------|-----------|
| **Fase 1** | [fase1_dataset_preparation.md](fase1_dataset_preparation.md) | ✅ Selesai | Dataset 5.278 citra, 7 kelas, 100% valid |
| **Fase 2** | [fase2_baseline_yolo26n.md](fase2_baseline_yolo26n.md) | ✅ Selesai | mAP@0.5 = 94.69%, F1 = 90.81% |
| **Fase 3** | `fase3_coordinate_attention.md` | ⏳ Belum dimulai | YOLO26n + Coordinate Attention |

---

## Ringkasan Cepat per Fase

### Fase 1 — Persiapan Dataset
- **Input**: 5.278 citra klasifikasi daun teh (7 kelas)
- **Output**: Dataset YOLO object detection (`tea_yolo/`)
- **Split**: Train 3.701 / Val 1.048 / Test 529 (leak-free stratified)
- **Kualitas**: 0 label error, 0 data leakage, 100% bounding box valid

### Fase 2 — Baseline Model YOLO26n
- **Model**: YOLO26n (2.57M params, 6.2 GFLOPs)
- **Training**: 50 epoch, CPU, ~6 jam, default Ultralytics recipe
- **Hasil Test Set**:

| Metric | Nilai |
|--------|-------|
| mAP@0.5 | **94.69%** |
| mAP@0.5:0.95 | **90.70%** |
| F1-Score | **90.81%** |
| Precision | 93.87% |
| Recall | 87.94% |

### Fase 3 — YOLO26n + Coordinate Attention *(belum dimulai)*
- Modifikasi arsitektur: menambahkan Coordinate Attention pada neck/backbone
- Target: meningkatkan Recall kelas lemah (Brown Blight, Tea algal leaf spot, Helopeltis)

---

## Struktur Folder Proyek

```
D:\SKRIPSI\
├── laporan\                        ← Folder ini (semua laporan .md)
│   ├── README.md                   ← Index ini
│   ├── fase1_dataset_preparation.md
│   ├── fase2_baseline_yolo26n.md
│   └── fase3_coordinate_attention.md (belum ada)
│
├── tea_yolo\                       ← Dataset YOLO (output Fase 1)
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
│
├── phase1_scripts\                 ← Script pipeline Fase 1
├── phase2_scripts\                 ← Script pipeline Fase 2
├── phase2_runs\                    ← Hasil training Fase 2
│   └── yolo26n_baseline_50ep\
│       ├── weights\best.pt         ← Model terbaik baseline
│       ├── results.csv
│       ├── test_eval\
│       └── visual_analysis\
│
└── yolo26n.pt                      ← Pretrained weights COCO
```

---

*Terakhir diperbarui: 2026-10-02*
