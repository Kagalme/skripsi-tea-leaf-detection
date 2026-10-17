"""
FASE 3 (FINE-TUNING EDITION): YOLO26n + Coordinate Attention (CA) + Precision Fine-Tuning
=========================================================================================
Paket modul Fase 3 dengan tambahan Stage-2 Precision Fine-Tuning:
- Penajaman bobot fungsi rugi klasifikasi (`cls=1.15`, `box=8.5`) untuk menekan False Positive
  dan Wrong Classification antar penyakit daun teh yang mirip secara visual.
- Cosine Annealing Learning Rate rendah (`lr0=5e-4 -> lrf=0.01`) dengan `AdamW` dan
  penonaktifan distorsi Mosaic (`mosaic=0.0`) agar model belajar dari kontur daun utuh.
- Dilengkapi White-Twig Occlusion Healing (v2.7) + Test-Time Augmentation (TTA) 2-View
  pada saat inferensi daun lapangan.
"""

