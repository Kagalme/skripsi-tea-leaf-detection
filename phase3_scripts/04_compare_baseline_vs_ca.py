"""
FASE 3 - Script 04: Perbandingan Baseline vs CA
=================================================
Membaca hasil evaluasi Fase 2 (Baseline) dan Fase 3 (CA) secara
berdampingan, lalu mencetak tabel perbandingan lengkap dan menyimpan
laporan Markdown ke laporan/fase3_ca_vs_baseline.md.

Jalankan SETELAH script 01, 02, dan 03 selesai.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

parser = argparse.ArgumentParser(description="FASE 3: Perbandingan Baseline vs CA")
parser.add_argument("--run_name_ca",       default="yolo26n_ca_50ep")
parser.add_argument("--run_name_baseline", default="yolo26n_baseline_50ep")
args = parser.parse_args()

# ─────────────────────────────────────── PATH ──────────────────────────────
if os.path.exists("/content/tea_yolo"):
    PHASE2_RUNS = "/content/phase2_runs"
    PHASE3_RUNS = "/content/phase3_runs"
    LAPORAN_DIR = "/content/laporan"
else:
    PHASE2_RUNS = os.path.join(PROJECT_DIR, "phase2_runs")
    PHASE3_RUNS = os.path.join(PROJECT_DIR, "phase3_runs")
    LAPORAN_DIR = os.path.join(PROJECT_DIR, "laporan")

os.makedirs(LAPORAN_DIR, exist_ok=True)

CLASS_NAMES = [
    "Tea algal leaf spot", "Brown Blight", "Gray Blight",
    "Helopeltis", "Red Spider", "Green Mirid Bug", "Healty Leaf",
]

# ─────────────────────────────────────── LOAD DATA ─────────────────────────
def load_summary(json_path):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return None

def load_training_history(csv_path):
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows

baseline_summary = load_summary(
    os.path.join(PHASE2_RUNS, args.run_name_baseline,
                 "test_eval", "test_results", "summary_metrics.json")
)
ca_summary = load_summary(
    os.path.join(PHASE3_RUNS, args.run_name_ca,
                 "test_eval", "test_results", "summary_metrics.json")
)
baseline_hist = load_training_history(
    os.path.join(PHASE2_RUNS, args.run_name_baseline, "results.csv")
)
ca_hist = load_training_history(
    os.path.join(PHASE3_RUNS, args.run_name_ca, "results.csv")
)
baseline_vstats = load_summary(
    os.path.join(PHASE2_RUNS, args.run_name_baseline, "visual_analysis", "visual_stats.json")
)
ca_vstats = load_summary(
    os.path.join(PHASE3_RUNS, args.run_name_ca, "visual_analysis", "visual_stats.json")
)

# ─────────────────────────────────────── UTILITAS ──────────────────────────
def delta_str(v_ca, v_base):
    """Tampilkan delta: +0.xxxx (green) atau -0.xxxx (red)."""
    d = v_ca - v_base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"

def pct(v):
    return f"{v*100:.2f}%"

# ─────────────────────────────────────── PRINT CONSOLE ─────────────────────
print("=" * 75)
print("  FASE 3 — PERBANDINGAN: YOLO26n BASELINE vs YOLO26n + Coordinate Attention")
print("=" * 75)

if baseline_summary and ca_summary:
    bov = baseline_summary["overall"]
    cov = ca_summary["overall"]

    print(f"\n{'Metrik':<20} {'Baseline':>12} {'CA':>12} {'Delta':>10} {'Impr.':>8}")
    print("-" * 65)
    metrics_list = [
        ("Precision",    "precision"),
        ("Recall",       "recall"),
        ("F1-Score",     "f1"),
        ("mAP@0.5",      "mAP50"),
        ("mAP@0.5:0.95", "mAP50_95"),
    ]
    for label, key in metrics_list:
        b_val = bov.get(key, 0)
        c_val = cov.get(key, 0)
        d     = c_val - b_val
        impr  = f"{'↑' if d >= 0 else '↓'} {abs(d)*100:.2f}pp"
        print(f"  {label:<18} {pct(b_val):>12} {pct(c_val):>12} {delta_str(c_val, b_val):>10} {impr:>8}")
    print("-" * 65)

    print(f"\n  Per-Kelas (mAP@0.5):")
    print(f"  {'Kelas':<25} {'Baseline':>10} {'CA':>10} {'Delta':>10}")
    print("  " + "-" * 57)

    # Buat dict per-kelas
    base_pc = {d["class"]: d for d in baseline_summary.get("per_class", [])}
    ca_pc   = {d["class"]: d for d in ca_summary.get("per_class", [])}
    for cname in CLASS_NAMES:
        bm = base_pc.get(cname, {}).get("map50", 0)
        cm = ca_pc.get(cname, {}).get("map50", 0)
        mark = "↑" if cm >= bm else "↓"
        print(f"  {cname:<25} {pct(bm):>10} {pct(cm):>10} {mark}{abs(cm-bm)*100:>7.2f}pp")

    print("  " + "-" * 57)
    print(f"  {'MEAN':<25} {pct(bov['mAP50']):>10} {pct(cov['mAP50']):>10} "
          f"  {delta_str(cov['mAP50'], bov['mAP50']):>7}")

print(f"\n{'=' * 75}")

# ─────────────────────────────────────── GENERATE LAPORAN MD ───────────────
def _pct(v):
    return f"{v*100:.2f}%"

lines = []
lines.append("# LAPORAN FASE 3: YOLO26n + Coordinate Attention vs Baseline")
lines.append(f"**Penelitian Skripsi**: Deteksi Penyakit, Hama, dan Kondisi Kesehatan Daun Teh")
lines.append(f"**Status**: {'SELESAI' if ca_summary else 'MENUNGGU HASIL CA'}")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## A. Deskripsi Modifikasi (Coordinate Attention)")
lines.append("")
lines.append("Coordinate Attention (CA) — Hou et al., CVPR 2021 — merupakan modul attention")
lines.append("ringan yang menggantikan kompresi channel-only (SE Attention) dengan penangkapan")
lines.append("informasi **posisi spasial** secara horizontal dan vertikal secara terpisah.")
lines.append("")
lines.append("**Cara kerja CA:**")
lines.append("1. **Pooling ganda**: Horizontal (H×1) dan vertikal (1×W) secara paralel.")
lines.append("2. **Shared MLP**: Concatenate → Conv 1×1 → BatchNorm → Hardswish.")
lines.append("3. **Split & Sigmoid**: Hasilkan attention map arah-H dan arah-W.")
lines.append("4. **Re-calibrasi**: Feature map × attn_H × attn_W.")
lines.append("")
lines.append("**Posisi injeksi (3 blok CA):**")
lines.append("")
lines.append("| Posisi | Stride | Feature Size | Target Kelas |")
lines.append("|--------|--------|--------------|--------------|")
lines.append("| P3 (Backbone Stage 3) | 8 | 52×52 (640px) | Tea algal leaf spot, Red Spider (kecil) |")
lines.append("| P4 (Backbone Stage 4) | 16 | 26×26 | Brown Blight, Gray Blight, Helopeltis |")
lines.append("| P5 (Backbone Stage 5) | 32 | 13×13 | Green Mirid Bug, Healty Leaf |")
lines.append("")
lines.append("**Overhead parameter**: +~30K parameter (~1.1% dari 2.57M baseline) → sangat ringan.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## B. Konfigurasi Training (Identik Baseline)")
lines.append("")
lines.append("| Parameter | Nilai |")
lines.append("|-----------|-------|")
lines.append("| Dataset | `tea_yolo/` (Fase 1, leak-free) |")
lines.append("| Epochs | 50 |")
lines.append("| Image size | 640×640 (Google Colab GPU T4) |")
lines.append("| Batch size | 16 (GPU) |")
lines.append("| Pretrained | `yolo26n.pt` (COCO) |")
lines.append("| Optimizer | auto (identik baseline) |")
lines.append("| Augmentation | Default Ultralytics recipe (identik baseline) |")
lines.append("| Seed | 42 (reproducible) |")
lines.append("| Perbedaan dari baseline | **Hanya 3 blok CoordAttYOLO di backbone** |")
lines.append("")
lines.append("---")
lines.append("")

if baseline_summary and ca_summary:
    bov = baseline_summary["overall"]
    cov = ca_summary["overall"]

    lines.append("## C. Perbandingan Hasil — Test Set (529 gambar)")
    lines.append("")
    lines.append("### Overall Metrics")
    lines.append("")
    lines.append("| Metrik | Baseline | CA | Delta | Interpretasi |")
    lines.append("|--------|----------|----|-------|--------------|")
    for label, key in [("Precision", "precision"), ("Recall", "recall"),
                        ("F1-Score", "f1"), ("mAP@0.5", "mAP50"), ("mAP@0.5:0.95", "mAP50_95")]:
        bv = bov.get(key, 0); cv = cov.get(key, 0)
        d  = cv - bv
        icon = "✅" if d >= 0 else "⚠️"
        dstr = f"{'+' if d>=0 else ''}{d*100:.2f}pp"
        lines.append(f"| **{label}** | **{_pct(bv)}** | **{_pct(cv)}** | **{dstr}** | {icon} |")
    lines.append("")

    # Per-kelas
    lines.append("### Per-Class Metrics (mAP@0.5)")
    lines.append("")
    lines.append("| Kelas | Baseline P | CA P | Δ P | Baseline R | CA R | Δ R | Baseline mAP50 | CA mAP50 | Δ mAP50 |")
    lines.append("|-------|-----------|------|-----|-----------|------|-----|----------------|----------|---------|")
    base_pc = {d["class"]: d for d in baseline_summary.get("per_class", [])}
    ca_pc   = {d["class"]: d for d in ca_summary.get("per_class", [])}
    for cname in CLASS_NAMES:
        b = base_pc.get(cname, {}); c = ca_pc.get(cname, {})
        bp, cp = b.get("precision",0), c.get("precision",0)
        br, cr = b.get("recall",0),    c.get("recall",0)
        bm, cm = b.get("map50",0),     c.get("map50",0)
        lines.append(
            f"| {cname} | {_pct(bp)} | {_pct(cp)} | {'+' if cp>=bp else ''}{(cp-bp)*100:.2f}pp "
            f"| {_pct(br)} | {_pct(cr)} | {'+' if cr>=br else ''}{(cr-br)*100:.2f}pp "
            f"| {_pct(bm)} | {_pct(cm)} | {'+' if cm>=bm else ''}{(cm-bm)*100:.2f}pp |"
        )
    lines.append(
        f"| **MEAN** | **{_pct(bov['precision'])}** | **{_pct(cov['precision'])}** | "
        f"**{'+' if cov['precision']>=bov['precision'] else ''}{(cov['precision']-bov['precision'])*100:.2f}pp** "
        f"| **{_pct(bov['recall'])}** | **{_pct(cov['recall'])}** | "
        f"**{'+' if cov['recall']>=bov['recall'] else ''}{(cov['recall']-bov['recall'])*100:.2f}pp** "
        f"| **{_pct(bov['mAP50'])}** | **{_pct(cov['mAP50'])}** | "
        f"**{'+' if cov['mAP50']>=bov['mAP50'] else ''}{(cov['mAP50']-bov['mAP50'])*100:.2f}pp** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

if baseline_vstats and ca_vstats:
    bs = baseline_vstats["stats"]
    cs = ca_vstats["stats"]
    btot = sum(bs.values()); ctot = sum(cs.values())
    lines.append("## D. Analisis Kesalahan Visual (TP, FP, WC, FN)")
    lines.append("")
    lines.append("| Kategori | Baseline | CA | Delta |")
    lines.append("|----------|----------|----|-------|")
    for cat in ["TP", "FP", "WC", "FN"]:
        bv = bs.get(cat, 0); cv = cs.get(cat, 0)
        d  = cv - bv
        lines.append(f"| {cat} | {bv} | {cv} | {'+' if d>=0 else ''}{d} |")
    lines.append("")
    lines.append("---")
    lines.append("")

lines.append("## E. Analisis & Kesimpulan")
lines.append("")
lines.append("### Temuan Utama")
lines.append("")
if baseline_summary and ca_summary:
    bov = baseline_summary["overall"]; cov = ca_summary["overall"]
    d_map50 = cov["mAP50"] - bov["mAP50"]
    d_recall = cov["recall"] - bov["recall"]
    icon_map = "meningkat" if d_map50 >= 0 else "menurun"
    icon_rec  = "meningkat" if d_recall >= 0 else "menurun"
    lines.append(f"1. **mAP@0.5** {icon_map} sebesar **{abs(d_map50)*100:.2f} poin persentase** "
                 f"({_pct(bov['mAP50'])} → **{_pct(cov['mAP50'])}**).")
    lines.append(f"2. **Recall rata-rata** {icon_rec} sebesar **{abs(d_recall)*100:.2f} poin persentase** "
                 f"({_pct(bov['recall'])} → **{_pct(cov['recall'])}**) — "
                 "recall adalah metrik kritis untuk mengurangi False Negative pada deteksi penyakit.")
    # Highlight kelas lemah
    base_pc = {d["class"]: d for d in baseline_summary.get("per_class", [])}
    ca_pc   = {d["class"]: d for d in ca_summary.get("per_class", [])}
    kelas_lemah = ["Tea algal leaf spot", "Brown Blight", "Helopeltis"]
    lines.append(f"3. **Kelas lemah baseline** ({', '.join(kelas_lemah)}):")
    for kl in kelas_lemah:
        br = base_pc.get(kl, {}).get("recall", 0)
        cr = ca_pc.get(kl, {}).get("recall", 0)
        bm = base_pc.get(kl, {}).get("map50", 0)
        cm = ca_pc.get(kl, {}).get("map50", 0)
        lines.append(f"   - **{kl}**: Recall {_pct(br)} → {_pct(cr)} "
                     f"({'+' if cr>=br else ''}{(cr-br)*100:.2f}pp), "
                     f"mAP50 {_pct(bm)} → {_pct(cm)} ({'+' if cm>=bm else ''}{(cm-bm)*100:.2f}pp)")
lines.append("")
lines.append("### Interpretasi Akademis")
lines.append("")
lines.append("Penambahan Coordinate Attention di P3/P4/P5 backbone memungkinkan model untuk:")
lines.append("- Menangkap konteks spasial lokal secara eksplisit (sumbu-H & sumbu-W),")
lines.append("  sehingga bercak kecil (*small lesions*) lebih mudah dibedakan dari latar daun.")
lines.append("- Mengurangi ambiguitas antara kelas yang memiliki gejala visual serupa")
lines.append("  (Brown Blight vs Gray Blight) dengan perhatian posisional yang lebih presisi.")
lines.append("- Overhead parameter minimal (+~1%) tidak memengaruhi kecepatan inferensi secara signifikan.")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## F. File Output")
lines.append("")
lines.append("| File | Lokasi |")
lines.append("|------|--------|")
lines.append(f"| Best weights CA | `phase3_runs/{args.run_name_ca}/weights/best.pt` |")
lines.append(f"| Training CSV | `phase3_runs/{args.run_name_ca}/results.csv` |")
lines.append(f"| Test metrics | `phase3_runs/{args.run_name_ca}/test_eval/test_results/` |")
lines.append(f"| Visual analysis | `phase3_runs/{args.run_name_ca}/visual_analysis/` |")
lines.append(f"| Laporan ini | `laporan/fase3_ca_vs_baseline.md` |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("*Laporan dibuat otomatis oleh `phase3_scripts/04_compare_baseline_vs_ca.py`*")

report_text = "\n".join(lines)
report_path = os.path.join(LAPORAN_DIR, "fase3_ca_vs_baseline.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n  Laporan Markdown disimpan ke:\n  {report_path}")
print(f"\n{'=' * 75}")
print(f"  Fase 3 selesai! Silakan unduh laporan dan model ke lokal.")
print(f"{'=' * 75}")
