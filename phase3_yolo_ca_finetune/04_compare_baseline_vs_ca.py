"""
FASE 3 (FINE-TUNING EDITION) - Script 04: Perbandingan Baseline (640x640) vs CA Fine-Tuned (640x640)
====================================================================================================
Membandingkan metrik YOLO26n + CA setelah Precision Fine-Tuning (`yolo26n_ca_finetune`)
terhadap YOLO26n Baseline (640x640) serta Stage-1 (`yolo26n_ca_50ep` jika tersedia).
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

parser = argparse.ArgumentParser(description="FASE 3 (Fine-Tune): Perbandingan Baseline vs CA Fine-Tuned")
parser.add_argument("--run_name_ca", default="yolo26n_ca_finetune")
parser.add_argument("--run_name_stage1", default="yolo26n_ca_50ep")
parser.add_argument("--run_name_baseline", default="yolo26n")
args = parser.parse_args()

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
    "Tea algal leaf spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red Spider",
    "Green Mirid Bug",
    "Healty Leaf",
]

BUILTIN_BASELINE_640 = {
    "model": "YOLO26n Baseline (640x640, 50 Epochs)",
    "overall": {
        "precision": 0.9233714142366357,
        "recall": 0.8936392687496827,
        "f1": 0.9082620796761603,
        "mAP50": 0.9476120921179382,
        "mAP50_95": 0.9040892583328416,
    },
    "per_class": [
        {"class": "Tea algal leaf spot", "precision": 0.835601, "recall": 0.833333, "f1": 0.834466, "map50": 0.920041, "map50_95": 0.871423},
        {"class": "Brown Blight",        "precision": 0.873459, "recall": 0.812168, "f1": 0.841699, "map50": 0.884365, "map50_95": 0.831357},
        {"class": "Gray Blight",         "precision": 0.899563, "recall": 0.871287, "f1": 0.885200, "map50": 0.947416, "map50_95": 0.887355},
        {"class": "Helopeltis",          "precision": 0.981664, "recall": 0.877738, "f1": 0.926797, "map50": 0.958135, "map50_95": 0.949357},
        {"class": "Red Spider",          "precision": 0.940943, "recall": 0.919254, "f1": 0.929972, "map50": 0.958876, "map50_95": 0.917189},
        {"class": "Green Mirid Bug",     "precision": 0.953704, "recall": 0.965654, "f1": 0.959642, "map50": 0.980961, "map50_95": 0.938799},
        {"class": "Healty Leaf",         "precision": 0.978666, "recall": 0.976040, "f1": 0.977351, "map50": 0.983491, "map50_95": 0.933146},
    ],
}


def normalize_summary_dict(raw: dict) -> dict:
    if not raw:
        return None
    ov = raw.get("overall", raw)
    prec = float(ov.get("precision", 0.0))
    rec = float(ov.get("recall", 0.0))
    f1 = float(ov.get("f1", 2 * prec * rec / (prec + rec + 1e-9)))
    map50 = float(ov.get("mAP50", ov.get("map50", 0.0)))
    map50_95 = float(ov.get("mAP50_95", ov.get("map50_95", 0.0)))

    norm_pc = []
    for item in raw.get("per_class", []):
        cname = item.get("class") or item.get("class_name")
        p = float(item.get("precision", 0.0))
        r = float(item.get("recall", 0.0))
        cf1 = float(item.get("f1", 2 * p * r / (p + r + 1e-9)))
        m50 = float(item.get("map50", item.get("mAP50", 0.0)))
        m95 = float(item.get("map50_95", item.get("mAP50_95", 0.0)))
        norm_pc.append({
            "class": cname,
            "precision": p,
            "recall": r,
            "f1": cf1,
            "map50": m50,
            "map50_95": m95,
        })

    return {
        "overall": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mAP50": map50,
            "mAP50_95": map50_95,
        },
        "per_class": norm_pc,
    }


def load_baseline_summary() -> dict:
    candidates = [
        os.path.join(PHASE2_RUNS, args.run_name_baseline, "test_evaluation_summary.json"),
        os.path.join(PHASE2_RUNS, args.run_name_baseline, "test_eval", "test_results", "summary_metrics.json"),
        os.path.join(PHASE2_RUNS, "yolo26n", "test_evaluation_summary.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return normalize_summary_dict(json.load(f))
    return normalize_summary_dict(BUILTIN_BASELINE_640)


def load_ca_summary(rname: str) -> dict:
    ca_path = os.path.join(PHASE3_RUNS, rname, "test_eval", "test_results", "summary_metrics.json")
    if os.path.exists(ca_path):
        with open(ca_path, "r", encoding="utf-8") as f:
            return normalize_summary_dict(json.load(f))
    return None


baseline_summary = load_baseline_summary()
ca_summary = load_ca_summary(args.run_name_ca) or load_ca_summary(args.run_name_stage1)


def pct(v):
    return f"{v*100:.2f}%"


def delta_str(v_ca, v_base):
    d = v_ca - v_base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


print("=" * 78)
print("  FASE 3 (FINE-TUNING EDITION) — BASELINE vs YOLO26n + CA (FINE-TUNED)")
print("=" * 78)

if baseline_summary and ca_summary:
    bov = baseline_summary["overall"]
    cov = ca_summary["overall"]

    print(f"\n{'Metrik':<20} {'Baseline (640)':>15} {'CA Fine-Tuned':>17} {'Delta':>10} {'Perubahan':>10}")
    print("-" * 76)
    for label, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("mAP@0.5", "mAP50"),
        ("mAP@0.5:0.95", "mAP50_95"),
    ]:
        bv = bov[key]
        cv = cov[key]
        d = cv - bv
        impr = f"{'↑' if d >= 0 else '↓'} {abs(d)*100:.2f}pp"
        print(f"  {label:<18} {pct(bv):>15} {pct(cv):>17} {delta_str(cv, bv):>10} {impr:>10}")
    print("-" * 76)

    base_pc = {d["class"]: d for d in baseline_summary["per_class"]}
    ca_pc = {d["class"]: d for d in ca_summary["per_class"]}

    print(f"\n  Perbandingan Per-Kelas (Precision & mAP@0.5):")
    print(f"  {'Kelas':<24} {'Base P':>9} {'CA-FT P':>9} {'Δ P':>9} | {'Base mAP50':>10} {'CA-FT mAP50':>11} {'Δ mAP50':>9}")
    print("  " + "-" * 88)
    for cname in CLASS_NAMES:
        b = base_pc.get(cname, {})
        c = ca_pc.get(cname, {})
        bp, cp = b.get("precision", 0.0), c.get("precision", 0.0)
        bm, cm = b.get("map50", 0.0), c.get("map50", 0.0)
        dp = (cp - bp) * 100
        dm = (cm - bm) * 100
        print(
            f"  {cname:<24} {pct(bp):>9} {pct(cp):>9} {'+' if dp>=0 else ''}{dp:>7.2f}pp | "
            f"{pct(bm):>10} {pct(cm):>11} {'+' if dm>=0 else ''}{dm:>7.2f}pp"
        )
else:
    print("[WARNING] Hasil evaluasi CA Fine-Tuned belum tersedia. Jalankan 02_evaluate_testset.py terlebih dahulu.")

lines = [
    "# LAPORAN FASE 3 (FINE-TUNING EDITION): YOLO26n + CA (Precision Fine-Tuned) vs Baseline (640×640)",
    "**Penelitian Skripsi**: Deteksi Penyakit, Hama, dan Kondisi Kesehatan Daun Teh",
    "**Arsitektur**: YOLO26n Asli + 3× `CoordAttYOLO` (P3, P4, P5) + Stage-2 Precision Fine-Tuning",
    "**Strategi Fine-Tuning**: `AdamW (lr0=5e-4, cos_lr=True)`, `cls=1.15`, `box=8.5`, `mosaic=0.0`, `erasing=0.15`, `freeze=4`",
    "",
    "---",
    "",
]

if baseline_summary and ca_summary:
    bov = baseline_summary["overall"]
    cov = ca_summary["overall"]
    lines.append("## A. Tabel Komparasi Metrik Keseluruhan (Held-Out Test Set: 529 Gambar)")
    lines.append("")
    lines.append("| Metrik | YOLO26n Baseline (640×640) | YOLO26n + CA Fine-Tuned (640×640) | Delta (pp) |")
    lines.append("|---|---|---|---|")
    for label, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("mAP@0.5", "mAP50"),
        ("mAP@0.5:0.95", "mAP50_95"),
    ]:
        bv = bov[key]
        cv = cov[key]
        d = (cv - bv) * 100
        lines.append(f"| **{label}** | **{pct(bv)}** | **{pct(cv)}** | **{'+' if d>=0 else ''}{d:.2f}pp** |")
    lines.append("")
    lines.append("## B. Tabel Komparasi Per-Kelas (Test Set)")
    lines.append("")
    lines.append("| Kelas | Base P | CA-FT P | Δ P | Base R | CA-FT R | Base mAP50 | CA-FT mAP50 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    base_pc = {d["class"]: d for d in baseline_summary["per_class"]}
    ca_pc = {d["class"]: d for d in ca_summary["per_class"]}
    for cname in CLASS_NAMES:
        b = base_pc.get(cname, {})
        c = ca_pc.get(cname, {})
        bp, cp = b.get("precision", 0.0), c.get("precision", 0.0)
        br, cr = b.get("recall", 0.0), c.get("recall", 0.0)
        bm, cm = b.get("map50", 0.0), c.get("map50", 0.0)
        dp = (cp - bp) * 100
        lines.append(
            f"| {cname} | {pct(bp)} | {pct(cp)} | {'+' if dp>=0 else ''}{dp:.2f}pp | "
            f"{pct(br)} | {pct(cr)} | {pct(bm)} | {pct(cm)} |"
        )
    lines.append("")

report_path = os.path.join(LAPORAN_DIR, "fase3_ca_finetune_vs_baseline.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n[SUKSES] Laporan komparasi disimpan di: {report_path}")
print("=" * 78)

