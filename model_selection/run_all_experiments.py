"""
FASE 2: MODEL SELECTION EXPERIMENT - STEP-BY-STEP ORCHESTRATOR
===============================================================
Script orkestrator terkontrol untuk menjalankan eksperimen 1 MODEL PER TAHAP:
1. RetinaNet (ResNet-50-FPN)
2. SSD (SSD300-VGG16)
3. Faster R-CNN (ResNet-50-FPN)
4. RT-DETR (RT-DETR-L)
5. YOLO26n Baseline

FITUR KONTROL PENGGUNA:
- TIDAK OTOMATIS BERGANTIAN: Satu model dijalankan hingga selesai (Train -> Eval -> Visual).
- Setelah 1 model selesai, skrip BERHENTI (PAUSE) dan menunggu konfirmasi eksplisit dari pengguna
  sebelum berpindah ke model berikutnya.
- Dilengkapi MENU INTERAKTIF: Pengguna dapat memilih model mana yang ingin dirun sekarang.
- Menyimpan status progres (Experiment Status Tracker) di phase2_runs/model_selection/experiment_status.json.
- Mode aman: default tidak mengeksekusi tanpa persetujuan (--dry-run aktif secara default).
"""

import os
import sys
import argparse
import subprocess
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_selection.config import (
    BASE_DIR,
    RUNS_DIR,
    CANDIDATE_MODELS,
    DEFAULT_IMGSZ,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_DEVICE,
    get_model_dir
)
from model_selection.models import MODEL_REGISTRY

STATUS_FILE = os.path.join(RUNS_DIR, "experiment_status.json")


def load_experiment_status():
    """Memuat riwayat status penyelesaian eksperimen per model."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {k: {"status": "BELUM", "completed_at": None} for k in CANDIDATE_MODELS}


def save_experiment_status(status_dict):
    """Menyimpan riwayat status penyelesaian eksperimen."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status_dict, f, indent=4)


def run_command(cmd, dry_run=False):
    """Menjalankan perintah terminal atau menampilkan perintah jika mode dry-run."""
    print(f"\n[EXEC] {cmd}")
    if dry_run:
        print("  -> Status: SKIPPED (Dry-Run Mode aktif - perintah tidak dieksekusi)")
        return 0
    else:
        ret = subprocess.call(cmd, shell=True)
        return ret


def execute_single_model_pipeline(model_key, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                                 imgsz=DEFAULT_IMGSZ, device="cpu", dry_run=False):
    """
    Menjalankan alur lengkap 1 model:
    1. Training baseline (dengan validasi per epoch)
    2. Evaluasi pada Test Set (529 gambar)
    3. Analisis visual prediksi (TP, FP, FN, Misklasifikasi, Lokalisasi)
    """
    if model_key not in MODEL_REGISTRY:
        print(f"[ERROR] Model '{model_key}' tidak ditemukan di registri!")
        return False

    spec = MODEL_REGISTRY[model_key]
    print("\n" + "=" * 80)
    print(f"  MEMULAI PIPELINE MODEL: {spec['name'].upper()}")
    print(f"  Landasan Paper: {spec['paper']}")
    print(f"  Paradigma     : {spec['paradigm']}")
    print(f"  Backbone      : {spec['backbone']}")
    print("=" * 80)

    # 1. Training Command
    train_cmd = (
        f"python -m model_selection.train_detector "
        f"--model {model_key} "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--imgsz {imgsz} "
        f"--device {device} "
        f"--seed {DEFAULT_SEED}"
    )
    ret_train = run_command(train_cmd, dry_run=dry_run)
    if ret_train != 0 and not dry_run:
        print(f"[ERROR] Proses training {model_key} gagal atau dibatalkan!")
        return False

    # 2. Evaluation Command (pada Test Set setelah training selesai)
    eval_cmd = (
        f"python -m model_selection.evaluate_detector "
        f"--model {model_key} "
        f"--imgsz {imgsz} "
        f"--device {device}"
    )
    ret_eval = run_command(eval_cmd, dry_run=dry_run)
    if ret_eval != 0 and not dry_run:
        print(f"[ERROR] Proses evaluasi test set {model_key} gagal!")
        return False

    # 3. Visual Analysis Command
    vis_cmd = (
        f"python -m model_selection.visualize_predictions "
        f"--model {model_key} "
        f"--imgsz {imgsz} "
        f"--device {device}"
    )
    ret_vis = run_command(vis_cmd, dry_run=dry_run)
    if ret_vis != 0 and not dry_run:
        print(f"[ERROR] Proses analisis visual {model_key} gagal!")
        return False

    if not dry_run:
        status_dict = load_experiment_status()
        status_dict[model_key] = {
            "status": "SELESAI",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_experiment_status(status_dict)

    model_dir = get_model_dir(model_key)
    print("\n" + "-" * 80)
    print(f"  [SUKSES] Seluruh tahapan untuk model '{spec['name']}' telah tuntas!")
    print(f"  Hasil log, checkpoint, dan visualisasi tersimpan di direktori khusus:")
    print(f"  -> Direktori Model : {model_dir}")
    print(f"  -> Laporan Run      : {os.path.join(model_dir, 'laporan_hasil_run.md')}")
    print("-" * 80)
    return True


def interactive_menu(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, imgsz=DEFAULT_IMGSZ, device="cpu", dry_run=False):
    """
    Menu interaktif terminal untuk memilih dan menjalankan model satu demi satu.
    """
    while True:
        status_dict = load_experiment_status()

        print("\n" + "=" * 80)
        print("  FASE 2: MODEL SELECTION - KONTROL EKSPERIMEN (1 MODEL PER TAHAP)")
        print("=" * 80)
        print(f"Status Eksekusi  : {'DRY RUN (Preview)' if dry_run else 'ACTIVE EXECUTION'}")
        print(f"Target Epochs    : {epochs} | Batch: {batch_size} | Res: {imgsz}x{imgsz} | Dev: {device}\n")
        print("Pilih model yang ingin dijalankan sekarang:")

        model_list = [
            ("1", "retinanet", "RetinaNet (ResNet-50-FPN)"),
            ("2", "ssd", "SSD (SSD300-VGG16)"),
            ("3", "faster_rcnn", "Faster R-CNN (ResNet-50-FPN)"),
            ("4", "rtdetr", "RT-DETR (RT-DETR-L)"),
            ("5", "yolo26n", "YOLO26n Baseline"),
        ]

        for num, key, label in model_list:
            curr_status = status_dict.get(key, {}).get("status", "BELUM")
            badge = "[SUDAH SELESAI]" if curr_status == "SELESAI" else "[BELUM DIUJI]"
            print(f"  [{num}] {label:<32} {badge}")

        print("\n  [A] Jalankan Antrean Terpandu (Konfirmasi manual tiap selesai 1 model)")
        print("  [Q] Keluar / Selesai\n")

        pilihan = input("Masukkan pilihan Anda [1-5 / A / Q]: ").strip().upper()

        if pilihan == "Q":
            print("\n[INFO] Keluar dari kontrol eksperimen.")
            break

        model_map = {"1": "retinanet", "2": "ssd", "3": "faster_rcnn", "4": "rtdetr", "5": "yolo26n"}

        if pilihan in model_map:
            chosen_key = model_map[pilihan]
            spec = MODEL_REGISTRY[chosen_key]
            print(f"\nAnda memilih: {spec['name']}")
            konfirm = input(f"Jalankan eksperimen model ini sekarang? (y/n) [y]: ").strip().lower()
            if konfirm in ["", "y", "ya", "yes"]:
                execute_single_model_pipeline(chosen_key, epochs=epochs, batch_size=batch_size,
                                              imgsz=imgsz, device=device, dry_run=dry_run)
                # PAUSE eksplisit setelah selesai
                input("\n>> Model selesai! Tekan ENTER untuk kembali ke menu dan meninjau model lain... ")
            else:
                print(f"[INFO] Eksekusi {spec['name']} dibatalkan.")

        elif pilihan == "A":
            print("\n[MODE ANTREAN TERPANDU]")
            print("Setiap model akan dijalankan satu per satu. Setelah 1 model selesai,")
            print("sistem akan BERHENTI dan meminta konfirmasi Anda sebelum lanjut ke model berikutnya.\n")
            
            for num, key, label in model_list:
                curr_status = status_dict.get(key, {}).get("status", "BELUM")
                if curr_status == "SELESAI":
                    print(f"[-] Model '{label}' sudah selesai sebelumnya. Lewati? (y/n) [y]: ", end="")
                    skip = input().strip().lower()
                    if skip in ["", "y", "ya", "yes"]:
                        continue

                lanjut = input(f"\n>> Siap menjalankan model [{label}]? (y/n/q) [y]: ").strip().lower()
                if lanjut == "q":
                    print("[INFO] Antrean dihentikan oleh pengguna.")
                    break
                if lanjut not in ["", "y", "ya", "yes"]:
                    print(f"[INFO] Melewati model {label}.")
                    continue

                # Eksekusi hanya 1 model ini
                execute_single_model_pipeline(key, epochs=epochs, batch_size=batch_size,
                                              imgsz=imgsz, device=device, dry_run=dry_run)

                # PAUSE MUTLAK: Jangan langsung lanjut!
                print(f"\n[PAUSE] Model '{label}' telah selesai 100%.")
                tanya_lanjut = input("Apakah Anda ingin lanjut ke model berikutnya sekarang? (y/N) [n]: ").strip().lower()
                if tanya_lanjut not in ["y", "ya", "yes"]:
                    print("[INFO] Eksekusi dihentikan. Anda dapat memeriksa hasil model ini terlebih dahulu.")
                    break
        else:
            print("[WARNING] Pilihan tidak valid, silakan coba lagi.")


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator - Step-by-Step Model Selection Controller")
    parser.add_argument("--model", type=str, choices=CANDIDATE_MODELS, default=None,
                        help="Jalankan 1 model spesifik secara langsung (contoh: --model retinanet)")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Buka menu interaktif satu per satu")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Menampilkan alur eksekusi tanpa menjalankan training (Default: True)")
    parser.add_argument("--run-now", action="store_true", default=False,
                        help="Kunci otorisasi untuk memulai eksekusi sesungguhnya")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    is_dry_run = args.dry_run and not args.run_now

    # Kasus 1: Pengguna menentukan 1 model spesifik via CLI
    if args.model is not None:
        print("=" * 80)
        print(f"  FASE 2: EKSEKUSI MODEL TUNGGAL: {args.model.upper()}")
        print(f"  Mode Operasi: {'DRY RUN (Preview)' if is_dry_run else 'ACTIVE EXECUTION'}")
        print("=" * 80)
        execute_single_model_pipeline(
            args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            dry_run=is_dry_run
        )
        print(f"\n[SELESAI] Eksekusi model {args.model} selesai.")
        return

    # Kasus 2: Mode default (atau --interactive) -> Menu Interaktif Terpandu
    interactive_menu(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        dry_run=is_dry_run
    )


if __name__ == "__main__":
    main()
