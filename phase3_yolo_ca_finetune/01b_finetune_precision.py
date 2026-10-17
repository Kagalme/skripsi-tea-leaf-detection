"""
FASE 3 (FINE-TUNING EDITION) - Script 01b: Dedicated Stage-2 Precision Fine-Tuning
==================================================================================
Skrip khusus untuk langsung menjalankan Fine-Tuning Precision di atas `best.pt`
hasil training 50 epoch yang sudah ada (tanpa mengulang training 50 epoch dari awal).
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "01_train_ca.py")

if __name__ == "__main__":
    cmd = [sys.executable, TRAIN_SCRIPT] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))

