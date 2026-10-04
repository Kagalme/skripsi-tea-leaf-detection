"""
FASE 2 - Script 01: Environment Check
======================================
Verifikasi semua komponen yang diperlukan untuk training YOLO26n.
"""

import sys
import platform
import os
import time

print("=" * 60)
print("  FASE 2 - Environment Check")
print("=" * 60)

print(f"\n[1] System Info")
print(f"    OS       : {platform.system()} {platform.release()}")
print(f"    Machine  : {platform.machine()}")
print(f"    Python   : {sys.version.split()[0]}")

# CPU info
try:
    import multiprocessing
    print(f"    CPU cores: {multiprocessing.cpu_count()}")
except:
    pass

print(f"\n[2] PyTorch")
try:
    import torch
    print(f"    Version  : {torch.__version__}")
    print(f"    CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU      : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"    VRAM     : {props.total_memory / 1024**3:.2f} GB")
    else:
        print(f"    Device   : CPU only (training akan lambat)")
    
    # Test tensor operation
    t = torch.randn(3, 3)
    _ = t @ t
    print(f"    Tensor op: OK")
except ImportError:
    print("    PyTorch not found. Install: pip install torch torchvision")
    sys.exit(1)

print(f"\n[3] Ultralytics / YOLO")
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"    Version  : {ultralytics.__version__}")
    
    # Try loading YOLO26n
    model_path = "yolo26n.pt"
    if os.path.exists(model_path):
        print(f"    Model    : {model_path} ditemukan")
        model = YOLO(model_path)
        model.info()
        print(f"    Info     : loaded OK")
    else:
        print(f"    Model    : {model_path} tidak ditemukan, akan di-download saat training")
        try:
            model = YOLO("yolo26n.pt")
            model.info()
            print(f"    Hub load : OK")
        except Exception as e:
            print(f"    Hub load : {e}")
except ImportError:
    print("    Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

print(f"\n[4] Dataset")
data_yaml = "tea_yolo/data.yaml"
if os.path.exists(data_yaml):
    print(f"    data.yaml: ditemukan")
    with open(data_yaml) as f:
        content = f.read()
    for l in content.strip().splitlines():
        print(f"    {l}")
else:
    print(f"    {data_yaml} tidak ditemukan!")

# Count images per split
for split in ["train", "val", "test"]:
    img_dir = f"tea_yolo/images/{split}"
    if os.path.exists(img_dir):
        n = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"    {split:5s}    : {n} images")

print(f"\n[5] Disk Space")
import shutil
total, used, free = shutil.disk_usage("d:/")
print(f"    Total: {total / 1024**3:.1f} GB")
print(f"    Used : {used / 1024**3:.1f} GB")
print(f"    Free : {free / 1024**3:.1f} GB")

print("\n" + "=" * 60)
print("  Environment check selesai.")
print("=" * 60)
