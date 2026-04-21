import os
import shutil
import numpy as np
from pathlib import Path

SRC  = Path("/gpfs/home/s001/ssubram7/gscratch/isic2019")
DST  = Path("/dev/shm/isic2019/")
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
SAMPLE_RATIO = 1.0
SEED = 42

rng = np.random.default_rng(SEED)

total = 0
for cls in CLASSES:
    src_dir = SRC / cls
    dst_dir = DST / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_dir.glob("*.jpg"))
    n = max(1, int(len(images) * SAMPLE_RATIO))
    selected = rng.choice(images, size=n, replace=False)

    for img in selected:
        shutil.copy2(img, dst_dir / img.name)

    total += n
    print(f"{cls:<6}: {n:>4} / {len(images)} copied")

print(f"\nTotal: {total} images at {DST}")
