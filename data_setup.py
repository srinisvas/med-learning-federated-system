"""
load_data_scratch.py — Idempotent ISIC 2019 dataset loader for HPC scratch.

Run this at the top of every SLURM job before training starts.
Fast path (data already on scratch): completes in ~2 seconds.
Slow path (scratch was wiped): downloads ~9 GB and reorganizes.

Usage (standalone):
    python load_data_scratch.py

Usage (inside SLURM script — captures the final data root):
    export ISIC_DATA_ROOT=$(python load_data_scratch.py --path-only)

The --path-only flag suppresses all log output and prints only the
final data root path, making it safe to use with $(...) capture.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all under scratch so compute nodes can access them
# ---------------------------------------------------------------------------

SCRATCH_ROOT   = Path("/gpfs/home/s001/ssubram7/gscratch")
KAGGLE_CACHE   = SCRATCH_ROOT / "kagglehub_cache"
ISIC_DATA_ROOT = SCRATCH_ROOT / "isic2019"

KAGGLE_DATASET = "salviohexia/isic-2019-skin-lesion-images-for-classification"
ISIC_CLASSES   = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MIN_IMAGES_PER_CLASS = 50   # threshold to consider a class dir "complete"

GT_CSV_CANDIDATES = [
    "ISIC_2019_Training_GroundTruth.csv",
    "ISIC_2019_Training_GroundTruth_meta.csv",
    "ground_truth.csv",
    "labels.csv",
]


# ---------------------------------------------------------------------------
# Logging — suppressed when --path-only is set
# ---------------------------------------------------------------------------

_verbose = True

def log(msg: str) -> None:
    if _verbose:
        print(f"[data] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fast existence check
# ---------------------------------------------------------------------------

def data_is_ready(root: Path) -> bool:
    """
    Returns True if all 8 class dirs exist and each has at least
    MIN_IMAGES_PER_CLASS images. Runs in ~1-2 seconds on GPFS.
    """
    if not root.is_dir():
        return False
    for cls in ISIC_CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            return False
        # Count only .jpg files — enough to detect a wiped or incomplete dir
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
        if len(imgs) < MIN_IMAGES_PER_CLASS:
            log(f"Class {cls} looks incomplete ({len(imgs)} images). Rebuilding.")
            return False
    return True


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download() -> Path:
    try:
        import kagglehub
    except ImportError:
        log("kagglehub not found — installing...")
        ret = os.system(f"{sys.executable} -m pip install kagglehub -q")
        if ret != 0:
            sys.exit("[data] ERROR: Failed to install kagglehub. "
                     "Activate your conda env before running this script.")
        import kagglehub

    if not Path("~/.kaggle/kaggle.json").expanduser().exists():
        sys.exit(
            "[data] ERROR: ~/.kaggle/kaggle.json not found.\n"
            "Get your API token from kaggle.com -> Account -> Create New API Token\n"
            "then: mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )

    KAGGLE_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CACHE_DIR"] = str(KAGGLE_CACHE)

    log(f"Downloading {KAGGLE_DATASET} (~9 GB) ...")
    log(f"Cache dir: {KAGGLE_CACHE}")
    raw_path = kagglehub.dataset_download(KAGGLE_DATASET)
    log(f"Download complete: {raw_path}")
    return Path(raw_path)


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def detect_layout(root: Path):
    """
    Returns ('imagefolder', root) or ('flat_csv', (image_dir, csv_path)).
    Exits with an error if the layout is unrecognised.
    """
    # Check for class subdirectories
    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.upper() in ISIC_CLASSES]
    if len(class_dirs) >= 6:
        n_imgs = sum(
            len(list(d.glob("*.jpg")) + list(d.glob("*.JPG"))) for d in class_dirs
        )
        if n_imgs > 1000:
            log(f"ImageFolder layout detected ({len(class_dirs)} class dirs, {n_imgs} images).")
            return "imagefolder", root

    # Search for flat image dir + ground truth CSV
    search_dirs = [root] + [d for d in root.iterdir() if d.is_dir()]
    csv_file = None
    for d in search_dirs:
        for name in GT_CSV_CANDIDATES:
            p = d / name
            if p.is_file():
                csv_file = p
                break
        if csv_file:
            break

    # Find the directory with the most .jpg files
    best_dir, best_count = None, 0
    for d in search_dirs:
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.JPG")))
        if n > best_count:
            best_count, best_dir = n, d

    if csv_file and best_dir and best_count > 100:
        log(f"Flat+CSV layout detected.")
        log(f"  Images : {best_dir}  ({best_count} files)")
        log(f"  CSV    : {csv_file}")
        return "flat_csv", (best_dir, csv_file)

    # Unknown — print directory tree for manual diagnosis
    log("ERROR: Unrecognised layout. Top-level contents:")
    for item in sorted(root.iterdir()):
        tag  = "DIR " if item.is_dir() else "FILE"
        size = f"{len(list(item.iterdir()))} items" if item.is_dir() else f"{item.stat().st_size // 1024} KB"
        log(f"  {tag}  {item.name}  ({size})")
    sys.exit(
        "[data] Cannot determine dataset layout automatically.\n"
        "Please organize images into class subfolders manually:\n"
        f"  {ISIC_DATA_ROOT}/MEL/*.jpg\n"
        f"  {ISIC_DATA_ROOT}/NV/*.jpg  ... etc."
    )


# ---------------------------------------------------------------------------
# Reorganize flat+CSV -> ImageFolder
# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path) -> dict:
    """Parse one-hot ISIC ground-truth CSV. Returns {image_name: class_str}."""
    label_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        class_cols = [h for h in reader.fieldnames if h.strip().upper() in ISIC_CLASSES]
        if not class_cols:
            sys.exit(f"[data] No class columns found in CSV. Headers: {reader.fieldnames}")
        for row in reader:
            img = row.get("image", row.get("image_name", "")).strip()
            if not img:
                continue
            for col in class_cols:
                try:
                    if float(row[col]) == 1.0:
                        label_map[img] = col.strip().upper()
                        break
                except (ValueError, KeyError):
                    continue
    log(f"Parsed {len(label_map)} labels from CSV.")
    return label_map


def reorganize(image_dir: Path, csv_path: Path, output_dir: Path) -> None:
    """
    Copies images from flat `image_dir` into output_dir/<CLASS>/*.jpg.
    Skips files already present (safe to re-run on partial copies).
    Uses shutil.copy2 to preserve metadata.
    """
    label_map = parse_csv(csv_path)
    for cls in ISIC_CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    copied = skipped = missing = 0
    total = len(label_map)

    for img_name, cls in label_map.items():
        # CSV rows may or may not include the extension
        src = None
        for candidate in [f"{img_name}.jpg", f"{img_name}.JPG", img_name]:
            p = image_dir / candidate
            if p.exists():
                src = p
                break

        if src is None:
            missing += 1
            continue

        dst = output_dir / cls / src.name
        if dst.exists():
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1
        if copied % 2000 == 0:
            log(f"  Copied {copied}/{total} ...")

    log(f"Reorganization done: {copied} copied, {skipped} skipped, {missing} missing.")

    log("Class distribution:")
    for cls in ISIC_CLASSES:
        n = len(list((output_dir / cls).glob("*.jpg")))
        log(f"  {cls:<6}: {n:>6} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _verbose

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-only", action="store_true",
        help="Print only the final ISIC_DATA_ROOT path (no log output). "
             "Use with: export ISIC_DATA_ROOT=$(python load_data_scratch.py --path-only)"
    )
    args = parser.parse_args()

    if args.path_only:
        _verbose = False

    # Fast path — data already on scratch
    if data_is_ready(ISIC_DATA_ROOT):
        log(f"Data ready on scratch: {ISIC_DATA_ROOT}")
        print(ISIC_DATA_ROOT)
        return

    # Slow path — download and set up
    log("Data not found on scratch. Starting full setup...")
    raw_path = download()

    layout, payload = detect_layout(raw_path)

    if layout == "imagefolder":
        # Already in the right format — symlink or just point at it directly
        if not ISIC_DATA_ROOT.exists():
            log(f"Creating symlink: {ISIC_DATA_ROOT} -> {payload}")
            ISIC_DATA_ROOT.symlink_to(payload)
        else:
            log(f"Using existing path: {ISIC_DATA_ROOT}")

    elif layout == "flat_csv":
        image_dir, csv_path = payload
        log(f"Reorganizing into: {ISIC_DATA_ROOT}")
        reorganize(image_dir, csv_path, ISIC_DATA_ROOT)

    # Final verification
    if not data_is_ready(ISIC_DATA_ROOT):
        sys.exit(
            f"[data] ERROR: Setup completed but data verification failed at {ISIC_DATA_ROOT}.\n"
            "Check for missing images or CSV parsing errors above."
        )

    log(f"Setup complete: {ISIC_DATA_ROOT}")
    print(ISIC_DATA_ROOT)


if __name__ == "__main__":
    main()
