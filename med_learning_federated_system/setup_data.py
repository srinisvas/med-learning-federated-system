import os
import sys
import shutil
import pathlib

SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR / "data" / "isic2019"
DATASET_SLUG = "salviohexia/isic-2019-skin-lesion-images-for-classification"

# Expected top-level contents after download
REQUIRED_FILES = [
    "ISIC_2019_Training_GroundTruth.csv",
    "ISIC_2019_Training_Input",
]


def _check_already_present() -> bool:
    return all((DATA_DIR / f).exists() for f in REQUIRED_FILES)


def _symlink_or_copy(src: pathlib.Path, dst: pathlib.Path):

    if dst.exists() or dst.is_symlink():
        return  # already linked / copied
    try:
        dst.symlink_to(src)
        print(f"  Linked  {src.name} -> {dst}")
    except (OSError, NotImplementedError):
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  Copied  {src.name} -> {dst}")


def main():
    if _check_already_present():
        print(
            f"Dataset already present at {DATA_DIR}\n"
            "Nothing to do.  Delete data/isic2019/ and re-run to force a fresh download."
        )
        return

    try:
        import kagglehub
    except ImportError:
        print(
            "ERROR: kagglehub is not installed.\n"
            "Install it with:  pip install kagglehub",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading dataset: {DATASET_SLUG}")
    print("This may take a while on the first run (~10 GB).\n")

    try:
        download_path = pathlib.Path(
            kagglehub.dataset_download(DATASET_SLUG)
        )
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        print(
            "\nMake sure your Kaggle credentials are set up:\n"
            "  export KAGGLE_USERNAME=<your_username>\n"
            "  export KAGGLE_KEY=<your_api_key>\n"
            "or create ~/.kaggle/kaggle.json with your credentials.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloaded to: {download_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nPopulating {DATA_DIR} ...")

    for item in download_path.iterdir():
        _symlink_or_copy(item, DATA_DIR / item.name)

    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print(
            f"\nWARNING: Expected files not found after setup: {missing}\n"
            "The directory structure inside the Kaggle archive may have changed.\n"
            f"Inspect {DATA_DIR} and update REQUIRED_FILES / DATA_DIR paths in\n"
            "setup_data.py and task.py accordingly.",
            file=sys.stderr,
        )
    else:
        print(
            f"\nDataset ready at {DATA_DIR}\n"
            "You can now run the federated experiment."
        )

if __name__ == "__main__":
    main()