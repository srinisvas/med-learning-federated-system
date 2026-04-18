# Med Learning Federated System
## Setup & Execution Guide

Federated learning system for ISIC 2019 skin lesion classification (8-class, 224×224 dermoscopy images) using Flower + PyTorch.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Option A — Windows (WSL)](#option-a--windows-wsl)
- [Option B — HPC](#option-b--hpc)
- [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Common Errors](#common-errors)

---

## Prerequisites

Regardless of environment, you need:

1. **Kaggle account + API token** — dataset is downloaded via kagglehub
   - Go to [kaggle.com](https://kaggle.com) → profile picture → **Settings** → **API** → **Create New Token**
   - This gives you a username and key value — keep them handy

2. **Git access** to this repository

3. **~15 GB free disk space** (9 GB dataset + model checkpoints + environment)

---

## Option A — Windows (WSL)

### 1. Install WSL (if not already done)

Open PowerShell as Administrator and run:
```powershell
wsl --install
```
Restart your machine when prompted. This installs Ubuntu by default. Open the **Ubuntu** app from the Start menu to continue.

### 2. Install Miniconda inside WSL

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init bash
source ~/.bashrc
```

### 3. Clone the repository

```bash
cd ~
git clone <repository-url> med-learning-federated-system
cd med-learning-federated-system
```

### 4. Create and activate the conda environment

```bash
conda create -n fed-learning-env python=3.10 -y
conda activate fed-learning-env
```

### 5. Install dependencies

```bash
pip install -e .
```

### 6. Set up Kaggle credentials

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_TOKEN_VALUE"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

Replace `YOUR_KAGGLE_USERNAME` and `YOUR_TOKEN_VALUE` with your actual values.

### 7. Set dataset path

WSL uses your local filesystem. The dataset will download to kagglehub's default cache. Set the data root:

```bash
# Add to ~/.bashrc so it persists across sessions
echo 'export ISIC_DATA_ROOT="$HOME/.cache/kagglehub/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/versions/1"' >> ~/.bashrc
source ~/.bashrc
```

### 8. Download and set up the dataset

```bash
python data_setup.py
```

This downloads ~9 GB — takes a few minutes. When it finishes you should see:
```
[data] ImageFolder layout detected (8 class dirs, 25331 images).
[data] Setup complete: <path>
```

Verify:
```bash
ls $ISIC_DATA_ROOT
# Expected: AK  BCC  BKL  DF  MEL  NV  SCC  VASC
```

> **WSL GPU note:** If you have an NVIDIA GPU, WSL2 can access it directly — no extra steps needed if your Windows NVIDIA driver is up to date. Run `nvidia-smi` inside WSL to confirm. If no GPU is available, pretraining will run on CPU and take significantly longer (several hours vs ~30 min on GPU).

---

## Option B — HPC

### 1. SSH into the cluster and navigate to your project space

```bash
ssh ssubram7@<cluster-login-hostname>
cd /gpfs/home/s001/ssubram7/projects
```

### 2. Clone the repository

```bash
git clone <repository-url> med-learning-federated-system
cd med-learning-federated-system
```

### 3. Activate the conda environment

The environment is already set up on this cluster:
```bash
conda activate fed-learning-env
```

If starting fresh on a new cluster, create it:
```bash
conda create -n fed-learning-env python=3.10 -y
conda activate fed-learning-env
pip install -e .
```

### 4. Install the package

```bash
pip install -e .
```

### 5. Set up Kaggle credentials

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_TOKEN_VALUE"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

### 6. Request an interactive GPU session

Do not run training on the login node. Request a compute node:
```bash
interact -p gpu -g 1 -c 8 --mem=32G --time=3:00:00
```

Adjust flags to match your cluster's scheduler syntax. Once inside the session:
```bash
conda activate fed-learning-env
cd /gpfs/home/s001/ssubram7/projects/med-learning-federated-system
```

### 7. Set the dataset path

Scratch storage is temporary and gets wiped periodically. The `data_setup.py` script handles re-downloading when needed. Set the path every session:

```bash
export ISIC_DATA_ROOT="/gpfs/home/s001/ssubram7/gscratch/isic2019"
```

To avoid typing this every session, add a shell alias to `~/.bashrc`:
```bash
echo '
alias medfl="cd /gpfs/home/s001/ssubram7/projects/med-learning-federated-system && \
  conda activate fed-learning-env && \
  export ISIC_DATA_ROOT=/gpfs/home/s001/ssubram7/gscratch/isic2019 && \
  echo ISIC_DATA_ROOT=\$ISIC_DATA_ROOT"
' >> ~/.bashrc
source ~/.bashrc
```

Then just run `medfl` at the start of each session.

### 8. Download and set up the dataset

Run this on the login node (has internet access) — not inside the interactive GPU session:
```bash
python data_setup.py
```

> **Scratch is temporary.** If the cluster wipes your scratch between sessions, just re-run `python data_setup.py`. It checks for existing data first and only re-downloads if needed (~2 seconds if data is present, full download if wiped).

---

## Running the System

These steps are the same for both WSL and HPC once setup is complete.

### Step 1 — Verify GPU and data

```bash
nvidia-smi                          # confirm GPU is visible
ls $ISIC_DATA_ROOT                  # confirm: AK BCC BKL DF MEL NV SCC VASC
```

### Step 2 — Centralized pretraining (recommended before FL)

ISIC is an 8-class medical imaging task. Starting FL from a pretrained checkpoint significantly reduces the number of rounds needed for convergence. Skip this only for quick experiments.

```bash
python med_learning_federated_system/pre_train_model.py
```

- Runs 60 epochs with cosine LR schedule
- Saves best checkpoint to `isic_pretrained_bw32.pth` in the project root
- Takes ~20-30 min on an A100, longer on smaller GPUs
- Expected final accuracy: ~75-82% on the held-out test set

### Step 3 — Federated learning

```bash
# With pretrained starting point (recommended)
ISIC_PRETRAINED_PATH=isic_pretrained_bw32.pth flwr run .

# Without pretraining (random init)
flwr run .
```

Per-round metrics are printed to console and saved to `results/<simulation-id>_rounds.csv`.

### Configuring the FL run

Edit `pyproject.toml` to change experiment parameters:

```toml
[tool.flwr.app.config]
num-server-rounds = 50      # total FL rounds
fraction-fit = 0.1          # fraction of clients sampled per round (10 of 100)
local-epochs = 2            # local training epochs per client per round
num-clients = 100           # total number of virtual clients
simulation-id = "isic-exp1" # used for CSV log filename
```

---

## Project Structure

```
med-learning-federated-system/
├── pyproject.toml                          # Flower config + dependencies
├── data_setup.py                           # Dataset download + setup script
├── med_learning_federated_system/
│   ├── task.py                             # Data loading, train/test, model getter
│   ├── client_app.py                       # Flower client (benign FL)
│   ├── server_app.py                       # Flower server
│   ├── pre_train_model.py                  # Centralized pretraining
│   ├── models/
│   │   └── resnet_cnn_model.py             # MedTinyResNet18 (224x224, 8-class)
│   ├── state/
│   │   └── server_strategy.py              # FedAvg with MTA logging
│   └── utils/
│       └── dirichlet_partition.py          # Non-IID data partitioning
└── results/                                # Created at runtime — round CSV logs
```

---

## Common Errors

**`No module named 'med_learning_federated_system'`**
```bash
# Run from the project root
cd /path/to/med-learning-federated-system
pip install -e .
```

**`ISIC data root not found`**
```bash
# Data path not set or scratch was wiped
python data_setup.py
export ISIC_DATA_ROOT="/path/to/gscratch/isic2019"   # HPC
export ISIC_DATA_ROOT="$HOME/.cache/kagglehub/..."   # WSL
```

**`~/.kaggle/kaggle.json not found`**
```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_TOKEN_VALUE"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

**Pretraining running on CPU (very slow)**
```bash
# Confirm CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# If False on HPC: you're on the login node — request a GPU interact session first
# If False on WSL: update your Windows NVIDIA driver
```
