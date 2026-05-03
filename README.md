# Medical Federated Learning System

This project trains a skin lesion classification model using Federated Learning with Flower.

## Project Structure

| File / Folder | Purpose |
|---|---|
| `models/resnet_cnn_model.py` | Defines the main image classification model. |
| `task.py` | Handles data loading, transforms, training, testing, and model weights. |
| `client_app.py` | Flower client. Each client trains on its own data split. |
| `server_app.py` | Flower server. Starts the FL run and manages aggregation. |
| `state/server_strategy.py` | FedAvg strategy with logging, metrics, and plots. |
| `pre_train_model.py` | Pre-trains the model`. |
| `central_train_model.py` | Runs centralized training for comparison. |
| `setup_data.py` / `data_setup.py` | Scripts for preparing the dataset. |
| `data_export.py` | Helper script to export data or results. |
| `results/` | Stores logs, metrics, and plots after training. |
| `pyproject.toml` | Flower app settings and project dependencies. |

## Dataset Format

Keep the dataset in this format:

```text
isic2019/
├── AK/
├── BCC/
├── BKL/
├── DF/
├── MEL/
├── NV/
├── SCC/
└── VASC/