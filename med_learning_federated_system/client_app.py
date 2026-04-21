"""
client_app.py — Pure benign Flower client for ISIC 2019 federated learning.

Clients use GPU via fractional allocation (num-gpus=0.2 in pyproject.toml).
Ray assigns each actor 20% of the GPU, allowing 5 clients to run concurrently
on a single A100 without CUDA context conflicts.
"""

import random
from collections import OrderedDict

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from med_learning_federated_system.task import (
    get_isic_model,
    get_weights,
    set_weights,
    load_data,
    test,
    train,
)


class ISICFlowerClient(NumPyClient):

    def __init__(self, net: torch.nn.Module, local_epochs: int, context: Context):
        self.net = net
        self.local_epochs = local_epochs
        self.context = context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.partition_id  = int(context.node_config["partition-id"])
        self.num_partitions = int(context.node_config["num-partitions"])

        self._train_loader = None
        self._val_loader   = None

    def _ensure_data_loaded(self) -> None:
        if self._train_loader is None:
            self._train_loader, self._val_loader = load_data(
                partition_id=self.partition_id,
                num_partitions=self.num_partitions,
            )

    def get_properties(self, config):
        return {"partition_id": str(self.partition_id)}

    def fit(self, parameters, config):
        self._ensure_data_loaded()
        set_weights(self.net, parameters)

        lr     = float(config.get("local-lr", random.choice([0.005, 0.008, 0.01])))
        epochs = int(config.get("local-epochs", self.local_epochs))

        train_loss, _ = train(
            net=self.net,
            train_loader=self._train_loader,
            epochs=epochs,
            device=self.device,
            lr=lr,
        )

        return get_weights(self.net), len(self._train_loader.dataset), {
            "train_loss": float(train_loss),
            "local_epochs": epochs,
            "local_lr": lr,
        }

    def evaluate(self, parameters, config):
        self._ensure_data_loaded()
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self._val_loader, self.device)
        print(f"[Client {self.partition_id}] eval — loss={loss:.4f}, acc={accuracy:.4f}")
        return float(loss), len(self._val_loader.dataset), {"mta": float(accuracy)}


def client_fn(context: Context):
    net          = get_isic_model()
    local_epochs = int(context.run_config.get("local-epochs", 2))
    return ISICFlowerClient(net, local_epochs, context).to_client()


app = ClientApp(client_fn=client_fn)