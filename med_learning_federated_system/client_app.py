"""Flower client for ISIC 2019 federated learning — no attack logic."""
import random
from collections import OrderedDict

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from med_learning_federated_system.task import (
    get_weights, load_data, set_weights, test, train, get_resnet_cnn_model,
)



class FlowerClient(NumPyClient):
    def __init__(self, net, local_epochs: int, context: Context):
        self.net           = net
        self.local_epochs  = local_epochs
        self.context       = context
        self.partition_id  = int(context.node_config["partition-id"])
        self.num_partitions = int(context.node_config["num-partitions"])
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.training_set = None
        self.test_set     = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        # Lazy-load partition data on first call
        if self.training_set is None:
            alpha_val = float(config.get("alpha", 0.5))
            self.training_set, _ = load_data(
                self.partition_id, self.num_partitions, alpha_val=alpha_val
            )

        # Allow the server to optionally vary LR and epochs per round
        lr     = float(config.get("lr",     random.choice([0.003, 0.004, 0.005])))
        epochs = int(config.get("epochs",   random.choice([1, 2, 3])))

        current_round = config.get("current-round", "N/A")
        print(
            f"[Client {self.partition_id}] Round {current_round} | "
            f"epochs={epochs}, lr={lr:.4f}"
        )

        train_loss, _ = train(
            self.net, self.training_set, epochs, self.device, lr
        )

        return (
            get_weights(self.net),
            len(self.training_set.dataset),
            {"train_loss": train_loss},
        )

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        if self.test_set is None:
            alpha_val = float(config.get("alpha", 0.5))
            _, self.test_set = load_data(
                self.partition_id, self.num_partitions, alpha_val=alpha_val
            )

        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)

        print(
            f"[Client {self.partition_id}] eval: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}"
        )
        return loss, len(self.test_set.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# client_fn + ClientApp
# ---------------------------------------------------------------------------

def client_fn(context: Context):
    net          = get_resnet_cnn_model()
    local_epochs = int(context.run_config["local-epochs"])
    return FlowerClient(net, local_epochs, context).to_client()


app = ClientApp(client_fn=client_fn)