"""Centralised evaluation function passed to the Flower strategy."""
from typing import Optional
import torch
from flwr.common import NDArrays, Scalar
from med_learning_federated_system.task import test, set_weights

def get_evaluate_fn(model, test_data, device):
    """
    Return a closure that Flower calls at the end of every round for
    centralised (server-side) evaluation.

    The strategy passes this to `evaluate_fn` in FedAvg.  Flower calls it as:
        loss, metrics = evaluate_fn(server_round, parameters, config)
    """

    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        set_weights(model, parameters)
        model.to(device)
        loss, accuracy = test(model, test_data, device)
        print(
            f"[Server eval | Round {server_round}] "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}"
        )
        return loss, {"accuracy": accuracy}

    return evaluate
