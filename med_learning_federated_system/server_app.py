"""Flower server for ISIC 2019 federated learning."""
import os

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from med_learning_federated_system.state.server_strategy import SaveFedAvgMetricsStrategy
from med_learning_federated_system.task import (
    get_resnet_cnn_model, get_weights, load_test_data_for_eval,
)
from med_learning_federated_system.utils.evaluate import get_evaluate_fn


def server_fn(context: Context):
    # ------------------------------------------------------------------
    # Read run config
    # ------------------------------------------------------------------
    num_rounds         = int(context.run_config["num-server-rounds"])
    num_clients        = int(context.run_config["num-clients"])
    fraction_fit       = float(context.run_config.get("fraction-fit", 0.1))
    simulation_id      = context.run_config.get("simulation-id", "isic-fedavg")
    aggregation_method = context.run_config.get("aggregation-method", "fedavg").lower()
    alpha              = float(context.run_config.get("alpha", 0.5))

    # ------------------------------------------------------------------
    # Initial model parameters
    # ------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = get_resnet_cnn_model()
    model.to(device)

    # Optionally load a pretrained checkpoint to warm-start the global model
    pretrained_path = context.run_config.get("pretrained-checkpoint", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Loading pretrained global model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

    parameters = ndarrays_to_parameters(get_weights(model))

    # ------------------------------------------------------------------
    # Server-side centralised evaluation data
    # ------------------------------------------------------------------
    test_data = load_test_data_for_eval(batch_size=64)

    # ------------------------------------------------------------------
    # per-round fit config
    # ------------------------------------------------------------------
    def on_fit_config_fn(server_round: int) -> dict:
        return {
            "current-round": server_round,
            "alpha":         alpha,
            # Clients may read these to tune local training
            "lr":            0.005,
            "epochs":        2,
        }

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    eval_model = get_resnet_cnn_model().to(device)

    strategy = SaveFedAvgMetricsStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=int(num_clients * fraction_fit),
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(
            model=eval_model,
            test_data=test_data,
            device=device,
        ),
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config_fn,
        simulation_id=simulation_id,
        num_clients=num_clients,
        num_rounds=num_rounds,
        aggregation_method=aggregation_method,
        alpha=alpha,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)