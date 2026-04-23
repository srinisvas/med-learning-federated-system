import os

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from med_learning_federated_system.task import (
    get_isic_model,
    get_weights,
    load_test_data_for_eval,
    test,
    set_weights,
)
from med_learning_federated_system.state.server_strategy import ISICFedAvgStrategy


def get_evaluate_fn(model: torch.nn.Module, test_loader):
    _state = {"device": None}

    def evaluate_fn(server_round: int, parameters, config):
        if _state["device"] is None:
            _state["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(_state["device"])
        set_weights(model, parameters)
        loss, accuracy = test(model, test_loader, _state["device"])
        return loss, {"mta": accuracy}

    return evaluate_fn


def server_fn(context: Context):
    num_rounds    = int(context.run_config["num-server-rounds"])
    fraction_fit  = float(context.run_config.get("fraction-fit", 0.5))
    num_clients   = int(context.run_config.get("num-clients", 10))
    simulation_id = str(context.run_config.get("simulation-id", "isic-exp"))

    model = get_isic_model()
    pretrained_path = os.environ.get("ISIC_PRETRAINED_PATH", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"[Server] Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    else:
        print("[Server] No pretrained checkpoint found — starting from random init.")

    initial_parameters = ndarrays_to_parameters(get_weights(model))
    test_loader = load_test_data_for_eval(batch_size=32)
    eval_model  = get_isic_model()
    evaluate_fn = get_evaluate_fn(model=eval_model, test_loader=test_loader)

    strategy = ISICFedAvgStrategy(
        simulation_id=simulation_id,
        num_rounds=num_rounds,
        final_eval_model=eval_model,
        final_eval_loader=test_loader,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=max(1, int(num_clients * fraction_fit)),
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        # SGD in task.train() splits this as:
        #   backbone LR = 0.0005 * 0.1 = 0.00005  (very conservative)
        #   head LR     = 0.0005                   (adapts to local data)
        # Lower than the AdamW regime because SGD at equivalent effective
        # LR is more aggressive per step without adaptive dampening.
        on_fit_config_fn=lambda rnd: {
            "current-round": rnd,
            "local-lr": 0.0005,
        },
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)