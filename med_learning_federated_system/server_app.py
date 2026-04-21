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
    """
    Lazy CUDA init — defers model.to(device) until the first evaluate_fn call.

    Flower runs server_fn in a background thread. If we call model.to(cuda)
    eagerly here, it races with Ray's worker pool initialization and hits
    "CUDA device busy or unavailable" in exclusive GPU mode. Deferring to
    the first actual call guarantees Ray has fully initialized before any
    CUDA context is created in the server thread.
    """
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

    # ---- Model initialisation ----
    model = get_isic_model()
    pretrained_path = os.environ.get("ISIC_PRETRAINED_PATH", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"[Server] Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    else:
        print("[Server] No pretrained checkpoint found — starting from random init.")

    initial_parameters = ndarrays_to_parameters(get_weights(model))

    # ---- Global test set for centralized evaluation ----
    # num_workers=2 and pin_memory=True are safe here — server runs in the
    # main process and has legitimate GPU access after Ray has initialized.
    test_loader = load_test_data_for_eval(batch_size=32)
    evaluate_fn = get_evaluate_fn(
        model=get_isic_model(),  # separate instance from the one above
        test_loader=test_loader,
    )

    # ---- Strategy ----
    strategy = ISICFedAvgStrategy(
        simulation_id=simulation_id,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=max(1, int(num_clients * fraction_fit)),
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"current-round": rnd},
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)