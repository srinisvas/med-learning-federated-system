"""Server-side strategy: FedAvg with per-round metrics aggregation."""
import flwr as fl
from flwr.common import FitIns


class SaveFedAvgMetricsStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "fedavg",
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.simulation_id      = simulation_id
        self.num_clients        = num_clients
        self.num_rounds         = num_rounds
        self.aggregation_method = aggregation_method
        self.alpha              = alpha

        # History accumulated during the run
        self.history = {"round": [], "accuracy": []}
        self.central_accuracy_history = []

    # ------------------------------------------------------------------
    # configure_fit  — inject per-round config into every client
    # ------------------------------------------------------------------

    def configure_fit(self, server_round: int, parameters, client_manager):
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))

        fit_ins_list = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config["current-round"] = server_round
            fit_ins_list.append((client, FitIns(parameters, config)))

        return fit_ins_list

    # ------------------------------------------------------------------
    # aggregate_evaluate  — collect distributed accuracy, print summary
    # ------------------------------------------------------------------

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        if results:
            acc_vals = [res.metrics.get("accuracy", 0.0) for _, res in results]
            avg_acc  = sum(acc_vals) / len(acc_vals)
        else:
            avg_acc = 0.0

        self.history["round"].append(rnd)
        self.history["accuracy"].append(avg_acc)

        print(f"[Round {rnd}] Distributed accuracy={avg_acc:.4f}  "
              f"(failures={len(failures)})")

        if rnd >= self.num_rounds:
            self._print_summary()

        return aggregated

    # ------------------------------------------------------------------
    # record_centralized_eval  — called externally from evaluate_fn
    # ------------------------------------------------------------------

    def record_centralized_eval(self, rnd: int, loss: float, accuracy: float):
        self.central_accuracy_history.append(accuracy)
        print(f"[Round {rnd}] Centralized accuracy={accuracy:.4f}, loss={loss:.4f}")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _print_summary(self):
        rounds    = self.history["round"]
        accs      = self.history["accuracy"]
        best_rnd  = rounds[accs.index(max(accs))] if accs else "N/A"
        best_acc  = max(accs) if accs else 0.0
        final_acc = accs[-1]  if accs else 0.0

        print("\n" + "=" * 60)
        print(f"  Experiment summary  [{self.simulation_id}]")
        print(f"  Aggregation : {self.aggregation_method}")
        print(f"  Rounds      : {self.num_rounds}")
        print(f"  Clients     : {self.num_clients}")
        print(f"  Alpha       : {self.alpha}")
        print(f"  Best acc    : {best_acc:.4f}  (round {best_rnd})")
        print(f"  Final acc   : {final_acc:.4f}")
        if self.central_accuracy_history:
            print(f"  Central acc : {self.central_accuracy_history[-1]:.4f}")
        print("=" * 60 + "\n")