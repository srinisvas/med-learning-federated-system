import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Scalar, Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy


LOG_DIR = os.environ.get("FL_LOG_DIR", "results")


class ISICFedAvgStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        simulation_id: str = "isic-exp",
        num_rounds: int = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.simulation_id = simulation_id
        self.num_rounds = num_rounds

        self._central_mta_history: List[float] = []
        self._dist_mta_history:    List[float] = []
        self._loss_history:        List[float] = []

        # Buffer for centralized result — written to CSV in aggregate_evaluate
        self._pending_central: Optional[Tuple[int, float, float]] = None  # (round, loss, mta)

        os.makedirs(LOG_DIR, exist_ok=True)
        self._csv_path = os.path.join(LOG_DIR, f"{simulation_id}_rounds.csv")
        self._init_csv()

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "central_mta", "dist_mta", "central_loss"])

    def _write_row(
        self,
        rnd: int,
        central_mta: float,
        dist_mta: float,
        central_loss: float,
    ) -> None:
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [rnd, f"{central_mta:.4f}", f"{dist_mta:.4f}", f"{central_loss:.4f}"]
            )

    # ------------------------------------------------------------------
    # evaluate — centralized eval; buffer result, don't write CSV yet
    # ------------------------------------------------------------------

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        result = super().evaluate(server_round, parameters)
        if result is None:
            return result

        loss, metrics = result
        central_mta = float(metrics.get("mta", 0.0))

        self._central_mta_history.append(central_mta)
        self._loss_history.append(loss)

        # Buffer — CSV row written once aggregate_evaluate gives us dist_mta
        self._pending_central = (server_round, loss, central_mta)

        print(f"[Round {server_round}] Centralized — loss={loss:.4f}, MTA={central_mta:.4f}")
        return loss, metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate — distributed eval; write complete CSV row here
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)

        if results:
            total_examples = sum(r.num_examples for _, r in results)
            dist_mta = sum(
                r.metrics.get("mta", 0.0) * r.num_examples for _, r in results
            ) / max(total_examples, 1)
        else:
            dist_mta = 0.0

        self._dist_mta_history.append(dist_mta)
        print(f"[Round {server_round}] Distributed MTA: {dist_mta:.4f}")

        # Write complete row now that both metrics are from the same round
        if self._pending_central is not None:
            pending_rnd, pending_loss, pending_mta = self._pending_central
            self._write_row(pending_rnd, pending_mta, dist_mta, pending_loss)
            self._pending_central = None

        if server_round == self.num_rounds:
            self._print_summary()

        return agg_loss, {"dist_mta": dist_mta}

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"Experiment: {self.simulation_id}")
        print(f"Rounds completed: {self.num_rounds}")
        if self._central_mta_history:
            print(f"Final centralized MTA : {self._central_mta_history[-1]:.4f}")
            print(f"Peak centralized MTA  : {max(self._central_mta_history):.4f}")
        if self._dist_mta_history:
            print(f"Final distributed MTA : {self._dist_mta_history[-1]:.4f}")
        print(f"Round log saved to    : {self._csv_path}")
        print("=" * 60 + "\n")