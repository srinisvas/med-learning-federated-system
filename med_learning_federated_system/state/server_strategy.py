import csv
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Scalar, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required on HPC (no display)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize


LOG_DIR = os.environ.get("FL_LOG_DIR", "results")

ISIC_CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]


class ISICFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg with:
      - Per-round CSV logging (central_mta, dist_mta, central_loss)
      - Full sklearn metrics suite at the end of the final round:
        accuracy, weighted precision/recall/F1, confusion matrix,
        per-class ROC curves (AUC), precision-recall curves
      - All plots saved as PNG to LOG_DIR — no display required
    """

    def __init__(
        self,
        simulation_id: str = "isic-exp",
        num_rounds: int = 50,
        final_eval_model: Optional[nn.Module] = None,
        final_eval_loader: Optional[DataLoader] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.simulation_id = simulation_id
        self.num_rounds    = num_rounds

        # Model and loader for final-round full metrics.
        # These are the same instances used by evaluate_fn in server_app.py —
        # by the time _run_final_evaluation() is called, the model already has
        # the final aggregated weights loaded (set by evaluate_fn this round).
        self._final_eval_model  = final_eval_model
        self._final_eval_loader = final_eval_loader

        # _central_mta_history includes round 0 (init eval) + rounds 1..N  -> length N+1
        # _dist_mta_history includes rounds 1..N only                        -> length N
        # Keep track of round numbers separately to avoid x-axis mismatch.
        self._central_mta_history: List[float] = []
        self._central_rounds:      List[int]   = []
        self._dist_mta_history:    List[float] = []
        self._dist_rounds:         List[int]   = []
        self._loss_history:        List[float] = []

        self._pending_central: Optional[Tuple[int, float, float]] = None

        os.makedirs(LOG_DIR, exist_ok=True)
        self._csv_path = os.path.join(LOG_DIR, f"{simulation_id}_rounds.csv")
        self._init_csv()

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "central_mta", "dist_mta", "central_loss"])

    def _write_row(self, rnd: int, central_mta: float, dist_mta: float, central_loss: float) -> None:
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [rnd, f"{central_mta:.4f}", f"{dist_mta:.4f}", f"{central_loss:.4f}"]
            )

    # ------------------------------------------------------------------
    # evaluate — buffer centralized result, write CSV in aggregate_evaluate
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
        self._central_rounds.append(server_round)
        self._loss_history.append(loss)
        self._pending_central = (server_round, loss, central_mta)

        print(f"[Round {server_round}] Centralized — loss={loss:.4f}, MTA={central_mta:.4f}")
        return loss, metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate — write CSV row; run full metrics on final round
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
        self._dist_rounds.append(server_round)
        print(f"[Round {server_round}] Distributed MTA: {dist_mta:.4f}")

        if self._pending_central is not None:
            pending_rnd, pending_loss, pending_mta = self._pending_central
            self._write_row(pending_rnd, pending_mta, dist_mta, pending_loss)
            self._pending_central = None

        if server_round == self.num_rounds:
            self._print_summary()
            self._run_final_evaluation()

        return agg_loss, {"dist_mta": dist_mta}

    # ------------------------------------------------------------------
    # Final-round full evaluation
    # ------------------------------------------------------------------

    def _run_final_evaluation(self) -> None:
        if self._final_eval_model is None or self._final_eval_loader is None:
            print("[Metrics] No eval model/loader provided — skipping final metrics.")
            return

        print("\n[Metrics] Running final evaluation on held-out test set...")

        model  = self._final_eval_model
        loader = self._final_eval_loader
        device = next(model.parameters()).device

        model.eval()
        all_preds  = []
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                probs   = torch.softmax(outputs, dim=1)
                preds   = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds  = np.array(all_preds)
        all_probs  = np.array(all_probs)

        acc       = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm        = confusion_matrix(all_labels, all_preds)

        per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_recall    = recall_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_f1        = f1_score(all_labels, all_preds, average=None, zero_division=0)

        print("\n" + "=" * 60)
        print(f"Final FL Evaluation — {self.simulation_id}")
        print("=" * 60)
        print(f"Accuracy           : {acc*100:.2f}%")
        print(f"Weighted Precision : {precision:.4f}")
        print(f"Weighted Recall    : {recall:.4f}")
        print(f"Weighted F1        : {f1:.4f}")
        print("\nPer-class breakdown:")
        print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*42}")
        for i, cls in enumerate(ISIC_CLASS_NAMES):
            print(
                f"  {cls:<8} {per_class_precision[i]:>10.4f} "
                f"{per_class_recall[i]:>10.4f} {per_class_f1[i]:>10.4f}"
            )
        print("=" * 60)

        metrics_csv = os.path.join(LOG_DIR, f"{self.simulation_id}_final_metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1"])
            for i, cls in enumerate(ISIC_CLASS_NAMES):
                w.writerow([cls, f"{per_class_precision[i]:.4f}",
                             f"{per_class_recall[i]:.4f}", f"{per_class_f1[i]:.4f}"])
            w.writerow(["weighted", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
        print(f"[Metrics] Per-class CSV saved: {metrics_csv}")

        self._plot_confusion_matrix(cm)
        self._plot_roc_curves(all_labels, all_probs)
        self._plot_pr_curves(all_labels, all_probs)
        self._plot_training_curves()

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(ISIC_CLASS_NAMES)))
        ax.set_yticks(range(len(ISIC_CLASS_NAMES)))
        ax.set_xticklabels(ISIC_CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(ISIC_CLASS_NAMES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix — {self.simulation_id}")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)

        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"{self.simulation_id}_confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Metrics] Confusion matrix saved: {path}")

    def _plot_roc_curves(self, all_labels: np.ndarray, all_probs: np.ndarray) -> None:
        y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(ISIC_CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Per-class ROC Curves — {self.simulation_id}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"{self.simulation_id}_roc_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Metrics] ROC curves saved: {path}")

    def _plot_pr_curves(self, all_labels: np.ndarray, all_probs: np.ndarray) -> None:
        y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(ISIC_CLASS_NAMES):
            prec_vals, rec_vals, _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
            pr_auc = auc(rec_vals, prec_vals)
            ax.plot(rec_vals, prec_vals, label=f"{cls} (AUC={pr_auc:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Per-class Precision-Recall Curves — {self.simulation_id}")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"{self.simulation_id}_pr_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Metrics] PR curves saved: {path}")

    def _plot_training_curves(self) -> None:
        """
        Plot MTA and loss over rounds.

        _central_mta_history includes round 0 (init eval) through round N -> use _central_rounds
        _dist_mta_history includes rounds 1..N only                        -> use _dist_rounds
        Storing round numbers explicitly avoids any x-axis length mismatch.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MTA curves — central starts from round 0, distributed from round 1
        axes[0].plot(self._central_rounds, self._central_mta_history,
                     label="Centralized MTA", marker="o", markersize=3)
        if self._dist_mta_history:
            axes[0].plot(self._dist_rounds, self._dist_mta_history,
                         label="Distributed MTA", marker="s", markersize=3, linestyle="--")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("MTA over FL Rounds")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss curve
        axes[1].plot(self._central_rounds, self._loss_history,
                     label="Centralized Loss", color="orange", marker="o", markersize=3)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Centralized Loss over FL Rounds")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"FL Training Curves — {self.simulation_id}", fontsize=13)
        plt.tight_layout()
        path = os.path.join(LOG_DIR, f"{self.simulation_id}_training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Metrics] Training curves saved: {path}")

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"Experiment        : {self.simulation_id}")
        print(f"Rounds completed  : {self.num_rounds}")
        if self._central_mta_history:
            print(f"Final central MTA : {self._central_mta_history[-1]:.4f}")
            print(f"Peak central MTA  : {max(self._central_mta_history):.4f}")
        if self._dist_mta_history:
            print(f"Final dist MTA    : {self._dist_mta_history[-1]:.4f}")
        print(f"Round log         : {self._csv_path}")
        print("=" * 60 + "\n")