import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from parser_utils import validate_dsl, create_modern_nn, DSL_PRESETS
from trainer import TrainingEngine
from auto_researcher import AutoResearcher


@dataclass
class ExperimentResult:
    """Lightweight container for a single lab run."""
    id: int
    timestamp: str
    preset: str
    dsl_code: str
    epochs: int
    final_loss: float
    final_val_loss: float
    accuracy: float
    f1: float
    suggestions: List[str]
    notes: str

    def to_csv_row(self) -> List[Any]:
        return [
            self.id,
            self.timestamp,
            self.preset,
            self.dsl_code,
            self.epochs,
            self.final_loss,
            self.final_val_loss,
            self.accuracy,
            self.f1,
            " | ".join(self.suggestions),
            self.notes,
        ]


class InnovationLab:
    """
    Autonomous Lab Mode
    -------------------
    A tiny orchestrator that:
      1) Picks a DSL preset
      2) Uses AutoResearcher to critique it
      3) Builds a model + runs a short synthetic training loop
      4) Logs metrics + research hints to CSV / JSON

    This is intentionally self-contained so it can be launched
    from the CLI without the GUI or HTTP agent.
    """

    def __init__(
        self,
        preset: str,
        epochs: int = 50,
        out_csv: str = "innovation_lab_runs.csv",
        out_json: str = "innovation_lab_summary.json",
    ) -> None:
        if preset not in DSL_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(DSL_PRESETS.keys())}")

        self.preset = preset
        self.dsl_code = DSL_PRESETS[preset]
        self.epochs = int(epochs)
        self.out_csv = out_csv
        self.out_json = out_json

        self.researcher = AutoResearcher()
        self.experiments: List[ExperimentResult] = []

    # --------------------------------------------
    # Core experiment runner
    # --------------------------------------------

    def run_experiments(self, iterations: int = 3) -> None:
        for i in range(1, iterations + 1):
            print(f"\n=== Innovation Lab Experiment {i}/{iterations} ({self.preset}) ===")
            result = self._run_single_experiment(exp_id=i)
            if result is not None:
                self.experiments.append(result)
                self._append_to_csv(result)
                self._update_json_summary()

        print(f"\n[InnovationLab] Completed {len(self.experiments)} experiments.")
        print(f"  CSV log  -> {os.path.abspath(self.out_csv)}")
        print(f"  JSON top -> {os.path.abspath(self.out_json)}")

    def _run_single_experiment(self, exp_id: int) -> Optional[ExperimentResult]:
        # 1) Validate DSL + get research suggestions
        issues, layer_defs = validate_dsl(self.dsl_code)
        if layer_defs is None:
            print("[InnovationLab] DSL validation failed; skipping experiment.")
            for severity, msg in issues:
                print(f"  {severity}: {msg}")
            return None

        suggestions = self.researcher.analyze_dsl(self.dsl_code)
        random_fact = self.researcher.get_random_fact()

        print("[InnovationLab] Researcher suggestions:")
        for s in suggestions:
            print(f"  - {s}")
        print(f"  - EXTRA: {random_fact}")

        # 2) Build model
        model = create_modern_nn(layer_defs)
        if model is None:
            print("[InnovationLab] Model creation returned None; skipping.")
            return None

        # Try to infer IO dims from model attributes (matches GUI logic)
        in_dim = getattr(model, "input_dim", 8)
        out_dim = getattr(model, "output_dim", 1)

        # 3) TrainingEngine + synthetic data
        trainer = TrainingEngine(
            model,
            loss_fn="MSE",
            max_epochs=self.epochs,
            base_lr=0.01,
            aux_loss_coef=0.0,
            use_ema=True,
        )

        X, y = trainer.generate_dummy_data(in_dim, out_dim)

        best_val_loss = float("inf")
        last_loss = float("inf")
        last_acc = 0.0
        last_f1 = 0.0

        print(f"[InnovationLab] Training for {self.epochs} epochs on synthetic data...")
        for epoch in range(self.epochs):
            loss, lr, grad_norm, acc, f1 = trainer.train_step(X, y)
            last_loss, last_acc, last_f1 = loss, acc, f1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_loss, val_acc, val_f1 = trainer.evaluate(X, y)
                best_val_loss = min(best_val_loss, val_loss)
                print(
                    f"  Epoch {epoch+1:4d}/{self.epochs}: "
                    f"loss={loss:.5f}, val={val_loss:.5f}, lr={lr:.6f}, "
                    f"acc={val_acc:.3f}, f1={val_f1:.3f}"
                )

        timestamp = datetime.utcnow().isoformat() + "Z"
        notes = f"AutoResearcher fact: {random_fact}"

        val_loss, val_acc, val_f1 = trainer.evaluate(X, y)
        best_val_loss = min(best_val_loss, val_loss)

        result = ExperimentResult(
            id=exp_id,
            timestamp=timestamp,
            preset=self.preset,
            dsl_code=self.dsl_code,
            epochs=self.epochs,
            final_loss=float(last_loss),
            final_val_loss=float(best_val_loss),
            accuracy=float(last_acc),
            f1=float(last_f1),
            suggestions=suggestions,
            notes=notes,
        )

        print(
            f"[InnovationLab] Experiment {exp_id} summary -> "
            f"final_loss={result.final_loss:.5f}, "
            f"best_val={result.final_val_loss:.5f}, "
            f"acc={result.accuracy:.3f}, f1={result.f1:.3f}"
        )
        return result

    # --------------------------------------------
    # Logging helpers
    # --------------------------------------------

    def _append_to_csv(self, result: ExperimentResult) -> None:
        header = [
            "id",
            "timestamp",
            "preset",
            "dsl_code",
            "epochs",
            "final_loss",
            "final_val_loss",
            "accuracy",
            "f1",
            "suggestions",
            "notes",
        ]
        file_exists = os.path.exists(self.out_csv)
        with open(self.out_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(result.to_csv_row())

    def _update_json_summary(self) -> None:
        """
        Maintain a small JSON summary of the best experiments so far.
        Sorted by validation loss ascending.
        """
        if not self.experiments:
            return

        # Sort by validation loss (smaller is better)
        ranked = sorted(self.experiments, key=lambda r: r.final_val_loss)
        summary = [asdict(r) for r in ranked[:20]]
        payload = {
            "preset": self.preset,
            "total_experiments": len(self.experiments),
            "best_runs": summary,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroDSL Innovation Lab — autonomous preset experimentation"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="Research Frontier",
        help="DSL preset name to start from (see parser_utils.DSL_PRESETS)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of repeated experiments to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per experiment.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="innovation_lab_runs.csv",
        help="Path to CSV log file.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="innovation_lab_summary.json",
        help="Path to JSON summary file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lab = InnovationLab(
        preset=args.preset,
        epochs=args.epochs,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    start = time.time()
    lab.run_experiments(iterations=args.iterations)
    elapsed = time.time() - start
    print(f"[InnovationLab] Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

