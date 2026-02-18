import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

from parser_utils import validate_dsl, create_modern_nn, DSL_PRESETS
from trainer import TrainingEngine
from auto_researcher import AutoResearcher


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mutate_dsl(dsl_code: str, mutation_strength: float = 0.3) -> str:
    """
    Lightweight DSL mutation for architecture search.
    Mutates numeric dimensions in [..., ...] patterns.
    """
    def replace_num(m):
        val = int(m.group(0))
        delta = max(1, int(val * mutation_strength * (random.random() * 2 - 1)))
        new_val = max(8, min(512, val + delta))
        if random.random() < 0.3:
            new_val = (new_val // 8) * 8
        return str(new_val)

    return re.sub(r"\d+", replace_num, dsl_code)


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
    model_path: str = ""
    loss_fn: str = "MSE"
    lr: float = 0.01

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
            self.model_path,
            self.loss_fn,
            self.lr,
        ]


class InnovationLab:
    """
    Autonomous Lab Mode
    -------------------
    A tiny orchestrator that:
      1) Picks a DSL preset (or multiple)
      2) Optionally mutates DSL between iterations for architecture search
      3) Uses AutoResearcher to critique it
      4) Builds a model + runs training (synthetic or CSV data)
      5) Early stopping, export best model, logs to CSV / JSON
    """

    def __init__(
        self,
        preset: str,
        epochs: int = 50,
        out_csv: str = "innovation_lab_runs.csv",
        out_json: str = "innovation_lab_summary.json",
        data_csv: Optional[str] = None,
        export_best: bool = False,
        export_dir: str = "innovation_lab_models",
        patience: int = 0,
        mutate: bool = False,
        mutation_strength: float = 0.2,
        loss_fn: str = "MSE",
        lr: float = 0.01,
        quiet: bool = False,
    ) -> None:
        if preset not in DSL_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(DSL_PRESETS.keys())}")

        self.preset = preset
        self.dsl_code = DSL_PRESETS[preset]
        self.epochs = int(epochs)
        self.out_csv = out_csv
        self.out_json = out_json
        self.data_csv = data_csv
        self.export_best = export_best
        self.export_dir = export_dir
        self.patience = max(0, int(patience))
        self.mutate = mutate
        self.mutation_strength = mutation_strength
        self.loss_fn = loss_fn
        self.lr = lr
        self.quiet = quiet

        self.researcher = AutoResearcher()
        self.experiments: List[ExperimentResult] = []
        self._best_val_loss = float("inf")
        self._best_model = None
        self._best_dsl = ""

        if export_best:
            os.makedirs(export_dir, exist_ok=True)

    # --------------------------------------------
    # Core experiment runner
    # --------------------------------------------

    def run_experiments(self, iterations: int = 3, presets: Optional[List[str]] = None) -> None:
        preset_list = presets or [self.preset]
        exp_id = 0
        for preset in preset_list:
            if preset not in DSL_PRESETS:
                if not self.quiet:
                    print(f"[InnovationLab] Unknown preset '{preset}', skipping.")
                continue
            self.preset = preset
            self.dsl_code = DSL_PRESETS[preset]
            for i in range(1, iterations + 1):
                exp_id += 1
                if self.mutate and i > 1:
                    self.dsl_code = mutate_dsl(self.dsl_code, self.mutation_strength)
                    if not self.quiet:
                        print(f"[InnovationLab] Mutated DSL: {self.dsl_code[:80]}...")
                if not self.quiet:
                    print(f"\n=== Innovation Lab Experiment {exp_id} ({self.preset}) ===")
                result = self._run_single_experiment(exp_id=exp_id)
                if result is not None:
                    self.experiments.append(result)
                    self._append_to_csv(result)
                    self._update_json_summary()
                    if self.export_best and result.final_val_loss < self._best_val_loss:
                        self._best_val_loss = result.final_val_loss
                        self._best_model = result
                        best_path = os.path.join(self.export_dir, "best_model.pt")
                        if result.model_path:
                            shutil.copy(result.model_path, best_path)

        if not self.quiet:
            print(f"\n[InnovationLab] Completed {len(self.experiments)} experiments.")
            print(f"  CSV log  -> {os.path.abspath(self.out_csv)}")
            print(f"  JSON top -> {os.path.abspath(self.out_json)}")
            if self.export_best and self._best_model:
                print(f"  Best model -> {os.path.abspath(self.export_dir)}/")

    def _run_single_experiment(self, exp_id: int) -> Optional[ExperimentResult]:
        # 1) Validate DSL + get research suggestions
        issues, layer_defs = validate_dsl(self.dsl_code)
        if layer_defs is None:
            if not self.quiet:
                print("[InnovationLab] DSL validation failed; skipping experiment.")
                for severity, msg in issues:
                    print(f"  {severity}: {msg}")
            return None

        suggestions = self.researcher.analyze_dsl(self.dsl_code)
        random_fact = self.researcher.get_random_fact()

        if not self.quiet:
            print("[InnovationLab] Researcher suggestions:")
            for s in suggestions:
                print(f"  - {s}")
            print(f"  - EXTRA: {random_fact}")

        # 2) Build model
        model = create_modern_nn(layer_defs)
        if model is None:
            if not self.quiet:
                print("[InnovationLab] Model creation returned None; skipping.")
            return None

        # Infer IO dims: ModernMLP may not set input_dim; get from first/last layer
        in_dim = getattr(model, "input_dim", None)
        out_dim = getattr(model, "output_dim", None)
        if in_dim is None and layer_defs:
            first = layer_defs[0]
            if first.get("type") == "linear":
                in_dim = first.get("in", 8)
            else:
                in_dim = first.get("dim", 8)
        if out_dim is None and layer_defs:
            last = layer_defs[-1]
            if last.get("type") == "linear":
                out_dim = last.get("out", 1)
            else:
                out_dim = last.get("dim", 1)
        in_dim = in_dim or 8
        out_dim = out_dim or 1

        # 3) TrainingEngine
        trainer = TrainingEngine(
            model,
            loss_fn=self.loss_fn,
            max_epochs=self.epochs,
            base_lr=self.lr,
            aux_loss_coef=0.0,
            use_ema=True,
        )

        # 4) Data: CSV or synthetic
        if self.data_csv and os.path.exists(self.data_csv):
            try:
                X, y = trainer.load_csv_data(self.data_csv)
                if not self.quiet:
                    print(f"[InnovationLab] Loaded CSV: {X.shape[0]} samples, {X.shape[1]} features")
            except Exception as e:
                if not self.quiet:
                    print(f"[InnovationLab] CSV load failed: {e}; falling back to synthetic.")
                X, y = trainer.generate_dummy_data(in_dim, out_dim)
        else:
            X, y = trainer.generate_dummy_data(in_dim, out_dim)

        # Train/val split
        n = len(X)
        val_n = max(1, int(n * 0.2))
        X_val, y_val = X[:val_n], y[:val_n]
        X_train, y_train = X[val_n:], y[val_n:]

        best_val_loss = float("inf")
        last_loss = float("inf")
        last_acc = 0.0
        last_f1 = 0.0
        patience_counter = 0
        actual_epochs = 0

        if not self.quiet:
            data_src = "CSV" if (self.data_csv and os.path.exists(self.data_csv)) else "synthetic"
            print(f"[InnovationLab] Training for up to {self.epochs} epochs ({data_src})...")
        for epoch in range(self.epochs):
            loss, lr, grad_norm, acc, f1 = trainer.train_step(X_train, y_train)
            last_loss, last_acc, last_f1 = loss, acc, f1
            actual_epochs = epoch + 1

            val_loss, val_acc, val_f1 = trainer.evaluate(X_val, y_val)
            old_best = best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            if self.patience > 0:
                if val_loss >= old_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter >= self.patience:
                    if not self.quiet:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

            if not self.quiet and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch+1:4d}/{self.epochs}: "
                    f"loss={loss:.5f}, val={val_loss:.5f}, lr={lr:.6f}, "
                    f"acc={val_acc:.3f}, f1={val_f1:.3f}"
                )

        timestamp = datetime.utcnow().isoformat() + "Z"
        notes = f"AutoResearcher fact: {random_fact}"

        model_path = ""
        if self.export_best:
            path = os.path.join(self.export_dir, f"exp_{exp_id}.pt")
            torch.save(model.state_dict(), path)
            model_path = path

        result = ExperimentResult(
            id=exp_id,
            timestamp=timestamp,
            preset=self.preset,
            dsl_code=self.dsl_code,
            epochs=actual_epochs,
            final_loss=float(last_loss),
            final_val_loss=float(best_val_loss),
            accuracy=float(last_acc),
            f1=float(last_f1),
            suggestions=suggestions,
            notes=notes,
            model_path=model_path,
            loss_fn=self.loss_fn,
            lr=self.lr,
        )

        if not self.quiet:
            print(
                f"[InnovationLab] Experiment {exp_id} -> "
                f"final_loss={result.final_loss:.5f}, best_val={result.final_val_loss:.5f}, "
                f"acc={result.accuracy:.3f}, f1={result.f1:.3f}"
            )
        return result

    # --------------------------------------------
    # Logging helpers
    # --------------------------------------------

    def _append_to_csv(self, result: ExperimentResult) -> None:
        header = [
            "id", "timestamp", "preset", "dsl_code", "epochs",
            "final_loss", "final_val_loss", "accuracy", "f1",
            "suggestions", "notes", "model_path", "loss_fn", "lr",
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
        help="DSL preset name (or use --presets for multiple)",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help="Comma-separated presets, e.g. 'Research Frontier,Ultra-Efficient,Symbolic Reasoner'",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of experiments per preset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per experiment.",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Path to CSV for real data (target = last column). If missing, uses synthetic.",
    )
    parser.add_argument(
        "--export-best",
        action="store_true",
        help="Save best model to export_dir/best_model.pt",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="innovation_lab_models",
        help="Directory for exported models (with --export-best).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (epochs without val improvement). 0 = disabled.",
    )
    parser.add_argument(
        "--mutate",
        action="store_true",
        help="Mutate DSL between iterations for lightweight architecture search.",
    )
    parser.add_argument(
        "--mutation-strength",
        type=float,
        default=0.2,
        help="DSL mutation strength (0.1–0.5) when --mutate.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="MSE",
        choices=["MSE", "CrossEntropy", "Huber", "MAE", "Focal", "LabelSmooth"],
        help="Loss function.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output.",
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
    if args.seed is not None:
        set_seed(args.seed)

    presets = None
    if args.presets:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    if not presets:
        presets = [args.preset]

    lab = InnovationLab(
        preset=presets[0],
        epochs=args.epochs,
        out_csv=args.out_csv,
        out_json=args.out_json,
        data_csv=args.data_csv,
        export_best=args.export_best,
        export_dir=args.export_dir,
        patience=args.patience,
        mutate=args.mutate,
        mutation_strength=args.mutation_strength,
        loss_fn=args.loss,
        lr=args.lr,
        quiet=args.quiet,
    )
    start = time.time()
    lab.run_experiments(iterations=args.iterations, presets=presets)
    elapsed = time.time() - start
    if not args.quiet:
        print(f"[InnovationLab] Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

