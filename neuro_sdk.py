import json
import random
from typing import Any, Dict, List, Optional, Sequence

import torch

import codex_client
import parser_utils
from device_utils import resolve_device
from trainer import TrainingEngine


class NeuroLab:
    """
    Headless SDK for NeuroDSL.

    Designed for:
    - scripts
    - autonomous agents
    - reproducible train/eval/infer loops
    """

    def __init__(self, api_key: Optional[str] = None, device: str = "auto"):
        self.api_key = api_key
        self.device = resolve_device(device)
        self.model = None
        self.trainer = None
        self.history: List[Dict[str, Any]] = []
        self.dsl_code: str = ""
        self.layer_defs: List[Dict[str, Any]] = []
        self.last_train_report: Dict[str, Any] = {}

    def capabilities(self) -> Dict[str, Any]:
        return {
            "sdk": "NeuroLab",
            "device": str(self.device),
            "features": [
                "build",
                "train",
                "infer",
                "evaluate",
                "evolve",
                "autopilot",
                "save_session",
                "load_session",
            ],
            "loss_functions": list(TrainingEngine.LOSS_FUNCTIONS.keys()),
        }

    def build(self, dsl_code: str):
        """Build a model from NeuroDSL text."""
        issues, layer_defs = parser_utils.validate_dsl(dsl_code)
        if layer_defs is None:
            raise ValueError(f"Invalid DSL code. Issues: {issues}")

        model = parser_utils.create_modern_nn(layer_defs)
        model = model.to(self.device)
        model.config = layer_defs

        self.model = model
        self.dsl_code = dsl_code
        self.layer_defs = layer_defs

        total_params = sum(p.numel() for p in model.parameters())
        print(f"SDK: Model built on {self.device}. Params: {total_params:,}")
        return self.model

    def _resolve_dims(self) -> tuple:
        if not self.layer_defs:
            raise ValueError("No layer definitions available.")
        first = self.layer_defs[0]
        last = self.layer_defs[-1]
        in_dim = int(first.get("in", first.get("dim")))
        out_dim = int(last.get("out", last.get("dim")))
        return in_dim, out_dim

    def train(
        self,
        X=None,
        y=None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 5,
        loss_fn: str = "MSE",
        grad_clip: float = 1.0,
        warmup_steps: int = 5,
        aux_loss_coef: float = 0.02,
        synthetic_samples: int = 512,
    ) -> float:
        """
        Train the active model.

        Returns:
            best loss as float (kept for backwards compatibility).
        """
        if self.model is None:
            raise ValueError("No model built.")

        self.trainer = TrainingEngine(
            self.model,
            loss_fn=loss_fn,
            max_epochs=int(epochs),
            grad_clip=float(grad_clip),
            warmup_steps=int(warmup_steps),
            base_lr=float(lr),
            aux_loss_coef=float(aux_loss_coef),
        )

        if X is None or y is None:
            in_dim, out_dim = self._resolve_dims()
            X, y = self.trainer.generate_dummy_data(in_dim, out_dim, n_samples=int(synthetic_samples))
        else:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)

        X = X.to(self.device)
        y = y.to(self.device)

        best_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        losses = []

        for epoch in range(int(epochs)):
            loss, curr_lr, grad_norm = self.trainer.train_step(X, y)
            losses.append(float(loss))

            if float(loss) < best_loss:
                best_loss = float(loss)
                best_epoch = epoch
                patience_counter = 0
                self.history.append(
                    {
                        "epoch": epoch,
                        "loss": float(loss),
                        "lr": float(curr_lr),
                        "grad_norm": float(grad_norm),
                    }
                )
            else:
                patience_counter += 1

            if epochs <= 20 or epoch % max(1, int(epochs) // 10) == 0:
                print(
                    f"SDK: epoch={epoch:04d} loss={float(loss):.6f} "
                    f"lr={float(curr_lr):.6f} grad={float(grad_norm):.4f}"
                )

            if patience_counter >= int(patience):
                print(f"SDK: Early stopping at epoch {epoch}")
                break

        self.last_train_report = {
            "epochs_requested": int(epochs),
            "epochs_ran": len(losses),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "final_loss": float(losses[-1] if losses else best_loss),
            "batch_size": int(batch_size),
            "loss_fn": loss_fn,
        }
        return float(best_loss)

    def train_report(self) -> Dict[str, Any]:
        return dict(self.last_train_report)

    def infer(self, inputs: Sequence[float]) -> List[float]:
        if self.model is None:
            raise ValueError("No model built.")

        vals = [float(v) for v in inputs]
        in_dim, _ = self._resolve_dims()
        if len(vals) != in_dim:
            raise ValueError(f"Expected {in_dim} input values, got {len(vals)}")

        tensor_in = torch.tensor([vals], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor_in).detach().cpu().flatten().tolist()
        return output

    def evaluate(self, X, y) -> float:
        if self.model is None:
            raise ValueError("No model built.")
        if self.trainer is None:
            self.trainer = TrainingEngine(self.model)
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        return float(self.trainer.evaluate(X.to(self.device), y.to(self.device)))

    def evolve(self, current_dsl: str, feedback: str = "Optimize for accuracy and stability") -> str:
        """Use Codex to improve an architecture."""
        if not self.api_key:
            raise ValueError("API key required for evolution.")

        prompt = f"{feedback}\n\nCurrent DSL:\n{current_dsl}"
        success, new_dsl = codex_client.optimize_dsl(self.api_key, prompt)
        if success:
            return new_dsl
        print(f"SDK: Evolution failed: {new_dsl}")
        return current_dsl

    def _fallback_candidates(
        self,
        input_dim: int,
        output_dim: int,
        count: int,
        seed: int,
    ) -> List[str]:
        rng = random.Random(int(seed))
        candidates = []
        for _ in range(max(1, int(count))):
            hidden = rng.choice([32, 48, 64, 96, 128, 192])
            dsl = (
                f"[{int(input_dim)}, {hidden}], "
                f"moe: [{hidden}, {rng.choice([4, 6, 8])}, 1], "
                f"mod: [{hidden}, {rng.choice([2, 4])}, 0.35], "
                f"dropout: [{rng.choice([0.05, 0.1, 0.2])}], "
                f"[{hidden}, {int(output_dim)}]"
            )
            candidates.append(dsl)
        return candidates

    def autopilot(
        self,
        objective: str,
        input_dim: int,
        output_dim: int,
        candidates: int = 6,
        epochs_per_candidate: int = 25,
        samples: int = 512,
        seed: int = 42,
        loss_fn: str = "MSE",
        param_penalty: float = 0.002,
        use_ai: bool = True,
    ) -> Dict[str, Any]:
        """
        Autonomous DSL search loop.
        - proposes candidates (AI or fallback)
        - runs quick train loop
        - returns ranked report and keeps best model active
        """
        device = self.device
        in_dim = int(input_dim)
        out_dim = int(output_dim)
        trials = max(1, int(candidates))
        torch.manual_seed(int(seed))

        ai_items: List[Dict[str, Any]] = []
        if use_ai and self.api_key:
            ok, content = codex_client.generate_dsl_candidates(
                self.api_key,
                objective,
                input_dim=in_dim,
                output_dim=out_dim,
                count=trials,
            )
            if ok:
                try:
                    payload = json.loads(content)
                    if isinstance(payload, dict):
                        raw = payload.get("candidates", [])
                    elif isinstance(payload, list):
                        raw = payload
                    else:
                        raw = []
                    for item in raw:
                        if isinstance(item, str):
                            ai_items.append({"dsl": item, "notes": "ai"})
                        elif isinstance(item, dict) and item.get("dsl"):
                            ai_items.append(
                                {"dsl": str(item.get("dsl")), "notes": str(item.get("notes", "ai"))}
                            )
                except Exception:
                    ai_items = []

        candidate_dsls = [c["dsl"] for c in ai_items if c.get("dsl")]
        if len(candidate_dsls) < trials:
            candidate_dsls.extend(
                self._fallback_candidates(in_dim, out_dim, trials - len(candidate_dsls), seed=seed)
            )
        candidate_dsls = candidate_dsls[:trials]

        X = torch.randn(int(samples), in_dim, device=device)
        y = torch.tanh(X[:, : min(in_dim, out_dim)])
        if y.shape[1] < out_dim:
            extra = torch.sin(X[:, [0]].repeat(1, out_dim - y.shape[1]))
            y = torch.cat([y, extra], dim=1)
        y = y[:, :out_dim]

        results: List[Dict[str, Any]] = []
        best_idx = -1
        best_score = float("inf")
        best_model = None
        best_dsl = ""
        best_layer_defs: List[Dict[str, Any]] = []

        for idx, dsl in enumerate(candidate_dsls):
            issues, layer_defs = parser_utils.validate_dsl(dsl)
            if not layer_defs:
                results.append({"idx": idx, "dsl": dsl, "valid": False, "issues": issues})
                continue

            first = layer_defs[0]
            last = layer_defs[-1]
            c_in = int(first.get("in", first.get("dim")))
            c_out = int(last.get("out", last.get("dim")))
            if c_in != in_dim or c_out != out_dim:
                results.append(
                    {
                        "idx": idx,
                        "dsl": dsl,
                        "valid": False,
                        "issues": issues,
                        "error": f"dim_mismatch expected ({in_dim}->{out_dim}) got ({c_in}->{c_out})",
                    }
                )
                continue

            try:
                model = parser_utils.create_modern_nn(layer_defs).to(device)
                trainer = TrainingEngine(model, loss_fn=loss_fn, max_epochs=epochs_per_candidate)
                final_loss = None
                for _ in range(int(epochs_per_candidate)):
                    final_loss, _, _ = trainer.train_step(X, y)
                if final_loss is None:
                    raise RuntimeError("no_training_steps")
                params = int(sum(p.numel() for p in model.parameters()))
                score = float(final_loss) + float(param_penalty) * (params / 1_000_000.0)
                result = {
                    "idx": idx,
                    "dsl": dsl,
                    "valid": True,
                    "issues": issues,
                    "final_loss": float(final_loss),
                    "params": params,
                    "fitness": float(score),
                }
                results.append(result)
                if score < best_score:
                    best_score = float(score)
                    best_idx = idx
                    best_model = model
                    best_dsl = dsl
                    best_layer_defs = layer_defs
            except Exception as exc:
                results.append(
                    {
                        "idx": idx,
                        "dsl": dsl,
                        "valid": False,
                        "issues": issues,
                        "error": str(exc),
                    }
                )

        if best_model is None:
            raise RuntimeError("Autopilot failed to produce a valid candidate.")

        self.model = best_model
        self.dsl_code = best_dsl
        self.layer_defs = best_layer_defs

        report = {
            "objective": objective,
            "device": str(device),
            "input_dim": in_dim,
            "output_dim": out_dim,
            "best_idx": best_idx,
            "best_dsl": best_dsl,
            "best_fitness": best_score,
            "results": results,
        }
        self.last_train_report = {
            "mode": "autopilot",
            "best_idx": best_idx,
            "best_fitness": best_score,
        }
        return report

    def save_session(self, path: str):
        """Save model session to disk."""
        if self.model is None:
            raise ValueError("No model available to save.")
        state = {
            "dsl": self.dsl_code,
            "config": getattr(self.model, "config", self.layer_defs),
            "weights": self.model.state_dict(),
            "history_summary": [
                {"epoch": h.get("epoch"), "loss": h.get("loss")}
                for h in self.history
                if "epoch" in h
            ],
            "device": str(self.device),
            "last_train_report": self.last_train_report,
        }
        torch.save(state, path)
        print(f"SDK: Session saved to {path}")

    def load_session(self, path: str):
        """Load model session from disk."""
        state = torch.load(path, map_location="cpu")
        dsl = state.get("dsl", "")
        config = state.get("config", [])
        if dsl:
            self.build(dsl)
        else:
            self.build_from_config(config)
        self.model.load_state_dict(state["weights"], strict=False)
        self.model = self.model.to(self.device)
        self.dsl_code = dsl or ""
        self.layer_defs = config if isinstance(config, list) else self.layer_defs
        self.last_train_report = dict(state.get("last_train_report", {}))
        print(f"SDK: Session loaded from {path}")

    def build_from_config(self, config: List[Dict[str, Any]]):
        """Helper to build directly from parsed layer config."""
        if not config:
            raise ValueError("Empty config.")
        self.layer_defs = list(config)
        self.model = parser_utils.create_modern_nn(self.layer_defs).to(self.device)
        self.model.config = self.layer_defs
        return self.model
