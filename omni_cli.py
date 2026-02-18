"""Unified command line for NeuroDSL Infinity Studio.

This CLI orchestrates:
- existing scripts (main.py / verify.py / run_model.py / test_functional.py)
- tabular model training/inference
- image mode training/generation
- multimodal mode training/inference
- web checkpoint download
- accelerator selection including NPU/XPU paths when available
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

import codex_client
from data_preprocessor import preprocess_data
from device_utils import format_device_report, resolve_device
from experimental_models import (
    ExperimentalTrainConfig,
    ImageAutoencoder,
    MultiModalFusionModel,
    generate_uncanny_images,
    generate_interpolation_images,
    load_images_from_folder,
    save_image_grid,
    text_to_feature_vector,
    train_image_autoencoder,
    train_multimodal,
)
from parser_utils import parse_program
from trainer import TrainingEngine
from web_model_utils import download_file, download_from_hf
from advanced_training import AdvancedTrainingEngine, CurriculumLearning, KnowledgeDistillation, create_cyclic_scheduler, create_cosine_annealing_warmup
from viz_utils import ModelVisualizer

def _dsl_text(args) -> str:
    if args.dsl and args.dsl_file:
        raise ValueError("Use only one of --dsl or --dsl-file.")
    if args.dsl_file:
        return Path(args.dsl_file).read_text(encoding="utf-8")
    return args.dsl or ""


def _dsl_text_from_values(dsl: str = "", dsl_file: str = "") -> str:
    if dsl and dsl_file:
        raise ValueError("Use only one of --dsl or --dsl-file.")
    if dsl_file:
        return Path(dsl_file).read_text(encoding="utf-8")
    return dsl or ""


def _run_existing_script(script: str, script_args: str = "") -> int:
    cmd = [sys.executable, script]
    if script_args:
        cmd.extend(shlex.split(script_args))
    print(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def cmd_devices(_args):
    print(format_device_report())
    return 0


def cmd_download_model(args):
    if args.url:
        out = download_file(args.url, args.output)
        print(f"[ok] downloaded: {out}")
        return 0
    if args.hf_repo and args.hf_file:
        out = download_from_hf(args.hf_repo, args.hf_file, local_dir=args.hf_dir)
        print(f"[ok] downloaded: {out}")
        return 0
    raise ValueError("Provide either --url + --output, or --hf-repo + --hf-file.")


def cmd_run_script(args):
    scripts = {
        "main": "main.py",
        "verify": "verify.py",
        "functional": "test_functional.py",
        "run_model": "run_model.py",
        "omni_gui": "omni_studio.py",
        "console_app": "console_app.py",
    }
    script = scripts.get(args.script, args.script)
    if not os.path.exists(script):
        raise FileNotFoundError(f"Script not found: {script}")
    return _run_existing_script(script, args.script_args)


def _print_or_save(text: str, output_path: str = ""):
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"[ok] wrote {output_path}")
    else:
        print(text)


def cmd_ai_connection(args):
    ok, content = codex_client.test_connection(api_key=args.api_key, model=args.model)
    if not ok:
        raise RuntimeError(content)
    print(f"[ok] connection test: {content}")
    return 0


def cmd_ai_dsl(args):
    ok, content = codex_client.generate_dsl(args.api_key, args.prompt, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output_dsl)
    return 0


def cmd_ai_explain(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.explain_dsl(args.api_key, dsl, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output)
    return 0


def cmd_ai_optimize(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.optimize_dsl(args.api_key, dsl, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output_dsl)
    return 0


def cmd_ai_hyperparams(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.suggest_hyperparams(args.api_key, dsl, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output_json)
    return 0


def cmd_ai_codegen(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.generate_pytorch_code(args.api_key, dsl, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output_py)
    return 0


def cmd_ai_latency(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.estimate_latency(
        args.api_key,
        dsl,
        hardware=args.hardware,
        model=args.model,
    )
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output)
    return 0


def cmd_ai_diagram(args):
    dsl = _dsl_text_from_values(args.dsl, args.dsl_file)
    if not dsl:
        raise ValueError("Provide --dsl or --dsl-file.")
    ok, content = codex_client.generate_ascii_diagram(args.api_key, dsl, model=args.model)
    if not ok:
        raise RuntimeError(content)
    _print_or_save(content, args.output)
    return 0


def _extract_first_json_object(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("{") or text.startswith("["):
        return text

    # Recover JSON from extra tokens if a model returns prose around it.
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [s for s in (start_obj, start_arr) if s >= 0]
    if not starts:
        return ""
    start = min(starts)
    candidate = text[start:]

    # Try object first by balancing braces.
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[: i + 1]

    # Fallback: array balancing.
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return candidate[: i + 1]
    return ""


def _parse_ai_candidate_payload(content: str) -> list:
    cleaned = _extract_first_json_object(content)
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except Exception:
        return []

    if isinstance(payload, dict):
        items = payload.get("candidates", [])
        if isinstance(items, list):
            return items
        return []
    if isinstance(payload, list):
        return payload
    return []


def cmd_ai_autopilot(args):
    from parser_utils import create_modern_nn, validate_dsl

    in_dim = int(args.input_dim)
    out_dim = int(args.output_dim)
    trials = int(max(1, args.candidates))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)

    ai_candidates = []
    if not args.offline:
        ok, content = codex_client.generate_dsl_candidates(
            args.api_key,
            args.objective,
            input_dim=in_dim,
            output_dim=out_dim,
            count=trials,
            model=args.model,
        )
        if ok:
            ai_candidates = _parse_ai_candidate_payload(content)
        else:
            print(f"[warn] ai candidate generation failed: {content}")

    candidates = []
    for item in ai_candidates:
        if isinstance(item, str):
            dsl = item.strip()
            notes = "ai"
        elif isinstance(item, dict):
            dsl = str(item.get("dsl", "")).strip()
            notes = str(item.get("notes", "")).strip()
        else:
            continue
        if dsl:
            candidates.append({"dsl": dsl, "notes": notes or "ai"})

    while len(candidates) < trials:
        candidates.append(
            {
                "dsl": _random_dsl(in_dim, out_dim, rng),
                "notes": "fallback_random",
            }
        )
    candidates = candidates[:trials]

    n_samples = int(max(8, args.samples))
    X = torch.randn(n_samples, in_dim, device=device)
    y = torch.tanh(X[:, : min(in_dim, out_dim)])
    if y.shape[1] < out_dim:
        extra = torch.sin(X[:, [0]].repeat(1, out_dim - y.shape[1]))
        y = torch.cat([y, extra], dim=1)
    y = y[:, :out_dim]

    results = []
    best = None
    best_state = None

    for idx, candidate in enumerate(candidates):
        dsl = candidate["dsl"]
        issues, defs = validate_dsl(dsl)
        if not defs:
            results.append(
                {
                    "idx": idx,
                    "dsl": dsl,
                    "notes": candidate.get("notes", ""),
                    "valid": False,
                    "error": "parse_or_validation_failed",
                    "issues": issues,
                }
            )
            print(f"[autopilot] candidate={idx:02d} invalid")
            continue

        first = defs[0]
        last = defs[-1]
        c_in = int(first.get("in", first.get("dim", -1)))
        c_out = int(last.get("out", last.get("dim", -1)))
        if c_in != in_dim or c_out != out_dim:
            results.append(
                {
                    "idx": idx,
                    "dsl": dsl,
                    "notes": candidate.get("notes", ""),
                    "valid": False,
                    "error": f"dim_mismatch expected ({in_dim}->{out_dim}) got ({c_in}->{c_out})",
                    "issues": issues,
                }
            )
            print(f"[autopilot] candidate={idx:02d} dim mismatch")
            continue

        try:
            model = create_modern_nn(defs).to(device)
            trainer = TrainingEngine(
                model,
                loss_fn=args.loss,
                max_epochs=args.epochs_per_candidate,
                grad_clip=args.grad_clip,
                warmup_steps=args.warmup_steps,
                aux_loss_coef=args.aux_loss_coef,
            )

            final_loss = None
            for _ in range(int(args.epochs_per_candidate)):
                final_loss, _, _ = trainer.train_step(X, y)
            if final_loss is None:
                raise RuntimeError("No training steps executed.")

            params = int(sum(p.numel() for p in model.parameters()))
            fitness = float(final_loss) + float(args.param_penalty) * (params / 1_000_000.0)
            result = {
                "idx": idx,
                "dsl": dsl,
                "notes": candidate.get("notes", ""),
                "valid": True,
                "final_loss": float(final_loss),
                "fitness": fitness,
                "params": params,
                "issues": issues,
            }
            results.append(result)
            print(
                f"[autopilot] candidate={idx:02d} loss={final_loss:.6f} "
                f"fitness={fitness:.6f} params={params}"
            )

            if best is None or fitness < best["fitness"]:
                best = result
                best_state = model.state_dict()
        except Exception as exc:
            results.append(
                {
                    "idx": idx,
                    "dsl": dsl,
                    "notes": candidate.get("notes", ""),
                    "valid": False,
                    "error": str(exc),
                    "issues": issues,
                }
            )
            print(f"[autopilot] candidate={idx:02d} failed: {exc}")

    if best is None or best_state is None:
        raise RuntimeError("Autopilot could not produce a valid trained candidate.")

    report = {
        "objective": args.objective,
        "input_dim": in_dim,
        "output_dim": out_dim,
        "device": str(device),
        "best": best,
        "results": results,
    }
    report_path = out_dir / "autopilot_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "best.dsl").write_text(best["dsl"], encoding="utf-8")
    torch.save(
        {
            "state_dict": best_state,
            "dsl": best["dsl"],
            "fitness": best["fitness"],
            "final_loss": best["final_loss"],
            "params": best["params"],
        },
        out_dir / "best.pth",
    )

    if not args.offline:
        ok, explanation = codex_client.explain_dsl(args.api_key, best["dsl"], model=args.model)
        if ok:
            (out_dir / "best_explain.txt").write_text(explanation, encoding="utf-8")

    print(f"[autopilot] best_fitness={best['fitness']:.6f}")
    print(f"[autopilot] best_loss={best['final_loss']:.6f}")
    print(f"[autopilot] best_dsl={best['dsl']}")
    print(f"[ok] wrote report: {report_path}")
    print(f"[ok] wrote best dsl: {out_dir / 'best.dsl'}")
    print(f"[ok] wrote best weights: {out_dir / 'best.pth'}")
    return 0


def _agent_manifest() -> dict:
    return {
        "name": "NeuroDSL Agent API",
        "version": "1.0",
        "commands": {
            "dsl.validate": {
                "method": "POST",
                "path": "/dsl/validate",
                "body": {"dsl": "string"},
            },
            "session.build": {
                "method": "POST",
                "path": "/session/build",
                "body": {"dsl": "string"},
            },
            "session.train": {
                "method": "POST",
                "path": "/session/train",
                "body": {
                    "epochs": "int (optional)",
                    "samples": "int (optional)",
                    "loss": "MSE|CrossEntropy|Huber|MAE (optional)",
                    "grad_clip": "float (optional)",
                    "warmup_steps": "int (optional)",
                    "aux_loss_coef": "float (optional)",
                    "save_pth": "string path (optional)",
                },
            },
            "session.infer": {
                "method": "POST",
                "path": "/session/infer",
                "body": {"inputs": "list[float] or list[list[float]]"},
            },
            "session.reset": {
                "method": "POST",
                "path": "/session/reset",
                "body": {},
            },
            "meta.manifest": {"method": "GET", "path": "/manifest"},
            "meta.health": {"method": "GET", "path": "/health"},
            "meta.session": {"method": "GET", "path": "/session"},
        },
    }


def cmd_agent_manifest(args):
    payload = json.dumps(_agent_manifest(), indent=2)
    _print_or_save(payload, args.output_json)
    return 0


def cmd_serve_agent_api(args):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from parser_utils import create_modern_nn, validate_dsl

    device = resolve_device(args.device)
    state = {
        "model": None,
        "dsl": "",
        "expected_in": None,
        "expected_out": None,
    }
    lock = threading.Lock()

    def _send_json(handler: BaseHTTPRequestHandler, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                _send_json(self, 200, {"status": "ok", "device": str(device)})
                return
            if self.path.startswith("/manifest"):
                _send_json(self, 200, _agent_manifest())
                return
            if self.path.startswith("/session"):
                with lock:
                    info = {
                        "has_model": state["model"] is not None,
                        "dsl": state["dsl"],
                        "expected_in": state["expected_in"],
                        "expected_out": state["expected_out"],
                        "device": str(device),
                    }
                _send_json(self, 200, info)
                return
            _send_json(self, 404, {"error": "Not found"})

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_POST(self):
            try:
                data = self._read_json()
            except Exception:
                _send_json(self, 400, {"error": "Invalid JSON"})
                return

            if self.path.startswith("/dsl/validate"):
                dsl = str(data.get("dsl", "")).strip()
                if not dsl:
                    _send_json(self, 400, {"error": "Missing dsl"})
                    return
                issues, layer_defs = validate_dsl(dsl)
                _send_json(
                    self,
                    200,
                    {
                        "ok": layer_defs is not None,
                        "issues": issues,
                        "layer_defs": layer_defs,
                    },
                )
                return

            if self.path.startswith("/session/build"):
                dsl = str(data.get("dsl", "")).strip()
                if not dsl:
                    _send_json(self, 400, {"error": "Missing dsl"})
                    return
                issues, layer_defs = validate_dsl(dsl)
                if not layer_defs:
                    _send_json(self, 400, {"error": "Invalid DSL", "issues": issues})
                    return
                try:
                    model = create_modern_nn(layer_defs).to(device)
                    model.eval()
                    first = layer_defs[0]
                    last = layer_defs[-1]
                    expected_in = int(first.get("in", first.get("dim")))
                    expected_out = int(last.get("out", last.get("dim")))
                    params = int(sum(p.numel() for p in model.parameters()))
                    with lock:
                        state["model"] = model
                        state["dsl"] = dsl
                        state["expected_in"] = expected_in
                        state["expected_out"] = expected_out
                    _send_json(
                        self,
                        200,
                        {
                            "ok": True,
                            "issues": issues,
                            "expected_in": expected_in,
                            "expected_out": expected_out,
                            "params": params,
                        },
                    )
                except Exception as exc:
                    _send_json(self, 500, {"error": str(exc)})
                return

            if self.path.startswith("/session/train"):
                with lock:
                    model = state["model"]
                    in_dim = state["expected_in"]
                    out_dim = state["expected_out"]
                if model is None:
                    _send_json(self, 400, {"error": "No active model. Call /session/build first."})
                    return
                try:
                    epochs = int(data.get("epochs", 30))
                    samples = int(data.get("samples", 512))
                    loss = str(data.get("loss", "MSE"))
                    grad_clip = float(data.get("grad_clip", 1.0))
                    warmup_steps = int(data.get("warmup_steps", 5))
                    aux_loss_coef = float(data.get("aux_loss_coef", 0.02))
                    save_pth = str(data.get("save_pth", "")).strip()

                    trainer = TrainingEngine(
                        model,
                        loss_fn=loss,
                        max_epochs=epochs,
                        grad_clip=grad_clip,
                        warmup_steps=warmup_steps,
                        aux_loss_coef=aux_loss_coef,
                    )
                    X, y = trainer.generate_dummy_data(int(in_dim), int(out_dim), n_samples=samples)
                    X = X.to(device)
                    y = y.to(device)
                    final_loss = None
                    for _ in range(epochs):
                        final_loss, _, _ = trainer.train_step(X, y)
                    payload = {
                        "ok": True,
                        "final_loss": float(final_loss if final_loss is not None else 0.0),
                        "epochs": epochs,
                        "samples": samples,
                    }
                    if save_pth:
                        Path(save_pth).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "dsl": state["dsl"],
                                "final_loss": payload["final_loss"],
                            },
                            save_pth,
                        )
                        payload["save_pth"] = save_pth
                    _send_json(self, 200, payload)
                except Exception as exc:
                    _send_json(self, 500, {"error": str(exc)})
                return

            if self.path.startswith("/session/infer"):
                with lock:
                    model = state["model"]
                    expected_in = state["expected_in"]
                if model is None:
                    _send_json(self, 400, {"error": "No active model. Call /session/build first."})
                    return
                rows = data.get("inputs")
                if rows is None:
                    _send_json(self, 400, {"error": "Missing inputs"})
                    return
                if not isinstance(rows, list) or not rows:
                    _send_json(self, 400, {"error": "inputs must be a non-empty list"})
                    return
                if not isinstance(rows[0], list):
                    rows = [rows]
                for row in rows:
                    if len(row) != int(expected_in):
                        _send_json(self, 400, {"error": f"Each row must have dim {expected_in}"})
                        return
                try:
                    x = torch.tensor(rows, dtype=torch.float32, device=device)
                    with lock:
                        with torch.no_grad():
                            out = state["model"](x).cpu().tolist()
                    _send_json(self, 200, {"outputs": out})
                except Exception as exc:
                    _send_json(self, 500, {"error": str(exc)})
                return

            if self.path.startswith("/session/reset"):
                with lock:
                    state["model"] = None
                    state["dsl"] = ""
                    state["expected_in"] = None
                    state["expected_out"] = None
                _send_json(self, 200, {"ok": True})
                return

            _send_json(self, 404, {"error": "Not found"})

    host = args.host
    port = int(args.port)
    print(f"[agent-api] starting http://{host}:{port}")
    print("[agent-api] endpoints: /health /manifest /session /dsl/validate /session/build /session/train /session/infer /session/reset")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[agent-api] stopping...")
    finally:
        server.server_close()
    return 0


def _parse_seed_list(seed_text: str) -> list:
    raw = (seed_text or "").strip()
    if not raw:
        return [42]
    out = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        out = [42]
    return out


def cmd_ai_autopilot_sweep(args):
    seeds = _parse_seed_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objective = args.objective
    records = []

    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "omni_cli.py",
            "ai-autopilot",
            "--objective",
            objective,
            "--input-dim",
            str(int(args.input_dim)),
            "--output-dim",
            str(int(args.output_dim)),
            "--candidates",
            str(int(args.candidates)),
            "--epochs-per-candidate",
            str(int(args.epochs_per_candidate)),
            "--samples",
            str(int(args.samples)),
            "--loss",
            args.loss,
            "--grad-clip",
            str(float(args.grad_clip)),
            "--warmup-steps",
            str(int(args.warmup_steps)),
            "--aux-loss-coef",
            str(float(args.aux_loss_coef)),
            "--param-penalty",
            str(float(args.param_penalty)),
            "--seed",
            str(int(seed)),
            "--device",
            args.device,
            "--out-dir",
            str(seed_dir),
        ]
        if args.offline:
            cmd.append("--offline")
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.model:
            cmd.extend(["--model", args.model])

        print(f"[sweep] seed={seed} running...")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.returncode != 0:
            if proc.stderr.strip():
                print(proc.stderr.strip())
            print(f"[sweep] seed={seed} failed with rc={proc.returncode}")
            continue

        report_path = seed_dir / "autopilot_report.json"
        if not report_path.exists():
            print(f"[sweep] seed={seed} missing report: {report_path}")
            continue

        report = json.loads(report_path.read_text(encoding="utf-8"))
        best = report.get("best", {})
        if not best:
            print(f"[sweep] seed={seed} report has no best result")
            continue
        record = {
            "seed": int(seed),
            "fitness": float(best.get("fitness", 1e9)),
            "loss": float(best.get("final_loss", 1e9)),
            "dsl": str(best.get("dsl", "")),
            "dir": str(seed_dir),
        }
        records.append(record)

    if not records:
        raise RuntimeError("Autopilot sweep failed for all seeds.")

    records.sort(key=lambda r: r["fitness"])
    global_best = records[0]
    summary = {
        "objective": objective,
        "input_dim": int(args.input_dim),
        "output_dim": int(args.output_dim),
        "device": args.device,
        "seeds": seeds,
        "runs": records,
        "global_best": global_best,
    }
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {summary_path}")
    print(f"[sweep] best seed={global_best['seed']} fitness={global_best['fitness']:.6f}")

    best_dir = Path(global_best["dir"])
    best_dsl_src = best_dir / "best.dsl"
    best_pth_src = best_dir / "best.pth"

    if args.promote_best:
        promoted_dsl = out_dir / "champion.dsl"
        promoted_pth = out_dir / "champion.pth"
        shutil.copy2(best_dsl_src, promoted_dsl)
        shutil.copy2(best_pth_src, promoted_pth)
        print(f"[sweep] promoted best dsl: {promoted_dsl}")
        print(f"[sweep] promoted best pth: {promoted_pth}")

        if int(args.fine_tune_epochs) > 0:
            fine_out = out_dir / args.fine_tune_output
            cmd = [
                sys.executable,
                "omni_cli.py",
                "tabular-train",
                "--dsl-file",
                str(promoted_dsl),
                "--epochs",
                str(int(args.fine_tune_epochs)),
                "--samples",
                str(int(args.fine_tune_samples)),
                "--loss",
                args.loss,
                "--grad-clip",
                str(float(args.grad_clip)),
                "--warmup-steps",
                str(int(args.warmup_steps)),
                "--aux-loss-coef",
                str(float(args.aux_loss_coef)),
                "--log-every",
                str(int(args.fine_tune_log_every)),
                "--device",
                args.device,
                "--save-pth",
                str(fine_out),
            ]
            print(f"[sweep] fine-tuning promoted champion for {args.fine_tune_epochs} epochs...")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError("Champion fine-tuning failed.")
            print(f"[sweep] fine-tuned checkpoint: {fine_out}")
    return 0


def _champion_script_template(default_model_name: str, default_dsl_name: str, default_device: str) -> str:
    return f'''"""Interactive runner for a packaged NeuroDSL champion model.

Modes:
- repl: interactive shell for inference and benchmark
- infer: one-shot inference for a single vector
- batch: CSV batch inference
- benchmark: latency benchmark
- serve: lightweight HTTP inference API
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from device_utils import resolve_device
from parser_utils import create_modern_nn, parse_program


DEFAULT_MODEL = "{default_model_name}"
DEFAULT_DSL = "{default_dsl_name}"
DEFAULT_DEVICE = "{default_device}"


def _candidate_paths(filename: str):
    paths = []
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    if exe_dir is not None:
        paths.append(exe_dir / filename)
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        paths.append(Path(meipass) / filename)
    paths.append(Path(__file__).resolve().parent / filename)
    paths.append(Path(filename))
    return paths


def _resolve_path(path_or_name: str) -> str:
    p = Path(path_or_name)
    if p.exists():
        return str(p)
    for cand in _candidate_paths(path_or_name):
        if cand.exists():
            return str(cand)
    return path_or_name


def _parse_vector(text: str):
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("Input vector is empty.")
    return vals


def _load_model(model_path: str, dsl_path: str, device):
    model_path = _resolve_path(model_path)
    dsl_path = _resolve_path(dsl_path) if dsl_path else dsl_path

    ext = os.path.splitext(model_path)[1].lower()
    expected_dim = None

    if ext == ".pth":
        if not dsl_path or not os.path.exists(dsl_path):
            raise ValueError("A .pth checkpoint requires a valid DSL file.")
        dsl_text = Path(dsl_path).read_text(encoding="utf-8")
        layer_defs = parse_program(dsl_text)
        if not layer_defs:
            raise ValueError("Failed to parse DSL file.")
        model = create_modern_nn(layer_defs).to(device)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint format: expected a state_dict dict.")
        cleaned = {{}}
        for k, v in state.items():
            cleaned[k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k] = v
        model.load_state_dict(cleaned, strict=False)
        first = layer_defs[0]
        expected_dim = int(first.get("in", first.get("dim")))
    elif ext == ".pt":
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    else:
        raise ValueError("Unsupported model extension. Use .pth or .pt")

    model.eval()
    return model, expected_dim


def _infer_rows(model, rows, device):
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x).cpu().tolist()
    return out


def _load_csv_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        maybe_header = next(reader, None)
        if maybe_header:
            try:
                rows.append([float(v.strip()) for v in maybe_header if v.strip()])
            except Exception:
                pass
        for row in reader:
            try:
                vals = [float(v.strip()) for v in row if v.strip()]
                if vals:
                    rows.append(vals)
            except Exception:
                continue
    if not rows:
        raise ValueError("No numeric rows found in CSV.")
    return rows


def _write_csv(path, outputs):
    out_dim = len(outputs[0]) if outputs else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"out_{{i}}" for i in range(out_dim)])
        for row in outputs:
            w.writerow(row)


def _benchmark(model, rows, device, runs):
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(min(3, runs)):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / max(1, runs)


def _run_server(model, expected_dim, device, host, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {{"status": "ok", "device": str(device), "expected_dim": expected_dim}})
                return
            self._send(404, {{"error": "not found"}})

        def do_POST(self):
            if not self.path.startswith("/infer"):
                self._send(404, {{"error": "not found"}})
                return
            try:
                ln = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(ln)
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send(400, {{"error": "invalid json"}})
                return
            rows = data.get("inputs")
            if rows is None:
                self._send(400, {{"error": "missing inputs"}})
                return
            if not isinstance(rows, list) or not rows:
                self._send(400, {{"error": "inputs must be a non-empty list"}})
                return
            if not isinstance(rows[0], list):
                rows = [rows]
            if expected_dim is not None:
                for r in rows:
                    if len(r) != expected_dim:
                        self._send(400, {{"error": f"each row must have dim {{expected_dim}}"}})
                        return
            out = _infer_rows(model, rows, device)
            self._send(200, {{"outputs": out}})

    srv = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"[server] http://{{host}}:{{port}}")
    print("[server] GET /health, POST /infer")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[server] stopping...")
    finally:
        srv.server_close()


def _run_repl(model, expected_dim, device):
    print("Champion REPL ready. Commands: infer <csv>, random, bench [runs], batch <in.csv> [out.csv], help, exit")
    while True:
        try:
            line = input("champion> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line in ("help", "?"):
            print("infer <v1,v2,...> | random | bench [runs] | batch <in.csv> [out.csv] | exit")
            continue
        if line == "random":
            if expected_dim is None:
                print("[error] random requires known input dim")
                continue
            rows = torch.randn(1, expected_dim).tolist()
            out = _infer_rows(model, rows, device)
            print(json.dumps({{"inputs": rows[0], "outputs": out[0]}}, indent=2))
            continue
        if line.startswith("bench"):
            parts = line.split()
            runs = int(parts[1]) if len(parts) > 1 else 50
            if expected_dim is None:
                print("[error] bench requires known input dim")
                continue
            rows = torch.randn(4, expected_dim).tolist()
            ms = _benchmark(model, rows, device, runs=max(1, runs))
            print(f"[bench] {{ms:.3f}} ms/run (batch=4)")
            continue
        if line.startswith("infer "):
            raw = line[len("infer "):].strip()
            vals = _parse_vector(raw)
            if expected_dim is not None and len(vals) != expected_dim:
                print(f"[error] expected dim {{expected_dim}}, got {{len(vals)}}")
                continue
            out = _infer_rows(model, [vals], device)
            print(json.dumps({{"output": out[0]}}, indent=2))
            continue
        if line.startswith("batch "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: batch <in.csv> [out.csv]")
                continue
            in_csv = parts[1]
            out_csv = parts[2] if len(parts) > 2 else "preds.csv"
            rows = _load_csv_rows(in_csv)
            if expected_dim is not None:
                for i, row in enumerate(rows):
                    if len(row) != expected_dim:
                        raise ValueError(f"row {{i}} has dim {{len(row)}}, expected {{expected_dim}}")
            out = _infer_rows(model, rows, device)
            _write_csv(out_csv, out)
            print(f"[ok] wrote {{out_csv}} ({{len(out)}} rows)")
            continue
        print("[error] unknown command; type help")


def main():
    parser = argparse.ArgumentParser(description="Champion model runner")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dsl", default=DEFAULT_DSL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--mode", default="repl", choices=["repl", "infer", "batch", "benchmark", "serve"])
    parser.add_argument("--input", default="")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, expected_dim = _load_model(args.model, args.dsl, device)
    print(f"[info] model loaded on {{device}}")
    if expected_dim is not None:
        print(f"[info] expected input dim: {{expected_dim}}")

    if args.mode == "serve":
        _run_server(model, expected_dim, device, args.host, args.port)
        return

    if args.mode == "repl":
        _run_repl(model, expected_dim, device)
        return

    if args.mode == "infer":
        if not args.input:
            raise ValueError("--input is required for infer mode")
        vals = _parse_vector(args.input)
        if expected_dim is not None and len(vals) != expected_dim:
            raise ValueError(f"expected dim {{expected_dim}}, got {{len(vals)}}")
        out = _infer_rows(model, [vals], device)
        payload = {{"input": vals, "output": out[0]}}
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if args.mode == "batch":
        if not args.input_csv:
            raise ValueError("--input-csv is required for batch mode")
        rows = _load_csv_rows(args.input_csv)
        if expected_dim is not None:
            for i, row in enumerate(rows):
                if len(row) != expected_dim:
                    raise ValueError(f"row {{i}} has dim {{len(row)}}, expected {{expected_dim}}")
        out = _infer_rows(model, rows, device)
        payload = {{"rows": len(out), "outputs": out}}
        print(f"[info] rows={{len(out)}}")
        if args.output_csv:
            _write_csv(args.output_csv, out)
            print(f"[ok] wrote {{args.output_csv}}")
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {{args.output_json}}")
        return

    if args.mode == "benchmark":
        if expected_dim is None:
            raise ValueError("benchmark mode needs known input dim")
        rows = torch.randn(8, expected_dim).tolist()
        ms = _benchmark(model, rows, device, runs=max(1, int(args.runs)))
        print(f"[bench] {{ms:.3f}} ms/run (batch=8)")
        return


if __name__ == "__main__":
    main()
'''


def _build_champion_exe(
    script_path: Path,
    model_file: Path,
    dsl_file: Path | None,
    exe_name: str,
):
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        exe_name,
        "--add-data",
        f"{model_file.name}{sep}.",
        "--hidden-import",
        "parser_utils",
        "--hidden-import",
        "network",
        "--hidden-import",
        "device_utils",
        script_path.name,
    ]
    if dsl_file is not None:
        cmd.extend(["--add-data", f"{dsl_file.name}{sep}."])
    subprocess.run(cmd, cwd=str(script_path.parent), check=True)
    exe_suffix = ".exe" if os.name == "nt" else ""
    return script_path.parent / "dist" / f"{exe_name}{exe_suffix}"


def cmd_champion_package(args):
    model_src = Path(args.model)
    if not model_src.exists():
        raise FileNotFoundError(f"Model file not found: {model_src}")

    ext = model_src.suffix.lower()
    if ext not in (".pth", ".pt"):
        raise ValueError("Model must be .pth or .pt")

    dsl_text = _dsl_text_from_values(args.dsl, args.dsl_file)
    if ext == ".pth" and not dsl_text:
        raise ValueError("A .pth model requires --dsl or --dsl-file")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or f"champion_model{ext}"
    model_dst = out_dir / model_name
    shutil.copy2(model_src, model_dst)

    dsl_dst = None
    expected_dim = None
    if dsl_text:
        dsl_name = args.dsl_name or "champion_model.dsl"
        dsl_dst = out_dir / dsl_name
        dsl_dst.write_text(dsl_text, encoding="utf-8")
        defs = parse_program(dsl_text)
        if defs:
            first = defs[0]
            expected_dim = int(first.get("in", first.get("dim")))

    script_name = args.script_name or "champion_interact.py"
    script_path = out_dir / script_name
    script_path.write_text(
        _champion_script_template(
            default_model_name=model_name,
            default_dsl_name=(dsl_dst.name if dsl_dst is not None else ""),
            default_device=args.default_device,
        ),
        encoding="utf-8",
    )

    support_files = []
    for support_name in ("device_utils.py", "parser_utils.py", "network.py"):
        support_src = Path(support_name)
        if support_src.exists():
            support_dst = out_dir / support_name
            shutil.copy2(support_src, support_dst)
            support_files.append(support_name)

    run_bat = out_dir / "run_champion.bat"
    run_bat.write_text(
        f"@echo off\r\n{sys.executable} {script_name} --mode repl\r\n",
        encoding="utf-8",
    )

    manifest = {
        "name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_name,
        "dsl_file": dsl_dst.name if dsl_dst is not None else "",
        "script_file": script_name,
        "default_device": args.default_device,
        "expected_input_dim": expected_dim,
        "source_model_path": str(model_src),
        "source_dsl_path": args.dsl_file or "",
        "support_files": support_files,
    }
    if args.report_json:
        rp = Path(args.report_json)
        if rp.exists():
            try:
                manifest["report"] = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                manifest["report_path"] = str(rp)
    (out_dir / "champion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme_path = out_dir / "README_champion.txt"
    readme_path.write_text(
        "\n".join(
            [
                f"Champion Package: {args.name}",
                "",
                "Run with Python:",
                f"  {sys.executable} {script_name} --mode repl",
                "",
                "One-shot inference:",
                f"  {sys.executable} {script_name} --mode infer --input \"0.1,0.2,...\"",
                "",
                "Batch inference:",
                f"  {sys.executable} {script_name} --mode batch --input-csv inputs.csv --output-csv preds.csv",
                "",
                "Serve HTTP:",
                f"  {sys.executable} {script_name} --mode serve --host 127.0.0.1 --port 8092",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exe_path = None
    if args.build_exe:
        if args.install_pyinstaller:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        exe_path = _build_champion_exe(
            script_path=script_path,
            model_file=model_dst,
            dsl_file=dsl_dst,
            exe_name=args.exe_name or "ChampionModel",
        )

    print(f"[ok] champion package dir: {out_dir}")
    print(f"[ok] model: {model_dst}")
    if dsl_dst is not None:
        print(f"[ok] dsl: {dsl_dst}")
    print(f"[ok] interactive script: {script_path}")
    print(f"[ok] manifest: {out_dir / 'champion_manifest.json'}")
    if exe_path is not None:
        print(f"[ok] exe: {exe_path}")
    return 0


def cmd_tabular_train(args):
    from parser_utils import create_modern_nn

    dsl = _dsl_text(args)
    defs = parse_program(dsl)
    if not defs:
        raise ValueError("Failed to parse DSL.")

    model = create_modern_nn(defs)
    device = resolve_device(args.device)
    model = model.to(device)

    in_dim = defs[0].get("in", defs[0].get("dim"))
    out_dim = defs[-1].get("out", defs[-1].get("dim"))
    trainer = TrainingEngine(
        model,
        loss_fn=args.loss,
        max_epochs=args.epochs,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        aux_loss_coef=args.aux_loss_coef,
    )
    X, y = trainer.generate_dummy_data(in_dim, out_dim, n_samples=args.samples)
    X = X.to(device)
    y = y.to(device)

    for epoch in range(args.epochs):
        step_out = trainer.train_step(X, y)
        if isinstance(step_out, tuple):
            loss = float(step_out[0]) if len(step_out) > 0 else 0.0
            lr = float(step_out[1]) if len(step_out) > 1 else 0.0
            gn = float(step_out[2]) if len(step_out) > 2 else 0.0
        else:
            loss = float(step_out)
            lr = 0.0
            gn = 0.0
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs - 1:
            print(
                f"[train] epoch={epoch:04d} loss={loss:.6f} base={trainer.last_base_loss:.6f} "
                f"aux={trainer.last_aux_loss:.6f} lr={lr:.6f} grad={gn:.4f}"
            )

    if args.save_pth:
        torch.save(model.state_dict(), args.save_pth)
        print(f"[ok] saved weights: {args.save_pth}")
    if args.save_ts:
        trainer.export_torchscript(args.save_ts, in_dim)
        print(f"[ok] saved torchscript: {args.save_ts}")
    return 0


def cmd_tabular_run(args):
    passthrough = (
        f"--model {shlex.quote(args.model)} "
        f"{f'--dsl-file {shlex.quote(args.dsl_file)}' if args.dsl_file else ''} "
        f"{f'--dsl {shlex.quote(args.dsl)}' if args.dsl else ''} "
        f"{f'--input {shlex.quote(args.input)}' if args.input else ''} "
        f"{f'--input-csv {shlex.quote(args.input_csv)}' if args.input_csv else ''} "
        f"{f'--output-csv {shlex.quote(args.output_csv)}' if args.output_csv else ''} "
        f"{f'--output-json {shlex.quote(args.output_json)}' if args.output_json else ''} "
        f"--device {shlex.quote(args.device)} "
        f"{'--compile ' if args.compile_model else ''}"
        f"{'--as-probs ' if args.as_probs else ''}"
        f"{f'--random-samples {args.random_samples} ' if args.random_samples > 0 else ''}"
        f"--creative-samples {args.creative_samples} "
        f"--temperature {args.temperature} "
        f"--top-k {args.top_k} --top-p {args.top_p} "
        f"--mc-dropout-samples {args.mc_dropout_samples} "
        f"--benchmark-runs {args.benchmark_runs}"
    )
    if args.ensemble_model:
        passthrough += " " + " ".join(
            f"--ensemble-model {shlex.quote(path)}" for path in args.ensemble_model
        )
    return _run_existing_script("run_model.py", passthrough)


def _random_dsl(input_dim: int, output_dim: int, rng: random.Random) -> str:
    hidden = rng.choice([32, 48, 64, 96, 128, 192, 256])
    parts = [f"[{input_dim}, {hidden}]"]
    # Stochastic stack inspired by evolutionary architecture search.
    if rng.random() < 0.8:
        parts.append(f"moe: [{hidden}, {rng.choice([4, 6, 8, 10])}, {rng.choice([0, 1, 2])}]")
    if rng.random() < 0.7:
        parts.append(f"mod: [{hidden}, {rng.choice([2, 4, 6])}, {rng.choice([0.25, 0.35, 0.45])}]")
    if rng.random() < 0.6:
        parts.append(f"residual: [{hidden}, {rng.choice([2, 4])}]")
    if rng.random() < 0.5:
        parts.append(f"dropout: [{rng.choice([0.05, 0.1, 0.2, 0.3])}]")
    if rng.random() < 0.4:
        parts.append(f"gqa: [{hidden}, {rng.choice([4, 8])}, {rng.choice([1, 2])}]")
    parts.append(f"[{hidden}, {output_dim}]")
    return ", ".join(parts)


def cmd_tabular_search(args):
    from parser_utils import create_modern_nn

    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    best = None
    in_dim = int(args.input_dim)
    out_dim = int(args.output_dim)
    n_samples = int(args.samples)

    X = torch.randn(n_samples, in_dim, device=device)
    # Synthetic target family with mixed nonlinearities.
    y = torch.tanh(X[:, : min(in_dim, out_dim)])
    if y.shape[1] < out_dim:
        extra = torch.sin(X[:, [0]].repeat(1, out_dim - y.shape[1]))
        y = torch.cat([y, extra], dim=1)
    y = y[:, :out_dim]

    for trial in range(int(args.trials)):
        dsl = _random_dsl(in_dim, out_dim, rng)
        defs = parse_program(dsl)
        if not defs:
            continue
        model = create_modern_nn(defs).to(device)
        trainer = TrainingEngine(
            model,
            loss_fn=args.loss,
            max_epochs=args.epochs_per_trial,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            aux_loss_coef=args.aux_loss_coef,
        )
        last = None
        for _ in range(int(args.epochs_per_trial)):
            last, _, _ = trainer.train_step(X, y)
        score = float(last if last is not None else 1e9)
        print(f"[search] trial={trial:03d} loss={score:.6f} dsl={dsl}")
        if best is None or score < best["loss"]:
            best = {"loss": score, "dsl": dsl, "state_dict": model.state_dict()}

    if best is None:
        raise RuntimeError("No valid candidate architecture found.")

    Path(args.out_dsl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_dsl).write_text(best["dsl"], encoding="utf-8")
    print(f"[search] best_loss={best['loss']:.6f}")
    print(f"[search] best_dsl={best['dsl']}")
    print(f"[ok] wrote best dsl: {args.out_dsl}")

    if args.out_pth:
        Path(args.out_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best["state_dict"], args.out_pth)
        print(f"[ok] wrote best weights: {args.out_pth}")
    return 0


def _save_checkpoint(path: str, payload: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def cmd_image_train(args):
    device = resolve_device(args.device)
    model = ImageAutoencoder(image_size=args.image_size, latent_dim=args.latent_dim, channels=3)
    cfg = ExperimentalTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mixup_alpha=args.mixup_alpha,
        latent_noise=args.latent_noise,
        ema_decay=args.ema_decay,
        use_ema=not args.no_ema,
    )

    dataset = None
    if args.image_folder:
        dataset = load_images_from_folder(args.image_folder, image_size=args.image_size)

    history = train_image_autoencoder(model, cfg, device=device, dataset=dataset)
    for i, loss in enumerate(history):
        if i % max(1, args.log_every) == 0 or i == len(history) - 1:
            print(f"[image-train] epoch={i:04d} loss={loss:.6f}")

    payload = {
        "type": "image_autoencoder",
        "config": {"image_size": args.image_size, "latent_dim": args.latent_dim},
        "state_dict": model.state_dict(),
        "history": history,
    }
    _save_checkpoint(args.save_model, payload)
    print(f"[ok] saved image checkpoint: {args.save_model}")
    return 0


def cmd_image_generate(args):
    ckpt = _load_checkpoint(args.checkpoint)
    if ckpt.get("type") != "image_autoencoder":
        raise ValueError("Checkpoint is not an image_autoencoder checkpoint.")

    cfg = ckpt["config"]
    model = ImageAutoencoder(image_size=int(cfg["image_size"]), latent_dim=int(cfg["latent_dim"]), channels=3)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    device = resolve_device(args.device)
    imgs = generate_uncanny_images(
        model,
        n_samples=args.samples,
        chaos_strength=args.chaos_strength,
        seed=args.seed,
        device=device,
    )
    save_image_grid(imgs, args.output_grid, cols=args.cols)
    print(f"[ok] saved generated image grid: {args.output_grid}")
    return 0


def cmd_image_interpolate(args):
    ckpt = _load_checkpoint(args.checkpoint)
    if ckpt.get("type") != "image_autoencoder":
        raise ValueError("Checkpoint is not an image_autoencoder checkpoint.")

    cfg = ckpt["config"]
    model = ImageAutoencoder(image_size=int(cfg["image_size"]), latent_dim=int(cfg["latent_dim"]), channels=3)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    device = resolve_device(args.device)
    imgs = generate_interpolation_images(
        model,
        steps=args.steps,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        device=device,
    )
    save_image_grid(imgs, args.output_grid, cols=min(args.cols, max(1, imgs.shape[0])))
    print(f"[ok] saved interpolation grid: {args.output_grid}")
    return 0


def cmd_multimodal_train(args):
    device = resolve_device(args.device)
    model = MultiModalFusionModel(
        image_size=args.image_size,
        vec_dim=args.vec_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    cfg = ExperimentalTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_ema=False,
    )
    history = train_multimodal(model, cfg, device=device, dataset=None)
    for i, loss in enumerate(history):
        if i % max(1, args.log_every) == 0 or i == len(history) - 1:
            print(f"[multi-train] epoch={i:04d} loss={loss:.6f}")

    payload = {
        "type": "multimodal_fusion",
        "config": {
            "image_size": args.image_size,
            "vec_dim": args.vec_dim,
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
        },
        "state_dict": model.state_dict(),
        "history": history,
    }
    _save_checkpoint(args.save_model, payload)
    print(f"[ok] saved multimodal checkpoint: {args.save_model}")
    return 0


def cmd_multimodal_run(args):
    ckpt = _load_checkpoint(args.checkpoint)
    if ckpt.get("type") != "multimodal_fusion":
        raise ValueError("Checkpoint is not a multimodal_fusion checkpoint.")

    cfg = ckpt["config"]
    model = MultiModalFusionModel(
        image_size=int(cfg["image_size"]),
        vec_dim=int(cfg["vec_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        out_dim=int(cfg["out_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()

    if args.vector:
        vector = torch.tensor([float(v.strip()) for v in args.vector.split(",") if v.strip()], dtype=torch.float32)
    elif args.text:
        vector = text_to_feature_vector(args.text, int(cfg["vec_dim"]))
    else:
        vector = torch.randn(int(cfg["vec_dim"]))
    if vector.numel() != int(cfg["vec_dim"]):
        raise ValueError(f"vector must have {cfg['vec_dim']} values.")

    image = torch.rand(3, int(cfg["image_size"]), int(cfg["image_size"]))
    image = image.unsqueeze(0).to(device)
    vector = vector.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(image, vector).cpu().squeeze(0)
    print(f"[multi-run] output={out.tolist()}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"output": out.tolist()}, indent=2), encoding="utf-8")
        print(f"[ok] wrote {args.output_json}")
    return 0


def _run_check_command(name: str, cmd: list) -> dict:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "name": name,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def cmd_platform_health(args):
    checks = []
    checks.append(_run_check_command("compileall", [sys.executable, "-m", "compileall", "-q", "."]))
    checks.append(_run_check_command("verify", [sys.executable, "verify.py"]))
    if args.include_functional:
        checks.append(_run_check_command("functional", [sys.executable, "test_functional.py"]))
    if args.include_codex:
        if os.path.exists("verify_codex.py"):
            checks.append(_run_check_command("verify_codex", [sys.executable, "verify_codex.py"]))

    overall_ok = all(c["returncode"] == 0 for c in checks)
    report = {
        "overall_ok": overall_ok,
        "checks": checks,
    }
    payload = json.dumps(report, indent=2)
    _print_or_save(payload, args.output_json)
    if not overall_ok:
        raise RuntimeError("Platform health checks failed.")
    return 0


def cmd_export_bundle(args):
    output_zip = args.output_zip
    sources = [p for p in args.include if p]
    if not sources:
        raise ValueError("Provide at least one --include path.")

    Path(output_zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sources:
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for fname in files:
                        fp = os.path.join(root, fname)
                        arc = os.path.relpath(fp, start=".")
                        zf.write(fp, arcname=arc)
            elif os.path.exists(src):
                arc = os.path.relpath(src, start=".")
                zf.write(src, arcname=arc)
            else:
                print(f"[warn] missing path skipped: {src}")
    print(f"[ok] bundle exported: {output_zip}")
    return 0


def _load_tabular_model_for_server(model_path: str, device, dsl: str = "", dsl_file: str = ""):
    from parser_utils import create_modern_nn

    ext = os.path.splitext(model_path)[1].lower()
    layer_defs = None
    if ext == ".pth":
        text = dsl
        if dsl_file:
            text = Path(dsl_file).read_text(encoding="utf-8")
        if not text:
            raise ValueError(".pth server mode requires --dsl or --dsl-file")
        layer_defs = parse_program(text)
        if not layer_defs:
            raise ValueError("Failed to parse DSL.")
        model = create_modern_nn(layer_defs).to(device)
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        cleaned = {}
        for key, value in state.items():
            cleaned_key = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
            cleaned[cleaned_key] = value
        model.load_state_dict(cleaned, strict=False)
    elif ext == ".pt":
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    else:
        raise ValueError("Unsupported model format. Use .pth or .pt")

    model.eval()
    expected_dim = None
    if layer_defs:
        first = layer_defs[0]
        expected_dim = int(first.get("in", first.get("dim")))
    return model, expected_dim


def cmd_serve_tabular(args):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    device = resolve_device(args.device)
    model, expected_dim = _load_tabular_model_for_server(args.model, device, dsl=args.dsl, dsl_file=args.dsl_file)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send_json(200, {"status": "ok"})
                return
            if self.path.startswith("/meta"):
                self._send_json(200, {"expected_dim": expected_dim, "device": str(device)})
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            if not self.path.startswith("/infer"):
                self._send_json(404, {"error": "Not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            x = data.get("inputs")
            if x is None:
                self._send_json(400, {"error": "Missing 'inputs'"})
                return
            if not isinstance(x, list) or not x:
                self._send_json(400, {"error": "'inputs' must be a non-empty list"})
                return
            rows = x if isinstance(x[0], list) else [x]
            if expected_dim is not None:
                for row in rows:
                    if len(row) != expected_dim:
                        self._send_json(400, {"error": f"Each input row must have dim {expected_dim}"})
                        return
            tensor = torch.tensor(rows, dtype=torch.float32, device=device)
            with lock:
                with torch.no_grad():
                    out = model(tensor).cpu().tolist()
            self._send_json(200, {"outputs": out})

    host = args.host
    port = int(args.port)
    print(f"[server] starting http://{host}:{port}")
    print("[server] endpoints: GET /health, GET /meta, POST /infer")
    srv = ThreadingHTTPServer((host, port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[server] stopping...")
    finally:
        srv.server_close()
    return 0


def cmd_sim_generate_data(args):
    from simulation_lab import export_dataset_csv, generate_simulation_dataset

    dataset = generate_simulation_dataset(
        episodes=args.episodes,
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        max_steps=args.max_steps,
        policy_model=None,
    )
    csv_path = export_dataset_csv(dataset, args.output_csv)
    summary = {
        "rows": int(dataset.observations.shape[0]),
        "episodes": int(dataset.episodes),
        "success_rate": float(dataset.success_rate),
        "output_csv": csv_path,
    }
    payload = json.dumps(summary, indent=2)
    _print_or_save(payload, args.output_json)
    return 0


def cmd_sim_train_agent(args):
    from simulation_lab import train_agent_with_self_play

    report = train_agent_with_self_play(
        dsl=args.dsl or "",
        cycles=args.cycles,
        episodes_per_cycle=args.episodes_per_cycle,
        epochs_per_cycle=args.epochs_per_cycle,
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        max_steps=args.max_steps,
        device=args.device,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2)
    _print_or_save(payload, args.output_json)
    return 0


def cmd_platform_init(args):
    from platform_db import get_snapshot, init_db, seed_common_phrases

    db_path = init_db(args.db_path)
    seeded = 0
    if args.seed_phrases:
        seeded = seed_common_phrases(db_path=db_path)
    snapshot = get_snapshot(db_path=db_path)
    snapshot["db_path"] = db_path
    snapshot["seeded_phrases"] = seeded
    print(json.dumps(snapshot, indent=2))
    return 0


def cmd_account_create(args):
    from platform_db import create_account

    account = create_account(
        username=args.username,
        password=args.password,
        db_path=args.db_path,
        role=args.role,
        preferred_lang=args.lang,
    )
    print(json.dumps(account, indent=2))
    return 0


def cmd_account_auth(args):
    from platform_db import authenticate_account

    auth = authenticate_account(args.username, args.password, db_path=args.db_path)
    print(json.dumps(auth, indent=2))
    if not auth.get("ok"):
        raise RuntimeError("authentication failed")
    return 0


def cmd_project_create(args):
    from platform_db import create_project

    project = create_project(
        owner_username=args.owner,
        name=args.name,
        description=args.description,
        stack_json=args.stack_json,
        db_path=args.db_path,
    )
    print(json.dumps(project, indent=2))
    return 0


def cmd_project_list(args):
    from platform_db import list_projects

    rows = list_projects(db_path=args.db_path, owner_username=args.owner)
    print(json.dumps(rows, indent=2))
    return 0


def cmd_model_register(args):
    from platform_db import register_model

    record = register_model(
        project_name=args.project,
        owner_username=args.owner,
        model_name=args.name,
        checkpoint_path=args.checkpoint,
        dsl=args.dsl,
        metrics_json=args.metrics_json,
        agent_name=args.agent_name,
        db_path=args.db_path,
    )
    print(json.dumps(record, indent=2))
    return 0


def cmd_model_list(args):
    from platform_db import list_models

    rows = list_models(db_path=args.db_path, project_name=args.project, owner_username=args.owner)
    print(json.dumps(rows, indent=2))
    return 0


def cmd_events_sync(args):
    from internet_hub import extract_keyphrases, fetch_world_events
    from platform_db import add_events, bulk_upsert_phrases

    events = fetch_world_events(
        max_items=args.max_items,
        include_network=not args.offline,
        timeout=args.timeout,
    )
    inserted = add_events(events, db_path=args.db_path)
    phrase_updates = extract_keyphrases(
        [f"{e.get('title', '')} {e.get('summary', '')}".strip() for e in events],
        top_k=args.top_k_phrases,
    )
    bulk_upsert_phrases(phrase_updates, db_path=args.db_path)
    payload = {
        "events_seen": len(events),
        "events_inserted": inserted,
        "phrase_updates": len(phrase_updates),
        "offline_mode": bool(args.offline),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_phrase_list(args):
    from platform_db import list_phrases

    rows = list_phrases(db_path=args.db_path, language=args.language, limit=args.limit)
    print(json.dumps(rows, indent=2))
    return 0


def cmd_polyglot_scaffold(args):
    from polyglot_bridge import scaffold_polyglot_connectors

    manifest = scaffold_polyglot_connectors(out_dir=args.out_dir, base_url=args.base_url)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_agents_run(args):
    from project_agents import ModelAgent, ProjectRuntimeManager

    manager = ProjectRuntimeManager(db_path=args.db_path, max_workers=args.max_workers)
    names = [n.strip() for n in args.agents.split(",") if n.strip()]
    if not names:
        raise ValueError("Provide --agents as comma-separated names.")

    for name in names:
        manager.add_agent(ModelAgent(name=name, dsl=args.dsl))
    report = manager.run_all(
        project_name=args.project,
        owner_username=args.owner,
        out_root=args.out_root,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_console_app(args):
    cmd = [sys.executable, "console_app.py", "--db-path", args.db_path]
    if args.command:
        cmd.extend(["--cmd", args.command])
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_parser():
    p = argparse.ArgumentParser(description="NeuroDSL Omni CLI")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("devices", help="List available compute backends.")
    s.set_defaults(func=cmd_devices)

    s = sub.add_parser("download-model", help="Download a model from web URL or Hugging Face.")
    s.add_argument("--url", default="", help="Direct file URL.")
    s.add_argument("--output", default="downloads/model.bin", help="Output path for --url.")
    s.add_argument("--hf-repo", default="", help="Hugging Face repo id.")
    s.add_argument("--hf-file", default="", help="Hugging Face filename.")
    s.add_argument("--hf-dir", default="downloads", help="Local directory for Hugging Face downloads.")
    s.set_defaults(func=cmd_download_model)

    s = sub.add_parser("run-script", help="Run existing project scripts.")
    s.add_argument("--script", required=True, help="Known alias (main, verify, functional, run_model, omni_gui) or script path.")
    s.add_argument("--script-args", default="", help="Arguments forwarded to the script.")
    s.set_defaults(func=cmd_run_script)

    s = sub.add_parser("ai-connection", help="Test OpenAI connectivity.")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.set_defaults(func=cmd_ai_connection)

    s = sub.add_parser("ai-dsl", help="Generate DSL from natural language.")
    s.add_argument("--prompt", required=True)
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output-dsl", default="")
    s.set_defaults(func=cmd_ai_dsl)

    s = sub.add_parser("ai-explain", help="Explain a DSL architecture.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output", default="")
    s.set_defaults(func=cmd_ai_explain)

    s = sub.add_parser("ai-optimize", help="Optimize DSL with AI.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output-dsl", default="")
    s.set_defaults(func=cmd_ai_optimize)

    s = sub.add_parser("ai-hyperparams", help="Suggest training hyperparameters from DSL.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_ai_hyperparams)

    s = sub.add_parser("ai-codegen", help="Generate standalone PyTorch code from DSL.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output-py", default="")
    s.set_defaults(func=cmd_ai_codegen)

    s = sub.add_parser("ai-latency", help="Estimate hardware latency from DSL.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--hardware", default="NVIDIA A100")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output", default="")
    s.set_defaults(func=cmd_ai_latency)

    s = sub.add_parser("ai-diagram", help="Generate ASCII architecture diagram.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--output", default="")
    s.set_defaults(func=cmd_ai_diagram)

    s = sub.add_parser("ai-autopilot", help="Autonomous AI candidate generation + quick train-and-rank loop.")
    s.add_argument("--objective", required=True, help="Natural language objective for architecture search.")
    s.add_argument("--input-dim", type=int, default=32)
    s.add_argument("--output-dim", type=int, default=10)
    s.add_argument("--candidates", type=int, default=8, help="Number of candidates to evaluate.")
    s.add_argument("--epochs-per-candidate", type=int, default=30)
    s.add_argument("--samples", type=int, default=512)
    s.add_argument("--loss", default="MSE", choices=list(TrainingEngine.LOSS_FUNCTIONS.keys()))
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--warmup-steps", type=int, default=5)
    s.add_argument("--aux-loss-coef", type=float, default=0.02)
    s.add_argument("--param-penalty", type=float, default=0.002, help="Penalty factor on model size during selection.")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--device", default="auto")
    s.add_argument("--offline", action="store_true", help="Skip LLM generation and use random DSL candidates only.")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--out-dir", default="outputs/autopilot")
    s.set_defaults(func=cmd_ai_autopilot)

    s = sub.add_parser("ai-autopilot-sweep", help="Run multi-seed autopilot sweep and optionally promote/fine-tune champion.")
    s.add_argument("--objective", required=True, help="Natural language objective for architecture search.")
    s.add_argument("--input-dim", type=int, default=32)
    s.add_argument("--output-dim", type=int, default=10)
    s.add_argument("--candidates", type=int, default=12)
    s.add_argument("--epochs-per-candidate", type=int, default=3)
    s.add_argument("--samples", type=int, default=1024)
    s.add_argument("--loss", default="MSE", choices=list(TrainingEngine.LOSS_FUNCTIONS.keys()))
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--warmup-steps", type=int, default=5)
    s.add_argument("--aux-loss-coef", type=float, default=0.02)
    s.add_argument("--param-penalty", type=float, default=0.001)
    s.add_argument("--seeds", default="11,23,37,53", help="Comma-separated seed list.")
    s.add_argument("--device", default="auto")
    s.add_argument("--offline", action="store_true", help="Skip LLM generation and use random DSL candidates only.")
    s.add_argument("--api-key", default="", help="Optional API key (else OPENAI_API_KEY).")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--out-dir", default="outputs/autopilot_sweep")
    s.add_argument("--promote-best", action="store_true", help="Copy best seed artifacts to champion.* in out-dir.")
    s.add_argument("--fine-tune-epochs", type=int, default=0, help="If >0, run tabular-train on promoted champion DSL.")
    s.add_argument("--fine-tune-samples", type=int, default=4096)
    s.add_argument("--fine-tune-log-every", type=int, default=20)
    s.add_argument("--fine-tune-output", default="champion_final.pth")
    s.set_defaults(func=cmd_ai_autopilot_sweep)

    s = sub.add_parser("agent-manifest", help="Print machine-readable API capability manifest for AI agents.")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_agent_manifest)

    s = sub.add_parser("serve-agent-api", help="Serve local HTTP API for AI agents to validate, build, train, and infer.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_serve_agent_api)

    s = sub.add_parser("tabular-train", help="Train a DSL tabular model.")
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--epochs", type=int, default=200)
    s.add_argument("--samples", type=int, default=512)
    s.add_argument("--loss", default="MSE", choices=list(TrainingEngine.LOSS_FUNCTIONS.keys()))
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--warmup-steps", type=int, default=10)
    s.add_argument("--aux-loss-coef", type=float, default=0.02)
    s.add_argument("--log-every", type=int, default=10)
    s.add_argument("--device", default="auto")
    s.add_argument("--save-pth", default="outputs/tabular_model.pth")
    s.add_argument("--save-ts", default="")
    s.set_defaults(func=cmd_tabular_train)

    s = sub.add_parser("tabular-run", help="Run inference via run_model.py.")
    s.add_argument("--model", required=True)
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--input", default="")
    s.add_argument("--input-csv", default="")
    s.add_argument("--output-csv", default="")
    s.add_argument("--output-json", default="")
    s.add_argument("--random-samples", type=int, default=0)
    s.add_argument("--device", default="auto")
    s.add_argument("--compile-model", action="store_true")
    s.add_argument("--as-probs", action="store_true")
    s.add_argument("--ensemble-model", action="append", default=[], help="Additional model path(s) for ensemble averaging.")
    s.add_argument("--creative-samples", type=int, default=0)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--mc-dropout-samples", type=int, default=1)
    s.add_argument("--benchmark-runs", type=int, default=0)
    s.set_defaults(func=cmd_tabular_run)

    s = sub.add_parser("tabular-search", help="Neuro-evolution search for strong DSL candidates.")
    s.add_argument("--input-dim", type=int, required=True)
    s.add_argument("--output-dim", type=int, required=True)
    s.add_argument("--trials", type=int, default=24)
    s.add_argument("--epochs-per-trial", type=int, default=30)
    s.add_argument("--samples", type=int, default=512)
    s.add_argument("--loss", default="MSE", choices=list(TrainingEngine.LOSS_FUNCTIONS.keys()))
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--warmup-steps", type=int, default=5)
    s.add_argument("--aux-loss-coef", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--device", default="auto")
    s.add_argument("--out-dsl", default="outputs/search_best.dsl")
    s.add_argument("--out-pth", default="outputs/search_best.pth")
    s.set_defaults(func=cmd_tabular_search)

    s = sub.add_parser("serve-tabular", help="Serve a tabular model over HTTP.")
    s.add_argument("--model", required=True)
    s.add_argument("--dsl", default="")
    s.add_argument("--dsl-file", default="")
    s.add_argument("--device", default="auto")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_serve_tabular)

    s = sub.add_parser("platform-health", help="Run platform validation checks and emit a health report.")
    s.add_argument("--include-functional", action="store_true", help="Include test_functional.py check.")
    s.add_argument("--include-codex", action="store_true", help="Include verify_codex.py check if present.")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_platform_health)

    s = sub.add_parser("export-bundle", help="Export selected files/folders into a zip bundle.")
    s.add_argument("--include", nargs="+", required=True, help="Paths to include in the bundle.")
    s.add_argument("--output-zip", default="outputs/platform_bundle.zip")
    s.set_defaults(func=cmd_export_bundle)

    s = sub.add_parser("champion-package", help="Create distributable champion package (model + DSL + interactive script + optional EXE).")
    s.add_argument("--model", required=True, help="Path to .pth or .pt champion model file.")
    s.add_argument("--dsl", default="", help="DSL text (required for .pth when --dsl-file is absent).")
    s.add_argument("--dsl-file", default="", help="Path to DSL file.")
    s.add_argument("--report-json", default="", help="Optional JSON report to embed into manifest.")
    s.add_argument("--name", default="NeuroDSL Champion")
    s.add_argument("--output-dir", default="outputs/champion_package")
    s.add_argument("--model-name", default="", help="Output model filename inside package.")
    s.add_argument("--dsl-name", default="", help="Output DSL filename inside package.")
    s.add_argument("--script-name", default="", help="Output interactive script filename.")
    s.add_argument("--default-device", default="auto")
    s.add_argument("--build-exe", action="store_true", help="Build onefile EXE using PyInstaller.")
    s.add_argument("--install-pyinstaller", action="store_true", help="Auto-install pyinstaller before build.")
    s.add_argument("--exe-name", default="ChampionModel")
    s.set_defaults(func=cmd_champion_package)

    s = sub.add_parser("image-train", help="Train experimental image autoencoder.")
    s.add_argument("--image-folder", default="", help="Optional image folder. If omitted, synthetic data is used.")
    s.add_argument("--image-size", type=int, default=32)
    s.add_argument("--latent-dim", type=int, default=128)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--mixup-alpha", type=float, default=0.1)
    s.add_argument("--latent-noise", type=float, default=0.03)
    s.add_argument("--ema-decay", type=float, default=0.995)
    s.add_argument("--no-ema", action="store_true")
    s.add_argument("--log-every", type=int, default=5)
    s.add_argument("--device", default="auto")
    s.add_argument("--save-model", default="outputs/image_model.pth")
    s.set_defaults(func=cmd_image_train)

    s = sub.add_parser("image-generate", help="Generate uncanny images from image checkpoint.")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--samples", type=int, default=8)
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--chaos-strength", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--device", default="auto")
    s.add_argument("--output-grid", default="outputs/generated_grid.png")
    s.set_defaults(func=cmd_image_generate)

    s = sub.add_parser("image-interpolate", help="Generate latent interpolation grid from image checkpoint.")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--steps", type=int, default=8)
    s.add_argument("--seed-a", type=int, default=7)
    s.add_argument("--seed-b", type=int, default=19)
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--device", default="auto")
    s.add_argument("--output-grid", default="outputs/interpolation_grid.png")
    s.set_defaults(func=cmd_image_interpolate)

    s = sub.add_parser("multimodal-train", help="Train synthetic multimodal fusion model.")
    s.add_argument("--image-size", type=int, default=32)
    s.add_argument("--vec-dim", type=int, default=16)
    s.add_argument("--hidden-dim", type=int, default=128)
    s.add_argument("--out-dim", type=int, default=8)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--log-every", type=int, default=5)
    s.add_argument("--device", default="auto")
    s.add_argument("--save-model", default="outputs/multimodal_model.pth")
    s.set_defaults(func=cmd_multimodal_train)

    s = sub.add_parser("multimodal-run", help="Run multimodal checkpoint on a sample.")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--vector", default="", help="Optional comma-separated vector.")
    s.add_argument("--text", default="", help="Optional text prompt converted to a feature vector.")
    s.add_argument("--device", default="auto")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_multimodal_run)

    s = sub.add_parser("sim-generate-data", help="Generate simulation dataset using diamond-topology observations.")
    s.add_argument("--episodes", type=int, default=32)
    s.add_argument("--grid-size", type=int, default=12)
    s.add_argument("--obstacle-prob", type=float, default=0.12)
    s.add_argument("--max-steps", type=int, default=80)
    s.add_argument("--output-csv", default="outputs/sim_lab/sim_dataset.csv")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_sim_generate_data)

    s = sub.add_parser("sim-train-agent", help="Train a simulation agent with self-play + synthetic data loops.")
    s.add_argument("--dsl", default="", help="Optional DSL. Defaults to a diamond-layer simulation architecture.")
    s.add_argument("--cycles", type=int, default=4)
    s.add_argument("--episodes-per-cycle", type=int, default=24)
    s.add_argument("--epochs-per-cycle", type=int, default=20)
    s.add_argument("--grid-size", type=int, default=12)
    s.add_argument("--obstacle-prob", type=float, default=0.12)
    s.add_argument("--max-steps", type=int, default=80)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--device", default="auto")
    s.add_argument("--out-dir", default="outputs/sim_lab")
    s.add_argument("--output-json", default="")
    s.set_defaults(func=cmd_sim_train_agent)

    s = sub.add_parser("platform-init", help="Initialize account/project/model/event database.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--seed-phrases", action="store_true")
    s.set_defaults(func=cmd_platform_init)

    s = sub.add_parser("account-create", help="Create a platform account.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--username", required=True)
    s.add_argument("--password", required=True)
    s.add_argument("--role", default="user")
    s.add_argument("--lang", default="en")
    s.set_defaults(func=cmd_account_create)

    s = sub.add_parser("account-auth", help="Authenticate an account.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--username", required=True)
    s.add_argument("--password", required=True)
    s.set_defaults(func=cmd_account_auth)

    s = sub.add_parser("project-create", help="Create a project under an account owner.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--owner", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--description", default="")
    s.add_argument("--stack-json", default='["python","javascript","html","css"]')
    s.set_defaults(func=cmd_project_create)

    s = sub.add_parser("project-list", help="List projects.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--owner", default="")
    s.set_defaults(func=cmd_project_list)

    s = sub.add_parser("model-register", help="Register a trained model artifact to a project.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--owner", required=True)
    s.add_argument("--project", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--checkpoint", default="")
    s.add_argument("--dsl", default="")
    s.add_argument("--metrics-json", default="{}")
    s.add_argument("--agent-name", default="")
    s.set_defaults(func=cmd_model_register)

    s = sub.add_parser("model-list", help="List registered models.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--owner", default="")
    s.add_argument("--project", default="")
    s.set_defaults(func=cmd_model_list)

    s = sub.add_parser("events-sync", help="Sync world events and derive multilingual phrase candidates.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--max-items", type=int, default=50)
    s.add_argument("--top-k-phrases", type=int, default=80)
    s.add_argument("--timeout", type=float, default=8.0)
    s.add_argument("--offline", action="store_true", help="Use builtin events only (no network fetch).")
    s.set_defaults(func=cmd_events_sync)

    s = sub.add_parser("phrase-list", help="List phrase entries from database.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--language", default="")
    s.add_argument("--limit", type=int, default=40)
    s.set_defaults(func=cmd_phrase_list)

    s = sub.add_parser("polyglot-scaffold", help="Create HTML/CSS/JS/Python/Go/Java/C# connectors for model APIs.")
    s.add_argument("--out-dir", default="outputs/polyglot_bridge")
    s.add_argument("--base-url", default="http://127.0.0.1:8090")
    s.set_defaults(func=cmd_polyglot_scaffold)

    s = sub.add_parser("agents-run", help="Run multiple model agents and register outputs to project DB.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--owner", required=True)
    s.add_argument("--project", required=True)
    s.add_argument("--agents", default="alpha,beta")
    s.add_argument("--dsl", default="")
    s.add_argument("--device", default="auto")
    s.add_argument("--max-workers", type=int, default=2)
    s.add_argument("--out-root", default="outputs/agent_runtime")
    s.set_defaults(func=cmd_agents_run)

    s = sub.add_parser("console-app", help="Launch the integrated console application.")
    s.add_argument("--db-path", default="outputs/neuro_platform.db")
    s.add_argument("--command", default="", help='Optional one-shot console command, e.g. "status".')
    s.set_defaults(func=cmd_console_app)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        rc = args.func(args)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
