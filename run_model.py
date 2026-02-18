"""Run trained NeuroDSL models from the terminal.

Supports:
- .pth weights + DSL architecture reconstruction
- TorchScript (.pt) models
- Ensemble averaging across multiple checkpoints
- Single vector or CSV batch inference
- Optional compile() acceleration
- Optional MC-dropout uncertainty estimates
- Optional creative class sampling (temperature/top-k/top-p)
"""

import argparse
import csv
import json
import os
import time
from typing import List, Optional

import torch

from device_utils import resolve_device
from parser_utils import create_modern_nn, parse_program
from web_model_utils import download_file, download_from_hf


def _read_dsl_text(args: argparse.Namespace) -> Optional[str]:
    if args.dsl and args.dsl_file:
        raise ValueError("Use either --dsl or --dsl-file, not both.")
    if args.dsl:
        return args.dsl
    if args.dsl_file:
        with open(args.dsl_file, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _load_model(
    model_path: str,
    device,
    dsl_text: Optional[str],
    compile_model: bool,
):
    ext = os.path.splitext(model_path)[1].lower()
    layer_defs = None

    if ext == ".pth":
        if not dsl_text:
            raise ValueError("Loading .pth requires --dsl or --dsl-file.")
        layer_defs = parse_program(dsl_text)
        if not layer_defs:
            raise ValueError("Failed to parse DSL for .pth model reconstruction.")

        model = create_modern_nn(layer_defs)
        model = model.to(device)

        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Expected a state_dict in .pth file.")

        cleaned = {}
        for key, value in state.items():
            cleaned_key = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
            cleaned[cleaned_key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[warn] Missing keys while loading: {missing[:8]}")
        if unexpected:
            print(f"[warn] Unexpected keys while loading: {unexpected[:8]}")
    elif ext == ".pt":
        model = torch.jit.load(model_path, map_location="cpu")
    else:
        raise ValueError("Unsupported model format. Use .pth or .pt")

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[info] torch.compile enabled.")
        except Exception as exc:
            print(f"[warn] compile skipped: {exc}")

    model = model.to(device)
    model.eval()
    return model, layer_defs


def _parse_vector(raw: str) -> List[float]:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Input vector is empty.")
    return vals


def _load_input_rows(args: argparse.Namespace, expected_dim: Optional[int]) -> List[List[float]]:
    if args.input and args.input_csv:
        raise ValueError("Use either --input or --input-csv, not both.")

    rows: List[List[float]] = []
    if args.input:
        rows = [_parse_vector(args.input)]
    elif args.input_csv:
        with open(args.input_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            maybe_header = next(reader, None)
            if maybe_header:
                try:
                    [float(v.strip()) for v in maybe_header if v.strip()]
                    rows.append([float(v.strip()) for v in maybe_header if v.strip()])
                except ValueError:
                    pass
            for row in reader:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        rows.append(vals)
                except ValueError:
                    continue
    elif args.random_samples > 0:
        if expected_dim is None:
            raise ValueError("Random sampling requires known input dim. Provide --dsl for .pth.")
        rows = torch.randn(args.random_samples, expected_dim).tolist()
    else:
        raise ValueError("Provide --input, --input-csv, or --random-samples.")

    if expected_dim is not None:
        for idx, row in enumerate(rows):
            if len(row) != expected_dim:
                raise ValueError(
                    f"Input row {idx} has dim {len(row)} but model expects {expected_dim}."
                )
    return rows


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    out = logits.clone()

    if top_k > 0 and top_k < out.shape[-1]:
        kth = torch.topk(out, top_k, dim=-1).values[..., -1].unsqueeze(-1)
        out = torch.where(out < kth, torch.full_like(out, float("-inf")), out)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(out, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        out = torch.full_like(out, float("-inf"))
        out.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return out


def _creative_samples(
    logits: torch.Tensor,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[List[int]]:
    if num_samples <= 0:
        return []
    scaled = logits / max(temperature, 1e-5)
    scaled = _top_k_top_p_filter(scaled, top_k=max(0, top_k), top_p=min(1.0, max(0.0, top_p)))
    probs = torch.softmax(scaled, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    picks = []
    for _ in range(num_samples):
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        picks.append(sampled.cpu().tolist())

    by_row = []
    for row_idx in range(logits.shape[0]):
        by_row.append([picks[s_idx][row_idx] for s_idx in range(num_samples)])
    return by_row


def _run_forward(
    model,
    x: torch.Tensor,
    mc_dropout_samples: int,
):
    if mc_dropout_samples <= 1:
        model.eval()
        with torch.no_grad():
            preds = model(x)
        return preds, None

    draws = []
    model.train()
    with torch.no_grad():
        for _ in range(mc_dropout_samples):
            draws.append(model(x))
    model.eval()

    stacked = torch.stack(draws, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


def _run_ensemble_forward(
    models: List,
    x: torch.Tensor,
    mc_dropout_samples: int,
):
    if not models:
        raise ValueError("No models provided for ensemble forward.")

    mean_preds = []
    model_stds = []
    for model in models:
        pred, std = _run_forward(model, x, mc_dropout_samples=mc_dropout_samples)
        mean_preds.append(pred)
        if std is not None:
            model_stds.append(std)

    if len(mean_preds) == 1:
        return mean_preds[0], (model_stds[0] if model_stds else None)

    stacked = torch.stack(mean_preds, dim=0)
    ensemble_mean = stacked.mean(dim=0)
    ensemble_std = stacked.std(dim=0, unbiased=False)

    # Combine between-model variance with within-model MC-dropout variance when available.
    total_var = ensemble_std.pow(2)
    if model_stds:
        model_var = torch.stack([s.pow(2) for s in model_stds], dim=0).mean(dim=0)
        total_var = total_var + model_var
    return ensemble_mean, total_var.sqrt()


def _benchmark_ms(model, x: torch.Tensor, runs: int) -> Optional[float]:
    if runs <= 0:
        return None
    model.eval()
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
        elapsed = time.perf_counter() - t0
    return (elapsed / runs) * 1000.0


def _benchmark_ensemble_ms(models: List, x: torch.Tensor, runs: int) -> Optional[float]:
    if runs <= 0:
        return None
    if not models:
        return None
    for model in models:
        model.eval()
    with torch.no_grad():
        for _ in range(min(3, runs)):
            for model in models:
                _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            for model in models:
                _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return (elapsed / runs) * 1000.0


def _maybe_download_model(args: argparse.Namespace) -> str:
    if os.path.exists(args.model):
        return args.model

    if args.model_url:
        out = download_file(args.model_url, args.model)
        print(f"[info] downloaded model from URL to {out}")
        return out

    if args.hf_repo and args.hf_file:
        out = download_from_hf(args.hf_repo, args.hf_file, local_dir=args.hf_dir)
        print(f"[info] downloaded model from hf://{args.hf_repo}/{args.hf_file} to {out}")
        return out

    raise FileNotFoundError(
        f"Model path not found: {args.model}. Provide --model-url or --hf-repo + --hf-file."
    )


def _write_output_csv(
    output_csv: str,
    preds: torch.Tensor,
    mc_std: Optional[torch.Tensor],
    creative: List[List[int]],
):
    preds_cpu = preds.cpu()
    out_dim = preds_cpu.shape[1]
    header = [f"out_{i}" for i in range(out_dim)]

    include_std = mc_std is not None
    if include_std:
        header.extend([f"std_{i}" for i in range(out_dim)])

    creative_count = len(creative[0]) if creative else 0
    if creative_count > 0:
        header.extend([f"creative_class_{i}" for i in range(creative_count)])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in range(preds_cpu.shape[0]):
            row = preds_cpu[idx].tolist()
            if include_std:
                row.extend(mc_std[idx].cpu().tolist())
            if creative_count > 0:
                row.extend(creative[idx])
            writer.writerow(row)


def _write_output_json(
    output_json: str,
    preds: torch.Tensor,
    mc_std: Optional[torch.Tensor],
    creative: List[List[int]],
):
    payload = {
        "outputs": preds.cpu().tolist(),
        "uncertainty_std": mc_std.cpu().tolist() if mc_std is not None else None,
        "creative_samples": creative if creative else None,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run trained NeuroDSL models.")
    parser.add_argument("--model", required=True, help="Path to .pth or .pt model.")
    parser.add_argument("--model-url", default="", help="Optional URL to download model when --model path does not exist.")
    parser.add_argument("--hf-repo", default="", help="Optional Hugging Face repo id for model download.")
    parser.add_argument("--hf-file", default="", help="Optional Hugging Face filename for model download.")
    parser.add_argument("--hf-dir", default="downloads", help="Local dir for Hugging Face download.")
    parser.add_argument("--dsl", default="", help="DSL string (required for .pth unless --dsl-file is used).")
    parser.add_argument("--dsl-file", default="", help="Path to DSL file (alternative to --dsl).")
    parser.add_argument("--ensemble-model", action="append", default=[], help="Additional model path(s) for ensemble averaging.")
    parser.add_argument("--input", default="", help="Comma-separated input vector.")
    parser.add_argument("--input-csv", default="", help="CSV path for batch inference.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--random-samples", type=int, default=0, help="Generate random inputs when no --input/--input-csv.")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps/xpu/npu/dml")
    parser.add_argument("--compile", action="store_true", dest="compile_model", help="Enable torch.compile when possible.")
    parser.add_argument("--mc-dropout-samples", type=int, default=1, help="MC-dropout forward passes (>=2 enables uncertainty stats).")
    parser.add_argument("--creative-samples", type=int, default=0, help="Sample class IDs from output logits.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--as-probs", action="store_true", help="Apply softmax to outputs before printing/saving.")
    parser.add_argument("--benchmark-runs", type=int, default=0, help="Optional speed benchmark loop count.")
    parser.add_argument("--print-limit", type=int, default=5, help="Max rows to print in terminal.")

    args = parser.parse_args()
    model_path = _maybe_download_model(args)
    device = resolve_device(args.device)
    dsl_text = _read_dsl_text(args)

    model_paths = [model_path] + [p for p in args.ensemble_model if p]
    models = []
    layer_defs = None
    for idx, path in enumerate(model_paths):
        model, defs = _load_model(
            model_path=path,
            device=device,
            dsl_text=dsl_text,
            compile_model=args.compile_model,
        )
        models.append(model)
        if idx == 0:
            layer_defs = defs

    expected_dim = None
    if layer_defs:
        first = layer_defs[0]
        expected_dim = first.get("in", first.get("dim"))

    input_rows = _load_input_rows(args, expected_dim=expected_dim)
    x = torch.tensor(input_rows, dtype=torch.float32, device=device)

    preds, mc_std = _run_ensemble_forward(models, x, mc_dropout_samples=max(1, args.mc_dropout_samples))
    if args.as_probs:
        preds = torch.softmax(preds, dim=-1)
    creative = _creative_samples(
        preds,
        num_samples=max(0, args.creative_samples),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    if len(models) > 1:
        bench_ms = _benchmark_ensemble_ms(models, x, runs=max(0, args.benchmark_runs))
    else:
        bench_ms = _benchmark_ms(models[0], x, runs=max(0, args.benchmark_runs))
    if bench_ms is not None:
        if len(models) > 1:
            print(f"[bench] ensemble latency: {bench_ms:.3f} ms/run for batch={x.shape[0]} models={len(models)}")
        else:
            print(f"[bench] mean latency: {bench_ms:.3f} ms/run for batch={x.shape[0]}")

    preds_cpu = preds.cpu()
    limit = max(1, args.print_limit)
    if len(models) > 1:
        print(f"[info] ensemble models: {len(models)}")
    print(f"[info] output shape: {tuple(preds_cpu.shape)}")
    for i in range(min(limit, preds_cpu.shape[0])):
        print(f"[{i}] out={preds_cpu[i].tolist()}")
        if mc_std is not None:
            print(f"    std={mc_std[i].cpu().tolist()}")
        if creative:
            print(f"    creative={creative[i]}")

    if args.output_csv:
        _write_output_csv(args.output_csv, preds, mc_std, creative)
        print(f"[info] wrote predictions to {args.output_csv}")
    if args.output_json:
        _write_output_json(args.output_json, preds, mc_std, creative)
        print(f"[info] wrote predictions to {args.output_json}")


if __name__ == "__main__":
    main()
