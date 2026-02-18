"""Interactive runner for a packaged NeuroDSL champion model.

Modes:
- repl: interactive shell for inference and benchmark
- infer: one-shot inference for a single vector
- compare/sweep/importance/sensmap/pipeline/interpolate/stability/stress/goalseek/counterfactual/pareto/portfolio/batchlab/drift/sentinel/cohort/watchtower/simlab/policylab/profile/autolab: advanced analysis modes
- batch: CSV batch inference
- benchmark: latency benchmark
- serve: lightweight HTTP inference API

Advanced options:
- Monte Carlo sampling (`mc_samples`) for uncertainty estimation
- Probability output mode (`as_probs`)
- Top-K extraction (`topk`)
- Robustness and trajectory analysis tools (`stability`, `interpolate`, `stress`)
- Goal-directed search (`goalseek`) for controlled feature optimization
- Counterfactual search (`counterfactual`) for minimal-change target flipping
- Pareto search (`pareto`) for multi-objective candidate exploration
- Portfolio search (`portfolio`) for ranking curated candidate sets with diversity control
- BatchLab search (`batchlab`) for batch-level scouting, ranking, and outlier-aware triage
- Drift search (`drift`) for reference-vs-current shift detection across input/output spaces
- Sentinel search (`sentinel`) for batch-level anomaly triage with entropy and uncertainty risk scores
- Cohort search (`cohort`) for predicted-class cohort profiling with risk-ranked group diagnostics
- Sensitivity map (`sensmap`) for stable feature-influence analysis under noise
- Runtime profiler and auto-orchestrated strategy synthesis (`profile`, `autolab`)
- Watchtower orchestration (`watchtower`) for unified drift/anomaly/cohort risk governance
- Scenario simulation lab (`simlab`) for risk-frontier and resilience stress testing
- Policy optimizer lab (`policylab`) for watchtower weight/threshold auto-tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from device_utils import resolve_device
from parser_utils import create_modern_nn, parse_program
from platform_db import (
    authenticate_account,
    authenticate_api_token,
    create_account,
    create_api_session,
    ensure_project,
    get_snapshot,
    init_db,
    list_accounts,
    list_api_sessions,
    list_model_runs,
    list_models,
    list_projects,
    log_model_run,
    register_model,
    revoke_api_token,
)


DEFAULT_MODEL = "champion_model.pth"
DEFAULT_DSL = "champion_model.dsl"
DEFAULT_DEVICE = "auto"
DEFAULT_DB = "champion_registry.db"
DEFAULT_REGISTRY_OWNER = "owner"
DEFAULT_REGISTRY_PROJECT = "power_champion"


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


def _normalize_feature_indices(value):
    if value is None:
        return None
    raw_items = value if isinstance(value, list) else str(value).split(",")
    out = []
    for raw in raw_items:
        try:
            idx = int(raw)
        except Exception:
            continue
        out.append(idx)
    return out if out else None


def _normalize_positive_int_list(value, defaults=None, low=1, high=8192):
    raw_items = []
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    elif isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = [value]

    out = []
    for raw in raw_items:
        try:
            iv = int(raw)
        except Exception:
            continue
        if iv < int(low) or iv > int(high):
            continue
        out.append(int(iv))

    if not out:
        dflt = defaults or []
        for raw in dflt:
            try:
                iv = int(raw)
            except Exception:
                continue
            if iv < int(low) or iv > int(high):
                continue
            out.append(int(iv))

    if not out:
        out = [int(low)]

    uniq = []
    seen = set()
    for iv in out:
        if iv in seen:
            continue
        seen.add(iv)
        uniq.append(int(iv))
    return uniq


def _normalize_non_negative_float_list(value, defaults=None, low=0.0, high=1e6):
    raw_items = []
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    elif isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = [value]

    out = []
    for raw in raw_items:
        try:
            fv = float(raw)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        if fv < float(low) or fv > float(high):
            continue
        out.append(float(fv))

    if not out:
        dflt = defaults or []
        for raw in dflt:
            try:
                fv = float(raw)
            except Exception:
                continue
            if not math.isfinite(fv):
                continue
            if fv < float(low) or fv > float(high):
                continue
            out.append(float(fv))

    if not out:
        out = [float(low)]

    uniq = []
    seen = set()
    for fv in out:
        key = round(float(fv), 12)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(fv))
    uniq.sort()
    return uniq


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("1", "true", "yes", "y", "on")
    return bool(value)


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
        cleaned = {}
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


def _topk_row(row, k):
    kk = max(1, min(int(k), len(row)))
    return [{"index": int(i), "value": float(v)} for i, v in sorted(enumerate(row), key=lambda it: it[1], reverse=True)[:kk]]


def _stats(outputs, stds):
    if not outputs:
        return {"rows": 0, "out_dim": 0}
    flat = [float(v) for row in outputs for v in row]
    flat_std = [float(v) for row in (stds or []) for v in row] if stds else []
    stats = {
        "rows": len(outputs),
        "out_dim": len(outputs[0]) if outputs and outputs[0] else 0,
        "min": min(flat),
        "max": max(flat),
        "mean": sum(flat) / max(1, len(flat)),
    }
    if flat_std:
        stats["uncertainty_mean"] = sum(flat_std) / max(1, len(flat_std))
        stats["uncertainty_max"] = max(flat_std)
    return stats


def _predict_tensor(model, x, mc_samples=1, as_probs=False):
    mc = max(1, int(mc_samples))
    prev_training = model.training

    if mc == 1:
        model.eval()
        with torch.no_grad():
            y = model(x)
            if as_probs:
                y = torch.softmax(y, dim=-1)
        model.train(prev_training)
        return y, torch.zeros_like(y)

    preds = []
    model.train(True)
    with torch.no_grad():
        for _ in range(mc):
            y = model(x)
            if as_probs:
                y = torch.softmax(y, dim=-1)
            preds.append(y.unsqueeze(0))
    stack = torch.cat(preds, dim=0)
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False)
    model.train(prev_training)
    return mean, std


def _infer_rows(model, rows, device, mc_samples=1, as_probs=False, topk=0):
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    out_t, std_t = _predict_tensor(model, x, mc_samples=mc_samples, as_probs=as_probs)
    out = out_t.cpu().tolist()
    std = std_t.cpu().tolist()
    payload = {
        "outputs": out,
        "stds": std,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
        "stats": _stats(out, std),
    }
    if int(topk) > 0:
        payload["topk"] = [_topk_row(row, int(topk)) for row in out]
    return payload


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


def _write_csv(path, outputs, stds=None):
    out_dim = len(outputs[0]) if outputs else 0
    has_std = bool(stds) and len(stds) == len(outputs) and len(stds[0]) == out_dim
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [f"out_{i}" for i in range(out_dim)]
        if has_std:
            cols += [f"std_{i}" for i in range(out_dim)]
        w.writerow(cols)
        for i, row in enumerate(outputs):
            vals = list(row)
            if has_std:
                vals.extend(stds[i])
            w.writerow(vals)


def _benchmark(model, rows, device, runs, mc_samples=1, as_probs=False):
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(min(3, runs)):
            _ = _predict_tensor(model, x, mc_samples=mc_samples, as_probs=as_probs)[0]
        if x.is_cuda:
            torch.cuda.synchronize()
        latencies = []
        t_total0 = time.perf_counter()
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = _predict_tensor(model, x, mc_samples=mc_samples, as_probs=as_probs)[0]
            if x.is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        elapsed = time.perf_counter() - t_total0
    avg = sum(latencies) / max(1, len(latencies))
    return {
        "runs": int(runs),
        "batch_size": int(x.shape[0]),
        "avg_ms": float(avg),
        "min_ms": float(min(latencies) if latencies else 0.0),
        "max_ms": float(max(latencies) if latencies else 0.0),
        "throughput_sps": float((runs * x.shape[0]) / max(elapsed, 1e-9)),
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _runtime_diagnostics(device, expected_dim, db_path):
    temp_dir = Path(tempfile.gettempdir())
    temp_ok = False
    temp_err = ""
    probe = temp_dir / "neurodsl_temp_probe.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        _ = probe.read_text(encoding="utf-8")
        temp_ok = True
    except Exception as exc:
        temp_err = str(exc)
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except Exception:
            pass

    temp_usage = {}
    try:
        du = shutil.disk_usage(temp_dir)
        temp_usage = {
            "total_gb": float(du.total / (1024 ** 3)),
            "used_gb": float(du.used / (1024 ** 3)),
            "free_gb": float(du.free / (1024 ** 3)),
        }
    except Exception:
        temp_usage = {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    return {
        "status": "ok",
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "script_dir": str(Path(__file__).resolve().parent),
        "device": str(device),
        "expected_dim": expected_dim,
        "temp_dir": str(temp_dir),
        "temp_write_ok": bool(temp_ok),
        "temp_write_error": temp_err,
        "temp_disk": temp_usage,
        "db_path": str(Path(db_path).resolve()),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }


def _resolve_target_index(values, target_index):
    if not values:
        return 0
    if target_index is None:
        return int(max(range(len(values)), key=lambda i: values[i]))
    try:
        idx = int(target_index)
    except Exception:
        idx = -1
    if idx < 0 or idx >= len(values):
        return int(max(range(len(values)), key=lambda i: values[i]))
    return idx


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / max(1, len(values)))
    var = float(sum((float(v) - mean) ** 2 for v in values) / max(1, len(values)))
    return mean, math.sqrt(max(0.0, var))


def _cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(dot / (na * nb))


def _tool_compare(
    model,
    device,
    expected_dim,
    input_a,
    input_b,
    mc_samples=1,
    as_probs=False,
    topk=5,
):
    rows = _validate_rows([input_a, input_b], expected_dim)
    pred = _infer_rows(
        model,
        rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(0, int(topk)),
    )
    out_a = [float(v) for v in pred["outputs"][0]]
    out_b = [float(v) for v in pred["outputs"][1]]
    std_a = [float(v) for v in pred["stds"][0]]
    std_b = [float(v) for v in pred["stds"][1]]
    delta = [float(b - a) for a, b in zip(out_a, out_b)]
    delta_std = [float(math.sqrt(sa * sa + sb * sb)) for sa, sb in zip(std_a, std_b)]
    k = max(1, min(int(topk), len(delta)))
    payload = {
        "input_a": rows[0],
        "input_b": rows[1],
        "output_a": out_a,
        "output_b": out_b,
        "std_a": std_a,
        "std_b": std_b,
        "delta": delta,
        "delta_std": delta_std,
        "metrics": {
            "l1_delta": float(sum(abs(v) for v in delta)),
            "l2_delta": float(math.sqrt(sum(v * v for v in delta))),
            "max_abs_delta": float(max(abs(v) for v in delta) if delta else 0.0),
            "cosine_similarity": _cosine_similarity(out_a, out_b),
            "mean_delta": float(sum(delta) / max(1, len(delta))),
        },
        "stats_delta": _stats([delta], [delta_std]),
        "topk_delta": _topk_row(delta, k),
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }
    if pred.get("topk"):
        payload["topk_a"] = pred["topk"][0] if len(pred["topk"]) > 0 else []
        payload["topk_b"] = pred["topk"][1] if len(pred["topk"]) > 1 else []
    return payload


def _tool_sweep(
    model,
    device,
    expected_dim,
    base_input,
    feature_indices=None,
    radius=0.3,
    steps=7,
    target_index=None,
    mc_samples=1,
    as_probs=False,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    base_pred = _infer_rows(
        model,
        [base_row],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    tgt = _resolve_target_index(base_out, target_index)

    dim = len(base_row)
    if feature_indices is None:
        indices = list(range(dim))
    else:
        indices = []
        for raw in feature_indices:
            try:
                idx = int(raw)
            except Exception:
                continue
            if 0 <= idx < dim:
                indices.append(idx)
        indices = sorted(set(indices))
    if not indices:
        raise ValueError("feature_indices is empty after validation")

    rad = abs(float(radius))
    st = max(1, int(steps))
    deltas = [0.0] if st == 1 else [(-rad + (2.0 * rad * i) / (st - 1)) for i in range(st)]

    trial_rows = []
    trial_meta = []
    for idx in indices:
        for d in deltas:
            row = list(base_row)
            row[idx] = float(row[idx] + d)
            trial_rows.append(row)
            trial_meta.append((idx, float(d), float(row[idx])))

    pred = _infer_rows(
        model,
        trial_rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    outputs = pred["outputs"]
    stds = pred["stds"]

    per_feature = {
        idx: {
            "feature_index": idx,
            "base_value": float(base_row[idx]),
            "best_delta": 0.0,
            "best_value": float(base_row[idx]),
            "best_score": -float("inf"),
            "best_uncertainty": 0.0,
            "worst_delta": 0.0,
            "worst_value": float(base_row[idx]),
            "worst_score": float("inf"),
            "worst_uncertainty": 0.0,
            "mean_score": 0.0,
            "mean_uncertainty": 0.0,
            "count": 0,
        }
        for idx in indices
    }

    global_best = {"score": -float("inf"), "feature_index": -1, "delta": 0.0, "value": 0.0}
    global_worst = {"score": float("inf"), "feature_index": -1, "delta": 0.0, "value": 0.0}

    for i, (idx, d, value) in enumerate(trial_meta):
        out = outputs[i]
        std = stds[i]
        score = float(out[tgt])
        unc = float(std[tgt]) if tgt < len(std) else 0.0
        feat = per_feature[idx]
        feat["count"] += 1
        feat["mean_score"] += score
        feat["mean_uncertainty"] += unc
        if score > feat["best_score"]:
            feat["best_score"] = score
            feat["best_delta"] = d
            feat["best_value"] = value
            feat["best_uncertainty"] = unc
        if score < feat["worst_score"]:
            feat["worst_score"] = score
            feat["worst_delta"] = d
            feat["worst_value"] = value
            feat["worst_uncertainty"] = unc
        if score > global_best["score"]:
            global_best = {"score": score, "feature_index": idx, "delta": d, "value": value}
        if score < global_worst["score"]:
            global_worst = {"score": score, "feature_index": idx, "delta": d, "value": value}

    summaries = []
    for idx in indices:
        feat = per_feature[idx]
        c = max(1, int(feat["count"]))
        feat["mean_score"] = float(feat["mean_score"] / c)
        feat["mean_uncertainty"] = float(feat["mean_uncertainty"] / c)
        feat["sensitivity"] = float(feat["best_score"] - feat["worst_score"])
        summaries.append(feat)
    summaries.sort(key=lambda it: abs(float(it.get("sensitivity", 0.0))), reverse=True)

    return {
        "base_input": base_row,
        "base_output": base_out,
        "base_uncertainty": base_std,
        "target_index": int(tgt),
        "base_target_score": float(base_out[tgt] if tgt < len(base_out) else 0.0),
        "radius": float(rad),
        "steps": int(st),
        "evaluated_rows": len(trial_rows),
        "feature_count": len(indices),
        "summaries": summaries,
        "recommended_adjustment": global_best,
        "worst_adjustment": global_worst,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_importance(
    model,
    device,
    expected_dim,
    input_vec,
    epsilon=0.01,
    target_index=None,
    mc_samples=1,
    as_probs=False,
    top_features=12,
):
    vec = _validate_rows([input_vec], expected_dim)[0]
    base_pred = _infer_rows(
        model,
        [vec],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    tgt = _resolve_target_index(base_out, target_index)
    eps = abs(float(epsilon))
    if eps <= 1e-8:
        eps = 1e-3

    rows = []
    for i in range(len(vec)):
        plus = list(vec)
        minus = list(vec)
        plus[i] += eps
        minus[i] -= eps
        rows.append(plus)
        rows.append(minus)

    pred = _infer_rows(
        model,
        rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    out_rows = pred["outputs"]
    std_rows = pred["stds"]

    feats = []
    max_abs = 1e-12
    for i in range(len(vec)):
        plus = out_rows[2 * i]
        minus = out_rows[2 * i + 1]
        score_plus = float(plus[tgt])
        score_minus = float(minus[tgt])
        grad = float((score_plus - score_minus) / (2.0 * eps))
        abs_grad = abs(grad)
        max_abs = max(max_abs, abs_grad)
        unc = float((std_rows[2 * i][tgt] + std_rows[2 * i + 1][tgt]) * 0.5) if tgt < len(std_rows[2 * i]) else 0.0
        feats.append(
            {
                "feature_index": i,
                "base_value": float(vec[i]),
                "score_plus": score_plus,
                "score_minus": score_minus,
                "importance": grad,
                "abs_importance": abs_grad,
                "uncertainty": unc,
            }
        )

    for f in feats:
        f["normalized_abs_importance"] = float(f["abs_importance"] / max_abs)

    feats.sort(key=lambda item: float(item["abs_importance"]), reverse=True)
    topn = max(1, min(int(top_features), len(feats)))
    return {
        "input": vec,
        "target_index": int(tgt),
        "base_target_score": float(base_out[tgt] if tgt < len(base_out) else 0.0),
        "epsilon": float(eps),
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
        "importance": feats,
        "top_features": feats[:topn],
    }


def _tool_sensmap(
    model,
    device,
    expected_dim,
    base_input,
    samples=16,
    noise_std=0.04,
    epsilon=0.01,
    target_index=None,
    mc_samples=1,
    as_probs=False,
    top_features=12,
    seed=None,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    dim = len(base_row)
    sample_count = max(1, int(samples))
    sigma = abs(float(noise_std))
    eps = abs(float(epsilon))
    if eps <= 1e-8:
        eps = 1e-3

    base_pred = _infer_rows(
        model,
        [base_row],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    tgt = _resolve_target_index(base_out, target_index)
    base_target_score = float(base_out[tgt] if tgt < len(base_out) else 0.0)

    gen = None
    if seed is not None:
        try:
            seed_i = int(seed)
        except Exception:
            seed_i = -1
        if seed_i >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed_i)

    rows = [list(base_row)]
    if sample_count > 1:
        if sigma <= 1e-12:
            for _ in range(sample_count - 1):
                rows.append(list(base_row))
        else:
            base_t = torch.tensor(base_row, dtype=torch.float32)
            noise = torch.randn((sample_count - 1, dim), generator=gen, dtype=torch.float32)
            noisy = (base_t.unsqueeze(0) + (noise * sigma)).tolist()
            rows.extend(noisy)

    sample_pred = _infer_rows(
        model,
        rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    sample_out = [[float(v) for v in row] for row in sample_pred["outputs"]]
    sample_std = [[float(v) for v in row] for row in sample_pred["stds"]]
    target_scores = [float(row[tgt] if tgt < len(row) else 0.0) for row in sample_out]
    target_mean, target_std = _mean_std(target_scores)

    sum_imp = [0.0 for _ in range(dim)]
    sum_abs = [0.0 for _ in range(dim)]
    sum_sq = [0.0 for _ in range(dim)]
    sum_unc = [0.0 for _ in range(dim)]
    sign_sum = [0.0 for _ in range(dim)]

    for row in rows:
        imp = _tool_importance(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=row,
            epsilon=float(eps),
            target_index=tgt,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            top_features=max(1, dim),
        )
        imp_rows = imp.get("importance", [])
        by_idx = {}
        for item in imp_rows:
            try:
                idx = int(item.get("feature_index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= dim:
                continue
            by_idx[idx] = {
                "importance": float(item.get("importance", 0.0)),
                "uncertainty": float(item.get("uncertainty", 0.0)),
            }

        for j in range(dim):
            val = float(by_idx.get(j, {}).get("importance", 0.0))
            unc = float(by_idx.get(j, {}).get("uncertainty", 0.0))
            sum_imp[j] += val
            sum_abs[j] += abs(val)
            sum_sq[j] += (val * val)
            sum_unc[j] += unc
            sign_sum[j] += (1.0 if val >= 0.0 else -1.0)

    n = max(1, len(rows))
    features = []
    max_abs_mean = 1e-12
    for j in range(dim):
        mean_imp = float(sum_imp[j] / n)
        abs_mean = float(sum_abs[j] / n)
        var = float(max(0.0, (sum_sq[j] / n) - (mean_imp * mean_imp)))
        std_imp = float(math.sqrt(var))
        sign_consistency = float(abs(sign_sum[j]) / n)
        mean_unc = float(sum_unc[j] / n)
        max_abs_mean = max(max_abs_mean, abs_mean)
        features.append(
            {
                "feature_index": int(j),
                "base_value": float(base_row[j]),
                "mean_importance": mean_imp,
                "mean_abs_importance": abs_mean,
                "std_importance": std_imp,
                "sign_consistency": sign_consistency,
                "mean_uncertainty": mean_unc,
            }
        )

    for item in features:
        item["normalized_abs_importance"] = float(item["mean_abs_importance"] / max_abs_mean)
        item["stability_score"] = float(
            item["normalized_abs_importance"] * (0.4 + 0.6 * item["sign_consistency"]) / max(1e-6, 1.0 + item["mean_uncertainty"])
        )

    features.sort(key=lambda x: float(x.get("mean_abs_importance", 0.0)), reverse=True)
    topn = max(1, min(int(top_features), len(features)))
    probe_size = float(max(0.01, min(0.5, max(sigma, eps) * 2.0)))
    probes = []
    for item in features[:topn]:
        direction = 1.0 if float(item.get("mean_importance", 0.0)) >= 0.0 else -1.0
        probes.append(
            {
                "action": "probe_feature_direction",
                "feature_index": int(item.get("feature_index", -1)),
                "delta": float(direction * probe_size),
                "reason": "high stable sensitivity",
                "confidence": float(item.get("sign_consistency", 0.0)),
            }
        )

    stable_count = int(
        sum(
            1
            for item in features
            if float(item.get("sign_consistency", 0.0)) >= 0.75 and float(item.get("normalized_abs_importance", 0.0)) >= 0.10
        )
    )
    preview = []
    for i in range(min(8, len(rows))):
        preview.append(
            {
                "sample_index": int(i),
                "target_score": float(target_scores[i] if i < len(target_scores) else 0.0),
                "target_uncertainty": float(sample_std[i][tgt] if i < len(sample_std) and tgt < len(sample_std[i]) else 0.0),
                "input": rows[i],
            }
        )

    return {
        "base_input": base_row,
        "base_output": base_out,
        "base_uncertainty": base_std,
        "target_index": int(tgt),
        "base_target_score": float(base_target_score),
        "samples": int(sample_count),
        "noise_std": float(sigma),
        "epsilon": float(eps),
        "summary": {
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "target_min": float(min(target_scores) if target_scores else 0.0),
            "target_max": float(max(target_scores) if target_scores else 0.0),
            "stable_feature_count": stable_count,
            "probe_size": probe_size,
        },
        "importance_map": features,
        "top_features": features[:topn],
        "recommended_probes": probes,
        "sample_preview": preview,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_pipeline(
    model,
    device,
    expected_dim,
    base_input,
    input_b=None,
    feature_indices=None,
    radius=0.3,
    steps=7,
    epsilon=0.01,
    target_index=None,
    mc_samples=1,
    as_probs=False,
    topk=5,
    top_features=12,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    if input_b is None:
        vec_b = list(base_row)
        if vec_b:
            vec_b[0] = float(vec_b[0] + abs(float(radius)))
    else:
        vec_b = _validate_rows([input_b], expected_dim)[0]

    compare = _tool_compare(
        model=model,
        device=device,
        expected_dim=expected_dim,
        input_a=base_row,
        input_b=vec_b,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
    )
    sweep = _tool_sweep(
        model=model,
        device=device,
        expected_dim=expected_dim,
        base_input=base_row,
        feature_indices=feature_indices,
        radius=float(radius),
        steps=max(1, int(steps)),
        target_index=target_index,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
    )
    importance = _tool_importance(
        model=model,
        device=device,
        expected_dim=expected_dim,
        input_vec=base_row,
        epsilon=float(epsilon),
        target_index=sweep.get("target_index"),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        top_features=max(1, int(top_features)),
    )

    best_adj = sweep.get("recommended_adjustment", {}) if isinstance(sweep, dict) else {}
    top_imp = (importance.get("top_features") or [{}])[0] if isinstance(importance, dict) else {}
    top_delta = (compare.get("topk_delta") or [{}])[0] if isinstance(compare, dict) else {}
    base_target_score = float(sweep.get("base_target_score", 0.0))
    best_target_score = float(best_adj.get("score", base_target_score))
    expected_gain = float(best_target_score - base_target_score)
    imp_value = float(top_imp.get("importance", 0.0))
    delta_std = [float(v) for v in compare.get("delta_std", [])]
    avg_unc = float(sum(delta_std) / max(1, len(delta_std)))
    signal = abs(expected_gain) + abs(float(top_imp.get("abs_importance", 0.0))) + abs(float(compare.get("metrics", {}).get("l2_delta", 0.0)))
    exploration_score = float(signal / max(1e-6, avg_unc + 1e-4))

    recommended_actions = []
    if isinstance(best_adj, dict) and best_adj.get("feature_index", -1) >= 0:
        recommended_actions.append(
            {
                "action": "adjust_feature",
                "feature_index": int(best_adj.get("feature_index", -1)),
                "delta": float(best_adj.get("delta", 0.0)),
                "to_value": float(best_adj.get("value", 0.0)),
                "reason": "sweep best target improvement",
            }
        )
    if isinstance(top_imp, dict) and top_imp.get("feature_index", -1) >= 0:
        imp_direction = 1.0 if imp_value >= 0.0 else -1.0
        imp_delta = max(0.01, min(abs(float(radius)) * 0.5, 0.5))
        recommended_actions.append(
            {
                "action": "probe_feature_direction",
                "feature_index": int(top_imp.get("feature_index", -1)),
                "delta": float(imp_direction * imp_delta),
                "reason": "importance gradient probe",
            }
        )
    if isinstance(top_delta, dict) and top_delta.get("index", -1) >= 0:
        recommended_actions.append(
            {
                "action": "watch_output_shift",
                "output_index": int(top_delta.get("index", -1)),
                "delta": float(top_delta.get("value", 0.0)),
                "reason": "largest compare delta",
            }
        )

    summary = {
        "target_index": int(sweep.get("target_index", 0)),
        "base_target_score": base_target_score,
        "best_target_score": best_target_score,
        "expected_gain": expected_gain,
        "top_sweep_feature": int(best_adj.get("feature_index", -1)),
        "top_importance_feature": int(top_imp.get("feature_index", -1)),
        "compare_l2_delta": float(compare.get("metrics", {}).get("l2_delta", 0.0)),
        "compare_cosine_similarity": float(compare.get("metrics", {}).get("cosine_similarity", 0.0)),
        "exploration_score": exploration_score,
        "recommended_action_count": len(recommended_actions),
    }

    return {
        "base_input": base_row,
        "input_b": vec_b,
        "summary": summary,
        "recommended_actions": recommended_actions,
        "compare": compare,
        "sweep": sweep,
        "importance": importance,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_interpolate(
    model,
    device,
    expected_dim,
    input_a,
    input_b,
    steps=9,
    target_index=None,
    mc_samples=1,
    as_probs=False,
    topk=5,
):
    vec_a = _validate_rows([input_a], expected_dim)[0]
    vec_b = _validate_rows([input_b], expected_dim)[0]
    st = max(2, int(steps))
    alphas = [float(i / max(1, st - 1)) for i in range(st)]
    rows = []
    for a in alphas:
        row = [(1.0 - a) * float(x) + a * float(y) for x, y in zip(vec_a, vec_b)]
        rows.append(row)

    pred = _infer_rows(
        model,
        rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(0, int(topk)),
    )
    outputs = [[float(v) for v in row] for row in pred["outputs"]]
    stds = [[float(v) for v in row] for row in pred["stds"]]
    if not outputs:
        raise ValueError("interpolate produced no outputs")

    tgt = _resolve_target_index(outputs[0], target_index)
    target_scores = [float(row[tgt]) if tgt < len(row) else 0.0 for row in outputs]
    target_unc = [float(row[tgt]) if tgt < len(row) else 0.0 for row in stds]
    target_mean, target_std = _mean_std(target_scores)
    unc_mean, unc_std = _mean_std(target_unc)

    jumps = []
    for i in range(max(0, len(outputs) - 1)):
        arow = outputs[i]
        brow = outputs[i + 1]
        dim = min(len(arow), len(brow))
        jump = math.sqrt(sum((float(brow[j]) - float(arow[j])) ** 2 for j in range(dim)))
        jumps.append(float(jump))
    path_length = float(sum(jumps))
    start_end = 0.0
    if outputs:
        dim = min(len(outputs[0]), len(outputs[-1]))
        start_end = float(math.sqrt(sum((float(outputs[-1][j]) - float(outputs[0][j])) ** 2 for j in range(dim))))
    max_jump = float(max(jumps) if jumps else 0.0)
    mean_jump = float(path_length / max(1, len(jumps)))
    smoothness = float(start_end / max(1e-9, path_length)) if path_length > 0 else 1.0

    samples = []
    for i, alpha in enumerate(alphas):
        item = {
            "step": i,
            "alpha": float(alpha),
            "input": rows[i],
            "output": outputs[i],
            "target_score": float(target_scores[i] if i < len(target_scores) else 0.0),
            "target_uncertainty": float(target_unc[i] if i < len(target_unc) else 0.0),
        }
        if pred.get("topk") and i < len(pred["topk"]):
            item["topk"] = pred["topk"][i]
        samples.append(item)

    return {
        "input_a": vec_a,
        "input_b": vec_b,
        "steps": int(st),
        "target_index": int(tgt),
        "target_scores": target_scores,
        "target_uncertainty": target_unc,
        "target_stats": {
            "mean": target_mean,
            "std": target_std,
            "uncertainty_mean": unc_mean,
            "uncertainty_std": unc_std,
            "min": float(min(target_scores) if target_scores else 0.0),
            "max": float(max(target_scores) if target_scores else 0.0),
        },
        "metrics": {
            "path_length_l2": path_length,
            "start_end_l2": start_end,
            "max_jump_l2": max_jump,
            "mean_jump_l2": mean_jump,
            "smoothness_ratio": smoothness,
        },
        "samples": samples,
        "start_topk": _topk_row(outputs[0], max(1, int(topk))),
        "end_topk": _topk_row(outputs[-1], max(1, int(topk))),
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_stability(
    model,
    device,
    expected_dim,
    input_vec,
    samples=64,
    noise_std=0.05,
    target_index=None,
    mc_samples=1,
    as_probs=False,
    topk=5,
    seed=None,
):
    vec = _validate_rows([input_vec], expected_dim)[0]
    sample_count = max(4, int(samples))
    sigma = abs(float(noise_std))

    base_pred = _infer_rows(
        model,
        [vec],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    tgt = _resolve_target_index(base_out, target_index)
    base_target = float(base_out[tgt] if tgt < len(base_out) else 0.0)

    gen = None
    if seed is not None:
        try:
            seed_i = int(seed)
        except Exception:
            seed_i = -1
        if seed_i >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed_i)

    if sigma <= 1e-12:
        noisy_rows = [list(vec) for _ in range(sample_count)]
    else:
        base_t = torch.tensor(vec, dtype=torch.float32)
        noise = torch.randn((sample_count, len(vec)), generator=gen, dtype=torch.float32)
        noisy_rows = (base_t.unsqueeze(0) + (noise * sigma)).tolist()

    pred = _infer_rows(
        model,
        noisy_rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    out_rows = [[float(v) for v in row] for row in pred["outputs"]]
    std_rows = [[float(v) for v in row] for row in pred["stds"]]
    if not out_rows:
        raise ValueError("stability produced no outputs")

    out_dim = len(out_rows[0])
    mean_output = []
    std_output = []
    mean_abs_delta = []
    for j in range(out_dim):
        col = [float(row[j]) for row in out_rows]
        m, s = _mean_std(col)
        mean_output.append(float(m))
        std_output.append(float(s))
        base_j = float(base_out[j] if j < len(base_out) else 0.0)
        mean_abs_delta.append(float(abs(m - base_j)))

    target_scores = [float(row[tgt]) if tgt < len(row) else 0.0 for row in out_rows]
    target_mean, target_std = _mean_std(target_scores)
    target_min = float(min(target_scores) if target_scores else 0.0)
    target_max = float(max(target_scores) if target_scores else 0.0)
    below_base_fraction = float(sum(1 for s in target_scores if s < base_target) / max(1, len(target_scores)))
    robust_score = float(max(0.0, 1.0 / (1.0 + target_std + abs(target_mean - base_target) + (sum(mean_abs_delta) / max(1, len(mean_abs_delta))))))

    worst = sorted(
        [{"index": int(i), "mean_abs_delta": float(v)} for i, v in enumerate(mean_abs_delta)],
        key=lambda item: item["mean_abs_delta"],
        reverse=True,
    )[:max(1, int(topk))]

    recommendation = "stable"
    if robust_score < 0.35:
        recommendation = "increase data augmentation and reduce feature noise"
    elif robust_score < 0.65:
        recommendation = "improve robustness with perturbation-aware training"

    sample_preview = []
    preview_n = min(8, len(noisy_rows))
    for i in range(preview_n):
        sample_preview.append(
            {
                "sample_index": int(i),
                "input": noisy_rows[i],
                "target_score": float(target_scores[i] if i < len(target_scores) else 0.0),
                "target_uncertainty": float(std_rows[i][tgt] if tgt >= 0 and tgt < len(std_rows[i]) else 0.0),
            }
        )

    return {
        "input": vec,
        "samples": int(sample_count),
        "noise_std": float(sigma),
        "target_index": int(tgt),
        "base_output": base_out,
        "base_uncertainty": base_std,
        "base_target_score": float(base_target),
        "mean_output": mean_output,
        "std_output": std_output,
        "mean_abs_delta": mean_abs_delta,
        "worst_output_shift": worst,
        "target_stats": {
            "mean": float(target_mean),
            "std": float(target_std),
            "min": target_min,
            "max": target_max,
            "below_base_fraction": below_base_fraction,
        },
        "robust_score": robust_score,
        "recommendation": recommendation,
        "sample_preview": sample_preview,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_stress(
    model,
    device,
    expected_dim,
    input_vec,
    noise_levels=None,
    samples=48,
    target_index=None,
    robust_threshold=0.5,
    mc_samples=1,
    as_probs=False,
    topk=5,
    seed=None,
):
    vec = _validate_rows([input_vec], expected_dim)[0]
    levels = _normalize_non_negative_float_list(
        noise_levels,
        defaults=[0.0, 0.01, 0.03, 0.05, 0.1],
        low=0.0,
        high=100.0,
    )
    sample_count = max(4, int(samples))
    threshold = max(0.0, min(1.0, float(robust_threshold)))

    resolved_target = target_index
    ladder = []
    first_breakdown = None
    prev_robust = None

    for sigma in levels:
        res = _tool_stability(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=vec,
            samples=sample_count,
            noise_std=float(sigma),
            target_index=resolved_target,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            topk=max(1, int(topk)),
            seed=seed,
        )
        if resolved_target is None:
            resolved_target = res.get("target_index")
        robust = float(res.get("robust_score", 0.0))
        tstats = res.get("target_stats", {})
        point = {
            "noise_std": float(sigma),
            "robust_score": robust,
            "target_mean": float(tstats.get("mean", 0.0)),
            "target_std": float(tstats.get("std", 0.0)),
            "below_base_fraction": float(tstats.get("below_base_fraction", 0.0)),
            "recommendation": str(res.get("recommendation", "")),
        }
        if prev_robust is not None:
            point["delta_robust"] = float(robust - prev_robust)
        prev_robust = robust
        if first_breakdown is None and robust < threshold:
            first_breakdown = float(sigma)
        ladder.append(point)

    if not ladder:
        raise ValueError("stress produced no levels")

    robust_values = [float(item.get("robust_score", 0.0)) for item in ladder]
    mean_robust = float(sum(robust_values) / max(1, len(robust_values)))
    min_robust = float(min(robust_values))
    max_robust = float(max(robust_values))

    xs = [float(item.get("noise_std", 0.0)) for item in ladder]
    ys = [float(item.get("robust_score", 0.0)) for item in ladder]
    slope = 0.0
    if len(xs) >= 2:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) * (x - mx) for x in xs)
        if abs(den) > 1e-12:
            slope = float(num / den)

    auc = mean_robust
    if len(xs) >= 2:
        area = 0.0
        for i in range(len(xs) - 1):
            width = float(max(0.0, xs[i + 1] - xs[i]))
            area += width * 0.5 * (ys[i] + ys[i + 1])
        span = float(max(1e-9, xs[-1] - xs[0]))
        auc = float(area / span)

    recommendation = "stress_profile_ready"
    if first_breakdown is not None and first_breakdown <= float(xs[len(xs) // 2]):
        recommendation = "prioritize noise-aware training and stronger augmentation"
    elif min_robust < 0.55:
        recommendation = "improve robustness for high-noise edge cases"
    elif slope < -0.9:
        recommendation = "robustness drops quickly with noise; tune regularization"
    elif auc >= 0.75:
        recommendation = "stable across tested noise envelope"

    return {
        "input": vec,
        "target_index": int(resolved_target if resolved_target is not None else 0),
        "noise_levels": levels,
        "samples_per_level": int(sample_count),
        "robust_threshold": float(threshold),
        "summary": {
            "mean_robust_score": mean_robust,
            "min_robust_score": min_robust,
            "max_robust_score": max_robust,
            "robust_auc": float(max(0.0, min(1.0, auc))),
            "robust_slope": float(slope),
            "first_breakdown_noise": first_breakdown,
            "recommendation": recommendation,
            "level_count": len(ladder),
        },
        "levels": ladder,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_goalseek(
    model,
    device,
    expected_dim,
    base_input,
    target_index=None,
    target_score=None,
    feature_indices=None,
    steps=12,
    step_size=0.05,
    radius=0.4,
    epsilon=0.01,
    top_features=8,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    dim = len(base_row)
    idxs = []
    if feature_indices is None:
        idxs = list(range(dim))
    else:
        for raw in feature_indices:
            try:
                i = int(raw)
            except Exception:
                continue
            if 0 <= i < dim:
                idxs.append(i)
        idxs = sorted(set(idxs))
    if not idxs:
        raise ValueError("goalseek feature_indices is empty after validation")

    target_goal = None
    try:
        if target_score is not None:
            tgs = float(target_score)
            if math.isfinite(tgs):
                target_goal = float(tgs)
    except Exception:
        target_goal = None

    iters = max(1, int(steps))
    base_step = abs(float(step_size))
    if base_step <= 1e-8:
        base_step = 0.01
    dyn_step = float(base_step)
    rad = abs(float(radius))
    eps = abs(float(epsilon))
    if eps <= 1e-8:
        eps = 1e-3
    topn = max(1, min(int(top_features), len(idxs)))

    base_pred = _infer_rows(
        model,
        [base_row],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    tgt = _resolve_target_index(base_out, target_index)
    base_score = float(base_out[tgt] if tgt < len(base_out) else 0.0)

    def objective(score: float) -> float:
        if target_goal is None:
            return float(score)
        return float(-abs(float(score) - float(target_goal)))

    current_row = list(base_row)
    current_out = list(base_out)
    current_score = float(base_score)
    current_obj = objective(current_score)

    best_row = list(current_row)
    best_out = list(current_out)
    best_score = float(current_score)
    best_obj = float(current_obj)
    best_std = list(base_std)
    accepted = 0
    history = []

    for step_i in range(iters):
        imp = _tool_importance(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=current_row,
            epsilon=float(eps),
            target_index=tgt,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            top_features=max(topn, len(idxs)),
        )
        imp_rows = imp.get("importance", [])
        imp_by_idx = {}
        for row in imp_rows:
            try:
                ii = int(row.get("feature_index", -1))
            except Exception:
                continue
            if ii in idxs:
                imp_by_idx[ii] = float(row.get("importance", 0.0))

        ranked = sorted(
            [{"feature_index": int(i), "importance": float(imp_by_idx.get(i, 0.0))} for i in idxs],
            key=lambda x: abs(float(x.get("importance", 0.0))),
            reverse=True,
        )[:topn]

        candidate = list(current_row)
        moves = []
        for rank, item in enumerate(ranked):
            idx = int(item.get("feature_index", -1))
            direction = 1.0 if float(item.get("importance", 0.0)) >= 0.0 else -1.0
            delta = float(direction * dyn_step / max(1.0, 1.0 + (0.15 * rank)))
            next_val = float(candidate[idx] + delta)
            if rad > 0.0:
                lo = float(base_row[idx] - rad)
                hi = float(base_row[idx] + rad)
                next_val = max(lo, min(hi, next_val))
            applied = float(next_val - candidate[idx])
            candidate[idx] = next_val
            moves.append(
                {
                    "feature_index": idx,
                    "importance": float(item.get("importance", 0.0)),
                    "delta": applied,
                    "new_value": float(next_val),
                }
            )

        pred = _infer_rows(
            model,
            [candidate],
            device,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            topk=0,
        )
        cand_out = [float(v) for v in pred["outputs"][0]]
        cand_std = [float(v) for v in pred["stds"][0]]
        cand_score = float(cand_out[tgt] if tgt < len(cand_out) else 0.0)
        cand_obj = objective(cand_score)
        accept = bool(cand_obj >= current_obj + 1e-12)

        history.append(
            {
                "step": int(step_i + 1),
                "accepted": bool(accept),
                "current_score": float(current_score),
                "candidate_score": float(cand_score),
                "current_objective": float(current_obj),
                "candidate_objective": float(cand_obj),
                "delta_objective": float(cand_obj - current_obj),
                "step_size": float(dyn_step),
                "moves": moves,
            }
        )

        if accept:
            accepted += 1
            current_row = candidate
            current_out = cand_out
            current_score = cand_score
            current_obj = cand_obj
            dyn_step = min(base_step * 1.6, max(base_step * 0.75, dyn_step * 1.05))
            if cand_obj > best_obj + 1e-12:
                best_obj = float(cand_obj)
                best_score = float(cand_score)
                best_row = list(candidate)
                best_out = list(cand_out)
                best_std = list(cand_std)
        else:
            dyn_step = max(base_step * 0.2, dyn_step * 0.75)

        if target_goal is not None and abs(current_score - float(target_goal)) <= max(1e-6, eps):
            break

    final_topk = _topk_row(best_out, max(1, int(topk)))
    return {
        "base_input": base_row,
        "optimized_input": best_row,
        "target_index": int(tgt),
        "target_score_goal": target_goal,
        "base_target_score": float(base_score),
        "optimized_target_score": float(best_score),
        "expected_gain": float(best_score - base_score),
        "objective_improvement": float(best_obj - objective(base_score)),
        "accepted_steps": int(accepted),
        "total_steps": int(len(history)),
        "radius": float(rad),
        "step_size": float(base_step),
        "epsilon": float(eps),
        "feature_indices": idxs,
        "top_features": int(topn),
        "history": history,
        "final_output": best_out,
        "final_uncertainty": best_std,
        "final_topk": final_topk,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }

def _tool_counterfactual(
    model,
    device,
    expected_dim,
    base_input,
    desired_index=None,
    feature_indices=None,
    steps=14,
    step_size=0.04,
    radius=0.35,
    epsilon=0.01,
    top_features=8,
    margin=0.02,
    l1_penalty=0.05,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    dim = len(base_row)
    idxs = []
    if feature_indices is None:
        idxs = list(range(dim))
    else:
        for raw in feature_indices:
            try:
                i = int(raw)
            except Exception:
                continue
            if 0 <= i < dim:
                idxs.append(i)
        idxs = sorted(set(idxs))
    if not idxs:
        raise ValueError("counterfactual feature_indices is empty after validation")

    iters = max(1, int(steps))
    base_step = abs(float(step_size))
    if base_step <= 1e-8:
        base_step = 0.01
    dyn_step = float(base_step)
    rad = abs(float(radius))
    eps = abs(float(epsilon))
    if eps <= 1e-8:
        eps = 1e-3
    topn = max(1, min(int(top_features), len(idxs)))
    margin_target = max(0.0, float(margin))
    penalty = max(0.0, float(l1_penalty))

    base_pred = _infer_rows(
        model,
        [base_row],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    out_dim = len(base_out)
    if out_dim <= 0:
        raise ValueError("counterfactual produced empty output")

    base_pred_idx = _resolve_target_index(base_out, None)
    if desired_index is None:
        ranked = sorted(range(out_dim), key=lambda i: float(base_out[i]), reverse=True)
        desired = int(ranked[1] if len(ranked) > 1 else ranked[0])
    else:
        try:
            desired = int(desired_index)
        except Exception:
            desired = -1
        if desired < 0 or desired >= out_dim:
            raise ValueError(f"desired_index must be in [0, {out_dim - 1}]")

    def _margin_objective(vec, out):
        desired_score = float(out[desired] if desired < len(out) else 0.0)
        if len(out) <= 1:
            competitor_idx = int(desired)
            competitor_score = 0.0
            margin_val = desired_score
        else:
            competitor_idx = int(max((i for i in range(len(out)) if i != desired), key=lambda i: out[i]))
            competitor_score = float(out[competitor_idx])
            margin_val = float(desired_score - competitor_score)
        l1 = float(sum(abs(float(vec[i]) - float(base_row[i])) for i in idxs) / max(1, len(idxs)))
        l2 = float(math.sqrt(sum((float(vec[i]) - float(base_row[i])) ** 2 for i in idxs)))
        objective = float(margin_val - (penalty * l1))
        return {
            "desired_score": desired_score,
            "competitor_index": competitor_idx,
            "competitor_score": competitor_score,
            "margin": margin_val,
            "distance_l1": l1,
            "distance_l2": l2,
            "objective": objective,
        }

    current_row = list(base_row)
    current_out = list(base_out)
    current_stats = _margin_objective(current_row, current_out)
    base_stats = dict(current_stats)

    best_row = list(current_row)
    best_out = list(current_out)
    best_std = list(base_std)
    best_stats = dict(current_stats)
    accepted = 0
    history = []

    def _build_importance_map(vec, tgt_idx):
        imp = _tool_importance(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=vec,
            epsilon=float(eps),
            target_index=int(tgt_idx),
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            top_features=max(1, len(idxs)),
        )
        out = {}
        for row in imp.get("importance", []):
            try:
                i = int(row.get("feature_index", -1))
            except Exception:
                continue
            if i in idxs:
                out[i] = float(row.get("importance", 0.0))
        return out

    for step_i in range(iters):
        imp_desired = _build_importance_map(current_row, desired)
        comp_idx = int(current_stats.get("competitor_index", desired))
        imp_comp = {}
        if out_dim > 1 and comp_idx != desired:
            imp_comp = _build_importance_map(current_row, comp_idx)

        ranked = sorted(
            [
                {
                    "feature_index": int(i),
                    "importance": float(imp_desired.get(i, 0.0) - imp_comp.get(i, 0.0)),
                }
                for i in idxs
            ],
            key=lambda x: abs(float(x.get("importance", 0.0))),
            reverse=True,
        )[:topn]

        candidate = list(current_row)
        moves = []
        for rank, item in enumerate(ranked):
            idx = int(item.get("feature_index", -1))
            direction = 1.0 if float(item.get("importance", 0.0)) >= 0.0 else -1.0
            delta = float(direction * dyn_step / max(1.0, 1.0 + (0.20 * rank)))
            next_val = float(candidate[idx] + delta)
            if rad > 0.0:
                lo = float(base_row[idx] - rad)
                hi = float(base_row[idx] + rad)
                next_val = max(lo, min(hi, next_val))
            applied = float(next_val - candidate[idx])
            candidate[idx] = next_val
            moves.append(
                {
                    "feature_index": idx,
                    "importance": float(item.get("importance", 0.0)),
                    "delta": applied,
                    "new_value": float(next_val),
                }
            )

        pred = _infer_rows(
            model,
            [candidate],
            device,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            topk=0,
        )
        cand_out = [float(v) for v in pred["outputs"][0]]
        cand_std = [float(v) for v in pred["stds"][0]]
        cand_stats = _margin_objective(candidate, cand_out)
        accept = bool(cand_stats["objective"] >= current_stats["objective"] + 1e-12)

        history.append(
            {
                "step": int(step_i + 1),
                "accepted": bool(accept),
                "competitor_index": int(comp_idx),
                "current_margin": float(current_stats["margin"]),
                "candidate_margin": float(cand_stats["margin"]),
                "current_objective": float(current_stats["objective"]),
                "candidate_objective": float(cand_stats["objective"]),
                "delta_objective": float(cand_stats["objective"] - current_stats["objective"]),
                "step_size": float(dyn_step),
                "moves": moves,
            }
        )

        if accept:
            accepted += 1
            current_row = candidate
            current_out = cand_out
            current_stats = cand_stats
            dyn_step = min(base_step * 1.8, max(base_step * 0.75, dyn_step * 1.07))
            if cand_stats["objective"] > best_stats["objective"] + 1e-12:
                best_row = list(candidate)
                best_out = list(cand_out)
                best_std = list(cand_std)
                best_stats = dict(cand_stats)
        else:
            dyn_step = max(base_step * 0.2, dyn_step * 0.75)

        current_pred = _resolve_target_index(current_out, None)
        margin_hit = (
            current_stats["margin"] >= margin_target
            if out_dim > 1
            else (current_stats["desired_score"] - float(base_out[0])) >= margin_target
        )
        if current_pred == desired and margin_hit:
            break

    final_pred_idx = _resolve_target_index(best_out, None)
    success = False
    if out_dim > 1:
        success = bool(final_pred_idx == desired and best_stats["margin"] >= margin_target)
    else:
        success = bool((best_stats["desired_score"] - float(base_out[0])) >= margin_target)

    deltas = []
    for idx in idxs:
        dv = float(best_row[idx] - base_row[idx])
        if abs(dv) <= 1e-12:
            continue
        deltas.append(
            {
                "feature_index": int(idx),
                "delta": dv,
                "base_value": float(base_row[idx]),
                "counterfactual_value": float(best_row[idx]),
            }
        )
    deltas.sort(key=lambda x: abs(float(x.get("delta", 0.0))), reverse=True)

    final_topk = _topk_row(best_out, max(1, int(topk)))
    return {
        "base_input": base_row,
        "counterfactual_input": best_row,
        "base_output": base_out,
        "counterfactual_output": best_out,
        "base_uncertainty": base_std,
        "counterfactual_uncertainty": best_std,
        "desired_index": int(desired),
        "base_predicted_index": int(base_pred_idx),
        "final_predicted_index": int(final_pred_idx),
        "margin_target": float(margin_target),
        "base_desired_score": float(base_stats["desired_score"]),
        "counterfactual_desired_score": float(best_stats["desired_score"]),
        "base_competitor_score": float(base_stats["competitor_score"]),
        "counterfactual_competitor_score": float(best_stats["competitor_score"]),
        "base_margin": float(base_stats["margin"]),
        "counterfactual_margin": float(best_stats["margin"]),
        "distance_l1": float(best_stats["distance_l1"]),
        "distance_l2": float(best_stats["distance_l2"]),
        "objective_improvement": float(best_stats["objective"] - base_stats["objective"]),
        "success": bool(success),
        "accepted_steps": int(accepted),
        "total_steps": int(len(history)),
        "radius": float(rad),
        "step_size": float(base_step),
        "epsilon": float(eps),
        "l1_penalty": float(penalty),
        "feature_indices": idxs,
        "top_features": int(topn),
        "changed_features": deltas[: max(1, int(topn))],
        "history": history,
        "final_topk": final_topk,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_pareto(
    model,
    device,
    expected_dim,
    base_input,
    target_index=None,
    target_score=None,
    feature_indices=None,
    samples=128,
    radius=0.3,
    sparsity=0.75,
    l1_penalty=0.05,
    uncertainty_penalty=0.10,
    top_candidates=12,
    topk=5,
    mc_samples=1,
    as_probs=False,
    seed=None,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    dim = len(base_row)
    idxs = []
    if feature_indices is None:
        idxs = list(range(dim))
    else:
        for raw in feature_indices:
            try:
                i = int(raw)
            except Exception:
                continue
            if 0 <= i < dim:
                idxs.append(i)
        idxs = sorted(set(idxs))
    if not idxs:
        raise ValueError("pareto feature_indices is empty after validation")

    sample_count = max(8, int(samples))
    rad = abs(float(radius))
    sparse = float(max(0.0, min(0.99, float(sparsity))))
    l1w = max(0.0, float(l1_penalty))
    uncw = max(0.0, float(uncertainty_penalty))
    topn = max(1, int(top_candidates))

    target_goal = None
    try:
        if target_score is not None:
            tgs = float(target_score)
            if math.isfinite(tgs):
                target_goal = float(tgs)
    except Exception:
        target_goal = None

    base_pred = _infer_rows(
        model,
        [base_row],
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    base_out = [float(v) for v in base_pred["outputs"][0]]
    base_std = [float(v) for v in base_pred["stds"][0]]
    tgt = _resolve_target_index(base_out, target_index)
    base_score = float(base_out[tgt] if tgt < len(base_out) else 0.0)
    base_unc = float(base_std[tgt] if tgt < len(base_std) else 0.0)

    gen = None
    if seed is not None:
        try:
            seed_i = int(seed)
        except Exception:
            seed_i = -1
        if seed_i >= 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed_i)

    rows = [list(base_row)]
    changes = [[]]
    random_count = max(0, sample_count - 1)
    if random_count > 0:
        if rad <= 1e-12:
            for _ in range(random_count):
                rows.append(list(base_row))
                changes.append([])
        else:
            noise = ((torch.rand((random_count, len(idxs)), generator=gen, dtype=torch.float32) * 2.0) - 1.0) * rad
            if sparse > 0.0:
                mask = (torch.rand((random_count, len(idxs)), generator=gen, dtype=torch.float32) >= sparse).float()
                noise = noise * mask
            for r in range(random_count):
                row = list(base_row)
                row_changes = []
                for j, idx in enumerate(idxs):
                    d = float(noise[r, j].item())
                    if abs(d) <= 1e-12:
                        continue
                    row[idx] = float(base_row[idx] + d)
                    row_changes.append(
                        {
                            "feature_index": int(idx),
                            "delta": float(d),
                            "value": float(row[idx]),
                        }
                    )
                row_changes.sort(key=lambda x: abs(float(x.get("delta", 0.0))), reverse=True)
                rows.append(row)
                changes.append(row_changes)

    pred = _infer_rows(
        model,
        rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    outs = [[float(v) for v in row] for row in pred["outputs"]]
    stds = [[float(v) for v in row] for row in pred["stds"]]

    def _objective(score: float) -> float:
        if target_goal is None:
            return float(score)
        return float(-abs(float(score) - float(target_goal)))

    cands = []
    for i in range(len(rows)):
        row = rows[i]
        out = outs[i] if i < len(outs) else []
        std = stds[i] if i < len(stds) else []
        score = float(out[tgt] if tgt < len(out) else 0.0)
        unc = float(std[tgt] if tgt < len(std) else 0.0)
        l1 = float(sum(abs(float(row[idx]) - float(base_row[idx])) for idx in idxs) / max(1, len(idxs)))
        l2 = float(math.sqrt(sum((float(row[idx]) - float(base_row[idx])) ** 2 for idx in idxs)))
        obj = _objective(score)
        utility = float(obj - (l1w * l1) - (uncw * unc))
        cands.append(
            {
                "candidate_index": int(i),
                "target_score": score,
                "objective": float(obj),
                "utility": utility,
                "uncertainty": unc,
                "distance_l1": l1,
                "distance_l2": l2,
                "predicted_index": int(_resolve_target_index(out, None)) if out else 0,
                "changed_count": int(len(changes[i]) if i < len(changes) else 0),
                "changes": (changes[i] if i < len(changes) else [])[: max(1, min(32, topn))],
                "input": row,
                "output": out,
            }
        )

    if not cands:
        raise ValueError("pareto produced no candidates")

    cands.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)

    pareto = []
    for cand in cands:
        dominated = False
        for other in cands:
            if other is cand:
                continue
            better_obj = float(other.get("objective", 0.0)) >= float(cand.get("objective", 0.0))
            better_l1 = float(other.get("distance_l1", 0.0)) <= float(cand.get("distance_l1", 0.0))
            better_unc = float(other.get("uncertainty", 0.0)) <= float(cand.get("uncertainty", 0.0))
            strictly = (
                float(other.get("objective", 0.0)) > float(cand.get("objective", 0.0))
                or float(other.get("distance_l1", 0.0)) < float(cand.get("distance_l1", 0.0))
                or float(other.get("uncertainty", 0.0)) < float(cand.get("uncertainty", 0.0))
            )
            if better_obj and better_l1 and better_unc and strictly:
                dominated = True
                break
        if not dominated:
            pareto.append(cand)

    pareto.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)
    best = cands[0]
    best_topk = _topk_row(best.get("output", []), max(1, int(topk)))
    base_topk = _topk_row(base_out, max(1, int(topk)))

    utilities = [float(c.get("utility", 0.0)) for c in cands]
    avg_utility = float(sum(utilities) / max(1, len(utilities)))
    avg_l1 = float(sum(float(c.get("distance_l1", 0.0)) for c in cands) / max(1, len(cands)))
    avg_unc = float(sum(float(c.get("uncertainty", 0.0)) for c in cands) / max(1, len(cands)))

    target_hit = None
    if target_goal is not None:
        threshold = max(1e-6, 0.01 * max(1.0, abs(target_goal)))
        target_hit = bool(abs(float(best.get("target_score", 0.0)) - float(target_goal)) <= threshold)

    return {
        "base_input": base_row,
        "target_index": int(tgt),
        "target_score_goal": target_goal,
        "base_target_score": float(base_score),
        "base_target_uncertainty": float(base_unc),
        "samples": int(sample_count),
        "radius": float(rad),
        "sparsity": float(sparse),
        "l1_penalty": float(l1w),
        "uncertainty_penalty": float(uncw),
        "top_candidates": int(topn),
        "feature_indices": idxs,
        "summary": {
            "best_utility": float(best.get("utility", 0.0)),
            "best_target_score": float(best.get("target_score", 0.0)),
            "best_distance_l1": float(best.get("distance_l1", 0.0)),
            "best_uncertainty": float(best.get("uncertainty", 0.0)),
            "improvement_vs_base": float(float(best.get("target_score", 0.0)) - base_score),
            "avg_utility": avg_utility,
            "avg_distance_l1": avg_l1,
            "avg_uncertainty": avg_unc,
            "pareto_count": int(len(pareto)),
            "target_hit": target_hit,
        },
        "best_candidate": {
            k: best.get(k)
            for k in (
                "candidate_index",
                "target_score",
                "objective",
                "utility",
                "uncertainty",
                "distance_l1",
                "distance_l2",
                "predicted_index",
                "changed_count",
                "changes",
                "input",
                "output",
            )
        },
        "top_list": cands[:topn],
        "pareto_front": pareto[:topn],
        "base_topk": base_topk,
        "best_topk": best_topk,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_portfolio(
    model,
    device,
    expected_dim,
    base_input,
    candidates,
    target_index=None,
    top_candidates=8,
    uncertainty_penalty=0.10,
    novelty_weight=0.15,
    diversity_weight=0.10,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    cand_rows = _validate_rows(candidates, expected_dim)
    if not cand_rows:
        raise ValueError("portfolio requires at least one candidate row")

    topn = max(1, int(top_candidates))
    uncw = max(0.0, float(uncertainty_penalty))
    novw = max(0.0, float(novelty_weight))
    divw = max(0.0, float(diversity_weight))

    pred = _infer_rows(
        model,
        [base_row] + cand_rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    outs = [[float(v) for v in row] for row in pred.get("outputs", [])]
    stds = [[float(v) for v in row] for row in pred.get("stds", [])]
    if not outs:
        raise ValueError("portfolio inference produced no outputs")

    base_out = outs[0]
    base_std = stds[0] if stds else [0.0 for _ in base_out]
    tgt = _resolve_target_index(base_out, target_index)
    base_score = float(base_out[tgt] if tgt < len(base_out) else 0.0)
    base_unc = float(base_std[tgt] if tgt < len(base_std) else 0.0)

    dim = max(1, len(base_row))

    def _dist_l1(a, b):
        return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / max(1, dim))

    def _dist_l2(a, b):
        return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / max(1, dim)))

    ranked = []
    for i, row in enumerate(cand_rows):
        out = outs[i + 1] if i + 1 < len(outs) else []
        std = stds[i + 1] if i + 1 < len(stds) else []
        score = float(out[tgt] if tgt < len(out) else 0.0)
        unc = float(std[tgt] if tgt < len(std) else 0.0)
        novelty_l1 = _dist_l1(row, base_row)
        novelty_l2 = _dist_l2(row, base_row)
        utility = float(score - (uncw * unc) + (novw * novelty_l2))
        ranked.append(
            {
                "candidate_index": int(i),
                "target_score": score,
                "uncertainty": unc,
                "novelty_l1": novelty_l1,
                "novelty_l2": novelty_l2,
                "utility": utility,
                "predicted_index": int(_resolve_target_index(out, None)) if out else 0,
                "input": row,
                "output": out,
            }
        )

    ranked.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)
    if not ranked:
        raise ValueError("portfolio produced no candidates")

    selected = []
    used = set()
    select_n = min(topn, len(ranked))
    while len(selected) < select_n:
        best_item = None
        best_score = -float("inf")
        best_bonus = 0.0
        best_min_dist = 0.0
        for cand in ranked:
            cid = int(cand.get("candidate_index", -1))
            if cid in used:
                continue
            min_dist = 0.0
            bonus = 0.0
            if selected:
                dists = [_dist_l2(cand.get("input", []), pick.get("input", [])) for pick in selected]
                min_dist = float(min(dists) if dists else 0.0)
                bonus = float(divw * min_dist)
            score = float(cand.get("utility", 0.0) + bonus)
            if score > best_score:
                best_score = score
                best_item = cand
                best_bonus = bonus
                best_min_dist = min_dist
        if best_item is None:
            break
        picked = dict(best_item)
        picked["selection_score"] = float(best_score)
        picked["diversity_bonus"] = float(best_bonus)
        picked["min_distance_to_selected"] = float(best_min_dist)
        selected.append(picked)
        used.add(int(best_item.get("candidate_index", -1)))

    best = ranked[0]
    best_topk = _topk_row(best.get("output", []), max(1, int(topk)))
    base_topk = _topk_row(base_out, max(1, int(topk)))

    avg_utility = float(sum(float(c.get("utility", 0.0)) for c in ranked) / max(1, len(ranked)))
    avg_target = float(sum(float(c.get("target_score", 0.0)) for c in ranked) / max(1, len(ranked)))
    avg_unc = float(sum(float(c.get("uncertainty", 0.0)) for c in ranked) / max(1, len(ranked)))
    selected_avg_utility = float(sum(float(c.get("utility", 0.0)) for c in selected) / max(1, len(selected)))
    selected_avg_score = float(sum(float(c.get("selection_score", 0.0)) for c in selected) / max(1, len(selected)))
    selected_avg_target = float(sum(float(c.get("target_score", 0.0)) for c in selected) / max(1, len(selected)))
    selected_avg_unc = float(sum(float(c.get("uncertainty", 0.0)) for c in selected) / max(1, len(selected)))

    pairwise_sum = 0.0
    pairwise_count = 0
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairwise_sum += _dist_l2(selected[i].get("input", []), selected[j].get("input", []))
            pairwise_count += 1
    selected_diversity = float(pairwise_sum / max(1, pairwise_count)) if pairwise_count > 0 else 0.0

    return {
        "base_input": base_row,
        "base_output": base_out,
        "base_uncertainty": base_std,
        "target_index": int(tgt),
        "base_target_score": float(base_score),
        "base_target_uncertainty": float(base_unc),
        "candidate_count": int(len(cand_rows)),
        "top_candidates": int(topn),
        "uncertainty_penalty": float(uncw),
        "novelty_weight": float(novw),
        "diversity_weight": float(divw),
        "summary": {
            "best_utility": float(best.get("utility", 0.0)),
            "best_target_score": float(best.get("target_score", 0.0)),
            "best_uncertainty": float(best.get("uncertainty", 0.0)),
            "improvement_vs_base": float(float(best.get("target_score", 0.0)) - base_score),
            "avg_utility": avg_utility,
            "avg_target_score": avg_target,
            "avg_uncertainty": avg_unc,
            "selected_count": int(len(selected)),
            "selected_avg_utility": selected_avg_utility,
            "selected_avg_selection_score": selected_avg_score,
            "selected_avg_target_score": selected_avg_target,
            "selected_avg_uncertainty": selected_avg_unc,
            "selected_diversity_l2": selected_diversity,
        },
        "best_candidate": {
            k: best.get(k)
            for k in (
                "candidate_index",
                "target_score",
                "uncertainty",
                "novelty_l1",
                "novelty_l2",
                "utility",
                "predicted_index",
                "input",
                "output",
            )
        },
        "ranking": ranked[:topn],
        "selected_portfolio": selected,
        "base_topk": base_topk,
        "best_topk": best_topk,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_batchlab(
    model,
    device,
    expected_dim,
    rows,
    base_input=None,
    target_index=None,
    top_rows=8,
    outlier_weight=0.20,
    centroid_weight=0.10,
    uncertainty_penalty=0.10,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    clean_rows = _validate_rows(rows, expected_dim)
    if not clean_rows:
        raise ValueError("batchlab requires at least one input row")

    if base_input is None:
        base_row = list(clean_rows[0])
    else:
        base_row = _validate_rows([base_input], expected_dim)[0]

    topn = max(1, int(top_rows))
    outw = max(0.0, float(outlier_weight))
    centw = max(0.0, float(centroid_weight))
    uncw = max(0.0, float(uncertainty_penalty))

    pred = _infer_rows(
        model,
        clean_rows,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    outs = [[float(v) for v in row] for row in pred.get("outputs", [])]
    stds = [[float(v) for v in row] for row in pred.get("stds", [])]
    if not outs:
        raise ValueError("batchlab inference produced no outputs")

    out_dim = len(outs[0]) if outs and outs[0] else 0
    if out_dim <= 0:
        raise ValueError("batchlab produced empty output vectors")

    if target_index is None:
        means = []
        for j in range(out_dim):
            means.append(float(sum(float(row[j]) for row in outs) / max(1, len(outs))))
        tgt = int(max(range(out_dim), key=lambda i: means[i]))
    else:
        tgt = _resolve_target_index(outs[0], target_index)

    dim = max(1, len(base_row))
    centroid = []
    for j in range(len(clean_rows[0])):
        centroid.append(float(sum(float(row[j]) for row in clean_rows) / max(1, len(clean_rows))))

    def _dist_l1(a, b):
        return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / max(1, dim))

    def _dist_l2(a, b):
        return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / max(1, dim)))

    target_scores = [float(row[tgt] if tgt < len(row) else 0.0) for row in outs]
    target_mean, target_std = _mean_std(target_scores)
    uncertainty_vals = []
    for row in stds:
        uncertainty_vals.append(float(row[tgt] if tgt < len(row) else 0.0))
    unc_mean, _ = _mean_std(uncertainty_vals)
    unc_max = float(max(uncertainty_vals) if uncertainty_vals else 0.0)

    ranking = []
    for i, row in enumerate(clean_rows):
        out = outs[i] if i < len(outs) else []
        std = stds[i] if i < len(stds) else []
        score = float(target_scores[i] if i < len(target_scores) else 0.0)
        unc = float(uncertainty_vals[i] if i < len(uncertainty_vals) else 0.0)
        novelty_base_l1 = _dist_l1(row, base_row)
        novelty_base_l2 = _dist_l2(row, base_row)
        novelty_centroid_l1 = _dist_l1(row, centroid)
        novelty_centroid_l2 = _dist_l2(row, centroid)
        z_score = float((score - target_mean) / target_std) if target_std > 1e-12 else 0.0
        outlier_score = float(abs(z_score))
        utility = float(score + (outw * outlier_score) + (centw * novelty_centroid_l2) - (uncw * unc))
        stability_score = float(score - unc)
        ranking.append(
            {
                "row_index": int(i),
                "target_score": score,
                "uncertainty": unc,
                "z_score": z_score,
                "outlier_score": outlier_score,
                "stability_score": stability_score,
                "novelty_to_base_l1": novelty_base_l1,
                "novelty_to_base_l2": novelty_base_l2,
                "novelty_to_centroid_l1": novelty_centroid_l1,
                "novelty_to_centroid_l2": novelty_centroid_l2,
                "utility": utility,
                "predicted_index": int(_resolve_target_index(out, None)) if out else 0,
                "input": row,
                "output": out,
            }
        )

    ranking.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)
    selected = ranking[:topn]
    stable = sorted(ranking, key=lambda x: float(x.get("stability_score", 0.0)), reverse=True)[:topn]
    explore = sorted(
        ranking,
        key=lambda x: (float(x.get("novelty_to_centroid_l2", 0.0)) - (uncw * float(x.get("uncertainty", 0.0)))),
        reverse=True,
    )[:topn]
    outliers = sorted(ranking, key=lambda x: float(x.get("outlier_score", 0.0)), reverse=True)[:topn]

    best = ranking[0]
    best_out = best.get("output", [])
    best_topk = _topk_row(best_out, max(1, int(topk)))

    centroid_novelty_mean = float(
        sum(float(item.get("novelty_to_centroid_l2", 0.0)) for item in ranking) / max(1, len(ranking))
    )
    best_idx = int(best.get("row_index", -1))
    outlier_idx = int(outliers[0].get("row_index", -1)) if outliers else -1
    stable_idx = int(stable[0].get("row_index", -1)) if stable else -1

    return {
        "base_input": base_row,
        "centroid_input": centroid,
        "target_index": int(tgt),
        "row_count": int(len(clean_rows)),
        "top_rows": int(topn),
        "outlier_weight": float(outw),
        "centroid_weight": float(centw),
        "uncertainty_penalty": float(uncw),
        "summary": {
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "target_min": float(min(target_scores) if target_scores else 0.0),
            "target_max": float(max(target_scores) if target_scores else 0.0),
            "uncertainty_mean": float(unc_mean),
            "uncertainty_max": float(unc_max),
            "centroid_novelty_mean_l2": centroid_novelty_mean,
            "best_row_index": int(best_idx),
            "best_utility": float(best.get("utility", 0.0)),
            "best_target_score": float(best.get("target_score", 0.0)),
            "outlier_row_index": int(outlier_idx),
            "stable_row_index": int(stable_idx),
            "selected_count": int(len(selected)),
        },
        "best_row": best,
        "ranking": ranking,
        "selected_rows": selected,
        "stable_rows": stable,
        "outlier_rows": outliers,
        "explore_rows": explore,
        "best_topk": best_topk,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_drift(
    model,
    device,
    expected_dim,
    current_rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    top_features=12,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    cur = _validate_rows(current_rows, expected_dim)
    if not cur:
        raise ValueError("drift requires non-empty current_rows")

    ref = None
    if reference_rows is not None:
        ref = _validate_rows(reference_rows, expected_dim)
    elif reference_input is not None:
        ref = _validate_rows([reference_input], expected_dim)
    else:
        ref = [list(cur[0])]
    if not ref:
        raise ValueError("drift requires non-empty reference rows")

    pred_ref = _infer_rows(
        model,
        ref,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    pred_cur = _infer_rows(
        model,
        cur,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )

    out_ref = [[float(v) for v in row] for row in pred_ref.get("outputs", [])]
    out_cur = [[float(v) for v in row] for row in pred_cur.get("outputs", [])]
    std_ref = [[float(v) for v in row] for row in pred_ref.get("stds", [])]
    std_cur = [[float(v) for v in row] for row in pred_cur.get("stds", [])]
    if not out_ref or not out_cur:
        raise ValueError("drift inference produced empty outputs")

    in_dim = len(cur[0])
    out_dim = len(out_cur[0]) if out_cur and out_cur[0] else 0
    if out_dim <= 0:
        raise ValueError("drift produced empty output vectors")

    def _mean_by_dim(rows):
        if not rows:
            return []
        d = len(rows[0])
        vals = []
        for j in range(d):
            vals.append(float(sum(float(r[j]) for r in rows) / max(1, len(rows))))
        return vals

    in_ref_mean = _mean_by_dim(ref)
    in_cur_mean = _mean_by_dim(cur)
    out_ref_mean = _mean_by_dim(out_ref)
    out_cur_mean = _mean_by_dim(out_cur)

    tgt = _resolve_target_index(out_ref_mean, target_index)

    target_ref = [float(row[tgt] if tgt < len(row) else 0.0) for row in out_ref]
    target_cur = [float(row[tgt] if tgt < len(row) else 0.0) for row in out_cur]
    target_ref_mean, target_ref_std = _mean_std(target_ref)
    target_cur_mean, target_cur_std = _mean_std(target_cur)
    target_delta = float(target_cur_mean - target_ref_mean)

    unc_ref_vals = [float(row[tgt] if tgt < len(row) else 0.0) for row in std_ref] if std_ref else [0.0]
    unc_cur_vals = [float(row[tgt] if tgt < len(row) else 0.0) for row in std_cur] if std_cur else [0.0]
    unc_ref_mean, _ = _mean_std(unc_ref_vals)
    unc_cur_mean, _ = _mean_std(unc_cur_vals)

    topn = max(1, min(int(top_features), in_dim))
    feature_shift = []
    for j in range(in_dim):
        ref_col = [float(r[j]) for r in ref]
        cur_col = [float(r[j]) for r in cur]
        ref_mean, ref_std = _mean_std(ref_col)
        cur_mean, cur_std = _mean_std(cur_col)
        delta = float(cur_mean - ref_mean)
        pooled = math.sqrt(max(1e-12, (ref_std * ref_std) + (cur_std * cur_std)))
        effect_size = float(delta / pooled) if pooled > 1e-12 else 0.0
        feature_shift.append(
            {
                "feature_index": int(j),
                "reference_mean": float(ref_mean),
                "current_mean": float(cur_mean),
                "delta": float(delta),
                "abs_delta": float(abs(delta)),
                "reference_std": float(ref_std),
                "current_std": float(cur_std),
                "effect_size": float(effect_size),
            }
        )
    feature_shift.sort(key=lambda x: float(x.get("abs_delta", 0.0)), reverse=True)

    output_shift = []
    out_l2 = 0.0
    for j in range(out_dim):
        delta = float(out_cur_mean[j] - out_ref_mean[j])
        out_l2 += delta * delta
        output_shift.append(
            {
                "output_index": int(j),
                "reference_mean": float(out_ref_mean[j]),
                "current_mean": float(out_cur_mean[j]),
                "delta": float(delta),
                "abs_delta": float(abs(delta)),
            }
        )
    output_shift.sort(key=lambda x: float(x.get("abs_delta", 0.0)), reverse=True)
    output_l2_delta = float(math.sqrt(max(0.0, out_l2)))

    mean_abs_feature_delta = float(
        sum(float(item.get("abs_delta", 0.0)) for item in feature_shift) / max(1, len(feature_shift))
    )

    def _to_prob(vec):
        arr = [float(v) for v in vec]
        if not arr:
            return [1.0]
        if as_probs:
            s = sum(max(0.0, v) for v in arr)
            if s > 1e-12:
                return [max(0.0, v) / s for v in arr]
        m = max(arr)
        ex = [math.exp(v - m) for v in arr]
        s = sum(ex)
        if s <= 1e-12:
            return [1.0 / len(arr) for _ in arr]
        return [v / s for v in ex]

    p_ref = _to_prob(out_ref_mean)
    p_cur = _to_prob(out_cur_mean)
    p_mix = [0.5 * (a + b) for a, b in zip(p_ref, p_cur)]

    def _kl(p, q):
        out = 0.0
        for a, b in zip(p, q):
            aa = max(1e-12, float(a))
            bb = max(1e-12, float(b))
            out += aa * math.log(aa / bb)
        return float(out)

    js_div = float(0.5 * _kl(p_ref, p_mix) + 0.5 * _kl(p_cur, p_mix))
    drift_score = float(abs(target_delta) + output_l2_delta + mean_abs_feature_delta + js_div + abs(unc_cur_mean - unc_ref_mean))

    recommendation = "monitor"
    if drift_score >= 0.75:
        recommendation = "high_drift_retrain_or_recalibrate"
    elif drift_score >= 0.30:
        recommendation = "moderate_drift_review_thresholds"

    return {
        "target_index": int(tgt),
        "reference_row_count": int(len(ref)),
        "current_row_count": int(len(cur)),
        "input_dim": int(in_dim),
        "output_dim": int(out_dim),
        "reference_input_mean": in_ref_mean,
        "current_input_mean": in_cur_mean,
        "reference_output_mean": out_ref_mean,
        "current_output_mean": out_cur_mean,
        "reference_topk": _topk_row(out_ref_mean, max(1, int(topk))),
        "current_topk": _topk_row(out_cur_mean, max(1, int(topk))),
        "summary": {
            "target_reference_mean": float(target_ref_mean),
            "target_current_mean": float(target_cur_mean),
            "target_delta": float(target_delta),
            "target_reference_std": float(target_ref_std),
            "target_current_std": float(target_cur_std),
            "uncertainty_reference_mean": float(unc_ref_mean),
            "uncertainty_current_mean": float(unc_cur_mean),
            "output_l2_delta": float(output_l2_delta),
            "mean_abs_feature_delta": float(mean_abs_feature_delta),
            "js_divergence": float(js_div),
            "drift_score": float(drift_score),
            "recommendation": recommendation,
        },
        "feature_shift": feature_shift,
        "top_feature_shift": feature_shift[:topn],
        "output_shift": output_shift,
        "top_output_shift": output_shift[: max(1, min(int(topn), out_dim))],
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_sentinel(
    model,
    device,
    expected_dim,
    rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    top_rows=8,
    uncertainty_weight=0.50,
    entropy_weight=1.00,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    cur = _validate_rows(rows, expected_dim)
    if not cur:
        raise ValueError("sentinel requires non-empty rows")

    use_self_reference = reference_rows is None and reference_input is None
    if reference_rows is not None:
        ref = _validate_rows(reference_rows, expected_dim)
    elif reference_input is not None:
        ref = _validate_rows([reference_input], expected_dim)
    else:
        ref = [list(r) for r in cur]
    if not ref:
        raise ValueError("sentinel requires non-empty reference rows")

    pred_cur = _infer_rows(
        model,
        cur,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    if use_self_reference:
        pred_ref = pred_cur
    else:
        pred_ref = _infer_rows(
            model,
            ref,
            device,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            topk=0,
        )

    out_cur = [[float(v) for v in row] for row in pred_cur.get("outputs", [])]
    std_cur = [[float(v) for v in row] for row in pred_cur.get("stds", [])]
    out_ref = [[float(v) for v in row] for row in pred_ref.get("outputs", [])]
    std_ref = [[float(v) for v in row] for row in pred_ref.get("stds", [])]
    if not out_cur:
        raise ValueError("sentinel inference produced empty outputs")
    if not out_ref:
        raise ValueError("sentinel reference inference produced empty outputs")

    out_dim = len(out_cur[0]) if out_cur and out_cur[0] else 0
    if out_dim <= 0:
        raise ValueError("sentinel produced empty output vectors")
    tgt = _resolve_target_index(
        [
            float(sum(float(row[j]) for row in out_ref) / max(1, len(out_ref)))
            for j in range(out_dim)
        ],
        target_index,
    )

    def _to_prob(row):
        arr = [float(v) for v in row]
        if not arr:
            return [1.0]
        if as_probs:
            total = sum(max(0.0, v) for v in arr)
            if total > 1e-12:
                return [max(0.0, v) / total for v in arr]
        m = max(arr)
        ex = [math.exp(v - m) for v in arr]
        s = sum(ex)
        if s <= 1e-12:
            return [1.0 / len(arr) for _ in arr]
        return [v / s for v in ex]

    def _entropy(row):
        p = _to_prob(row)
        if len(p) <= 1:
            return 0.0
        ent = 0.0
        for v in p:
            pv = max(1e-12, float(v))
            ent -= pv * math.log(pv)
        return float(ent / max(1e-12, math.log(len(p))))

    ref_target_vals = [float((row[tgt] if tgt < len(row) else 0.0)) for row in out_ref]
    ref_unc_vals = [float((row[tgt] if tgt < len(row) else 0.0)) for row in std_ref] if std_ref else [0.0]
    ref_entropy_vals = [float(_entropy(row)) for row in out_ref]
    ref_target_mean, ref_target_std = _mean_std(ref_target_vals)
    ref_unc_mean, ref_unc_std = _mean_std(ref_unc_vals)
    ref_entropy_mean, ref_entropy_std = _mean_std(ref_entropy_vals)
    target_den = max(float(ref_target_std), max(1e-3, abs(float(ref_target_mean)) * 0.05))
    unc_den = max(float(ref_unc_std), max(1e-4, abs(float(ref_unc_mean)) * 0.10 + 1e-4))
    entropy_den = max(float(ref_entropy_std), 1e-3)

    topn = max(1, int(top_rows))
    uncw = max(0.0, float(uncertainty_weight))
    entw = max(0.0, float(entropy_weight))

    ranking = []
    for i, row in enumerate(cur):
        out = out_cur[i] if i < len(out_cur) else []
        std = std_cur[i] if i < len(std_cur) else []
        target_score = float(out[tgt] if tgt < len(out) else 0.0)
        uncertainty = float(std[tgt] if tgt < len(std) else 0.0)
        entropy = float(_entropy(out))
        z_target = float((target_score - ref_target_mean) / max(1e-6, target_den))
        z_unc = float((uncertainty - ref_unc_mean) / max(1e-6, unc_den))
        z_entropy = float((entropy - ref_entropy_mean) / max(1e-6, entropy_den))
        anomaly_score = float(abs(z_target) + (uncw * max(0.0, z_unc)) + (entw * abs(z_entropy)))
        risk = "low"
        if anomaly_score >= 3.0 or z_target <= -2.5 or z_unc >= 2.5:
            risk = "critical"
        elif anomaly_score >= 2.0 or z_target <= -1.75 or z_unc >= 1.75:
            risk = "high"
        elif anomaly_score >= 1.0:
            risk = "medium"
        ranking.append(
            {
                "row_index": int(i),
                "target_score": float(target_score),
                "uncertainty": float(uncertainty),
                "entropy": float(entropy),
                "z_target": float(z_target),
                "z_uncertainty": float(z_unc),
                "z_entropy": float(z_entropy),
                "anomaly_score": float(anomaly_score),
                "risk_level": risk,
                "predicted_index": int(_resolve_target_index(out, None)) if out else 0,
                "input": row,
                "output": out,
            }
        )

    ranking.sort(key=lambda item: float(item.get("anomaly_score", 0.0)), reverse=True)
    stable_rows = sorted(ranking, key=lambda item: float(item.get("anomaly_score", 0.0)))[:topn]
    top_anomalies = ranking[:topn]

    anomalies = [float(item.get("anomaly_score", 0.0)) for item in ranking]
    anomaly_mean, anomaly_std = _mean_std(anomalies)
    anomaly_max = float(max(anomalies) if anomalies else 0.0)
    anomaly_min = float(min(anomalies) if anomalies else 0.0)

    critical_count = sum(1 for item in ranking if str(item.get("risk_level", "")) == "critical")
    high_count = sum(1 for item in ranking if str(item.get("risk_level", "")) == "high")
    medium_count = sum(1 for item in ranking if str(item.get("risk_level", "")) == "medium")
    low_count = max(0, len(ranking) - critical_count - high_count - medium_count)
    high_risk_count = critical_count + high_count
    high_risk_fraction = float(high_risk_count / max(1, len(ranking)))

    recommendation = "normal_monitoring"
    if high_risk_fraction >= 0.30 or anomaly_max >= 4.0:
        recommendation = "activate_guardrails_and_refresh_baseline"
    elif high_risk_fraction >= 0.12 or anomaly_mean >= 1.20:
        recommendation = "monitor_closely_collect_feedback"

    worst = ranking[0] if ranking else {}
    worst_out = worst.get("output", []) if isinstance(worst, dict) else []

    return {
        "target_index": int(tgt),
        "row_count": int(len(cur)),
        "reference_row_count": int(len(ref)),
        "top_rows": int(topn),
        "uncertainty_weight": float(uncw),
        "entropy_weight": float(entw),
        "reference_stats": {
            "target_mean": float(ref_target_mean),
            "target_std": float(ref_target_std),
            "uncertainty_mean": float(ref_unc_mean),
            "uncertainty_std": float(ref_unc_std),
            "entropy_mean": float(ref_entropy_mean),
            "entropy_std": float(ref_entropy_std),
        },
        "summary": {
            "anomaly_mean": float(anomaly_mean),
            "anomaly_std": float(anomaly_std),
            "anomaly_max": float(anomaly_max),
            "anomaly_min": float(anomaly_min),
            "critical_count": int(critical_count),
            "high_count": int(high_count),
            "medium_count": int(medium_count),
            "low_count": int(low_count),
            "high_risk_count": int(high_risk_count),
            "high_risk_fraction": float(high_risk_fraction),
            "recommendation": recommendation,
        },
        "top_anomalies": top_anomalies,
        "stable_rows": stable_rows,
        "ranking": ranking,
        "worst_topk": _topk_row(worst_out, max(1, int(topk))) if worst_out else [],
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_cohort(
    model,
    device,
    expected_dim,
    rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    top_groups=6,
    uncertainty_weight=0.50,
    entropy_weight=0.50,
    margin_weight=0.30,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    cur = _validate_rows(rows, expected_dim)
    if not cur:
        raise ValueError("cohort requires non-empty rows")

    use_self_reference = reference_rows is None and reference_input is None
    if reference_rows is not None:
        ref = _validate_rows(reference_rows, expected_dim)
    elif reference_input is not None:
        ref = _validate_rows([reference_input], expected_dim)
    else:
        ref = [list(r) for r in cur]
    if not ref:
        raise ValueError("cohort requires non-empty reference rows")

    pred_cur = _infer_rows(
        model,
        cur,
        device,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=0,
    )
    if use_self_reference:
        pred_ref = pred_cur
    else:
        pred_ref = _infer_rows(
            model,
            ref,
            device,
            mc_samples=max(1, int(mc_samples)),
            as_probs=bool(as_probs),
            topk=0,
        )

    out_cur = [[float(v) for v in row] for row in pred_cur.get("outputs", [])]
    std_cur = [[float(v) for v in row] for row in pred_cur.get("stds", [])]
    out_ref = [[float(v) for v in row] for row in pred_ref.get("outputs", [])]
    if not out_cur:
        raise ValueError("cohort inference produced empty outputs")
    if not out_ref:
        raise ValueError("cohort reference inference produced empty outputs")
    out_dim = len(out_cur[0]) if out_cur and out_cur[0] else 0
    if out_dim <= 0:
        raise ValueError("cohort produced empty output vectors")

    ref_mean = []
    for j in range(out_dim):
        ref_mean.append(float(sum(float(row[j]) for row in out_ref) / max(1, len(out_ref))))
    tgt = _resolve_target_index(ref_mean, target_index)

    def _to_prob(row):
        arr = [float(v) for v in row]
        if not arr:
            return [1.0]
        if as_probs:
            total = sum(max(0.0, v) for v in arr)
            if total > 1e-12:
                return [max(0.0, v) / total for v in arr]
        m = max(arr)
        ex = [math.exp(v - m) for v in arr]
        s = sum(ex)
        if s <= 1e-12:
            return [1.0 / len(arr) for _ in arr]
        return [v / s for v in ex]

    def _entropy(row):
        p = _to_prob(row)
        if len(p) <= 1:
            return 0.0
        ent = 0.0
        for v in p:
            pv = max(1e-12, float(v))
            ent -= pv * math.log(pv)
        return float(ent / max(1e-12, math.log(len(p))))

    def _margin(row):
        vals = [float(v) for v in row]
        if not vals:
            return 0.0
        if len(vals) == 1:
            return float(vals[0])
        top2 = sorted(vals, reverse=True)[:2]
        return float(top2[0] - top2[1])

    target_vals = [float(row[tgt] if tgt < len(row) else 0.0) for row in out_cur]
    unc_vals = [float((std_cur[i][tgt] if i < len(std_cur) and tgt < len(std_cur[i]) else 0.0)) for i in range(len(out_cur))]
    ent_vals = [float(_entropy(row)) for row in out_cur]
    margin_vals = [float(_margin(row)) for row in out_cur]

    g_target_mean, g_target_std = _mean_std(target_vals)
    g_unc_mean, g_unc_std = _mean_std(unc_vals)
    g_entropy_mean, g_entropy_std = _mean_std(ent_vals)
    g_margin_mean, g_margin_std = _mean_std(margin_vals)

    t_den = max(float(g_target_std), max(1e-3, abs(float(g_target_mean)) * 0.05))
    u_den = max(float(g_unc_std), max(1e-4, abs(float(g_unc_mean)) * 0.10 + 1e-4))
    e_den = max(float(g_entropy_std), 1e-3)
    m_den = max(float(g_margin_std), max(1e-4, abs(float(g_margin_mean)) * 0.10 + 1e-4))

    uncw = max(0.0, float(uncertainty_weight))
    entw = max(0.0, float(entropy_weight))
    marw = max(0.0, float(margin_weight))
    topn = max(1, int(top_groups))

    row_records = []
    for i, row in enumerate(cur):
        out = out_cur[i] if i < len(out_cur) else []
        score = float(target_vals[i] if i < len(target_vals) else 0.0)
        unc = float(unc_vals[i] if i < len(unc_vals) else 0.0)
        ent = float(ent_vals[i] if i < len(ent_vals) else 0.0)
        margin = float(margin_vals[i] if i < len(margin_vals) else 0.0)
        z_target = float((score - g_target_mean) / max(1e-6, t_den))
        z_unc = float((unc - g_unc_mean) / max(1e-6, u_den))
        z_entropy = float((ent - g_entropy_mean) / max(1e-6, e_den))
        z_margin = float((margin - g_margin_mean) / max(1e-6, m_den))
        row_risk = float(max(0.0, -z_target) + (uncw * max(0.0, z_unc)) + (entw * max(0.0, z_entropy)) + (marw * max(0.0, -z_margin)))
        pred_idx = int(_resolve_target_index(out, None)) if out else 0
        row_records.append(
            {
                "row_index": int(i),
                "predicted_index": int(pred_idx),
                "target_score": float(score),
                "uncertainty": float(unc),
                "entropy": float(ent),
                "margin": float(margin),
                "z_target": float(z_target),
                "z_uncertainty": float(z_unc),
                "z_entropy": float(z_entropy),
                "z_margin": float(z_margin),
                "row_risk": float(row_risk),
                "input": row,
                "output": out,
            }
        )

    groups = {}
    for item in row_records:
        key = int(item.get("predicted_index", 0))
        groups.setdefault(key, []).append(item)

    cohorts = []
    for key, items in groups.items():
        count = int(len(items))
        share = float(count / max(1, len(row_records)))
        t_vals = [float(x.get("target_score", 0.0)) for x in items]
        u_vals = [float(x.get("uncertainty", 0.0)) for x in items]
        e_vals = [float(x.get("entropy", 0.0)) for x in items]
        m_vals = [float(x.get("margin", 0.0)) for x in items]
        r_vals = [float(x.get("row_risk", 0.0)) for x in items]
        t_mean, t_std = _mean_std(t_vals)
        u_mean, _ = _mean_std(u_vals)
        e_mean, _ = _mean_std(e_vals)
        m_mean, _ = _mean_std(m_vals)
        r_mean, r_std = _mean_std(r_vals)
        z_target = float((t_mean - g_target_mean) / max(1e-6, t_den))
        z_unc = float((u_mean - g_unc_mean) / max(1e-6, u_den))
        z_entropy = float((e_mean - g_entropy_mean) / max(1e-6, e_den))
        z_margin = float((m_mean - g_margin_mean) / max(1e-6, m_den))
        risk_score = float(max(0.0, -z_target) + (uncw * max(0.0, z_unc)) + (entw * max(0.0, z_entropy)) + (marw * max(0.0, -z_margin)))
        risk_level = "low"
        if risk_score >= 2.5:
            risk_level = "critical"
        elif risk_score >= 1.5:
            risk_level = "high"
        elif risk_score >= 0.75:
            risk_level = "medium"
        reps = sorted(items, key=lambda x: float(x.get("row_risk", 0.0)), reverse=True)[: min(3, len(items))]
        cohorts.append(
            {
                "predicted_index": int(key),
                "count": int(count),
                "share": float(share),
                "target_mean": float(t_mean),
                "target_std": float(t_std),
                "uncertainty_mean": float(u_mean),
                "entropy_mean": float(e_mean),
                "margin_mean": float(m_mean),
                "row_risk_mean": float(r_mean),
                "row_risk_std": float(r_std),
                "z_target": float(z_target),
                "z_uncertainty": float(z_unc),
                "z_entropy": float(z_entropy),
                "z_margin": float(z_margin),
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "representative_rows": reps,
            }
        )

    cohorts.sort(key=lambda x: (float(x.get("risk_score", 0.0)), float(x.get("share", 0.0))), reverse=True)
    dominant = max(cohorts, key=lambda x: int(x.get("count", 0))) if cohorts else {}
    top_risky = cohorts[:topn]

    critical_count = sum(1 for c in cohorts if str(c.get("risk_level", "")) == "critical")
    high_count = sum(1 for c in cohorts if str(c.get("risk_level", "")) == "high")
    medium_count = sum(1 for c in cohorts if str(c.get("risk_level", "")) == "medium")
    low_count = max(0, len(cohorts) - critical_count - high_count - medium_count)
    high_risk_count = critical_count + high_count
    high_risk_fraction = float(high_risk_count / max(1, len(cohorts)))
    top_risk_score = float(top_risky[0].get("risk_score", 0.0)) if top_risky else 0.0

    recommendation = "cohorts_stable"
    if high_risk_fraction >= 0.40 or top_risk_score >= 2.50:
        recommendation = "rebalance_or_collect_more_data_for_dominant_cohorts"
    elif high_risk_fraction >= 0.20 or top_risk_score >= 1.40:
        recommendation = "monitor_shifted_cohorts_and_review_thresholds"

    worst_rep = []
    if top_risky:
        reps = top_risky[0].get("representative_rows", [])
        if reps:
            worst_rep = reps[0].get("output", [])

    return {
        "target_index": int(tgt),
        "row_count": int(len(cur)),
        "reference_row_count": int(len(ref)),
        "cohort_count": int(len(cohorts)),
        "top_groups": int(topn),
        "uncertainty_weight": float(uncw),
        "entropy_weight": float(entw),
        "margin_weight": float(marw),
        "global_stats": {
            "target_mean": float(g_target_mean),
            "target_std": float(g_target_std),
            "uncertainty_mean": float(g_unc_mean),
            "uncertainty_std": float(g_unc_std),
            "entropy_mean": float(g_entropy_mean),
            "entropy_std": float(g_entropy_std),
            "margin_mean": float(g_margin_mean),
            "margin_std": float(g_margin_std),
        },
        "summary": {
            "dominant_predicted_index": int(dominant.get("predicted_index", 0)) if dominant else 0,
            "dominant_share": float(dominant.get("share", 0.0)) if dominant else 0.0,
            "top_risk_predicted_index": int(top_risky[0].get("predicted_index", 0)) if top_risky else 0,
            "top_risk_score": float(top_risk_score),
            "critical_count": int(critical_count),
            "high_count": int(high_count),
            "medium_count": int(medium_count),
            "low_count": int(low_count),
            "high_risk_count": int(high_risk_count),
            "high_risk_fraction": float(high_risk_fraction),
            "recommendation": recommendation,
        },
        "cohorts": cohorts,
        "top_risky_cohorts": top_risky,
        "dominant_cohort": dominant if dominant else {},
        "worst_topk": _topk_row(worst_rep, max(1, int(topk))) if worst_rep else [],
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_watchtower(
    model,
    device,
    expected_dim,
    rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    top_features=12,
    top_rows=8,
    top_groups=6,
    uncertainty_weight=0.50,
    entropy_weight=0.50,
    margin_weight=0.30,
    drift_weight=1.00,
    sentinel_weight=1.00,
    cohort_weight=1.00,
    medium_threshold=0.35,
    high_threshold=0.60,
    critical_threshold=0.85,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    drift_payload = _tool_drift(
        model=model,
        device=device,
        expected_dim=expected_dim,
        current_rows=rows,
        reference_rows=reference_rows,
        reference_input=reference_input,
        target_index=target_index,
        top_features=max(1, int(top_features)),
        topk=max(1, int(topk)),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
    )
    sentinel_payload = _tool_sentinel(
        model=model,
        device=device,
        expected_dim=expected_dim,
        rows=rows,
        reference_rows=reference_rows,
        reference_input=reference_input,
        target_index=target_index,
        top_rows=max(1, int(top_rows)),
        uncertainty_weight=max(0.0, float(uncertainty_weight)),
        entropy_weight=max(0.0, float(entropy_weight)),
        topk=max(1, int(topk)),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
    )
    cohort_payload = _tool_cohort(
        model=model,
        device=device,
        expected_dim=expected_dim,
        rows=rows,
        reference_rows=reference_rows,
        reference_input=reference_input,
        target_index=target_index,
        top_groups=max(1, int(top_groups)),
        uncertainty_weight=max(0.0, float(uncertainty_weight)),
        entropy_weight=max(0.0, float(entropy_weight)),
        margin_weight=max(0.0, float(margin_weight)),
        topk=max(1, int(topk)),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
    )

    drift_summary = drift_payload.get("summary", {})
    sentinel_summary = sentinel_payload.get("summary", {})
    cohort_summary = cohort_payload.get("summary", {})

    drift_score = float(drift_summary.get("drift_score", 0.0))
    sentinel_anomaly_mean = float(sentinel_summary.get("anomaly_mean", 0.0))
    sentinel_anomaly_max = float(sentinel_summary.get("anomaly_max", 0.0))
    sentinel_high_risk_fraction = float(sentinel_summary.get("high_risk_fraction", 0.0))
    cohort_top_risk_score = float(cohort_summary.get("top_risk_score", 0.0))
    cohort_high_risk_fraction = float(cohort_summary.get("high_risk_fraction", 0.0))

    drift_norm = float(max(0.0, min(1.0, drift_score / 0.75)))
    sentinel_norm = float(
        max(
            0.0,
            min(
                1.0,
                max(
                    sentinel_anomaly_mean / 2.0,
                    sentinel_anomaly_max / 4.0,
                    sentinel_high_risk_fraction / 0.30,
                ),
            ),
        )
    )
    cohort_norm = float(
        max(
            0.0,
            min(
                1.0,
                max(
                    cohort_top_risk_score / 2.50,
                    cohort_high_risk_fraction / 0.40,
                ),
            ),
        )
    )

    w_drift = max(0.0, float(drift_weight))
    w_sentinel = max(0.0, float(sentinel_weight))
    w_cohort = max(0.0, float(cohort_weight))
    w_sum = w_drift + w_sentinel + w_cohort
    if w_sum <= 1e-12:
        w_drift, w_sentinel, w_cohort = 1.0, 1.0, 1.0
        w_sum = 3.0
    combined_risk_score = float(
        ((w_drift * drift_norm) + (w_sentinel * sentinel_norm) + (w_cohort * cohort_norm)) / w_sum
    )

    med_th = max(0.0, float(medium_threshold))
    high_th = max(med_th + 1e-6, float(high_threshold))
    critical_th = max(high_th + 1e-6, float(critical_threshold))

    drift_severe = drift_score >= 0.75 or str(drift_summary.get("recommendation", "")) == "high_drift_retrain_or_recalibrate"
    sentinel_severe = sentinel_high_risk_fraction >= 0.30 or sentinel_anomaly_max >= 4.0
    cohort_severe = cohort_high_risk_fraction >= 0.40 or cohort_top_risk_score >= 2.50
    severe_count = int((1 if drift_severe else 0) + (1 if sentinel_severe else 0) + (1 if cohort_severe else 0))

    if combined_risk_score >= critical_th:
        risk_level = "critical"
    elif combined_risk_score >= high_th:
        risk_level = "high"
    elif combined_risk_score >= med_th:
        risk_level = "medium"
    else:
        risk_level = "low"

    if severe_count >= 2:
        risk_level = "critical"
    elif severe_count == 1 and risk_level == "medium":
        risk_level = "high"

    recommendation = "normal_monitoring"
    if risk_level == "critical":
        recommendation = "halt_auto_actions_and_retrain"
    elif risk_level == "high":
        recommendation = "activate_guardrails_and_review_data"
    elif risk_level == "medium":
        recommendation = "monitor_shift_and_collect_feedback"

    action_plan = []
    if drift_severe:
        action_plan.append(
            {
                "priority": "p0",
                "area": "drift",
                "action": "retrain_or_recalibrate",
                "reason": "drift score exceeded high-risk boundary",
                "metric": {"drift_score": float(drift_score)},
            }
        )
    if sentinel_severe:
        action_plan.append(
            {
                "priority": "p0",
                "area": "sentinel",
                "action": "activate_guardrails",
                "reason": "anomaly pressure indicates unstable batch behavior",
                "metric": {
                    "anomaly_max": float(sentinel_anomaly_max),
                    "high_risk_fraction": float(sentinel_high_risk_fraction),
                },
            }
        )
    if cohort_severe:
        action_plan.append(
            {
                "priority": "p1",
                "area": "cohort",
                "action": "rebalance_or_collect_targeted_data",
                "reason": "cohort imbalance and risk concentration detected",
                "metric": {
                    "top_risk_score": float(cohort_top_risk_score),
                    "high_risk_fraction": float(cohort_high_risk_fraction),
                },
            }
        )
    if not action_plan and risk_level in ("medium", "high"):
        action_plan.append(
            {
                "priority": "p2",
                "area": "monitoring",
                "action": "increase_observability",
                "reason": "aggregate risk is elevated despite no single severe signal",
            }
        )
    if not action_plan:
        action_plan.append(
            {
                "priority": "p3",
                "area": "monitoring",
                "action": "continue_normal_monitoring",
                "reason": "signals are within normal operating envelope",
            }
        )

    signals = [
        {
            "signal": "drift",
            "raw_score": float(drift_score),
            "normalized_score": float(drift_norm),
            "weight": float(w_drift),
            "status": "severe" if drift_severe else "ok",
            "recommendation": drift_summary.get("recommendation", "monitor"),
        },
        {
            "signal": "sentinel",
            "raw_score": float(sentinel_anomaly_mean),
            "normalized_score": float(sentinel_norm),
            "weight": float(w_sentinel),
            "status": "severe" if sentinel_severe else "ok",
            "recommendation": sentinel_summary.get("recommendation", "normal_monitoring"),
        },
        {
            "signal": "cohort",
            "raw_score": float(cohort_top_risk_score),
            "normalized_score": float(cohort_norm),
            "weight": float(w_cohort),
            "status": "severe" if cohort_severe else "ok",
            "recommendation": cohort_summary.get("recommendation", "cohorts_stable"),
        },
    ]

    top_anomalies = sentinel_payload.get("top_anomalies", [])
    top_risky_cohorts = cohort_payload.get("top_risky_cohorts", [])
    top_feature_shift = drift_payload.get("top_feature_shift", [])

    return {
        "target_index": int(drift_payload.get("target_index", 0)),
        "row_count": int(sentinel_payload.get("row_count", 0)),
        "reference_row_count": int(sentinel_payload.get("reference_row_count", 0)),
        "weights": {
            "drift_weight": float(w_drift),
            "sentinel_weight": float(w_sentinel),
            "cohort_weight": float(w_cohort),
        },
        "thresholds": {
            "medium": float(med_th),
            "high": float(high_th),
            "critical": float(critical_th),
        },
        "summary": {
            "risk_level": risk_level,
            "combined_risk_score": float(combined_risk_score),
            "severe_signal_count": int(severe_count),
            "drift_score": float(drift_score),
            "sentinel_anomaly_mean": float(sentinel_anomaly_mean),
            "sentinel_high_risk_fraction": float(sentinel_high_risk_fraction),
            "cohort_top_risk_score": float(cohort_top_risk_score),
            "cohort_high_risk_fraction": float(cohort_high_risk_fraction),
            "recommendation": recommendation,
        },
        "signals": signals,
        "action_plan": action_plan,
        "top_watch_items": {
            "top_feature_shift": top_feature_shift[: max(1, min(5, len(top_feature_shift)))],
            "top_anomalies": top_anomalies[: max(1, min(5, len(top_anomalies)))],
            "top_risky_cohorts": top_risky_cohorts[: max(1, min(5, len(top_risky_cohorts)))],
        },
        "drift": {
            "summary": drift_summary,
            "top_feature_shift": drift_payload.get("top_feature_shift", []),
            "top_output_shift": drift_payload.get("top_output_shift", []),
            "reference_topk": drift_payload.get("reference_topk", []),
            "current_topk": drift_payload.get("current_topk", []),
        },
        "sentinel": {
            "summary": sentinel_summary,
            "top_anomalies": top_anomalies,
            "stable_rows": sentinel_payload.get("stable_rows", []),
            "worst_topk": sentinel_payload.get("worst_topk", []),
        },
        "cohort": {
            "summary": cohort_summary,
            "top_risky_cohorts": top_risky_cohorts,
            "dominant_cohort": cohort_payload.get("dominant_cohort", {}),
            "worst_topk": cohort_payload.get("worst_topk", []),
        },
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_simlab(
    model,
    device,
    expected_dim,
    rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    noise_levels=None,
    repeats=3,
    drift_bias=0.02,
    seed=None,
    top_features=12,
    top_rows=8,
    top_groups=6,
    uncertainty_weight=0.50,
    entropy_weight=0.50,
    margin_weight=0.30,
    drift_weight=1.00,
    sentinel_weight=1.00,
    cohort_weight=1.00,
    medium_threshold=0.35,
    high_threshold=0.60,
    critical_threshold=0.85,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    clean_rows = _validate_rows(rows, expected_dim)
    if not clean_rows:
        raise ValueError("simlab requires non-empty rows")

    ref_rows = _validate_rows(reference_rows, expected_dim) if reference_rows is not None else None
    ref_input = _validate_rows([reference_input], expected_dim)[0] if reference_input is not None else None
    levels = _normalize_non_negative_float_list(
        noise_levels,
        defaults=[0.0, 0.01, 0.03, 0.05, 0.10],
        low=0.0,
        high=10.0,
    )
    reps = max(1, int(repeats))
    bias = max(0.0, abs(float(drift_bias)))
    fixed_seed = int(seed) if seed is not None else None

    def _perturb_rows(base_rows, noise_level, rep_idx):
        lvl = max(0.0, float(noise_level))
        if lvl <= 0.0 and bias <= 0.0:
            return [list(r) for r in base_rows]
        gen = torch.Generator(device="cpu")
        if fixed_seed is not None:
            gen.manual_seed(max(0, int(fixed_seed + (rep_idx * 1009) + int(round(lvl * 1_000_000)))))
        out = []
        for i, row in enumerate(base_rows):
            new_row = []
            for j, val in enumerate(row):
                nv = float(val)
                if lvl > 0.0:
                    nv += float(torch.randn(1, generator=gen).item()) * lvl
                if bias > 0.0:
                    direction = 1.0 if ((i + j + rep_idx) % 2 == 0) else -1.0
                    nv += direction * (bias * lvl)
                new_row.append(float(nv))
            out.append(new_row)
        return out

    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    scenarios = []
    risk_curve = []
    worst = None

    for lvl in levels:
        sim_rows_ref = _perturb_rows(clean_rows, float(lvl), 0)
        runs = []
        worst_run = None
        for rep_idx in range(reps):
            sim_rows = sim_rows_ref if rep_idx == 0 else _perturb_rows(clean_rows, float(lvl), rep_idx)
            wt = _tool_watchtower(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=sim_rows,
                reference_rows=ref_rows,
                reference_input=ref_input,
                target_index=target_index,
                top_features=top_features,
                top_rows=top_rows,
                top_groups=top_groups,
                uncertainty_weight=uncertainty_weight,
                entropy_weight=entropy_weight,
                margin_weight=margin_weight,
                drift_weight=drift_weight,
                sentinel_weight=sentinel_weight,
                cohort_weight=cohort_weight,
                medium_threshold=medium_threshold,
                high_threshold=high_threshold,
                critical_threshold=critical_threshold,
                topk=topk,
                mc_samples=mc_samples,
                as_probs=as_probs,
            )
            s = wt.get("summary", {})
            run = {
                "repeat_index": int(rep_idx),
                "risk_level": str(s.get("risk_level", "low")),
                "combined_risk_score": float(s.get("combined_risk_score", 0.0)),
                "severe_signal_count": int(s.get("severe_signal_count", 0)),
                "recommendation": s.get("recommendation", "normal_monitoring"),
                "top_action": (wt.get("action_plan", []) or [{}])[0].get("action", "none"),
                "watch_items": wt.get("top_watch_items", {}),
            }
            runs.append(run)
            if worst_run is None or float(run.get("combined_risk_score", 0.0)) > float(
                worst_run.get("combined_risk_score", 0.0)
            ):
                worst_run = run

        scores = [float(r.get("combined_risk_score", 0.0)) for r in runs]
        score_mean, score_std = _mean_std(scores)
        score_max = float(max(scores) if scores else 0.0)
        score_min = float(min(scores) if scores else 0.0)
        agg_level = "low"
        if runs:
            agg_level = max(runs, key=lambda r: severity_rank.get(str(r.get("risk_level", "low")), 0)).get("risk_level", "low")
        scenario = {
            "noise_level": float(lvl),
            "repeat_count": int(reps),
            "risk_level": str(agg_level),
            "mean_risk_score": float(score_mean),
            "std_risk_score": float(score_std),
            "max_risk_score": float(score_max),
            "min_risk_score": float(score_min),
            "worst_run": worst_run if worst_run else {},
            "runs": runs,
        }
        scenarios.append(scenario)
        risk_curve.append({"noise_level": float(lvl), "risk_score": float(score_mean)})
        if worst is None or score_max > float(worst.get("max_risk_score", 0.0)):
            worst = scenario

    scenarios.sort(key=lambda item: float(item.get("noise_level", 0.0)))
    risk_curve.sort(key=lambda item: float(item.get("noise_level", 0.0)))

    auc = 0.0
    if len(risk_curve) == 1:
        auc = float(risk_curve[0].get("risk_score", 0.0))
    else:
        for i in range(1, len(risk_curve)):
            x0 = float(risk_curve[i - 1].get("noise_level", 0.0))
            x1 = float(risk_curve[i].get("noise_level", 0.0))
            y0 = float(risk_curve[i - 1].get("risk_score", 0.0))
            y1 = float(risk_curve[i].get("risk_score", 0.0))
            width = max(0.0, x1 - x0)
            auc += 0.5 * (y0 + y1) * width
        span = max(1e-9, float(risk_curve[-1].get("noise_level", 0.0)) - float(risk_curve[0].get("noise_level", 0.0)))
        auc = float(auc / span)
    auc = float(max(0.0, min(1.0, auc)))

    first_high_noise = None
    first_critical_noise = None
    for row in scenarios:
        lvl = str(row.get("risk_level", "low"))
        noise = float(row.get("noise_level", 0.0))
        if first_high_noise is None and lvl in ("high", "critical"):
            first_high_noise = noise
        if first_critical_noise is None and lvl == "critical":
            first_critical_noise = noise

    resilience_score = float(max(0.0, min(1.0, 1.0 - auc)))
    resilience_grade = "strong"
    if resilience_score < 0.45:
        resilience_grade = "fragile"
    elif resilience_score < 0.70:
        resilience_grade = "moderate"

    recommendation = "maintain_and_monitor"
    if resilience_grade == "fragile":
        recommendation = "prioritize_robust_training_and_guardrails"
    elif resilience_grade == "moderate":
        recommendation = "targeted_augmentation_and_threshold_tuning"

    action_plan = []
    if first_critical_noise is not None:
        action_plan.append(
            {
                "priority": "p0",
                "action": "critical_noise_hardening",
                "trigger_noise_level": float(first_critical_noise),
                "reason": "critical risk appears under simulation",
            }
        )
    if first_high_noise is not None:
        action_plan.append(
            {
                "priority": "p1",
                "action": "high_risk_noise_guardrails",
                "trigger_noise_level": float(first_high_noise),
                "reason": "high risk appears before critical threshold",
            }
        )
    if resilience_grade == "fragile":
        action_plan.append(
            {
                "priority": "p1",
                "action": "augment_training_with_noisy_shifted_batches",
                "noise_levels": levels,
                "drift_bias": float(bias),
            }
        )
    elif resilience_grade == "moderate":
        action_plan.append(
            {
                "priority": "p2",
                "action": "tune_watchtower_thresholds",
                "current_thresholds": {
                    "medium": float(medium_threshold),
                    "high": float(high_threshold),
                    "critical": float(critical_threshold),
                },
            }
        )
    else:
        action_plan.append(
            {
                "priority": "p3",
                "action": "continue_monitoring_with_periodic_simlab",
                "cadence": "weekly",
            }
        )

    baseline = min(scenarios, key=lambda item: float(item.get("noise_level", 0.0))) if scenarios else {}

    return {
        "target_index": int(target_index) if target_index is not None else None,
        "row_count": int(len(clean_rows)),
        "reference_row_count": int(len(ref_rows) if ref_rows else (1 if ref_input is not None else 0)),
        "noise_levels": levels,
        "repeats": int(reps),
        "drift_bias": float(bias),
        "seed": int(fixed_seed) if fixed_seed is not None else None,
        "summary": {
            "resilience_score": float(resilience_score),
            "resilience_grade": resilience_grade,
            "risk_auc": float(auc),
            "first_high_noise": first_high_noise,
            "first_critical_noise": first_critical_noise,
            "worst_noise_level": float(worst.get("noise_level", 0.0)) if worst else 0.0,
            "worst_risk_score": float(worst.get("max_risk_score", 0.0)) if worst else 0.0,
            "recommendation": recommendation,
        },
        "risk_curve": risk_curve,
        "scenarios": scenarios,
        "baseline_scenario": baseline if baseline else {},
        "worst_scenario": worst if worst else {},
        "action_plan": action_plan,
        "weights": {
            "drift_weight": float(max(0.0, drift_weight)),
            "sentinel_weight": float(max(0.0, sentinel_weight)),
            "cohort_weight": float(max(0.0, cohort_weight)),
        },
        "thresholds": {
            "medium": float(max(0.0, medium_threshold)),
            "high": float(max(0.0, high_threshold)),
            "critical": float(max(0.0, critical_threshold)),
        },
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_policylab(
    model,
    device,
    expected_dim,
    rows,
    reference_rows=None,
    reference_input=None,
    target_index=None,
    noise_levels=None,
    repeats=3,
    drift_bias=0.02,
    search_iters=10,
    max_weight_shift=0.35,
    threshold_margin=0.06,
    seed=None,
    top_features=12,
    top_rows=8,
    top_groups=6,
    uncertainty_weight=0.50,
    entropy_weight=0.50,
    margin_weight=0.30,
    drift_weight=1.00,
    sentinel_weight=1.00,
    cohort_weight=1.00,
    medium_threshold=0.35,
    high_threshold=0.60,
    critical_threshold=0.85,
    topk=5,
    mc_samples=1,
    as_probs=False,
):
    clean_rows = _validate_rows(rows, expected_dim)
    if not clean_rows:
        raise ValueError("policylab requires non-empty rows")

    ref_rows = _validate_rows(reference_rows, expected_dim) if reference_rows is not None else None
    ref_input = _validate_rows([reference_input], expected_dim)[0] if reference_input is not None else None
    levels = _normalize_non_negative_float_list(
        noise_levels,
        defaults=[0.0, 0.01, 0.03, 0.05, 0.10],
        low=0.0,
        high=10.0,
    )
    reps = max(1, int(repeats))
    bias = max(0.0, abs(float(drift_bias)))
    iters = max(1, int(search_iters))
    max_shift = max(0.01, abs(float(max_weight_shift)))
    th_margin = max(0.0, abs(float(threshold_margin)))
    fixed_seed = int(seed) if seed is not None else None

    def _clamp01(x):
        return float(max(0.0, min(1.0, float(x))))

    def _norm_thresholds(med, high, critical):
        m = max(0.0, float(med))
        h = max(m + 1e-6, float(high))
        c = max(h + 1e-6, float(critical))
        return float(m), float(h), float(c)

    def _rand_uniform(gen, low, high):
        lo = float(low)
        hi = float(high)
        if hi <= lo:
            return lo
        if gen is None:
            return float(lo + (hi - lo) * torch.rand(1).item())
        return float(lo + (hi - lo) * torch.rand(1, generator=gen).item())

    base_weights = {
        "drift_weight": float(max(0.0, drift_weight)),
        "sentinel_weight": float(max(0.0, sentinel_weight)),
        "cohort_weight": float(max(0.0, cohort_weight)),
    }
    b_med, b_high, b_critical = _norm_thresholds(medium_threshold, high_threshold, critical_threshold)
    base_thresholds = {
        "medium": float(b_med),
        "high": float(b_high),
        "critical": float(b_critical),
    }
    max_noise = float(max(levels) if levels else 1.0)
    noise_den = max(1e-9, max_noise)

    def _score_policy(sim_payload):
        summary = sim_payload.get("summary", {})
        scenarios = sim_payload.get("scenarios", [])
        resilience = _clamp01(summary.get("resilience_score", 0.0))
        worst_risk = _clamp01(summary.get("worst_risk_score", 0.0))
        risk_auc = _clamp01(summary.get("risk_auc", 0.0))
        low_noise_risk = _clamp01(scenarios[0].get("mean_risk_score", 0.0)) if scenarios else 0.0
        high_noise_risk = _clamp01(scenarios[-1].get("mean_risk_score", 0.0)) if scenarios else 0.0
        first_high = summary.get("first_high_noise")
        first_critical = summary.get("first_critical_noise")
        first_high_term = 1.0 if first_high is None else _clamp01(float(first_high) / noise_den)
        first_critical_term = 1.0 if first_critical is None else _clamp01(float(first_critical) / noise_den)

        objective = float(
            (1.35 * resilience)
            + (0.30 * first_high_term)
            + (0.35 * first_critical_term)
            - (0.40 * low_noise_risk)
            - (0.15 * high_noise_risk)
            - (0.20 * worst_risk)
            - (0.10 * risk_auc)
        )
        return {
            "objective_score": float(objective),
            "resilience_score": float(resilience),
            "risk_auc": float(risk_auc),
            "low_noise_risk": float(low_noise_risk),
            "high_noise_risk": float(high_noise_risk),
            "worst_risk_score": float(worst_risk),
            "first_high_noise": first_high,
            "first_critical_noise": first_critical,
        }

    candidates = []

    def _add_candidate(name, policy_type, weights, thresholds):
        m, h, c = _norm_thresholds(
            thresholds.get("medium", base_thresholds["medium"]),
            thresholds.get("high", base_thresholds["high"]),
            thresholds.get("critical", base_thresholds["critical"]),
        )
        dw = max(0.0, float(weights.get("drift_weight", base_weights["drift_weight"])))
        sw = max(0.0, float(weights.get("sentinel_weight", base_weights["sentinel_weight"])))
        cw = max(0.0, float(weights.get("cohort_weight", base_weights["cohort_weight"])))
        if (dw + sw + cw) <= 1e-12:
            dw = float(base_weights["drift_weight"])
            sw = float(base_weights["sentinel_weight"])
            cw = float(base_weights["cohort_weight"])
        candidates.append(
            {
                "name": str(name),
                "policy_type": str(policy_type),
                "weights": {
                    "drift_weight": float(dw),
                    "sentinel_weight": float(sw),
                    "cohort_weight": float(cw),
                },
                "thresholds": {
                    "medium": float(m),
                    "high": float(h),
                    "critical": float(c),
                },
            }
        )

    _add_candidate("baseline", "baseline", dict(base_weights), dict(base_thresholds))
    _add_candidate(
        "balanced_strict",
        "heuristic",
        dict(base_weights),
        {
            "medium": base_thresholds["medium"] + th_margin,
            "high": base_thresholds["high"] + th_margin,
            "critical": base_thresholds["critical"] + th_margin,
        },
    )
    _add_candidate(
        "balanced_sensitive",
        "heuristic",
        dict(base_weights),
        {
            "medium": max(0.0, base_thresholds["medium"] - th_margin),
            "high": max(0.0, base_thresholds["high"] - th_margin),
            "critical": max(0.0, base_thresholds["critical"] - th_margin),
        },
    )
    _add_candidate(
        "drift_guard",
        "heuristic",
        {
            "drift_weight": base_weights["drift_weight"] * (1.0 + (0.80 * max_shift)),
            "sentinel_weight": base_weights["sentinel_weight"] * max(0.10, 1.0 - (0.35 * max_shift)),
            "cohort_weight": base_weights["cohort_weight"] * max(0.10, 1.0 - (0.25 * max_shift)),
        },
        {
            "medium": base_thresholds["medium"] + (0.40 * th_margin),
            "high": base_thresholds["high"] + (0.40 * th_margin),
            "critical": base_thresholds["critical"] + (0.40 * th_margin),
        },
    )
    _add_candidate(
        "sentinel_guard",
        "heuristic",
        {
            "drift_weight": base_weights["drift_weight"] * max(0.10, 1.0 - (0.25 * max_shift)),
            "sentinel_weight": base_weights["sentinel_weight"] * (1.0 + (0.85 * max_shift)),
            "cohort_weight": base_weights["cohort_weight"] * max(0.10, 1.0 - (0.20 * max_shift)),
        },
        {
            "medium": base_thresholds["medium"] + (0.15 * th_margin),
            "high": base_thresholds["high"] + (0.15 * th_margin),
            "critical": base_thresholds["critical"] + (0.15 * th_margin),
        },
    )
    _add_candidate(
        "cohort_guard",
        "heuristic",
        {
            "drift_weight": base_weights["drift_weight"] * max(0.10, 1.0 - (0.20 * max_shift)),
            "sentinel_weight": base_weights["sentinel_weight"] * max(0.10, 1.0 - (0.20 * max_shift)),
            "cohort_weight": base_weights["cohort_weight"] * (1.0 + (0.90 * max_shift)),
        },
        {
            "medium": base_thresholds["medium"] + (0.20 * th_margin),
            "high": base_thresholds["high"] + (0.20 * th_margin),
            "critical": base_thresholds["critical"] + (0.20 * th_margin),
        },
    )

    random_count = max(0, iters - len(candidates))
    gen = torch.Generator(device="cpu")
    if fixed_seed is not None:
        gen.manual_seed(max(0, fixed_seed))
    else:
        gen = None

    for i in range(random_count):
        mult_d = 1.0 + _rand_uniform(gen, -max_shift, max_shift)
        mult_s = 1.0 + _rand_uniform(gen, -max_shift, max_shift)
        mult_c = 1.0 + _rand_uniform(gen, -max_shift, max_shift)
        dw = base_weights["drift_weight"] * mult_d
        sw = base_weights["sentinel_weight"] * mult_s
        cw = base_weights["cohort_weight"] * mult_c
        t_med = base_thresholds["medium"] + _rand_uniform(gen, -th_margin, th_margin)
        t_high = base_thresholds["high"] + _rand_uniform(gen, -th_margin, th_margin)
        t_critical = base_thresholds["critical"] + _rand_uniform(gen, -th_margin, th_margin)
        _add_candidate(
            f"search_{i + 1}",
            "search",
            {
                "drift_weight": float(max(0.0, dw)),
                "sentinel_weight": float(max(0.0, sw)),
                "cohort_weight": float(max(0.0, cw)),
            },
            {
                "medium": float(max(0.0, t_med)),
                "high": float(max(0.0, t_high)),
                "critical": float(max(0.0, t_critical)),
            },
        )

    leaderboard = []
    for idx, cand in enumerate(candidates):
        sim_seed = None if fixed_seed is None else int(fixed_seed + (idx * 7919))
        sim_payload = _tool_simlab(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=clean_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            noise_levels=levels,
            repeats=reps,
            drift_bias=bias,
            seed=sim_seed,
            top_features=top_features,
            top_rows=top_rows,
            top_groups=top_groups,
            uncertainty_weight=uncertainty_weight,
            entropy_weight=entropy_weight,
            margin_weight=margin_weight,
            drift_weight=cand["weights"]["drift_weight"],
            sentinel_weight=cand["weights"]["sentinel_weight"],
            cohort_weight=cand["weights"]["cohort_weight"],
            medium_threshold=cand["thresholds"]["medium"],
            high_threshold=cand["thresholds"]["high"],
            critical_threshold=cand["thresholds"]["critical"],
            topk=topk,
            mc_samples=mc_samples,
            as_probs=as_probs,
        )
        score = _score_policy(sim_payload)
        leaderboard.append(
            {
                "name": cand["name"],
                "policy_type": cand["policy_type"],
                "weights": cand["weights"],
                "thresholds": cand["thresholds"],
                "score": score,
                "summary": sim_payload.get("summary", {}),
                "risk_curve": sim_payload.get("risk_curve", []),
                "action_plan": sim_payload.get("action_plan", []),
            }
        )

    if not leaderboard:
        raise ValueError("policylab could not evaluate policy candidates")

    ranked = sorted(
        leaderboard,
        key=lambda item: float((item.get("score", {}) or {}).get("objective_score", -1e9)),
        reverse=True,
    )
    baseline_entry = next((it for it in ranked if it.get("name") == "baseline"), ranked[0])
    best_entry = ranked[0]

    baseline_score = float((baseline_entry.get("score", {}) or {}).get("objective_score", 0.0))
    best_score = float((best_entry.get("score", {}) or {}).get("objective_score", 0.0))
    improvement = float(best_score - baseline_score)
    adopt = bool(best_entry.get("name") != "baseline" and improvement >= 0.015)

    recommendation = "keep_baseline_policy"
    if adopt:
        recommendation = "adopt_recommended_policy"
    elif improvement > 0:
        recommendation = "monitor_candidate_policy"

    action_plan = []
    if adopt:
        action_plan.append(
            {
                "priority": "p1",
                "action": "update_watchtower_policy",
                "policy_name": best_entry.get("name"),
                "weights": best_entry.get("weights", {}),
                "thresholds": best_entry.get("thresholds", {}),
                "objective_improvement": float(improvement),
            }
        )
        action_plan.append(
            {
                "priority": "p2",
                "action": "run_simlab_regression_gate",
                "noise_levels": levels,
                "repeats": int(reps),
                "reason": "validate tuned policy before production rollout",
            }
        )
    else:
        action_plan.append(
            {
                "priority": "p2",
                "action": "retain_baseline_policy",
                "reason": "candidate policies did not clear adoption threshold",
                "best_candidate": best_entry.get("name"),
                "objective_improvement": float(improvement),
            }
        )
    action_plan.append(
        {
            "priority": "p3",
            "action": "schedule_periodic_policylab",
            "cadence": "weekly",
        }
    )

    return {
        "target_index": int(target_index) if target_index is not None else None,
        "row_count": int(len(clean_rows)),
        "reference_row_count": int(len(ref_rows) if ref_rows else (1 if ref_input is not None else 0)),
        "noise_levels": levels,
        "repeats": int(reps),
        "drift_bias": float(bias),
        "search_iters": int(iters),
        "max_weight_shift": float(max_shift),
        "threshold_margin": float(th_margin),
        "seed": int(fixed_seed) if fixed_seed is not None else None,
        "summary": {
            "baseline_objective_score": float(baseline_score),
            "recommended_objective_score": float(best_score),
            "objective_improvement": float(improvement),
            "baseline_resilience_score": float((baseline_entry.get("score", {}) or {}).get("resilience_score", 0.0)),
            "recommended_resilience_score": float((best_entry.get("score", {}) or {}).get("resilience_score", 0.0)),
            "baseline_risk_auc": float((baseline_entry.get("score", {}) or {}).get("risk_auc", 0.0)),
            "recommended_risk_auc": float((best_entry.get("score", {}) or {}).get("risk_auc", 0.0)),
            "adopt_recommended_policy": bool(adopt),
            "recommendation": recommendation,
            "best_policy_name": best_entry.get("name", "baseline"),
        },
        "baseline_policy": baseline_entry,
        "recommended_policy": best_entry,
        "policy_candidates": ranked[: max(1, min(12, len(ranked)))],
        "baseline_risk_curve": baseline_entry.get("risk_curve", []),
        "recommended_risk_curve": best_entry.get("risk_curve", []),
        "action_plan": action_plan,
        "weights": {
            "baseline": dict(base_weights),
            "recommended": dict(best_entry.get("weights", {})),
        },
        "thresholds": {
            "baseline": dict(base_thresholds),
            "recommended": dict(best_entry.get("thresholds", {})),
        },
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _tool_profile(
    model,
    device,
    expected_dim,
    base_input=None,
    batch_sizes=None,
    mc_grid=None,
    runs=30,
    as_probs=False,
):
    if base_input is None:
        if expected_dim is None:
            raise ValueError("profile requires --input when expected input dim is unknown")
        base_row = torch.randn(expected_dim).tolist()
    else:
        base_row = _validate_rows([base_input], expected_dim)[0]

    batches = _normalize_positive_int_list(batch_sizes, defaults=[1, 2, 4, 8, 16], low=1, high=4096)
    mcs = _normalize_positive_int_list(mc_grid, defaults=[1, 2, 4], low=1, high=128)
    rr = max(1, int(runs))
    cases = []
    for bs in batches:
        rows = [list(base_row) for _ in range(int(bs))]
        for i, row in enumerate(rows):
            if row:
                row[0] = float(row[0] + (float(i) * 1e-6))
        for mc in mcs:
            bench = _benchmark(
                model=model,
                rows=rows,
                device=device,
                runs=rr,
                mc_samples=max(1, int(mc)),
                as_probs=bool(as_probs),
            )
            avg_ms = float(bench.get("avg_ms", 0.0))
            throughput = float(bench.get("throughput_sps", 0.0))
            score = float(throughput / max(1e-9, avg_ms))
            cases.append(
                {
                    "batch_size": int(bs),
                    "mc_samples": int(mc),
                    "runs": int(rr),
                    "avg_ms": avg_ms,
                    "min_ms": float(bench.get("min_ms", 0.0)),
                    "max_ms": float(bench.get("max_ms", 0.0)),
                    "throughput_sps": throughput,
                    "efficiency_score": score,
                }
            )

    if not cases:
        raise ValueError("profile produced no benchmark cases")

    best_latency = min(cases, key=lambda c: float(c.get("avg_ms", 0.0)))
    best_throughput = max(cases, key=lambda c: float(c.get("throughput_sps", 0.0)))
    best_efficiency = max(cases, key=lambda c: float(c.get("efficiency_score", 0.0)))

    pareto = []
    for case in cases:
        dominated = False
        for other in cases:
            if other is case:
                continue
            better_latency = float(other.get("avg_ms", 0.0)) <= float(case.get("avg_ms", 0.0))
            better_throughput = float(other.get("throughput_sps", 0.0)) >= float(case.get("throughput_sps", 0.0))
            strictly_better = (
                float(other.get("avg_ms", 0.0)) < float(case.get("avg_ms", 0.0))
                or float(other.get("throughput_sps", 0.0)) > float(case.get("throughput_sps", 0.0))
            )
            if better_latency and better_throughput and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(case)
    pareto.sort(key=lambda c: float(c.get("avg_ms", 0.0)))

    return {
        "base_input": base_row,
        "batch_sizes": batches,
        "mc_grid": mcs,
        "runs_per_case": int(rr),
        "case_count": len(cases),
        "cases": cases,
        "pareto_frontier": pareto,
        "recommended": {
            "lowest_latency": best_latency,
            "highest_throughput": best_throughput,
            "best_efficiency": best_efficiency,
        },
        "as_probs": bool(as_probs),
    }


def _tool_autolab(
    model,
    device,
    expected_dim,
    base_input,
    input_b=None,
    feature_indices=None,
    radius=0.3,
    steps=7,
    interp_steps=9,
    epsilon=0.01,
    target_index=None,
    top_features=12,
    topk=5,
    stability_samples=64,
    noise_std=0.05,
    seed=None,
    mc_samples=1,
    as_probs=False,
):
    base_row = _validate_rows([base_input], expected_dim)[0]
    if input_b is None:
        vec_b = list(base_row)
        if vec_b:
            vec_b[0] = float(vec_b[0] + abs(float(radius)))
    else:
        vec_b = _validate_rows([input_b], expected_dim)[0]

    pipe = _tool_pipeline(
        model=model,
        device=device,
        expected_dim=expected_dim,
        base_input=base_row,
        input_b=vec_b,
        feature_indices=feature_indices,
        radius=float(radius),
        steps=max(1, int(steps)),
        epsilon=float(epsilon),
        target_index=target_index,
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
        top_features=max(1, int(top_features)),
    )
    interp = _tool_interpolate(
        model=model,
        device=device,
        expected_dim=expected_dim,
        input_a=base_row,
        input_b=vec_b,
        steps=max(2, int(interp_steps)),
        target_index=pipe.get("summary", {}).get("target_index"),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
    )
    stability = _tool_stability(
        model=model,
        device=device,
        expected_dim=expected_dim,
        input_vec=base_row,
        samples=max(4, int(stability_samples)),
        noise_std=abs(float(noise_std)),
        target_index=pipe.get("summary", {}).get("target_index"),
        mc_samples=max(1, int(mc_samples)),
        as_probs=bool(as_probs),
        topk=max(1, int(topk)),
        seed=seed,
    )

    p_summary = pipe.get("summary", {})
    i_metrics = interp.get("metrics", {})
    s_stats = stability.get("target_stats", {})
    gain = float(p_summary.get("expected_gain", 0.0))
    exploration = float(p_summary.get("exploration_score", 0.0))
    smoothness = float(i_metrics.get("smoothness_ratio", 0.0))
    robust = float(stability.get("robust_score", 0.0))
    jump_penalty = max(0.0, float(i_metrics.get("max_jump_l2", 0.0)) - float(i_metrics.get("mean_jump_l2", 0.0)))
    gain_score = math.tanh(max(0.0, gain) * 30.0)
    exploration_score = math.tanh(max(0.0, exploration) * 0.2)
    composite_raw = (
        (0.30 * robust)
        + (0.25 * smoothness)
        + (0.25 * gain_score)
        + (0.20 * exploration_score)
        - (0.10 * min(1.0, jump_penalty))
    )
    composite_score = float(max(0.0, min(1.0, composite_raw)))

    strategy = "balanced_explore"
    if robust < 0.55:
        strategy = "robustness_first"
    elif gain > 0.01 and robust >= 0.65:
        strategy = "exploit_high_confidence"
    elif smoothness < 0.80:
        strategy = "stabilize_transition"

    action_plan = list(pipe.get("recommended_actions", []))
    if robust < 0.65:
        action_plan.append(
            {
                "action": "robustness_train_step",
                "noise_std": float(noise_std),
                "samples": int(stability_samples),
                "reason": "stability score below preferred range",
            }
        )
    if smoothness < 0.85:
        action_plan.append(
            {
                "action": "smooth_transition_regularization",
                "interp_steps": int(max(2, interp_steps)),
                "reason": "interpolation path is not smooth enough",
            }
        )
    if gain <= 0.0:
        action_plan.append(
            {
                "action": "expand_search",
                "radius": float(max(0.4, abs(float(radius)) * 1.5)),
                "reason": "expected gain is non-positive",
            }
        )

    return {
        "base_input": base_row,
        "input_b": vec_b,
        "summary": {
            "strategy": strategy,
            "composite_score": composite_score,
            "expected_gain": gain,
            "robust_score": robust,
            "smoothness_ratio": smoothness,
            "exploration_score": exploration,
            "target_index": int(p_summary.get("target_index", 0)),
            "target_mean": float(s_stats.get("mean", 0.0)),
            "target_std": float(s_stats.get("std", 0.0)),
            "action_count": len(action_plan),
        },
        "action_plan": action_plan,
        "pipeline": pipe,
        "interpolate": interp,
        "stability": stability,
        "mc_samples": int(max(1, mc_samples)),
        "as_probs": bool(as_probs),
    }


def _validate_rows(rows, expected_dim):
    if rows is None:
        raise ValueError("missing inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("inputs must be a non-empty list")
    if not isinstance(rows[0], list):
        rows = [rows]
    clean_rows = []
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"row {i} must be a list")
        try:
            clean = [float(v) for v in row]
        except Exception:
            raise ValueError(f"row {i} contains non-numeric values")
        clean_rows.append(clean)
    if expected_dim is not None:
        for i, row in enumerate(clean_rows):
            if len(row) != expected_dim:
                raise ValueError(f"row {i} must have dim {expected_dim}")
    return clean_rows


def _json_dumps(payload):
    return json.dumps(payload, ensure_ascii=True)


def _extract_token_from_headers(headers):
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip()
    x_token = headers.get("X-Auth-Token", "")
    if x_token:
        return x_token.strip()
    return ""


def _auth_from_request(headers, db_path):
    token = _extract_token_from_headers(headers)
    if not token:
        return {"ok": False, "error": "missing token"}
    return authenticate_api_token(token=token, db_path=db_path, touch=True)


def _bootstrap_registry(
    db_path: str,
    owner_username: str,
    bootstrap_password: str,
    project_name: str,
    model_name: str,
    model_path: str,
    dsl_path: str,
    expected_dim: int | None,
):
    init_db(db_path)
    accounts = list_accounts(db_path=db_path)
    owner_names = {str(a.get("username", "")) for a in accounts}
    if owner_username not in owner_names:
        create_account(
            username=owner_username,
            password=bootstrap_password,
            db_path=db_path,
            role="admin",
            preferred_lang="en",
        )
    ensure_project(
        owner_username=owner_username,
        name=project_name,
        description="Auto-bootstrapped project for champion runtime.",
        stack_json='["python","pytorch","neurodsl"]',
        db_path=db_path,
    )
    models = list_models(db_path=db_path, project_name=project_name, owner_username=owner_username)
    checkpoint_abs = str(Path(model_path).resolve())
    already = False
    for m in models:
        if str(m.get("checkpoint_path", "")) == checkpoint_abs and str(m.get("name", "")) == model_name:
            already = True
            break
    if not already:
        metrics = {"expected_dim": expected_dim, "runtime": "champion_interact", "auto_registered": True}
        dsl_text = ""
        try:
            dsl_text = Path(dsl_path).read_text(encoding="utf-8")
        except Exception:
            dsl_text = ""
        register_model(
            project_name=project_name,
            model_name=model_name,
            checkpoint_path=checkpoint_abs,
            dsl=dsl_text,
            metrics_json=_json_dumps(metrics),
            agent_name="runtime_server",
            db_path=db_path,
            owner_username=owner_username,
        )


def _qint(query: dict, key: str, default: int, low: int = 1, high: int = 100000) -> int:
    raw = (query.get(key) or [str(default)])[0]
    try:
        val = int(raw)
    except Exception:
        val = default
    return max(low, min(high, val))


def _run_server(
    model,
    expected_dim,
    device,
    host,
    port,
    db_path,
    require_auth,
    registry_owner,
    registry_project,
    registry_model_name,
    model_path,
    dsl_path,
    bootstrap_password,
):
    init_db(db_path)
    _bootstrap_registry(
        db_path=db_path,
        owner_username=registry_owner,
        bootstrap_password=bootstrap_password,
        project_name=registry_project,
        model_name=registry_model_name,
        model_path=model_path,
        dsl_path=dsl_path,
        expected_dim=expected_dim,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _auth(self):
            return _auth_from_request(self.headers, db_path=db_path)

        def _need_auth(self):
            auth = self._auth()
            if not auth.get("ok"):
                self._send(401, {"error": auth.get("error", "unauthorized")})
                return None
            return auth

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Auth-Token")
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "device": str(device),
                        "expected_dim": expected_dim,
                        "auth_required": bool(require_auth),
                        "db_path": str(Path(db_path).resolve()),
                        "registry_owner": registry_owner,
                        "registry_project": registry_project,
                        "registry_model_name": registry_model_name,
                        "features": {
                            "infer": True,
                            "benchmark": True,
                            "mc_samples": True,
                            "as_probs": True,
                            "topk": True,
                            "uncertainty": True,
                            "accounts": True,
                            "session_auth": True,
                            "registry": True,
                            "run_logging": True,
                            "tools": [
                                "compare",
                                "sweep",
                                "importance",
                                "sensmap",
                                "pipeline",
                                "interpolate",
                                "stability",
                                "stress",
                                "goalseek",
                                "counterfactual",
                                "pareto",
                                "portfolio",
                                "batchlab",
                                "drift",
                                "sentinel",
                                "cohort",
                                "watchtower",
                                "simlab",
                                "policylab",
                                "profile",
                                "autolab",
                            ],
                            "diagnostics": True,
                        },
                    },
                )
                return
            if path == "/schema":
                self._send(
                    200,
                    {
                        "inputs": {"expected_dim": expected_dim},
                        "infer_body": {
                            "inputs": "[[...], ...] or [...]",
                            "mc_samples": "int >= 1",
                            "as_probs": "bool",
                            "topk": "int >= 1",
                        },
                        "benchmark_body": {
                            "runs": "int >= 1",
                            "inputs": "optional rows (defaults random if expected_dim is known)",
                            "mc_samples": "int >= 1",
                            "as_probs": "bool",
                        },
                        "tool_bodies": {
                            "compare": {
                                "input_a": "[...]",
                                "input_b": "[...]",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                                "topk": "int >= 1",
                            },
                            "sweep": {
                                "base_input": "[...]",
                                "feature_indices": "[int, ...] optional",
                                "radius": "float >= 0",
                                "steps": "int >= 1",
                                "target_index": "optional int",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "importance": {
                                "input": "[...]",
                                "epsilon": "float > 0",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "sensmap": {
                                "base_input": "[...]",
                                "samples": "int >= 1",
                                "noise_std": "float >= 0",
                                "epsilon": "float > 0",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                                "seed": "optional int >= 0",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "pipeline": {
                                "base_input": "[...]",
                                "input_b": "[...] optional",
                                "feature_indices": "[int, ...] optional",
                                "radius": "float >= 0",
                                "steps": "int >= 1",
                                "epsilon": "float > 0",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                            },
                            "interpolate": {
                                "input_a": "[...]",
                                "input_b": "[...]",
                                "steps": "int >= 2",
                                "target_index": "optional int",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "stability": {
                                "input": "[...]",
                                "samples": "int >= 4",
                                "noise_std": "float >= 0",
                                "target_index": "optional int",
                                "topk": "int >= 1",
                                "seed": "optional int >= 0",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "stress": {
                                "input": "[...]",
                                "noise_levels": "[0.0,0.01,...] or csv string",
                                "samples": "int >= 4",
                                "robust_threshold": "float in [0,1]",
                                "target_index": "optional int",
                                "topk": "int >= 1",
                                "seed": "optional int >= 0",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "goalseek": {
                                "base_input": "[...]",
                                "target_index": "optional int",
                                "target_score": "optional float",
                                "feature_indices": "[int, ...] optional",
                                "steps": "int >= 1",
                                "step_size": "float > 0",
                                "radius": "float >= 0",
                                "epsilon": "float > 0",
                                "top_features": "int >= 1",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "counterfactual": {
                                "base_input": "[...]",
                                "desired_index": "optional int (defaults to 2nd-highest base output)",
                                "feature_indices": "[int, ...] optional",
                                "steps": "int >= 1",
                                "step_size": "float > 0",
                                "radius": "float >= 0",
                                "epsilon": "float > 0",
                                "top_features": "int >= 1",
                                "margin": "float >= 0",
                                "l1_penalty": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "pareto": {
                                "base_input": "[...]",
                                "target_index": "optional int",
                                "target_score": "optional float",
                                "feature_indices": "[int, ...] optional",
                                "samples": "int >= 8",
                                "radius": "float >= 0",
                                "sparsity": "float in [0,1)",
                                "l1_penalty": "float >= 0",
                                "uncertainty_penalty": "float >= 0",
                                "top_candidates": "int >= 1",
                                "topk": "int >= 1",
                                "seed": "optional int >= 0",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "portfolio": {
                                "base_input": "[...]",
                                "candidates": "[[...], ...]",
                                "target_index": "optional int",
                                "top_candidates": "int >= 1",
                                "uncertainty_penalty": "float >= 0",
                                "novelty_weight": "float >= 0",
                                "diversity_weight": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "batchlab": {
                                "inputs": "[[...], ...] or csv rows list",
                                "base_input": "[...] optional",
                                "target_index": "optional int",
                                "top_rows": "int >= 1",
                                "outlier_weight": "float >= 0",
                                "centroid_weight": "float >= 0",
                                "uncertainty_penalty": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "drift": {
                                "current_rows": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "sentinel": {
                                "inputs": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "top_rows": "int >= 1",
                                "uncertainty_weight": "float >= 0",
                                "entropy_weight": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "cohort": {
                                "inputs": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "top_groups": "int >= 1",
                                "uncertainty_weight": "float >= 0",
                                "entropy_weight": "float >= 0",
                                "margin_weight": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "watchtower": {
                                "inputs": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                                "top_rows": "int >= 1",
                                "top_groups": "int >= 1",
                                "uncertainty_weight": "float >= 0",
                                "entropy_weight": "float >= 0",
                                "margin_weight": "float >= 0",
                                "drift_weight": "float >= 0",
                                "sentinel_weight": "float >= 0",
                                "cohort_weight": "float >= 0",
                                "medium_threshold": "float >= 0",
                                "high_threshold": "float >= 0",
                                "critical_threshold": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "simlab": {
                                "inputs": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "noise_levels": "[0.0,0.01,...] or csv string",
                                "repeats": "int >= 1",
                                "drift_bias": "float >= 0",
                                "seed": "optional int >= 0",
                                "top_features": "int >= 1",
                                "top_rows": "int >= 1",
                                "top_groups": "int >= 1",
                                "uncertainty_weight": "float >= 0",
                                "entropy_weight": "float >= 0",
                                "margin_weight": "float >= 0",
                                "drift_weight": "float >= 0",
                                "sentinel_weight": "float >= 0",
                                "cohort_weight": "float >= 0",
                                "medium_threshold": "float >= 0",
                                "high_threshold": "float >= 0",
                                "critical_threshold": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "policylab": {
                                "inputs": "[[...], ...] or rows list",
                                "reference_rows": "[[...], ...] optional",
                                "reference_input": "[...] optional",
                                "target_index": "optional int",
                                "noise_levels": "[0.0,0.01,...] or csv string",
                                "repeats": "int >= 1",
                                "drift_bias": "float >= 0",
                                "search_iters": "int >= 1",
                                "max_weight_shift": "float > 0",
                                "threshold_margin": "float >= 0",
                                "seed": "optional int >= 0",
                                "top_features": "int >= 1",
                                "top_rows": "int >= 1",
                                "top_groups": "int >= 1",
                                "uncertainty_weight": "float >= 0",
                                "entropy_weight": "float >= 0",
                                "margin_weight": "float >= 0",
                                "drift_weight": "float >= 0",
                                "sentinel_weight": "float >= 0",
                                "cohort_weight": "float >= 0",
                                "medium_threshold": "float >= 0",
                                "high_threshold": "float >= 0",
                                "critical_threshold": "float >= 0",
                                "topk": "int >= 1",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                            "profile": {
                                "input": "[...] optional",
                                "batch_sizes": "[1,2,4,...] or csv string",
                                "mc_grid": "[1,2,4,...] or csv string",
                                "runs": "int >= 1",
                                "as_probs": "bool",
                            },
                            "autolab": {
                                "base_input": "[...]",
                                "input_b": "[...] optional",
                                "feature_indices": "[int, ...] optional",
                                "radius": "float >= 0",
                                "steps": "int >= 1",
                                "interp_steps": "int >= 2",
                                "epsilon": "float > 0",
                                "target_index": "optional int",
                                "top_features": "int >= 1",
                                "topk": "int >= 1",
                                "stability_samples": "int >= 4",
                                "noise_std": "float >= 0",
                                "seed": "optional int >= 0",
                                "mc_samples": "int >= 1",
                                "as_probs": "bool",
                            },
                        },
                        "auth_endpoints": [
                            "POST /auth/register",
                            "POST /auth/login",
                            "POST /auth/logout",
                            "GET /auth/me",
                        ],
                        "registry_endpoints": [
                            "GET /registry/snapshot",
                            "GET /registry/accounts",
                            "GET /registry/projects",
                            "GET /registry/models",
                            "GET /registry/sessions",
                            "GET /registry/runs",
                            "GET /registry/metrics",
                            "POST /registry/register-model",
                        ],
                        "diagnostics_endpoints": [
                            "GET /diagnostics/runtime",
                        ],
                        "tool_endpoints": [
                            "POST /tools/compare",
                            "POST /tools/sweep",
                            "POST /tools/importance",
                            "POST /tools/sensmap",
                            "POST /tools/pipeline",
                            "POST /tools/interpolate",
                            "POST /tools/stability",
                            "POST /tools/stress",
                            "POST /tools/goalseek",
                            "POST /tools/counterfactual",
                            "POST /tools/pareto",
                            "POST /tools/portfolio",
                            "POST /tools/batchlab",
                            "POST /tools/drift",
                            "POST /tools/sentinel",
                            "POST /tools/cohort",
                            "POST /tools/watchtower",
                            "POST /tools/simlab",
                            "POST /tools/policylab",
                            "POST /tools/profile",
                            "POST /tools/autolab",
                        ],
                    },
                )
                return
            if path == "/diagnostics/runtime":
                self._send(200, {"ok": True, "diagnostics": _runtime_diagnostics(device, expected_dim, db_path)})
                return
            if path == "/auth/me":
                auth = self._need_auth()
                if auth is None:
                    return
                self._send(200, {"ok": True, "session": auth})
                return
            if path == "/registry/snapshot":
                self._send(200, {"ok": True, "snapshot": get_snapshot(db_path=db_path)})
                return
            if path == "/registry/accounts":
                auth = self._need_auth()
                if auth is None:
                    return
                self._send(200, {"ok": True, "accounts": list_accounts(db_path=db_path)})
                return
            if path == "/registry/projects":
                owner = (query.get("owner") or [""])[0]
                self._send(200, {"ok": True, "projects": list_projects(db_path=db_path, owner_username=owner)})
                return
            if path == "/registry/models":
                owner = (query.get("owner") or [""])[0]
                project = (query.get("project") or [""])[0]
                self._send(
                    200,
                    {
                        "ok": True,
                        "models": list_models(
                            db_path=db_path,
                            project_name=project,
                            owner_username=owner,
                        ),
                    },
                )
                return
            if path == "/registry/sessions":
                auth = self._need_auth()
                if auth is None:
                    return
                user = (query.get("username") or [""])[0]
                limit = _qint(query, "limit", 100, low=1, high=1000)
                self._send(
                    200,
                    {"ok": True, "sessions": list_api_sessions(db_path=db_path, username=user, limit=limit)},
                )
                return
            if path == "/registry/runs":
                auth = self._need_auth()
                if auth is None:
                    return
                user = (query.get("username") or [""])[0]
                limit = _qint(query, "limit", 200, low=1, high=5000)
                self._send(
                    200,
                    {"ok": True, "runs": list_model_runs(db_path=db_path, username=user, limit=limit)},
                )
                return
            if path == "/registry/metrics":
                auth = self._need_auth()
                if auth is None:
                    return
                user = (query.get("username") or [""])[0]
                limit = _qint(query, "limit", 2000, low=1, high=20000)
                rows = list_model_runs(db_path=db_path, username=user, limit=limit)
                modes = {}
                total_latency = 0.0
                for row in rows:
                    mode = str(row.get("run_mode", "unknown"))
                    ms = float(row.get("latency_ms", 0.0))
                    total_latency += ms
                    bucket = modes.setdefault(mode, {"count": 0, "avg_latency_ms": 0.0, "total_latency_ms": 0.0})
                    bucket["count"] += 1
                    bucket["total_latency_ms"] += ms
                for mode in modes:
                    c = max(1, int(modes[mode]["count"]))
                    modes[mode]["avg_latency_ms"] = modes[mode]["total_latency_ms"] / c
                payload = {
                    "ok": True,
                    "total_runs": len(rows),
                    "avg_latency_ms": (total_latency / len(rows)) if rows else 0.0,
                    "modes": modes,
                    "latest": rows[0] if rows else {},
                }
                self._send(200, payload)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                ln = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(ln)
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._send(400, {"error": "invalid json"})
                return

            if path == "/auth/register":
                try:
                    username = str(data.get("username", "")).strip()
                    password = str(data.get("password", ""))
                    role = str(data.get("role", "user")).strip() or "user"
                    pref = str(data.get("preferred_lang", "en")).strip() or "en"
                    acc = create_account(
                        username=username,
                        password=password,
                        role=role,
                        preferred_lang=pref,
                        db_path=db_path,
                    )
                    self._send(200, {"ok": True, "account": acc})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            if path == "/auth/login":
                try:
                    username = str(data.get("username", "")).strip()
                    password = str(data.get("password", ""))
                    ttl_hours = int(data.get("ttl_hours", 24))
                    meta = data.get("meta", {})
                    sess = create_api_session(
                        username=username,
                        password=password,
                        db_path=db_path,
                        ttl_hours=ttl_hours,
                        meta_json=_json_dumps(meta if isinstance(meta, dict) else {}),
                    )
                    if not sess.get("ok"):
                        self._send(401, {"ok": False, "error": "invalid credentials"})
                        return
                    self._send(200, sess)
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            if path == "/auth/logout":
                token = str(data.get("token", "")).strip() or _extract_token_from_headers(self.headers)
                out = revoke_api_token(token=token, db_path=db_path)
                code = 200 if out.get("ok") else 400
                self._send(code, out)
                return

            if path == "/registry/register-model":
                auth = self._need_auth()
                if auth is None:
                    return
                try:
                    owner = str(data.get("owner", auth.get("username", registry_owner))).strip() or registry_owner
                    project_name = str(data.get("project_name", registry_project)).strip() or registry_project
                    model_name = str(data.get("model_name", registry_model_name)).strip() or registry_model_name
                    checkpoint_path = str(data.get("checkpoint_path", model_path)).strip() or model_path
                    dsl = str(data.get("dsl", ""))
                    metrics = data.get("metrics", {})
                    ensure_project(
                        owner_username=owner,
                        name=project_name,
                        description="Project created via runtime registry API.",
                        stack_json='["python","pytorch","neurodsl"]',
                        db_path=db_path,
                    )
                    rec = register_model(
                        project_name=project_name,
                        model_name=model_name,
                        checkpoint_path=checkpoint_path,
                        dsl=dsl,
                        metrics_json=_json_dumps(metrics if isinstance(metrics, dict) else {}),
                        agent_name="runtime_api",
                        db_path=db_path,
                        owner_username=owner,
                    )
                    self._send(200, {"ok": True, "model": rec})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            auth = self._auth()
            if require_auth and not auth.get("ok"):
                self._send(401, {"error": auth.get("error", "unauthorized")})
                return
            session_id = str(auth.get("session_id", "")) if auth.get("ok") else ""
            account_id = int(auth.get("account_id")) if auth.get("ok") and auth.get("account_id") is not None else None

            if path == "/infer":
                try:
                    rows = _validate_rows(data.get("inputs"), expected_dim)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(0, int(data.get("topk", 0)))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return

                t0 = time.perf_counter()
                payload = _infer_rows(
                    model,
                    rows,
                    device,
                    mc_samples=mc_samples,
                    as_probs=as_probs,
                    topk=topk,
                )
                payload["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                payload["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="infer",
                        request_obj={
                            "rows": len(rows),
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                            "topk": topk,
                        },
                        response_obj={
                            "stats": payload.get("stats", {}),
                            "out_dim": len(payload.get("outputs", [[0]])[0]) if payload.get("outputs") else 0,
                        },
                        latency_ms=float(payload["latency_ms"]),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, payload)
                return

            if path == "/benchmark":
                try:
                    runs = max(1, int(data.get("runs", 50)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    rows = data.get("inputs")
                    if rows is None:
                        if expected_dim is None:
                            raise ValueError("inputs required when expected_dim is unknown")
                        rows = torch.randn(8, expected_dim).tolist()
                    rows = _validate_rows(rows, expected_dim)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return

                result = _benchmark(
                    model,
                    rows,
                    device,
                    runs=runs,
                    mc_samples=mc_samples,
                    as_probs=as_probs,
                )
                result["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="benchmark",
                        request_obj={
                            "rows": len(rows),
                            "runs": runs,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "avg_ms": result.get("avg_ms", 0.0),
                            "throughput_sps": result.get("throughput_sps", 0.0),
                        },
                        latency_ms=float(result.get("avg_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/compare":
                try:
                    input_a = data.get("input_a")
                    input_b = data.get("input_b")
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(1, int(data.get("topk", 5)))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_compare(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        input_a=input_a,
                        input_b=input_b,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        topk=topk,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="tool_compare",
                        request_obj={
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                            "topk": topk,
                        },
                        response_obj={
                            "l2_delta": result.get("metrics", {}).get("l2_delta", 0.0),
                            "max_abs_delta": result.get("metrics", {}).get("max_abs_delta", 0.0),
                            "topk_delta_count": len(result.get("topk_delta", [])),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/sweep":
                try:
                    base_input = data.get("base_input")
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    radius = float(data.get("radius", 0.3))
                    steps = max(1, int(data.get("steps", 7)))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_sweep(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        feature_indices=feature_indices,
                        radius=radius,
                        steps=steps,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="tool_sweep",
                        request_obj={
                            "radius": radius,
                            "steps": steps,
                            "feature_count": len(result.get("summaries", [])),
                            "target_index": result.get("target_index"),
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "evaluated_rows": result.get("evaluated_rows", 0),
                            "best_feature": result.get("recommended_adjustment", {}).get("feature_index", -1),
                            "best_score": result.get("recommended_adjustment", {}).get("score", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/importance":
                try:
                    vec = data.get("input", data.get("input_vec"))
                    epsilon = float(data.get("epsilon", 0.01))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    top_features = max(1, int(data.get("top_features", 12)))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_importance(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        input_vec=vec,
                        epsilon=epsilon,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        top_features=top_features,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    top = result.get("top_features", [{}])[0] if result.get("top_features") else {}
                    log_model_run(
                        run_mode="tool_importance",
                        request_obj={
                            "epsilon": epsilon,
                            "target_index": result.get("target_index"),
                            "top_features": top_features,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "feature_count": len(result.get("importance", [])),
                            "top_feature": top.get("feature_index", -1),
                            "top_abs_importance": top.get("abs_importance", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/sensmap":
                try:
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    samples = max(1, int(data.get("samples", 16)))
                    noise_std = abs(float(data.get("noise_std", data.get("noise", 0.04))))
                    epsilon = float(data.get("epsilon", 0.01))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    top_features = max(1, int(data.get("top_features", 12)))
                    seed = data.get("seed", None)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_sensmap(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        samples=samples,
                        noise_std=noise_std,
                        epsilon=epsilon,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        top_features=top_features,
                        seed=seed,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    top = result.get("top_features", [{}])[0] if result.get("top_features") else {}
                    log_model_run(
                        run_mode="tool_sensmap",
                        request_obj={
                            "samples": samples,
                            "noise_std": noise_std,
                            "epsilon": epsilon,
                            "target_index": result.get("target_index"),
                            "top_features": top_features,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "target_mean": summary.get("target_mean", 0.0),
                            "target_std": summary.get("target_std", 0.0),
                            "stable_feature_count": summary.get("stable_feature_count", 0),
                            "top_feature": top.get("feature_index", -1),
                            "top_mean_abs_importance": top.get("mean_abs_importance", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/pipeline":
                try:
                    base_input = data.get("base_input", data.get("input"))
                    input_b = data.get("input_b", None)
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    radius = float(data.get("radius", 0.3))
                    steps = max(1, int(data.get("steps", 7)))
                    epsilon = float(data.get("epsilon", 0.01))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(1, int(data.get("topk", 5)))
                    top_features = max(1, int(data.get("top_features", 12)))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_pipeline(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        input_b=input_b,
                        feature_indices=feature_indices,
                        radius=radius,
                        steps=steps,
                        epsilon=epsilon,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        topk=topk,
                        top_features=top_features,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_pipeline",
                        request_obj={
                            "radius": radius,
                            "steps": steps,
                            "epsilon": epsilon,
                            "topk": topk,
                            "top_features": top_features,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "target_index": summary.get("target_index", -1),
                            "expected_gain": summary.get("expected_gain", 0.0),
                            "exploration_score": summary.get("exploration_score", 0.0),
                            "action_count": summary.get("recommended_action_count", 0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/interpolate":
                try:
                    input_a = data.get("input_a", data.get("base_input", data.get("input")))
                    input_b = data.get("input_b")
                    steps = max(2, int(data.get("steps", 9)))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(1, int(data.get("topk", 5)))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_interpolate(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        input_a=input_a,
                        input_b=input_b,
                        steps=steps,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        topk=topk,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    metrics = result.get("metrics", {})
                    stats = result.get("target_stats", {})
                    log_model_run(
                        run_mode="tool_interpolate",
                        request_obj={
                            "steps": steps,
                            "target_index": result.get("target_index"),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "smoothness_ratio": metrics.get("smoothness_ratio", 0.0),
                            "path_length_l2": metrics.get("path_length_l2", 0.0),
                            "target_mean": stats.get("mean", 0.0),
                            "target_std": stats.get("std", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/stability":
                try:
                    vec = data.get("input", data.get("input_vec", data.get("base_input")))
                    samples = max(4, int(data.get("samples", 64)))
                    noise_std = abs(float(data.get("noise_std", data.get("noise", 0.05))))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(1, int(data.get("topk", 5)))
                    seed = data.get("seed", None)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_stability(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        input_vec=vec,
                        samples=samples,
                        noise_std=noise_std,
                        target_index=target_index,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        topk=topk,
                        seed=seed,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    tstats = result.get("target_stats", {})
                    log_model_run(
                        run_mode="tool_stability",
                        request_obj={
                            "samples": samples,
                            "noise_std": noise_std,
                            "target_index": result.get("target_index"),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "robust_score": result.get("robust_score", 0.0),
                            "target_std": tstats.get("std", 0.0),
                            "target_mean": tstats.get("mean", 0.0),
                            "below_base_fraction": tstats.get("below_base_fraction", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/stress":
                try:
                    vec = data.get("input", data.get("input_vec", data.get("base_input")))
                    noise_levels = data.get("noise_levels", data.get("noise_grid", [0.0, 0.01, 0.03, 0.05, 0.1]))
                    samples = max(4, int(data.get("samples", 48)))
                    robust_threshold = float(data.get("robust_threshold", 0.5))
                    target_index = data.get("target_index", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    topk = max(1, int(data.get("topk", 5)))
                    seed = data.get("seed", None)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_stress(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        input_vec=vec,
                        noise_levels=noise_levels,
                        samples=samples,
                        target_index=target_index,
                        robust_threshold=robust_threshold,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        topk=topk,
                        seed=seed,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_stress",
                        request_obj={
                            "noise_levels": result.get("noise_levels", []),
                            "samples": samples,
                            "robust_threshold": robust_threshold,
                            "target_index": result.get("target_index"),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "mean_robust_score": summary.get("mean_robust_score", 0.0),
                            "min_robust_score": summary.get("min_robust_score", 0.0),
                            "robust_auc": summary.get("robust_auc", 0.0),
                            "first_breakdown_noise": summary.get("first_breakdown_noise"),
                            "recommendation": summary.get("recommendation", ""),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/goalseek":
                try:
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    target_index = data.get("target_index", None)
                    target_score = data.get("target_score", None)
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    steps = max(1, int(data.get("steps", 12)))
                    step_size = abs(float(data.get("step_size", data.get("delta", 0.05))))
                    radius = abs(float(data.get("radius", 0.4)))
                    epsilon = float(data.get("epsilon", 0.01))
                    top_features = max(1, int(data.get("top_features", 8)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_goalseek(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        target_index=target_index,
                        target_score=target_score,
                        feature_indices=feature_indices,
                        steps=steps,
                        step_size=step_size,
                        radius=radius,
                        epsilon=epsilon,
                        top_features=top_features,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="tool_goalseek",
                        request_obj={
                            "steps": steps,
                            "step_size": step_size,
                            "radius": radius,
                            "epsilon": epsilon,
                            "target_index": result.get("target_index"),
                            "target_score": target_score,
                            "top_features": top_features,
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "base_target_score": result.get("base_target_score", 0.0),
                            "optimized_target_score": result.get("optimized_target_score", 0.0),
                            "expected_gain": result.get("expected_gain", 0.0),
                            "objective_improvement": result.get("objective_improvement", 0.0),
                            "accepted_steps": result.get("accepted_steps", 0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/counterfactual":
                try:
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    desired_index = data.get("desired_index", data.get("target_index", None))
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    steps = max(1, int(data.get("steps", 14)))
                    step_size = abs(float(data.get("step_size", data.get("delta", 0.04))))
                    radius = abs(float(data.get("radius", 0.35)))
                    epsilon = float(data.get("epsilon", 0.01))
                    top_features = max(1, int(data.get("top_features", 8)))
                    margin = max(0.0, float(data.get("margin", 0.02)))
                    l1_penalty = max(0.0, float(data.get("l1_penalty", 0.05)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_counterfactual(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        desired_index=desired_index,
                        feature_indices=feature_indices,
                        steps=steps,
                        step_size=step_size,
                        radius=radius,
                        epsilon=epsilon,
                        top_features=top_features,
                        margin=margin,
                        l1_penalty=l1_penalty,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    log_model_run(
                        run_mode="tool_counterfactual",
                        request_obj={
                            "desired_index": result.get("desired_index"),
                            "steps": steps,
                            "step_size": step_size,
                            "radius": radius,
                            "epsilon": epsilon,
                            "top_features": top_features,
                            "margin": margin,
                            "l1_penalty": l1_penalty,
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "success": bool(result.get("success", False)),
                            "base_margin": result.get("base_margin", 0.0),
                            "counterfactual_margin": result.get("counterfactual_margin", 0.0),
                            "distance_l1": result.get("distance_l1", 0.0),
                            "distance_l2": result.get("distance_l2", 0.0),
                            "final_predicted_index": result.get("final_predicted_index"),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/pareto":
                try:
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    target_index = data.get("target_index", None)
                    target_score = data.get("target_score", None)
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    samples = max(8, int(data.get("samples", 128)))
                    radius = abs(float(data.get("radius", 0.3)))
                    sparsity = float(data.get("sparsity", 0.75))
                    l1_penalty = max(0.0, float(data.get("l1_penalty", 0.05)))
                    uncertainty_penalty = max(0.0, float(data.get("uncertainty_penalty", 0.10)))
                    top_candidates = max(1, int(data.get("top_candidates", 12)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                    seed = data.get("seed", None)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_pareto(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        target_index=target_index,
                        target_score=target_score,
                        feature_indices=feature_indices,
                        samples=samples,
                        radius=radius,
                        sparsity=sparsity,
                        l1_penalty=l1_penalty,
                        uncertainty_penalty=uncertainty_penalty,
                        top_candidates=top_candidates,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                        seed=seed,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    best = result.get("best_candidate", {})
                    log_model_run(
                        run_mode="tool_pareto",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "target_score_goal": result.get("target_score_goal"),
                            "samples": result.get("samples"),
                            "radius": result.get("radius"),
                            "sparsity": result.get("sparsity"),
                            "l1_penalty": result.get("l1_penalty"),
                            "uncertainty_penalty": result.get("uncertainty_penalty"),
                            "top_candidates": result.get("top_candidates"),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "best_utility": summary.get("best_utility", 0.0),
                            "best_target_score": summary.get("best_target_score", 0.0),
                            "improvement_vs_base": summary.get("improvement_vs_base", 0.0),
                            "pareto_count": summary.get("pareto_count", 0),
                            "best_changed_count": best.get("changed_count", 0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/portfolio":
                try:
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    candidates = data.get("candidates", data.get("inputs", data.get("rows")))
                    target_index = data.get("target_index", None)
                    top_candidates = max(1, int(data.get("top_candidates", 8)))
                    uncertainty_penalty = max(0.0, float(data.get("uncertainty_penalty", 0.10)))
                    novelty_weight = max(0.0, float(data.get("novelty_weight", 0.15)))
                    diversity_weight = max(0.0, float(data.get("diversity_weight", 0.10)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_portfolio(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        candidates=candidates,
                        target_index=target_index,
                        top_candidates=top_candidates,
                        uncertainty_penalty=uncertainty_penalty,
                        novelty_weight=novelty_weight,
                        diversity_weight=diversity_weight,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    best = result.get("best_candidate", {})
                    log_model_run(
                        run_mode="tool_portfolio",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "top_candidates": result.get("top_candidates"),
                            "candidate_count": result.get("candidate_count", 0),
                            "uncertainty_penalty": result.get("uncertainty_penalty", 0.0),
                            "novelty_weight": result.get("novelty_weight", 0.0),
                            "diversity_weight": result.get("diversity_weight", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "best_utility": summary.get("best_utility", 0.0),
                            "best_target_score": summary.get("best_target_score", 0.0),
                            "improvement_vs_base": summary.get("improvement_vs_base", 0.0),
                            "selected_count": summary.get("selected_count", 0),
                            "selected_diversity_l2": summary.get("selected_diversity_l2", 0.0),
                            "best_candidate_index": best.get("candidate_index", -1),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/batchlab":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("candidates")))
                    base_input = data.get("base_input", data.get("input", data.get("input_vec")))
                    target_index = data.get("target_index", None)
                    top_rows = max(1, int(data.get("top_rows", data.get("top_candidates", 8))))
                    outlier_weight = max(0.0, float(data.get("outlier_weight", 0.20)))
                    centroid_weight = max(0.0, float(data.get("centroid_weight", 0.10)))
                    uncertainty_penalty = max(0.0, float(data.get("uncertainty_penalty", 0.10)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_batchlab(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        base_input=base_input,
                        target_index=target_index,
                        top_rows=top_rows,
                        outlier_weight=outlier_weight,
                        centroid_weight=centroid_weight,
                        uncertainty_penalty=uncertainty_penalty,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    best = result.get("best_row", {})
                    log_model_run(
                        run_mode="tool_batchlab",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "row_count": result.get("row_count", 0),
                            "top_rows": result.get("top_rows", 0),
                            "outlier_weight": result.get("outlier_weight", 0.0),
                            "centroid_weight": result.get("centroid_weight", 0.0),
                            "uncertainty_penalty": result.get("uncertainty_penalty", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "best_row_index": summary.get("best_row_index", -1),
                            "best_utility": summary.get("best_utility", 0.0),
                            "best_target_score": summary.get("best_target_score", 0.0),
                            "target_mean": summary.get("target_mean", 0.0),
                            "target_std": summary.get("target_std", 0.0),
                            "selected_count": summary.get("selected_count", 0),
                            "best_predicted_index": best.get("predicted_index", -1),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/drift":
                try:
                    current_rows = data.get("current_rows", data.get("inputs", data.get("rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    top_features = max(1, int(data.get("top_features", 12)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_drift(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        current_rows=current_rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        top_features=top_features,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_drift",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "reference_rows": result.get("reference_row_count", 0),
                            "current_rows": result.get("current_row_count", 0),
                            "top_features": top_features,
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "drift_score": summary.get("drift_score", 0.0),
                            "target_delta": summary.get("target_delta", 0.0),
                            "output_l2_delta": summary.get("output_l2_delta", 0.0),
                            "js_divergence": summary.get("js_divergence", 0.0),
                            "recommendation": summary.get("recommendation", "monitor"),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/sentinel":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("current_rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    top_rows = max(1, int(data.get("top_rows", data.get("top_candidates", 8))))
                    uncertainty_weight = max(0.0, float(data.get("uncertainty_weight", 0.50)))
                    entropy_weight = max(0.0, float(data.get("entropy_weight", 1.00)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_sentinel(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        top_rows=top_rows,
                        uncertainty_weight=uncertainty_weight,
                        entropy_weight=entropy_weight,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    top = result.get("top_anomalies", [])
                    worst = top[0] if top else {}
                    log_model_run(
                        run_mode="tool_sentinel",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "rows": result.get("row_count", 0),
                            "reference_rows": result.get("reference_row_count", 0),
                            "top_rows": result.get("top_rows", 0),
                            "uncertainty_weight": result.get("uncertainty_weight", 0.0),
                            "entropy_weight": result.get("entropy_weight", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "anomaly_mean": summary.get("anomaly_mean", 0.0),
                            "anomaly_max": summary.get("anomaly_max", 0.0),
                            "high_risk_count": summary.get("high_risk_count", 0),
                            "high_risk_fraction": summary.get("high_risk_fraction", 0.0),
                            "recommendation": summary.get("recommendation", "normal_monitoring"),
                            "worst_row_index": worst.get("row_index", -1),
                            "worst_anomaly_score": worst.get("anomaly_score", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/cohort":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("current_rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    top_groups = max(1, int(data.get("top_groups", data.get("top_rows", 6))))
                    uncertainty_weight = max(0.0, float(data.get("uncertainty_weight", 0.50)))
                    entropy_weight = max(0.0, float(data.get("entropy_weight", 0.50)))
                    margin_weight = max(0.0, float(data.get("margin_weight", 0.30)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_cohort(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        top_groups=top_groups,
                        uncertainty_weight=uncertainty_weight,
                        entropy_weight=entropy_weight,
                        margin_weight=margin_weight,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    top = result.get("top_risky_cohorts", [])
                    worst = top[0] if top else {}
                    log_model_run(
                        run_mode="tool_cohort",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "rows": result.get("row_count", 0),
                            "reference_rows": result.get("reference_row_count", 0),
                            "cohort_count": result.get("cohort_count", 0),
                            "top_groups": result.get("top_groups", 0),
                            "uncertainty_weight": result.get("uncertainty_weight", 0.0),
                            "entropy_weight": result.get("entropy_weight", 0.0),
                            "margin_weight": result.get("margin_weight", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "dominant_predicted_index": summary.get("dominant_predicted_index", 0),
                            "dominant_share": summary.get("dominant_share", 0.0),
                            "top_risk_predicted_index": summary.get("top_risk_predicted_index", 0),
                            "top_risk_score": summary.get("top_risk_score", 0.0),
                            "high_risk_fraction": summary.get("high_risk_fraction", 0.0),
                            "recommendation": summary.get("recommendation", "cohorts_stable"),
                            "worst_predicted_index": worst.get("predicted_index", -1),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/watchtower":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("current_rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    top_features = max(1, int(data.get("top_features", 12)))
                    top_rows = max(1, int(data.get("top_rows", 8)))
                    top_groups = max(1, int(data.get("top_groups", 6)))
                    uncertainty_weight = max(0.0, float(data.get("uncertainty_weight", 0.50)))
                    entropy_weight = max(0.0, float(data.get("entropy_weight", 0.50)))
                    margin_weight = max(0.0, float(data.get("margin_weight", 0.30)))
                    drift_weight = max(0.0, float(data.get("drift_weight", 1.00)))
                    sentinel_weight = max(0.0, float(data.get("sentinel_weight", 1.00)))
                    cohort_weight = max(0.0, float(data.get("cohort_weight", 1.00)))
                    medium_threshold = max(0.0, float(data.get("medium_threshold", 0.35)))
                    high_threshold = max(0.0, float(data.get("high_threshold", 0.60)))
                    critical_threshold = max(0.0, float(data.get("critical_threshold", 0.85)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_watchtower(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        top_features=top_features,
                        top_rows=top_rows,
                        top_groups=top_groups,
                        uncertainty_weight=uncertainty_weight,
                        entropy_weight=entropy_weight,
                        margin_weight=margin_weight,
                        drift_weight=drift_weight,
                        sentinel_weight=sentinel_weight,
                        cohort_weight=cohort_weight,
                        medium_threshold=medium_threshold,
                        high_threshold=high_threshold,
                        critical_threshold=critical_threshold,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_watchtower",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "rows": result.get("row_count", 0),
                            "reference_rows": result.get("reference_row_count", 0),
                            "top_features": top_features,
                            "top_rows": top_rows,
                            "top_groups": top_groups,
                            "drift_weight": drift_weight,
                            "sentinel_weight": sentinel_weight,
                            "cohort_weight": cohort_weight,
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "risk_level": summary.get("risk_level", "low"),
                            "combined_risk_score": summary.get("combined_risk_score", 0.0),
                            "severe_signal_count": summary.get("severe_signal_count", 0),
                            "recommendation": summary.get("recommendation", "normal_monitoring"),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/simlab":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("current_rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    noise_levels = data.get("noise_levels", data.get("levels", "0,0.01,0.03,0.05,0.1"))
                    repeats = max(1, int(data.get("repeats", 3)))
                    drift_bias = max(0.0, float(data.get("drift_bias", 0.02)))
                    seed = data.get("seed", None)
                    top_features = max(1, int(data.get("top_features", 12)))
                    top_rows = max(1, int(data.get("top_rows", 8)))
                    top_groups = max(1, int(data.get("top_groups", 6)))
                    uncertainty_weight = max(0.0, float(data.get("uncertainty_weight", 0.50)))
                    entropy_weight = max(0.0, float(data.get("entropy_weight", 0.50)))
                    margin_weight = max(0.0, float(data.get("margin_weight", 0.30)))
                    drift_weight = max(0.0, float(data.get("drift_weight", 1.00)))
                    sentinel_weight = max(0.0, float(data.get("sentinel_weight", 1.00)))
                    cohort_weight = max(0.0, float(data.get("cohort_weight", 1.00)))
                    medium_threshold = max(0.0, float(data.get("medium_threshold", 0.35)))
                    high_threshold = max(0.0, float(data.get("high_threshold", 0.60)))
                    critical_threshold = max(0.0, float(data.get("critical_threshold", 0.85)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_simlab(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        noise_levels=noise_levels,
                        repeats=repeats,
                        drift_bias=drift_bias,
                        seed=seed,
                        top_features=top_features,
                        top_rows=top_rows,
                        top_groups=top_groups,
                        uncertainty_weight=uncertainty_weight,
                        entropy_weight=entropy_weight,
                        margin_weight=margin_weight,
                        drift_weight=drift_weight,
                        sentinel_weight=sentinel_weight,
                        cohort_weight=cohort_weight,
                        medium_threshold=medium_threshold,
                        high_threshold=high_threshold,
                        critical_threshold=critical_threshold,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_simlab",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "rows": result.get("row_count", 0),
                            "reference_rows": result.get("reference_row_count", 0),
                            "noise_levels": result.get("noise_levels", []),
                            "repeats": result.get("repeats", 1),
                            "drift_bias": result.get("drift_bias", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "resilience_score": summary.get("resilience_score", 0.0),
                            "resilience_grade": summary.get("resilience_grade", "moderate"),
                            "risk_auc": summary.get("risk_auc", 0.0),
                            "first_high_noise": summary.get("first_high_noise"),
                            "first_critical_noise": summary.get("first_critical_noise"),
                            "recommendation": summary.get("recommendation", "maintain_and_monitor"),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/policylab":
                try:
                    rows = data.get("inputs", data.get("rows", data.get("current_rows")))
                    reference_rows = data.get("reference_rows", data.get("baseline_rows"))
                    reference_input = data.get("reference_input", data.get("base_input", data.get("input")))
                    target_index = data.get("target_index", None)
                    noise_levels = data.get("noise_levels", data.get("levels", "0,0.01,0.03,0.05,0.1"))
                    repeats = max(1, int(data.get("repeats", 3)))
                    drift_bias = max(0.0, float(data.get("drift_bias", 0.02)))
                    search_iters = max(1, int(data.get("search_iters", 10)))
                    max_weight_shift = max(0.01, float(data.get("max_weight_shift", 0.35)))
                    threshold_margin = max(0.0, float(data.get("threshold_margin", 0.06)))
                    seed = data.get("seed", None)
                    top_features = max(1, int(data.get("top_features", 12)))
                    top_rows = max(1, int(data.get("top_rows", 8)))
                    top_groups = max(1, int(data.get("top_groups", 6)))
                    uncertainty_weight = max(0.0, float(data.get("uncertainty_weight", 0.50)))
                    entropy_weight = max(0.0, float(data.get("entropy_weight", 0.50)))
                    margin_weight = max(0.0, float(data.get("margin_weight", 0.30)))
                    drift_weight = max(0.0, float(data.get("drift_weight", 1.00)))
                    sentinel_weight = max(0.0, float(data.get("sentinel_weight", 1.00)))
                    cohort_weight = max(0.0, float(data.get("cohort_weight", 1.00)))
                    medium_threshold = max(0.0, float(data.get("medium_threshold", 0.35)))
                    high_threshold = max(0.0, float(data.get("high_threshold", 0.60)))
                    critical_threshold = max(0.0, float(data.get("critical_threshold", 0.85)))
                    topk = max(1, int(data.get("topk", 5)))
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_policylab(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        rows=rows,
                        reference_rows=reference_rows,
                        reference_input=reference_input,
                        target_index=target_index,
                        noise_levels=noise_levels,
                        repeats=repeats,
                        drift_bias=drift_bias,
                        search_iters=search_iters,
                        max_weight_shift=max_weight_shift,
                        threshold_margin=threshold_margin,
                        seed=seed,
                        top_features=top_features,
                        top_rows=top_rows,
                        top_groups=top_groups,
                        uncertainty_weight=uncertainty_weight,
                        entropy_weight=entropy_weight,
                        margin_weight=margin_weight,
                        drift_weight=drift_weight,
                        sentinel_weight=sentinel_weight,
                        cohort_weight=cohort_weight,
                        medium_threshold=medium_threshold,
                        high_threshold=high_threshold,
                        critical_threshold=critical_threshold,
                        topk=topk,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_policylab",
                        request_obj={
                            "target_index": result.get("target_index"),
                            "rows": result.get("row_count", 0),
                            "reference_rows": result.get("reference_row_count", 0),
                            "noise_levels": result.get("noise_levels", []),
                            "repeats": result.get("repeats", 1),
                            "search_iters": result.get("search_iters", 1),
                            "max_weight_shift": result.get("max_weight_shift", 0.0),
                            "threshold_margin": result.get("threshold_margin", 0.0),
                            "topk": topk,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "baseline_objective_score": summary.get("baseline_objective_score", 0.0),
                            "recommended_objective_score": summary.get("recommended_objective_score", 0.0),
                            "objective_improvement": summary.get("objective_improvement", 0.0),
                            "adopt_recommended_policy": summary.get("adopt_recommended_policy", False),
                            "recommendation": summary.get("recommendation", "keep_baseline_policy"),
                            "best_policy_name": summary.get("best_policy_name", "baseline"),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/profile":
                try:
                    base_input = data.get("input", data.get("base_input"))
                    batch_sizes = data.get("batch_sizes", data.get("batches", [1, 2, 4, 8, 16]))
                    mc_grid = data.get("mc_grid", data.get("mc_samples_grid", [1, 2, 4]))
                    runs = max(1, int(data.get("runs", 30)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_profile(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        batch_sizes=batch_sizes,
                        mc_grid=mc_grid,
                        runs=runs,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    rec = result.get("recommended", {})
                    log_model_run(
                        run_mode="tool_profile",
                        request_obj={
                            "case_count": result.get("case_count", 0),
                            "runs": runs,
                            "batch_sizes": result.get("batch_sizes", []),
                            "mc_grid": result.get("mc_grid", []),
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "best_latency_ms": rec.get("lowest_latency", {}).get("avg_ms", 0.0),
                            "best_throughput_sps": rec.get("highest_throughput", {}).get("throughput_sps", 0.0),
                            "best_efficiency": rec.get("best_efficiency", {}).get("efficiency_score", 0.0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            if path == "/tools/autolab":
                try:
                    base_input = data.get("base_input", data.get("input"))
                    input_b = data.get("input_b", None)
                    feature_indices = _normalize_feature_indices(data.get("feature_indices"))
                    radius = float(data.get("radius", 0.3))
                    steps = max(1, int(data.get("steps", 7)))
                    interp_steps = max(2, int(data.get("interp_steps", 9)))
                    epsilon = float(data.get("epsilon", 0.01))
                    target_index = data.get("target_index", None)
                    top_features = max(1, int(data.get("top_features", 12)))
                    topk = max(1, int(data.get("topk", 5)))
                    stability_samples = max(4, int(data.get("stability_samples", data.get("samples", 64))))
                    noise_std = abs(float(data.get("noise_std", data.get("noise", 0.05))))
                    seed = data.get("seed", None)
                    mc_samples = max(1, int(data.get("mc_samples", 1)))
                    as_probs = _as_bool(data.get("as_probs", False))
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                t0 = time.perf_counter()
                try:
                    result = _tool_autolab(
                        model=model,
                        device=device,
                        expected_dim=expected_dim,
                        base_input=base_input,
                        input_b=input_b,
                        feature_indices=feature_indices,
                        radius=radius,
                        steps=steps,
                        interp_steps=interp_steps,
                        epsilon=epsilon,
                        target_index=target_index,
                        top_features=top_features,
                        topk=topk,
                        stability_samples=stability_samples,
                        noise_std=noise_std,
                        seed=seed,
                        mc_samples=mc_samples,
                        as_probs=as_probs,
                    )
                except Exception as exc:
                    self._send(400, {"error": str(exc)})
                    return
                result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                result["session_id"] = session_id
                try:
                    summary = result.get("summary", {})
                    log_model_run(
                        run_mode="tool_autolab",
                        request_obj={
                            "radius": radius,
                            "steps": steps,
                            "interp_steps": interp_steps,
                            "epsilon": epsilon,
                            "stability_samples": stability_samples,
                            "noise_std": noise_std,
                            "mc_samples": mc_samples,
                            "as_probs": as_probs,
                        },
                        response_obj={
                            "strategy": summary.get("strategy", ""),
                            "composite_score": summary.get("composite_score", 0.0),
                            "expected_gain": summary.get("expected_gain", 0.0),
                            "robust_score": summary.get("robust_score", 0.0),
                            "action_count": summary.get("action_count", 0),
                        },
                        latency_ms=float(result.get("latency_ms", 0.0)),
                        db_path=db_path,
                        session_id=session_id,
                        account_id=account_id,
                    )
                except Exception:
                    pass
                self._send(200, result)
                return

            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"[server] http://{host}:{port}")
    print("[server] GET /health, /schema, /registry/*, /diagnostics/runtime | POST /auth/*, /infer, /benchmark, /tools/*, /registry/register-model")
    print(f"[server] auth_required={bool(require_auth)} db={Path(db_path).resolve()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[server] stopping...")
    finally:
        srv.server_close()


def _run_repl(model, expected_dim, device):
    print(
        "Champion REPL ready. Commands: infer <csv>, compare <csvA>|<csvB>, sweep <csv> [radius] [steps] [target], "
        "importance <csv> [epsilon] [topn] [target], pipeline <csvA>|<csvB> [target], "
        "sensmap <csv> [samples] [noise] [epsilon] [target], "
        "interpolate <csvA>|<csvB> [steps] [target], stability <csv> [samples] [noise] [target], "
        "stress <csv> [noise_csv] [samples] [target], goalseek <csv> [target] [goal] [steps] [step_size], "
        "counterfactual <csv> [desired] [steps] [step_size] [margin] [l1_penalty], random, bench [runs], "
        "pareto <csv> [target] [samples] [radius] [sparsity], "
        "portfolio <base_csv> <candidates.csv> [topn] [target], profile [batches_csv] [mc_csv] [runs], "
        "batchlab <in.csv> [topn] [target], "
        "drift <current.csv> [reference.csv] [top_features] [target], "
        "sentinel <current.csv> [reference.csv] [top_rows] [target], "
        "cohort <current.csv> [reference.csv] [top_groups] [target], "
        "watchtower <current.csv> [reference.csv] [top_rows] [target], "
        "simlab <current.csv> [reference.csv] [noise_csv] [target], "
        "policylab <current.csv> [reference.csv] [noise_csv] [target], "
        "autolab <csvA>|<csvB> [target], batch <in.csv> [out.csv], help, exit"
    )
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
            print(
                "infer <v1,v2,...> | compare <v1,..>|<v1,..> | sweep <v1,..> [radius] [steps] [target] "
                "| importance <v1,..> [epsilon] [topn] [target] | pipeline <v1,..>|<u1,..> [target] "
                "| sensmap <v1,..> [samples] [noise] [epsilon] [target] "
                "| interpolate <v1,..>|<u1,..> [steps] [target] | stability <v1,..> [samples] [noise] [target] "
                "| stress <v1,..> [noise_csv] [samples] [target] | goalseek <v1,..> [target] [goal] [steps] [step_size] "
                "| counterfactual <v1,..> [desired] [steps] [step_size] [margin] [l1_penalty] "
                "| pareto <v1,..> [target] [samples] [radius] [sparsity] "
                "| portfolio <v1,..> <candidates.csv> [topn] [target] "
                "| batchlab <in.csv> [topn] [target] "
                "| drift <current.csv> [reference.csv] [top_features] [target] "
                "| sentinel <current.csv> [reference.csv] [top_rows] [target] "
                "| cohort <current.csv> [reference.csv] [top_groups] [target] "
                "| watchtower <current.csv> [reference.csv] [top_rows] [target] "
                "| simlab <current.csv> [reference.csv] [noise_csv] [target] "
                "| policylab <current.csv> [reference.csv] [noise_csv] [target] "
                "| profile [batches_csv] [mc_csv] [runs] | autolab <v1,..>|<u1,..> [target] "
                "| random | bench [runs] | batch <in.csv> [out.csv] | exit"
            )
            continue
        if line == "random":
            if expected_dim is None:
                print("[error] random requires known input dim")
                continue
            rows = torch.randn(1, expected_dim).tolist()
            res = _infer_rows(model, rows, device)
            print(json.dumps({"inputs": rows[0], "outputs": res["outputs"][0], "stds": res["stds"][0]}, indent=2))
            continue
        if line.startswith("bench"):
            parts = line.split()
            runs = int(parts[1]) if len(parts) > 1 else 50
            if expected_dim is None:
                print("[error] bench requires known input dim")
                continue
            rows = torch.randn(4, expected_dim).tolist()
            bench = _benchmark(model, rows, device, runs=max(1, runs))
            print(f"[bench] avg={bench['avg_ms']:.3f}ms min={bench['min_ms']:.3f} max={bench['max_ms']:.3f} throughput={bench['throughput_sps']:.2f} sps")
            continue
        if line.startswith("infer "):
            raw = line[len("infer "):].strip()
            vals = _parse_vector(raw)
            if expected_dim is not None and len(vals) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vals)}")
                continue
            res = _infer_rows(model, [vals], device, topk=5)
            print(json.dumps({"output": res["outputs"][0], "std": res["stds"][0], "topk": res.get("topk", [])[0]}, indent=2))
            continue
        if line.startswith("compare "):
            raw = line[len("compare "):].strip()
            if "|" not in raw:
                print("[error] usage: compare <v1,v2,...>|<u1,u2,...>")
                continue
            left, right = raw.split("|", 1)
            vec_a = _parse_vector(left.strip())
            vec_b = _parse_vector(right.strip())
            if expected_dim is not None and (len(vec_a) != expected_dim or len(vec_b) != expected_dim):
                print(f"[error] expected dim {expected_dim}, got {len(vec_a)} and {len(vec_b)}")
                continue
            out = _tool_compare(
                model=model,
                device=device,
                expected_dim=expected_dim,
                input_a=vec_a,
                input_b=vec_b,
                mc_samples=1,
                as_probs=False,
                topk=5,
            )
            print(
                json.dumps(
                    {
                        "metrics": out.get("metrics", {}),
                        "topk_delta": out.get("topk_delta", []),
                        "topk_a": out.get("topk_a", []),
                        "topk_b": out.get("topk_b", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("sweep "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: sweep <v1,v2,...> [radius] [steps] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            radius = float(parts[2]) if len(parts) > 2 else 0.3
            steps = int(parts[3]) if len(parts) > 3 else 7
            target = int(parts[4]) if len(parts) > 4 else None
            out = _tool_sweep(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                feature_indices=None,
                radius=radius,
                steps=steps,
                target_index=target,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "recommended_adjustment": out.get("recommended_adjustment", {}),
                        "worst_adjustment": out.get("worst_adjustment", {}),
                        "top_sensitivity": out.get("summaries", [])[:8],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("importance "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: importance <v1,v2,...> [epsilon] [topn] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            eps = float(parts[2]) if len(parts) > 2 else 0.01
            topn = int(parts[3]) if len(parts) > 3 else 12
            target = int(parts[4]) if len(parts) > 4 else None
            out = _tool_importance(
                model=model,
                device=device,
                expected_dim=expected_dim,
                input_vec=vec,
                epsilon=eps,
                target_index=target,
                mc_samples=1,
                as_probs=False,
                top_features=topn,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "base_target_score": out.get("base_target_score"),
                        "top_features": out.get("top_features", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("sensmap "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: sensmap <v1,..> [samples] [noise] [epsilon] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            samples = int(parts[2]) if len(parts) > 2 else 16
            noise = float(parts[3]) if len(parts) > 3 else 0.04
            eps = float(parts[4]) if len(parts) > 4 else 0.01
            target = int(parts[5]) if len(parts) > 5 else None
            out = _tool_sensmap(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                samples=max(1, int(samples)),
                noise_std=abs(float(noise)),
                epsilon=float(eps),
                target_index=target,
                mc_samples=1,
                as_probs=False,
                top_features=12,
                seed=42,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "top_features": out.get("top_features", [])[:8],
                        "recommended_probes": out.get("recommended_probes", [])[:8],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("pipeline "):
            raw = line[len("pipeline "):].strip()
            if not raw:
                print("[error] usage: pipeline <v1,..>|<u1,..> [target]")
                continue
            parts = raw.split()
            vec_spec = parts[0]
            target = int(parts[1]) if len(parts) > 1 else None
            if "|" in vec_spec:
                left, right = vec_spec.split("|", 1)
            else:
                left, right = vec_spec, ""
            vec_a = _parse_vector(left.strip())
            vec_b = _parse_vector(right.strip()) if right.strip() else None
            if expected_dim is not None and len(vec_a) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec_a)}")
                continue
            if expected_dim is not None and vec_b is not None and len(vec_b) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec_b)}")
                continue
            out = _tool_pipeline(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec_a,
                input_b=vec_b,
                feature_indices=None,
                radius=0.3,
                steps=7,
                epsilon=0.01,
                target_index=target,
                mc_samples=1,
                as_probs=False,
                topk=5,
                top_features=8,
            )
            print(
                json.dumps(
                    {
                        "summary": out.get("summary", {}),
                        "recommended_actions": out.get("recommended_actions", []),
                        "top_importance": out.get("importance", {}).get("top_features", [])[:5],
                        "top_sweep": out.get("sweep", {}).get("summaries", [])[:5],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("interpolate "):
            raw = line[len("interpolate "):].strip()
            if "|" not in raw:
                print("[error] usage: interpolate <v1,..>|<u1,..> [steps] [target]")
                continue
            parts = raw.split()
            vec_spec = parts[0]
            steps = int(parts[1]) if len(parts) > 1 else 9
            target = int(parts[2]) if len(parts) > 2 else None
            left, right = vec_spec.split("|", 1)
            vec_a = _parse_vector(left.strip())
            vec_b = _parse_vector(right.strip())
            if expected_dim is not None and (len(vec_a) != expected_dim or len(vec_b) != expected_dim):
                print(f"[error] expected dim {expected_dim}, got {len(vec_a)} and {len(vec_b)}")
                continue
            out = _tool_interpolate(
                model=model,
                device=device,
                expected_dim=expected_dim,
                input_a=vec_a,
                input_b=vec_b,
                steps=max(2, int(steps)),
                target_index=target,
                mc_samples=1,
                as_probs=False,
                topk=5,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "target_stats": out.get("target_stats", {}),
                        "metrics": out.get("metrics", {}),
                        "start_topk": out.get("start_topk", []),
                        "end_topk": out.get("end_topk", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("stability "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: stability <v1,..> [samples] [noise] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            samples = int(parts[2]) if len(parts) > 2 else 64
            noise = float(parts[3]) if len(parts) > 3 else 0.05
            target = int(parts[4]) if len(parts) > 4 else None
            out = _tool_stability(
                model=model,
                device=device,
                expected_dim=expected_dim,
                input_vec=vec,
                samples=max(4, int(samples)),
                noise_std=abs(float(noise)),
                target_index=target,
                mc_samples=1,
                as_probs=False,
                topk=5,
                seed=42,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "base_target_score": out.get("base_target_score"),
                        "target_stats": out.get("target_stats", {}),
                        "robust_score": out.get("robust_score"),
                        "recommendation": out.get("recommendation"),
                        "worst_output_shift": out.get("worst_output_shift", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("stress "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: stress <v1,..> [noise_csv] [samples] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            noise_levels = parts[2] if len(parts) > 2 else "0,0.01,0.03,0.05,0.1"
            samples = int(parts[3]) if len(parts) > 3 else 48
            target = int(parts[4]) if len(parts) > 4 else None
            out = _tool_stress(
                model=model,
                device=device,
                expected_dim=expected_dim,
                input_vec=vec,
                noise_levels=noise_levels,
                samples=max(4, int(samples)),
                target_index=target,
                robust_threshold=0.5,
                mc_samples=1,
                as_probs=False,
                topk=5,
                seed=42,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "levels": out.get("levels", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("goalseek "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: goalseek <v1,..> [target] [goal] [steps] [step_size]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            target = int(parts[2]) if len(parts) > 2 else None
            goal = float(parts[3]) if len(parts) > 3 else None
            steps = int(parts[4]) if len(parts) > 4 else 12
            step_size = float(parts[5]) if len(parts) > 5 else 0.05
            out = _tool_goalseek(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                target_index=target,
                target_score=goal,
                feature_indices=None,
                steps=max(1, int(steps)),
                step_size=abs(float(step_size)),
                radius=0.4,
                epsilon=0.01,
                top_features=8,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "base_target_score": out.get("base_target_score"),
                        "optimized_target_score": out.get("optimized_target_score"),
                        "expected_gain": out.get("expected_gain"),
                        "objective_improvement": out.get("objective_improvement"),
                        "accepted_steps": out.get("accepted_steps"),
                        "final_topk": out.get("final_topk", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("counterfactual "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: counterfactual <v1,..> [desired] [steps] [step_size] [margin] [l1_penalty]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            desired = int(parts[2]) if len(parts) > 2 else None
            steps = int(parts[3]) if len(parts) > 3 else 14
            step_size = float(parts[4]) if len(parts) > 4 else 0.04
            margin = float(parts[5]) if len(parts) > 5 else 0.02
            l1_penalty = float(parts[6]) if len(parts) > 6 else 0.05
            out = _tool_counterfactual(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                desired_index=desired,
                feature_indices=None,
                steps=max(1, int(steps)),
                step_size=abs(float(step_size)),
                radius=0.35,
                epsilon=0.01,
                top_features=8,
                margin=max(0.0, float(margin)),
                l1_penalty=max(0.0, float(l1_penalty)),
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "desired_index": out.get("desired_index"),
                        "base_predicted_index": out.get("base_predicted_index"),
                        "final_predicted_index": out.get("final_predicted_index"),
                        "base_margin": out.get("base_margin"),
                        "counterfactual_margin": out.get("counterfactual_margin"),
                        "distance_l1": out.get("distance_l1"),
                        "success": out.get("success"),
                        "changed_features": out.get("changed_features", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("pareto "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: pareto <v1,..> [target] [samples] [radius] [sparsity]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            target = int(parts[2]) if len(parts) > 2 else None
            samples = int(parts[3]) if len(parts) > 3 else 128
            radius = float(parts[4]) if len(parts) > 4 else 0.3
            sparsity = float(parts[5]) if len(parts) > 5 else 0.75
            out = _tool_pareto(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                target_index=target,
                target_score=None,
                feature_indices=None,
                samples=max(8, int(samples)),
                radius=abs(float(radius)),
                sparsity=float(max(0.0, min(0.99, float(sparsity)))),
                l1_penalty=0.05,
                uncertainty_penalty=0.10,
                top_candidates=8,
                topk=5,
                mc_samples=1,
                as_probs=False,
                seed=42,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "best_candidate": out.get("best_candidate", {}),
                        "pareto_front": out.get("pareto_front", [])[:5],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("portfolio "):
            parts = line.split()
            if len(parts) < 3:
                print("[error] usage: portfolio <v1,..> <candidates.csv> [topn] [target]")
                continue
            vec = _parse_vector(parts[1])
            if expected_dim is not None and len(vec) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec)}")
                continue
            candidate_csv = parts[2]
            topn = int(parts[3]) if len(parts) > 3 else 8
            target = int(parts[4]) if len(parts) > 4 else None
            rows = _load_csv_rows(candidate_csv)
            out = _tool_portfolio(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec,
                candidates=rows,
                target_index=target,
                top_candidates=max(1, int(topn)),
                uncertainty_penalty=0.10,
                novelty_weight=0.15,
                diversity_weight=0.10,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "best_candidate": out.get("best_candidate", {}),
                        "selected_portfolio": out.get("selected_portfolio", [])[:5],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("batchlab "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: batchlab <in.csv> [topn] [target]")
                continue
            in_csv = parts[1]
            topn = int(parts[2]) if len(parts) > 2 else 8
            target = int(parts[3]) if len(parts) > 3 else None
            rows = _load_csv_rows(in_csv)
            out = _tool_batchlab(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=rows,
                base_input=None,
                target_index=target,
                top_rows=max(1, int(topn)),
                outlier_weight=0.20,
                centroid_weight=0.10,
                uncertainty_penalty=0.10,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "best_row": out.get("best_row", {}),
                        "selected_rows": out.get("selected_rows", [])[:5],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("drift "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: drift <current.csv> [reference.csv] [top_features] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            if len(parts) > 2:
                try:
                    _ = int(parts[2])
                except Exception:
                    ref_csv = parts[2]
                    idx = 3
            topn = int(parts[idx]) if len(parts) > idx else 12
            target = int(parts[idx + 1]) if len(parts) > (idx + 1) else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_drift(
                model=model,
                device=device,
                expected_dim=expected_dim,
                current_rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                top_features=max(1, int(topn)),
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "top_feature_shift": out.get("top_feature_shift", [])[:8],
                        "top_output_shift": out.get("top_output_shift", [])[:8],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("sentinel "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: sentinel <current.csv> [reference.csv] [top_rows] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            if len(parts) > 2:
                try:
                    _ = int(parts[2])
                except Exception:
                    ref_csv = parts[2]
                    idx = 3
            topn = int(parts[idx]) if len(parts) > idx else 8
            target = int(parts[idx + 1]) if len(parts) > (idx + 1) else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_sentinel(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                top_rows=max(1, int(topn)),
                uncertainty_weight=0.50,
                entropy_weight=1.00,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "top_anomalies": out.get("top_anomalies", [])[:8],
                        "stable_rows": out.get("stable_rows", [])[:8],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("cohort "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: cohort <current.csv> [reference.csv] [top_groups] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            if len(parts) > 2:
                try:
                    _ = int(parts[2])
                except Exception:
                    ref_csv = parts[2]
                    idx = 3
            topn = int(parts[idx]) if len(parts) > idx else 6
            target = int(parts[idx + 1]) if len(parts) > (idx + 1) else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_cohort(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                top_groups=max(1, int(topn)),
                uncertainty_weight=0.50,
                entropy_weight=0.50,
                margin_weight=0.30,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "top_risky_cohorts": out.get("top_risky_cohorts", [])[:8],
                        "dominant_cohort": out.get("dominant_cohort", {}),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("watchtower "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: watchtower <current.csv> [reference.csv] [top_rows] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            if len(parts) > 2:
                try:
                    _ = int(parts[2])
                except Exception:
                    ref_csv = parts[2]
                    idx = 3
            topn = int(parts[idx]) if len(parts) > idx else 8
            target = int(parts[idx + 1]) if len(parts) > (idx + 1) else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_watchtower(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                top_features=max(4, int(topn) * 2),
                top_rows=max(1, int(topn)),
                top_groups=max(1, int(topn)),
                uncertainty_weight=0.50,
                entropy_weight=0.50,
                margin_weight=0.30,
                drift_weight=1.00,
                sentinel_weight=1.00,
                cohort_weight=1.00,
                medium_threshold=0.35,
                high_threshold=0.60,
                critical_threshold=0.85,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "signals": out.get("signals", []),
                        "action_plan": out.get("action_plan", []),
                        "top_watch_items": out.get("top_watch_items", {}),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("simlab "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: simlab <current.csv> [reference.csv] [noise_csv] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            noise_levels = "0,0.01,0.03,0.05,0.1"
            if len(parts) > 2:
                p2 = parts[2]
                if "," in p2:
                    noise_levels = p2
                    idx = 3
                else:
                    try:
                        _ = int(p2)
                    except Exception:
                        ref_csv = p2
                        idx = 3
            if len(parts) > idx and "," in parts[idx]:
                noise_levels = parts[idx]
                idx += 1
            target = int(parts[idx]) if len(parts) > idx else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_simlab(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                noise_levels=noise_levels,
                repeats=2,
                drift_bias=0.02,
                seed=42,
                top_features=12,
                top_rows=8,
                top_groups=6,
                uncertainty_weight=0.50,
                entropy_weight=0.50,
                margin_weight=0.30,
                drift_weight=1.00,
                sentinel_weight=1.00,
                cohort_weight=1.00,
                medium_threshold=0.35,
                high_threshold=0.60,
                critical_threshold=0.85,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "risk_curve": out.get("risk_curve", []),
                        "action_plan": out.get("action_plan", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("policylab "):
            parts = line.split()
            if len(parts) < 2:
                print("[error] usage: policylab <current.csv> [reference.csv] [noise_csv] [target]")
                continue
            current_csv = parts[1]
            ref_csv = None
            idx = 2
            noise_levels = "0,0.01,0.03,0.05,0.1"
            if len(parts) > 2:
                p2 = parts[2]
                if "," in p2:
                    noise_levels = p2
                    idx = 3
                else:
                    try:
                        _ = int(p2)
                    except Exception:
                        ref_csv = p2
                        idx = 3
            if len(parts) > idx and "," in parts[idx]:
                noise_levels = parts[idx]
                idx += 1
            target = int(parts[idx]) if len(parts) > idx else None
            cur_rows = _load_csv_rows(current_csv)
            ref_rows = _load_csv_rows(ref_csv) if ref_csv else None
            out = _tool_policylab(
                model=model,
                device=device,
                expected_dim=expected_dim,
                rows=cur_rows,
                reference_rows=ref_rows,
                reference_input=None,
                target_index=target,
                noise_levels=noise_levels,
                repeats=2,
                drift_bias=0.02,
                search_iters=8,
                max_weight_shift=0.35,
                threshold_margin=0.06,
                seed=42,
                top_features=12,
                top_rows=8,
                top_groups=6,
                uncertainty_weight=0.50,
                entropy_weight=0.50,
                margin_weight=0.30,
                drift_weight=1.00,
                sentinel_weight=1.00,
                cohort_weight=1.00,
                medium_threshold=0.35,
                high_threshold=0.60,
                critical_threshold=0.85,
                topk=5,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "target_index": out.get("target_index"),
                        "summary": out.get("summary", {}),
                        "baseline_policy": out.get("baseline_policy", {}),
                        "recommended_policy": out.get("recommended_policy", {}),
                        "action_plan": out.get("action_plan", []),
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("profile"):
            parts = line.split()
            batches = parts[1] if len(parts) > 1 else "1,2,4,8,16"
            mc_grid = parts[2] if len(parts) > 2 else "1,2,4"
            runs = int(parts[3]) if len(parts) > 3 else 20
            out = _tool_profile(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=None,
                batch_sizes=batches,
                mc_grid=mc_grid,
                runs=max(1, int(runs)),
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "case_count": out.get("case_count"),
                        "recommended": out.get("recommended", {}),
                        "pareto_frontier": out.get("pareto_frontier", [])[:10],
                    },
                    indent=2,
                )
            )
            continue
        if line.startswith("autolab "):
            raw = line[len("autolab "):].strip()
            if not raw:
                print("[error] usage: autolab <v1,..>|<u1,..> [target]")
                continue
            parts = raw.split()
            vec_spec = parts[0]
            target = int(parts[1]) if len(parts) > 1 else None
            if "|" in vec_spec:
                left, right = vec_spec.split("|", 1)
            else:
                left, right = vec_spec, ""
            vec_a = _parse_vector(left.strip())
            vec_b = _parse_vector(right.strip()) if right.strip() else None
            if expected_dim is not None and len(vec_a) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec_a)}")
                continue
            if expected_dim is not None and vec_b is not None and len(vec_b) != expected_dim:
                print(f"[error] expected dim {expected_dim}, got {len(vec_b)}")
                continue
            out = _tool_autolab(
                model=model,
                device=device,
                expected_dim=expected_dim,
                base_input=vec_a,
                input_b=vec_b,
                feature_indices=None,
                radius=0.3,
                steps=7,
                interp_steps=9,
                epsilon=0.01,
                target_index=target,
                top_features=8,
                topk=5,
                stability_samples=64,
                noise_std=0.05,
                seed=42,
                mc_samples=1,
                as_probs=False,
            )
            print(
                json.dumps(
                    {
                        "summary": out.get("summary", {}),
                        "action_plan": out.get("action_plan", []),
                        "pipeline_summary": out.get("pipeline", {}).get("summary", {}),
                    },
                    indent=2,
                )
            )
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
                        raise ValueError(f"row {i} has dim {len(row)}, expected {expected_dim}")
            res = _infer_rows(model, rows, device)
            _write_csv(out_csv, res["outputs"], res["stds"])
            print(f"[ok] wrote {out_csv} ({len(res['outputs'])} rows)")
            continue
        print("[error] unknown command; type help")


def main():
    parser = argparse.ArgumentParser(description="Champion model runner")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dsl", default=DEFAULT_DSL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--mode",
        default="repl",
        choices=[
            "repl",
            "diagnose",
            "infer",
            "compare",
            "sweep",
            "importance",
            "sensmap",
            "pipeline",
            "interpolate",
            "stability",
            "stress",
            "goalseek",
            "counterfactual",
            "pareto",
            "portfolio",
            "batchlab",
            "drift",
            "sentinel",
            "cohort",
            "watchtower",
            "simlab",
            "policylab",
            "profile",
            "autolab",
            "batch",
            "benchmark",
            "serve",
        ],
    )
    parser.add_argument("--input", default="")
    parser.add_argument("--input-b", default="")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--input-b-csv", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--profile-batches", default="1,2,4,8,16")
    parser.add_argument("--profile-mc-grid", default="1,2,4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--as-probs", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--interp-steps", type=int, default=9)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--noise-levels", default="0,0.01,0.03,0.05,0.1")
    parser.add_argument("--robust-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--goal-step-size", type=float, default=0.05)
    parser.add_argument("--target-score", type=float, default=float("nan"))
    parser.add_argument("--cf-margin", type=float, default=0.02)
    parser.add_argument("--cf-l1-penalty", type=float, default=0.05)
    parser.add_argument("--search-sparsity", type=float, default=0.75)
    parser.add_argument("--search-l1-penalty", type=float, default=0.05)
    parser.add_argument("--search-uncertainty-penalty", type=float, default=0.10)
    parser.add_argument("--search-top-candidates", type=int, default=12)
    parser.add_argument("--portfolio-top-candidates", type=int, default=8)
    parser.add_argument("--portfolio-uncertainty-penalty", type=float, default=0.10)
    parser.add_argument("--portfolio-novelty-weight", type=float, default=0.15)
    parser.add_argument("--portfolio-diversity-weight", type=float, default=0.10)
    parser.add_argument("--batchlab-top-rows", type=int, default=8)
    parser.add_argument("--batchlab-outlier-weight", type=float, default=0.20)
    parser.add_argument("--batchlab-centroid-weight", type=float, default=0.10)
    parser.add_argument("--batchlab-uncertainty-penalty", type=float, default=0.10)
    parser.add_argument("--drift-top-features", type=int, default=12)
    parser.add_argument("--sentinel-top-rows", type=int, default=8)
    parser.add_argument("--sentinel-uncertainty-weight", type=float, default=0.50)
    parser.add_argument("--sentinel-entropy-weight", type=float, default=1.00)
    parser.add_argument("--cohort-top-groups", type=int, default=6)
    parser.add_argument("--cohort-uncertainty-weight", type=float, default=0.50)
    parser.add_argument("--cohort-entropy-weight", type=float, default=0.50)
    parser.add_argument("--cohort-margin-weight", type=float, default=0.30)
    parser.add_argument("--watchtower-drift-weight", type=float, default=1.00)
    parser.add_argument("--watchtower-sentinel-weight", type=float, default=1.00)
    parser.add_argument("--watchtower-cohort-weight", type=float, default=1.00)
    parser.add_argument("--watchtower-medium-threshold", type=float, default=0.35)
    parser.add_argument("--watchtower-high-threshold", type=float, default=0.60)
    parser.add_argument("--watchtower-critical-threshold", type=float, default=0.85)
    parser.add_argument("--simlab-repeats", type=int, default=3)
    parser.add_argument("--simlab-drift-bias", type=float, default=0.02)
    parser.add_argument("--policylab-search-iters", type=int, default=10)
    parser.add_argument("--policylab-max-weight-shift", type=float, default=0.35)
    parser.add_argument("--policylab-threshold-margin", type=float, default=0.06)
    parser.add_argument("--feature-indices", default="")
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--registry-owner", default=DEFAULT_REGISTRY_OWNER)
    parser.add_argument("--registry-project", default=DEFAULT_REGISTRY_PROJECT)
    parser.add_argument("--registry-model-name", default="champion_runtime")
    parser.add_argument("--bootstrap-password", default="changeme123")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, expected_dim = _load_model(args.model, args.dsl, device)
    print(f"[info] model loaded on {device}")
    if expected_dim is not None:
        print(f"[info] expected input dim: {expected_dim}")

    if args.mode == "serve":
        _run_server(
            model=model,
            expected_dim=expected_dim,
            device=device,
            host=args.host,
            port=args.port,
            db_path=args.db_path,
            require_auth=bool(args.require_auth),
            registry_owner=str(args.registry_owner).strip() or DEFAULT_REGISTRY_OWNER,
            registry_project=str(args.registry_project).strip() or DEFAULT_REGISTRY_PROJECT,
            registry_model_name=str(args.registry_model_name).strip() or "champion_runtime",
            model_path=_resolve_path(args.model),
            dsl_path=_resolve_path(args.dsl),
            bootstrap_password=str(args.bootstrap_password),
        )
        return

    if args.mode == "repl":
        _run_repl(model, expected_dim, device)
        return

    if args.mode == "diagnose":
        payload = _runtime_diagnostics(device=device, expected_dim=expected_dim, db_path=args.db_path)
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "infer":
        if not args.input:
            raise ValueError("--input is required for infer mode")
        vals = _parse_vector(args.input)
        if expected_dim is not None and len(vals) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vals)}")
        res = _infer_rows(
            model,
            [vals],
            device,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(0, int(args.topk)),
        )
        payload = {
            "input": vals,
            "output": res["outputs"][0],
            "std": res["stds"][0],
            "stats": res["stats"],
            "mc_samples": res["mc_samples"],
            "as_probs": res["as_probs"],
        }
        if res.get("topk"):
            payload["topk"] = res["topk"][0]
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if args.mode == "compare":
        if not args.input or not args.input_b:
            raise ValueError("compare mode needs --input and --input-b")
        vec_a = _parse_vector(args.input)
        vec_b = _parse_vector(args.input_b)
        if expected_dim is not None and (len(vec_a) != expected_dim or len(vec_b) != expected_dim):
            raise ValueError(f"expected dim {expected_dim}, got {len(vec_a)} and {len(vec_b)}")
        payload = _tool_compare(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_a=vec_a,
            input_b=vec_b,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(1, int(args.topk or 5)),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "sweep":
        if not args.input:
            raise ValueError("sweep mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        payload = _tool_sweep(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            feature_indices=feature_indices,
            radius=float(args.radius),
            steps=max(1, int(args.steps)),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "importance":
        if not args.input:
            raise ValueError("importance mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_importance(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=vec,
            epsilon=float(args.epsilon),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            top_features=max(1, int(args.top_features)),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "sensmap":
        if not args.input:
            raise ValueError("sensmap mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_sensmap(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            samples=max(1, int(args.samples)),
            noise_std=abs(float(args.noise_std)),
            epsilon=float(args.epsilon),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            top_features=max(1, int(args.top_features)),
            seed=seed,
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "pipeline":
        if not args.input:
            raise ValueError("pipeline mode needs --input")
        vec = _parse_vector(args.input)
        vec_b = _parse_vector(args.input_b) if args.input_b else None
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        if expected_dim is not None and vec_b is not None and len(vec_b) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec_b)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        payload = _tool_pipeline(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            input_b=vec_b,
            feature_indices=feature_indices,
            radius=float(args.radius),
            steps=max(1, int(args.steps)),
            epsilon=float(args.epsilon),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(1, int(args.topk or 5)),
            top_features=max(1, int(args.top_features)),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "interpolate":
        if not args.input or not args.input_b:
            raise ValueError("interpolate mode needs --input and --input-b")
        vec_a = _parse_vector(args.input)
        vec_b = _parse_vector(args.input_b)
        if expected_dim is not None and (len(vec_a) != expected_dim or len(vec_b) != expected_dim):
            raise ValueError(f"expected dim {expected_dim}, got {len(vec_a)} and {len(vec_b)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_interpolate(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_a=vec_a,
            input_b=vec_b,
            steps=max(2, int(args.steps)),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(1, int(args.topk or 5)),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "stability":
        if not args.input:
            raise ValueError("stability mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_stability(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=vec,
            samples=max(4, int(args.samples)),
            noise_std=abs(float(args.noise_std)),
            target_index=target_index,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(1, int(args.topk or 5)),
            seed=seed,
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "stress":
        if not args.input:
            raise ValueError("stress mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_stress(
            model=model,
            device=device,
            expected_dim=expected_dim,
            input_vec=vec,
            noise_levels=args.noise_levels,
            samples=max(4, int(args.samples)),
            target_index=target_index,
            robust_threshold=float(args.robust_threshold),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(1, int(args.topk or 5)),
            seed=seed,
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "goalseek":
        if not args.input:
            raise ValueError("goalseek mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        target_score = None
        try:
            if math.isfinite(float(args.target_score)):
                target_score = float(args.target_score)
        except Exception:
            target_score = None
        payload = _tool_goalseek(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            target_index=target_index,
            target_score=target_score,
            feature_indices=feature_indices,
            steps=max(1, int(args.steps)),
            step_size=abs(float(args.goal_step_size)),
            radius=abs(float(args.radius)),
            epsilon=float(args.epsilon),
            top_features=max(1, int(args.top_features)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "counterfactual":
        if not args.input:
            raise ValueError("counterfactual mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        desired_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        payload = _tool_counterfactual(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            desired_index=desired_index,
            feature_indices=feature_indices,
            steps=max(1, int(args.steps)),
            step_size=abs(float(args.goal_step_size)),
            radius=abs(float(args.radius)),
            epsilon=float(args.epsilon),
            top_features=max(1, int(args.top_features)),
            margin=max(0.0, float(args.cf_margin)),
            l1_penalty=max(0.0, float(args.cf_l1_penalty)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "pareto":
        if not args.input:
            raise ValueError("pareto mode needs --input")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        target_score = None
        try:
            if math.isfinite(float(args.target_score)):
                target_score = float(args.target_score)
        except Exception:
            target_score = None
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_pareto(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            target_index=target_index,
            target_score=target_score,
            feature_indices=feature_indices,
            samples=max(8, int(args.samples)),
            radius=abs(float(args.radius)),
            sparsity=float(max(0.0, min(0.99, float(args.search_sparsity)))),
            l1_penalty=max(0.0, float(args.search_l1_penalty)),
            uncertainty_penalty=max(0.0, float(args.search_uncertainty_penalty)),
            top_candidates=max(1, int(args.search_top_candidates)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            seed=seed,
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "portfolio":
        if not args.input:
            raise ValueError("portfolio mode needs --input")
        if not args.input_csv:
            raise ValueError("portfolio mode needs --input-csv with candidate rows")
        vec = _parse_vector(args.input)
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        rows = _load_csv_rows(args.input_csv)
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_portfolio(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            candidates=rows,
            target_index=target_index,
            top_candidates=max(1, int(args.portfolio_top_candidates)),
            uncertainty_penalty=max(0.0, float(args.portfolio_uncertainty_penalty)),
            novelty_weight=max(0.0, float(args.portfolio_novelty_weight)),
            diversity_weight=max(0.0, float(args.portfolio_diversity_weight)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "batchlab":
        if not args.input_csv:
            raise ValueError("batchlab mode needs --input-csv")
        rows = _load_csv_rows(args.input_csv)
        base_vec = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and base_vec is not None and len(base_vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(base_vec)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_batchlab(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=rows,
            base_input=base_vec,
            target_index=target_index,
            top_rows=max(1, int(args.batchlab_top_rows)),
            outlier_weight=max(0.0, float(args.batchlab_outlier_weight)),
            centroid_weight=max(0.0, float(args.batchlab_centroid_weight)),
            uncertainty_penalty=max(0.0, float(args.batchlab_uncertainty_penalty)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "drift":
        if not args.input_csv:
            raise ValueError("drift mode needs --input-csv for current rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_drift(
            model=model,
            device=device,
            expected_dim=expected_dim,
            current_rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            top_features=max(1, int(args.drift_top_features)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "sentinel":
        if not args.input_csv:
            raise ValueError("sentinel mode needs --input-csv for rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_sentinel(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            top_rows=max(1, int(args.sentinel_top_rows)),
            uncertainty_weight=max(0.0, float(args.sentinel_uncertainty_weight)),
            entropy_weight=max(0.0, float(args.sentinel_entropy_weight)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "cohort":
        if not args.input_csv:
            raise ValueError("cohort mode needs --input-csv for rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_cohort(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            top_groups=max(1, int(args.cohort_top_groups)),
            uncertainty_weight=max(0.0, float(args.cohort_uncertainty_weight)),
            entropy_weight=max(0.0, float(args.cohort_entropy_weight)),
            margin_weight=max(0.0, float(args.cohort_margin_weight)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "watchtower":
        if not args.input_csv:
            raise ValueError("watchtower mode needs --input-csv for rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        payload = _tool_watchtower(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            top_features=max(1, int(args.drift_top_features)),
            top_rows=max(1, int(args.sentinel_top_rows)),
            top_groups=max(1, int(args.cohort_top_groups)),
            uncertainty_weight=max(0.0, float(args.cohort_uncertainty_weight)),
            entropy_weight=max(0.0, float(args.cohort_entropy_weight)),
            margin_weight=max(0.0, float(args.cohort_margin_weight)),
            drift_weight=max(0.0, float(args.watchtower_drift_weight)),
            sentinel_weight=max(0.0, float(args.watchtower_sentinel_weight)),
            cohort_weight=max(0.0, float(args.watchtower_cohort_weight)),
            medium_threshold=max(0.0, float(args.watchtower_medium_threshold)),
            high_threshold=max(0.0, float(args.watchtower_high_threshold)),
            critical_threshold=max(0.0, float(args.watchtower_critical_threshold)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "simlab":
        if not args.input_csv:
            raise ValueError("simlab mode needs --input-csv for rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_simlab(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            noise_levels=args.noise_levels,
            repeats=max(1, int(args.simlab_repeats)),
            drift_bias=max(0.0, float(args.simlab_drift_bias)),
            seed=seed,
            top_features=max(1, int(args.drift_top_features)),
            top_rows=max(1, int(args.sentinel_top_rows)),
            top_groups=max(1, int(args.cohort_top_groups)),
            uncertainty_weight=max(0.0, float(args.cohort_uncertainty_weight)),
            entropy_weight=max(0.0, float(args.cohort_entropy_weight)),
            margin_weight=max(0.0, float(args.cohort_margin_weight)),
            drift_weight=max(0.0, float(args.watchtower_drift_weight)),
            sentinel_weight=max(0.0, float(args.watchtower_sentinel_weight)),
            cohort_weight=max(0.0, float(args.watchtower_cohort_weight)),
            medium_threshold=max(0.0, float(args.watchtower_medium_threshold)),
            high_threshold=max(0.0, float(args.watchtower_high_threshold)),
            critical_threshold=max(0.0, float(args.watchtower_critical_threshold)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "policylab":
        if not args.input_csv:
            raise ValueError("policylab mode needs --input-csv for rows")
        cur_rows = _load_csv_rows(args.input_csv)
        ref_rows = _load_csv_rows(args.input_b_csv) if args.input_b_csv else None
        ref_input = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and ref_input is not None and len(ref_input) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(ref_input)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_policylab(
            model=model,
            device=device,
            expected_dim=expected_dim,
            rows=cur_rows,
            reference_rows=ref_rows,
            reference_input=ref_input,
            target_index=target_index,
            noise_levels=args.noise_levels,
            repeats=max(1, int(args.simlab_repeats)),
            drift_bias=max(0.0, float(args.simlab_drift_bias)),
            search_iters=max(1, int(args.policylab_search_iters)),
            max_weight_shift=max(0.01, float(args.policylab_max_weight_shift)),
            threshold_margin=max(0.0, float(args.policylab_threshold_margin)),
            seed=seed,
            top_features=max(1, int(args.drift_top_features)),
            top_rows=max(1, int(args.sentinel_top_rows)),
            top_groups=max(1, int(args.cohort_top_groups)),
            uncertainty_weight=max(0.0, float(args.cohort_uncertainty_weight)),
            entropy_weight=max(0.0, float(args.cohort_entropy_weight)),
            margin_weight=max(0.0, float(args.cohort_margin_weight)),
            drift_weight=max(0.0, float(args.watchtower_drift_weight)),
            sentinel_weight=max(0.0, float(args.watchtower_sentinel_weight)),
            cohort_weight=max(0.0, float(args.watchtower_cohort_weight)),
            medium_threshold=max(0.0, float(args.watchtower_medium_threshold)),
            high_threshold=max(0.0, float(args.watchtower_high_threshold)),
            critical_threshold=max(0.0, float(args.watchtower_critical_threshold)),
            topk=max(1, int(args.topk or 5)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "profile":
        base_vec = _parse_vector(args.input) if args.input else None
        if expected_dim is not None and base_vec is not None and len(base_vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(base_vec)}")
        payload = _tool_profile(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=base_vec,
            batch_sizes=args.profile_batches,
            mc_grid=args.profile_mc_grid,
            runs=max(1, int(args.runs)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "autolab":
        if not args.input:
            raise ValueError("autolab mode needs --input")
        vec = _parse_vector(args.input)
        vec_b = _parse_vector(args.input_b) if args.input_b else None
        if expected_dim is not None and len(vec) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec)}")
        if expected_dim is not None and vec_b is not None and len(vec_b) != expected_dim:
            raise ValueError(f"expected dim {expected_dim}, got {len(vec_b)}")
        target_index = None if int(args.target_index) < 0 else int(args.target_index)
        feature_indices = _normalize_feature_indices(args.feature_indices)
        seed = int(args.seed) if int(args.seed) >= 0 else None
        payload = _tool_autolab(
            model=model,
            device=device,
            expected_dim=expected_dim,
            base_input=vec,
            input_b=vec_b,
            feature_indices=feature_indices,
            radius=float(args.radius),
            steps=max(1, int(args.steps)),
            interp_steps=max(2, int(args.interp_steps)),
            epsilon=float(args.epsilon),
            target_index=target_index,
            top_features=max(1, int(args.top_features)),
            topk=max(1, int(args.topk or 5)),
            stability_samples=max(4, int(args.samples)),
            noise_std=abs(float(args.noise_std)),
            seed=seed,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(json.dumps(payload, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "batch":
        if not args.input_csv:
            raise ValueError("--input-csv is required for batch mode")
        rows = _load_csv_rows(args.input_csv)
        if expected_dim is not None:
            for i, row in enumerate(rows):
                if len(row) != expected_dim:
                    raise ValueError(f"row {i} has dim {len(row)}, expected {expected_dim}")
        res = _infer_rows(
            model,
            rows,
            device,
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
            topk=max(0, int(args.topk)),
        )
        payload = {
            "rows": len(res["outputs"]),
            "outputs": res["outputs"],
            "stds": res["stds"],
            "stats": res["stats"],
            "mc_samples": res["mc_samples"],
            "as_probs": res["as_probs"],
        }
        if res.get("topk"):
            payload["topk"] = res["topk"]
        print(f"[info] rows={payload['rows']}")
        if args.output_csv:
            _write_csv(args.output_csv, payload["outputs"], payload["stds"])
            print(f"[ok] wrote {args.output_csv}")
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return

    if args.mode == "benchmark":
        if expected_dim is None:
            raise ValueError("benchmark mode needs known input dim")
        rows = torch.randn(8, expected_dim).tolist()
        bench = _benchmark(
            model,
            rows,
            device,
            runs=max(1, int(args.runs)),
            mc_samples=max(1, int(args.mc_samples)),
            as_probs=bool(args.as_probs),
        )
        print(
            "[bench] avg={avg:.3f} ms/run min={minv:.3f} max={maxv:.3f} throughput={th:.2f} samples/sec (batch={bs})".format(
                avg=bench["avg_ms"],
                minv=bench["min_ms"],
                maxv=bench["max_ms"],
                th=bench["throughput_sps"],
                bs=bench["batch_size"],
            )
        )
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(bench, indent=2), encoding="utf-8")
            print(f"[ok] wrote {args.output_json}")
        return


if __name__ == "__main__":
    main()
