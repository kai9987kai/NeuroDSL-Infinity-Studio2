from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

import torch

from network import ModernMLP


PRESETS_FILE = "user_presets.json"


DSL_PRESETS: Dict[str, str] = {
    "Classifier (MLP)": "[128, 64], dropout: [0.2], [64, 32], [32, 10]",
    "Deep Classifier": "[256, 128], residual: [128], dropout: [0.3], [128, 64], [64, 10]",
    "AutoEncoder": "[784, 256], [256, 64], [64, 256], [256, 784]",
    "Transformer Block": "fractal: [256, 2], trans: [256], [256, 10]",
    "MoE Heavy": "[128, 256], moe: [256, 8], dropout: [0.1], [256, 10]",
    "Adaptive Creative": "[128, 256], moe: [256, 10, 2], mod: [256, 4, 0.4], gqa: [256, 8, 2], [256, 10]",
    "Attention Pipeline": "[64, 128], gqa: [128, 8, 2], residual: [128], [128, 10]",
    "Conv-LSTM Hybrid": "[64, 128], lstm: [128], [128, 10]",
    "MoE-Heavy": "moe: [128, 8, 2], moe: [128, 8, 2], [128, 10]",
    "Fractal Deep": "fractal: [64, 4], [64, 10]",
    "Kitchen Sink": "fractal: [256, 2], moe: [256, 8, 1], mod: [256, 4, 0.35], gqa: [256, 8, 2], residual: [256], conv1d: [256, 5], lstm: [256], dropout: [0.1], [256, 10]",
    "Omni-Model (SOTA)": "[16, 32], conv3d: [32, 3], conv1d: [32, 3], trans: [32], moe: [32, 4, 1], fractal: [32, 2], lstm: [32, 2], [32, 10]",
    "ASI Omni-Intelligence": "mamba: [256], hyper: [256], liquid: [256], trans: [256], moe: [256, 16, 2], [256, 10]",
    "Quantum-Fractal ASI": "quantum: [128], fractal_synth: [64], [64, 10]",
    "Research Frontier": "kan: [128], diff_attn: [128], lora: [128, 16], [128, 10]",
    "Stable Training": "[128, 256], specnorm: [256], gcp: [256], residual: [256], [256, 10]",
    "Ultra-Efficient": "bitlinear: [128], bitlinear: [128], retention: [128], [128, 10]",
    "RetNet-Style": "[128, 256], retention: [256, 8], mix_depth: [256], retention: [256, 8], [256, 10]",
    "Symbolic Reasoner": "[128, 64], logic: [64, 16], graph: [64], concept: [64], [64, 10]",
    "Manifold Explorer": "[128, 64], sphere: [64], poincare: [64], topo_attn: [64, 4], [64, 10]",
    "Singularity Nexus": "[128, 64], holographic: [64], alchemy: [64], logic: [64, 16], [64, 10]",
    "Ethereal Synthesis": "[128, 64], hypernet: [64], node: [64], xfusion: [64], [64, 10]",
    "Ethereal Flow": "[128, 64], turbulence: [64], fluid: [64], fluid: [64], [64, 10]",
    "Frontier Intelligence": "gla: [128, 4], xlstm: [128], ttt: [128], sparse_attn: [128, 8], [128, 10]",
    "Adaptive Nexus": "hyena: [128], geglu: [128], conv_mixer: [128], stoch_depth: [128], [128, 10]",
    "Singularity Synthesis": "mhla: [128, 8, 32, 64], mambaconv: [128], sparse_k: [128, 16], saliency: [128, 0.3], [128, 10]",
    "Singularity Horizon": "mhla: [256, 16], mambaconv: [256, 32], saliency: [256], bitnet: [256, 128], [128, 10]",
    "BitNet Efficiency": "bitnet: [128, 256], bitnet: [256, 128], [128, 10]",
    "Reversible Nexus": "rev_res: [256], rev_res: [256], [256, 128], [128, 10]",
    "Mixture of Attention": "moa: [128], bitnet: [128, 64], [64, 10]",
    "Cross-Attention Module": "[64, 128], crossattn: [128, 8], [128, 10]",
    "Gated Cross-Attention": "[64, 128], gatedcrossattn: [128, 8], [128, 10]",
    "Complex Neural Module": "[64, 256], complex: [256, 8, 4.0, 3], [256, 10]",
    "Multimodal Fusion": "[64, 128], crossattn: [128, 8], gatedcrossattn: [128, 4], complex: [128, 8, 2.0, 2], [128, 10]",
    "Innovation Helix": "[128, 256], kan: [256, 7, 3], retention: [256, 8], diff_logic: [256, 24], adaptive_rank: [256, 32], [256, 10]",
    "Singularity Synthesis+": "[128, 256], mhla: [256, 8], mambaconv: [256, 16, 7], topk_sparse: [256, 32], saliency_prune: [256, 0.2], rev_res: [256], moa: [256, 4, 2], bitnet: [256, 256], [256, 10]",
}


_ALIASES: Dict[str, str] = {
    "graph": "graph_conv",
    "graphconv": "graph_conv",
    "logic": "diff_logic",
    "stochastic_depth": "stoch_depth",
    "gated_crossattn": "gatedcrossattn",
    "cross_attn": "crossattn",
    "mamba_conv": "mambaconv",
    "sparse_topk": "topk_sparse",
    "saliency": "saliency_prune",
    "saliency_pruning": "saliency_prune",
    "reversible": "rev_res",
    "reversible_residual": "rev_res",
    "latent_attn": "mhla",
    "multi_head_latent_attention": "mhla",
    "mixture_of_attention": "moa",
}


_KNOWN_TYPES = {
    "linear",
    "attn",
    "gqa",
    "moe",
    "trans",
    "fractal",
    "dropout",
    "residual",
    "conv1d",
    "lstm",
    "mod",
    "conv3d",
    "mamba",
    "liquid",
    "hyper",
    "script",
    "diamond",
    "highway",
    "ensemble",
    "imagine",
    "quantum",
    "fractal_synth",
    "chrono",
    "evolve",
    "kan",
    "diff_attn",
    "lora",
    "specnorm",
    "gcp",
    "bitlinear",
    "retention",
    "mix_depth",
    "graph_conv",
    "diff_logic",
    "concept",
    "sphere",
    "poincare",
    "topo_attn",
    "holographic",
    "alchemy",
    "xfusion",
    "node",
    "hypernet",
    "fluid",
    "turbulence",
    "gla",
    "xlstm",
    "ttt",
    "contrastive",
    "sparse_attn",
    "distill",
    "hyena",
    "geglu",
    "conv_mixer",
    "adaptive_rank",
    "stoch_depth",
    "crossattn",
    "gatedcrossattn",
    "complex",
    "mhla",
    "mambaconv",
    "topk_sparse",
    "saliency_prune",
    "bitnet",
    "rev_res",
    "moa",
}


def load_presets() -> Dict[str, str]:
    presets = DSL_PRESETS.copy()
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as fh:
                user_presets = json.load(fh)
            if isinstance(user_presets, dict):
                presets.update({str(k): str(v) for k, v in user_presets.items()})
        except Exception:
            pass
    return presets


def save_preset(name: str, dsl: str) -> None:
    name = str(name).strip()
    dsl = str(dsl).strip()
    if not name:
        raise ValueError("Preset name cannot be empty.")
    if not dsl:
        raise ValueError("Preset DSL cannot be empty.")

    existing: Dict[str, str] = {}
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            existing = {}

    existing[name] = dsl
    with open(PRESETS_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    bracket_depth = 0
    paren_depth = 0
    brace_depth = 0
    quote: Optional[str] = None
    escaped = False

    for ch in text:
        if quote is not None:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
            continue
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1

        if ch == sep and bracket_depth == 0 and paren_depth == 0 and brace_depth == 0:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            continue
        buf.append(ch)

    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out


def _parse_scalar(token: str) -> Any:
    t = token.strip()
    if not t:
        raise ValueError("Empty token in DSL argument list.")
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]

    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False

    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", t) or re.fullmatch(
        r"[+-]?\d+(?:[eE][+-]?\d+)", t
    ):
        return float(t)
    return t


def _parse_args(raw_args: str) -> List[Any]:
    raw_args = raw_args.strip()
    if raw_args == "":
        return []
    return [_parse_scalar(tok) for tok in _split_top_level(raw_args, ",")]


def _as_int(args: List[Any], idx: int, default: int) -> int:
    if idx >= len(args):
        return int(default)
    value = args[idx]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Expected numeric argument at index {idx}, got {value!r}")


def _as_float(args: List[Any], idx: int, default: float) -> float:
    if idx >= len(args):
        return float(default)
    value = args[idx]
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric argument at index {idx}, got {value!r}")


def _normalize_rate(value: float) -> float:
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, float(value)))


def _normalize_keyword(keyword: str) -> str:
    k = keyword.strip().lower()
    return _ALIASES.get(k, k)


def _layer_from_keyword(keyword: str, args: List[Any]) -> Dict[str, Any]:
    lt = _normalize_keyword(keyword)
    if lt not in _KNOWN_TYPES:
        suggestion = get_close_matches(lt, sorted(_KNOWN_TYPES), n=1)
        if suggestion:
            raise ValueError(f"Unknown layer type '{keyword}'. Did you mean '{suggestion[0]}'?")
        raise ValueError(f"Unknown layer type '{keyword}'.")

    if lt == "attn":
        return {"type": lt, "dim": _as_int(args, 0, 128)}
    if lt == "gqa":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "heads": _as_int(args, 1, 8),
            "groups": _as_int(args, 2, 2),
        }
    if lt == "moe":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "experts": _as_int(args, 1, 8),
            "shared": _as_int(args, 2, 0),
            "top_k": max(1, _as_int(args, 3, 2)),
        }
    if lt == "trans":
        return {"type": lt, "dim": _as_int(args, 0, 128)}
    if lt == "fractal":
        return {"type": lt, "dim": _as_int(args, 0, 128), "depth": max(1, _as_int(args, 1, 2))}
    if lt == "dropout":
        return {"type": lt, "rate": _normalize_rate(_as_float(args, 0, 0.1))}
    if lt == "residual":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 4))}
    if lt == "conv1d":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "kernel": max(1, _as_int(args, 1, 3)),
            "groups": max(1, _as_int(args, 2, 1)),
        }
    if lt == "lstm":
        return {"type": lt, "dim": _as_int(args, 0, 128), "layers": max(1, _as_int(args, 1, 1))}
    if lt == "mod":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "expansion": max(1, _as_int(args, 1, 4)),
            "threshold": _normalize_rate(_as_float(args, 2, 0.35)),
        }
    if lt == "conv3d":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "kernel": max(1, _as_int(args, 1, 3)),
            "groups": max(1, _as_int(args, 2, 1)),
        }
    if lt in {"mamba", "liquid", "hyper", "diamond", "imagine", "quantum", "fractal_synth", "chrono"}:
        return {"type": lt, "dim": _as_int(args, 0, 128)}
    if lt == "script":
        code = args[0] if args else "x"
        return {"type": lt, "code": str(code)}
    if lt == "highway":
        mode = "average"
        models: List[str] = []
        for a in args:
            if isinstance(a, str) and a.lower().startswith("mode="):
                mode = a.split("=", 1)[1].strip() or mode
            elif isinstance(a, str):
                models.append(a)
        return {"type": lt, "models": models, "mode": mode, "out_dim": _as_int(args, 0, 8)}
    if lt == "ensemble":
        path = str(args[0]) if args else ""
        return {"type": lt, "path": path}
    if lt == "evolve":
        pool = [str(a) for a in args if isinstance(a, str)]
        if not pool:
            pool = ["mamba", "moe", "liquid", "quantum"]
        return {"type": lt, "pool": pool}
    if lt == "kan":
        return {"type": lt, "dim": _as_int(args, 0, 128), "grid": max(2, _as_int(args, 1, 5)), "order": max(1, _as_int(args, 2, 3))}
    if lt == "diff_attn":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 8))}
    if lt == "lora":
        return {"type": lt, "dim": _as_int(args, 0, 128), "rank": max(1, _as_int(args, 1, 16)), "alpha": _as_float(args, 2, 1.0)}
    if lt == "specnorm":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 4))}
    if lt == "gcp":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 4))}
    if lt == "bitlinear":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 4))}
    if lt == "retention":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 4))}
    if lt == "mix_depth":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "expansion": max(1, _as_int(args, 1, 4)),
            "capacity": _normalize_rate(_as_float(args, 2, 0.5)),
        }
    if lt == "graph_conv":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 1))}
    if lt == "diff_logic":
        return {"type": lt, "dim": _as_int(args, 0, 128), "rules": max(1, _as_int(args, 1, 16))}
    if lt in {"concept", "sphere", "poincare", "holographic", "alchemy", "xfusion", "node", "hypernet", "fluid", "turbulence", "xlstm", "ttt", "distill"}:
        out = {"type": lt, "dim": _as_int(args, 0, 128)}
        if lt == "distill":
            out["momentum"] = max(0.0, min(0.9999, _as_float(args, 1, 0.996)))
        if lt == "ttt":
            out["ttt_lr"] = _as_float(args, 1, 0.01)
        return out
    if lt == "topo_attn":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 4))}
    if lt == "gla":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 4))}
    if lt == "contrastive":
        return {"type": lt, "dim": _as_int(args, 0, 128), "proj_dim": max(2, _as_int(args, 1, 64))}
    if lt == "sparse_attn":
        return {"type": lt, "dim": _as_int(args, 0, 128), "k": max(1, _as_int(args, 1, 8)), "heads": max(1, _as_int(args, 2, 4))}
    if lt == "hyena":
        return {"type": lt, "dim": _as_int(args, 0, 128), "kernel": max(1, _as_int(args, 1, 7)), "expand": max(1, _as_int(args, 2, 2))}
    if lt == "geglu":
        return {"type": lt, "dim": _as_int(args, 0, 128), "expansion": max(1, _as_int(args, 1, 4))}
    if lt == "conv_mixer":
        return {"type": lt, "dim": _as_int(args, 0, 128), "kernel": max(1, _as_int(args, 1, 7))}
    if lt == "adaptive_rank":
        return {"type": lt, "dim": _as_int(args, 0, 128), "rank": max(1, _as_int(args, 1, 16)), "energy": _as_float(args, 2, 0.9)}
    if lt == "stoch_depth":
        return {"type": lt, "dim": _as_int(args, 0, 128), "drop_prob": _normalize_rate(_as_float(args, 1, 0.1))}
    if lt == "crossattn":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 8))}
    if lt == "gatedcrossattn":
        return {"type": lt, "dim": _as_int(args, 0, 128), "heads": max(1, _as_int(args, 1, 8))}
    if lt == "complex":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "heads": max(1, _as_int(args, 1, 8)),
            "expansion": max(1, _as_int(args, 2, 4)),
            "depth": max(1, _as_int(args, 3, 2)),
        }
    if lt == "mhla":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "heads": max(1, _as_int(args, 1, 8)),
            "q_rank": max(1, _as_int(args, 2, 32)),
            "kv_rank": max(1, _as_int(args, 3, 64)),
        }
    if lt == "mambaconv":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "d_state": max(1, _as_int(args, 1, 16)),
            "d_conv": max(1, _as_int(args, 2, 7)),
        }
    if lt == "topk_sparse":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "k": max(1, _as_int(args, 1, 8)),
        }
    if lt == "saliency_prune":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "threshold": _normalize_rate(_as_float(args, 1, 0.2)),
        }
    if lt == "bitnet":
        in_dim = _as_int(args, 0, 128)
        out_dim = _as_int(args, 1, in_dim)
        return {"type": lt, "in": in_dim, "out": out_dim}
    if lt == "rev_res":
        return {"type": lt, "dim": _as_int(args, 0, 128)}
    if lt == "moa":
        return {
            "type": lt,
            "dim": _as_int(args, 0, 128),
            "experts": max(1, _as_int(args, 1, 4)),
            "heads_per_expert": max(1, _as_int(args, 2, 2)),
        }
    return {"type": lt}


def parse_program(program: str) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(program, str):
        return None
    try:
        cleaned_lines = [line.split("#", 1)[0] for line in program.splitlines()]
        cleaned = ",".join(part.strip() for part in cleaned_lines if part.strip())
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        if cleaned.lower().startswith("nn:"):
            cleaned = cleaned[3:].strip()

        tokens = _split_top_level(cleaned, ",")
        layer_defs: List[Dict[str, Any]] = []
        for raw in tokens:
            spec = raw.strip()
            if not spec:
                continue
            if spec.startswith("[") and spec.endswith("]"):
                args = _parse_args(spec[1:-1])
                if len(args) < 2:
                    raise ValueError(f"Linear layer requires 2 args, got {len(args)} in {spec!r}")
                layer_defs.append({"type": "linear", "in": int(args[0]), "out": int(args[1])})
                continue

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[(.*)\]\s*$", spec)
            if not m:
                raise ValueError(f"Invalid layer spec: {spec!r}")
            keyword = m.group(1)
            args = _parse_args(m.group(2))
            layer_defs.append(_layer_from_keyword(keyword, args))

        if not layer_defs:
            return None
        return layer_defs
    except Exception:
        return None


def validate_dsl(program: str) -> Tuple[List[Tuple[str, str]], Optional[List[Dict[str, Any]]]]:
    issues: List[Tuple[str, str]] = []
    layer_defs = parse_program(program)
    if layer_defs is None:
        issues.append(("ERROR", "Failed to parse DSL. Check bracket/keyword syntax."))
        return issues, None

    current_dim: Optional[int] = None
    dim_preserving = {
        "attn",
        "gqa",
        "moe",
        "trans",
        "fractal",
        "residual",
        "conv1d",
        "lstm",
        "mod",
        "conv3d",
        "mamba",
        "liquid",
        "hyper",
        "diamond",
        "imagine",
        "quantum",
        "fractal_synth",
        "chrono",
        "kan",
        "diff_attn",
        "lora",
        "specnorm",
        "gcp",
        "bitlinear",
        "retention",
        "mix_depth",
        "graph_conv",
        "diff_logic",
        "concept",
        "sphere",
        "poincare",
        "topo_attn",
        "holographic",
        "alchemy",
        "xfusion",
        "node",
        "hypernet",
        "fluid",
        "turbulence",
        "gla",
        "xlstm",
        "ttt",
        "contrastive",
        "sparse_attn",
        "distill",
        "hyena",
        "geglu",
        "conv_mixer",
        "adaptive_rank",
        "stoch_depth",
        "crossattn",
        "gatedcrossattn",
        "complex",
        "mhla",
        "mambaconv",
        "topk_sparse",
        "saliency_prune",
        "rev_res",
        "moa",
    }

    for idx, layer in enumerate(layer_defs):
        ltype = layer.get("type", "")
        if ltype == "linear":
            in_dim = int(layer["in"])
            out_dim = int(layer["out"])
            if current_dim is not None and current_dim != in_dim:
                issues.append(("WARNING", f"Layer {idx}: input {in_dim} != previous output {current_dim}"))
            current_dim = out_dim
            continue

        if ltype == "dropout":
            rate = float(layer.get("rate", 0.0))
            if rate < 0.0 or rate > 1.0:
                issues.append(("WARNING", f"Layer {idx} dropout rate out of range: {rate}"))
            continue

        if ltype in {"script", "highway", "ensemble", "evolve"}:
            continue

        if ltype in dim_preserving:
            layer_dim = layer.get("dim")
            if layer_dim is not None:
                layer_dim = int(layer_dim)
                if current_dim is not None and current_dim != layer_dim:
                    issues.append(("WARNING", f"Layer {idx} ({ltype}) dim {layer_dim} != previous output {current_dim}"))
                current_dim = layer_dim

        if layer.get("heads", 1) <= 0:
            issues.append(("WARNING", f"Layer {idx} ({ltype}) has non-positive heads."))
        if layer.get("experts", 1) <= 0:
            issues.append(("WARNING", f"Layer {idx} ({ltype}) has non-positive experts."))

    if len(layer_defs) > 80:
        issues.append(("WARNING", f"Program has {len(layer_defs)} layers; this may be expensive to train."))
    if not any(ld.get("type") == "linear" for ld in layer_defs):
        issues.append(("WARNING", "No explicit linear layers found. Ensure input/output dims are intentional."))
    return issues, layer_defs


def create_modern_nn(
    layer_defs: Any,
    in_dim: Optional[int] = None,
    out_dim: Optional[int] = None,
) -> Optional[ModernMLP]:
    _ = in_dim
    _ = out_dim
    if isinstance(layer_defs, str):
        layer_defs = parse_program(layer_defs)
    if not layer_defs:
        return None
    model = ModernMLP(layer_defs)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def capability_report(layer_defs: Any) -> Dict[str, Any]:
    if isinstance(layer_defs, str):
        layer_defs = parse_program(layer_defs)
    if not layer_defs:
        return {
            "layer_count": 0,
            "scores": {},
            "notes": ["No parsable layers."],
        }

    types = [str(ld.get("type", "")) for ld in layer_defs]
    type_set = set(types)

    def _score(keys: set[str], weight: float = 1.0) -> float:
        hits = sum(1 for t in types if t in keys)
        return min(100.0, round(100.0 * (hits / max(1, len(types))) * weight, 2))

    scores = {
        "adaptivity": _score({"moe", "mod", "mix_depth", "ttt", "adaptive_rank", "lora"}, weight=2.2),
        "reasoning": _score({"diff_logic", "concept", "graph_conv", "sphere", "poincare", "topo_attn"}, weight=2.5),
        "efficiency": _score({"bitlinear", "bitnet", "retention", "gla", "sparse_attn", "topk_sparse", "adaptive_rank"}, weight=2.4),
        "robustness": _score({"residual", "dropout", "specnorm", "gcp", "distill", "stoch_depth"}, weight=2.1),
        "novelty": _score(
            {
                "kan",
                "diff_attn",
                "hyena",
                "geglu",
                "conv_mixer",
                "mhla",
                "mambaconv",
                "moa",
                "holographic",
                "alchemy",
                "quantum",
            },
            weight=2.6,
        ),
    }

    notes: List[str] = []
    if "linear" in type_set and len(type_set) <= 3:
        notes.append("Architecture is simple; consider adding adaptive or reasoning layers.")
    if "dropout" not in type_set and "stoch_depth" not in type_set:
        notes.append("No explicit regularization block detected.")
    if scores["efficiency"] < 20 and len(layer_defs) > 20:
        notes.append("Large model without efficiency-oriented layers may be expensive.")
    if not notes:
        notes.append("Balanced architecture profile.")

    return {
        "layer_count": len(layer_defs),
        "unique_layers": sorted(type_set),
        "scores": scores,
        "notes": notes,
    }


__all__ = [
    "DSL_PRESETS",
    "load_presets",
    "save_preset",
    "parse_program",
    "validate_dsl",
    "create_modern_nn",
    "capability_report",
]
