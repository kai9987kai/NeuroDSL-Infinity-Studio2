"""Device and accelerator utilities for NeuroDSL tools.

Supports dynamic detection for:
- CPU
- CUDA
- MPS
- XPU (Intel PyTorch backend)
- NPU (torch_npu ecosystem)
- DirectML (if torch-directml is installed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class DeviceInfo:
    name: str
    available: bool
    backend: str
    note: str = ""


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _has_xpu() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def _has_npu() -> bool:
    # Ascend backend is typically exposed through torch_npu and torch.npu.
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    try:
        return bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        return False


def _has_dml() -> bool:
    try:
        import torch_directml  # noqa: F401
        return True
    except Exception:
        return False


def detect_devices() -> Dict[str, DeviceInfo]:
    devices: Dict[str, DeviceInfo] = {
        "cpu": DeviceInfo(name="cpu", available=True, backend="torch", note="Always available"),
        "cuda": DeviceInfo(name="cuda", available=torch.cuda.is_available(), backend="torch"),
        "mps": DeviceInfo(name="mps", available=_has_mps(), backend="torch"),
        "xpu": DeviceInfo(name="xpu", available=_has_xpu(), backend="torch"),
        "npu": DeviceInfo(name="npu", available=_has_npu(), backend="torch_npu"),
        "dml": DeviceInfo(name="dml", available=_has_dml(), backend="torch_directml"),
    }
    return devices


def available_device_names(include_cpu: bool = True) -> List[str]:
    detected = detect_devices()
    out = []
    for name, info in detected.items():
        if not include_cpu and name == "cpu":
            continue
        if info.available:
            out.append(name)
    return out


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower().strip()
    detected = detect_devices()

    if preferred == "auto":
        for candidate in ("npu", "xpu", "cuda", "mps", "dml", "cpu"):
            if detected[candidate].available:
                preferred = candidate
                break

    if preferred not in detected:
        raise ValueError(f"Unknown device '{preferred}'.")
    if not detected[preferred].available:
        raise RuntimeError(f"Requested device '{preferred}' is not available on this system.")

    if preferred == "dml":
        # DML uses a custom device abstraction from torch_directml.
        try:
            import torch_directml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("DirectML requested but torch_directml is not importable.") from exc
        return torch_directml.device()

    return torch.device(preferred)


def format_device_report() -> str:
    detected = detect_devices()
    lines = ["Detected compute backends:"]
    for name in ("cpu", "cuda", "mps", "xpu", "npu", "dml"):
        info = detected[name]
        status = "YES" if info.available else "no"
        suffix = f" | {info.note}" if info.note else ""
        lines.append(f"- {name:<4} available={status:<3} backend={info.backend}{suffix}")
    return "\n".join(lines)


def maybe_set_device_for_model(model: torch.nn.Module, preferred: str = "auto") -> torch.nn.Module:
    device = resolve_device(preferred)
    return model.to(device)

