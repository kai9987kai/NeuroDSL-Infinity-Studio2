"""Web model access helpers.

Provides checkpoint download by:
- direct URL
- optional Hugging Face repo+filename (if huggingface_hub is installed)
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional


def download_file(url: str, output_path: str, timeout: int = 120) -> str:
    if not url:
        raise ValueError("URL is required.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "NeuroDSL-Omni/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
        f.write(r.read())
    return str(out)


def download_from_hf(repo_id: str, filename: str, local_dir: str = "downloads") -> str:
    if not repo_id or not filename:
        raise ValueError("repo_id and filename are required.")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

