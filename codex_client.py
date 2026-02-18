import json
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple


SYSTEM_PROMPT = """You are an expert NeuroDSL Architect. Your job is to translate natural language descriptions of neural networks into NeuroDSL specifications.

### NeuroDSL Syntax Rules
Return ONLY a valid NeuroDSL string (comma-separated list of layers). No markdown, no explanations.

**Layer Types:**
1. **Linear**: `[in, out]`
2. **Self-Attention**: `attn: [dim]`
3. **Grouped Query Attention**: `gqa: [dim, heads, groups]`
4. **MoE (Mixture of Experts)**: `moe: [dim, experts, shared]`
5. **Transformer Block**: `trans: [dim]`
6. **Fractal Block**: `fractal: [dim, depth]`
7. **Dropout**: `dropout: [rate]`
8. **Residual FFN**: `residual: [dim, expansion]`
9. **Conv1D**: `conv1d: [dim, kernel]`
10. **BiLSTM**: `lstm: [dim, layers]`
11. **Adaptive Compute**: `mod: [dim, expansion, threshold]`
12. **Conv3D**: `conv3d: [dim, kernel]`

### Rules
- Dimension flow must match from layer to layer.
- Return only the DSL text, no markdown fences.
"""


EXPLAIN_SYSTEM_PROMPT = """You are a neural architecture explainer.
Be concise and practical. Explain data flow, key blocks, and likely strengths/risks.
"""


CODE_SYSTEM_PROMPT = """You are an expert PyTorch engineer.
Return production-ready Python code only.
"""


JSON_SYSTEM_PROMPT = """You are a deep learning tuning expert.
Return valid JSON only.
"""


CANDIDATE_JSON_SYSTEM_PROMPT = """You are a neural architecture search copilot.
Return strict JSON only. No markdown, no comments.
"""


def _get_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key and api_key.strip():
        return api_key.strip()
    env = os.getenv("OPENAI_API_KEY", "").strip()
    return env or None


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    for fence in ("```python", "```json", "```text", "```"):
        text = text.replace(fence, "")
    return text.strip()


def _extract_content(response_obj: dict) -> str:
    choices = response_obj.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return _strip_code_fences(msg.get("content", ""))


def _chat_completion(
    api_key: str,
    user_prompt: str,
    *,
    system_prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
    retries: int = 2,
    fallback_models: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    models_to_try = [model] + [m for m in (fallback_models or []) if m and m != model]
    last_error = "Unknown error"

    for mdl in models_to_try:
        for attempt in range(retries + 1):
            payload = {
                "model": mdl,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = _extract_content(result)
                if not content:
                    return False, "Empty response from model."
                return True, content
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    body = ""
                last_error = f"API Error {e.code}: {e.reason} {body}".strip()
                # Retry on server/rate limits, otherwise try fallback model.
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                break
            except Exception as e:
                last_error = f"Connection Error: {str(e)}"
                if attempt < retries:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                break
    return False, last_error


def generate_dsl(api_key, user_prompt, model="gpt-3.5-turbo"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    return _chat_completion(
        key,
        user_prompt,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        temperature=0.2,
        max_tokens=200,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def explain_dsl(api_key, dsl_code, model="gpt-3.5-turbo"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Explain this NeuroDSL architecture in clear English.
Describe data flow, main components, and potential strengths/risks.

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=EXPLAIN_SYSTEM_PROMPT,
        model=model,
        temperature=0.2,
        max_tokens=400,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def fix_dsl(api_key, broken_dsl, error_msg, model="gpt-3.5-turbo"):
    prompt = f"""The following NeuroDSL has an error. Fix it and return only corrected DSL text.

Broken DSL:
{broken_dsl}

Error:
{error_msg}
"""
    return generate_dsl(api_key, prompt, model=model)


def optimize_dsl(api_key, dsl_code, model="gpt-3.5-turbo"):
    prompt = f"""Optimize this NeuroDSL for stronger training stability, efficiency, and representation quality.
Return only valid DSL text.

Original:
{dsl_code}
"""
    return generate_dsl(api_key, prompt, model=model)


def generate_pytorch_code(api_key, dsl_code, model="gpt-4"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Convert this NeuroDSL into a complete standalone PyTorch file.
Include imports, model class, and a __main__ smoke test.
Return only Python code.

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=CODE_SYSTEM_PROMPT,
        model=model,
        temperature=0.2,
        max_tokens=1600,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def suggest_hyperparams(api_key, dsl_code, model="gpt-3.5-turbo"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Suggest training hyperparameters for this NeuroDSL.
Return ONLY JSON with keys: epochs (int), lr (float), clip (float), warmup_steps (int), weight_decay (float).

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=JSON_SYSTEM_PROMPT,
        model=model,
        temperature=0.1,
        max_tokens=200,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def generate_synthetic_data(api_key, task_desc, n_samples=1000, model="gpt-4"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Write a Python script to generate synthetic data for:
"{task_desc}"

Requirements:
- use numpy or sklearn
- generate n_samples={n_samples}
- expose X and y arrays
- print X.shape and y.shape
- return code only
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=CODE_SYSTEM_PROMPT,
        model=model,
        temperature=0.35,
        max_tokens=1200,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def generate_unit_tests(api_key, dsl_code, model="gpt-4"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Write a standalone unittest script for this NeuroDSL.
Assume parser_utils.py and network.py are in the same folder.
Test:
- parse/build succeeds
- forward shape for batch=4
- backward pass runs
Return Python code only.

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=CODE_SYSTEM_PROMPT,
        model=model,
        temperature=0.15,
        max_tokens=1200,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def estimate_latency(api_key, dsl_code, hardware="NVIDIA A100", model="gpt-3.5-turbo"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Estimate inference performance for this NeuroDSL on {hardware}.
Include:
1) approximate FLOPs
2) memory footprint trend
3) latency estimate for batch=1 and batch=64
4) two optimization suggestions

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=EXPLAIN_SYSTEM_PROMPT,
        model=model,
        temperature=0.2,
        max_tokens=450,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def generate_ascii_diagram(api_key, dsl_code, model="gpt-4"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    prompt = f"""Generate a vertical ASCII dataflow diagram for this DSL.
Use boxes and arrows. Include key dims when possible.
Return only ASCII diagram text.

NeuroDSL:
{dsl_code}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=EXPLAIN_SYSTEM_PROMPT,
        model=model,
        temperature=0.25,
        max_tokens=500,
        fallback_models=["gpt-4o-mini", "gpt-4.1-mini"],
    )


def test_connection(api_key=None, model="gpt-4o-mini"):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."
    return _chat_completion(
        key,
        "Reply with exactly: OK",
        system_prompt="You are a connectivity test service. Reply with only OK.",
        model=model,
        temperature=0.0,
        max_tokens=5,
        retries=1,
        fallback_models=[],
    )


def generate_dsl_candidates(
    api_key,
    objective,
    input_dim,
    output_dim,
    count=6,
    model="gpt-4o-mini",
):
    key = _get_api_key(api_key)
    if not key:
        return False, "Missing API key. Pass key or set OPENAI_API_KEY."

    try:
        n = int(count)
    except Exception:
        n = 6
    n = max(1, min(20, n))
    in_dim = int(input_dim)
    out_dim = int(output_dim)

    prompt = f"""Design {n} diverse NeuroDSL candidates for this objective:
{objective}

Hard constraints:
- First layer must accept input dim {in_dim}
- Last layer must output dim {out_dim}
- Use only supported NeuroDSL layers (linear, attn, gqa, moe, trans, fractal, dropout, residual, conv1d, lstm, mod, conv3d)
- Keep architectures valid for dimension flow.

Return JSON with this exact schema:
{{
  "candidates": [
    {{"dsl": "<neurodsl string>", "notes": "<short rationale>"}}
  ]
}}
"""
    return _chat_completion(
        key,
        prompt,
        system_prompt=CANDIDATE_JSON_SYSTEM_PROMPT,
        model=model,
        temperature=0.5,
        max_tokens=1400,
        fallback_models=["gpt-4.1-mini", "gpt-4o-mini"],
    )
