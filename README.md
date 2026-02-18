# NeuroDSL Infinity Studio (v4.0)

A **GUI-first neural architecture sandbox** that lets you describe modern PyTorch networks in a compact **DSL (Domain Specific Language)**, compile them into a runnable model, **train** (synthetic or CSV), **run inference** (single or batch), **visualize layer stats**, and **export** (PTH / ONNX / TorchScript).

> Built around:
> - **NeuroDSL** (pyparsing-based parser + validator)
> - **ModernMLP** (a composable “MLP+” stack that supports attention, MoE, fractal blocks, residuals, conv1d, BiLSTM, dropout)
> - **FreeSimpleGUI** desktop app (“Infinity Studio”)

---

## What you get

### ✅ NeuroDSL → PyTorch compiler
Write an architecture like:

[64, 128], fractal: [128, 2], gqa: [128, 8, 2], moe: [128, 8], dropout: [0.1], [128, 10]


…and the studio will:

* parse it
* validate dimension flow (warn if mismatched)
* build a PyTorch model
* show param counts + “memory estimate”
* let you train + infer + export

### ✅ GUI Studio (Dark UI)

Tabs include:

* **Training Studio**: epochs slider, loss choice, gradient clipping, LR warmup, live loss curve
* **Inference Lab**: single-vector inference + random input generator
* **Batch Inference**: run a whole CSV through the model and preview outputs
* **Architecture Viz**: text table of layers + params + trainable params
* **Neural Stream**: live log feed + build/train traces

### ✅ Training Engine (v4.0)

* Losses: **MSE**, **CrossEntropy**, **Huber**, **MAE**
* Optimizer: **AdamW** + weight decay
* Scheduler: **CosineAnnealingLR**
* **Linear warmup** (first N steps)
* **Gradient clipping**
* **AMP** (automatic mixed precision) when CUDA is available
* Data:

  * **Synthetic dummy data** generator for quick experiments
  * **CSV training** loader for real numeric datasets

### ✅ Export

* **Save/Load weights** (`.pth`)
* **Export ONNX** (`.onnx`) *(optional deps)*
* **Export TorchScript** (`.pt`) *(always available)*

---

## Repo layout

```text
.
├─ main.py           # FreeSimpleGUI app (build/train/infer/export)
├─ omni_studio.py    # Unified GUI control center (scripts + tabular + image + multimodal)
├─ omni_cli.py       # Unified CLI across all major modes
├─ parser_utils.py   # DSL presets + parser + validator + model factory
├─ network.py        # ModernMLP + blocks (MoE/GQA/Transformer/Fractal/etc.)
├─ trainer.py        # TrainingEngine (multi-loss, warmup, clip, AMP, exports)
├─ experimental_models.py # Image and multimodal experimental engines
├─ device_utils.py   # Device/NPU/XPU backend detection and routing
├─ web_model_utils.py # URL/Hugging Face checkpoint download helpers
├─ verify.py         # Verification tests for parsing, blocks, training, exports
├─ LICENSE           # MIT
├─ CODE_OF_CONDUCT.md
├─ CONTRIBUTING.md
└─ SECURITY.md
```

---

## Install

### 1) Clone

```bash
git clone https://github.com/kai9987kai/NeuroDSL-Infinity-Studio.git
cd NeuroDSL-Infinity-Studio
```

### 2) Create a venv

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install -U pip
python -m pip install torch numpy pyparsing FreeSimpleGUI
```

#### Optional (for ONNX export)

```bash
python -m pip install onnx onnxscript
```

> Notes:
>
> * `network.py` uses `torch.nn.functional.scaled_dot_product_attention`, so **PyTorch 2.x** is recommended.
> * FreeSimpleGUI typically uses **tkinter** by default; on Linux you may need your distro’s tkinter package.

---

## Run the Studio

```bash
python main.py
```

You should see **“NeuroDSL Infinity Studio v4.0”**.

---

## NeuroDSL reference

NeuroDSL is a **comma-separated list of layer specs**.

### Core syntax

* **Linear layer**

  * `[in, out]`
* **Keyword layer**

  * `keyword: [args...]`

### Supported layer types

#### 1) Linear

```text
[128, 64]
```

Creates: `Linear(128→64) + RMSNorm(64) + SiLU`

#### 2) Self-attention

```text
attn: [dim]
```

Creates: `SOTAAttention(dim)` (single-token attention over features)

#### 3) Grouped Query Attention (GQA)

```text
gqa: [dim, heads, groups]
```

Defaults:

* `heads = 8`
* `groups = 2`

#### 4) Mixture of Experts (MoE)

```text
moe: [dim, experts, shared]
```

Defaults:

* `experts = 8`
* `shared = 0`
  Internals:
* top-k routing (`top_k=2`)
* mixture of `SwiGLU` and lightweight `FractalBlock` experts
* optional shared experts + load-balancing auxiliary loss

#### 5) Transformer block

```text
trans: [dim]
```

Creates: RMSNorm + Attention + RMSNorm + SwiGLU MLP + LayerScale skips

#### 6) Fractal block

```text
fractal: [dim, depth]
```

Defaults:

* `depth = 2`
  Internals:
* recursive-ish “extreme depth” block using repeated SwiGLU + stochastic depth + layer scale

---

## v4.0 layers

#### 7) Dropout

```text
dropout: [0.2]
```

* Accepts `0–1` as probability
* If you pass `20`, it’s treated as `20%` → `0.2`

#### 8) Residual FFN block

```text
residual: [dim, expansion]
```

Defaults:

* `expansion = 4`
  Creates: pre-norm FFN + skip + layer scale + stochastic depth

#### 9) Conv1D block

```text
conv1d: [dim, kernel]
```

Defaults:

* `kernel = 3`
  Treats the feature dim as channels, adds a length-1 “time axis”, convs, then squeezes back.

#### 10) BiLSTM block

```text
lstm: [dim, layers]
```

Defaults:

* `layers = 1`
  Wraps features into a length-1 sequence, runs a **bidirectional LSTM**, then projects back.

#### 11) Adaptive compute block (MoD-inspired)

```text
mod: [dim, expansion, threshold]
```

Defaults:

* `expansion = 4`
* `threshold = 0.35`
  Uses a learned router and can skip expensive compute for easy samples at inference time.

---

## Dimension flow rules (important)

The validator checks that dimensions chain correctly:

* If you do `[128, 64]`, the next block expecting `dim` should be `64`
* Keywords like `fractal/moe/gqa/trans/attn/residual/conv1d/lstm/mod` are treated as dim-preserving blocks and should match the previous output.

Example (valid):

```text
[32, 64], residual: [64], [64, 10]
```

Example (warns):

```text
[128, 64], [32, 10]
```

---

## Built-in presets

The GUI includes presets like:

* **Classifier (MLP)**
* **Deep Classifier**
* **AutoEncoder**
* **Transformer Block**
* **MoE Heavy**
* **Adaptive Creative**
* **Attention Pipeline**
* **Conv-LSTM Hybrid**
* **Kitchen Sink** (everything)

Select a preset from the dropdown to populate the DSL field.

---

## Using the GUI (workflow)

1. **Pick a preset** or type your own DSL
2. Click **VALIDATE** to see warnings/errors in *Neural Stream*
3. Click **INITIALIZE CORE** to compile the model
4. Choose training settings (loss, grad clip, warmup, epochs)
5. Click **START** (synthetic) or **TRAIN CSV** (real data)
6. Use **Inference Lab** or **Batch Inference**
7. Save/export from the sidebar

---

## CSV training format

### Training CSV (`TRAIN CSV`)

* Header row is allowed (skipped)
* Non-numeric rows are skipped
* Default: **last column is target** and all previous columns are features
* Target shape is loaded as `(N, 1)`

**Regression example**

```csv
x1,x2,x3,y
0.1,0.2,0.3,0.9
...
```

Use **MSE/Huber/MAE**.

**Classification notes (CrossEntropy)**

* Use `CrossEntropy` loss in the GUI.
* Your model output dimension must equal **num_classes** (final linear out = C).
* The trainer will convert `y` into class indices if possible.

  * Best practice: store the label as an integer in the last column.

---

## Batch inference CSV format

### Batch inference (`BATCH RUN`)

* Header row allowed
* Each row should contain **only input features**
* The studio will run all rows through the model and preview outputs.

---

## Exports

### Save / Load weights (PTH)

* Saves `state_dict()` to `.pth`
* Loads back onto the model device
* Has a small compatibility shim for models wrapped by compilation (`_orig_mod`)

### Export ONNX

* Uses `torch.onnx.export(..., opset_version=11)`
* If you get an ONNX-related error, install:

  ```bash
  pip install onnx onnxscript
  ```

### Export TorchScript

* Uses tracing (`torch.jit.trace`)
* Outputs `.pt`

---

## Run trained models from terminal

Use `run_model.py` for scripted inference without opening the GUI.

### 1) Run a `.pth` model (requires DSL)

```bash
python run_model.py --model model.pth --dsl "[64,128], moe: [128,8,1], mod: [128,4,0.35], [128,10]" --input "0.1,0.2,0.3,0.4"
```

### 2) Run a TorchScript `.pt` model

```bash
python run_model.py --model model.pt --input-csv inputs.csv --output-csv preds.csv --benchmark-runs 50
```

Additional runtime options:

* web model download: `--model-url` or `--hf-repo` + `--hf-file`
* expanded device targeting: `--device auto/cpu/cuda/mps/xpu/npu/dml`
* compile acceleration: `--compile`
* probability view: `--as-probs`
* JSON outputs: `--output-json preds.json`
* ensemble inference: repeat `--ensemble-model other_model.pth`

### 3) Creative class sampling

```bash
python run_model.py --model model.pth --dsl-file model.dsl --input-csv inputs.csv --creative-samples 5 --temperature 0.9 --top-p 0.9
```

### 4) MC-dropout uncertainty

```bash
python run_model.py --model model.pth --dsl-file model.dsl --input-csv inputs.csv --mc-dropout-samples 8 --output-csv preds_with_std.csv
```

---

## Research-inspired upgrades

Recent upgrades include:

* shared-expert MoE path via `moe: [dim, experts, shared]` (DeepSeekMoE-style shared specialists),
* adaptive compute routing via `mod: [dim, expansion, threshold]` (Mixture-of-Depths inspired),
* MoE load-balance auxiliary loss, surfaced through `model.get_aux_loss()` and consumed by `TrainingEngine`.

These target smarter routing, faster easy-sample inference, and more creative outputs through CLI sampling controls.

---

## Omni control layer (new)

### Unified CLI

Use `omni_cli.py` as the single command layer across scripts and modes:

```bash
python omni_cli.py devices
python omni_cli.py run-script --script verify
python omni_cli.py tabular-train --dsl "[32,64], moe: [64,6,1], mod: [64,4,0.35], [64,10]"
python omni_cli.py tabular-search --input-dim 32 --output-dim 10 --trials 24
python omni_cli.py serve-tabular --model outputs/tabular_model.pth --dsl-file outputs/search_best.dsl --port 8080
python omni_cli.py image-train --epochs 40 --save-model outputs/image_model.pth
python omni_cli.py image-generate --checkpoint outputs/image_model.pth --output-grid outputs/generated_grid.png
python omni_cli.py image-interpolate --checkpoint outputs/image_model.pth --steps 12 --output-grid outputs/interp.png
python omni_cli.py multimodal-train --epochs 40 --save-model outputs/multimodal_model.pth
python omni_cli.py multimodal-run --checkpoint outputs/multimodal_model.pth --text "uncanny biomech cathedral"
python omni_cli.py ai-dsl --prompt "hybrid attention model for 32-d features"
python omni_cli.py ai-optimize --dsl-file outputs/search_best.dsl
python omni_cli.py ai-autopilot --objective "robust compact model for 32->10 regression" --input-dim 32 --output-dim 10 --candidates 8 --out-dir outputs/autopilot
python omni_cli.py ai-autopilot-sweep --objective "robust compact model for 32->10 regression" --seeds "11,23,37,53" --promote-best --fine-tune-epochs 300 --fine-tune-output champion_final.pth
python omni_cli.py agent-manifest --output-json outputs/agent_manifest.json
python omni_cli.py serve-agent-api --host 127.0.0.1 --port 8090
python omni_cli.py champion-package --model outputs/autopilot_sweep/champion_final.pth --dsl-file outputs/autopilot_sweep/champion.dsl --build-exe --install-pyinstaller --output-dir outputs/champion_package
python omni_cli.py sim-generate-data --episodes 64 --output-csv outputs/sim_lab/sim_dataset.csv
python omni_cli.py sim-train-agent --cycles 4 --episodes-per-cycle 24 --out-dir outputs/sim_lab
python omni_cli.py platform-init --seed-phrases
python omni_cli.py account-create --username kai --password secret --lang en
python omni_cli.py project-create --owner kai --name uncanny_lab
python omni_cli.py events-sync --max-items 50
python omni_cli.py polyglot-scaffold --out-dir outputs/polyglot_bridge --base-url http://127.0.0.1:8090
python omni_cli.py console-app
python omni_cli.py platform-health --include-functional --include-codex --output-json outputs/platform_health.json
python omni_cli.py export-bundle --include outputs parser_utils.py network.py trainer.py --output-zip outputs/platform_bundle.zip
```

### Unified GUI

Run:

```bash
python omni_studio.py
```

This GUI can launch and monitor:

* existing scripts (main/verify/functional/run_model)
* tabular training + inference
* neural architecture search + local inference API server
* image mode training + uncanny image generation
* image latent interpolation experiments
* multimodal training + inference

Keyboard controls:

* `F5` start current tab action
* `F6` stop active process
* `Ctrl+R` run script tab
* `Ctrl+T` tabular train
* `Ctrl+I` image train
* `Ctrl+M` multimodal train

Local API mode:

* `omni_cli.py serve-tabular` starts an HTTP endpoint
* `GET /health` status check
* `GET /meta` input dimension + device info
* `POST /infer` JSON `{ "inputs": [...] }` or `{ "inputs": [[...], ...] }`
* `omni_cli.py serve-agent-api` starts an agent API with:
* `GET /manifest` machine-readable capability map
* `GET /session` active in-memory model state
* `POST /dsl/validate`, `POST /session/build`, `POST /session/train`, `POST /session/infer`, `POST /session/reset`

AI workflow mode:

* `ai-connection` quick OpenAI connectivity check
* `ai-dsl` natural-language to DSL generation
* `ai-explain` architecture explanation
* `ai-optimize` DSL optimization
* `ai-hyperparams` JSON hyperparameter suggestions
* `ai-codegen` standalone PyTorch code generation
* `ai-latency` hardware latency report
* `ai-diagram` ASCII architecture diagram
* `ai-autopilot` autonomous AI candidate generation + local train/rank loop
* `ai-autopilot-sweep` multi-seed champion search + optional promotion/fine-tuning
* `champion-package` creates model package with interactive runner and optional EXE

---

## Device + Web Model Access

`device_utils.py` detects and routes to available accelerators:

* `cpu`
* `cuda`
* `mps`
* `xpu`
* `npu` (when `torch_npu` stack is installed)
* `dml` (when `torch-directml` is installed)

`run_model.py` and `omni_cli.py` support web model access:

* direct URL download (`--model-url`)
* Hugging Face download (`--hf-repo` + `--hf-file`)

---

## Verification tests

Run:

```bash
python verify.py
```

It covers:

* parsing + validation warnings
* Infinity blocks (Fractal, GQA, MoE + shared experts)
* v4.0 layers (Dropout, Residual, Conv1D, LSTM, MoD)
* training engine sanity checks + TorchScript export
* presets parse cleanly
* model summary output

---

## Implementation notes (for devs)

### Where to add a new DSL keyword

1. `parser_utils.py`

   * define a `*_layer` parser
   * add it to the `layer = (...)` union in the right precedence order
2. `network.py`

   * implement the corresponding module/block
   * handle it inside `ModernMLP.__init__`
3. `verify.py`

   * add a small shape test + parser test

### Current tensor shape convention

The model operates primarily on **2D tensors**:

* `x.shape == (batch, dim)`

Several “sequence-like” blocks (attention, transformer, LSTM) currently operate on a **single-step pseudo-sequence** internally (length = 1). If you want true sequence modeling, the next step is to extend the DSL and model pipeline to support `(batch, seq, dim)` end-to-end.

---

## Contributing

PRs are welcome. If you add a new layer:

* update parser + network + tests
* ensure `verify.py` stays green

See `CODE_OF_CONDUCT.md` for community standards.
`SECURITY.md` exists — feel free to open an issue/PR to formalize reporting details.

---

## License

MIT — see `LICENSE`.

---

## Credits

* **PyTorch** for the training/runtime stack
* **pyparsing** for building the DSL parser
* **FreeSimpleGUI** (PySimpleGUI-style API) for the desktop UI

```

::contentReference[oaicite:0]{index=0}
```





