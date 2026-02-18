"""Automated functional test - exercises all code paths without GUI."""
import csv
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import torch

from network import ModernMLP
from parser_utils import DSL_PRESETS, create_modern_nn, parse_program, validate_dsl
from trainer import TrainingEngine


print("=" * 60)
print("NeuroDSL Infinity v4.0 - Full Functional Test")
print("=" * 60)
errors = []

# --------------------------------------------------
# TEST 1: All DSL presets build and forward pass
# --------------------------------------------------
print("\n[TEST 1] Building and running all DSL presets...")
for name, dsl in DSL_PRESETS.items():
    try:
        result = parse_program(dsl)
        assert result, f"Preset '{name}' parse failed"
        model = create_modern_nn(result)

        in_dim = result[0].get("in", result[0].get("dim"))
        x = torch.randn(4, in_dim)
        with torch.no_grad():
            y = model(x)
        print(f"  [PASS] {name}: input={in_dim} -> output={tuple(y.shape)}")
    except Exception as e:
        errors.append(f"Preset '{name}': {e}")
        print(f"  [FAIL] {name}: {e}")

# --------------------------------------------------
# TEST 2: DSL Validation catches dim mismatches
# --------------------------------------------------
print("\n[TEST 2] DSL Validation...")
try:
    issues, result = validate_dsl("[128, 64], [32, 10]")
    assert result is not None
    assert any(("match" in msg.lower()) or ("!=" in msg) for _, msg in issues), "Should detect dim mismatch"
    print(f"  [PASS] Detects dimension mismatch ({len(issues)} warnings)")
except Exception as e:
    errors.append(f"Validation: {e}")
    print(f"  [FAIL] {e}")

try:
    issues, result = validate_dsl("invalid garbage text")
    assert result is None
    print("  [PASS] Rejects invalid DSL")
except Exception as e:
    errors.append(f"Invalid DSL rejection: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# TEST 3: Training engine with all loss functions
# --------------------------------------------------
print("\n[TEST 3] Training engine with all loss functions...")
for loss_fn in ["MSE", "Huber", "MAE"]:
    try:
        defs = [{"type": "linear", "in": 8, "out": 16}, {"type": "linear", "in": 16, "out": 4}]
        model = ModernMLP(defs)
        trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=1.0, warmup_steps=3)
        X, y = trainer.generate_dummy_data(8, 4, n_samples=50)

        losses = []
        for _ in range(20):
            loss, lr, gn = trainer.train_step(X, y)
            losses.append(loss)

        print(f"  [PASS] {loss_fn}: loss {losses[0]:.4f} -> {losses[-1]:.4f} | LR: {lr:.6f} | grad: {gn:.3f}")
    except Exception as e:
        errors.append(f"Training {loss_fn}: {e}")
        print(f"  [FAIL] {loss_fn}: {e}")

# --------------------------------------------------
# TEST 4: CrossEntropy with integer targets
# --------------------------------------------------
print("\n[TEST 4] CrossEntropy loss...")
try:
    defs = [{"type": "linear", "in": 8, "out": 10}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model, loss_fn="CrossEntropy", grad_clip=1.0, warmup_steps=2)
    X = torch.randn(50, 8)
    y = torch.randint(0, 10, (50, 1)).float()

    for _ in range(10):
        loss, lr, gn = trainer.train_step(X, y)
    print(f"  [PASS] CrossEntropy: loss={loss:.4f}")
except Exception as e:
    errors.append(f"CrossEntropy: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# TEST 5: CSV data loading
# --------------------------------------------------
print("\n[TEST 5] CSV data loading...")
try:
    csv_path = os.path.join(tempfile.gettempdir(), "test_neurodsl.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1", "f2", "f3", "target"])
        for _ in range(20):
            import random

            writer.writerow([random.random(), random.random(), random.random(), random.randint(0, 1)])

    defs = [{"type": "linear", "in": 3, "out": 1}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model, loss_fn="MSE")
    X, y = trainer.load_csv_data(csv_path)
    assert X.shape == (20, 3), f"Expected (20, 3), got {X.shape}"
    assert y.shape == (20, 1), f"Expected (20, 1), got {y.shape}"

    loss, lr, gn = trainer.train_step(X, y)
    print(f"  [PASS] CSV loaded: {X.shape[0]} samples, {X.shape[1]} features -> loss={loss:.4f}")
    os.remove(csv_path)
except Exception as e:
    errors.append(f"CSV loading: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# TEST 6: TorchScript export
# --------------------------------------------------
print("\n[TEST 6] TorchScript export...")
try:
    defs = [{"type": "linear", "in": 4, "out": 2}]
    model = ModernMLP(defs)
    trainer = TrainingEngine(model)
    ts_path = os.path.join(tempfile.gettempdir(), "test_model.pt")
    trainer.export_torchscript(ts_path, 4)
    assert os.path.exists(ts_path)

    loaded = torch.jit.load(ts_path)
    out = loaded(torch.randn(1, 4))
    print(f"  [PASS] TorchScript exported and loaded: output shape={tuple(out.shape)}")
    os.remove(ts_path)
except Exception as e:
    errors.append(f"TorchScript: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# TEST 7: Model summary generation
# --------------------------------------------------
print("\n[TEST 7] Model summary...")
try:
    defs = parse_program(
        "fractal: [128, 2], moe: [128, 4, 1], mod: [128, 4, 0.35], residual: [128], dropout: [0.2], [128, 10]"
    )
    model = ModernMLP(defs)
    summary, total = model.get_summary()
    print(f"  [PASS] Summary: {len(summary)} layer entries, {total:,} total params")
    for entry in summary:
        print(f"     [{entry['index']:2}] {entry['type']:<20} params={entry['params']:>8,}")
except Exception as e:
    errors.append(f"Summary: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# TEST 8: Complex architecture end-to-end
# --------------------------------------------------
print("\n[TEST 8] Kitchen Sink architecture end-to-end...")
try:
    dsl = DSL_PRESETS["Kitchen Sink"]
    defs = parse_program(dsl)
    model = create_modern_nn(defs)

    in_dim = defs[0].get("in", defs[0].get("dim"))
    trainer = TrainingEngine(model, loss_fn="MSE", grad_clip=0.5, warmup_steps=5)
    X, y = trainer.generate_dummy_data(in_dim, 10, n_samples=32)

    for _ in range(15):
        loss, lr, gn = trainer.train_step(X, y)

    model.eval()
    with torch.no_grad():
        pred = model(X[:1])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [PASS] Kitchen Sink: {total_params:,} params | Final loss={loss:.4f} | Output shape={tuple(pred.shape)}")
except Exception as e:
    errors.append(f"Kitchen Sink: {e}")
    print(f"  [FAIL] {e}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n[TEST 9] Omni CLI smoke commands...")
try:
    rc = subprocess.run(["python", "omni_cli.py", "devices"], check=False, capture_output=True, text=True)
    assert rc.returncode == 0, f"devices command failed: {rc.stderr}"
    assert "cpu" in rc.stdout.lower(), "device report should include CPU"

    ckpt = os.path.join(tempfile.gettempdir(), "omni_image_test.pth")
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "image-train",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--save-model",
            ckpt,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"image-train failed: {rc.stderr}\n{rc.stdout}"
    assert os.path.exists(ckpt), "omni image checkpoint not created"
    os.remove(ckpt)
    print("  [PASS] Omni CLI smoke")
except Exception as e:
    errors.append(f"Omni CLI smoke: {e}")
    print(f"  [FAIL] {e}")

print("\n[TEST 10] Advanced Omni features...")
try:
    best_dsl = os.path.join(tempfile.gettempdir(), "omni_best.dsl")
    best_pth = os.path.join(tempfile.gettempdir(), "omni_best.pth")
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "tabular-search",
            "--input-dim",
            "4",
            "--output-dim",
            "2",
            "--trials",
            "3",
            "--epochs-per-trial",
            "3",
            "--samples",
            "64",
            "--out-dsl",
            best_dsl,
            "--out-pth",
            best_pth,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"tabular-search failed: {rc.stderr}\n{rc.stdout}"
    assert os.path.exists(best_dsl), "search DSL not created"
    assert os.path.exists(best_pth), "search weights not created"

    img_ckpt = os.path.join(tempfile.gettempdir(), "omni_img_interp.pth")
    img_grid = os.path.join(tempfile.gettempdir(), "omni_interp_grid.png")
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "image-train",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--save-model",
            img_ckpt,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"image-train for interpolation failed: {rc.stderr}\n{rc.stdout}"
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "image-interpolate",
            "--checkpoint",
            img_ckpt,
            "--steps",
            "4",
            "--output-grid",
            img_grid,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"image-interpolate failed: {rc.stderr}\n{rc.stdout}"
    assert os.path.exists(img_grid), "interpolation grid not created"

    mm_ckpt = os.path.join(tempfile.gettempdir(), "omni_mm_text.pth")
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "multimodal-train",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--save-model",
            mm_ckpt,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"multimodal-train failed: {rc.stderr}\n{rc.stdout}"
    out_json = os.path.join(tempfile.gettempdir(), "omni_mm_out.json")
    rc = subprocess.run(
        [
            "python",
            "omni_cli.py",
            "multimodal-run",
            "--checkpoint",
            mm_ckpt,
            "--text",
            "an uncanny cybernetic owl staring through static",
            "--output-json",
            out_json,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, f"multimodal-run --text failed: {rc.stderr}\n{rc.stdout}"
    payload = json.loads(open(out_json, "r", encoding="utf-8").read())
    assert "output" in payload and len(payload["output"]) > 0, "multimodal output missing"

    # API server smoke test
    api_model = os.path.join(tempfile.gettempdir(), "omni_api_model.pth")
    dsl = "[4, 8], [8, 2]"
    model = create_modern_nn(parse_program(dsl))
    torch.save(model.state_dict(), api_model)
    proc = subprocess.Popen(
        [
            "python",
            "omni_cli.py",
            "serve-tabular",
            "--model",
            api_model,
            "--dsl",
            dsl,
            "--port",
            "8095",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        healthy = False
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8095/health", timeout=2) as r:
                    h = json.loads(r.read().decode("utf-8"))
                    healthy = h.get("status") == "ok"
                    if healthy:
                        break
            except Exception:
                time.sleep(0.8)
        assert healthy, "server did not become healthy in time"

        req = urllib.request.Request(
            "http://127.0.0.1:8095/infer",
            data=json.dumps({"inputs": [0.1, 0.2, 0.3, 0.4]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            out = json.loads(r.read().decode("utf-8"))
            assert "outputs" in out and len(out["outputs"]) == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    for p in [best_dsl, best_pth, img_ckpt, img_grid, mm_ckpt, out_json, api_model]:
        if os.path.exists(p):
            os.remove(p)
    print("  [PASS] Advanced Omni features")
except Exception as e:
    errors.append(f"Advanced Omni features: {e}")
    print(f"  [FAIL] {e}")

print("\n" + "=" * 60)
if errors:
    print(f"[FAIL] FAILED: {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("[PASS] ALL TESTS PASSED - No bugs found!")
print("=" * 60)
