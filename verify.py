import torch
from network import ModernMLP, RMSNorm
from parser_utils import parse_program, validate_dsl, DSL_PRESETS
from device_utils import detect_devices
from experimental_models import (
    ExperimentalTrainConfig,
    ImageAutoencoder,
    MultiModalFusionModel,
    SyntheticImageDataset,
    generate_interpolation_images,
    text_to_feature_vector,
    train_image_autoencoder,
    train_multimodal,
)

def test_infinity_blocks():
    print("Testing Infinity Blocks (MoE, GQA, Fractal)...")
    layer_defs = [
        {'type': 'linear', 'in': 64, 'out': 128},
        {'type': 'fractal', 'dim': 128, 'depth': 2},
        {'type': 'gqa', 'dim': 128, 'heads': 8, 'groups': 2},
        {'type': 'moe', 'dim': 128, 'experts': 8, 'shared': 1},
        {'type': 'linear', 'in': 128, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    x = torch.randn(4, 64)
    y = model(x)
    assert y.shape == (4, 10), f"Expected shape (4, 10), got {y.shape}"
    aux = model.get_aux_loss()
    assert torch.is_tensor(aux), "Expected MoE aux loss tensor"
    print("  PASS: Infinity Blocks")

def test_new_v4_layers():
    print("Testing v4.0 Layers (Dropout, Residual, Conv1D, LSTM, MoD)...")
    
    # Test DropoutBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    x = torch.randn(4, 32)
    y = model(x)
    assert y.shape == (4, 10), f"Dropout: Expected (4, 10), got {y.shape}"
    print("  PASS: DropoutBlock")
    
    # Test ResidualBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'residual', 'dim': 64, 'expansion': 4},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"Residual: Expected (4, 10), got {y.shape}"
    print("  PASS: ResidualBlock")
    
    # Test Conv1DBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'conv1d', 'dim': 64, 'kernel': 3},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"Conv1D: Expected (4, 10), got {y.shape}"
    print("  PASS: Conv1DBlock")
    
    # Test LSTMBlock
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'lstm', 'dim': 64, 'layers': 1},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"LSTM: Expected (4, 10), got {y.shape}"
    print("  PASS: LSTMBlock")

    # Test AdaptiveComputeBlock (Mixture-of-Depths inspired)
    layer_defs = [
        {'type': 'linear', 'in': 32, 'out': 64},
        {'type': 'mod', 'dim': 64, 'expansion': 4, 'threshold': 0.35},
        {'type': 'linear', 'in': 64, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    y = model(torch.randn(4, 32))
    assert y.shape == (4, 10), f"MoD: Expected (4, 10), got {y.shape}"
    print("  PASS: AdaptiveComputeBlock")

def test_training_engine():
    print("Testing Training Engine (multi-loss, grad clip, warmup)...")
    from trainer import TrainingEngine
    
    layer_defs = [{'type': 'linear', 'in': 4, 'out': 2}]
    model = ModernMLP(layer_defs)
    
    for loss_fn in ['MSE', 'Huber', 'MAE']:
        trainer = TrainingEngine(model, loss_fn=loss_fn, grad_clip=1.0, warmup_steps=3)
        X, y = trainer.generate_dummy_data(4, 2, n_samples=10)
        loss, lr, grad_norm = trainer.train_step(X, y)
        assert loss > 0, f"{loss_fn}: Loss should be positive"
        assert grad_norm >= 0, f"{loss_fn}: Grad norm should be non-negative"
    print("  PASS: Multi-loss training")
    
    # Test TorchScript export
    import os
    ts_path = "test_model.pt"
    trainer.export_torchscript(ts_path, 4)
    assert os.path.exists(ts_path), "TorchScript export failed"
    os.remove(ts_path)
    print("  PASS: TorchScript export")

def test_dsl_parsing():
    print("Testing DSL Parsing...")
    
    # Test basic parsing
    result = parse_program("[128, 64], [64, 10]")
    assert result is not None, "Basic parse failed"
    assert len(result) == 2, f"Expected 2 layers, got {len(result)}"
    print("  PASS: Basic linear layers")
    
    # Test new v4.0 layer parsing
    result = parse_program("dropout: [0.3], residual: [128], conv1d: [128, 5], lstm: [128], mod: [128, 4, 0.35]")
    assert result is not None, "v4.0 layer parse failed"
    assert len(result) == 5, f"Expected 5 layers, got {len(result)}"
    assert result[0]['type'] == 'dropout'
    assert result[1]['type'] == 'residual'
    assert result[2]['type'] == 'conv1d'
    assert result[3]['type'] == 'lstm'
    assert result[4]['type'] == 'mod'
    print("  PASS: v4.0 layer types")
    
    # Test full complex DSL
    result = parse_program("fractal: [256, 4], moe: [256, 16, 2], mod: [256, 4, 0.4], gqa: [256, 8, 2], [256, 10]")
    assert result is not None, "Complex parse failed"
    assert len(result) == 5
    print("  PASS: Complex mixed architecture")
    
    # Test validation
    issues, result = validate_dsl("[128, 64], [32, 10]")
    assert result is not None
    assert len(issues) > 0, "Should detect dimension mismatch"
    print("  PASS: DSL validation (dimension mismatch detection)")

def test_presets():
    print("Testing DSL Presets...")
    for name, dsl in DSL_PRESETS.items():
        result = parse_program(dsl)
        assert result is not None, f"Preset '{name}' failed to parse"
        assert len(result) > 0, f"Preset '{name}' produced empty result"
    print(f"  PASS: All {len(DSL_PRESETS)} presets parse correctly")

def test_model_summary():
    print("Testing Model Summary...")
    layer_defs = [
        {'type': 'linear', 'in': 64, 'out': 128},
        {'type': 'residual', 'dim': 128},
        {'type': 'linear', 'in': 128, 'out': 10}
    ]
    model = ModernMLP(layer_defs)
    summary, total = model.get_summary()
    assert len(summary) > 0, "Summary should have entries"
    assert total > 0, "Total params should be positive"
    print(f"  PASS: Model summary ({total:,} params)")

def test_device_backends():
    print("Testing device backend detection...")
    devices = detect_devices()
    assert "cpu" in devices, "CPU backend missing from detection map"
    assert devices["cpu"].available, "CPU should always be available"
    print("  PASS: Device backend detection")

def test_experimental_modes():
    print("Testing experimental image + multimodal modes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_model = ImageAutoencoder(image_size=32, latent_dim=32, channels=3)
    img_cfg = ExperimentalTrainConfig(epochs=1, batch_size=8, lr=1e-3, mixup_alpha=0.0, use_ema=False)
    img_ds = SyntheticImageDataset(n_samples=16, image_size=32)
    img_hist = train_image_autoencoder(img_model, img_cfg, device=device, dataset=img_ds)
    assert len(img_hist) == 1, "Image mode should produce one epoch loss"
    assert img_hist[0] > 0, "Image loss should be positive"
    interp = generate_interpolation_images(img_model, steps=5, seed_a=1, seed_b=2, device=device)
    assert interp.shape[0] == 5, "Interpolation should create requested number of frames"

    mm_model = MultiModalFusionModel(image_size=32, vec_dim=8, hidden_dim=32, out_dim=4)
    mm_cfg = ExperimentalTrainConfig(epochs=1, batch_size=8, lr=1e-3, use_ema=False)
    mm_hist = train_multimodal(mm_model, mm_cfg, device=device, dataset=None)
    assert len(mm_hist) == 1, "Multimodal mode should produce one epoch loss"
    assert mm_hist[0] > 0, "Multimodal loss should be positive"
    txt_vec = text_to_feature_vector("uncanny robotics", dim=8)
    assert txt_vec.shape[0] == 8, "Text vector dim mismatch"
    print("  PASS: Experimental modes")

if __name__ == "__main__":
    try:
        test_infinity_blocks()
        test_new_v4_layers()
        test_training_engine()
        test_dsl_parsing()
        test_presets()
        test_model_summary()
        test_device_backends()
        test_experimental_modes()
        print("\n[PASS] All NeuroDSL v4.0 Verification tests passed!")
    except Exception as e:
        print(f"\n[FAIL] Verification Failed: {e}")
        import traceback
        traceback.print_exc()



