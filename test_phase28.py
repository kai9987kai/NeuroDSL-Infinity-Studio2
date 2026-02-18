import torch
import torch.nn as nn
from parser_utils import create_modern_nn, validate_dsl
from network import (
    GatedLinearAttention, xLSTMBlock, TestTimeTrainLayer,
    ContrastiveHead, SparseAttention, SelfDistillBlock,
)
from trainer import ContrastiveLoss


def test_gated_linear_attention():
    print("Testing GatedLinearAttention...")
    dim = 64
    layer = GatedLinearAttention(dim, num_heads=4)
    x = torch.randn(4, dim)
    out = layer(x)
    assert out.shape == (4, dim), f"Shape mismatch: {out.shape}"
    # Gradient flow
    out.sum().backward()
    assert layer.q_proj.weight.grad is not None
    print("[ok] GatedLinearAttention pass")


def test_xlstm_block():
    print("Testing xLSTMBlock...")
    dim = 32
    layer = xLSTMBlock(dim)
    x = torch.randn(8, dim)
    out = layer(x)
    assert out.shape == (8, dim), f"Shape mismatch: {out.shape}"
    out.sum().backward()
    assert layer.W_i.weight.grad is not None
    print("[ok] xLSTMBlock pass")


def test_test_time_train_layer():
    print("Testing TestTimeTrainLayer...")
    dim = 64
    layer = TestTimeTrainLayer(dim, ttt_lr=0.01)

    x = torch.randn(4, dim)

    # Training mode: standard residual FFN
    layer.train()
    out_train = layer(x)
    assert out_train.shape == (4, dim), f"Train shape: {out_train.shape}"

    # Eval mode: TTT self-supervised update
    layer.eval()
    x_eval = torch.randn(4, dim, requires_grad=True)
    out_eval = layer(x_eval)
    assert out_eval.shape == (4, dim), f"Eval shape: {out_eval.shape}"
    print("[ok] TestTimeTrainLayer pass")


def test_contrastive_head():
    print("Testing ContrastiveHead...")
    dim = 128
    layer = ContrastiveHead(dim, proj_dim=64)
    x = torch.randn(4, dim)
    out = layer(x)
    assert out.shape == (4, dim), f"Shape mismatch: {out.shape}"
    out.sum().backward()
    print("[ok] ContrastiveHead pass")


def test_sparse_attention():
    print("Testing SparseAttention...")
    dim = 64
    layer = SparseAttention(dim, top_k=4, num_heads=4)
    x = torch.randn(4, dim)
    out = layer(x)
    assert out.shape == (4, dim), f"Shape mismatch: {out.shape}"
    out.sum().backward()
    print("[ok] SparseAttention pass")


def test_self_distill_block():
    print("Testing SelfDistillBlock...")
    dim = 32
    layer = SelfDistillBlock(dim, momentum=0.996)

    x = torch.randn(4, dim)

    # Training: teacher EMA update + distillation loss
    layer.train()
    out_train = layer(x)
    assert out_train.shape == (4, dim), f"Train shape: {out_train.shape}"
    aux = layer.get_aux_loss()
    assert aux.item() >= 0, f"Aux loss should be non-negative: {aux.item()}"

    # Eval mode
    layer.eval()
    out_eval = layer(x)
    assert out_eval.shape == (4, dim), f"Eval shape: {out_eval.shape}"
    print("[ok] SelfDistillBlock pass")


def test_contrastive_loss():
    print("Testing ContrastiveLoss...")
    loss_fn = ContrastiveLoss(temperature=0.07)
    z = torch.randn(8, 64)
    targets = torch.zeros(8)  # unused in contrastive but required by API
    loss = loss_fn(z, targets)
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
    assert torch.isfinite(loss), f"Loss should be finite: {loss.item()}"
    print(f"[ok] ContrastiveLoss pass (loss={loss.item():.4f})")


def test_dsl_integration():
    print("Testing DSL Integration...")
    program = "gla: [64], xlstm: [64], ttt: [64], sparse_attn: [64, 8], [64, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, f"Parse failed: {issues}"
    model = create_modern_nn(layer_defs)
    assert model is not None
    layers_found = [type(l).__name__ for l in model.layers]
    print(f"  Layers: {layers_found}")
    assert "GatedLinearAttention" in layers_found
    assert "xLSTMBlock" in layers_found
    assert "TestTimeTrainLayer" in layers_found
    assert "SparseAttention" in layers_found

    # Forward pass
    x = torch.randn(2, 64)
    model.eval()
    out = model(x)
    assert out.shape[0] == 2, f"Output batch mismatch: {out.shape}"
    print("[ok] DSL Integration pass")


def test_frontier_preset():
    print("Testing Frontier Intelligence preset...")
    program = "gla: [128, 4], xlstm: [128], ttt: [128], sparse_attn: [128, 8], [128, 10]"
    issues, layer_defs = validate_dsl(program)
    assert layer_defs is not None, f"Parse failed: {issues}"
    model = create_modern_nn(layer_defs)
    x = torch.randn(2, 128)
    model.eval()
    out = model(x)
    assert out.shape == (2, 10), f"Output shape mismatch: {out.shape}"
    print("[ok] Frontier Intelligence preset pass")


if __name__ == "__main__":
    test_gated_linear_attention()
    test_xlstm_block()
    test_test_time_train_layer()
    test_contrastive_head()
    test_sparse_attention()
    test_self_distill_block()
    test_contrastive_loss()
    test_dsl_integration()
    test_frontier_preset()
    print("\n[ok] ALL PHASE 28 TESTS PASSED")

