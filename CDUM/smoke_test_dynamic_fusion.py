#!/usr/bin/env python3
"""Smoke test suite for CPMDynamicFusion variant.

Validates the CPM Dynamic Fusion architecture combining representation-level
expert routing with VALOR treatment-gated branch and dynamic fusion router.

Run from the project root:
    python -m CDUM.smoke_test_dynamic_fusion
  or:
    python CDUM/smoke_test_dynamic_fusion.py
"""

import os
import sys
import tempfile
import traceback
from typing import Dict, List, Tuple

# Ensure project root is on the path so `CDUM` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def header(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def test_imports() -> bool:
    """Test 1: CPMDynamicFusion and related classes import correctly."""
    header("Test 1: Imports")
    from CDUM.cpm import CPM
    from CDUM.trainer import CPMTrainer
    from CDUM.variants.cpm_dynamic_fusion import (
        CPMDynamicFusion,
        DynamicFusionRouter,
        ValorTreatmentGatedBranch,
    )

    assert CPMDynamicFusion is not None
    assert DynamicFusionRouter is not None
    assert ValorTreatmentGatedBranch is not None
    assert CPM is not None
    assert CPMTrainer is not None
    print("  ✅ CPMDynamicFusion and variant components imported successfully")
    return True


def test_default_shapes_and_validations(
    B: int = 8,
    num_features: int = 12,
    num_bins: int = 101,
    emb_dim: int = 32,
    treatment_dim: int = 128,
    refine_hidden: int = 64,
    refine_dim: int = 32,
    num_experts: int = 3,
    expert_hidden: int = 128,
    expert_dim: int = 64,
    tower_hidden: int = 32,
) -> bool:
    """Test 2: Default architecture shapes, sum-to-1, mask [0,1], and finite values."""
    header("Test 2: Default Shapes and Intermediate Validations")
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(42)
    model = CPMDynamicFusion(
        num_features=num_features,
        num_bins=num_bins,
        embedding_dim=emb_dim,
        treatment_dim=treatment_dim,
        refine_hidden_dim=refine_hidden,
        refine_dim=refine_dim,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden,
        expert_dim=expert_dim,
        tower_hidden_dim=tower_hidden,
    )

    x_ids = torch.randint(0, num_bins, (B, num_features))
    t = torch.randint(0, 2, (B,))

    outputs = model(x_ids, t, return_diagnostics=True)
    assert "diagnostics" in outputs, "Diagnostics missing when return_diagnostics=True"
    diagnostics = outputs["diagnostics"]

    # Verify both treatment branches (0 and 1)
    for tid in [0, 1]:
        assert tid in diagnostics, f"Branch {tid} diagnostics missing"
        diag = diagnostics[tid]

        # e_x[B, 384], e_t[B, 128]
        expected_feature_dim = num_features * emb_dim  # 384
        assert diag["e_x"].shape == (B, expected_feature_dim), (
            f"tid={tid} e_x shape mismatch: expected ({B}, {expected_feature_dim}), got {diag['e_x'].shape}"
        )
        assert diag["e_t"].shape == (B, treatment_dim), (
            f"tid={tid} e_t shape mismatch: expected ({B}, {treatment_dim}), got {diag['e_t'].shape}"
        )

        # experts[B, 3, 64]
        assert diag["expert_outputs"].shape == (B, num_experts, expert_dim), (
            f"tid={tid} expert_outputs shape mismatch: expected ({B}, {num_experts}, {expert_dim}), "
            f"got {diag['expert_outputs'].shape}"
        )

        # a_C[B, 3]
        assert diag["a_C"].shape == (B, num_experts), (
            f"tid={tid} a_C shape mismatch: expected ({B}, {num_experts}), got {diag['a_C'].shape}"
        )

        # z_C, h_x, m_t, interaction, z_V, z_F all [B, 64]
        for key in ["z_C", "h_x", "m_t", "interaction", "z_V", "z_F"]:
            assert diag[key].shape == (B, expert_dim), (
                f"tid={tid} {key} shape mismatch: expected ({B}, {expert_dim}), got {diag[key].shape}"
            )

        # q_R[B, 128]
        expected_router_in = 2 * expert_dim
        assert diag["q_R"].shape == (B, expected_router_in), (
            f"tid={tid} q_R shape mismatch: expected ({B}, {expected_router_in}), got {diag['q_R'].shape}"
        )

        # logits/pi[B, 2]
        assert diag["router_logits"].shape == (B, 2), (
            f"tid={tid} router_logits shape mismatch: expected ({B}, 2), got {diag['router_logits'].shape}"
        )
        assert diag["pi"].shape == (B, 2), (
            f"tid={tid} pi shape mismatch: expected ({B}, 2), got {diag['pi'].shape}"
        )

        # pred[B, 1] for both treatments
        assert diag["y_hat"].shape == (B, 1), (
            f"tid={tid} y_hat shape mismatch: expected ({B}, 1), got {diag['y_hat'].shape}"
        )

        # Validation: pi sums to 1
        pi_sum = diag["pi"].sum(dim=-1)
        assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), (
            f"tid={tid} pi does not sum to 1: {pi_sum}"
        )

        # Validation: expert gates a_C sum to 1
        a_C_sum = diag["a_C"].sum(dim=-1)
        assert torch.allclose(a_C_sum, torch.ones_like(a_C_sum), atol=1e-5), (
            f"tid={tid} a_C does not sum to 1: {a_C_sum}"
        )

        # Validation: m_t mask in [0, 1]
        m_t = diag["m_t"]
        assert m_t.min() >= 0.0 and m_t.max() <= 1.0, (
            f"tid={tid} m_t out of [0, 1] bounds: min={m_t.min()}, max={m_t.max()}"
        )

        # Validation: finite tensors
        for key in ["e_x", "e_t", "expert_outputs", "a_C", "z_C", "h_x", "m_t", "interaction", "z_V", "q_R", "router_logits", "pi", "z_F", "y_hat"]:
            assert torch.isfinite(diag[key]).all(), f"tid={tid} non-finite values in {key}"

    # Check top-level prediction shapes
    for key in ["y_factual", "y0", "y1", "y0_hat", "y1_hat", "uplift"]:
        assert outputs[key].shape == (B, 1), f"{key} shape mismatch: {outputs[key].shape}"
        assert torch.isfinite(outputs[key]).all(), f"{key} contains non-finite values"

    # Assert valor_branch.mlp hidden/output dims and ReLU pipeline using current config
    mlp = model.valor_branch.mlp
    assert mlp.hidden.in_features == expert_dim, f"MLP hidden in_features mismatch: expected {expert_dim}, got {mlp.hidden.in_features}"
    assert mlp.hidden.out_features == expert_hidden, f"MLP hidden out_features mismatch: expected {expert_hidden}, got {mlp.hidden.out_features}"
    assert mlp.output.in_features == expert_hidden, f"MLP output in_features mismatch: expected {expert_hidden}, got {mlp.output.in_features}"
    assert mlp.output.out_features == expert_dim, f"MLP output out_features mismatch: expected {expert_dim}, got {mlp.output.out_features}"
    assert isinstance(mlp.relu, nn.ReLU), f"MLP relu expected nn.ReLU, got {type(mlp.relu)}"

    print("  ✅ All default shapes verified for treatments 0 & 1:")
    print("     e_x[B,384], e_t[B,128], experts[B,3,64], a_C[B,3]")
    print("     z_C/h_x/m_t/interaction/z_V/z_F[B,64], q_R[B,128], logits/pi[B,2], pred[B,1]")
    print("  ✅ pi sums to 1, a_C sums to 1, m_t in [0,1], all tensors finite")
    print("  ✅ valor_branch.mlp dimensions and ReLU pipeline verified")
    return True


def test_custom_dimensions_and_batch_one() -> bool:
    """Test 3: Custom dimensions (treatment_dim != refine_dim, expert_dim != both, custom router hidden) with B=1."""
    header("Test 3: Custom Dimensions and B=1")
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(123)
    B = 1
    num_features = 10
    emb_dim = 16
    treatment_dim = 80
    refine_dim = 40
    tower_hidden_dim = 40  # must equal refine_dim
    expert_dim = 50
    router_hidden_dim = 70
    num_experts = 4

    assert treatment_dim != refine_dim, "Sanity: treatment_dim must differ from refine_dim"
    assert expert_dim != treatment_dim and expert_dim != refine_dim, "Sanity: expert_dim must differ from both"
    assert router_hidden_dim != expert_dim, "Sanity: router_hidden_dim must differ from expert_dim"

    model = CPMDynamicFusion(
        num_features=num_features,
        num_bins=50,
        embedding_dim=emb_dim,
        treatment_dim=treatment_dim,
        refine_hidden_dim=30,
        refine_dim=refine_dim,
        tower_hidden_dim=tower_hidden_dim,
        num_experts=num_experts,
        expert_hidden_dim=60,
        expert_dim=expert_dim,
        router_hidden_dim=router_hidden_dim,
    )

    x_ids = torch.randint(0, 50, (B, num_features))
    t = torch.tensor([0], dtype=torch.long)

    outputs = model(x_ids, t, return_diagnostics=True)
    diag = outputs["diagnostics"]

    expected_feat_dim = num_features * emb_dim  # 160
    for tid in [0, 1]:
        d = diag[tid]
        assert d["e_x"].shape == (B, expected_feat_dim), f"e_x shape: {d['e_x'].shape}"
        assert d["e_t"].shape == (B, treatment_dim), f"e_t shape: {d['e_t'].shape}"
        assert d["expert_outputs"].shape == (B, num_experts, expert_dim), f"expert_outputs shape: {d['expert_outputs'].shape}"
        assert d["a_C"].shape == (B, num_experts), f"a_C shape: {d['a_C'].shape}"
        assert d["z_C"].shape == (B, expert_dim), f"z_C shape: {d['z_C'].shape}"
        assert d["h_x"].shape == (B, expert_dim), f"h_x shape: {d['h_x'].shape}"
        assert d["m_t"].shape == (B, expert_dim), f"m_t shape: {d['m_t'].shape}"
        assert d["interaction"].shape == (B, expert_dim), f"interaction shape: {d['interaction'].shape}"
        assert torch.isfinite(d["interaction"]).all(), f"tid={tid} non-finite values in interaction"
        assert d["z_V"].shape == (B, expert_dim), f"z_V shape: {d['z_V'].shape}"
        assert d["q_R"].shape == (B, 2 * expert_dim), f"q_R shape: {d['q_R'].shape}"
        assert d["router_logits"].shape == (B, 2), f"router_logits shape: {d['router_logits'].shape}"
        assert d["pi"].shape == (B, 2), f"pi shape: {d['pi'].shape}"
        assert d["z_F"].shape == (B, expert_dim), f"z_F shape: {d['z_F'].shape}"
        assert d["y_hat"].shape == (B, 1), f"y_hat shape: {d['y_hat'].shape}"
        assert torch.allclose(d["pi"].sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(d["a_C"].sum(dim=-1), torch.ones(B), atol=1e-5)

    assert outputs["y_factual"].shape == (B, 1)
    assert outputs["uplift"].shape == (B, 1)

    # Assert custom MLP hidden/output dims and ReLU pipeline
    custom_mlp = model.valor_branch.mlp
    assert custom_mlp.hidden.in_features == expert_dim, f"Custom MLP hidden in_features: {custom_mlp.hidden.in_features}"
    assert custom_mlp.hidden.out_features == 60, f"Custom MLP hidden out_features: {custom_mlp.hidden.out_features}"
    assert custom_mlp.output.in_features == 60, f"Custom MLP output in_features: {custom_mlp.output.in_features}"
    assert custom_mlp.output.out_features == expert_dim, f"Custom MLP output out_features: {custom_mlp.output.out_features}"
    assert isinstance(custom_mlp.relu, nn.ReLU), f"Custom MLP relu: {type(custom_mlp.relu)}"

    print(f"  ✅ Custom dims verified (treatment_dim={treatment_dim}, refine_dim={refine_dim}, "
          f"expert_dim={expert_dim}, router_hidden={router_hidden_dim}, B={B})")
    print("  ✅ Custom valor_branch.mlp dimensions (50->60->50) and ReLU pipeline verified")
    return True


def test_intermediates_algebra_and_tower_hooks() -> bool:
    """Test 4: Verify intermediate tensor algebra, raw treatment embedding for VALOR, and tower hook inputs."""
    header("Test 4: Intermediates Algebra, Raw Treatment Embedding, and Tower Hooks")
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(99)
    B = 4
    model = CPMDynamicFusion()

    x_ids = torch.randint(0, 101, (B, 12))
    t = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    # Hook tower inputs to verify z_F is fed into towers
    captured_tower_inputs: Dict[str, torch.Tensor] = {}

    def hook_control(module, inp, out):
        captured_tower_inputs["control"] = inp[0]

    def hook_treatment(module, inp, out):
        captured_tower_inputs["treatment"] = inp[0]

    h_c = model.control_tower.register_forward_hook(hook_control)
    h_t = model.treatment_tower.register_forward_hook(hook_treatment)

    try:
        outputs = model(x_ids, t, return_diagnostics=True)
    finally:
        h_c.remove()
        h_t.remove()

    diag0 = outputs["diagnostics"][0]
    diag1 = outputs["diagnostics"][1]

    # Verify tower inputs match z_F
    assert "control" in captured_tower_inputs, "Control tower hook did not capture input"
    assert "treatment" in captured_tower_inputs, "Treatment tower hook did not capture input"
    assert torch.allclose(captured_tower_inputs["control"], diag0["z_F"], atol=1e-6), (
        "Control tower input does NOT match z_F from branch 0"
    )
    assert torch.allclose(captured_tower_inputs["treatment"], diag1["z_F"], atol=1e-6), (
        "Treatment tower input does NOT match z_F from branch 1"
    )
    print("  ✅ Forward hooks verified: z_F is explicitly passed to both TreatmentTowers")

    # Verify VALOR uses raw treatment embedding e_t (not e_guidance or e_indicator)
    for tid, diag in [(0, diag0), (1, diag1)]:
        t_single = torch.full((B,), tid, dtype=torch.long)
        expected_raw_et = model.encoder.encode_treatment(t_single)
        assert torch.allclose(diag["e_t"], expected_raw_et, atol=1e-6), (
            f"tid={tid} VALOR e_t does not match raw encoder.encode_treatment(t)"
        )
        assert diag["e_t"].shape[-1] == model.treatment_dim
        assert diag["e_guidance"].shape[-1] == model.refine_dim
        # Dimension or value check ensures raw e_t was used
        assert diag["e_t"].shape[-1] != diag["e_guidance"].shape[-1] or not torch.allclose(diag["e_t"], diag["e_guidance"]), (
            f"tid={tid} raw e_t incorrectly matches refined guidance"
        )
    print("  ✅ VALOR branch verified to use raw treatment embedding e_t (never e_guidance/e_indicator)")

    # Verify mathematical algebra for intermediates in both branches
    for tid, diag in [(0, diag0), (1, diag1)]:
        # 1. z_C = sum(a_C * experts)
        reconstructed_z_C = (diag["expert_outputs"] * diag["a_C"].unsqueeze(-1)).sum(dim=1)
        assert torch.allclose(diag["z_C"], reconstructed_z_C, atol=1e-6), f"tid={tid} z_C algebra mismatch"

        # 2. h_x = Linear(e_x)
        expected_h_x = model.valor_branch.linear_x(diag["e_x"])
        assert torch.allclose(diag["h_x"], expected_h_x, atol=1e-6), f"tid={tid} h_x algebra mismatch"

        # 3. m_t = Sigmoid(Linear(e_t))
        expected_m_t = torch.sigmoid(model.valor_branch.linear_t(diag["e_t"]))
        assert torch.allclose(diag["m_t"], expected_m_t, atol=1e-6), f"tid={tid} m_t algebra mismatch"

        # 4. interaction = h_x * m_t
        expected_interaction = diag["h_x"] * diag["m_t"]
        assert torch.allclose(diag["interaction"], expected_interaction, atol=1e-6), f"tid={tid} interaction algebra mismatch"

        # 5. z_V = model.valor_branch.mlp(interaction)
        expected_z_V = model.valor_branch.mlp(diag["interaction"])
        assert torch.allclose(diag["z_V"], expected_z_V, atol=1e-6), f"tid={tid} z_V algebra mismatch"

        # Verify mlp architecture and pipeline: Linear(D, hidden) -> ReLU -> Linear(hidden, D) -> ReLU
        mlp_h = model.valor_branch.mlp.relu(model.valor_branch.mlp.hidden(diag["interaction"]))
        manual_z_V = model.valor_branch.mlp.relu(model.valor_branch.mlp.output(mlp_h))
        assert torch.allclose(diag["z_V"], manual_z_V, atol=1e-6), f"tid={tid} manual mlp pipeline mismatch"

        # 6. q_R = Concat(z_C, z_V) - post-MLP z_V is router input
        expected_q_R = torch.cat([diag["z_C"], diag["z_V"]], dim=-1)
        assert torch.allclose(diag["q_R"], expected_q_R, atol=1e-6), f"tid={tid} q_R algebra mismatch"
        # Confirm post-MLP z_V is router input (and not pre-MLP interaction)
        router_valor_input = diag["q_R"][:, diag["z_C"].shape[-1]:]
        assert torch.allclose(router_valor_input, diag["z_V"], atol=1e-6), f"tid={tid} router input is not post-MLP z_V"
        assert not torch.allclose(router_valor_input, diag["interaction"], atol=1e-6), f"tid={tid} router input equals pre-MLP interaction"

        # 7. router_logits = fc2(ReLU(fc1(q_R)))
        expected_logits = model.router.fc2(torch.relu(model.router.fc1(diag["q_R"])))
        assert torch.allclose(diag["router_logits"], expected_logits, atol=1e-6), f"tid={tid} logits algebra mismatch"

        # 8. pi = Softmax(router_logits, -1)
        expected_pi = torch.softmax(diag["router_logits"], dim=-1)
        assert torch.allclose(diag["pi"], expected_pi, atol=1e-6), f"tid={tid} pi algebra mismatch"

        # 9. z_F = pi[..., 0:1] * z_C + pi[..., 1:2] * z_V
        expected_z_F = diag["pi"][..., 0:1] * diag["z_C"] + diag["pi"][..., 1:2] * diag["z_V"]
        assert torch.allclose(diag["z_F"], expected_z_F, atol=1e-6), f"tid={tid} z_F algebra mismatch"

    print("  ✅ All intermediate algebra verified exactly: h_x, m_t, interaction, z_V (post-MLP), q_R, logits, pi, z_F")
    return True


def test_backprop_new_parameters_gradient_flow() -> bool:
    """Test 5: Backprop existing CPMTrainer.compute_loss and assert all 12 new parameter tensors have nonzero grad."""
    header("Test 5: CPMTrainer.compute_loss Backprop and New Parameter Gradients")
    from CDUM.trainer import CPMTrainer
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(42)
    model = CPMDynamicFusion()
    trainer = CPMTrainer(model=model, lr=1e-3, device="cpu")

    B = 16
    x_ids = torch.randint(0, 101, (B, 12))
    # Seeded main batch with balanced treatments
    t = torch.tensor([0, 1] * (B // 2), dtype=torch.long)
    outcome = torch.randn(B)

    model.zero_grad()
    loss = trainer.compute_loss(x_ids, t, outcome)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()

    new_param_dict = {
        "valor_branch.linear_x.weight": model.valor_branch.linear_x.weight,
        "valor_branch.linear_x.bias": model.valor_branch.linear_x.bias,
        "valor_branch.linear_t.weight": model.valor_branch.linear_t.weight,
        "valor_branch.linear_t.bias": model.valor_branch.linear_t.bias,
        "valor_branch.mlp.hidden.weight": model.valor_branch.mlp.hidden.weight,
        "valor_branch.mlp.hidden.bias": model.valor_branch.mlp.hidden.bias,
        "valor_branch.mlp.output.weight": model.valor_branch.mlp.output.weight,
        "valor_branch.mlp.output.bias": model.valor_branch.mlp.output.bias,
        "router.fc1.weight": model.router.fc1.weight,
        "router.fc1.bias": model.router.fc1.bias,
        "router.fc2.weight": model.router.fc2.weight,
        "router.fc2.bias": model.router.fc2.bias,
    }

    assert len(new_param_dict) == 12, f"Expected exactly 12 new parameter tensors, found {len(new_param_dict)}"

    for name, param in new_param_dict.items():
        assert param.grad is not None, f"Parameter {name} has None gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"
        grad_norm = param.grad.norm().item()
        assert grad_norm > 0.0, f"Parameter {name} has zero gradient norm"
        print(f"     - {name:32s}: grad norm = {grad_norm:.6f}")

    print("  ✅ All 12 new parameter tensors have non-None, finite gradients with nonzero norm")
    return True


def test_diagnostics_statistics_and_loss_invariance() -> bool:
    """Test 6: Capture/print pi_C/pi_V, entropy, a_C, m_t; verify diagnostics never affect loss."""
    header("Test 6: Diagnostics Statistics and Loss Invariance")
    import math
    from CDUM.trainer import CPMTrainer
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(2026)
    model = CPMDynamicFusion()
    trainer = CPMTrainer(model=model, device="cpu")

    B = 16
    x_ids = torch.randint(0, 101, (B, 12))
    t = torch.tensor([0, 1] * 8, dtype=torch.long)
    outcome = torch.randn(B)

    # Forward with diagnostics
    out_diag = model(x_ids, t, return_diagnostics=True)
    # Forward without diagnostics
    out_norm = model(x_ids, t, return_diagnostics=False)

    # 1. Capture and print distribution statistics
    for tid in [0, 1]:
        d = out_diag["diagnostics"][tid]
        pi = d["pi"]
        pi_C_mean = pi[:, 0].mean().item()
        pi_V_mean = pi[:, 1].mean().item()
        entropy = -(pi * torch.log(pi.clamp(min=1e-12))).sum(dim=-1).mean().item()
        a_C_mean = d["a_C"].mean(dim=0).tolist()
        m_t_mean = d["m_t"].mean().item()
        interaction_mean = d["interaction"].mean().item()
        z_V_mean = d["z_V"].mean().item()

        print(f"  Branch {tid} (Treatment {tid}):")
        print(f"     Mean pi_C (CPM branch): {pi_C_mean:.4f}")
        print(f"     Mean pi_V (VALOR):      {pi_V_mean:.4f}")
        print(f"     Entropy -sum(pi*log pi): {entropy:.4f}")
        print(f"     Mean a_C per expert:    {[round(x, 4) for x in a_C_mean]}")
        print(f"     Mean m_t gating mask:   {m_t_mean:.4f}")
        print(f"     Mean interaction:       {interaction_mean:.4f}")
        print(f"     Mean z_V (post-MLP):    {z_V_mean:.4f}")

    # 2. Verify "diagnostics never loss": outputs match and loss uses only y_factual
    for key in ["y_factual", "y0", "y1", "y0_hat", "y1_hat", "uplift", "g0", "g1", "ind0", "ind1"]:
        assert torch.allclose(out_diag[key], out_norm[key], atol=1e-7), (
            f"Key '{key}' differs between diagnostic and non-diagnostic forward passes"
        )

    # Verify trainer.compute_loss is pure Huber on y_factual
    loss_val = trainer.compute_loss(x_ids, t, outcome)
    manual_huber = nn.HuberLoss(delta=1.0)(out_norm["y_factual"], outcome.view(-1, 1))
    assert torch.allclose(loss_val, manual_huber, atol=1e-7), "compute_loss does not match Huber on y_factual"
    print("  ✅ Verified: diagnostics never alter forward outputs or objective Huber loss")
    return True


def test_baseline_preservation_and_forced_cpm() -> bool:
    """Test 7: Load base CPM weights into variant strict=False (missing only new branch/router),
    force router CPM-only via hook, and verify exact match with baseline across ALL keys.
    """
    header("Test 7: Baseline Preservation and Forced CPM Router")
    from CDUM.cpm import CPM
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(42)
    base_model = CPM()
    torch.manual_seed(42)
    variant_model = CPMDynamicFusion()

    # Load base weights into variant
    load_res = variant_model.load_state_dict(base_model.state_dict(), strict=False)
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys found: {load_res.unexpected_keys}"

    expected_missing = {
        "valor_branch.linear_x.weight",
        "valor_branch.linear_x.bias",
        "valor_branch.linear_t.weight",
        "valor_branch.linear_t.bias",
        "valor_branch.mlp.hidden.weight",
        "valor_branch.mlp.hidden.bias",
        "valor_branch.mlp.output.weight",
        "valor_branch.mlp.output.bias",
        "router.fc1.weight",
        "router.fc1.bias",
        "router.fc2.weight",
        "router.fc2.bias",
    }
    missing_set = set(load_res.missing_keys)
    assert missing_set == expected_missing, (
        f"Missing keys mismatch: expected {expected_missing}, got {missing_set}"
    )
    print("  ✅ Base CPM weights loaded into variant: missing ONLY the 12 new branch/router parameters")

    # Hook to force router CPM-only: returns z_C and pi=[1.0, 0.0]
    def force_cpm_hook(module, args, output):
        z_C, z_V = args[0], args[1]
        B = z_C.shape[0]
        pi_cpm = torch.zeros(B, 2, device=z_C.device, dtype=z_C.dtype)
        pi_cpm[:, 0] = 1.0  # 100% on z_C, 0% on z_V
        z_F = z_C
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            # return_intermediates=True
            diag = dict(output[1])
            diag["pi"] = pi_cpm
            diag["z_F"] = z_F
            return (z_F, diag)
        return (z_F, pi_cpm)

    hook_handle = variant_model.router.register_forward_hook(force_cpm_hook)

    try:
        torch.manual_seed(101)
        B = 12
        x_ids = torch.randint(0, 101, (B, 12))
        t = torch.randint(0, 2, (B,))

        base_out = base_model(x_ids, t)
        var_out_norm = variant_model(x_ids, t, return_diagnostics=False)
        var_out_diag = variant_model(x_ids, t, return_diagnostics=True)

        assert set(var_out_norm.keys()) == set(base_out.keys()), (
            f"Default variant output key set mismatch: {set(var_out_norm.keys())} != {set(base_out.keys())}"
        )
        all_base_keys = ["y_factual", "y0", "y1", "y0_hat", "y1_hat", "uplift", "g0", "g1", "ind0", "ind1"]
        for k in all_base_keys:
            assert k in var_out_norm, f"Key {k} missing from variant outputs"
            max_diff_norm = (base_out[k] - var_out_norm[k]).abs().max().item()
            max_diff_diag = (base_out[k] - var_out_diag[k]).abs().max().item()
            assert max_diff_norm < 1e-6, f"Key {k} normal diff too large: {max_diff_norm}"
            assert max_diff_diag < 1e-6, f"Key {k} diag diff too large: {max_diff_diag}"
            print(f"     Key '{k:10s}': max diff = {max_diff_norm:.2e}")
    finally:
        hook_handle.remove()

    print("  ✅ Forced CPM-only router perfectly reproduces base CPM outputs across ALL 10 keys")
    return True


def test_factual_masking_and_counterfactual_independence() -> bool:
    """Test 8: Verify factual masks match selected branches, and counterfactual y0/y1 are independent of observed t."""
    header("Test 8: Factual Masking and Counterfactual Independence")
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(777)
    model = CPMDynamicFusion()

    B = 16
    x_ids = torch.randint(0, 101, (B, 12))

    t_0 = torch.zeros(B, dtype=torch.long)
    t_1 = torch.ones(B, dtype=torch.long)
    t_mixed = torch.tensor([0, 1] * (B // 2), dtype=torch.long)

    out_0 = model(x_ids, t_0)
    out_1 = model(x_ids, t_1)
    out_m = model(x_ids, t_mixed)

    # 1. Factual masking
    # When t=0, y_factual must equal y0 and y0_hat
    assert torch.allclose(out_0["y_factual"], out_0["y0"], atol=1e-6)
    assert torch.allclose(out_0["y_factual"], out_0["y0_hat"], atol=1e-6)

    # When t=1, y_factual must equal y1 and y1_hat
    assert torch.allclose(out_1["y_factual"], out_1["y1"], atol=1e-6)
    assert torch.allclose(out_1["y_factual"], out_1["y1_hat"], atol=1e-6)

    # In mixed batch:
    mask0 = (t_mixed == 0)
    mask1 = (t_mixed == 1)
    assert torch.allclose(out_m["y_factual"][mask0], out_m["y0"][mask0], atol=1e-6)
    assert torch.allclose(out_m["y_factual"][mask1], out_m["y1"][mask1], atol=1e-6)
    print("  ✅ Factual masking verified (y_factual == y0 for t=0, y1 for t=1)")

    # 2. Counterfactual independence: y0 and y1 are independent of observed treatment t
    assert torch.allclose(out_0["y0"], out_1["y0"], atol=1e-6), "y0 varies with observed t!"
    assert torch.allclose(out_0["y0"], out_m["y0"], atol=1e-6), "y0 varies with observed t!"
    assert torch.allclose(out_0["y1"], out_1["y1"], atol=1e-6), "y1 varies with observed t!"
    assert torch.allclose(out_0["y1"], out_m["y1"], atol=1e-6), "y1 varies with observed t!"
    assert torch.allclose(out_0["uplift"], out_1["uplift"], atol=1e-6), "uplift varies with observed t!"
    print("  ✅ Counterfactual independence verified: y0, y1, uplift depend solely on x_ids, not observed t")
    return True


def test_factual_only_gradient_isolation() -> bool:
    """Test 9: Factual-only gradient isolation for all-control and all-treatment batches.
    Assert inactive tower/gate/treatment embedding row have zero gradients, while shared new parameters participate.
    """
    header("Test 9: Factual-Only Gradient Isolation")
    from CDUM.trainer import CPMTrainer
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(888)
    model = CPMDynamicFusion()
    trainer = CPMTrainer(model=model, device="cpu")

    B = 16
    x_ids = torch.randint(0, 101, (B, 12))
    outcome = torch.randn(B)

    # Shared new parameter references (12 tensors)
    new_params = [
        model.valor_branch.linear_x.weight,
        model.valor_branch.linear_x.bias,
        model.valor_branch.linear_t.weight,
        model.valor_branch.linear_t.bias,
        model.valor_branch.mlp.hidden.weight,
        model.valor_branch.mlp.hidden.bias,
        model.valor_branch.mlp.output.weight,
        model.valor_branch.mlp.output.bias,
        model.router.fc1.weight,
        model.router.fc1.bias,
        model.router.fc2.weight,
        model.router.fc2.bias,
    ]

    # -------------------------------------------------------------
    # Case A: All-control batch (t=0 for all samples)
    # -------------------------------------------------------------
    model.zero_grad()
    t_control = torch.zeros(B, dtype=torch.long)
    loss_ctrl = trainer.compute_loss(x_ids, t_control, outcome)
    loss_ctrl.backward()

    # Inactive treatment branch checks
    for name, p in model.treatment_tower.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0, (
            f"Control batch leaked gradient to inactive treatment_tower parameter {name}"
        )
    for name, p in model.treatment_gate.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0, (
            f"Control batch leaked gradient to inactive treatment_gate parameter {name}"
        )
    # Inactive treatment embedding row (row 1)
    treat_row1_grad = model.encoder.treatment_embeddings.weight.grad[1]
    assert treat_row1_grad.norm().item() == 0.0, "Control batch leaked gradient to treatment embedding row 1"

    # Active control branch checks
    ctrl_tower_grad_sum = sum(p.grad.norm().item() for p in model.control_tower.parameters() if p.grad is not None)
    ctrl_gate_grad_sum = sum(p.grad.norm().item() for p in model.control_gate.parameters() if p.grad is not None)
    treat_row0_grad = model.encoder.treatment_embeddings.weight.grad[0]
    assert ctrl_tower_grad_sum > 0.0, "Control tower received no gradients in control batch"
    assert ctrl_gate_grad_sum > 0.0, "Control gate received no gradients in control batch"
    assert treat_row0_grad.norm().item() > 0.0, "Treatment embedding row 0 received no gradients in control batch"

    # Shared new parameters participate (all 12 tensors including MLP)
    for i, p in enumerate(new_params):
        assert p.grad is not None and p.grad.norm().item() > 0.0, (
            f"Shared new parameter {i} did not participate in control batch gradient flow"
        )
    # Ensure MLP gets gradient on active treatment (control)
    for name, p in model.valor_branch.mlp.named_parameters():
        assert p.grad is not None and p.grad.norm().item() > 0.0, (
            f"MLP parameter {name} did not receive gradients in control batch"
        )
    print("  ✅ All-control batch isolation verified:")
    print("     - Inactive treatment_tower, treatment_gate, treatment embedding row 1 grads == 0")
    print("     - Active control_tower, control_gate, treatment embedding row 0 grads > 0")
    print("     - Shared new VALOR & router parameters (including 4 MLP tensors) actively participate")

    # -------------------------------------------------------------
    # Case B: All-treatment batch (t=1 for all samples)
    # -------------------------------------------------------------
    model.zero_grad()
    t_treatment = torch.ones(B, dtype=torch.long)
    loss_treat = trainer.compute_loss(x_ids, t_treatment, outcome)
    loss_treat.backward()

    # Inactive control branch checks
    for name, p in model.control_tower.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0, (
            f"Treatment batch leaked gradient to inactive control_tower parameter {name}"
        )
    for name, p in model.control_gate.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0, (
            f"Treatment batch leaked gradient to inactive control_gate parameter {name}"
        )
    # Inactive treatment embedding row (row 0)
    treat_row0_grad = model.encoder.treatment_embeddings.weight.grad[0]
    assert treat_row0_grad.norm().item() == 0.0, "Treatment batch leaked gradient to treatment embedding row 0"

    # Active treatment branch checks
    treat_tower_grad_sum = sum(p.grad.norm().item() for p in model.treatment_tower.parameters() if p.grad is not None)
    treat_gate_grad_sum = sum(p.grad.norm().item() for p in model.treatment_gate.parameters() if p.grad is not None)
    treat_row1_grad = model.encoder.treatment_embeddings.weight.grad[1]
    assert treat_tower_grad_sum > 0.0, "Treatment tower received no gradients in treatment batch"
    assert treat_gate_grad_sum > 0.0, "Treatment gate received no gradients in treatment batch"
    assert treat_row1_grad.norm().item() > 0.0, "Treatment embedding row 1 received no gradients in treatment batch"

    # Shared new parameters participate (all 12 tensors including MLP)
    for i, p in enumerate(new_params):
        assert p.grad is not None and p.grad.norm().item() > 0.0, (
            f"Shared new parameter {i} did not participate in treatment batch gradient flow"
        )
    # Ensure MLP gets gradient on active treatment
    for name, p in model.valor_branch.mlp.named_parameters():
        assert p.grad is not None and p.grad.norm().item() > 0.0, (
            f"MLP parameter {name} did not receive gradients in treatment batch"
        )
    print("  ✅ All-treatment batch isolation verified:")
    print("     - Inactive control_tower, control_gate, treatment embedding row 0 grads == 0")
    print("     - Active treatment_tower, treatment_gate, treatment embedding row 1 grads > 0")
    print("     - Shared new VALOR & router parameters (including 4 MLP tensors) actively participate")
    return True


def test_trainer_epoch_eval_and_strict_save_load() -> bool:
    """Test 10: CPMTrainer tiny training epoch, evaluation, and strict checkpoint save/load roundtrip."""
    header("Test 10: CPMTrainer Tiny Training & Strict Save/Load Roundtrip")
    from CDUM.trainer import CPMTrainer
    from CDUM.variants.cpm_dynamic_fusion import CPMDynamicFusion

    torch.manual_seed(42)
    N = 128
    B = 32
    num_features = 12
    num_bins = 101

    # Fake balanced dataset
    x_ids = torch.randint(0, num_bins, (N, num_features))
    t = torch.tensor([0, 1] * (N // 2), dtype=torch.long)
    y = torch.randn(N)

    dataset = TensorDataset(x_ids, t, y)
    train_loader = DataLoader(dataset, batch_size=B, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=B)

    model = CPMDynamicFusion(num_features=num_features, num_bins=num_bins)
    trainer = CPMTrainer(model=model, lr=1e-3, device="cpu")

    # Train 2 epochs
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
        early_stopping_patience=5,
        verbose=0,
    )

    assert len(history["train_loss"]) == 2, f"Expected 2 train losses, got {len(history['train_loss'])}"
    assert len(history["val_loss"]) == 2, f"Expected 2 val losses, got {len(history['val_loss'])}"
    for val in history["train_loss"] + history["val_loss"]:
        assert torch.isfinite(torch.tensor(val)), f"Non-finite loss value in history: {val}"
    print(f"  ✅ Training completed: epoch 1 loss={history['train_loss'][0]:.4f}, epoch 2 loss={history['train_loss'][1]:.4f}")

    # Evaluate
    eval_res = trainer.evaluate(val_loader, print_diagnostics=False)
    assert "loss" in eval_res, "Evaluation result missing 'loss' key"
    assert torch.isfinite(torch.tensor(eval_res["loss"])), "Evaluation loss is not finite"
    print(f"  ✅ Evaluation completed: loss={eval_res['loss']:.4f}")

    # Strict save / load roundtrip
    os.makedirs("/tmp/opencode", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "cpm_dynamic_fusion.pth")
        trainer.save(ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not created"

        fresh_model = CPMDynamicFusion(num_features=num_features, num_bins=num_bins)
        fresh_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

        model.eval()
        fresh_model.eval()

        eval_x = x_ids[:16]
        eval_t = t[:16]

        with torch.no_grad():
            orig_out = model(eval_x, eval_t)
            loaded_out = fresh_model(eval_x, eval_t)

        for key in ["y_factual", "y0", "y1", "y0_hat", "y1_hat", "uplift"]:
            assert torch.allclose(orig_out[key], loaded_out[key], atol=1e-7), (
                f"Mismatch after strict load for output key '{key}'"
            )
        print("  ✅ Strict save/load roundtrip verified: loaded state matches exactly")

    return True


def main() -> None:
    print("+" + "=" * 63 + "+")
    print("|      CPM Dynamic Fusion Smoke Test Suite                      |")
    print("+" + "=" * 63 + "+")

    passed = 0
    failed = 0
    results: List[Tuple[str, str]] = []

    tests = [
        ("Imports", test_imports),
        ("Default Shapes & Validations", test_default_shapes_and_validations),
        ("Custom Dimensions & B=1", test_custom_dimensions_and_batch_one),
        ("Intermediates Algebra & Tower Hooks", test_intermediates_algebra_and_tower_hooks),
        ("Compute Loss Backprop & 12 New Parameter Grads", test_backprop_new_parameters_gradient_flow),
        ("Diagnostics Stats & Loss Invariance", test_diagnostics_statistics_and_loss_invariance),
        ("Baseline Preservation & Forced CPM", test_baseline_preservation_and_forced_cpm),
        ("Factual Masking & Counterfactual Independence", test_factual_masking_and_counterfactual_independence),
        ("Factual-Only Gradient Isolation", test_factual_only_gradient_isolation),
        ("Trainer Integration & Strict Save/Load", test_trainer_epoch_eval_and_strict_save_load),
    ]

    for name, test_fn in tests:
        try:
            res = test_fn()
            if res is False:
                failed += 1
                results.append((name, "FAIL"))
            else:
                passed += 1
                results.append((name, "PASS"))
        except Exception:
            failed += 1
            results.append((name, "FAIL"))
            traceback.print_exc()

    header("Summary")
    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    total = passed + failed
    print(f"\n  Result: {passed}/{total} passed", end="")
    if failed:
        print(f" ({failed} FAILED)")
        sys.exit(1)
    else:
        print(" -- All clear!")
        sys.exit(0)


if __name__ == "__main__":
    main()
