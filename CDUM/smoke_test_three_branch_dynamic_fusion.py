#!/usr/bin/env python3
"""Semantic smoke tests for three-branch CPM (CPU, synthetic inputs only)."""

import inspect
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from unittest import mock

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CDUM.cpm import CPM
from CDUM.experts import Expert
from CDUM.trainer import CPMTrainer
from CDUM.variants import CPMDynamicFusion, CPMThreeBranchDynamicFusion, PrognosticBranch


class TestThreeBranchArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CPMThreeBranchDynamicFusion()
        self.x = torch.randint(0, 101, (16, 12))
        self.t = torch.tensor([0, 1] * 8)
        self.y = torch.tensor([0., 0., 1., 1.] * 4)

    def assert_algebra(self, model, x, t):
        out = model(x, t, return_diagnostics=True)
        for k, d in out["diagnostics"].items():
            B, D = x.shape[0], model.expert_dim
            shapes = {
                "e_x": (B, model.num_features * model.embedding_dim),
                "e_t": (B, model.treatment_dim),
                "e_guidance": (B, model.refine_dim), "e_indicator": (B, model.refine_dim),
                "expert_outputs": (B, model.num_experts, D), "a_C": (B, model.num_experts),
                "q_R": (B, 3 * D), "router_logits": (B, 3), "pi": (B, 3), "y_hat": (B, 1),
            }
            shapes.update({key: (B, D) for key in ("z_P", "z_C", "z_I", "h_x", "m_t", "interaction", "z_F")})
            for key, shape in shapes.items():
                self.assertEqual(tuple(d[key].shape), shape, key)
                self.assertTrue(torch.isfinite(d[key]).all(), key)
            torch.testing.assert_close(d["pi"].sum(-1), torch.ones(B))
            torch.testing.assert_close(d["a_C"].sum(-1), torch.ones(B))
            for key in ("pi", "m_t"):
                self.assertTrue(((d[key] >= 0) & (d[key] <= 1)).all())
            self.assertEqual(d["treatment_id"], k)
            raw_et = model.encoder.encode_treatment(torch.full((B,), k))
            torch.testing.assert_close(d["e_t"], raw_et)
            gui, ind = model.treatment_refine(raw_et)
            torch.testing.assert_close(d["e_guidance"], gui)
            torch.testing.assert_close(d["e_indicator"], ind)
            gate = model.control_gate if k == 0 else model.treatment_gate
            torch.testing.assert_close(d["a_C"], gate(gui))
            torch.testing.assert_close(d["z_C"], (d["expert_outputs"] * d["a_C"].unsqueeze(-1)).sum(1))
            torch.testing.assert_close(d["z_P"], model.prognostic_branch.mlp(d["e_x"]))
            torch.testing.assert_close(d["h_x"], model.valor_branch.linear_x(d["e_x"]))
            torch.testing.assert_close(d["m_t"], torch.sigmoid(model.valor_branch.linear_t(raw_et)))
            torch.testing.assert_close(d["interaction"], d["h_x"] * d["m_t"])
            torch.testing.assert_close(d["z_I"], model.valor_branch.mlp(d["interaction"]))
            torch.testing.assert_close(d["q_R"], torch.cat([d["z_P"], d["z_C"], d["z_I"]], -1))
            torch.testing.assert_close(d["router_logits"], model.router.fc2(torch.relu(model.router.fc1(d["q_R"]))))
            torch.testing.assert_close(d["pi"], torch.softmax(d["router_logits"], -1))
            expected = d["pi"][:, :1] * d["z_P"] + d["pi"][:, 1:2] * d["z_C"] + d["pi"][:, 2:3] * d["z_I"]
            torch.testing.assert_close(d["z_F"], expected, rtol=0, atol=0)
            tower = model.control_tower if k == 0 else model.treatment_tower
            # Verify the indicator is applied *inside* the existing tower.
            expected_y = tower.softplus(tower.layer2(tower.relu(tower.layer1(expected)) * ind))
            torch.testing.assert_close(out[f"y{k}_hat"], expected_y)
            self.assertIs(out[f"y{k}"], out[f"y{k}_hat"])
        self.assertIs(out["diagnostics"][0]["z_P"], out["diagnostics"][1]["z_P"])
        torch.testing.assert_close(out["uplift"], out["y1_hat"] - out["y0_hat"])
        return out

    def test_default_shapes_and_exact_algebra(self):
        self.assert_algebra(self.model, self.x, self.t)
        self.assertIsInstance(self.model.prognostic_branch.mlp, Expert)
        self.assertIsInstance(self.model.prognostic_branch.mlp.relu, nn.ReLU)
        self.assertEqual(self.model.prognostic_hidden_dim, self.model.expert_hidden_dim)
        self.assertEqual(self.model.router_hidden_dim, self.model.expert_dim)

    def test_custom_dimensions_and_batch_one(self):
        model = CPMThreeBranchDynamicFusion(
            num_features=7, num_bins=31, embedding_dim=16, treatment_dim=60,
            refine_dim=28, tower_hidden_dim=28, num_experts=4,
            expert_dim=44, expert_hidden_dim=90, prognostic_hidden_dim=75,
            router_hidden_dim=55, valor_hidden_dim=85,
        )
        self.assertEqual(model.router.fc1.in_features, 132)
        self.assertEqual(model.router.fc1.out_features, 55)
        self.assertEqual(model.prognostic_branch.mlp.hidden.out_features, 75)
        self.assertEqual(model.valor_branch.mlp.hidden.out_features, 85)
        self.assert_algebra(model, torch.randint(0, 31, (1, 7)), torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "must match refine_dim"):
            CPMThreeBranchDynamicFusion(refine_dim=28, tower_hidden_dim=32)

    def test_shared_call_counts_and_candidate_hooks(self):
        calls = defaultdict(list)
        handles = []
        names = ("prognostic_branch", "user_experts", "valor_branch", "router",
                 "control_gate", "treatment_gate", "control_tower", "treatment_tower")
        for name in names:
            def record(module, args, output, name=name):
                calls[name].append(args)
            handles.append(getattr(self.model, name).register_forward_hook(record))
        try:
            with mock.patch.object(self.model.encoder, "encode_features", wraps=self.model.encoder.encode_features) as encode:
                out = self.model(self.x, self.t, return_diagnostics=True)
                self.assertEqual(encode.call_count, 1)
        finally:
            for h in handles:
                h.remove()
        for name in names:
            self.assertEqual(len(calls[name]), 2 if name in ("valor_branch", "router") else 1, name)
        for k, prefix in ((0, "control"), (1, "treatment")):
            d = out["diagnostics"][k]
            self.assertIs(calls[prefix + "_gate"][0][0], d["e_guidance"])
            self.assertIs(calls["valor_branch"][k][1], d["e_t"])
            self.assertIs(calls[prefix + "_tower"][0][0], d["z_F"])
            self.assertIs(calls[prefix + "_tower"][0][1], d["e_indicator"])
            for i, key in enumerate(("z_P", "z_C", "z_I")):
                self.assertIs(calls["router"][k][i], d[key])
        self.assertEqual(len(calls["prognostic_branch"][0]), 1)
        self.assertIs(calls["prognostic_branch"][0][0], out["diagnostics"][0]["e_x"])

    def test_prognostic_has_no_treatment_dependency(self):
        self.assertEqual(list(inspect.signature(PrognosticBranch.forward).parameters), ["self", "e_x"])
        self.assertFalse(any(isinstance(m, nn.Embedding) for m in self.model.prognostic_branch.modules()))
        out = self.model(self.x, self.t, return_diagnostics=True)
        z_P = out["diagnostics"][0]["z_P"]
        grad = torch.autograd.grad(z_P.sum(), self.model.encoder.treatment_embeddings.weight, allow_unused=True)[0]
        self.assertIsNone(grad)
        with torch.no_grad():
            self.model.encoder.treatment_embeddings.weight.add_(5)
        changed = self.model(self.x, 1 - self.t, return_diagnostics=True)
        torch.testing.assert_close(z_P, changed["diagnostics"][1]["z_P"], rtol=0, atol=0)

    def test_counterfactual_independence_and_factual_selection(self):
        reference = self.model(self.x, self.t)
        for t in (self.t, torch.zeros_like(self.t), torch.ones_like(self.t), self.t[:, None]):
            out = self.model(self.x, t)
            for key in ("y0", "y1", "uplift"):
                torch.testing.assert_close(out[key], reference[key], rtol=0, atol=0)
            expected = torch.where(t.reshape(-1, 1).bool(), out["y1_hat"], out["y0_hat"])
            torch.testing.assert_close(out["y_factual"], expected, rtol=0, atol=0)

    def test_diagnostics_and_factual_loss_only(self):
        normal = self.model(self.x, self.t)
        diag = self.model(self.x, self.t, return_diagnostics=True)
        for key in normal:
            torch.testing.assert_close(normal[key], diag[key], rtol=0, atol=0)
        trainer = CPMTrainer(self.model, device="cpu")
        expected = nn.HuberLoss()(normal["y_factual"], self.y[:, None])
        # Poison evaluation-only outputs: loss must still use y_factual alone.
        def poison(module, args, out):
            return {**out, **{key: torch.full_like(out[key], float("nan")) for key in ("y0", "y1", "y0_hat", "y1_hat", "uplift")}}
        handle = self.model.register_forward_hook(poison)
        try:
            torch.testing.assert_close(trainer.compute_loss(self.x, self.t, self.y), expected)
        finally:
            handle.remove()

    def test_new_parameters_receive_factual_gradients(self):
        trainer = CPMTrainer(self.model, device="cpu")
        trainer.compute_loss(self.x, self.t, self.y).backward()
        new = {n: p for n, p in self.model.named_parameters() if n.startswith(("prognostic_branch.", "valor_branch.", "router."))}
        self.assertEqual(len(new), 16)
        for name, p in new.items():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.norm().item(), 0, name)

    def test_factual_gradient_isolation(self):
        trainer = CPMTrainer(self.model, device="cpu")
        for active in (0, 1):
            self.model.zero_grad(set_to_none=True)
            trainer.compute_loss(self.x, torch.full_like(self.t, active), self.y).backward()
            inactive_prefix = "treatment" if active == 0 else "control"
            for suffix in ("_tower", "_gate"):
                for p in getattr(self.model, inactive_prefix + suffix).parameters():
                    self.assertTrue(p.grad is None or p.grad.count_nonzero() == 0)
            emb_grad = self.model.encoder.treatment_embeddings.weight.grad
            self.assertEqual(emb_grad[1 - active].count_nonzero().item(), 0)
            self.assertGreater(emb_grad[active].norm().item(), 0)
            active_prefix = "control" if active == 0 else "treatment"
            for name in (active_prefix + "_gate", active_prefix + "_tower", "user_experts", "prognostic_branch", "valor_branch", "router"):
                norm = sum(p.grad.norm().item() for p in getattr(self.model, name).parameters() if p.grad is not None)
                self.assertGreater(norm, 0, name)

    def test_forced_router_all_three_paths_and_baseline(self):
        base = CPM()
        result = self.model.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(len(result.missing_keys), 16)
        self.assertTrue(all(n.startswith(("prognostic_branch.", "valor_branch.", "router.")) for n in result.missing_keys))
        for i, key in enumerate(("z_P", "z_C", "z_I")):
            with torch.no_grad():
                self.model.router.fc2.weight.zero_()
                self.model.router.fc2.bias.fill_(-1e9)
                self.model.router.fc2.bias[i] = 0
            out = self.model(self.x, self.t, return_diagnostics=True)
            for d in out["diagnostics"].values():
                torch.testing.assert_close(d["z_F"], d[key], rtol=0, atol=0)
            if i == 1:
                for name, value in base(self.x, self.t).items():
                    torch.testing.assert_close(out[name], value, rtol=0, atol=0)
        self.assertEqual(CPMDynamicFusion().router.fc2.out_features, 2)

    def test_trainer_fit_and_strict_roundtrip(self):
        # Matched covariates across treatments guarantee both groups are present
        # in the top-4 (30%) subset, where lift would otherwise be undefined.
        paired_x = self.x[:8].repeat_interleave(2, dim=0)
        loader = DataLoader(TensorDataset(paired_x, self.t, self.y), batch_size=8)
        trainer = CPMTrainer(self.model, device="cpu")
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            history = trainer.fit(loader, loader, epochs=2, checkpoint_dir=tmp, monitor="val_loss", verbose=0)
            self.assertEqual(len(history["train_loss"]), 2)
            self.assertTrue(all(torch.isfinite(torch.tensor(history[k])).all() for k in ("train_loss", "val_loss")))
            with self.assertLogs("CDUM.trainer", level="INFO") as logs:
                metrics = trainer.evaluate(loader)
            self.assertIn("Router [P,C,I]", "\n".join(logs.output))
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            restored = CPMTrainer(CPMThreeBranchDynamicFusion(), device="cpu")
            restored.load(os.path.join(tmp, "cpm_best.pth"))
            for key, value in trainer.model(self.x, self.t).items():
                torch.testing.assert_close(restored.model(self.x, self.t)[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
