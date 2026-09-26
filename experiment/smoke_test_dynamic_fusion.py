#!/usr/bin/env python3
"""Integration smoke test suite for CPMDynamicFusion model selection and evaluation pipeline.

Validates model-selection integration across experiment/main.py and experiment/evaluate.py:
  1. Factory checks: cdum/cpm build exact CPM; cpm_dynamic_fusion builds CPMDynamicFusion
     with preserved base kwargs, default/custom router width, and existing Huber delta behavior.
  2. YAML and CLI configuration: model.name, cdum/cpm settings, router config, and explicit CLI
     overriding YAML (including parser default model).
  3. main() dispatch: canonical model names ('cdum' for cpm/cdum, variant otherwise) and args passed to run_model.
  4. evaluate.main() integration: CLI and YAML variant selection, architecture dims and router preservation,
     checkpoint resolution, baseline CLI override of variant YAML, and no model guesses from checkpoint paths.
  5. EquidistantBucketer stability: fitted train only for variant and baseline (exact same IDs),
     and eval sets with higher maxima do NOT alter denominators.
  6. evaluate_single_checkpoint save/load: strict state_dict round-trip with custom dims and
     balanced synthetic loader, asserting finite evaluation loss and metrics.
  7. Exception swallowing: evaluator error handling verified with mocks ensuring execution config
     is strictly verified and not relying on process exit code alone.

All temporary artifacts are maintained under /tmp/opencode.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from preprocess.data_loader import CriteoDataset
from preprocess.cpm_processor import EquidistantBucketer
from CDUM.cpm import CPM
from CDUM.variants import CPMDynamicFusion
from experiment.main import (
    build_parser,
    build_model,
    load_yaml_config,
    merge_config_into_args,
    prepare_loaders_for_model,
)
import experiment.main as main_module
import experiment.evaluate as evaluate_module
from experiment.evaluate import (
    resolve_checkpoint_path,
    evaluate_single_checkpoint,
)


TMP_OPENCODE_DIR = "/tmp/opencode"
os.makedirs(TMP_OPENCODE_DIR, exist_ok=True)


class TestModelFactoryIntegration(unittest.TestCase):
    """Test 1: build_model factory builds exact CPM for cdum/cpm and CPMDynamicFusion for variant."""

    def setUp(self):
        self.parser = build_parser()

    def test_build_model_cpm_and_cdum_build_cpm_exactly(self):
        """cdum and cpm must build exact CPM instance (never CPMDynamicFusion)."""
        for model_name in ("cdum", "cpm"):
            args = self.parser.parse_args([])
            args.device = "cpu"
            trainer = build_model(model_name, args)
            self.assertIsInstance(
                trainer.model,
                CPM,
                f"Model built for '{model_name}' must be an instance of CPM",
            )
            self.assertNotIsInstance(
                trainer.model,
                CPMDynamicFusion,
                f"Model built for '{model_name}' must NOT be an instance of CPMDynamicFusion",
            )
            self.assertIs(
                type(trainer.model),
                CPM,
                f"Model built for '{model_name}' must be exactly CPM",
            )

    def test_build_model_dynamic_fusion_preserved_kwargs_and_default_router(self):
        """CPMDynamicFusion preserves custom base kwargs and defaults router_hidden_dim to expert_dim D."""
        args = self.parser.parse_args([])
        args.device = "cpu"
        args.cpm_num_features = 8
        args.input_dim = 8
        args.cpm_embedding_dim = 16
        args.cpm_treatment_dim = 64
        args.cpm_refine_hidden_dim = 48
        args.cpm_refine_dim = 24
        args.cpm_tower_hidden_dim = 24
        args.cpm_num_experts = 4
        args.cpm_expert_hidden_dim = 96
        args.cpm_expert_dim = 32
        args.cpm_activation = "relu"
        args.cpm_dropout = 0.1
        args.cpm_batch_norm = False
        args.router_hidden_dim = None  # None -> should default to expert_dim (32)

        trainer = build_model("cpm_dynamic_fusion", args)
        model = trainer.model

        self.assertIsInstance(
            model,
            CPMDynamicFusion,
            "Factory must build CPMDynamicFusion when requested",
        )
        # Verify preserved base kwargs
        self.assertEqual(model.num_features, 8)
        self.assertEqual(model.embedding_dim, 16)
        self.assertEqual(model.treatment_dim, 64)
        self.assertEqual(model.refine_dim, 24)
        self.assertEqual(model.tower_hidden_dim, 24)
        self.assertEqual(model.num_experts, 4)
        self.assertEqual(model.expert_hidden_dim, 96)
        self.assertEqual(model.expert_dim, 32)

        # Verify default router width matches expert_dim D
        self.assertEqual(model.router_hidden_dim, 32)
        self.assertEqual(model.router.hidden_dim, 32)
        self.assertEqual(model.router.fc1.in_features, 2 * 32)
        self.assertEqual(model.router.fc1.out_features, 32)
        self.assertEqual(model.router.fc2.in_features, 32)
        self.assertEqual(model.router.fc2.out_features, 2)

    def test_build_model_dynamic_fusion_custom_router_width(self):
        """CPMDynamicFusion honors explicit custom router width."""
        args = self.parser.parse_args([])
        args.device = "cpu"
        args.cpm_expert_dim = 32
        args.cpm_refine_dim = 24
        args.cpm_tower_hidden_dim = 24
        args.router_hidden_dim = 50

        trainer = build_model("cpm_dynamic_fusion", args)
        model = trainer.model

        self.assertEqual(model.router_hidden_dim, 50)
        self.assertEqual(model.router.hidden_dim, 50)
        self.assertEqual(model.router.fc1.out_features, 50)
        self.assertEqual(model.router.fc2.in_features, 50)

    def test_build_model_huber_delta_behavior(self):
        """Criterion delta matches default (1.0) and custom cpm_huber_delta."""
        # Default delta
        args_default = self.parser.parse_args([])
        args_default.device = "cpu"
        trainer_default = build_model("cpm_dynamic_fusion", args_default)
        self.assertIsInstance(trainer_default.criterion, nn.HuberLoss)
        self.assertEqual(trainer_default.criterion.delta, 1.0)

        # Custom delta
        args_custom = self.parser.parse_args([])
        args_custom.device = "cpu"
        args_custom.cpm_huber_delta = 2.5
        trainer_custom = build_model("cpm_dynamic_fusion", args_custom)
        self.assertIsInstance(trainer_custom.criterion, nn.HuberLoss)
        self.assertEqual(trainer_custom.criterion.delta, 2.5)


class TestYamlAndCliConfigIntegration(unittest.TestCase):
    """Test 2: YAML model.name, cpm/cdum settings, router config, and explicit CLI overrides."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=TMP_OPENCODE_DIR)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_yaml_model_and_router_honored(self):
        """YAML config with model.name and router_hidden_dim is correctly loaded and merged."""
        yaml_content = """
model:
  name: "cpm_dynamic_fusion"
  router_hidden_dim: 80
cdum:
  expert_dim: 64
  refine_dim: 32
  tower_hidden_dim: 32
"""
        cfg_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        parser = build_parser()
        args = parser.parse_args([])
        merged_args = merge_config_into_args(args, cfg_path, raw_argv=[])

        self.assertEqual(merged_args.model, "cpm_dynamic_fusion")
        self.assertEqual(merged_args.router_hidden_dim, 80)
        self.assertEqual(merged_args.cpm_expert_dim, 64)

    def test_cli_explicit_default_model_overrides_yaml(self):
        """Explicit CLI --model cdum overrides YAML even though 'cdum' is the parser default."""
        yaml_content = """
model:
  name: "cpm_dynamic_fusion"
  router_hidden_dim: 80
"""
        cfg_path = os.path.join(self.temp_dir.name, "variant_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        raw_argv = ["--config", cfg_path, "--model", "cdum"]
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        merged_args = merge_config_into_args(args, cfg_path, raw_argv=raw_argv)

        self.assertEqual(
            merged_args.model,
            "cdum",
            "Explicit CLI --model cdum must override variant YAML model.name",
        )

    def test_cli_explicit_override_yaml_parameters(self):
        """CLI arguments override corresponding YAML options."""
        yaml_content = """
model:
  name: "cdum"
  router_hidden_dim: 64
training:
  lr: 0.001
"""
        cfg_path = os.path.join(self.temp_dir.name, "baseline_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        raw_argv = [
            "--config", cfg_path,
            "--model", "cpm_dynamic_fusion",
            "--router_hidden_dim", "128",
            "--lr", "0.005",
        ]
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        merged_args = merge_config_into_args(args, cfg_path, raw_argv=raw_argv)

        self.assertEqual(merged_args.model, "cpm_dynamic_fusion")
        self.assertEqual(merged_args.router_hidden_dim, 128)
        self.assertEqual(merged_args.lr, 0.005)

    def test_yaml_cpm_dynamic_fusion_section_honored(self):
        """cpm_dynamic_fusion section settings in YAML are honored."""
        yaml_content = """
cpm_dynamic_fusion:
  router_hidden_dim: 96
  expert_dim: 128
  refine_dim: 32
  tower_hidden_dim: 32
"""
        cfg_path = os.path.join(self.temp_dir.name, "section_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        flat_cfg = load_yaml_config(cfg_path)
        self.assertEqual(flat_cfg.get("router_hidden_dim"), 96)
        self.assertEqual(flat_cfg.get("cpm_expert_dim"), 128)


class TestMainDispatchIntegration(unittest.TestCase):
    """Test 3: main() dispatches canonical name 'cdum' for cpm/cdum and variant otherwise."""

    @mock.patch("experiment.main.get_dataloaders")
    @mock.patch("experiment.main.run_model")
    def test_main_canonical_cpm_to_cdum(self, mock_run_model, mock_get_dataloaders):
        """CLI --model cpm passes canonical model 'cdum' to run_model."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        test_argv = ["main.py", "--model", "cpm", "--epochs", "1"]

        with mock.patch.object(sys, "argv", test_argv):
            main_module.main()

        mock_run_model.assert_called_once()
        passed_model_name = mock_run_model.call_args[0][0]
        passed_args = mock_run_model.call_args[0][1]

        self.assertEqual(passed_model_name, "cdum")
        self.assertEqual(passed_args.model, "cdum")

    @mock.patch("experiment.main.get_dataloaders")
    @mock.patch("experiment.main.run_model")
    def test_main_canonical_cdum_remains_cdum(self, mock_run_model, mock_get_dataloaders):
        """CLI --model cdum passes canonical model 'cdum' to run_model."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        test_argv = ["main.py", "--model", "cdum", "--epochs", "1"]

        with mock.patch.object(sys, "argv", test_argv):
            main_module.main()

        mock_run_model.assert_called_once()
        passed_model_name = mock_run_model.call_args[0][0]
        passed_args = mock_run_model.call_args[0][1]

        self.assertEqual(passed_model_name, "cdum")
        self.assertEqual(passed_args.model, "cdum")

    @mock.patch("experiment.main.get_dataloaders")
    @mock.patch("experiment.main.run_model")
    def test_main_variant_dispatch_and_args_passed(self, mock_run_model, mock_get_dataloaders):
        """CLI --model cpm_dynamic_fusion passes variant name and args to run_model."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        test_argv = [
            "main.py",
            "--model", "cpm_dynamic_fusion",
            "--router_hidden_dim", "44",
            "--epochs", "2",
        ]

        with mock.patch.object(sys, "argv", test_argv):
            main_module.main()

        mock_run_model.assert_called_once()
        passed_model_name = mock_run_model.call_args[0][0]
        passed_args = mock_run_model.call_args[0][1]

        self.assertEqual(passed_model_name, "cpm_dynamic_fusion")
        self.assertEqual(passed_args.model, "cpm_dynamic_fusion")
        self.assertEqual(passed_args.router_hidden_dim, 44)
        self.assertEqual(passed_args.epochs, 2)


class TestEvaluateMainIntegration(unittest.TestCase):
    """Test 4: evaluate.main() variant selection, router config, checkpoint resolution, and mock verification."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=TMP_OPENCODE_DIR)

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("experiment.evaluate.get_dataloaders")
    @mock.patch("experiment.evaluate.prepare_loaders_for_model")
    @mock.patch("experiment.evaluate.resolve_checkpoint_path")
    @mock.patch("experiment.evaluate.evaluate_single_checkpoint")
    def test_evaluate_cli_variant_selection(
        self,
        mock_eval_single,
        mock_resolve,
        mock_prepare,
        mock_get_dataloaders,
    ):
        """evaluate.main() with CLI variant selection passes model and router dim to evaluate_single_checkpoint."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        mock_prepare.return_value = (None, None, mock.MagicMock())
        fake_ckpt = os.path.join(self.temp_dir.name, "fake_ckpt.pth")
        mock_resolve.return_value = fake_ckpt
        mock_eval_single.return_value = {"loss": 0.2, "auuc": 0.3, "qini": 0.2, "lift@30%": 0.1}

        summary_csv = os.path.join(self.temp_dir.name, "summary.csv")
        test_argv = [
            "evaluate.py",
            "--model", "cpm_dynamic_fusion",
            "--router_hidden_dim", "56",
            "--seed", "1",
            "--save_csv", summary_csv,
        ]

        with mock.patch.object(sys, "argv", test_argv):
            evaluate_module.main()

        mock_eval_single.assert_called_once()
        kwargs = mock_eval_single.call_args.kwargs
        self.assertEqual(kwargs.get("model_name"), "cpm_dynamic_fusion")
        self.assertEqual(kwargs.get("ckpt_path"), fake_ckpt)
        self.assertEqual(kwargs.get("seed_label"), "1")
        self.assertEqual(kwargs.get("args").router_hidden_dim, 56)
        self.assertEqual(kwargs.get("args").model, "cpm_dynamic_fusion")
        mock_prepare.assert_called_once()
        self.assertEqual(mock_prepare.call_args[0][0], "cpm_dynamic_fusion")
        mock_resolve.assert_called_once()
        self.assertEqual(mock_resolve.call_args[0][1], "cpm_dynamic_fusion")

    @mock.patch("experiment.evaluate.get_dataloaders")
    @mock.patch("experiment.evaluate.prepare_loaders_for_model")
    @mock.patch("experiment.evaluate.resolve_checkpoint_path")
    @mock.patch("experiment.evaluate.evaluate_single_checkpoint")
    def test_evaluate_yaml_variant_selection(
        self,
        mock_eval_single,
        mock_resolve,
        mock_prepare,
        mock_get_dataloaders,
    ):
        """evaluate.main() with YAML config passes variant model and router dim."""
        yaml_content = """
model:
  name: "cpm_dynamic_fusion"
  router_hidden_dim: 68
cdum:
  expert_dim: 64
"""
        cfg_path = os.path.join(self.temp_dir.name, "variant_eval.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        mock_prepare.return_value = (None, None, mock.MagicMock())
        fake_ckpt = os.path.join(self.temp_dir.name, "variant_ep20.pth")
        mock_resolve.return_value = fake_ckpt
        mock_eval_single.return_value = {"loss": 0.15, "auuc": 0.25, "qini": 0.15, "lift@30%": 0.05}

        summary_csv = os.path.join(self.temp_dir.name, "summary.csv")
        test_argv = [
            "evaluate.py",
            "--config", cfg_path,
            "--seed", "2",
            "--save_csv", summary_csv,
        ]

        with mock.patch.object(sys, "argv", test_argv):
            evaluate_module.main()

        mock_eval_single.assert_called_once()
        kwargs = mock_eval_single.call_args.kwargs
        self.assertEqual(kwargs.get("model_name"), "cpm_dynamic_fusion")
        self.assertEqual(kwargs.get("ckpt_path"), fake_ckpt)
        self.assertEqual(kwargs.get("args").router_hidden_dim, 68)
        self.assertEqual(kwargs.get("args").cpm_expert_dim, 64)
        mock_prepare.assert_called_once()
        self.assertEqual(mock_prepare.call_args[0][0], "cpm_dynamic_fusion")
        mock_resolve.assert_called_once()
        self.assertEqual(mock_resolve.call_args[0][1], "cpm_dynamic_fusion")

    @mock.patch("experiment.evaluate.get_dataloaders")
    @mock.patch("experiment.evaluate.prepare_loaders_for_model")
    @mock.patch("experiment.evaluate.resolve_checkpoint_path")
    @mock.patch("experiment.evaluate.evaluate_single_checkpoint")
    def test_evaluate_explicit_baseline_cli_overrides_variant_yaml(
        self,
        mock_eval_single,
        mock_resolve,
        mock_prepare,
        mock_get_dataloaders,
    ):
        """Explicit CLI --model cdum overrides variant YAML in evaluate.main()."""
        yaml_content = """
model:
  name: "cpm_dynamic_fusion"
  router_hidden_dim: 70
"""
        cfg_path = os.path.join(self.temp_dir.name, "variant_cfg.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        mock_prepare.return_value = (None, None, mock.MagicMock())
        fake_ckpt = os.path.join(self.temp_dir.name, "cdum_best.pth")
        mock_resolve.return_value = fake_ckpt
        mock_eval_single.return_value = {"loss": 0.18, "auuc": 0.22, "qini": 0.18, "lift@30%": 0.08}

        summary_csv = os.path.join(self.temp_dir.name, "summary.csv")
        test_argv = [
            "evaluate.py",
            "--config", cfg_path,
            "--model", "cdum",
            "--seed", "1",
            "--save_csv", summary_csv,
        ]

        with mock.patch.object(sys, "argv", test_argv):
            evaluate_module.main()

        mock_eval_single.assert_called_once()
        kwargs = mock_eval_single.call_args.kwargs
        self.assertEqual(kwargs.get("model_name"), "cdum")
        self.assertEqual(kwargs.get("args").model, "cdum")

    @mock.patch("experiment.evaluate.get_dataloaders")
    @mock.patch("experiment.evaluate.prepare_loaders_for_model")
    @mock.patch("experiment.evaluate.evaluate_single_checkpoint")
    def test_evaluate_no_model_guesses_from_checkpoint_path(
        self,
        mock_eval_single,
        mock_prepare,
        mock_get_dataloaders,
    ):
        """evaluate.main() never infers model from checkpoint path or nearby config.json when absent --model."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        mock_prepare.return_value = (None, None, mock.MagicMock())
        mock_eval_single.return_value = {"loss": 0.15, "auuc": 0.25, "qini": 0.15, "lift@30%": 0.05}

        # Baseline YAML with model.name=cdum and router width
        yaml_content = """
model:
  name: "cdum"
  router_hidden_dim: 32
"""
        cfg_path = os.path.join(self.temp_dir.name, "baseline_config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        # Checkpoint path containing 'cpm_dynamic_fusion' and nearby config.json with variant settings
        ckpt_dir = os.path.join(self.temp_dir.name, "checkpoints", "cpm_dynamic_fusion_seed_1")
        os.makedirs(ckpt_dir, exist_ok=True)
        custom_ckpt = os.path.join(ckpt_dir, "best_auuc_cpm_dynamic_fusion.pth")
        with open(custom_ckpt, "w", encoding="utf-8") as f:
            f.write("fake checkpoint")

        config_json_path = os.path.join(ckpt_dir, "config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "cpm_dynamic_fusion",
                "router_hidden_dim": 128,
            }, f)

        # Provide --config and --checkpoint, but NO --model
        summary_csv = os.path.join(self.temp_dir.name, "summary.csv")
        test_argv = [
            "evaluate.py",
            "--config", cfg_path,
            "--checkpoint", custom_ckpt,
            "--save_csv", summary_csv,
        ]

        with mock.patch.object(sys, "argv", test_argv):
            evaluate_module.main()

        mock_eval_single.assert_called_once()
        kwargs = mock_eval_single.call_args.kwargs
        self.assertEqual(
            kwargs.get("model_name"),
            "cdum",
            "Evaluated model must remain cdum without guessing variant from checkpoint path",
        )
        self.assertEqual(kwargs.get("args").model, "cdum")
        self.assertEqual(
            kwargs.get("args").router_hidden_dim,
            32,
            "Config router width must not be replaced by nearby config.json",
        )
        self.assertNotEqual(kwargs.get("args").router_hidden_dim, 128)
        self.assertEqual(kwargs.get("ckpt_path"), custom_ckpt)
        mock_prepare.assert_called_once()
        self.assertEqual(mock_prepare.call_args[0][0], "cdum")

    @mock.patch("experiment.evaluate.get_dataloaders")
    @mock.patch("experiment.evaluate.prepare_loaders_for_model")
    @mock.patch("experiment.evaluate.resolve_checkpoint_path")
    @mock.patch("experiment.evaluate.evaluate_single_checkpoint")
    def test_evaluate_swallows_exception_detected_by_mock(
        self,
        mock_eval_single,
        mock_resolve,
        mock_prepare,
        mock_get_dataloaders,
    ):
        """evaluate.main() swallows exceptions in single checkpoint eval; mock asserts expected call config."""
        mock_get_dataloaders.return_value = (mock.MagicMock(), None, mock.MagicMock())
        mock_prepare.return_value = (None, None, mock.MagicMock())
        fake_ckpt = os.path.join(self.temp_dir.name, "corrupted_ckpt.pth")
        mock_resolve.return_value = fake_ckpt
        mock_eval_single.side_effect = RuntimeError("Simulated checkpoint load crash")

        summary_csv = os.path.join(self.temp_dir.name, "summary.csv")
        test_argv = [
            "evaluate.py",
            "--model", "cpm_dynamic_fusion",
            "--seed", "1",
            "--save_csv", summary_csv,
        ]

        # evaluate.main() does not raise because of internal try-except block
        with mock.patch.object(sys, "argv", test_argv):
            evaluate_module.main()

        # The mock verifies that the attempt occurred with the expected configuration
        mock_eval_single.assert_called_once()
        kwargs = mock_eval_single.call_args.kwargs
        self.assertEqual(kwargs.get("model_name"), "cpm_dynamic_fusion")
        self.assertEqual(kwargs.get("ckpt_path"), fake_ckpt)

    def test_resolve_checkpoint_path_resolution(self):
        """resolve_checkpoint_path finds variant checkpoint under results_dir/cpm_dynamic_fusion/seed_1/."""
        seed_dir = os.path.join(self.temp_dir.name, "cpm_dynamic_fusion", "seed_1")
        os.makedirs(seed_dir, exist_ok=True)
        expected_file = os.path.join(seed_dir, "best_auuc_checkpoint.pth")
        with open(expected_file, "w") as f:
            f.write("fake checkpoint")

        resolved = resolve_checkpoint_path(self.temp_dir.name, "cpm_dynamic_fusion", 1, "best_auuc")
        self.assertEqual(resolved, expected_file)

        # Non-existent seed should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_path(self.temp_dir.name, "cpm_dynamic_fusion", 999, "best_auuc")


class TestBucketerFittingAndStability(unittest.TestCase):
    """Test 5: EquidistantBucketer fitted train only for variant/baseline; eval higher maxima do not alter denoms."""

    def test_bucketer_train_only_and_higher_maxima_stability(self):
        """Bucketer fits train only; variant and baseline get identical IDs; eval higher maxima do not change denoms."""
        parser = build_parser()
        args = parser.parse_args([])
        args.cpm_num_bins = 101

        # Continuous train features: range [0.0, 50.0]
        torch.manual_seed(42)
        N_train = 100
        D = 12
        X_train = torch.rand(N_train, D) * 50.0
        t_train = torch.randint(0, 2, (N_train,))
        y_train = torch.randint(0, 2, (N_train,)).float()
        ds_train = CriteoDataset(X_train, t_train, y_train)
        train_loader = DataLoader(ds_train, batch_size=25, shuffle=False)

        # Continuous eval features with higher maxima (up to 250.0, strictly larger than train max)
        N_eval = 50
        X_eval = X_train[:N_eval].clone() * 5.0 + 10.0
        ds_eval = CriteoDataset(X_eval, t_train[:N_eval], y_train[:N_eval])
        eval_loader = DataLoader(ds_eval, batch_size=25, shuffle=False)

        # Prepare loaders for variant and baseline
        tr_var, _, te_var = prepare_loaders_for_model(
            "cpm_dynamic_fusion", args, train_loader, None, eval_loader
        )
        tr_base, _, te_base = prepare_loaders_for_model(
            "cdum", args, train_loader, None, eval_loader
        )

        # 1. Variant and baseline must yield identical bucket IDs on train
        b_var = next(iter(tr_var))[0]
        b_base = next(iter(tr_base))[0]
        self.assertTrue(
            torch.equal(b_var, b_base),
            "Variant and baseline must produce identical bucket IDs on train split",
        )

        # 2. Check fitted denominators match train split maximums
        initial_denoms = tr_var.bucketer.denominators.clone()
        expected_denoms = torch.clamp(X_train.max(dim=0).values, min=1e-8)
        self.assertTrue(
            torch.allclose(initial_denoms, expected_denoms),
            "Denominators must be computed exclusively from train split continuous features",
        )

        # 3. Iterate through eval loader with higher maxima
        for batch in te_var:
            x_eval_b = batch[0]
            self.assertTrue(
                (x_eval_b >= 0).all() and (x_eval_b <= 100).all(),
                "Eval bucket IDs must be within [0, 100] even for higher maxima",
            )
            # Features exceeding train max should clamp to bin 100
            self.assertTrue(
                (x_eval_b == 100).any(),
                "Higher maxima in eval split should map to clamped boundary bin 100",
            )

        # 4. Denominators MUST NOT change after eval transformation
        self.assertTrue(
            torch.equal(tr_var.bucketer.denominators, initial_denoms),
            "Eval features with higher maxima must not modify fitted bucketer denominators",
        )


class TestEvaluateSingleCheckpointSaveLoad(unittest.TestCase):
    """Test 6: evaluate_single_checkpoint save/load variant via factory strict state_dict with custom dims."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=TMP_OPENCODE_DIR)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_strict_state_dict_custom_dims_balanced_loader(self):
        """evaluate_single_checkpoint strictly loads checkpoint with custom dims and computes finite loss."""
        parser = build_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.eval_k = 0.3
        # Custom architecture dimensions
        args.input_dim = 6
        args.cpm_num_features = 6
        args.cpm_embedding_dim = 16
        args.cpm_treatment_dim = 64
        args.cpm_refine_hidden_dim = 32
        args.cpm_refine_dim = 16
        args.cpm_tower_hidden_dim = 16
        args.cpm_num_experts = 2
        args.cpm_expert_hidden_dim = 32
        args.cpm_expert_dim = 20
        args.router_hidden_dim = 24
        args.cpm_num_bins = 101

        # Build variant and save checkpoint
        trainer = build_model("cpm_dynamic_fusion", args)
        ckpt_path = os.path.join(self.temp_dir.name, "variant_custom_dims.pth")
        trainer.save(ckpt_path)

        # Balanced synthetic DataLoader (40 samples: 20 control, 20 treated, balanced outcomes)
        N = 40
        torch.manual_seed(123)
        x_discrete = torch.randint(0, 101, (N, 6), dtype=torch.long)
        t_balanced = torch.tensor([0] * 20 + [1] * 20, dtype=torch.long)
        y_balanced = torch.tensor([0.0, 1.0] * 20, dtype=torch.float32)
        dataset = TensorDataset(x_discrete, t_balanced, y_balanced)
        test_loader = DataLoader(dataset, batch_size=20, shuffle=False)

        # Run evaluate_single_checkpoint
        metrics = evaluate_single_checkpoint(
            model_name="cpm_dynamic_fusion",
            ckpt_path=ckpt_path,
            seed_label="custom_test",
            test_loader=test_loader,
            args=args,
        )

        self.assertIn("loss", metrics)
        self.assertTrue(
            np.isfinite(metrics["loss"]),
            f"Loss must be finite, got {metrics['loss']}",
        )
        self.assertGreaterEqual(
            metrics["loss"],
            0.0,
            f"Loss must be non-negative, got {metrics['loss']}",
        )
        self.assertIn("auuc", metrics)
        self.assertIn("qini", metrics)

    def test_strict_state_dict_fails_on_architecture_mismatch(self):
        """Strict state_dict load raises RuntimeError if checkpoint is missing router/variant keys."""
        parser = build_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.input_dim = 6
        args.cpm_num_features = 6
        args.cpm_embedding_dim = 16
        args.cpm_refine_dim = 16
        args.cpm_tower_hidden_dim = 16
        args.cpm_expert_dim = 20

        # Build baseline CPM and save checkpoint (lacks router and valor_branch weights)
        cpm_trainer = build_model("cdum", args)
        cpm_ckpt = os.path.join(self.temp_dir.name, "baseline_cpm.pth")
        cpm_trainer.save(cpm_ckpt)

        # Attempting to load baseline checkpoint into variant trainer must fail under strict=True
        variant_trainer = build_model("cpm_dynamic_fusion", args)
        with self.assertRaises(RuntimeError):
            variant_trainer.load(cpm_ckpt)


def run_all_tests():
    """Run full test suite with verbose reporting."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestModelFactoryIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestYamlAndCliConfigIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMainDispatchIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluateMainIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBucketerFittingAndStability))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluateSingleCheckpointSaveLoad))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
