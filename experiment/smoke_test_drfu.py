#!/usr/bin/env python3
"""DRFU factory, configuration, dispatch, bucketing and checkpoint tests."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from CDUM.cpm import CPM
from CDUM.variants import TwoBranchDynamicFusion, DRFU
from CDUM.inspect_checkpoint import detect_and_load_model, inspect_checkpoint, print_metrics
from preprocess.data_loader import CriteoDataset
from experiment import main as main_module, evaluate as evaluate_module
from experiment.main import build_parser, build_model, merge_config_into_args, prepare_loaders_for_model, BucketedDataLoader


MODEL = "drfu"


class TestDRFUIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = build_parser().parse_args(["--model", MODEL, "--device", "cpu"])

    def custom_args(self):
        args = self.args
        args.input_dim = 6
        args.cpm_num_features = None  # saved parser default must fall back to input_dim
        args.cpm_embedding_dim = 8
        args.cpm_treatment_dim = 30
        args.cpm_refine_hidden_dim = 18
        args.cpm_refine_dim = args.cpm_tower_hidden_dim = 14
        args.cpm_num_experts = 2
        args.cpm_expert_hidden_dim = 28
        args.cpm_expert_dim = 20
        args.router_hidden_dim = 17
        args.interaction_hidden_dim = 23
        args.prognostic_hidden_dim = 31
        args.results_dir = str(self.root)
        return args

    def fixture(self):
        args = self.custom_args()
        trainer = build_model(MODEL, args)
        ckpt = self.root / MODEL / "seed_1" / "best_auuc_checkpoint.pth"
        trainer.save(str(ckpt))
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps(vars(args)))
        x = torch.randint(0, 101, (40, 6))
        t = torch.tensor([0, 1] * 20)
        y = torch.tensor([0., 0., 1., 1.] * 10)
        loader = DataLoader(TensorDataset(x, t, y), batch_size=10)
        return trainer, ckpt, cfg, loader

    def test_factory_defaults_and_baseline_selection(self):
        self.args.cpm_expert_hidden_dim = 83
        self.args.cpm_expert_dim = 37
        trainer = build_model(MODEL, self.args)
        model = trainer.model
        self.assertIs(type(model), DRFU)
        self.assertEqual(model.prognostic_hidden_dim, 83)
        self.assertEqual(model.interaction_hidden_dim, 83)
        self.assertEqual(model.router_hidden_dim, 37)
        self.assertEqual(model.router.fc1.in_features, 111)
        self.assertEqual(model.router.fc2.out_features, 3)
        self.assertEqual(len(trainer.optimizer.param_groups), 1)
        for name, cls in (("cdum", CPM), ("cpm", CPM), ("two_branch_dynamic_fusion", TwoBranchDynamicFusion)):
            self.assertIs(type(build_model(name, self.args).model), cls)
        self.assertEqual(build_parser().parse_args([]).model, "cdum")

    def test_factory_custom_dimensions_and_huber(self):
        args = self.custom_args()
        args.cpm_huber_delta = 2.5
        trainer = build_model(MODEL, args)
        self.assertEqual(trainer.criterion.delta, 2.5)
        self.assertEqual(trainer.model.prognostic_hidden_dim, 31)
        self.assertEqual(trainer.model.router_hidden_dim, 17)
        self.assertEqual(trainer.model.interaction_hidden_dim, 23)
        self.assertEqual(trainer.model.encoder.num_features, 6)

    def test_yaml_sections_and_cli_precedence(self):
        for section in ("model", "cdum", "cpm", MODEL):
            with self.subTest(section=section):
                cfg = self.root / "config.yaml"
                content = {"model": {"name": MODEL}}
                content.setdefault(section, {}).update({
                    "prognostic_hidden_dim": 97,
                    "router_hidden_dim": 41,
                })
                cfg.write_text(yaml.safe_dump(content))
                merged = merge_config_into_args(build_parser().parse_args([]), str(cfg), raw_argv=[])
                self.assertEqual(merged.prognostic_hidden_dim, 97)
                self.assertEqual(merged.router_hidden_dim, 41)
                argv = ["--model", "cdum", "--cdum_prognostic_hidden_dim", "128"]
                overridden = merge_config_into_args(build_parser().parse_args(argv), str(cfg), raw_argv=argv)
                self.assertEqual(overridden.model, "cdum")
                self.assertEqual(overridden.prognostic_hidden_dim, 128)

    def test_saved_json_custom_dimensions_reconstruction(self):
        original, ckpt, cfg, loader = self.fixture()
        args = merge_config_into_args(build_parser().parse_args([]), str(cfg), raw_argv=[])
        restored = build_model(args.model, args)
        restored.load(str(ckpt))
        original.model.eval()
        restored.model.eval()
        x, t, _ = next(iter(loader))
        for key, value in original.model(x, t).items():
            torch.testing.assert_close(restored.model(x, t)[key], value, rtol=0, atol=0)
        overridden = merge_config_into_args(build_parser().parse_args(["--prognostic_hidden_dim", "19"]), str(cfg), raw_argv=["--prognostic_hidden_dim", "19"])
        self.assertEqual(overridden.prognostic_hidden_dim, 19)

    def test_main_dispatch_single_seed(self):
        argv = ["main.py", "--model", MODEL, "--prognostic_hidden_dim", "71", "--seed", "1", "--epochs", "1", "--max_samples", "1000"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(main_module, "get_dataloaders", return_value=([], [], [])) as data, mock.patch.object(main_module, "run_model") as run:
            main_module.main()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], MODEL)
        args = run.call_args.args[1]
        self.assertEqual(args.prognostic_hidden_dim, 71)
        self.assertEqual(args.seeds, [1])
        self.assertEqual(data.call_args.kwargs["max_samples"], 1000)

    def test_bucketing_matches_baselines_train_only(self):
        x = torch.rand(40, 12) * 20
        t = torch.tensor([0, 1] * 20)
        y = torch.tensor([0., 0., 1., 1.] * 10)
        train = DataLoader(CriteoDataset(x, t, y), batch_size=10)
        validation = DataLoader(CriteoDataset(x * 100, t, y), batch_size=10)
        expected = x.max(0).values.clamp(min=1e-8)
        reference = None
        for name in ("cdum", "two_branch_dynamic_fusion", MODEL):
            tr, va, te = prepare_loaders_for_model(name, self.args, train, validation, validation)
            self.assertIsInstance(tr, BucketedDataLoader)
            self.assertIs(tr.bucketer, va.bucketer)
            self.assertIs(tr.bucketer, te.bucketer)
            torch.testing.assert_close(tr.bucketer.denominators, expected)
            ids = next(iter(tr))[0]
            if reference is not None:
                torch.testing.assert_close(ids, reference)
            reference = ids
            for batch in te:
                self.assertTrue(((batch[0] >= 0) & (batch[0] <= 100)).all())
            torch.testing.assert_close(tr.bucketer.denominators, expected)

    def test_evaluate_cli_loads_real_checkpoint_and_writes_metrics(self):
        _, ckpt, cfg, loader = self.fixture()
        csv = self.root / "evaluation.csv"
        argv = ["evaluate.py", "--config", str(cfg), "--model", MODEL,
                "--prognostic_hidden_dim", "31", "--save_csv", str(csv)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(evaluate_module, "get_dataloaders", return_value=(loader, loader, loader)), mock.patch.object(evaluate_module, "evaluate_single_checkpoint", wraps=evaluate_module.evaluate_single_checkpoint) as evaluate:
            evaluate_module.main()
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.kwargs["args"].prognostic_hidden_dim, 31)
        frame = pd.read_csv(csv)  # catches the evaluator's swallowed exceptions
        self.assertEqual(frame.loc[0, "model"], MODEL)
        self.assertEqual(frame.loc[0, "checkpoint_path"], str(ckpt))
        self.assertFalse(frame[["loss", "auuc", "qini", "lift@30%"]].isna().any().any())

    def test_checkpoint_inspection_three_way_labels_and_norms(self):
        trainer, ckpt, cfg, loader = self.fixture()
        state = trainer.model.state_dict()
        for config in (str(cfg), None):
            model = detect_and_load_model(config, state)
            self.assertIs(type(model), DRFU)
            metrics = inspect_checkpoint(model, str(ckpt), next(iter(loader)))
            self.assertEqual(metrics["prognostic_candidate_diff"], 0)
            for name in ("P", "C", "I"):
                self.assertIn(f"router_zero_input_{name}", metrics)
                self.assertIn(f"z_{name}_ctrl_norm", metrics)
            self.assertAlmostEqual(sum(metrics[f"router_zero_input_{n}"] for n in ("P", "C", "I")), 1, places=6)
            self.assertTrue(metrics["router_logit_bias_free"])
            self.assertAlmostEqual(sum(metrics[f"router_ctrl_pi_{n}"] for n in ("P", "C", "I")), 1, places=6)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_metrics(metrics)
            for name in ("pi_P=", "pi_C=", "pi_I="):
                self.assertIn(name, output.getvalue())

    def test_two_way_checkpoints_remain_strictly_separate(self):
        trainer, ckpt, _, _ = self.fixture()
        two_way = build_model("two_branch_dynamic_fusion", self.args)
        with self.assertRaises(RuntimeError):
            two_way.load(str(ckpt))
        old_ckpt = self.root / "two_way.pth"
        two_way.save(str(old_ckpt))
        with self.assertRaises(RuntimeError):
            trainer.load(str(old_ckpt))
        metrics = inspect_checkpoint(two_way.model, str(old_ckpt))
        self.assertIn("router_prior_I", metrics)
        self.assertNotIn("router_prior_P", metrics)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
