import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quadmix.core.types import ProxyResult, QuaDMixConfig
from quadmix.pipeline.active_search import (
    ActiveSearchConfig,
    ActiveSearchController,
    FIXED_TRAIN_STEPS,
    _parameter_key,
    select_active_batch,
)
from quadmix.pipeline.param_sampler import ParameterSampler


class ActiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.quadmix_config = QuaDMixConfig(
            num_domains=4,
            num_quality_criteria=5,
        )
        self.candidates = ParameterSampler(
            self.quadmix_config, seed=123
        ).sample_batch(200)

    def test_fixed_train_steps_are_enforced(self):
        ActiveSearchConfig(train_steps=FIXED_TRAIN_STEPS).validate()
        with self.assertRaisesRegex(ValueError, "fixed at 5000"):
            ActiveSearchConfig(train_steps=1000).validate()

    def test_active_batch_is_unique_and_respects_exclusions(self):
        means = np.linspace(0.0, 1.0, len(self.candidates))
        stds = np.linspace(0.2, 0.0, len(self.candidates))
        excluded = {_parameter_key(self.candidates[0])}
        choices = select_active_batch(
            candidates=self.candidates,
            predicted_mean=means,
            predicted_std=stds,
            batch_size=32,
            beta=1.5,
            rng=np.random.default_rng(7),
            excluded_keys=excluded,
        )

        keys = [_parameter_key(choice.parameters) for choice in choices]
        self.assertEqual(len(choices), 32)
        self.assertEqual(len(set(keys)), 32)
        self.assertNotIn(_parameter_key(self.candidates[0]), keys)
        self.assertIn("exploit", {choice.source for choice in choices})
        self.assertIn("uncertain", {choice.source for choice in choices})
        self.assertIn("diverse", {choice.source for choice in choices})
        self.assertIn("random", {choice.source for choice in choices})

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            ActiveSearchController._atomic_json(path, {"round": 1})
            ActiveSearchController._atomic_json(path, {"round": 2})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "round": 2\n}')

    def test_completed_round_is_recovered_into_history(self):
        class DummyRunner:
            tiny_steps = FIXED_TRAIN_STEPS

        with tempfile.TemporaryDirectory() as directory:
            controller = ActiveSearchController(
                quadmix_config=self.quadmix_config,
                active_config=ActiveSearchConfig(active_rounds=0),
                proxy_runner=DummyRunner(),
                output_dir=directory,
            )
            round_dir = Path(directory) / "rounds" / "round_00"
            record = {
                "round_id": 0,
                "global_experiment_id": 0,
            }
            controller._atomic_json(round_dir / "round_records.json", [record])
            controller._atomic_json(round_dir / "round_complete.json", {"round_id": 0})

            self.assertEqual(controller._load_history(), [record])
            self.assertTrue(controller.history_path.exists())

    def test_controller_runs_rounds_and_resumes_without_retraining(self):
        class FakeRunner:
            tiny_steps = FIXED_TRAIN_STEPS
            device_type = "cpu"

            def __init__(self):
                self.output_dir = None
                self.calls = 0

            def precompute_samples(self, params):
                return [np.array([index]) for index in range(len(params))]

            def tokenize_all_needed(self, selected):
                return None

            def run_batch_parallel(self, params, selected, **kwargs):
                self.calls += 1
                return [
                    ProxyResult(
                        parameters=value,
                        validation_loss=float(np.sum(value.flatten())),
                        metadata={"experiment_id": index},
                    )
                    for index, value in enumerate(params)
                ]

        class FakeModel:
            def predict(self, params):
                return np.array([np.sum(value.flatten()) for value in params])

        class FakeOptimizer:
            def __init__(self, config):
                self.results = []
                self._bootstrap_models = [FakeModel(), FakeModel()]
                self.regressor = FakeModel()

            def add_proxy_results(self, results):
                self.results.extend(results)

            def train_regressor(self):
                return self.regressor

            def search_optimal(self, n_search_points, top_k):
                params = self.results[0].parameters
                return params, [params], np.array([self.results[0].validation_loss])

        runner = FakeRunner()
        with tempfile.TemporaryDirectory() as directory, patch(
            "quadmix.pipeline.active_search.QuaDMixOptimizer", FakeOptimizer
        ):
            controller = ActiveSearchController(
                quadmix_config=self.quadmix_config,
                active_config=ActiveSearchConfig(
                    initial_experiments=4,
                    active_rounds=1,
                    batch_size=2,
                    candidate_pool_size=20,
                    final_search_points=20,
                    top_k=1,
                    num_workers=1,
                ),
                proxy_runner=runner,
                output_dir=directory,
            )
            first = controller.run()
            second = controller.run()

            self.assertEqual(first["num_experiments"], 6)
            self.assertEqual(second["num_experiments"], 6)
            self.assertEqual(runner.calls, 2)
            self.assertTrue((Path(directory) / "optimal_parameters.json").exists())


if __name__ == "__main__":
    unittest.main()
