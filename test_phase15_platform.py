import os
import tempfile
import unittest

from internet_hub import extract_keyphrases, fetch_world_events
from platform_db import (
    authenticate_account,
    create_account,
    create_project,
    init_db,
    list_models,
    register_model,
    seed_common_phrases,
)
from polyglot_bridge import scaffold_polyglot_connectors
from simulation_lab import generate_simulation_dataset, train_agent_with_self_play


class TestPhase15Platform(unittest.TestCase):
    def test_db_account_project_model_flow(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "platform.db")
            init_db(db_path)
            create_account("kai", "secret", db_path=db_path, preferred_lang="en")
            auth = authenticate_account("kai", "secret", db_path=db_path)
            self.assertTrue(auth["ok"])

            proj = create_project("kai", "uncanny_lab", db_path=db_path)
            self.assertEqual(proj["name"], "uncanny_lab")

            model = register_model(
                owner_username="kai",
                project_name="uncanny_lab",
                model_name="sim_v1",
                checkpoint_path="outputs/sim_lab/sim_agent.pth",
                dsl="[16, 32], diamond: [32], [32, 5]",
                db_path=db_path,
            )
            self.assertEqual(model["name"], "sim_v1")
            rows = list_models(db_path=db_path, project_name="uncanny_lab", owner_username="kai")
            self.assertGreaterEqual(len(rows), 1)

    def test_simulation_dataset_generation(self):
        ds = generate_simulation_dataset(
            episodes=3,
            grid_size=8,
            obstacle_prob=0.05,
            max_steps=25,
            policy_model=None,
        )
        self.assertEqual(ds.observations.shape[1], 16)
        self.assertEqual(ds.actions.dim(), 2)
        self.assertGreaterEqual(int(ds.actions.min().item()), 0)
        self.assertLessEqual(int(ds.actions.max().item()), 4)

    def test_short_self_play_train(self):
        with tempfile.TemporaryDirectory() as td:
            report = train_agent_with_self_play(
                cycles=1,
                episodes_per_cycle=6,
                epochs_per_cycle=2,
                grid_size=8,
                obstacle_prob=0.08,
                max_steps=30,
                device="cpu",
                out_dir=td,
                seed=7,
            )
            self.assertTrue(os.path.exists(report["checkpoint"]))
            self.assertTrue(os.path.exists(report["dataset_csv"]))
            self.assertTrue(os.path.exists(report["report_json"]))

    def test_polyglot_scaffold(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = scaffold_polyglot_connectors(out_dir=td, base_url="http://127.0.0.1:8090")
            self.assertTrue(os.path.exists(manifest["manifest_path"]))
            self.assertTrue(os.path.exists(os.path.join(td, "web", "index.html")))
            self.assertTrue(os.path.exists(os.path.join(td, "javascript", "client.mjs")))

    def test_event_and_phrase_pipeline(self):
        events = fetch_world_events(max_items=5, include_network=False)
        self.assertGreaterEqual(len(events), 1)
        phrases = extract_keyphrases([f"{e.get('title', '')} {e.get('summary', '')}" for e in events], top_k=10)
        self.assertGreaterEqual(len(phrases), 1)

        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "phrases.db")
            init_db(db_path)
            seeded = seed_common_phrases(db_path=db_path)
            self.assertGreater(seeded, 0)


if __name__ == "__main__":
    unittest.main()
