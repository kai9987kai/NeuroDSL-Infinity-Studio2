"""Tests for DB-backed account, session, model registry, and run logging."""

from __future__ import annotations

import os
import unittest

from platform_db import (
    authenticate_api_token,
    create_account,
    create_api_session,
    ensure_project,
    get_snapshot,
    list_model_runs,
    register_model,
    revoke_api_token,
    log_model_run,
)


class TestPlatformRegistry(unittest.TestCase):
    def setUp(self):
        self.db_path = "outputs/test_registry.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_account_session_model_and_runs(self):
        acc = create_account("owner", "secret", db_path=self.db_path, role="admin")
        self.assertEqual(acc["username"], "owner")

        proj = ensure_project("owner", "demo_project", db_path=self.db_path)
        self.assertEqual(proj["name"], "demo_project")

        mdl = register_model(
            project_name="demo_project",
            model_name="demo_model",
            checkpoint_path="demo_model.pth",
            dsl="[32, 16], [16, 10]",
            metrics_json='{"acc":0.9}',
            owner_username="owner",
            db_path=self.db_path,
        )
        self.assertEqual(mdl["name"], "demo_model")

        sess = create_api_session("owner", "secret", db_path=self.db_path, ttl_hours=1)
        self.assertTrue(sess["ok"])
        token = sess["token"]

        auth = authenticate_api_token(token, db_path=self.db_path)
        self.assertTrue(auth["ok"])
        self.assertEqual(auth["username"], "owner")

        run = log_model_run(
            run_mode="infer",
            request_obj={"rows": 1},
            response_obj={"out_dim": 10},
            latency_ms=1.23,
            db_path=self.db_path,
            session_id=auth["session_id"],
            account_id=auth["account_id"],
        )
        self.assertGreater(run["id"], 0)

        runs = list_model_runs(db_path=self.db_path, username="owner", limit=10)
        self.assertGreaterEqual(len(runs), 1)

        snap = get_snapshot(db_path=self.db_path)
        self.assertEqual(snap["accounts"], 1)
        self.assertEqual(snap["projects"], 1)
        self.assertEqual(snap["models"], 1)
        self.assertEqual(snap["sessions"], 1)
        self.assertGreaterEqual(snap["runs"], 1)

        out = revoke_api_token(token, db_path=self.db_path)
        self.assertTrue(out["ok"])
        auth2 = authenticate_api_token(token, db_path=self.db_path)
        self.assertFalse(auth2["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
