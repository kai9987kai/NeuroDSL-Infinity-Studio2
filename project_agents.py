"""Project/runtime manager and model agents for coordinated experimentation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from platform_db import register_model
from simulation_lab import train_agent_with_self_play


class ModelAgent:
    """Execution wrapper for a trainable simulation agent."""

    def __init__(self, name: str, dsl: str = ""):
        self.name = name.strip() or "agent"
        self.dsl = dsl.strip()

    def run(self, out_dir: str, device: str = "auto", cycles: int = 3, episodes_per_cycle: int = 20) -> dict:
        report = train_agent_with_self_play(
            dsl=self.dsl,
            cycles=cycles,
            episodes_per_cycle=episodes_per_cycle,
            out_dir=out_dir,
            device=device,
        )
        report["agent_name"] = self.name
        return report


class ProjectRuntimeManager:
    """Runs multiple model agents and optionally registers artifacts in platform DB."""

    def __init__(self, db_path: str = "outputs/neuro_platform.db", max_workers: int = 2):
        self.db_path = db_path
        self.max_workers = max(1, int(max_workers))
        self.agents: list[ModelAgent] = []

    def add_agent(self, agent: ModelAgent) -> None:
        self.agents.append(agent)

    def run_all(
        self,
        project_name: str,
        owner_username: str,
        out_root: str = "outputs/agent_runtime",
        device: str = "auto",
    ) -> dict:
        if not self.agents:
            raise ValueError("no agents configured")

        runs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for idx, agent in enumerate(self.agents):
                out_dir = f"{out_root}/{agent.name}_{idx+1}"
                fut = pool.submit(agent.run, out_dir=out_dir, device=device)
                futures[fut] = (idx, agent)

            for fut in as_completed(futures):
                idx, agent = futures[fut]
                try:
                    report = fut.result()
                except Exception as exc:
                    runs.append(
                        {
                            "agent_index": idx,
                            "agent_name": agent.name,
                            "ok": False,
                            "error": str(exc),
                        }
                    )
                    continue

                checkpoint = str(report.get("checkpoint", ""))
                register_model(
                    project_name=project_name,
                    owner_username=owner_username,
                    model_name=f"{agent.name}_sim_model",
                    checkpoint_path=checkpoint,
                    dsl=str(report.get("dsl", "")),
                    metrics_json=json.dumps({"cycles": report.get("cycles", [])}),
                    agent_name=agent.name,
                    db_path=self.db_path,
                )
                runs.append(
                    {
                        "agent_index": idx,
                        "agent_name": agent.name,
                        "ok": True,
                        "checkpoint": checkpoint,
                        "eval_success_rate": (
                            report.get("cycles", [{}])[-1].get("eval_success_rate", 0.0)
                            if report.get("cycles")
                            else 0.0
                        ),
                    }
                )

        runs.sort(key=lambda r: r.get("agent_index", 0))
        return {"runs": runs, "count": len(runs)}
