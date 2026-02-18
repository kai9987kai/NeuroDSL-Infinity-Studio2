"""Neural simulation lab with diamond-topology observations and self-play training."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from device_utils import resolve_device
from parser_utils import create_modern_nn, validate_dsl
from trainer import TrainingEngine


ACTION_DELTAS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
    4: (0, 0),    # stay
}


@dataclass
class SimulationDataset:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    episodes: int
    success_count: int

    @property
    def success_rate(self) -> float:
        if self.episodes <= 0:
            return 0.0
        return float(self.success_count) / float(self.episodes)


class DiamondGridWorld:
    """Grid environment with obstacle awareness and a diamond observation profile."""

    def __init__(self, size: int = 12, obstacle_prob: float = 0.12, max_steps: int = 80):
        self.size = max(4, int(size))
        self.obstacle_prob = max(0.0, min(0.35, float(obstacle_prob)))
        self.max_steps = max(5, int(max_steps))
        self.agent = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.obstacles: set[tuple[int, int]] = set()
        self.steps = 0

    def reset(self) -> torch.Tensor:
        self.steps = 0
        self.obstacles = set()
        for y in range(self.size):
            for x in range(self.size):
                if random.random() < self.obstacle_prob:
                    self.obstacles.add((x, y))
        self.agent = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.obstacles.discard(self.agent)
        self.obstacles.discard(self.goal)
        return self.observe()

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_blocked(self, x: int, y: int) -> bool:
        return (not self._inside(x, y)) or ((x, y) in self.obstacles)

    def _directional_clearance(self, dx: int, dy: int, max_scan: int = 4) -> float:
        ax, ay = self.agent
        for k in range(1, max_scan + 1):
            tx = ax + dx * k
            ty = ay + dy * k
            if self._is_blocked(tx, ty):
                return float(k - 1) / float(max_scan)
        return 1.0

    def observe(self) -> torch.Tensor:
        ax, ay = self.agent
        gx, gy = self.goal
        size_f = float(max(1, self.size - 1))
        dx = (gx - ax) / size_f
        dy = (gy - ay) / size_f
        manhattan = (abs(gx - ax) + abs(gy - ay)) / float(max(1, 2 * (self.size - 1)))
        step_ratio = self.steps / float(max(1, self.max_steps))

        directional = [
            self._directional_clearance(0, -1),   # N
            self._directional_clearance(0, 1),    # S
            self._directional_clearance(1, 0),    # E
            self._directional_clearance(-1, 0),   # W
            self._directional_clearance(1, -1),   # NE
            self._directional_clearance(-1, -1),  # NW
            self._directional_clearance(1, 1),    # SE
            self._directional_clearance(-1, 1),   # SW
        ]

        obs = [
            ax / size_f,
            ay / size_f,
            gx / size_f,
            gy / size_f,
            dx,
            dy,
            manhattan,
            step_ratio,
            *directional,
        ]
        return torch.tensor(obs, dtype=torch.float32)

    def simulate_action(self, action: int) -> tuple[int, int]:
        move = ACTION_DELTAS.get(int(action), (0, 0))
        ax, ay = self.agent
        tx, ty = ax + move[0], ay + move[1]
        if self._is_blocked(tx, ty):
            return ax, ay
        return tx, ty

    def expert_action(self) -> int:
        gx, gy = self.goal
        best_action = 4
        best_score = float("inf")
        for action in range(5):
            tx, ty = self.simulate_action(action)
            blocked_penalty = 0.0 if (tx, ty) != self.agent or action == 4 else 3.0
            score = abs(gx - tx) + abs(gy - ty) + blocked_penalty
            if score < best_score:
                best_score = score
                best_action = action
        return best_action

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        prev = self.agent
        self.agent = self.simulate_action(action)
        self.steps += 1

        reward = -0.01
        if self.agent == prev and int(action) != 4:
            reward -= 0.08
        if self.agent == self.goal:
            reward += 1.0
            return self.observe(), reward, True
        if self.steps >= self.max_steps:
            return self.observe(), reward, True
        return self.observe(), reward, False


def default_diamond_sim_dsl(input_dim: int = 16, output_dim: int = 5) -> str:
    return (
        f"[{input_dim}, 96], diamond: [96], trans: [96], "
        f"moe: [96, 6, 1], mod: [96, 4, 0.4], [96, {output_dim}]"
    )


def generate_simulation_dataset(
    episodes: int = 32,
    grid_size: int = 12,
    obstacle_prob: float = 0.12,
    max_steps: int = 80,
    policy_model=None,
    policy_device: str = "cpu",
    agent_blend: float = 0.65,
    epsilon: float = 0.10,
) -> SimulationDataset:
    env = DiamondGridWorld(size=grid_size, obstacle_prob=obstacle_prob, max_steps=max_steps)
    observations = []
    actions = []
    rewards = []
    success_count = 0
    agent_blend = max(0.0, min(1.0, float(agent_blend)))
    epsilon = max(0.0, min(0.9, float(epsilon)))
    policy_device_obj = torch.device(policy_device)

    if policy_model is not None:
        policy_model.eval()

    for _ in range(max(1, int(episodes))):
        obs = env.reset()
        done = False
        while not done:
            expert = env.expert_action()
            chosen = expert
            if policy_model is not None and random.random() < agent_blend:
                if random.random() < epsilon:
                    chosen = random.randint(0, 4)
                else:
                    with torch.no_grad():
                        logits = policy_model(obs.unsqueeze(0).to(policy_device_obj))
                        chosen = int(torch.argmax(logits, dim=-1).item() % 5)

            next_obs, reward, done = env.step(chosen)
            observations.append(obs.clone())
            # Train against expert action so generated rollouts stay grounded.
            actions.append([int(expert)])
            rewards.append([float(reward)])
            obs = next_obs

        if env.agent == env.goal:
            success_count += 1

    if not observations:
        observations = [torch.zeros(16, dtype=torch.float32)]
        actions = [[4]]
        rewards = [[0.0]]

    return SimulationDataset(
        observations=torch.stack(observations).float(),
        actions=torch.tensor(actions, dtype=torch.long),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        episodes=max(1, int(episodes)),
        success_count=success_count,
    )


def evaluate_policy(
    model,
    episodes: int = 20,
    grid_size: int = 12,
    obstacle_prob: float = 0.12,
    max_steps: int = 80,
    device: str = "cpu",
) -> dict:
    env = DiamondGridWorld(size=grid_size, obstacle_prob=obstacle_prob, max_steps=max_steps)
    success = 0
    total_reward = 0.0
    dev = torch.device(device)
    model.eval()
    for _ in range(max(1, int(episodes))):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                logits = model(obs.unsqueeze(0).to(dev))
                action = int(torch.argmax(logits, dim=-1).item() % 5)
            obs, reward, done = env.step(action)
            ep_reward += float(reward)
        if env.agent == env.goal:
            success += 1
        total_reward += ep_reward
    return {
        "episodes": int(max(1, episodes)),
        "success_rate": float(success) / float(max(1, episodes)),
        "avg_reward": total_reward / float(max(1, episodes)),
    }


def export_dataset_csv(dataset: SimulationDataset, csv_path: str) -> str:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        n_features = int(dataset.observations.shape[1])
        writer.writerow([f"f{i}" for i in range(n_features)] + ["action", "reward"])
        for i in range(dataset.observations.shape[0]):
            row = dataset.observations[i].tolist()
            row.append(int(dataset.actions[i, 0].item()))
            row.append(float(dataset.rewards[i, 0].item()))
            writer.writerow(row)
    return str(out)


def train_agent_with_self_play(
    dsl: str = "",
    cycles: int = 4,
    episodes_per_cycle: int = 24,
    epochs_per_cycle: int = 20,
    grid_size: int = 12,
    obstacle_prob: float = 0.12,
    max_steps: int = 80,
    device: str = "auto",
    out_dir: str = "outputs/sim_lab",
    seed: int = 42,
) -> dict:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dsl = (dsl or "").strip() or default_diamond_sim_dsl()
    issues, defs = validate_dsl(dsl)
    if not defs:
        raise ValueError(f"simulation DSL failed validation: {issues}")

    first = defs[0]
    last = defs[-1]
    expected_in = int(first.get("in", first.get("dim", -1)))
    expected_out = int(last.get("out", last.get("dim", -1)))
    if expected_in != 16:
        raise ValueError(f"simulation DSL input dim must be 16, got {expected_in}")
    if expected_out < 5:
        raise ValueError(f"simulation DSL output dim must be >= 5, got {expected_out}")

    dev = resolve_device(device)
    model = create_modern_nn(defs).to(dev)
    trainer = TrainingEngine(
        model,
        loss_fn="CrossEntropy",
        max_epochs=max(1, int(epochs_per_cycle)),
        grad_clip=1.0,
        warmup_steps=3,
        aux_loss_coef=0.02,
    )

    cycle_reports = []
    last_dataset = None
    for cycle in range(max(1, int(cycles))):
        dataset = generate_simulation_dataset(
            episodes=max(4, int(episodes_per_cycle)),
            grid_size=grid_size,
            obstacle_prob=obstacle_prob,
            max_steps=max_steps,
            policy_model=model if cycle > 0 else None,
            policy_device=str(dev),
            agent_blend=min(0.85, 0.60 + 0.08 * cycle),
            epsilon=max(0.03, 0.15 - 0.02 * cycle),
        )
        last_dataset = dataset

        loss = 0.0
        for _ in range(max(1, int(epochs_per_cycle))):
            loss, _, _ = trainer.train_step(dataset.observations, dataset.actions)

        eval_stats = evaluate_policy(
            model,
            episodes=12,
            grid_size=grid_size,
            obstacle_prob=obstacle_prob,
            max_steps=max_steps,
            device=str(dev),
        )
        cycle_reports.append(
            {
                "cycle": cycle + 1,
                "train_loss": float(loss),
                "dataset_rows": int(dataset.observations.shape[0]),
                "dataset_success_rate": float(dataset.success_rate),
                "eval_success_rate": float(eval_stats["success_rate"]),
                "eval_avg_reward": float(eval_stats["avg_reward"]),
            }
        )

    checkpoint = out / "sim_agent.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dsl": dsl,
            "training": cycle_reports,
            "issues": issues,
        },
        checkpoint,
    )

    dataset_csv = out / "sim_dataset.csv"
    if last_dataset is not None:
        export_dataset_csv(last_dataset, str(dataset_csv))

    report = {
        "dsl": dsl,
        "device": str(dev),
        "issues": issues,
        "cycles": cycle_reports,
        "checkpoint": str(checkpoint),
        "dataset_csv": str(dataset_csv),
    }
    report_path = out / "sim_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path)
    return report
