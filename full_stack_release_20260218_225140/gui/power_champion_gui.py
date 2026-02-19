"""GUI runner for NeuroDSL champion model with visual output tools."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch

from device_utils import resolve_device
from parser_utils import create_modern_nn, parse_program


def _default_asset(name: str) -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / name,
        here / name,
        Path.cwd() / "outputs" / "power_champion_release_minimal_v2" / name,
        Path.cwd() / "outputs" / "power_champion_release" / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(Path.cwd() / "outputs" / "power_champion_release_minimal_v2" / name)


DEFAULT_MODEL = _default_asset("champion_model.pth")
DEFAULT_DSL = _default_asset("champion_model.dsl")


def _candidate_paths(filename: str) -> list[Path]:
    out: list[Path] = []
    if getattr(sys, "frozen", False):
        out.append(Path(sys.executable).resolve().parent / filename)
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        out.append(Path(meipass) / filename)
    out.append(Path.cwd() / filename)
    return out


def resolve_asset(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    for c in _candidate_paths(p.name):
        if c.exists():
            return str(c)
    return path


def load_model_from_files(model_path: str, dsl_path: str, device):
    model_path = resolve_asset(model_path)
    dsl_path = resolve_asset(dsl_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(dsl_path):
        raise FileNotFoundError(f"DSL file not found: {dsl_path}")

    dsl = Path(dsl_path).read_text(encoding="utf-8")
    defs = parse_program(dsl)
    if not defs:
        raise ValueError("Failed to parse DSL file.")
    model = create_modern_nn(defs).to(device)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        cleaned[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    in_dim = int(defs[0].get("in", defs[0].get("dim")))
    return model, in_dim, defs


class ChampionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroDSL Power Champion GUI")
        self.root.geometry("1180x780")
        self.device = resolve_device("auto")
        self.model = None
        self.in_dim = None
        self.out_dim = None
        self.layer_defs = []
        self.model_param_count = 0
        self.last_outputs = []
        self.last_stds = []
        self.last_payload = {}
        self.history = []
        self.optimized_input = None
        self.last_batch_summary = {}

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.dsl_var = tk.StringVar(value=DEFAULT_DSL)
        self.input_var = tk.StringVar(value="")
        self.input_b_var = tk.StringVar(value="")
        self.batch_csv_var = tk.StringVar(value="")
        self.bench_runs_var = tk.IntVar(value=60)
        self.mc_samples_var = tk.IntVar(value=1)
        self.mc_grid_var = tk.StringVar(value="1,2,4,8")
        self.mutation_count_var = tk.IntVar(value=24)
        self.mutation_scale_var = tk.DoubleVar(value=0.12)
        self.interp_steps_var = tk.IntVar(value=11)
        self.sensitivity_delta_var = tk.DoubleVar(value=0.05)
        self.sensitivity_relative_var = tk.BooleanVar(value=True)
        self.topk_var = tk.IntVar(value=5)
        self.as_probs_var = tk.BooleanVar(value=False)
        self.auto_mc_var = tk.BooleanVar(value=False)
        self.auto_mc_threshold_var = tk.DoubleVar(value=0.05)
        self.auto_mc_max_var = tk.IntVar(value=32)
        self.target_class_var = tk.IntVar(value=0)
        self.optimize_steps_var = tk.IntVar(value=36)
        self.optimize_lr_var = tk.DoubleVar(value=0.08)
        self.optimize_clip_var = tk.DoubleVar(value=2.5)
        self.optimize_l2_var = tk.DoubleVar(value=0.001)
        self.quick_cmd_var = tk.StringVar(value="")
        self.model_info_var = tk.StringVar(value="Layers: - | Output dim: - | Params: -")
        self.status_var = tk.StringVar(value=f"Device: {self.device} | Model: not loaded")

        self._build_ui()
        self._wire_shortcuts()
        self._log("GUI initialized.")

    def _wire_shortcuts(self):
        self.root.bind("<Control-l>", lambda _e: self.load_model())
        self.root.bind("<Control-r>", lambda _e: self.run_infer())
        self.root.bind("<Control-b>", lambda _e: self.run_benchmark())
        self.root.bind("<Control-s>", lambda _e: self.save_outputs())
        self.root.bind("<Control-o>", lambda _e: self.run_batch_from_saved())
        self.root.bind("<Control-e>", lambda _e: self.save_snapshot())
        self.root.bind("<Control-d>", lambda _e: self.run_compare())
        self.root.bind("<Control-m>", lambda _e: self.run_mc_profile())
        self.root.bind("<Control-t>", lambda _e: self.run_mutation_lab())
        self.root.bind("<Control-u>", lambda _e: self.run_input_optimizer())
        self.root.bind("<Control-j>", lambda _e: self.show_batch_insights())
        self.root.bind("<Control-i>", lambda _e: self.run_interpolate())
        self.root.bind("<Control-y>", lambda _e: self.run_sensitivity_scan())
        self.root.bind("<Control-k>", lambda _e: self.run_quick_command())

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(outer, text="Model Files", padding=8)
        top.pack(fill=tk.X, pady=4)

        ttk.Label(top, text="Model (.pth):").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_var, width=80).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse", command=self._pick_model).grid(row=0, column=2, sticky="ew")

        ttk.Label(top, text="DSL (.dsl):").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.dsl_var, width=80).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse", command=self._pick_dsl).grid(row=1, column=2, sticky="ew")

        ttk.Button(top, text="Load Model", command=self.load_model).grid(row=2, column=0, sticky="ew", pady=8)
        ttk.Button(top, text="Random Input", command=self.fill_random).grid(row=2, column=1, sticky="w", pady=8)

        intel = ttk.Frame(top)
        intel.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        ttk.Label(intel, text="Model Intel:").pack(side=tk.LEFT)
        ttk.Label(intel, textvariable=self.model_info_var).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(intel, text="Copy", command=self.copy_model_intel).pack(side=tk.RIGHT)
        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(outer)
        mid.pack(fill=tk.BOTH, expand=True, pady=4)

        left = ttk.LabelFrame(mid, text="Inference + Batch", padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ttk.Label(left, text="Input Vector (comma-separated)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.input_var).pack(fill=tk.X, pady=(2, 6))
        ttk.Label(left, text="Compare Input B (optional)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.input_b_var).pack(fill=tk.X, pady=(2, 6))

        batch_src = ttk.Frame(left)
        batch_src.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(batch_src, text="Batch CSV").pack(side=tk.LEFT)
        ttk.Entry(batch_src, textvariable=self.batch_csv_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(batch_src, text="Browse", command=self._pick_batch_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(batch_src, text="Run Batch Source", command=self.run_batch_from_saved).pack(side=tk.LEFT)

        actions = ttk.Frame(left)
        actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Run Infer", command=self.run_infer).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Run Compare", command=self.run_compare).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Load CSV + Run", command=self.run_batch).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Save Outputs CSV", command=self.save_outputs).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Export History", command=self.export_history_json).pack(side=tk.LEFT)

        actions2 = ttk.Frame(left)
        actions2.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(actions2, text="Copy Result", command=self.copy_result_to_clipboard).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions2, text="Save Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions2, text="Load Snapshot", command=self.load_snapshot).pack(side=tk.LEFT)

        actions3 = ttk.Frame(left)
        actions3.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(actions3, text="Batch Insights", command=self.show_batch_insights).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions3, text="Export Insights CSV", command=self.export_batch_insights_csv).pack(side=tk.LEFT)

        cmd_row = ttk.Frame(left)
        cmd_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(cmd_row, text="Command Prompt").pack(side=tk.LEFT)
        self.quick_entry = ttk.Entry(cmd_row, textvariable=self.quick_cmd_var)
        self.quick_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.quick_entry.bind("<Return>", lambda _e: self.run_quick_command())
        ttk.Button(cmd_row, text="Run Cmd", command=self.run_quick_command).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(cmd_row, text="Prompt Help", command=self.show_command_help).pack(side=tk.LEFT)

        self.output_box = tk.Text(left, height=18, wrap="none")
        self.output_box.pack(fill=tk.BOTH, expand=True, pady=6)

        right = ttk.LabelFrame(mid, text="Visual + Analytics", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        opts_row = ttk.Frame(right)
        opts_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(opts_row, text="As Probabilities", variable=self.as_probs_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(opts_row, text="Top-K").pack(side=tk.LEFT)
        ttk.Spinbox(opts_row, from_=1, to=50, textvariable=self.topk_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(opts_row, text="MC Samples").pack(side=tk.LEFT)
        ttk.Spinbox(opts_row, from_=1, to=64, textvariable=self.mc_samples_var, width=6).pack(side=tk.LEFT, padx=(4, 8))

        auto_row = ttk.Frame(right)
        auto_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(auto_row, text="Auto MC", variable=self.auto_mc_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(auto_row, text="Unc Threshold").pack(side=tk.LEFT)
        ttk.Spinbox(
            auto_row,
            from_=0.0,
            to=2.0,
            increment=0.01,
            textvariable=self.auto_mc_threshold_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(auto_row, text="MC Max").pack(side=tk.LEFT)
        ttk.Spinbox(auto_row, from_=1, to=256, textvariable=self.auto_mc_max_var, width=6).pack(side=tk.LEFT, padx=(6, 8))

        prof_row = ttk.Frame(right)
        prof_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(prof_row, text="MC Grid").pack(side=tk.LEFT)
        ttk.Entry(prof_row, textvariable=self.mc_grid_var, width=22).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(prof_row, text="MC Profile", command=self.run_mc_profile).pack(side=tk.LEFT)

        mut_row = ttk.Frame(right)
        mut_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(mut_row, text="Mutations").pack(side=tk.LEFT)
        ttk.Spinbox(mut_row, from_=4, to=256, textvariable=self.mutation_count_var, width=6).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(mut_row, text="Scale").pack(side=tk.LEFT)
        ttk.Spinbox(mut_row, from_=0.01, to=1.0, increment=0.01, textvariable=self.mutation_scale_var, width=7).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(mut_row, text="Mutation Lab", command=self.run_mutation_lab).pack(side=tk.LEFT)

        opt_row1 = ttk.Frame(right)
        opt_row1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(opt_row1, text="Target Class").pack(side=tk.LEFT)
        ttk.Spinbox(opt_row1, from_=0, to=4096, textvariable=self.target_class_var, width=7).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(opt_row1, text="Steps").pack(side=tk.LEFT)
        ttk.Spinbox(opt_row1, from_=1, to=1000, textvariable=self.optimize_steps_var, width=6).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(opt_row1, text="LR").pack(side=tk.LEFT)
        ttk.Spinbox(
            opt_row1,
            from_=0.001,
            to=2.0,
            increment=0.001,
            textvariable=self.optimize_lr_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(opt_row1, text="Optimize Input", command=self.run_input_optimizer).pack(side=tk.LEFT)

        opt_row2 = ttk.Frame(right)
        opt_row2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(opt_row2, text="Clip").pack(side=tk.LEFT)
        ttk.Spinbox(
            opt_row2,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.optimize_clip_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(opt_row2, text="L2").pack(side=tk.LEFT)
        ttk.Spinbox(
            opt_row2,
            from_=0.0,
            to=1.0,
            increment=0.0005,
            textvariable=self.optimize_l2_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(opt_row2, text="Use Optimized", command=self.use_optimized_input).pack(side=tk.LEFT)

        interp_row = ttk.Frame(right)
        interp_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(interp_row, text="Interp Steps").pack(side=tk.LEFT)
        ttk.Spinbox(interp_row, from_=3, to=201, textvariable=self.interp_steps_var, width=6).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(interp_row, text="Interpolate A->B", command=self.run_interpolate).pack(side=tk.LEFT)

        sens_row = ttk.Frame(right)
        sens_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(sens_row, text="Sens Delta").pack(side=tk.LEFT)
        ttk.Spinbox(
            sens_row,
            from_=0.001,
            to=2.0,
            increment=0.001,
            textvariable=self.sensitivity_delta_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Checkbutton(sens_row, text="Relative Delta", variable=self.sensitivity_relative_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(sens_row, text="Sensitivity Scan", command=self.run_sensitivity_scan).pack(side=tk.LEFT)

        bench_row = ttk.Frame(right)
        bench_row.pack(fill=tk.X, pady=4)
        ttk.Label(bench_row, text="Benchmark Runs").pack(side=tk.LEFT)
        ttk.Spinbox(bench_row, from_=1, to=5000, textvariable=self.bench_runs_var, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Button(bench_row, text="Benchmark", command=self.run_benchmark).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(right, bg="#111827", height=230, highlightthickness=0)
        self.canvas.pack(fill=tk.X, pady=6)

        ttk.Label(right, text="Analytics").pack(anchor="w")
        self.analysis_box = tk.Text(right, height=9, wrap="word")
        self.analysis_box.pack(fill=tk.X, pady=(2, 6))

        history_frame = ttk.Frame(right)
        history_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(history_frame, text="History").pack(anchor="w")
        self.history_list = tk.Listbox(history_frame, height=7)
        self.history_list.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        hist_actions = ttk.Frame(history_frame)
        hist_actions.pack(fill=tk.X)
        ttk.Button(hist_actions, text="Recall Selected", command=self.recall_selected_history).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(hist_actions, text="Clear History", command=self.clear_history).pack(side=tk.LEFT)

        ttk.Label(right, text="Activity Log").pack(anchor="w", pady=(8, 0))
        self.log_box = tk.Text(right, height=7, wrap="none")
        self.log_box.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(outer, textvariable=self.status_var)
        status.pack(fill=tk.X, pady=(6, 0))

    def _pick_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _pick_dsl(self):
        path = filedialog.askopenfilename(filetypes=[("DSL file", "*.dsl"), ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            self.dsl_var.set(path)

    def _pick_batch_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.batch_csv_var.set(path)

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def _set_output(self, text: str):
        self.output_box.delete("1.0", "end")
        self.output_box.insert("end", text)

    def _set_analysis(self, text: str):
        self.analysis_box.delete("1.0", "end")
        self.analysis_box.insert("end", text)

    def _update_model_intel(self):
        layers = len(self.layer_defs) if isinstance(self.layer_defs, list) else 0
        out_dim_txt = str(self.out_dim) if isinstance(self.out_dim, int) and self.out_dim > 0 else "-"
        params_txt = f"{int(self.model_param_count):,}" if int(self.model_param_count) > 0 else "-"
        self.model_info_var.set(f"Layers: {layers} | Output dim: {out_dim_txt} | Params: {params_txt}")

    def _sync_out_dim(self, out_dim: int):
        if int(out_dim) <= 0:
            return
        self.out_dim = int(out_dim)
        if int(self.target_class_var.get()) >= self.out_dim:
            self.target_class_var.set(max(0, self.out_dim - 1))
        self._update_model_intel()

    def copy_model_intel(self):
        text = self.model_info_var.get().strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._log("Copied model intel to clipboard.")

    def _load_rows_from_csv(self, path: str) -> list[list[float]]:
        rows: list[list[float]] = []
        with open(path, "r", newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            for row in rd:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        rows.append(vals)
                except Exception:
                    continue
        if not rows:
            raise ValueError("No numeric rows found.")
        if self.in_dim is not None:
            for i, r in enumerate(rows):
                if len(r) != self.in_dim:
                    raise ValueError(f"Row {i} has {len(r)} values, expected {self.in_dim}.")
        return rows

    def _refresh_history_list(self):
        self.history_list.delete(0, "end")
        for idx, item in enumerate(self.history):
            line = f"{idx+1:03d} | {item.get('time', '')} | {item.get('kind', '')} | dim={item.get('out_dim', 0)}"
            self.history_list.insert("end", line)

    def _record_history(self, payload: dict):
        self.history.append(payload)
        if len(self.history) > 200:
            self.history = self.history[-200:]
        self._refresh_history_list()

    def _predict(self, x: torch.Tensor, mc_samples: int, as_probs: bool):
        if self.model is None:
            raise ValueError("Model not loaded.")
        mc = max(1, int(mc_samples))
        prev_mode = self.model.training
        if mc == 1:
            self.model.eval()
            with torch.no_grad():
                y = self.model(x)
                if as_probs:
                    y = torch.softmax(y, dim=-1)
            self.model.train(prev_mode)
            return y, torch.zeros_like(y)

        preds = []
        self.model.train(True)
        with torch.no_grad():
            for _ in range(mc):
                y = self.model(x)
                if as_probs:
                    y = torch.softmax(y, dim=-1)
                preds.append(y.unsqueeze(0))
        stack = torch.cat(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0, unbiased=False)
        self.model.train(prev_mode)
        return mean, std

    def _predict_adaptive(
        self,
        x: torch.Tensor,
        mc_samples: int,
        as_probs: bool,
        auto_mc: bool,
        threshold: float,
        max_mc: int,
    ):
        base_mc = max(1, int(mc_samples))
        ceiling = max(base_mc, int(max_mc))
        out_t, std_t = self._predict(x, mc_samples=base_mc, as_probs=as_probs)
        diag = {
            "enabled": bool(auto_mc),
            "initial_mc_samples": int(base_mc),
            "threshold": float(max(0.0, threshold)),
            "max_mc_samples": int(ceiling),
            "adapted": False,
            "steps": [],
            "selected_mc_samples": int(base_mc),
        }
        if not auto_mc:
            return out_t, std_t, diag

        current_out = out_t
        current_std = std_t
        current_unc = float(current_std.abs().mean().item())
        last_mc = int(base_mc)
        while last_mc < ceiling and current_unc > float(max(0.0, threshold)):
            next_mc = min(int(ceiling), int(last_mc * 2))
            if next_mc <= last_mc:
                break
            probe_out, probe_std = self._predict(x, mc_samples=next_mc, as_probs=as_probs)
            probe_unc = float(probe_std.abs().mean().item())
            diag["steps"].append(
                {
                    "from_mc_samples": int(last_mc),
                    "to_mc_samples": int(next_mc),
                    "from_uncertainty": float(current_unc),
                    "to_uncertainty": float(probe_unc),
                }
            )
            current_out = probe_out
            current_std = probe_std
            current_unc = probe_unc
            last_mc = int(next_mc)
            diag["adapted"] = True
        diag["selected_mc_samples"] = int(last_mc)
        return current_out, current_std, diag

    def _confidence_summary(self, out: list[float], out_std: list[float], as_probs: bool) -> dict:
        if not out:
            return {"label": "unknown", "score": 0.0, "margin": 0.0, "avg_uncertainty": 0.0}
        top_sorted = sorted(out, reverse=True)
        top1 = float(top_sorted[0])
        top2 = float(top_sorted[1]) if len(top_sorted) > 1 else float(top1)
        margin = float(top1 - top2)
        avg_unc = float(sum(abs(v) for v in out_std) / max(1, len(out_std)))
        entropy = self._entropy(out) if as_probs else 0.0
        raw = (0.60 * math.tanh(max(0.0, margin) * 6.0)) + (0.40 * (1.0 - min(1.0, avg_unc)))
        if as_probs:
            raw = raw - (0.15 * min(1.0, entropy / max(1e-6, math.log(max(2, len(out))))))
        score = float(max(0.0, min(1.0, raw)))
        label = "high"
        if score < 0.45:
            label = "low"
        elif score < 0.70:
            label = "medium"
        return {
            "label": label,
            "score": score,
            "margin": margin,
            "avg_uncertainty": avg_unc,
            "entropy": float(entropy),
        }

    def _entropy(self, probs: list[float]) -> float:
        eps = 1e-12
        return float(-sum(p * math.log(max(p, eps)) for p in probs))

    def load_model(self):
        try:
            self.model, self.in_dim, self.layer_defs = load_model_from_files(
                self.model_var.get().strip(),
                self.dsl_var.get().strip(),
                self.device,
            )
            if self.layer_defs:
                last = self.layer_defs[-1]
                self.out_dim = int(last.get("out", last.get("dim", 0)))
            else:
                self.out_dim = None
            self.model_param_count = int(sum(p.numel() for p in self.model.parameters()))
            if self.out_dim and int(self.target_class_var.get()) >= int(self.out_dim):
                self.target_class_var.set(max(0, int(self.out_dim) - 1))
            self._update_model_intel()
            status_tail = f"input_dim={self.in_dim}"
            if self.out_dim:
                status_tail += f" | out_dim={self.out_dim}"
            self.status_var.set(f"Device: {self.device} | Model loaded | {status_tail}")
            self._log(f"Loaded model with input_dim={self.in_dim} out_dim={self.out_dim}")
            self.fill_random()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            self._log(f"Load error: {exc}")

    def _parse_input(self) -> list[float]:
        raw = self.input_var.get().strip()
        vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
        if not vals:
            raise ValueError("Input vector is empty.")
        if self.in_dim is not None and len(vals) != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} values, got {len(vals)}.")
        return vals

    def _parse_input_b(self) -> list[float]:
        raw = self.input_b_var.get().strip()
        vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
        if not vals:
            raise ValueError("Compare input B is empty.")
        if self.in_dim is not None and len(vals) != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} values for input B, got {len(vals)}.")
        return vals

    def _parse_mc_grid(self) -> list[int]:
        vals: list[int] = []
        for token in self.mc_grid_var.get().split(","):
            t = token.strip()
            if not t:
                continue
            n = int(t)
            if n <= 0:
                continue
            vals.append(min(256, n))
        if not vals:
            raise ValueError("MC grid is empty. Use values like 1,2,4,8.")
        return sorted(set(vals))

    def fill_random(self):
        if self.in_dim is None:
            self.input_var.set("0.1,0.2,0.3,0.4")
            return
        vals = [round(random.uniform(-1.0, 1.0), 4) for _ in range(self.in_dim)]
        self.input_var.set(",".join(str(v) for v in vals))

    def show_command_help(self):
        msg = (
            "Commands:\n"
            "infer | compare | interpolate [steps] | sensitivity | mutate | profile\n"
            "optimize | useopt | insights\n"
            "benchmark [runs] | random [infer]\n"
            "set mc <n> | set topk <n> | set auto <on|off> | set threshold <x>\n"
            "set mut <n> | set scale <x> | set interp <n> | set delta <x> | set relative <on|off>\n"
            "set target <idx> | set optsteps <n> | set optlr <x> | set optclip <x> | set optl2 <x>\n"
            "help"
        )
        messagebox.showinfo("Prompt Help", msg)
        self._log("Prompt help displayed.")

    def run_quick_command(self):
        raw = self.quick_cmd_var.get().strip()
        if not raw:
            return
        parts = raw.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("help", "?"):
                self.show_command_help()
                return
            if cmd == "infer":
                self.run_infer()
                return
            if cmd == "compare":
                self.run_compare()
                return
            if cmd in ("interpolate", "interp"):
                if len(parts) > 1:
                    self.interp_steps_var.set(max(3, min(201, int(parts[1]))))
                self.run_interpolate()
                return
            if cmd in ("sensitivity", "sens", "scan"):
                self.run_sensitivity_scan()
                return
            if cmd in ("mutate", "mutation", "mutlab"):
                self.run_mutation_lab()
                return
            if cmd in ("profile", "mcprofile"):
                self.run_mc_profile()
                return
            if cmd in ("optimize", "opt"):
                self.run_input_optimizer()
                return
            if cmd in ("useopt", "applyopt"):
                self.use_optimized_input()
                return
            if cmd in ("insights", "batchinsights"):
                self.show_batch_insights()
                return
            if cmd == "benchmark":
                if len(parts) > 1:
                    self.bench_runs_var.set(max(1, int(parts[1])))
                self.run_benchmark()
                return
            if cmd == "random":
                self.fill_random()
                if len(parts) > 1 and parts[1].lower() == "infer":
                    self.run_infer()
                else:
                    self._log("Random input generated.")
                return
            if cmd == "set":
                if len(parts) < 3:
                    raise ValueError("usage: set <mc|topk|auto|threshold|mut|scale|interp|delta|relative> <value>")
                key = parts[1].lower()
                val = parts[2]
                val_bool = val.strip().lower() in ("1", "true", "yes", "on")
                if key == "mc":
                    self.mc_samples_var.set(max(1, int(val)))
                elif key == "topk":
                    self.topk_var.set(max(1, int(val)))
                elif key == "auto":
                    self.auto_mc_var.set(val_bool)
                elif key == "threshold":
                    self.auto_mc_threshold_var.set(max(0.0, float(val)))
                elif key in ("mut", "mutations"):
                    self.mutation_count_var.set(max(4, int(val)))
                elif key == "scale":
                    self.mutation_scale_var.set(max(0.001, float(val)))
                elif key in ("interp", "steps"):
                    self.interp_steps_var.set(max(3, min(201, int(val))))
                elif key in ("delta", "sensdelta"):
                    self.sensitivity_delta_var.set(max(0.0001, float(val)))
                elif key in ("relative", "rel"):
                    self.sensitivity_relative_var.set(val_bool)
                elif key in ("target", "class"):
                    self.target_class_var.set(max(0, int(val)))
                elif key in ("optsteps", "steps_opt"):
                    self.optimize_steps_var.set(max(1, int(val)))
                elif key in ("optlr", "lr_opt"):
                    self.optimize_lr_var.set(max(0.0001, float(val)))
                elif key in ("optclip", "clip"):
                    self.optimize_clip_var.set(max(0.001, float(val)))
                elif key in ("optl2", "l2"):
                    self.optimize_l2_var.set(max(0.0, float(val)))
                else:
                    raise ValueError(f"unknown set key: {key}")
                self._log(f"Set {key}={val}")
                return
            raise ValueError(f"unknown command: {cmd}")
        except Exception as exc:
            messagebox.showerror("Command Error", str(exc))
            self._log(f"Command error: {exc}")

    def run_infer(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec = self._parse_input()
            x = torch.tensor([vec], dtype=torch.float32, device=self.device)
            mc = max(1, int(self.mc_samples_var.get()))
            as_probs = bool(self.as_probs_var.get())
            auto_mc = bool(self.auto_mc_var.get())
            auto_mc_threshold = max(0.0, float(self.auto_mc_threshold_var.get()))
            auto_mc_max = max(1, int(self.auto_mc_max_var.get()))
            out_t, std_t, auto_diag = self._predict_adaptive(
                x,
                mc_samples=mc,
                as_probs=as_probs,
                auto_mc=auto_mc,
                threshold=auto_mc_threshold,
                max_mc=auto_mc_max,
            )
            out = out_t.cpu().flatten().tolist()
            out_std = std_t.cpu().flatten().tolist()
            self._sync_out_dim(len(out))
            conf = self._confidence_summary(out, out_std, as_probs)

            k = min(max(1, int(self.topk_var.get())), len(out))
            topk = sorted(enumerate(out), key=lambda kv: kv[1], reverse=True)[:k]
            avg_unc = sum(out_std) / max(1, len(out_std))

            lines = []
            lines.append("Input:")
            lines.append(",".join(f"{v:.6f}" for v in vec))
            lines.append("")
            lines.append("Output:")
            lines.append(",".join(f"{v:.6f}" for v in out))
            lines.append("")
            lines.append("Uncertainty (std):")
            lines.append(",".join(f"{v:.6f}" for v in out_std))
            lines.append("")
            lines.append(f"Top-{k}:")
            for idx, val in topk:
                lines.append(f"  class {idx:02d} -> {val:.6f}")
            self._set_output("\n".join(lines))

            analytics = []
            analytics.append(f"Mode: {'probabilities' if as_probs else 'raw logits'}")
            analytics.append(
                f"MC samples: {auto_diag.get('selected_mc_samples', mc)} (base={mc}, auto={'on' if auto_mc else 'off'})"
            )
            analytics.append(f"Output min={min(out):.6f} max={max(out):.6f} mean={sum(out)/len(out):.6f}")
            analytics.append(f"Avg uncertainty={avg_unc:.6f}")
            analytics.append(
                f"Confidence={conf['label']} ({conf['score']:.3f}) margin={conf['margin']:.6f}"
            )
            if as_probs:
                analytics.append(f"Entropy={self._entropy(out):.6f}")
            self._set_analysis("\n".join(analytics))

            self._draw_bars(out, out_std)
            self._log(
                f"Inference done. out_dim={len(out)} mc={auto_diag.get('selected_mc_samples', mc)} probs={as_probs} conf={conf['label']}"
            )
            self.last_outputs = [out]
            self.last_stds = [out_std]
            hist = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "infer",
                "input": vec,
                "output": out,
                "std": out_std,
                "out_dim": len(out),
                "mc_samples": int(auto_diag.get("selected_mc_samples", mc)),
                "mc_samples_initial": mc,
                "as_probs": as_probs,
                "confidence": conf,
                "adaptive_mc": auto_diag,
            }
            self.last_payload = {"kind": "infer", "payload": hist}
            self._record_history(hist)
        except Exception as exc:
            messagebox.showerror("Inference Error", str(exc))
            self._log(f"Inference error: {exc}")

    def run_compare(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec_a = self._parse_input()
            vec_b = self._parse_input_b()
            mc = max(1, int(self.mc_samples_var.get()))
            as_probs = bool(self.as_probs_var.get())
            x = torch.tensor([vec_a, vec_b], dtype=torch.float32, device=self.device)
            out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
            out_a = out_t[0].detach().cpu().tolist()
            out_b = out_t[1].detach().cpu().tolist()
            std_a = std_t[0].detach().cpu().tolist()
            std_b = std_t[1].detach().cpu().tolist()
            self._sync_out_dim(len(out_a))
            delta = [b - a for a, b in zip(out_a, out_b)]

            k = min(max(1, int(self.topk_var.get())), len(delta))
            top_changes = sorted(enumerate(delta), key=lambda kv: abs(kv[1]), reverse=True)[:k]
            lines = [
                "Compare A -> B",
                "",
                "Input A:",
                ",".join(f"{v:.6f}" for v in vec_a),
                "Input B:",
                ",".join(f"{v:.6f}" for v in vec_b),
                "",
                "Delta Output (B - A):",
                ",".join(f"{v:.6f}" for v in delta),
                "",
                f"Top-{k} absolute shifts:",
            ]
            for idx, dv in top_changes:
                lines.append(f"  class {idx:02d} -> {dv:+.6f}")
            self._set_output("\n".join(lines))

            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            na = math.sqrt(sum(v * v for v in vec_a))
            nb = math.sqrt(sum(v * v for v in vec_b))
            cosine = dot / max(na * nb, 1e-12)
            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"MC samples: {mc}",
                f"Input cosine similarity={cosine:.6f}",
                f"Mean abs shift={sum(abs(v) for v in delta) / max(1, len(delta)):.6f}",
                f"Max abs shift={max(abs(v) for v in delta):.6f}",
                f"Avg uncertainty A={sum(std_a)/max(1, len(std_a)):.6f}",
                f"Avg uncertainty B={sum(std_b)/max(1, len(std_b)):.6f}",
            ]
            self._set_analysis("\n".join(analytics))
            self._draw_bars(delta, None)

            self.last_outputs = [out_a, out_b, delta]
            self.last_stds = [std_a, std_b]
            hist = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "compare",
                "input_a": vec_a,
                "input_b": vec_b,
                "delta": delta,
                "out_dim": len(delta),
                "mc_samples": mc,
                "as_probs": as_probs,
            }
            self.last_payload = {"kind": "compare", "payload": hist}
            self._record_history(hist)
            self._log(f"Compare done. out_dim={len(delta)} mc={mc} probs={as_probs}")
        except Exception as exc:
            messagebox.showerror("Compare Error", str(exc))
            self._log(f"Compare error: {exc}")

    def run_interpolate(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec_a = self._parse_input()
            vec_b = self._parse_input_b()
            steps = max(3, min(201, int(self.interp_steps_var.get())))
            mc = max(1, int(self.mc_samples_var.get()))
            as_probs = bool(self.as_probs_var.get())

            path_rows = []
            for i in range(steps):
                alpha = float(i / max(1, steps - 1))
                row = [((1.0 - alpha) * float(a)) + (alpha * float(b)) for a, b in zip(vec_a, vec_b)]
                path_rows.append(row)

            x = torch.tensor(path_rows, dtype=torch.float32, device=self.device)
            out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
            outs = out_t.detach().cpu().tolist()
            stds = std_t.detach().cpu().tolist()
            if outs:
                self._sync_out_dim(len(outs[0]))

            profile_rows = []
            for i, (out_row, std_row) in enumerate(zip(outs, stds)):
                top_idx = int(max(range(len(out_row)), key=lambda j: out_row[j]))
                top_score = float(out_row[top_idx])
                avg_unc = float(sum(abs(v) for v in std_row) / max(1, len(std_row)))
                profile_rows.append(
                    {
                        "step": int(i),
                        "alpha": float(i / max(1, steps - 1)),
                        "top_idx": top_idx,
                        "top_score": top_score,
                        "avg_uncertainty": avg_unc,
                    }
                )

            switch_points = []
            for i in range(1, len(profile_rows)):
                prev = profile_rows[i - 1]
                cur = profile_rows[i]
                if int(prev["top_idx"]) != int(cur["top_idx"]):
                    switch_points.append(
                        {
                            "from_top_idx": int(prev["top_idx"]),
                            "to_top_idx": int(cur["top_idx"]),
                            "from_step": int(prev["step"]),
                            "to_step": int(cur["step"]),
                            "alpha": float(cur["alpha"]),
                        }
                    )

            preview_cap = 120
            lines = ["Interpolation A -> B", "", "step,alpha,top_idx,top_score,avg_uncertainty"]
            for row in profile_rows[:preview_cap]:
                lines.append(
                    f"{row['step']},{row['alpha']:.4f},{row['top_idx']},{row['top_score']:.6f},{row['avg_uncertainty']:.6f}"
                )
            if len(profile_rows) > preview_cap:
                lines.append(f"... {len(profile_rows) - preview_cap} additional rows omitted ...")
            lines.append("")
            lines.append(f"Switch points: {len(switch_points)}")
            for sp in switch_points[:20]:
                lines.append(
                    f"  step {sp['from_step']}->{sp['to_step']} alpha={sp['alpha']:.4f} class {sp['from_top_idx']}->{sp['to_top_idx']}"
                )
            self._set_output("\n".join(lines))

            unc_vals = [float(r["avg_uncertainty"]) for r in profile_rows]
            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"MC samples: {mc}",
                f"Steps: {steps}",
                f"Top class start/end: {profile_rows[0]['top_idx']} -> {profile_rows[-1]['top_idx']}",
                f"Class switch count: {len(switch_points)}",
                f"Avg uncertainty min/max: {min(unc_vals):.6f} / {max(unc_vals):.6f}",
            ]
            if switch_points:
                analytics.append(
                    f"First switch at alpha={switch_points[0]['alpha']:.4f} ({switch_points[0]['from_top_idx']} -> {switch_points[0]['to_top_idx']})"
                )
            self._set_analysis("\n".join(analytics))

            draw_idx = len(profile_rows) // 2
            if switch_points:
                draw_idx = int(switch_points[0]["to_step"])
            self._draw_bars(outs[draw_idx], stds[draw_idx])

            self.last_outputs = outs
            self.last_stds = stds
            summary = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "interpolate",
                "input_a": vec_a,
                "input_b": vec_b,
                "steps": steps,
                "switch_count": len(switch_points),
                "switch_points": switch_points[:20],
                "start_top_idx": int(profile_rows[0]["top_idx"]),
                "end_top_idx": int(profile_rows[-1]["top_idx"]),
                "mc_samples": mc,
                "as_probs": as_probs,
                "out_dim": len(outs[0]) if outs else 0,
                "mid_output": outs[draw_idx],
                "mid_std": stds[draw_idx],
            }
            self.last_payload = {
                "kind": "interpolate",
                "summary": summary,
                "profile_rows": profile_rows,
            }
            self._record_history(summary)
            self._log(f"Interpolate done. steps={steps} switches={len(switch_points)} mc={mc} probs={as_probs}")
        except Exception as exc:
            messagebox.showerror("Interpolation Error", str(exc))
            self._log(f"Interpolation error: {exc}")

    def run_sensitivity_scan(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec = self._parse_input()
            mc = max(1, int(self.mc_samples_var.get()))
            as_probs = bool(self.as_probs_var.get())
            delta_base = max(0.0001, float(self.sensitivity_delta_var.get()))
            relative = bool(self.sensitivity_relative_var.get())

            candidates = [list(vec)]
            deltas = []
            for i, base_val in enumerate(vec):
                step = float(delta_base * max(1.0, abs(float(base_val)))) if relative else float(delta_base)
                deltas.append(step)
                plus = list(vec)
                minus = list(vec)
                plus[i] = float(plus[i]) + step
                minus[i] = float(minus[i]) - step
                candidates.append(plus)
                candidates.append(minus)

            x = torch.tensor(candidates, dtype=torch.float32, device=self.device)
            out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
            outs = out_t.detach().cpu().tolist()
            stds = std_t.detach().cpu().tolist()
            if outs:
                self._sync_out_dim(len(outs[0]))

            base_out = outs[0]
            base_top_idx = int(max(range(len(base_out)), key=lambda j: base_out[j]))
            base_top_score = float(base_out[base_top_idx])

            rows = []
            for i, step in enumerate(deltas):
                plus_idx = 1 + (2 * i)
                minus_idx = plus_idx + 1
                plus_out = outs[plus_idx]
                minus_out = outs[minus_idx]
                plus_std = stds[plus_idx]
                minus_std = stds[minus_idx]
                plus_top_idx = int(max(range(len(plus_out)), key=lambda j: plus_out[j]))
                minus_top_idx = int(max(range(len(minus_out)), key=lambda j: minus_out[j]))
                plus_score = float(plus_out[base_top_idx])
                minus_score = float(minus_out[base_top_idx])
                signed_sens = float((plus_score - minus_score) / max(2.0 * step, 1e-12))
                impact = float(0.5 * (abs(plus_score - base_top_score) + abs(minus_score - base_top_score)))
                local_unc = float(0.5 * (abs(plus_std[base_top_idx]) + abs(minus_std[base_top_idx])))
                rows.append(
                    {
                        "feature_idx": int(i),
                        "base_value": float(vec[i]),
                        "delta": float(step),
                        "impact": impact,
                        "signed_sensitivity": signed_sens,
                        "plus_top_idx": plus_top_idx,
                        "minus_top_idx": minus_top_idx,
                        "class_flip": bool((plus_top_idx != base_top_idx) or (minus_top_idx != base_top_idx)),
                        "local_uncertainty": local_unc,
                    }
                )

            ranked = sorted(rows, key=lambda r: (int(r["class_flip"]), r["impact"]), reverse=True)
            show_n = min(len(ranked), max(12, int(self.topk_var.get()) * 3))
            lines = [
                "Sensitivity Scan (around current input)",
                "",
                f"Base top class: {base_top_idx} score={base_top_score:.6f}",
                "feature,impact,signed_sens,delta,class_flip,plus_top,minus_top",
            ]
            for row in ranked[:show_n]:
                lines.append(
                    f"{row['feature_idx']},{row['impact']:.6f},{row['signed_sensitivity']:+.6f},"
                    f"{row['delta']:.6f},{int(row['class_flip'])},{row['plus_top_idx']},{row['minus_top_idx']}"
                )
            if len(ranked) > show_n:
                lines.append(f"... {len(ranked) - show_n} additional features omitted ...")
            self._set_output("\n".join(lines))

            signed_vec = [0.0 for _ in rows]
            impact_vec = [0.0 for _ in rows]
            for row in rows:
                idx = int(row["feature_idx"])
                signed_vec[idx] = float(row["signed_sensitivity"])
                impact_vec[idx] = float(row["impact"])

            flip_count = sum(1 for row in rows if bool(row["class_flip"]))
            mean_impact = sum(impact_vec) / max(1, len(impact_vec))
            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"MC samples: {mc}",
                f"Delta base: {delta_base:.6f} ({'relative' if relative else 'absolute'})",
                f"Feature count: {len(rows)}",
                f"Class flips: {flip_count}",
                f"Mean impact: {mean_impact:.6f}",
            ]
            if ranked:
                top = ranked[0]
                analytics.append(
                    f"Most sensitive feature: idx={top['feature_idx']} impact={top['impact']:.6f} sens={top['signed_sensitivity']:+.6f}"
                )
            self._set_analysis("\n".join(analytics))
            self._draw_bars(signed_vec, impact_vec)

            self.last_outputs = [signed_vec]
            self.last_stds = [impact_vec]
            summary = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "sensitivity",
                "input": vec,
                "base_top_idx": base_top_idx,
                "base_top_score": base_top_score,
                "delta_base": delta_base,
                "delta_mode": "relative" if relative else "absolute",
                "feature_count": len(rows),
                "class_flip_count": flip_count,
                "mc_samples": mc,
                "as_probs": as_probs,
                "top_features": ranked[:20],
                "signed_sensitivity": signed_vec,
                "impact_vector": impact_vec,
                "out_dim": len(signed_vec),
            }
            self.last_payload = {"kind": "sensitivity", "payload": summary}
            self._record_history(summary)
            self._log(
                f"Sensitivity scan done. features={len(rows)} flips={flip_count} mc={mc} probs={as_probs} relative={relative}"
            )
        except Exception as exc:
            messagebox.showerror("Sensitivity Error", str(exc))
            self._log(f"Sensitivity scan error: {exc}")

    def run_mc_profile(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec = self._parse_input()
            grid = self._parse_mc_grid()
            as_probs = bool(self.as_probs_var.get())
            x = torch.tensor([vec], dtype=torch.float32, device=self.device)
            rows = []
            last_out = []
            last_std = []
            for mc in grid:
                out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
                out = out_t[0].detach().cpu().tolist()
                std = std_t[0].detach().cpu().tolist()
                self._sync_out_dim(len(out))
                top_idx = int(max(range(len(out)), key=lambda i: out[i]))
                top_val = float(out[top_idx])
                avg_unc = float(sum(std) / max(1, len(std)))
                ent = self._entropy(out) if as_probs else float("nan")
                rows.append(
                    {
                        "mc": int(mc),
                        "top_idx": top_idx,
                        "top_score": top_val,
                        "avg_uncertainty": avg_unc,
                        "entropy": ent,
                    }
                )
                last_out = out
                last_std = std

            lines = ["MC Profile", "", "mc,top_idx,top_score,avg_uncertainty,entropy"]
            for r in rows:
                ent_txt = f"{r['entropy']:.6f}" if not math.isnan(float(r["entropy"])) else "n/a"
                lines.append(
                    f"{r['mc']},{r['top_idx']},{r['top_score']:.6f},{r['avg_uncertainty']:.6f},{ent_txt}"
                )
            self._set_output("\n".join(lines))

            unc_start = rows[0]["avg_uncertainty"]
            unc_end = rows[-1]["avg_uncertainty"]
            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"Grid: {','.join(str(v) for v in grid)}",
                f"Top class start/end: {rows[0]['top_idx']} -> {rows[-1]['top_idx']}",
                f"Avg uncertainty start/end: {unc_start:.6f} -> {unc_end:.6f}",
                f"Uncertainty delta (end-start): {unc_end - unc_start:+.6f}",
            ]
            self._set_analysis("\n".join(analytics))
            if last_out:
                self._draw_bars(last_out, last_std)

            self.last_outputs = [last_out] if last_out else []
            self.last_stds = [last_std] if last_std else []
            payload = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "mc_profile",
                "grid": grid,
                "rows": rows,
                "out_dim": len(last_out),
                "as_probs": as_probs,
                "input": vec,
            }
            self.last_payload = {"kind": "mc_profile", "payload": payload}
            self._record_history(payload)
            self._log(f"MC profile done. grid={grid} probs={as_probs}")
        except Exception as exc:
            messagebox.showerror("MC Profile Error", str(exc))
            self._log(f"MC profile error: {exc}")

    def run_mutation_lab(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            base = self._parse_input()
            as_probs = bool(self.as_probs_var.get())
            mc = max(1, int(self.mc_samples_var.get()))
            n = max(4, min(256, int(self.mutation_count_var.get())))
            scale = max(0.001, float(self.mutation_scale_var.get()))

            candidates = [list(base)]
            for _ in range(n):
                row = []
                for v in base:
                    noise = random.gauss(0.0, scale)
                    row.append(float(v) + float(noise))
                candidates.append(row)

            x = torch.tensor(candidates, dtype=torch.float32, device=self.device)
            out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
            outs = out_t.detach().cpu().tolist()
            stds = std_t.detach().cpu().tolist()
            if outs:
                self._sync_out_dim(len(outs[0]))

            def _row_score(out_row, std_row):
                top_val = max(out_row)
                avg_unc = sum(abs(v) for v in std_row) / max(1, len(std_row))
                return float(top_val) - (0.25 * float(avg_unc))

            scored = []
            for i in range(len(candidates)):
                out_row = outs[i]
                std_row = stds[i]
                top_idx = int(max(range(len(out_row)), key=lambda j: out_row[j]))
                score = _row_score(out_row, std_row)
                dist = math.sqrt(sum((candidates[i][j] - base[j]) ** 2 for j in range(len(base))))
                scored.append(
                    {
                        "idx": i,
                        "score": score,
                        "top_idx": top_idx,
                        "top_val": float(out_row[top_idx]),
                        "avg_uncertainty": float(sum(abs(v) for v in std_row) / max(1, len(std_row))),
                        "distance_l2": float(dist),
                    }
                )

            scored.sort(key=lambda r: r["score"], reverse=True)
            best = scored[0]
            best_idx = int(best["idx"])
            best_input = candidates[best_idx]
            best_out = outs[best_idx]
            best_std = stds[best_idx]

            base_score = next((r["score"] for r in scored if int(r["idx"]) == 0), scored[-1]["score"])
            lift = float(best["score"] - base_score)
            topk = min(10, len(scored))

            lines = [
                "Mutation Lab (uncertainty-aware)",
                "",
                f"Candidates: {len(candidates)}",
                f"Best score: {best['score']:.6f} | Baseline score: {base_score:.6f} | Lift: {lift:+.6f}",
                f"Best top class: {best['top_idx']} value={best['top_val']:.6f}",
                f"Best uncertainty: {best['avg_uncertainty']:.6f} | L2 distance: {best['distance_l2']:.6f}",
                "",
                "Top candidates (idx,score,class,val,unc,l2):",
            ]
            for row in scored[:topk]:
                lines.append(
                    f"  {row['idx']:>3},{row['score']:.6f},{row['top_idx']},{row['top_val']:.6f},{row['avg_uncertainty']:.6f},{row['distance_l2']:.6f}"
                )
            self._set_output("\n".join(lines))

            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"MC samples: {mc}",
                f"Mutation count: {n}",
                f"Mutation scale: {scale:.4f}",
                f"Best score lift: {lift:+.6f}",
            ]
            self._set_analysis("\n".join(analytics))
            self._draw_bars(best_out, best_std)

            self.last_outputs = [best_out]
            self.last_stds = [best_std]
            payload = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "mutation_lab",
                "base_input": base,
                "best_input": best_input,
                "best_score": float(best["score"]),
                "base_score": float(base_score),
                "score_lift": float(lift),
                "top_candidates": scored[:topk],
                "mc_samples": mc,
                "as_probs": as_probs,
                "out_dim": len(best_out),
            }
            self.last_payload = {"kind": "mutation_lab", "payload": payload}
            self._record_history(payload)
            self._log(f"Mutation lab done. candidates={len(candidates)} lift={lift:+.4f} mc={mc}")
        except Exception as exc:
            messagebox.showerror("Mutation Lab Error", str(exc))
            self._log(f"Mutation lab error: {exc}")

    def run_input_optimizer(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec = self._parse_input()
            as_probs = bool(self.as_probs_var.get())
            mc = max(1, int(self.mc_samples_var.get()))
            target = max(0, int(self.target_class_var.get()))
            steps = max(1, int(self.optimize_steps_var.get()))
            lr = max(0.0001, float(self.optimize_lr_var.get()))
            clip = max(0.001, abs(float(self.optimize_clip_var.get())))
            l2 = max(0.0, float(self.optimize_l2_var.get()))

            base_x = torch.tensor([vec], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                base_out = self.model(base_x)
                if base_out.dim() == 1:
                    base_out = base_out.unsqueeze(0)
                probe = torch.softmax(base_out, dim=-1) if as_probs else base_out
                self._sync_out_dim(int(probe.shape[-1]))
                if target >= int(probe.shape[-1]):
                    target = int(probe.shape[-1]) - 1
                    self.target_class_var.set(target)
                base_score = float(probe[0, target].item())

            x = base_x.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)
            best_score = float(base_score)
            best_vec = list(vec)
            score_curve = [float(base_score)]

            self.model.eval()
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                y = self.model(x)
                if y.dim() == 1:
                    y = y.unsqueeze(0)
                pred = torch.softmax(y, dim=-1) if as_probs else y
                score = pred[0, target]
                loss = -score + (l2 * torch.mean(x * x))
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-clip, clip)
                    y_now = self.model(x)
                    if y_now.dim() == 1:
                        y_now = y_now.unsqueeze(0)
                    pred_now = torch.softmax(y_now, dim=-1) if as_probs else y_now
                    score_now = float(pred_now[0, target].item())
                    score_curve.append(score_now)
                    if score_now > best_score:
                        best_score = score_now
                        best_vec = x.detach().cpu().flatten().tolist()

            self.optimized_input = [float(v) for v in best_vec]
            final_x = torch.tensor([self.optimized_input], dtype=torch.float32, device=self.device)
            out_t, std_t = self._predict(final_x, mc_samples=mc, as_probs=as_probs)
            out = out_t[0].detach().cpu().tolist()
            std = std_t[0].detach().cpu().tolist()
            self._sync_out_dim(len(out))

            ranked = sorted(enumerate(out), key=lambda kv: kv[1], reverse=True)
            rank_map = {int(idx): i + 1 for i, (idx, _val) in enumerate(ranked)}
            target_rank = int(rank_map.get(int(target), len(out)))

            lines = [
                "Targeted Input Optimization",
                "",
                f"Target class: {target}",
                f"Objective mode: {'probability' if as_probs else 'logit'}",
                f"Steps: {steps} | LR: {lr:.6f} | Clip: +/-{clip:.4f} | L2: {l2:.6f}",
                f"Target score (base -> best): {base_score:.6f} -> {best_score:.6f} ({best_score - base_score:+.6f})",
                "",
                "Optimized input:",
                ",".join(f"{v:.6f}" for v in self.optimized_input),
                "",
                f"Target class rank after optimization: {target_rank}/{len(out)}",
            ]
            topk = min(max(1, int(self.topk_var.get())), len(out))
            lines.append(f"Top-{topk} outputs:")
            for idx, val in ranked[:topk]:
                lines.append(f"  class {idx:02d} -> {val:.6f}")
            self._set_output("\n".join(lines))

            analytics = [
                f"Mode: {'probabilities' if as_probs else 'raw logits'}",
                f"MC samples: {mc}",
                f"Target class: {target}",
                f"Target score lift: {best_score - base_score:+.6f}",
                f"Best target score: {best_score:.6f}",
                f"Optimization curve start/end: {score_curve[0]:.6f} -> {score_curve[-1]:.6f}",
            ]
            self._set_analysis("\n".join(analytics))
            self._draw_bars(out, std)

            self.last_outputs = [out]
            self.last_stds = [std]
            payload = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "input_optimize",
                "target_class": int(target),
                "objective_mode": "probability" if as_probs else "logit",
                "steps": int(steps),
                "learning_rate": float(lr),
                "clip": float(clip),
                "l2": float(l2),
                "base_input": vec,
                "optimized_input": self.optimized_input,
                "base_score": float(base_score),
                "best_score": float(best_score),
                "score_lift": float(best_score - base_score),
                "target_rank": int(target_rank),
                "score_curve_preview": score_curve[:20],
                "mc_samples": int(mc),
                "as_probs": bool(as_probs),
                "out_dim": len(out),
            }
            self.last_payload = {"kind": "input_optimize", "payload": payload}
            self._record_history(payload)
            self._log(
                f"Input optimization done. target={target} lift={best_score - base_score:+.6f} steps={steps} lr={lr:.4f}"
            )
        except Exception as exc:
            messagebox.showerror("Input Optimizer Error", str(exc))
            self._log(f"Input optimizer error: {exc}")

    def use_optimized_input(self):
        if not self.optimized_input:
            messagebox.showinfo("Optimized Input", "Run Optimize Input first.")
            return
        self.input_var.set(",".join(f"{float(v):.6f}" for v in self.optimized_input))
        self._log("Optimized input moved into Input Vector field.")

    def _draw_bars(self, values: list[float], stds: list[float] | None = None):
        self.canvas.delete("all")
        if not values:
            return
        w = int(self.canvas.winfo_width() or 620)
        h = int(self.canvas.winfo_height() or 230)
        pad = 24
        n = len(values)
        bar_w = max(6, (w - 2 * pad) // max(1, n))
        max_abs = max(max(abs(v) for v in values), 1e-6)
        mid = h // 2
        self.canvas.create_line(pad, mid, w - pad, mid, fill="#334155")

        for i, v in enumerate(values):
            x0 = pad + i * bar_w + 1
            x1 = x0 + bar_w - 2
            y = int((abs(v) / max_abs) * (h * 0.40))
            if v >= 0:
                y0, y1 = mid - y, mid
                fill = "#3b82f6"
            else:
                y0, y1 = mid, mid + y
                fill = "#ef4444"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, width=0)
            if stds is not None and i < len(stds):
                unc = abs(stds[i]) / max_abs * (h * 0.40)
                cy = y0 if v >= 0 else y1
                self.canvas.create_line((x0 + x1) // 2, int(cy - unc), (x0 + x1) // 2, int(cy + unc), fill="#f59e0b", width=1)

    def _run_batch_rows(self, rows: list[list[float]], source: str):
        if self.model is None:
            raise ValueError("Load a model first.")
        x = torch.tensor(rows, dtype=torch.float32, device=self.device)
        mc = max(1, int(self.mc_samples_var.get()))
        as_probs = bool(self.as_probs_var.get())
        out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
        out = out_t.cpu().tolist()
        out_std = std_t.cpu().tolist()
        if out:
            self._sync_out_dim(len(out[0]))
        self.last_outputs = out
        self.last_stds = out_std
        preview = out[: min(10, len(out))]
        text = f"Rows: {len(out)}\nMode: {'probabilities' if as_probs else 'raw logits'}\nMC samples: {mc}\nPreview:\n"
        for i, row in enumerate(preview):
            text += f"{i+1:>3}: " + ",".join(f"{v:.6f}" for v in row) + "\n"
        self._set_output(text)
        if preview:
            self._draw_bars(preview[0], out_std[0] if out_std else None)

        out_col = torch.tensor(out, dtype=torch.float32)
        std_col = torch.tensor(out_std, dtype=torch.float32)
        self.last_batch_summary = self._build_batch_summary(out, out_std, source=source, as_probs=as_probs, mc=mc)
        analytics = []
        analytics.append(f"Batch rows: {len(out)}")
        analytics.append(f"Output mean={out_col.mean().item():.6f} std={out_col.std(unbiased=False).item():.6f}")
        analytics.append(f"Uncertainty mean={std_col.mean().item():.6f} max={std_col.max().item():.6f}")
        if out:
            row0 = out[0]
            top_idx = max(range(len(row0)), key=lambda i: row0[i])
            analytics.append(f"Top class row0: idx={top_idx} score={row0[top_idx]:.6f}")
        flagged = int(self.last_batch_summary.get("flagged_count", 0))
        analytics.append(f"Flagged rows (low margin / high uncertainty): {flagged}")
        self._set_analysis("\n".join(analytics))

        self._log(f"Batch inference done. rows={len(out)} mc={mc} probs={as_probs} source={source}")
        hist = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": "batch",
            "rows": len(out),
            "out_dim": len(out[0]) if out else 0,
            "mc_samples": mc,
            "as_probs": as_probs,
            "source": source,
        }
        self.last_payload = {
            "kind": "batch",
            "source": source,
            "rows": rows,
            "outputs": out,
            "stds": out_std,
            "summary": self.last_batch_summary,
            "mc_samples": mc,
            "as_probs": as_probs,
        }
        self._record_history(hist)

    def _build_batch_summary(self, out: list[list[float]], out_std: list[list[float]], source: str, as_probs: bool, mc: int):
        row_stats = []
        class_counts: dict[int, int] = {}
        margins = []
        uncertainties = []

        for i, row in enumerate(out):
            if not row:
                continue
            ranked = sorted(enumerate(row), key=lambda kv: kv[1], reverse=True)
            top_idx = int(ranked[0][0])
            top_val = float(ranked[0][1])
            second_val = float(ranked[1][1]) if len(ranked) > 1 else float(top_val)
            margin = float(top_val - second_val)
            std_row = out_std[i] if i < len(out_std) else []
            avg_unc = float(sum(abs(v) for v in std_row) / max(1, len(std_row)))
            class_counts[top_idx] = int(class_counts.get(top_idx, 0) + 1)
            margins.append(margin)
            uncertainties.append(avg_unc)
            row_stats.append(
                {
                    "row_index": int(i),
                    "top_idx": top_idx,
                    "top_score": top_val,
                    "margin": margin,
                    "avg_uncertainty": avg_unc,
                }
            )

        if not row_stats:
            return {
                "source": source,
                "rows": 0,
                "as_probs": bool(as_probs),
                "mc_samples": int(mc),
                "row_stats": [],
                "flagged_count": 0,
                "class_counts": {},
            }

        mean_margin = float(sum(margins) / max(1, len(margins)))
        mean_unc = float(sum(uncertainties) / max(1, len(uncertainties)))
        std_unc = math.sqrt(
            sum((u - mean_unc) ** 2 for u in uncertainties) / max(1, len(uncertainties))
        )
        low_margin_thr = float(max(0.0, mean_margin * 0.35))
        high_unc_thr = float(mean_unc + std_unc)

        flagged_rows = []
        for row in row_stats:
            low_margin = bool(float(row["margin"]) <= low_margin_thr)
            high_unc = bool(float(row["avg_uncertainty"]) >= high_unc_thr)
            row["low_margin"] = low_margin
            row["high_uncertainty"] = high_unc
            if low_margin or high_unc:
                flagged_rows.append(row)

        return {
            "source": source,
            "rows": int(len(row_stats)),
            "as_probs": bool(as_probs),
            "mc_samples": int(mc),
            "mean_margin": mean_margin,
            "mean_uncertainty": mean_unc,
            "low_margin_threshold": low_margin_thr,
            "high_uncertainty_threshold": high_unc_thr,
            "flagged_count": int(len(flagged_rows)),
            "flagged_rows": flagged_rows,
            "class_counts": class_counts,
            "row_stats": row_stats,
        }

    def show_batch_insights(self):
        summary = self.last_batch_summary or {}
        rows = summary.get("row_stats", [])
        if not rows:
            messagebox.showinfo("Batch Insights", "Run batch inference first.")
            return

        class_counts = summary.get("class_counts", {}) or {}
        class_lines = []
        total = max(1, int(summary.get("rows", 0)))
        for cls, count in sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]:
            pct = 100.0 * float(count) / float(total)
            class_lines.append(f"class {int(cls):02d}: {int(count)} ({pct:.1f}%)")

        flagged_rows = summary.get("flagged_rows", []) or []
        preview = flagged_rows[:20]
        lines = [
            "Batch Insights",
            "",
            f"Source: {summary.get('source', '')}",
            f"Rows: {summary.get('rows', 0)}",
            f"Mode: {'probabilities' if summary.get('as_probs') else 'raw logits'}",
            f"MC samples: {summary.get('mc_samples', 1)}",
            f"Mean margin: {float(summary.get('mean_margin', 0.0)):.6f}",
            f"Mean uncertainty: {float(summary.get('mean_uncertainty', 0.0)):.6f}",
            (
                "Thresholds: "
                f"margin<={float(summary.get('low_margin_threshold', 0.0)):.6f}, "
                f"unc>={float(summary.get('high_uncertainty_threshold', 0.0)):.6f}"
            ),
            f"Flagged rows: {summary.get('flagged_count', 0)}",
            "",
            "Top class distribution:",
        ]
        lines.extend(class_lines if class_lines else ["(none)"])
        lines.append("")
        lines.append("Flagged rows preview (row,top,score,margin,unc,low_margin,high_unc):")
        for row in preview:
            lines.append(
                f"{row['row_index']},{row['top_idx']},{row['top_score']:.6f},{row['margin']:.6f},"
                f"{row['avg_uncertainty']:.6f},{int(bool(row.get('low_margin')))},{int(bool(row.get('high_uncertainty')))}"
            )
        if len(flagged_rows) > len(preview):
            lines.append(f"... {len(flagged_rows) - len(preview)} additional flagged rows omitted ...")

        text = "\n".join(lines)
        self._set_analysis(text)
        self._set_output(text)
        self._log(f"Batch insights generated. flagged={summary.get('flagged_count', 0)}")

    def export_batch_insights_csv(self):
        summary = self.last_batch_summary or {}
        rows = summary.get("row_stats", [])
        if not rows:
            messagebox.showinfo("Export Insights", "Run batch inference first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    "row_index",
                    "top_idx",
                    "top_score",
                    "margin",
                    "avg_uncertainty",
                    "low_margin",
                    "high_uncertainty",
                ]
            )
            for row in rows:
                wr.writerow(
                    [
                        int(row.get("row_index", 0)),
                        int(row.get("top_idx", -1)),
                        float(row.get("top_score", 0.0)),
                        float(row.get("margin", 0.0)),
                        float(row.get("avg_uncertainty", 0.0)),
                        int(bool(row.get("low_margin", False))),
                        int(bool(row.get("high_uncertainty", False))),
                    ]
                )
        self._log(f"Batch insights exported: {path}")

    def run_batch_from_saved(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        path = self.batch_csv_var.get().strip()
        if not path:
            messagebox.showinfo("Batch Source", "Set a batch CSV path first.")
            return
        try:
            rows = self._load_rows_from_csv(path)
            self._run_batch_rows(rows, source=path)
        except Exception as exc:
            messagebox.showerror("Batch Error", str(exc))
            self._log(f"Batch error: {exc}")

    def run_batch(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.batch_csv_var.set(path)
        try:
            rows = self._load_rows_from_csv(path)
            self._run_batch_rows(rows, source=path)
        except Exception as exc:
            messagebox.showerror("Batch Error", str(exc))
            self._log(f"Batch error: {exc}")

    def save_outputs(self):
        if not getattr(self, "last_outputs", None):
            messagebox.showinfo("No Data", "Run infer or batch first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            out_dim = len(self.last_outputs[0]) if self.last_outputs else 0
            has_std = bool(self.last_stds) and len(self.last_stds[0]) == out_dim
            header = [f"out_{i}" for i in range(out_dim)]
            if has_std:
                header += [f"std_{i}" for i in range(out_dim)]
            wr.writerow(header)
            for i, row in enumerate(self.last_outputs):
                vals = list(row)
                if has_std:
                    vals += list(self.last_stds[i])
                wr.writerow(vals)
        self._log(f"Saved outputs to {path}")

    def run_benchmark(self):
        if self.model is None or self.in_dim is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        runs = max(1, int(self.bench_runs_var.get()))
        x = torch.randn(8, self.in_dim, device=self.device)
        with torch.no_grad():
            self.model.eval()
            for _ in range(3):
                _ = self.model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = self.model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms = elapsed * 1000.0 / runs
        throughput = (runs * x.shape[0]) / max(elapsed, 1e-9)
        self._log(f"Benchmark: {ms:.4f} ms/run | throughput={throughput:.2f} samples/sec")
        self._set_analysis(
            f"Benchmark runs: {runs}\n"
            f"Latency: {ms:.4f} ms/run\n"
            f"Throughput: {throughput:.2f} samples/sec\n"
            f"Batch size: {x.shape[0]}"
        )
        self._record_history(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "benchmark",
                "out_dim": 0,
                "runs": runs,
                "latency_ms": ms,
                "throughput_sps": throughput,
            }
        )
        self.last_payload = {
            "kind": "benchmark",
            "runs": runs,
            "latency_ms": ms,
            "throughput_sps": throughput,
            "batch_size": int(x.shape[0]),
        }
        messagebox.showinfo("Benchmark", f"{ms:.4f} ms/run\n{throughput:.2f} samples/sec")

    def recall_selected_history(self):
        sel = self.history_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.history):
            return
        item = self.history[idx]
        kind = str(item.get("kind", "unknown"))

        if item.get("input"):
            self.input_var.set(",".join(str(v) for v in item["input"]))
        if item.get("input_a"):
            self.input_var.set(",".join(str(v) for v in item["input_a"]))
        if item.get("input_b"):
            self.input_b_var.set(",".join(str(v) for v in item["input_b"]))
        if kind == "input_optimize" and item.get("optimized_input"):
            self.input_var.set(",".join(str(v) for v in item["optimized_input"]))
        if kind == "interpolate" and item.get("steps") is not None:
            self.interp_steps_var.set(max(3, min(201, int(item["steps"]))))
        if kind == "sensitivity" and item.get("delta_base") is not None:
            self.sensitivity_delta_var.set(max(0.0001, float(item["delta_base"])))
            self.sensitivity_relative_var.set(str(item.get("delta_mode", "absolute")) == "relative")
        if kind == "input_optimize" and item.get("target_class") is not None:
            self.target_class_var.set(max(0, int(item["target_class"])))

        if kind == "infer" and item.get("output"):
            self._draw_bars(item["output"], item.get("std"))
        elif kind == "compare" and item.get("delta"):
            self._draw_bars(item["delta"], None)
        elif kind == "interpolate" and item.get("mid_output"):
            self._draw_bars(item["mid_output"], item.get("mid_std"))
        elif kind == "sensitivity" and item.get("signed_sensitivity"):
            std_hint = item.get("impact_vector") if isinstance(item.get("impact_vector"), list) else None
            self._draw_bars(item["signed_sensitivity"], std_hint)
        elif kind == "input_optimize" and item.get("optimized_input"):
            self.optimized_input = list(item["optimized_input"])

        self._set_output(json.dumps(item, indent=2))
        self._set_analysis(f"Recalled {kind} item from history.")
        self.last_payload = {"kind": "history_recall", "payload": item}
        self._log(f"Recalled history item #{idx+1} ({kind})")

    def clear_history(self):
        self.history = []
        self._refresh_history_list()
        self._log("History cleared.")

    def export_history_json(self):
        if not self.history:
            messagebox.showinfo("No History", "History is empty.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        Path(path).write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        self._log(f"History exported: {path}")

    def copy_result_to_clipboard(self):
        text = self.output_box.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showinfo("Copy Result", "No result to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._log("Copied current result text to clipboard.")

    def save_snapshot(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_var.get(),
            "dsl_path": self.dsl_var.get(),
            "batch_csv_path": self.batch_csv_var.get(),
            "input_vector": self.input_var.get(),
            "input_vector_b": self.input_b_var.get(),
            "bench_runs": int(self.bench_runs_var.get()),
            "mc_samples": int(self.mc_samples_var.get()),
            "auto_mc": bool(self.auto_mc_var.get()),
            "auto_mc_threshold": float(self.auto_mc_threshold_var.get()),
            "auto_mc_max": int(self.auto_mc_max_var.get()),
            "mc_grid": self.mc_grid_var.get(),
            "target_class": int(self.target_class_var.get()),
            "optimize_steps": int(self.optimize_steps_var.get()),
            "optimize_lr": float(self.optimize_lr_var.get()),
            "optimize_clip": float(self.optimize_clip_var.get()),
            "optimize_l2": float(self.optimize_l2_var.get()),
            "mutation_count": int(self.mutation_count_var.get()),
            "mutation_scale": float(self.mutation_scale_var.get()),
            "interp_steps": int(self.interp_steps_var.get()),
            "sensitivity_delta": float(self.sensitivity_delta_var.get()),
            "sensitivity_relative": bool(self.sensitivity_relative_var.get()),
            "topk": int(self.topk_var.get()),
            "as_probs": bool(self.as_probs_var.get()),
            "model_info": self.model_info_var.get(),
            "optimized_input": self.optimized_input,
            "last_batch_summary": self.last_batch_summary,
            "last_outputs": self.last_outputs,
            "last_stds": self.last_stds,
            "last_payload": self.last_payload,
            "history": self.history[-200:],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Snapshot saved: {path}")

    def load_snapshot(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.model_var.set(str(data.get("model_path", self.model_var.get())))
            self.dsl_var.set(str(data.get("dsl_path", self.dsl_var.get())))
            self.batch_csv_var.set(str(data.get("batch_csv_path", self.batch_csv_var.get())))
            self.input_var.set(str(data.get("input_vector", self.input_var.get())))
            self.input_b_var.set(str(data.get("input_vector_b", self.input_b_var.get())))
            self.bench_runs_var.set(int(data.get("bench_runs", self.bench_runs_var.get())))
            self.mc_samples_var.set(int(data.get("mc_samples", self.mc_samples_var.get())))
            self.auto_mc_var.set(bool(data.get("auto_mc", self.auto_mc_var.get())))
            self.auto_mc_threshold_var.set(float(data.get("auto_mc_threshold", self.auto_mc_threshold_var.get())))
            self.auto_mc_max_var.set(int(data.get("auto_mc_max", self.auto_mc_max_var.get())))
            self.mc_grid_var.set(str(data.get("mc_grid", self.mc_grid_var.get())))
            self.target_class_var.set(int(data.get("target_class", self.target_class_var.get())))
            self.optimize_steps_var.set(int(data.get("optimize_steps", self.optimize_steps_var.get())))
            self.optimize_lr_var.set(float(data.get("optimize_lr", self.optimize_lr_var.get())))
            self.optimize_clip_var.set(float(data.get("optimize_clip", self.optimize_clip_var.get())))
            self.optimize_l2_var.set(float(data.get("optimize_l2", self.optimize_l2_var.get())))
            self.mutation_count_var.set(int(data.get("mutation_count", self.mutation_count_var.get())))
            self.mutation_scale_var.set(float(data.get("mutation_scale", self.mutation_scale_var.get())))
            self.interp_steps_var.set(int(data.get("interp_steps", self.interp_steps_var.get())))
            self.sensitivity_delta_var.set(float(data.get("sensitivity_delta", self.sensitivity_delta_var.get())))
            self.sensitivity_relative_var.set(bool(data.get("sensitivity_relative", self.sensitivity_relative_var.get())))
            self.topk_var.set(int(data.get("topk", self.topk_var.get())))
            self.as_probs_var.set(bool(data.get("as_probs", self.as_probs_var.get())))
            self.optimized_input = data.get("optimized_input", self.optimized_input)
            self.last_batch_summary = data.get("last_batch_summary", self.last_batch_summary) or {}
            self.last_outputs = data.get("last_outputs", []) or []
            self.last_stds = data.get("last_stds", []) or []
            self.last_payload = data.get("last_payload", {}) or {}
            self.history = data.get("history", []) or []
            self.model_info_var.set(str(data.get("model_info", self.model_info_var.get())))
            self._refresh_history_list()
            if self.last_outputs and isinstance(self.last_outputs[0], list):
                std0 = self.last_stds[0] if self.last_stds and isinstance(self.last_stds[0], list) else None
                self._draw_bars(self.last_outputs[0], std0)
            if self.last_payload:
                self._set_output(json.dumps(self.last_payload, indent=2))
            self._set_analysis("Snapshot loaded. Run Load Model if file paths changed.")
            self._log(f"Snapshot loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Load Snapshot Error", str(exc))
            self._log(f"Snapshot load error: {exc}")


def main():
    root = tk.Tk()
    app = ChampionGUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
