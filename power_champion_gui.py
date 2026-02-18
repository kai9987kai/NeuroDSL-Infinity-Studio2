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
    return model, in_dim


class ChampionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroDSL Power Champion GUI")
        self.root.geometry("1180x780")
        self.device = resolve_device("auto")
        self.model = None
        self.in_dim = None
        self.last_outputs = []
        self.last_stds = []
        self.last_payload = {}
        self.history = []

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.dsl_var = tk.StringVar(value=DEFAULT_DSL)
        self.input_var = tk.StringVar(value="")
        self.input_b_var = tk.StringVar(value="")
        self.batch_csv_var = tk.StringVar(value="")
        self.bench_runs_var = tk.IntVar(value=60)
        self.mc_samples_var = tk.IntVar(value=1)
        self.mc_grid_var = tk.StringVar(value="1,2,4,8")
        self.topk_var = tk.IntVar(value=5)
        self.as_probs_var = tk.BooleanVar(value=False)
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

        prof_row = ttk.Frame(right)
        prof_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(prof_row, text="MC Grid").pack(side=tk.LEFT)
        ttk.Entry(prof_row, textvariable=self.mc_grid_var, width=22).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(prof_row, text="MC Profile", command=self.run_mc_profile).pack(side=tk.LEFT)

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

    def _entropy(self, probs: list[float]) -> float:
        eps = 1e-12
        return float(-sum(p * math.log(max(p, eps)) for p in probs))

    def load_model(self):
        try:
            self.model, self.in_dim = load_model_from_files(self.model_var.get().strip(), self.dsl_var.get().strip(), self.device)
            self.status_var.set(f"Device: {self.device} | Model loaded | input_dim={self.in_dim}")
            self._log(f"Loaded model with input_dim={self.in_dim}")
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

    def run_infer(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Load a model first.")
            return
        try:
            vec = self._parse_input()
            x = torch.tensor([vec], dtype=torch.float32, device=self.device)
            mc = max(1, int(self.mc_samples_var.get()))
            as_probs = bool(self.as_probs_var.get())
            out_t, std_t = self._predict(x, mc_samples=mc, as_probs=as_probs)
            out = out_t.cpu().flatten().tolist()
            out_std = std_t.cpu().flatten().tolist()

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
            analytics.append(f"MC samples: {mc}")
            analytics.append(f"Output min={min(out):.6f} max={max(out):.6f} mean={sum(out)/len(out):.6f}")
            analytics.append(f"Avg uncertainty={avg_unc:.6f}")
            if as_probs:
                analytics.append(f"Entropy={self._entropy(out):.6f}")
            self._set_analysis("\n".join(analytics))

            self._draw_bars(out, out_std)
            self._log(f"Inference done. out_dim={len(out)} mc={mc} probs={as_probs}")
            self.last_outputs = [out]
            self.last_stds = [out_std]
            hist = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "kind": "infer",
                "input": vec,
                "output": out,
                "std": out_std,
                "out_dim": len(out),
                "mc_samples": mc,
                "as_probs": as_probs,
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
        analytics = []
        analytics.append(f"Batch rows: {len(out)}")
        analytics.append(f"Output mean={out_col.mean().item():.6f} std={out_col.std(unbiased=False).item():.6f}")
        analytics.append(f"Uncertainty mean={std_col.mean().item():.6f} max={std_col.max().item():.6f}")
        if out:
            row0 = out[0]
            top_idx = max(range(len(row0)), key=lambda i: row0[i])
            analytics.append(f"Top class row0: idx={top_idx} score={row0[top_idx]:.6f}")
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
            "mc_samples": mc,
            "as_probs": as_probs,
        }
        self._record_history(hist)

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
        if item.get("kind") == "infer" and item.get("input"):
            self.input_var.set(",".join(str(v) for v in item["input"]))
            if item.get("output"):
                self._draw_bars(item["output"], item.get("std"))
            self._set_output(json.dumps(item, indent=2))
            self._set_analysis("Recalled inference from history.")
            self.last_payload = {"kind": "history_recall", "payload": item}
            self._log(f"Recalled history item #{idx+1}")

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
            "mc_grid": self.mc_grid_var.get(),
            "topk": int(self.topk_var.get()),
            "as_probs": bool(self.as_probs_var.get()),
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
            self.mc_grid_var.set(str(data.get("mc_grid", self.mc_grid_var.get())))
            self.topk_var.set(int(data.get("topk", self.topk_var.get())))
            self.as_probs_var.set(bool(data.get("as_probs", self.as_probs_var.get())))
            self.last_outputs = data.get("last_outputs", []) or []
            self.last_stds = data.get("last_stds", []) or []
            self.last_payload = data.get("last_payload", {}) or {}
            self.history = data.get("history", []) or []
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
