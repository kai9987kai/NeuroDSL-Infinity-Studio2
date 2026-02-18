"""Unified GUI control center for NeuroDSL Infinity Studio."""

from __future__ import annotations

import os
import subprocess
import threading
from typing import List, Optional

import FreeSimpleGUI as sg

from device_utils import available_device_names, format_device_report


class ProcessRunner:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: List[str], window, tag: str):
        if self.is_running():
            window.write_event_value("-LOG-", (tag, "[warn] A process is already running. Stop it first."))
            return

        def _run():
            with self.lock:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    universal_newlines=True,
                )
            window.write_event_value("-LOG-", (tag, f"[cmd] {' '.join(cmd)}"))
            try:
                assert self.proc is not None
                for line in self.proc.stdout:
                    window.write_event_value("-LOG-", (tag, line.rstrip("\n")))
                rc = self.proc.wait()
            except Exception as exc:
                window.write_event_value("-LOG-", (tag, f"[error] {exc}"))
                rc = -1
            with self.lock:
                self.proc = None
            window.write_event_value("-DONE-", (tag, rc))

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self, window, tag: str = "SYSTEM"):
        with self.lock:
            proc = self.proc
        if proc is None:
            window.write_event_value("-LOG-", (tag, "[info] No active process."))
            return
        try:
            proc.terminate()
            window.write_event_value("-LOG-", (tag, "[info] Termination signal sent."))
        except Exception as exc:
            window.write_event_value("-LOG-", (tag, f"[error] Stop failed: {exc}"))


def _base_cmd() -> List[str]:
    return ["python", "omni_cli.py"]


def build_window():
    sg.theme("DarkGrey8")
    devices = ["auto"] + available_device_names(include_cpu=True)
    scripts = ["main", "verify", "functional", "run_model", "omni_gui"]

    tab_script = [
        [sg.Text("Script"), sg.Combo(scripts, default_value="verify", key="-SCRIPT-NAME-", readonly=True, size=(16, 1))],
        [sg.Text("Args"), sg.Input(key="-SCRIPT-ARGS-", size=(70, 1))],
        [sg.HorizontalSeparator()],
        [sg.Text("Web Model URL"), sg.Input("", key="-WEB-URL-", size=(65, 1))],
        [sg.Text("Save To"), sg.Input("downloads/web_model.pth", key="-WEB-OUT-", size=(55, 1)), sg.Button("Download Model", key="-DOWNLOAD-MODEL-")],
        [sg.HorizontalSeparator()],
        [sg.Text("AI Key (optional)"), sg.Input("", key="-AI-KEY-", password_char="*", size=(45, 1))],
        [sg.Text("AI Prompt"), sg.Input("Design a robust classifier with MoE and adaptive compute", key="-AI-PROMPT-", size=(80, 1))],
        [sg.Text("Autopilot In/Out/Cands"), sg.Input("32", key="-AUTO-IN-", size=(6, 1)), sg.Input("10", key="-AUTO-OUT-", size=(6, 1)), sg.Input("8", key="-AUTO-CANDS-", size=(6, 1))],
        [
            sg.Button("AI Generate DSL", key="-AI-GEN-DSL-", button_color=("white", "#805ad5")),
            sg.Button("AI Explain DSL", key="-AI-EXPLAIN-DSL-"),
            sg.Button("AI Autopilot", key="-AI-AUTOPILOT-"),
            sg.Button("Agent API Server", key="-AGENT-API-"),
            sg.Button("Platform Health", key="-PLATFORM-HEALTH-"),
        ],
        [sg.Text("Champion Model"), sg.Input("outputs/best_ai_run_sweep/best_final.pth", key="-CHAMP-MODEL-", size=(52, 1))],
        [sg.Text("Champion DSL"), sg.Input("outputs/best_ai_run_sweep/seed_53/best.dsl", key="-CHAMP-DSL-", size=(52, 1))],
        [sg.Text("Package Dir"), sg.Input("outputs/champion_package", key="-CHAMP-DIR-", size=(40, 1)), sg.Button("Build Champion Package", key="-BUILD-CHAMPION-")],
        [sg.Text("Bundle Include"), sg.Input("outputs parser_utils.py network.py trainer.py omni_cli.py omni_studio.py", key="-BUNDLE-INCLUDE-", size=(75, 1))],
        [sg.Text("Bundle Zip"), sg.Input("outputs/platform_bundle.zip", key="-BUNDLE-ZIP-", size=(45, 1)), sg.Button("Export Bundle", key="-EXPORT-BUNDLE-")],
        [sg.Button("Run Script", key="-RUN-SCRIPT-", button_color=("white", "#2b6cb0")), sg.Button("Stop", key="-STOP-")],
        [sg.Multiline(size=(110, 16), key="-LOG-SCRIPT-", autoscroll=True, background_color="#050505", text_color="#9de1ff")],
    ]

    tab_data_prep = [
        [sg.Text("Input CSV"), sg.Input(key="-DP-IN-", size=(60, 1)), sg.FileBrowse()],
        [sg.Text("Output CSV"), sg.Input(key="-DP-OUT-", size=(60, 1)), sg.SaveAs()],
        [sg.Checkbox("Normalize (Min-Max)", key="-DP-NORM-")],
        [sg.Text("Fill Missing Values"), sg.Combo(["", "mean", "median"], key="-DP-FILL-", readonly=True)],
        [sg.Button("Run Preprocessing", key="-DP-RUN-", button_color=("white", "#3182ce"))],
        [sg.Multiline(size=(110, 16), key="-LOG-DP-", autoscroll=True, background_color="#050505", text_color="#f0e68c")],
    ]

    tab_tabular = [
        [sg.Text("DSL"), sg.Input("[32,64], moe: [64,6,1], mod: [64,4,0.35], [64,10]", key="-TABULAR-DSL-", size=(95, 1))],
        [sg.Text("Epochs"), sg.Input("120", key="-TABULAR-EPOCHS-", size=(8, 1)), sg.Text("Loss"), sg.Combo(["MSE", "CrossEntropy", "Huber", "MAE"], default_value="MSE", key="-TABULAR-LOSS-", readonly=True, size=(14, 1))],
        [sg.Text("Save PTH"), sg.Input("outputs/tabular_model.pth", key="-TABULAR-SAVE-", size=(55, 1))],
        [sg.Text("Search In/Out"), sg.Input("32", key="-SEARCH-IN-", size=(6, 1)), sg.Input("10", key="-SEARCH-OUT-", size=(6, 1)), sg.Text("Trials"), sg.Input("16", key="-SEARCH-TRIALS-", size=(6, 1))],
        [sg.Text("Server Host"), sg.Input("127.0.0.1", key="-SERVER-HOST-", size=(14, 1)), sg.Text("Port"), sg.Input("8080", key="-SERVER-PORT-", size=(8, 1))],
        [
            sg.Button("Start Tabular Train", key="-RUN-TABULAR-", button_color=("white", "#2f855a")),
            sg.Button("Run Existing Model", key="-RUN-TABULAR-INFER-"),
            sg.Button("Search Architectures", key="-RUN-TABULAR-SEARCH-"),
            sg.Button("Start API Server", key="-RUN-TABULAR-SERVE-"),
        ],
        [sg.Multiline(size=(110, 16), key="-LOG-TABULAR-", autoscroll=True, background_color="#050505", text_color="#b9f6ca")],
    ]

    tab_image = [
        [sg.Text("Image Folder (optional)"), sg.Input("", key="-IMG-FOLDER-", size=(60, 1)), sg.FolderBrowse()],
        [sg.Text("Image Size"), sg.Input("32", key="-IMG-SIZE-", size=(8, 1)), sg.Text("Latent"), sg.Input("128", key="-IMG-LATENT-", size=(8, 1)), sg.Text("Epochs"), sg.Input("35", key="-IMG-EPOCHS-", size=(8, 1))],
        [sg.Text("Checkpoint"), sg.Input("outputs/image_model.pth", key="-IMG-CKPT-", size=(50, 1))],
        [sg.Text("Grid Output"), sg.Input("outputs/generated_grid.png", key="-IMG-GRID-", size=(50, 1))],
        [sg.Button("Train Image Mode", key="-RUN-IMAGE-TRAIN-", button_color=("white", "#805ad5")), sg.Button("Generate Uncanny Grid", key="-RUN-IMAGE-GEN-"), sg.Button("Interpolate Latents", key="-RUN-IMAGE-INTERP-")],
        [sg.Multiline(size=(110, 16), key="-LOG-IMAGE-", autoscroll=True, background_color="#050505", text_color="#f7c7ff")],
    ]

    tab_multi = [
        [sg.Text("Vec Dim"), sg.Input("16", key="-MM-VEC-", size=(8, 1)), sg.Text("Hidden"), sg.Input("128", key="-MM-HIDDEN-", size=(8, 1)), sg.Text("Out Dim"), sg.Input("8", key="-MM-OUT-", size=(8, 1)), sg.Text("Epochs"), sg.Input("40", key="-MM-EPOCHS-", size=(8, 1))],
        [sg.Text("Checkpoint"), sg.Input("outputs/multimodal_model.pth", key="-MM-CKPT-", size=(50, 1))],
        [sg.Text("Vector (optional)"), sg.Input("", key="-MM-VECTOR-", size=(60, 1))],
        [sg.Text("Text Prompt (optional)"), sg.Input("", key="-MM-TEXT-", size=(60, 1))],
        [sg.Button("Train Multimodal", key="-RUN-MM-TRAIN-", button_color=("white", "#d69e2e")), sg.Button("Run Multimodal", key="-RUN-MM-RUN-")],
        [sg.Multiline(size=(110, 16), key="-LOG-MM-", autoscroll=True, background_color="#050505", text_color="#fff3bf")],
    ]

    control_bar = [
        [
            sg.Text("Device"),
            sg.Combo(devices, default_value="auto", key="-DEVICE-", readonly=True, size=(10, 1)),
            sg.Button("Refresh Devices", key="-REFRESH-DEVICES-"),
            sg.Button("Show Device Report", key="-SHOW-DEVICES-"),
            sg.Push(),
            sg.Text("Hotkeys: F5 start current tab | F6 stop | Ctrl+R run script | Ctrl+T tabular | Ctrl+I image | Ctrl+M multimodal"),
        ]
    ]

    tabs = [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("Command Deck", tab_script),
                        sg.Tab("Data Preprocessing", tab_data_prep),
                        sg.Tab("Tabular Lab", tab_tabular),
                        sg.Tab("Image Mode", tab_image),
                        sg.Tab("Multimodal", tab_multi),
                    ]
                ],
                key="-TABGROUP-",
                enable_events=True,
                expand_x=True,
                expand_y=True,
            )
        ]
    ]

    layout = control_bar + tabs + [[sg.StatusBar("Omni Studio ready.", key="-STATUS-", size=(120, 1))]]
    window = sg.Window("NeuroDSL Omni Studio", layout, finalize=True, resizable=True, size=(1200, 780))
    window.bind("<F5>", "F5")
    window.bind("<F6>", "F6")
    window.bind("<Control-r>", "CTRL_R")
    window.bind("<Control-t>", "CTRL_T")
    window.bind("<Control-i>", "CTRL_I")
    window.bind("<Control-m>", "CTRL_M")
    return window


def append_log(window, tab: str, line: str):
    key = {
        "SCRIPT": "-LOG-SCRIPT-",
        "DATAPREP": "-LOG-DP-",
        "TABULAR": "-LOG-TABULAR-",
        "IMAGE": "-LOG-IMAGE-",
        "MULTI": "-LOG-MM-",
        "SYSTEM": "-LOG-SCRIPT-",
    }.get(tab, "-LOG-SCRIPT-")
    window[key].print(line)


def run_for_tab(window, runner: ProcessRunner, values, tab_name: str):
    device = values["-DEVICE-"]
    if tab_name == "Command Deck":
        cmd = _base_cmd() + ["run-script", "--script", values["-SCRIPT-NAME-"]]
        if values["-SCRIPT-ARGS-"].strip():
            cmd += ["--script-args", values["-SCRIPT-ARGS-"].strip()]
        runner.start(cmd, window, "SCRIPT")
        window["-STATUS-"].update("Running script...")
        return

    if tab_name == "Tabular Lab":
        cmd = _base_cmd() + [
            "tabular-train",
            "--dsl",
            values["-TABULAR-DSL-"],
            "--epochs",
            str(int(values["-TABULAR-EPOCHS-"])),
            "--loss",
            values["-TABULAR-LOSS-"],
            "--device",
            device,
            "--save-pth",
            values["-TABULAR-SAVE-"],
        ]
        runner.start(cmd, window, "TABULAR")
        window["-STATUS-"].update("Training tabular model...")
        return

    if tab_name == "Image Mode":
        cmd = _base_cmd() + [
            "image-train",
            "--image-size",
            str(int(values["-IMG-SIZE-"])),
            "--latent-dim",
            str(int(values["-IMG-LATENT-"])),
            "--epochs",
            str(int(values["-IMG-EPOCHS-"])),
            "--device",
            device,
            "--save-model",
            values["-IMG-CKPT-"],
        ]
        if values["-IMG-FOLDER-"].strip():
            cmd += ["--image-folder", values["-IMG-FOLDER-"].strip()]
        runner.start(cmd, window, "IMAGE")
        window["-STATUS-"].update("Training image model...")
        return

    if tab_name == "Multimodal":
        cmd = _base_cmd() + [
            "multimodal-train",
            "--vec-dim",
            str(int(values["-MM-VEC-"])),
            "--hidden-dim",
            str(int(values["-MM-HIDDEN-"])),
            "--out-dim",
            str(int(values["-MM-OUT-"])),
            "--epochs",
            str(int(values["-MM-EPOCHS-"])),
            "--device",
            device,
            "--save-model",
            values["-MM-CKPT-"],
        ]
        runner.start(cmd, window, "MULTI")
        window["-STATUS-"].update("Training multimodal model...")
        return


def main():
    window = build_window()
    runner = ProcessRunner()

    while True:
        event, values = window.read(timeout=150)
        if event == sg.WINDOW_CLOSED:
            if runner.is_running():
                runner.stop(window)
            break

        if event in ("-RUN-SCRIPT-", "CTRL_R"):
            run_for_tab(window, runner, values, "Command Deck")

        if event == "-DP-RUN-":
            cmd = _base_cmd() + [
                "preprocess",
                "--input-file",
                values["-DP-IN-"],
                "--output-file",
                values["-DP-OUT-"],
            ]
            if values["-DP-NORM-"]:
                cmd.append("--normalize")
            if values["-DP-FILL-"]:
                cmd.extend(["--fill-missing", values["-DP-FILL-"]])
            runner.start(cmd, window, "DATAPREP")

        if event == "-AI-GEN-DSL-":
            cmd = _base_cmd() + [
                "ai-dsl",
                "--prompt",
                values["-AI-PROMPT-"].strip(),
                "--output-dsl",
                "outputs/ai_generated.dsl",
            ]
            if values["-AI-KEY-"].strip():
                cmd += ["--api-key", values["-AI-KEY-"].strip()]
            runner.start(cmd, window, "SCRIPT")

        if event == "-AI-AUTOPILOT-":
            cmd = _base_cmd() + [
                "ai-autopilot",
                "--objective",
                values["-AI-PROMPT-"].strip(),
                "--input-dim",
                values["-AUTO-IN-"].strip(),
                "--output-dim",
                values["-AUTO-OUT-"].strip(),
                "--candidates",
                values["-AUTO-CANDS-"].strip(),
                "--device",
                values["-DEVICE-"],
                "--out-dir",
                "outputs/autopilot",
            ]
            if values["-AI-KEY-"].strip():
                cmd += ["--api-key", values["-AI-KEY-"].strip()]
            runner.start(cmd, window, "SCRIPT")

        if event == "-AGENT-API-":
            cmd = _base_cmd() + [
                "serve-agent-api",
                "--host",
                values.get("-SERVER-HOST-", "127.0.0.1").strip() or "127.0.0.1",
                "--port",
                values.get("-SERVER-PORT-", "8090").strip() or "8090",
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "SCRIPT")

        if event == "-PLATFORM-HEALTH-":
            cmd = _base_cmd() + [
                "platform-health",
                "--include-functional",
                "--include-codex",
                "--output-json",
                "outputs/platform_health.json",
            ]
            runner.start(cmd, window, "SCRIPT")

        if event == "-EXPORT-BUNDLE-":
            raw = values["-BUNDLE-INCLUDE-"].strip()
            includes = [p for p in raw.split(" ") if p]
            if not includes:
                append_log(window, "SCRIPT", "[warn] Provide include paths first.")
            else:
                cmd = _base_cmd() + ["export-bundle", "--include"] + includes + ["--output-zip", values["-BUNDLE-ZIP-"].strip()]
                runner.start(cmd, window, "SCRIPT")

        if event == "-BUILD-CHAMPION-":
            cmd = _base_cmd() + [
                "champion-package",
                "--model",
                values["-CHAMP-MODEL-"].strip(),
                "--dsl-file",
                values["-CHAMP-DSL-"].strip(),
                "--output-dir",
                values["-CHAMP-DIR-"].strip(),
                "--build-exe",
                "--install-pyinstaller",
                "--exe-name",
                "NeuroDSL_Champion",
            ]
            runner.start(cmd, window, "SCRIPT")

        if event == "-DOWNLOAD-MODEL-":
            if not values["-WEB-URL-"].strip():
                append_log(window, "SCRIPT", "[warn] Enter a URL first.")
            else:
                cmd = _base_cmd() + [
                    "download-model",
                    "--url",
                    values["-WEB-URL-"].strip(),
                    "--output",
                    values["-WEB-OUT-"].strip(),
                ]
                runner.start(cmd, window, "SCRIPT")

        if event in ("-RUN-TABULAR-", "CTRL_T"):
            run_for_tab(window, runner, values, "Tabular Lab")

        if event == "-RUN-TABULAR-INFER-":
            dsl = values["-TABULAR-DSL-"].strip()
            save = values["-TABULAR-SAVE-"].strip()
            cmd = _base_cmd() + [
                "tabular-run",
                "--model",
                save,
                "--dsl",
                dsl,
                "--input",
                "0.1,0.2,0.3,0.4",
                "--creative-samples",
                "3",
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "TABULAR")

        if event == "-RUN-TABULAR-SEARCH-":
            cmd = _base_cmd() + [
                "tabular-search",
                "--input-dim",
                values["-SEARCH-IN-"].strip(),
                "--output-dim",
                values["-SEARCH-OUT-"].strip(),
                "--trials",
                values["-SEARCH-TRIALS-"].strip(),
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "TABULAR")

        if event == "-RUN-TABULAR-SERVE-":
            cmd = _base_cmd() + [
                "serve-tabular",
                "--model",
                values["-TABULAR-SAVE-"].strip(),
                "--dsl",
                values["-TABULAR-DSL-"].strip(),
                "--host",
                values["-SERVER-HOST-"].strip(),
                "--port",
                values["-SERVER-PORT-"].strip(),
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "TABULAR")

        if event in ("-RUN-IMAGE-TRAIN-", "CTRL_I"):
            run_for_tab(window, runner, values, "Image Mode")

        if event == "-RUN-IMAGE-GEN-":
            cmd = _base_cmd() + [
                "image-generate",
                "--checkpoint",
                values["-IMG-CKPT-"],
                "--output-grid",
                values["-IMG-GRID-"],
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "IMAGE")

        if event == "-RUN-IMAGE-INTERP-":
            out = values["-IMG-GRID-"].strip()
            if out.lower().endswith(".png"):
                out = out[:-4] + "_interp.png"
            cmd = _base_cmd() + [
                "image-interpolate",
                "--checkpoint",
                values["-IMG-CKPT-"],
                "--output-grid",
                out,
                "--device",
                values["-DEVICE-"],
            ]
            runner.start(cmd, window, "IMAGE")

        if event in ("-RUN-MM-TRAIN-", "CTRL_M"):
            run_for_tab(window, runner, values, "Multimodal")

        if event == "-RUN-MM-RUN-":
            cmd = _base_cmd() + [
                "multimodal-run",
                "--checkpoint",
                values["-MM-CKPT-"],
                "--device",
                values["-DEVICE-"],
            ]
            if values["-MM-VECTOR-"].strip():
                cmd += ["--vector", values["-MM-VECTOR-"].strip()]
            elif values["-MM-TEXT-"].strip():
                cmd += ["--text", values["-MM-TEXT-"].strip()]
            runner.start(cmd, window, "MULTI")

        if event in ("-STOP-", "F6"):
            runner.stop(window)
            window["-STATUS-"].update("Stop signal sent.")

        if event == "-REFRESH-DEVICES-":
            devices = ["auto"] + available_device_names(include_cpu=True)
            window["-DEVICE-"].update(values=devices, value="auto")
            window["-STATUS-"].update("Device list refreshed.")

        if event == "-SHOW-DEVICES-":
            append_log(window, "SYSTEM", format_device_report())
            window["-STATUS-"].update("Printed device report.")

        if event == "-LOG-":
            tag, line = values[event]
            append_log(window, tag, line)

        if event == "-DONE-":
            tag, rc = values[event]
            append_log(window, tag, f"[done] return_code={rc}")
            window["-STATUS-"].update(f"Process done ({tag}) rc={rc}")

        if event == "F5":
            selected = values.get("-TABGROUP-", "Command Deck")
            run_for_tab(window, runner, values, selected)

    window.close()


if __name__ == "__main__":
    main()
