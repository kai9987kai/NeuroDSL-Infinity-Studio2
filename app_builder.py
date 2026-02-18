import os
import subprocess
import sys

def generate_inference_script(dsl_code, model_path, output_path="standalone_app.py"):
    """
    Generates a standalone Python script for inference with a GUI.
    """
    script_content = f"""
import torch
import FreeSimpleGUI as sg
import numpy as np
import os
import sys

# Ensure we can import local modules if they are bundled
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)

from network import ModernMLP
from parser_utils import parse_dsl

# --- Configuration ---
DSL_CODE = "{dsl_code}"
MODEL_PATH = "{os.path.basename(model_path)}"

def load_model():
    try:
        layer_defs = parse_dsl(DSL_CODE)
        model = ModernMLP(layer_defs)
        
        # Determine path to checkpoint
        if getattr(sys, 'frozen', False):
            ckpt_path = os.path.join(sys._MEIPASS, MODEL_PATH)
        else:
            ckpt_path = MODEL_PATH
            
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        else:
            sg.popup_error(f"Checkpoint not found: {{ckpt_path}}")
            return None
            
        model.eval()
        return model
    except Exception as e:
        sg.popup_error(f"Error loading model: {{e}}")
        return None

def main():
    sg.theme("DarkBlue3")
    layout = [
        [sg.Text("NeuroDSL Inference App", font=("Helvetica", 14, "bold"))],
        [sg.Text("Input Features (comma separated):")],
        [sg.Input(key="-INPUT-", expand_x=True)],
        [sg.Button("Predict", bind_return_key=True), sg.Button("Exit")],
        [sg.Text("Output:", size=(10, 1)), sg.Text("", key="-OUTPUT-", text_color="yellow", font=("Helvetica", 12))]
    ]
    
    window = sg.Window("NeuroDSL App", layout)
    model = load_model()
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
            
        if event == "Predict":
            if not model:
                sg.popup_error("Model not loaded.")
                continue
                
            try:
                # Parse inputs
                raw_input = values["-INPUT-"]
                inputs = [float(x.strip()) for x in raw_input.split(",")]
                tensor_in = torch.tensor([inputs], dtype=torch.float32)
                
                with torch.no_grad():
                    output = model(tensor_in)
                    
                window["-OUTPUT-"].update(str(output.numpy()[0]))
            except Exception as e:
                sg.popup_error(f"Inference Error: {{e}}")
                
    window.close()

if __name__ == "__main__":
    main()
"""
    with open(output_path, "w") as f:
        f.write(script_content)
    return output_path

def build_exe(script_path, model_path):
    """
    Builds an EXE using PyInstaller.
    """
    try:
        # Check if pyinstaller is installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
        # Build command
        # --add-data "model.pth;." (Windows separator is ;)
        add_data = f"{model_path};."
        
        cmd = [
            "pyinstaller",
            "--noconfirm",
            "--onefile",
            "--windowed",
            "--add-data", add_data,
            "--hidden-import", "network",
            "--hidden-import", "parser_utils",
            "--name", "NeuroDSL_App",
            script_path
        ]
        
        subprocess.check_call(cmd)
        return True, "dist/NeuroDSL_App.exe"
    except Exception as e:
        return False, str(e)
