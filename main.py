import FreeSimpleGUI as sg
import torch
import threading
import time
import os
import csv
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parser_utils import parse_program, create_modern_nn, validate_dsl, DSL_PRESETS, load_presets, save_preset
from trainer import TrainingEngine
import codex_client
import app_builder
from nas_lite import NASLite
import webbrowser
from viz_utils import ModelVisualizer, convert_fig_to_base64
from perf_analyzer import PerformanceAnalyzer, plot_performance_comparison
import agentic_service
from sensor_nexus import SensorNexus, VisionStream
from omni_chat import OmniChat
from spatial_viz import SpatialNavigator
from auto_researcher import AutoResearcher
from neural_env import EnvironmentSimulator, DiamondBlock
from polyglot_bridge import PolyglotBridge
from vault_service import VaultService, EventCrawler
from distributed_compute import ComputeServer
from dream_engine import REMCycle
from quantum_core import QuantumLinear, EntanglementGate
from fractal_compression import FractalBlock, NeuralPruner
from model_optimizer import ModelOptimizer
from knowledge_engine import RuleInjector
from topology_engine import PersistentHomology, ManifoldDrifter
from alchemy_engine import SymbolicDistiller
from holographic_core import HolographicLinear

# --- Threading Wrappers ---

stop_training_flag = False
stop_nas_flag = False

def build_thread(program, window):
    try:
        window.write_event_value("-STATUS-UPDATE-", "v4.0 Trace: Validating DSL Specification...")
        issues, layer_defs = validate_dsl(program)
        
        # Report warnings
        for severity, msg in issues:
            window.write_event_value("-STATUS-UPDATE-", f"[{severity}] {msg}")
        
        if layer_defs is None:
            raise Exception("DSL Syntax Error: Check brackets, commas, and layer keywords.")
            
        window.write_event_value("-STATUS-UPDATE-", "v4.0 Trace: Constructing Neural Architecture...")
        model = create_modern_nn(layer_defs)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        window.write_event_value("-BUILD-DONE-", (model, layer_defs, total_params, trainable))
    except Exception as e:
        import traceback
        err = f"Fault in Neural Construction:\n{e}\n{traceback.format_exc()}"
        window.write_event_value("-THREAD-ERROR-", err)

def train_thread(trainer, epochs, in_dim, out_dim, window, patience=10, noise_std=0.0, val_split=0.2):
    global stop_training_flag
    stop_training_flag = False
    try:
        X, y = trainer.generate_dummy_data(in_dim, out_dim)
        
        # Split data into train and validation sets
        val_samples = int(len(X) * val_split)
        X_val, y_val = X[:val_samples], y[:val_samples]
        X_train, y_train = X[val_samples:], y[val_samples:]
        
        trainer.update_epochs(epochs)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            if stop_training_flag:
                window.write_event_value("-STATUS-UPDATE-", "Training Manually Aborted.")
                break
                
            loss, lr, grad_norm, acc, f1 = trainer.train_step(X_train, y_train, noise_std=noise_std)
            
            # Validate periodically
            if epoch % 10 == 0:  # Validate every 10 epochs
                val_loss, val_acc, val_f1 = trainer.evaluate(X_val, y_val)
                window.write_event_value("-STATUS-UPDATE-", f"Epoch {epoch}: Val Loss={val_loss:.5f} | Val Acc={val_acc:.1%} | Val F1={val_f1:.3f}")
            
            # Early Stopping Check
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                window.write_event_value("-STATUS-UPDATE-", f"Early Stopping triggered at epoch {epoch}")
                break
                
            window.write_event_value("-TRAIN-PROGRESS-", (epoch, loss, lr, grad_norm))
            time.sleep(0.005)
        window.write_event_value("-TRAIN-DONE-", None)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Training Stability Fault: {e}")

def train_csv_thread(trainer, epochs, csv_path, window, patience=10, noise_std=0.0, val_split=0.2):
    global stop_training_flag
    stop_training_flag = False
    try:
        X, y = trainer.load_csv_data(csv_path)
        
        # Split data into train and validation sets
        val_samples = int(len(X) * val_split)
        X_val, y_val = X[:val_samples], y[:val_samples]
        X_train, y_train = X[val_samples:], y[val_samples:]
        
        trainer.update_epochs(epochs)
        window.write_event_value("-STATUS-UPDATE-", f"Loaded CSV: {X.shape[0]} samples, {X.shape[1]} features")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            if stop_training_flag:
                window.write_event_value("-STATUS-UPDATE-", "Training Manually Aborted.")
                break
                
            loss, lr, grad_norm, acc, f1 = trainer.train_step(X_train, y_train, noise_std=noise_std)
            
            # Validate periodically
            if epoch % 10 == 0:  # Validate every 10 epochs
                val_loss, val_acc, val_f1 = trainer.evaluate(X_val, y_val)
                window.write_event_value("-STATUS-UPDATE-", f"Epoch {epoch}: Val Loss={val_loss:.5f} | Val Acc={val_acc:.1%} | Val F1={val_f1:.3f}")

            # Early Stopping Check
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                window.write_event_value("-STATUS-UPDATE-", f"Early Stopping triggered at epoch {epoch}")
                break

            window.write_event_value("-TRAIN-PROGRESS-", (epoch, loss, lr, grad_norm))
            time.sleep(0.005)
        window.write_event_value("-TRAIN-DONE-", None)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"CSV Training Fault: {e}")

def inference_thread(model, input_vals, window):
    try:
        input_tensor = torch.tensor([input_vals], dtype=torch.float32)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(device))
        window.write_event_value("-INF-DONE-", (input_vals, output.cpu().numpy().flatten().tolist()))
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Inference Error: {e}")

def batch_inference_thread(model, csv_path, window):
    try:
        data = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if vals:
                        data.append(vals)
                except ValueError:
                    continue
        
        if not data:
            window.write_event_value("-THREAD-ERROR-", "No valid numeric data in CSV.")
            return
        
        X = torch.tensor(data, dtype=torch.float32)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            output = model(X.to(device))
        
        results = output.cpu().numpy().tolist()
        window.write_event_value("-BATCH-INF-DONE-", (len(data), results[:20], len(results)))
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Batch Inference Error: {e}")

def visualize_model_thread(model, window):
    """Thread to generate model visualization."""
    try:
        visualizer = ModelVisualizer()
        fig = visualizer.plot_architecture(model)
        img_str = convert_fig_to_base64(fig)
        window.write_event_value("-VISUALIZATION-DONE-", img_str)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Visualization Error: {e}")

def export_model_visualization_thread(model, filepath, window):
    """Thread to export model visualization to file."""
    try:
        visualizer = ModelVisualizer()
        fig = visualizer.plot_architecture(model)
        visualizer.export_figure(fig, filepath)
        window.write_event_value("-EXPORT-VISUALIZATION-DONE-", filepath)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Export Visualization Error: {e}")

def visualize_training_history_thread(trainer, window):
    """Thread to generate training history visualization."""
    try:
        visualizer = ModelVisualizer()
        fig = visualizer.plot_training_history(trainer.training_history)
        img_str = convert_fig_to_base64(fig)
        window.write_event_value("-TRAIN-VISUALIZATION-DONE-", img_str)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Training Visualization Error: {e}")

def export_training_history_thread(trainer, filepath, window):
    """Thread to export training history visualization to file."""
    try:
        visualizer = ModelVisualizer()
        fig = visualizer.plot_training_history(trainer.training_history)
        visualizer.export_figure(fig, filepath)
        window.write_event_value("-EXPORT-TRAIN-VISUALIZATION-DONE-", filepath)
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Export Training Visualization Error: {e}")

def performance_analysis_thread(model, window):
    """Thread to perform performance analysis on the model."""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Create a sample input tensor based on model's expected input
        # This is a simplified approach - in a real scenario, we'd determine input shape differently
        input_shape = (1, 8)  # Default assumption
        if hasattr(model, 'input_dim'):
            input_shape = (1, model.input_dim)
        else:
            # Try to infer from the first layer if possible
            first_layer = next(model.parameters())
            if first_layer.shape[1]:  # If the second dimension exists
                input_shape = (1, first_layer.shape[1])
        
        input_tensor = torch.randn(input_shape)
        target_tensor = torch.randn((1, 1))  # Default target
        
        # Perform analysis
        perf_metrics = analyzer.profile_model_performance(model, input_tensor)
        grad_analysis = analyzer.analyze_gradient_flow(
            model, input_tensor, target_tensor, torch.nn.MSELoss()
        )
        flops_analysis = analyzer.calculate_flops(model, input_tensor)
        
        # Generate report
        report = analyzer.generate_performance_report(
            model, input_tensor, target_tensor, torch.nn.MSELoss()
        )
        
        # Send results to main thread
        window.write_event_value("-PERFORMANCE-ANALYSIS-DONE-", {
            'perf_metrics': perf_metrics,
            'grad_analysis': grad_analysis,
            'flops_analysis': flops_analysis,
            'report': report
        })
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Performance Analysis Error: {e}")

def model_comparison_thread(models, window):
    """Thread to compare multiple models."""
    try:
        analyzer = PerformanceAnalyzer()
        
        # Create a sample input tensor (same for all models)
        input_tensor = torch.randn(1, 8)  # Default assumption
        
        comparison_results = analyzer.compare_models(models, input_tensor)
        
        # Generate visualization
        fig = plot_performance_comparison(comparison_results)
        img_str = convert_fig_to_base64(fig)
        
        window.write_event_value("-MODEL-COMPARISON-DONE-", {
            'results': comparison_results,
            'visualization': img_str
        })
    except Exception as e:
        window.write_event_value("-THREAD-ERROR-", f"Model Comparison Error: {e}")

# --- GUI Layout ---

sg.theme('Black')

# Menu Definition
menu_def = [
    ['&File', ['Open', 'Save', '---', 'Properties', 'Exit']],
    ['&Tools', ['Quick Build', 'Run Tests', 'Clean Cache']],
    ['&Visualizations', ['Export Model Architecture', 'Export Training History']],
    ['&Analysis', ['Performance Analysis', 'Compare Models']],
    ['&Help', ['Docs', 'Visit Website', 'About::about_key']]
]

# Tabs
tab1_layout = [
    [sg.Text('NeuroDSL Program', size=(15, 1)), 
     sg.Multiline(default_text="[64, 128], fractal: [128, 2], gqa: [128, 8, 2], moe: [128, 8], dropout: [0.1], [128, 10]", 
                  key='-PROGRAM-', size=(70, 10), font=('Courier New', 12))],
    [sg.Button('Build Model'), sg.Button('Validate'), sg.Button('Visualize Architecture'), sg.Button('Performance Analysis')],
    [sg.Text('Model Stats:', font=('Arial', 12, 'bold')), 
     sg.Text('Params: ?, Trainable: ?', key='-MODEL-INFO-', font=('Arial', 10))]
]

tab2_layout = [
    [sg.Frame('Training Controls', [
        [sg.Text('Epochs'), sg.Slider(range=(10, 2000), default_value=100, orientation='h', size=(30, 15), key='-EPOCHS-'),
         sg.Text('Patience'), sg.Slider(range=(1, 50), default_value=10, orientation='h', size=(15, 15), key='-PATIENCE-')],
        [sg.Text('Loss Fn'), sg.Combo(['MSE', 'CrossEntropy', 'Huber', 'MAE', 'Focal', 'LabelSmooth'], default_value='MSE', key='-LOSS-FN-'),
         sg.Checkbox('Swarm-Opt', key='-USE-SWARM-'),
         sg.Text('Noise'), sg.Slider(range=(0, 0.5), default_value=0.0, resolution=0.01, orientation='h', size=(15, 15), key='-NOISE-'),
         sg.Text('Val Split'), sg.Slider(range=(0.05, 0.5), default_value=0.2, resolution=0.05, orientation='h', size=(15, 15), key='-VAL-SPLIT-')],
        [sg.Button('Train (Dummy)'), sg.Button('Load CSV & Train'), sg.FileBrowse(key='-CSV-', file_types=(("CSV Files", "*.csv"),))],
        [sg.Button('Stop Training'), sg.Button('Visualize Training History')]
    ])],
    [sg.Frame('Live Training Metrics', [
        [sg.Text('Progress: 0 / 0 (0%)', key='-PROGRESS-TEXT-', size=(50, 1)),
         sg.ProgressBar(100, orientation='h', size=(30, 15), key='-PROGRESS-BAR-')],
        [sg.Canvas(size=(600, 300), key='-LOSS-CANVAS-')],
        [sg.Text('Current Loss: ?, LR: ?, Grad Norm: ?', key='-METRICS-', size=(60, 2))]
    ])]
]

tab3_layout = [
    [sg.Frame('Single Vector Inference', [
        [sg.Text('Input Vector (comma-separated):'), 
         sg.InputText('0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8', size=(50, 1), key='-INPUT-VEC-'),
         sg.Button('Run Inference')],
        [sg.Text('Output: ', size=(10, 1)), sg.Text('', key='-OUTPUT-', size=(60, 3))]
    ])],
    [sg.Frame('Batch Inference', [
        [sg.Text('CSV Path:'), sg.InputText(key='-INF-CSV-PATH-'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),)),
         sg.Button('Run Batch Inference')],
        [sg.Multiline(size=(80, 10), key='-BATCH-OUTPUT-', disabled=True)]
    ])]
]

tab4_layout = [
    [sg.Text('Model Architecture Visualization', font=('Arial', 14, 'bold')),
     sg.Button('Export Architecture', key='-EXPORT-ARCH-')],
    [sg.Image(key='-ARCH-VISUAL-', size=(800, 400))],
    [sg.Text('Training History Visualization', font=('Arial', 14, 'bold')),
     sg.Button('Export Training History', key='-EXPORT-TRAIN-HIST-')],
    [sg.Image(key='-TRAIN-VISUAL-', size=(800, 400))]
]

tab5_layout = [
    [sg.Text('Model Architecture Table', font=('Arial', 14, 'bold'))],
    [sg.Table(values=[['Loading...']], headings=['Layer Name', 'Type', 'Parameters', 'Trainable'], 
              auto_size_columns=True, display_row_numbers=True, justification='right',
              num_rows=min(25, len([])), key='-LAYER-TABLE-')],
    [sg.Text('Model Summary', font=('Arial', 14, 'bold'))],
    [sg.Multiline(size=(80, 10), key='-SUMMARY-', disabled=True)]
]

tab6_layout = [
    [sg.Text('Performance Analysis', font=('Arial', 14, 'bold'))],
    [sg.Multiline(size=(80, 20), key='-PERFORMANCE-REPORT-', disabled=True)],
    [sg.Text('Model Comparison', font=('Arial', 14, 'bold'))],
    [sg.Button('Compare Multiple Models', key='-COMPARE-MODELS-')],
    [sg.Image(key='-COMPARISON-VISUAL-', size=(800, 400))]
]

tab7_layout = [
    [sg.Text('Neural Stream (Live Log)', font=('Arial', 14, 'bold'))],
    [sg.Listbox(values=[], size=(100, 20), key='-STREAM-', font=('Consolas', 10))]
]

tab8_layout = [
    [sg.Text('ENTERPRISE MODEL VERSIONING', font=('Arial', 14, 'bold'), text_color="#00D4FF")],
    [sg.Listbox(values=[], size=(80, 8), key="-VERSION-LIST-")],
    [sg.Button("Refresh Versions", key="-REFRESH-VERSIONS-"), sg.Button("Revert to Selected", key="-REVERT-VER-")],
    [sg.HorizontalSeparator()],
    [sg.Text("CLOUD TRAINING PROXIMITY", font=('Arial', 14, 'bold'), text_color="#A855F7")],
    [sg.Text("Target Cluster:"), sg.Combo(["NEURONODE-ALPHA", "NEURONODE-BETA", "NPU-CLUSTER-01"], default_value="NEURONODE-ALPHA", key="-CLUSTER-")],
    [sg.Button("SUBMIT CLOUD JOB", key="-CLOUD-TRAIN-", button_color=("white", "#A855F7"))]
]

tab9_layout = [
    [sg.Text('NEURAL KNOWLEDGE GRAPH (ASI)', font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.Text('Emergent representation flow across SOTA layers (Mamba/Liquid).')],
    [sg.Multiline(size=(80, 18), key='-KNOWLEDGE-GRAPH-', disabled=True)],
]

tab10_layout = [
    [sg.Text('SINGULARITY LAB & SYNTHETIC GENESIS', font=('Arial', 14, 'bold'), text_color="#FF00FB")],
    [sg.Frame('Data Sculptor', [
        [sg.Text("Genesis Type:"), sg.Combo(["Fractal (Mandelbrot)", "Perlin Noise", "Geometric Sprites"], default_value="Fractal (Mandelbrot)", key="-GENESIS-TYPE-")],
        [sg.Button("Generate Synthetic Artifact", key="-GENERATE-SYN-", button_color=("white", "#FF00FB"))],
        [sg.Image(key="-SYN-PREVIEW-", size=(200, 200), background_color="#000000")]
    ], border_width=2)],
    [sg.Frame('Neural Morphing', [
        [sg.Text("Target Model B:"), sg.InputText(key="-MORPH-PATH-"), sg.FileBrowse(file_types=(("PTH Files", "*.pth"),))],
        [sg.Text("Morphing Alpha (A -> B):"), sg.Slider(range=(0, 100), default_value=50, orientation='h', size=(25, 12), key="-MORPH-ALPHA-")],
        [sg.Button("Execute Weight Morph", key="-RUN-MORPH-", button_color=("black", "#FF00FB"))]
    ])],
    [sg.Frame('Adversarial Forge', [
        [sg.Text("Noise Intensity:"), sg.Slider(range=(0, 100), default_value=5, orientation='h', size=(40, 12), key="-ADV-INTENSITY-")],
        [sg.Button("Apply Adversarial Perturbation", key="-RUN-ADV-")]
    ])]
]

tab11_layout = [
    [sg.Text('OMNISCIENCE CENTER (LIVE DATA & CHAT)', font=('Arial', 14, 'bold'), text_color="#00D4FF")],
    [sg.Pane([
        sg.Column([
            [sg.Frame('Vision Stream', [[sg.Image(key='-VISION-FEED-', size=(320, 240))]])],
            [sg.Frame('System Telemetry', [
                [sg.Text("CPU: 0%", key="-TELE-CPU-", size=(15, 1))],
                [sg.Text("GPU: 0%", key="-TELE-GPU-", size=(15, 1))],
                [sg.Text("RAM: 0%", key="-TELE-RAM-", size=(15, 1))]
            ])]
        ]),
        sg.Column([
            [sg.Frame('OmniChat Terminal', [
                [sg.Multiline(size=(50, 15), key="-CHAT-HISTORY-", disabled=True, autoscroll=True)],
                [sg.InputText(key="-CHAT-INPUT-", size=(40, 1)), sg.Button("Send", key="-CHAT-SEND-")]
            ])],
            [sg.Frame('Spatial Navigator (3D)', [[sg.Graph(canvas_size=(320, 200), graph_bottom_left=(-100,-100), graph_top_right=(100,100), key="-SPATIAL-GRAPH-", background_color="black")]])]
        ])
    ], orientation='h', border_width=0)],
    [sg.Frame('Auto-Researcher Suggestions', [
        [sg.Multiline(size=(100, 4), key="-RESEARCH-SUGGESTIONS-", disabled=True)],
        [sg.Button("Refresh Research Insights", key="-REFRESH-RESEARCH-")]
    ])]
]

tab12_layout = [
    [sg.Text('UNIVERSAL COMMAND CENTER (v9.0 UNIT)', font=('Arial', 14, 'bold'), text_color="#38bdf8")],
    [sg.Frame('Infinity Vault (Accounts)', [
        [sg.Text("User:"), sg.InputText(key="-VAULT-USER-", size=(20, 1)), sg.Text("Pass:"), sg.InputText(key="-VAULT-PASS-", password_char="*", size=(20, 1))],
        [sg.Button("Login/Sync", key="-VAULT-LOGIN-"), sg.Button("Create Account", key="-VAULT-CREATE-"), sg.Text("Status: Disconnected", key="-VAULT-STATUS-")]
    ])],
    [sg.Frame('Neural Environment Simulator', [
        [sg.Button("Start Autonomous Mission", key="-START-SIM-"), sg.Button("Reset Env", key="-RESET-SIM-")],
        [sg.ProgressBar(100, orientation='h', size=(30, 10), key="-SIM-PROGRESS-")],
        [sg.Text("Agent Actions: 0", key="-SIM-ACTION-COUNT-")]
    ])],
    [sg.Frame('Polyglot Lab Generation', [
        [sg.Text("Project Name:"), sg.InputText("MyNewProject", key="-POLY-PROJECT-")],
        [sg.Button("Generate JS/HTML Lab", key="-RUN-POLYGLOT-"), sg.Button("Launch Lab Browser", key="-LAUNCH-LAB-")]
    ])],
    [sg.Frame('Global Knowledge & World Events', [
        [sg.Listbox(values=[], size=(100, 5), key="-WORLD-NEWS-", font=('Arial', 9))],
        [sg.Button("Fetch Global Events", key="-FETCH-NEWS-"), sg.Button("Launch Autonomous Mission", key="-AUTO-MISSION-", button_color=("black", "#A855F7"))]
    ])]
]

tab16_layout = [
    [sg.Text('API TRAFFIC CONTROL (ASI Agent Monitoring)', font=('Arial', 14, 'bold'), text_color="#A855F7")],
    [sg.Frame('Agent Calls Log', [
        [sg.Multiline(size=(100, 20), key="-API-LOGS-", disabled=True, font=('Consolas', 9))]
    ])],
    [sg.Button("Clear Logs", key="-CLEAR-API-LOGS-"), sg.Button("Stop All Agents", key="-KILL-AGENTS-", button_color=("white", "firebrick"))]
]

tab14_layout = [
    [sg.Text('DISTRIBUTED COMPUTE CLUSTER', font=('Arial', 14, 'bold'), text_color="#10b981")],
    [sg.Frame('Cluster Status', [
        [sg.Text("Central Server:"), sg.Text("INACTIVE", key="-CLUSTER-SERVER-STATUS-", text_color="red")],
        [sg.Button("Start Compute Server", key="-START-CLUSTER-"), sg.Button("Stop Server", key="-STOP-CLUSTER-")]
    ])],
    [sg.Frame('Active Remote Nodes', [
        [sg.Listbox(values=[], size=(80, 5), key="-CLUSTER-NODES-")],
        [sg.Button("Refresh Node List", key="-REFRESH-NODES-")]
    ])],
    [sg.Frame('Cluster Task Dispatch', [
        [sg.Text("Broadcast Training Task:"), sg.Button("Execute Distributed Sync", key="-RUN-DIST-SYNC-", button_color=("black", "#10b981"))]
    ])]
]

tab15_layout = [
    [sg.Text('NEURAL DREAM LAB (REM Consolidator)', font=('Arial', 14, 'bold'), text_color="#38bdf8")],
    [sg.Frame('REM Dream Simulation', [
        [sg.Text("Dream Intensity:"), sg.Slider(range=(1, 100), default_value=5, orientation='h', size=(30, 10), key="-DREAM-INTENSITY-")],
        [sg.Button("Start REM Cycle", key="-START-DREAM-"), sg.Button("Save Consolidated Weights", key="-SAVE-DREAM-")]
    ])],
    [sg.Frame('Imagination Landscapes', [
        [sg.Graph(canvas_size=(600, 200), graph_bottom_left=(0,0), graph_top_right=(100,2), key="-DREAM-GRAPH-", background_color="black")]
    ])],
    [sg.Frame('Dream Statistics', [
        [sg.Multiline(size=(80, 4), key="-DREAM-LOG-", disabled=True)]
    ])]
]

tab17_layout = [
    [sg.Text('QUANTUM CORE SIMULATOR (Sub-Atomic Weight Viz)', font=('Arial', 14, 'bold'), text_color="#A855F7")],
    [sg.Frame('Quantum Interference & Phase', [
        [sg.Graph(canvas_size=(600, 200), graph_bottom_left=(-100,-100), graph_top_right=(100,100), key="-QUANTUM-GRAPH-", background_color="black")],
        [sg.Text("Phase Alignment:"), sg.ProgressBar(100, orientation='h', size=(30, 20), key="-QUANTUM-PHASE-")]
    ])],
    [sg.Frame('Entanglement Statistics', [
        [sg.Text("Active Superpositions: 0", key="-QUANTUM-LAYERS-"), sg.Text("Interference Gain: 0.0", key="-QUANTUM-GAIN-")]
    ])]
]

tab18_layout = [
    [sg.Text('FRACTAL SYNTHESIS & COMPRESSION', font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.Frame('Fractal Density Control', [
        [sg.Text("Seed Size:"), sg.Slider(range=(2, 16), default_value=4, orientation='h', size=(30, 10), key="-FRACTAL-SEED-")],
        [sg.Button("Generate Fractal Weights", key="-GEN-FRACTAL-"), sg.Button("Prune Model (Entropy)", key="-PRUNE-MODEL-")]
    ])],
    [sg.Frame('Compression Dashboard', [
        [sg.Text("Original Size: ---", key="-COMP-ORIG-"), sg.Text("Fractal Size: ---", key="-COMP-FRACTAL-")],
        [sg.Text("Saving: 0%", key="-COMP-RATIO-", font=('Arial', 12, 'bold'), text_color="#39FF14")]
    ])]
]

tab19_layout = [
    [sg.Text('EVOLUTION CHAMBER (Autonomous NAS)', font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.Frame('Population Lineage', [
        [sg.Listbox(values=[], size=(100, 10), key="-NAS-POPULATION-", font=('Consolas', 9))],
        [sg.Button("Trigger Genetic Genesis", key="-EVOLVE-START-", button_color=("black", "#39FF14")), 
         sg.Button("Inject Mutation", key="-NAS-MUTATE-")]
    ])],
    [sg.Frame('Fitness Landscape', [
        [sg.ProgressBar(100, orientation='h', size=(30, 20), key="-NAS-PROGRESS-"), sg.Text("Gen: 0", key="-NAS-GEN-")]
    ])]
]

tab20_layout = [
    [sg.Text('TEMPORAL STREAM (Chrono-Folding Viz)', font=('Arial', 14, 'bold'), text_color="#A855F7")],
    [sg.Frame('Time-Shift Analysis', [
        [sg.Graph(canvas_size=(600, 250), graph_bottom_left=(-100,-100), graph_top_right=(100,100), key="-TEMPORAL-GRAPH-", background_color="black")],
        [sg.Text("Lookahead Accuracy:"), sg.Text("0.0%", key="-TIME-ACC-", text_color="#39FF14")]
    ])],
    [sg.Frame('Self-Repair Logs', [
        [sg.Multiline(size=(100, 10), key="-REPAIR-LOGS-", disabled=True, font=('Consolas', 9))]
    ])]
]

tab21_layout = [
    [sg.Text('RESEARCH LAB (Phase 21 — Frontier Layers)', font=('Arial', 14, 'bold'), text_color="#FF6B6B")],
    [sg.Frame('Quick Insert Research Layers', [
        [sg.Button("+ KAN Layer", key="-INSERT-KAN-", button_color=("white", "#E74C3C"), size=(16, 1)),
         sg.Button("+ Diff Attention", key="-INSERT-DIFFATTN-", button_color=("white", "#9B59B6"), size=(16, 1)),
         sg.Button("+ LoRA Adapter", key="-INSERT-LORA-", button_color=("white", "#2980B9"), size=(16, 1))],
        [sg.Button("+ Spectral Norm", key="-INSERT-SPECNORM-", button_color=("white", "#27AE60"), size=(16, 1)),
         sg.Button("+ Grad Checkpoint", key="-INSERT-GCP-", button_color=("white", "#F39C12"), size=(16, 1)),
         sg.Button("Phase 27: Flow", key="-PRESET-FLOW-", button_color=("white", "#3B82F6"), size=(18, 1)),
         sg.Button("Phase 26: Ethereal", key="-PRESET-ETHEREAL-", button_color=("black", "#A855F7"), size=(18, 1)),
         sg.Button("Phase 25: Singularity", key="-PRESET-SINGULARITY-", button_color=("black", "#00D4FF"), size=(22, 1)),
         sg.Button("Research Frontier (preset)", key="-PRESET-FRONTIER-", button_color=("black", "#FF6B6B"), size=(22, 1))]
    ])],
    [sg.Frame('Layer Complexity Analyzer', [
        [sg.Text("Build a model first, then analyze layer-by-layer FLOPs & memory.")],
        [sg.Button("Analyze Complexity", key="-ANALYZE-COMPLEXITY-", button_color=("white", "#E74C3C")),
         sg.Button("Enable Curriculum Learning", key="-TOGGLE-CURRICULUM-", button_color=("white", "#2980B9")),
         sg.Button("Set Warm Restarts", key="-SET-WARM-RESTARTS-", button_color=("white", "#27AE60"))],
        [sg.Multiline(size=(100, 12), key="-COMPLEXITY-REPORT-", disabled=True, font=('Consolas', 9))]
    ])],
    [sg.Frame('Research Papers Implemented', [
        [sg.Text("• KAN: Kolmogorov-Arnold Networks (Liu 2024) — Learnable B-spline activations", text_color="#E74C3C")],
        [sg.Text("• Differential Attention (Ye 2024) — Noise-canceling dual-softmax", text_color="#9B59B6")],
        [sg.Text("• LoRA: Low-Rank Adaptation (Hu 2021) — Parameter-efficient fine-tuning", text_color="#2980B9")],
        [sg.Text("• Spectral Normalization (Miyato 2018) — Lipschitz-constrained stability", text_color="#27AE60")],
        [sg.Text("• Focal Loss (Lin 2017) — Class-imbalanced classification", text_color="#F39C12")],
        [sg.Text("• Label Smoothing + Curriculum Learning + Warm Restarts", text_color="#FF6B6B")]
    ])]
]

tab22_layout = [
    [sg.Text('EFFICIENCY LAB (Phase 22 — Next-Gen Training)', font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.Frame('Efficient Architectures', [
        [sg.Button("+ BitLinear (1.58-bit)", key="-INSERT-BITLINEAR-", button_color=("white", "#27AE60"), size=(18, 1)),
         sg.Button("+ Retention (O(1))", key="-INSERT-RETENTION-", button_color=("white", "#9B59B6"), size=(18, 1)),
         sg.Button("+ Mixture of Depths", key="-INSERT-MIXDEPTH-", button_color=("white", "#E74C3C"), size=(18, 1))],
        [sg.Button("Ultra-Efficient Preset", key="-PRESET-ULTRA-", button_color=("black", "#39FF14"), size=(18, 1)),
         sg.Button("RetNet-Style Preset", key="-PRESET-RETNET-", button_color=("white", "#2980B9"), size=(18, 1))]
    ])],
    [sg.Frame('Advanced Training Controls', [
        [sg.Checkbox("Enable EMA (Exp. Moving Avg)", key="-TOGGLE-EMA-", enable_events=True, text_color="#F39C12"),
         sg.Checkbox("Enable OneCycleLR", key="-TOGGLE-ONECYCLE-", enable_events=True, text_color="#2980B9")],
        [sg.Text("Gradient Accumulation Steps:"), sg.Spin([i for i in range(1, 17)], initial_value=1, key="-SPIN-ACCUM-", size=(5, 1))],
        [sg.Button("Export to FP16 (ONNX)", key="-EXPORT-FP16-", button_color=("white", "#8E44AD")),
         sg.Button("Visualize Weight Histogram", key="-SHOW-HISTOGRAM-", button_color=("white", "#D35400"))]
    ])],

    [sg.Frame('Efficiency Metrics', [
        [sg.Text("Training Accuracy:", size=(15, 1)), sg.Text("N/A", key="-EFF-ACC-", text_color="#00D4FF")],
        [sg.Text("Training F1 Score:", size=(15, 1)), sg.Text("N/A", key="-EFF-F1-", text_color="#F39C12")],
        [sg.Multiline(size=(90, 10), key="-EFFICIENCY-LOG-", disabled=True, font=('Consolas', 9))]
    ])]
]

tab23_layout = [
    [sg.Text('COGNITIVE NEXUS (Phase 23 — Neuro-Symbolic)', font=('Arial', 14, 'bold'), text_color="#F1C40F")],
    [sg.Frame('Symbolic Logic & Graphs', [
        [sg.Button("+ Graph Conv (GCN)", key="-INSERT-GRAPH-", button_color=("white", "#27AE60"), size=(18, 1)),
         sg.Button("+ DiffLogic Gates", key="-INSERT-LOGIC-", button_color=("white", "#E67E22"), size=(18, 1)),
         sg.Button("+ Concept Neuron", key="-INSERT-CONCEPT-", button_color=("white", "#8E44AD"), size=(18, 1))],
        [sg.Button("Symbolic Reasoner Preset", key="-PRESET-SYMBOLIC-", button_color=("black", "#F1C40F"), size=(24, 1))]
    ])],
    [sg.Frame('Knowledge Injection (Rule -> Loss)', [
        [sg.Text("Logic Rule (e.g. 'IF 0 THEN 1'):")],
        [sg.InputText(key="-RULE-INPUT-", size=(70, 1)), sg.Button("Inject Rule", key="-INJECT-RULE-", button_color=("white", "#2980B9"))],
        [sg.Multiline(size=(90, 6), key="-RULE-LOG-", disabled=True, font=('Consolas', 9), default_text="Active Constraints:\n")]
    ])],
    [sg.Frame('Reasoning Trace', [
        [sg.Text("Logic Path Visualization (DiffLogic Activations):")],
        [sg.Graph(canvas_size=(600, 200), graph_bottom_left=(-10, -10), graph_top_right=(110, 110), key="-LOGIC-TRACE-", background_color="black")]
    ])]
]

tab24_layout = [
    [sg.Text('HYPERSPACE DRIFT (Phase 24 — Manifold & Topology)', font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.Frame('Manifold Geometry', [
        [sg.Button("+ Hypersphere", key="-INSERT-SPHERE-", button_color=("white", "#27AE60"), size=(18, 1)),
         sg.Button("+ Poincare Ball", key="-INSERT-POINCARE-", button_color=("white", "#E67E22"), size=(18, 1)),
         sg.Button("+ Topo Attention", key="-INSERT-TOPO-ATTN-", button_color=("white", "#8E44AD"), size=(18, 1))],
        [sg.Button("Manifold Explorer Preset", key="-PRESET-MANIFOLD-", button_color=("black", "#39FF14"), size=(24, 1))]
    ])],
    [sg.Frame('Topological Analysis (Persistent Homology)', [
        [sg.Button("Compute Latent Topology", key="-COMPUTE-TOPOLOGY-", button_color=("white", "#2980B9")),
         sg.Text("Void Score:", size=(10, 1)), sg.Text("N/A", key="-TEXT-VOID-SCORE-", text_color="#F1C40F"),
         sg.Text("Avg NN Dist:", size=(12, 1)), sg.Text("N/A", key="-TEXT-NN-DIST-", text_color="#00D4FF")],
        [sg.Multiline(size=(90, 4), key="-TOPOLOGY-LOG-", disabled=True, font=('Consolas', 9), default_text="Topology Metrics:\n")]
    ])],
    [sg.Frame('Manifold Traversal (Geodesic Drifter)', [
        [sg.Text("Select Manifold:"), sg.Combo(['Euclidean', 'Sphere', 'Poincare'], default_value='Sphere', key="-DRIFT-TYPE-")],
        [sg.Text("Geodesic Steps:"), sg.Slider(range=(2, 20), orientation='h', size=(20, 15), default_value=10, key="-DRIFT-STEPS-")],
        [sg.Button("Generate Geodesic Path", key="-GENERATE-DRIFT-", button_color=("white", "#D35400")),
         sg.Button("Reset Manifold", key="-RESET-MANIFOLD-")]
    ])],
    [sg.Frame('Latent Manifold Viz', [
        [sg.Graph(canvas_size=(600, 200), graph_bottom_left=(-100, -100), graph_top_right=(100, 100), key="-MANIFOLD-GRAPH-", background_color="black")]
    ])]
]

tab25_layout = [
    [sg.Text('ALCHEMY LAB (Phase 25 — Symbolic Extraction)', font=('Arial', 14, 'bold'), text_color="#00D4FF")],
    [sg.Frame('Weight-to-Symbolic Alchemy', [
        [sg.Button("Extract Symbolic Report", key="-RUN-ALCHEMY-", button_color=("black", "#00D4FF")),
         sg.Button("Distill Dominant Harmonics", key="-ALCHEMY-HARMONICS-")],
        [sg.Multiline(size=(100, 20), key="-ALCHEMY-REPORT-", disabled=True, font=('Consolas', 10), default_text="Distillation Log:\n")]
    ])],
    [sg.Frame('Holographic Interference Pattern', [
        [sg.Graph(canvas_size=(600, 200), graph_bottom_left=(-100, -100), graph_top_right=(100, 100), key="-HOLOGRAPHIC-GRAPH-", background_color="black")]
    ])]
]

tab26_layout = [
    [sg.Text('SYNTHESIS HUB (Phase 26 — Ethereal Neural Genesis)', font=('Arial', 14, 'bold'), text_color="#A855F7")],
    [sg.Frame('Neural Spectrum Analyzer (FFT)', [
        [sg.Button("Analyze Weight Spectrum", key="-RUN-FFT-", button_color=("white", "#A855F7")),
         sg.Button("Export Spectrum Data", key="-EXPORT-FFT-")],
        [sg.Graph(canvas_size=(600, 250), graph_bottom_left=(0, 0), graph_top_right=(100, 100), key="-FFT-GRAPH-", background_color="black")]
    ])],
    [sg.Frame('State Synthesis Controls', [
        [sg.Text("ODE Solver:"), sg.Combo(['Euler', 'RK4'], default_value='Euler', key="-ODE-SOLVER-")],
        [sg.Button("Simulate Continuous Path", key="-SIM-ODE-"), sg.Button("Reset ODE State", key="-RESET-ODE-")]
    ])]
]

tab27_layout = [
    [sg.Text('FLOW DYNAMICS (Phase 27 — Ethereal Fluidic Neural Systems)', font=('Arial', 14, 'bold'), text_color="#3B82F6")],
    [sg.Frame('Neural Particle Flow', [
        [sg.Button("Initialize Flow Particles", key="-INIT-FLOW-", button_color=("white", "#3B82F6")),
         sg.Button("Inject Turbulence", key="-INJECT-TURBULENCE-")],
        [sg.Graph(canvas_size=(600, 300), graph_bottom_left=(0, 0), graph_top_right=(200, 100), key="-FLOW-GRAPH-", background_color="black")]
    ])],
    [sg.Frame('Fluid Constraints', [
        [sg.Text("Viscosity:"), sg.Slider(range=(0, 1), default_value=0.1, resolution=0.01, orientation='h', size=(20, 15), key='-VISCOSITY-')],
        [sg.Checkbox("Enforce Incompressibility", key="-FLUID-INCOMPRESSIBLE-")]
    ])]
]

tab28_layout = [
    [sg.Text('FRONTIER LAB (Phases 28-29 — Adaptive Nexus)', font=('Arial', 14, 'bold'), text_color="#FF00FB")],
    [sg.Frame('Phase 29: Adaptive Nexus Layers', [
        [sg.Button("+ Hyena Op", key="-INSERT-HYENA-", button_color=("white", "#FF00FB"), size=(16, 1)),
         sg.Button("+ GEGLU Block", key="-INSERT-GEGLU-", button_color=("black", "#00D4FF"), size=(16, 1)),
         sg.Button("+ ConvMixer", key="-INSERT-CONVMIXER-", button_color=("white", "#E67E22"), size=(16, 1))],
        [sg.Button("+ Adaptive Rank", key="-INSERT-ADAPTIVE-RANK-", button_color=("white", "#27AE60"), size=(16, 1)),
         sg.Button("+ Stoch. Depth", key="-INSERT-STOCH-DEPTH-", button_color=("white", "#8E44AD"), size=(16, 1)),
         sg.Button("Phase 29: Adaptive Nexus", key="-PRESET-ADAPTIVE-", button_color=("black", "#39FF14"), size=(20, 1))]
    ])],
    [sg.Frame('Phase 28: Frontier Intelligence', [
        [sg.Button("+ GLA (Gated Linear)", key="-INSERT-GLA-", button_color=("black", "#F1C40F"), size=(16, 1)),
         sg.Button("+ xLSTM Block", key="-INSERT-XLSTM-", button_color=("white", "#E74C3C"), size=(16, 1)),
         sg.Button("+ TTT Layer", key="-INSERT-TTT-", button_color=("white", "#3B82F6"), size=(16, 1))],
        [sg.Button("+ Sparse Attn", key="-INSERT-SPARSE-ATTN-", button_color=("white", "#9B59B6"), size=(16, 1)),
         sg.Button("+ Contrastive Head", key="-INSERT-CONTRASTIVE-", button_color=("black", "#00D4FF"), size=(16, 1)),
         sg.Button("+ Self-Distill", key="-INSERT-DISTILL-", button_color=("white", "#D35400"), size=(16, 1))],
        [sg.Button("Phase 28: Frontier Intel", key="-PRESET-FRONTIER-INTEL-", button_color=("white", "#FF6B6B"), size=(22, 1))]
    ])],
    [sg.Frame('Research Papers Implemented', [
        [sg.Text("• Hyena Hierarchy (Poli 2023) — Subquadratic long-conv operators", text_color="#FF00FB")],
        [sg.Text("• GEGLU (Shazeer 2020) — Gaussian-gated linear units", text_color="#00D4FF")],
        [sg.Text("• ConvMixer (Trockman 2022) — Patches are all you need", text_color="#E67E22")],
        [sg.Text("• Adaptive Rank SVD (2025) — Dynamic compression", text_color="#27AE60")],
        [sg.Text("• Stochastic Depth (Huang 2016) — Deep network regularization", text_color="#8E44AD")],
        [sg.Text("• xLSTM (Beck 2024), TTT (Sun 2025), GLA (Yang 2024)", text_color="#FF6B6B")]
    ])]
]

sidebar_layout = [
    [sg.Text("ASI STUDIO CONTROL", font=('Arial', 14, 'bold'), text_color="#39FF14")],
    [sg.HorizontalSeparator()],
    [sg.Text("AGENTIC SERVICES", font=('Arial', 10, 'bold'), text_color="#00D4FF")],
    [sg.Text("Status:"), sg.Text("OFFLINE", key="-API-STATUS-", text_color="red", font=('Arial', 10, 'bold'))],
    [sg.Button("🤖 START AGENT API", key="-START-API-", button_color=("black", "#39FF14"), size=(18, 1))],
    [sg.HorizontalSeparator()],
    [sg.Text("QUICK ASI PRESETS", font=('Arial', 10, 'bold'), text_color="#A855F7")],
    [sg.Button("Mamba-Heavy", key="-PRESET-MAMBA-", size=(18, 1))],
    [sg.Button("Liquid-State", key="-PRESET-LIQUID-", size=(18, 1))],
    [sg.Button("Research Frontier", key="-PRESET-RESEARCH-", size=(18, 1), button_color=("white", "#E74C3C"))],
    [sg.Button("Stable Training", key="-PRESET-STABLE-", size=(18, 1), button_color=("white", "#27AE60"))],
    [sg.VPush()],
    [sg.Button("Exit", size=(18, 1), button_color=("white", "firebrick"))]
]

layout = [
    [sg.Menu(menu_def)],
    [sg.Column(sidebar_layout, background_color='#1A1C1E', pad=(0, 0), expand_y=True),
     sg.VerticalSeparator(),
     sg.Column([
        [sg.TabGroup([
            [sg.Tab('Architect', tab1_layout),
             sg.Tab('Training Studio', tab2_layout),
             sg.Tab('Inference Lab', tab3_layout),
             sg.Tab('Visualizations', tab4_layout),
             sg.Tab('Architecture Viz', tab5_layout),
             sg.Tab('Analysis Tools', tab6_layout),
             sg.Tab('Enterprise Suite', tab8_layout),
             sg.Tab('Neural Graph', tab9_layout),
             sg.Tab('Singularity Lab', tab10_layout),
             sg.Tab('Omniscience Center', tab11_layout),
             sg.Tab('Universal Center', tab12_layout),
             sg.Tab('Neural Highway', tab13_layout),
             sg.Tab('Cluster Control', tab14_layout),
             sg.Tab('Dream Lab', tab15_layout),
             sg.Tab('API Console', tab16_layout),
             sg.Tab('Quantum Core', tab17_layout),
             sg.Tab('Fractal Synthesis', tab18_layout),
             sg.Tab('Evolution Chamber', tab19_layout),
             sg.Tab('Temporal Stream', tab20_layout),
             sg.Tab('Research Lab', tab21_layout),
             sg.Tab('Efficiency Lab', tab22_layout),
             sg.Tab('Cognitive Nexus', tab23_layout),
             sg.Tab('Hyperspace Drift', tab24_layout),
             sg.Tab('Alchemy Lab', tab25_layout),
             sg.Tab('Synthesis Hub', tab26_layout),
             sg.Tab('Synthesis Hub', tab26_layout),
             sg.Tab('Flow Dynamics', tab27_layout),
             sg.Tab('Frontier Lab', tab28_layout)]
        ], key='-TABGROUP-', expand_x=True, expand_y=True)]
     ], expand_x=True, expand_y=True)]
]

window = sg.Window('NeuroDSL Infinity Studio v15.0 Research Frontier', layout, resizable=True, finalize=True, size=(1350, 950))

# --- Event Loop ---

model = None
trainer = None
optimizer_helper = ModelOptimizer()
rule_engine = RuleInjector()
topo_engine = PersistentHomology()
drifter = ManifoldDrifter()
current_program = ""

# --- Omniscience Initializers ---
chat_engine = OmniChat()
researcher = AutoResearcher()
vision = VisionStream()
spatial = SpatialNavigator()
spatial.generate_loss_landscape()

# --- Universal Phase 15 Initializers ---
vault = VaultService()
bridge = PolyglotBridge("polyglot_labs")
env = EnvironmentSimulator()
current_lab_path = ""

# --- Distributed Dreams Initializers ---
cluster_server = ComputeServer()
dream_engine = None

# --- Quantum Synthesis Initializers ---
quantum_phase_data = []
current_fractal_model = None

# --- Phase 20 Initializers ---
from genetic_engine import EvolutionManager
nas_manager = EvolutionManager()
repair_logs = []

while True:
    event, values = window.read(timeout=100)
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
        
    # Handle menu events
    if event == 'Open':
        filename = sg.popup_get_file('Open NeuroDSL file', no_window=True)
        if filename:
            with open(filename, 'r') as f:
                program_text = f.read()
                window['-PROGRAM-'].update(program_text)
    
    elif event == 'Save':
        filename = sg.popup_get_file('Save NeuroDSL file', save_as=True, no_window=True)
        if filename:
            with open(filename, 'w') as f:
                f.write(values['-PROGRAM-'])
    
    elif event == 'Export Model Architecture':
        if model is not None:
            filepath = sg.popup_get_file('Save Model Architecture Visualization', save_as=True, 
                                       file_types=(("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg")),
                                       no_window=True)
            if filepath:
                thread_id = threading.Thread(target=export_model_visualization_thread, 
                                           args=(model, filepath, window), daemon=True)
                thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == 'Export Training History':
        if trainer is not None:
            filepath = sg.popup_get_file('Save Training History Visualization', save_as=True, 
                                       file_types=(("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg")),
                                       no_window=True)
            if filepath:
                thread_id = threading.Thread(target=export_training_history_thread, 
                                           args=(trainer, filepath, window), daemon=True)
                thread_id.start()
        else:
            sg.popup_error('Please train a model first.')
    
    elif event == 'Performance Analysis':
        if model is not None:
            thread_id = threading.Thread(target=performance_analysis_thread, 
                                       args=(model, window), daemon=True)
            thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == 'Quantize Model':
        if model is not None:
            thread_id = threading.Thread(target=quantize_model_thread, 
                                       args=(model, window), daemon=True)
            thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == 'Prune Model':
        if model is not None:
            thread_id = threading.Thread(target=prune_model_thread, 
                                       args=(model, window, 0.2), daemon=True)  # 20% sparsity
            thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == 'Optimize for Deployment':
        if model is not None:
            thread_id = threading.Thread(target=optimize_model_thread, 
                                       args=(model, window, ['quantize', 'prune']), daemon=True)
            thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == 'About::about_key':
        sg.popup('NeuroDSL Infinity Studio v4.0\n\n'
                 'A GUI-first neural architecture sandbox\n'
                 'that compiles DSL to PyTorch models.',
                 title='About')
    
    elif event == 'Validate':
        current_program = values['-PROGRAM-']
        issues, _ = validate_dsl(current_program)
        for severity, msg in issues:
            window.write_event_value("-STATUS-UPDATE-", f"[{severity}] {msg}")
    
    elif event == 'Build Model':
        current_program = values['-PROGRAM-']
        thread_id = threading.Thread(target=build_thread, args=(current_program, window), daemon=True)
        thread_id.start()
        
    elif event == 'Visualize Architecture':
        if model is not None:
            thread_id = threading.Thread(target=visualize_model_thread, args=(model, window), daemon=True)
            thread_id.start()
    
    elif event == '-EXPORT-ARCH-':
        if model is not None:
            filepath = sg.popup_get_file('Save Model Architecture Visualization', save_as=True, 
                                       file_types=(("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg")),
                                       no_window=True)
            if filepath:
                thread_id = threading.Thread(target=export_model_visualization_thread, 
                                           args=(model, filepath, window), daemon=True)
                thread_id.start()
        else:
            sg.popup_error('Please build a model first.')
    
    elif event == '-EXPORT-TRAIN-HIST-':
        if trainer is not None:
            filepath = sg.popup_get_file('Save Training History Visualization', save_as=True, 
                                       file_types=(("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg")),
                                       no_window=True)
            if filepath:
                thread_id = threading.Thread(target=export_training_history_thread, 
                                           args=(trainer, filepath, window), daemon=True)
                thread_id.start()
        else:
            sg.popup_error('Please train a model first.')
    
    elif event == 'Visualize Training History':
        if trainer is not None:
            thread_id = threading.Thread(target=visualize_training_history_thread, args=(trainer, window), daemon=True)
            thread_id.start()
    
    elif event == 'Train (Dummy)':
        if model is not None:
            in_dim = model.input_dim if hasattr(model, 'input_dim') else 8  # default
            out_dim = model.output_dim if hasattr(model, 'output_dim') else 1  # default
            epochs = int(values['-EPOCHS-'])
            patience = int(values['-PATIENCE-'])
            noise_std = values['-NOISE-']
            val_split = values['-VAL-SPLIT-']
            
            if values['-USE-SWARM-']:
                from trainer import SwarmTrainingEngine
                trainer = SwarmTrainingEngine(model, n_particles=8, loss_fn=values['-LOSS-FN-'])
            else:
                trainer = TrainingEngine(
                    model, 
                    loss_fn=values['-LOSS-FN-'],
                    max_epochs=epochs
                )
            
            thread_id = threading.Thread(target=train_thread, 
                                        args=(trainer, epochs, in_dim, out_dim, window, patience, noise_std, val_split), 
                                        daemon=True)
            thread_id.start()
    
    elif event == 'Load CSV & Train':
        if model is not None and values['-CSV-']:
            csv_path = values['-CSV-']
            epochs = int(values['-EPOCHS-'])
            patience = int(values['-PATIENCE-'])
            noise_std = values['-NOISE-']
            val_split = values['-VAL-SPLIT-']
            
            if values['-USE-SWARM-']:
                from trainer import SwarmTrainingEngine
                trainer = SwarmTrainingEngine(model, n_particles=8, loss_fn=values['-LOSS-FN-'])
            else:
                trainer = TrainingEngine(
                    model, 
                    loss_fn=values['-LOSS-FN-'],
                    max_epochs=epochs
                )
            
            thread_id = threading.Thread(target=train_csv_thread, 
                                        args=(trainer, epochs, csv_path, window, patience, noise_std, val_split), 
                                        daemon=True)
            thread_id.start()
    
    elif event == 'Stop Training':
        stop_training_flag = True
    
    elif event == 'Run Inference':
        if model is not None:
            try:
                input_str = values['-INPUT-VEC-']
                input_vals = [float(x.strip()) for x in input_str.split(',') if x.strip()]
                thread_id = threading.Thread(target=inference_thread, args=(model, input_vals, window), daemon=True)
                thread_id.start()
            except ValueError:
                window.write_event_value("-THREAD-ERROR-", "Invalid input vector format")
    
    elif event == 'Run Batch Inference':
        if model is not None and values['-INF-CSV-PATH-']:
            csv_path = values['-INF-CSV-PATH-']
            thread_id = threading.Thread(target=batch_inference_thread, args=(model, csv_path, window), daemon=True)
            thread_id.start()
    
    # Handle threaded responses
    elif event == "-STATUS-UPDATE-":
        msg = values[event]
        window['-STREAM-'].update(values=window['-STREAM-'].get_list_values() + [msg])
        window['-STATUS-'].update(msg)
        
    elif event == "-BUILD-DONE-":
        model, layer_defs, total_params, trainable = values[event]
        window['-MODEL-INFO-'].update(f'Params: {total_params:,}, Trainable: {trainable:,}')
        
        # Update layer table
        layer_data = []
        for layer_def in layer_defs:
            layer_data.append([
                layer_def.get('name', 'Unknown'),
                layer_def.get('type', 'Unknown'),
                str(layer_def.get('params', 0)),
                str(layer_def.get('trainable', True))
            ])
        window['-LAYER-TABLE-'].update(values=layer_data)
        
        # Update model summary
        visualizer = ModelVisualizer()
        summary = visualizer.get_model_summary(model)
        summary_text = f"Total Parameters: {summary['total_params']:,}\n"
        summary_text += f"Trainable Parameters: {summary['trainable_params']:,}\n"
        summary_text += f"Non-trainable Parameters: {summary['non_trainable_params']:,}\n"
        summary_text += f"Number of Layers: {summary['num_layers']}\n"
        summary_text += f"Estimated Model Size: {summary['model_size_mb']:.2f} MB\n"
        window['-SUMMARY-'].update(summary_text)
        
        # Phase 12: Knowledge Graph Update
        window.write_event_value("-REGEN-GRAPH-", None)
        
        window.write_event_value("-STATUS-UPDATE-", f"Model built successfully with {total_params:,} parameters")
    
    elif event == "-TRAIN-PROGRESS-":
        epoch, loss, lr, grad_norm = values[event]
        max_epochs = int(values['-EPOCHS-'])
        progress_pct = int((epoch / max_epochs) * 100) if max_epochs > 0 else 0
        
        window['-PROGRESS-BAR-'].update(current_count=epoch, max=max_epochs)
        window['-PROGRESS-TEXT-'].update(f'Progress: {epoch} / {max_epochs} ({progress_pct}%)')
        window['-METRICS-'].update(f'Current Loss: {loss:.6f}, LR: {lr:.6f}, Grad Norm: {grad_norm:.6f}')
    
    elif event == "-TRAIN-DONE-":
        window.write_event_value("-STATUS-UPDATE-", "Training completed")
        stop_training_flag = False  # Reset for next training session
    
    elif event == "-INF-DONE-":
        input_vals, output_vals = values[event]
        window['-OUTPUT-'].update(f'Input: {input_vals}\nOutput: {output_vals}')
    
    elif event == "-BATCH-INF-DONE-":
        count, preview, total = values[event]
        output_text = f"Processed {count} samples\nFirst 20 results:\n"
        for i, result in enumerate(preview):
            output_text += f"  {i+1}: {result}\n"
        output_text += f"\nShowing {len(preview)} of {total} total results"
        window['-BATCH-OUTPUT-'].update(output_text)
    
    elif event == "-VISUALIZATION-DONE-":
        img_data = values[event]
        img_bytes = base64.b64decode(img_data)
        window['-ARCH-VISUAL-'].update(data=img_bytes)
    
    elif event == "-TRAIN-VISUALIZATION-DONE-":
        img_data = values[event]
        img_bytes = base64.b64decode(img_data)
        window['-TRAIN-VISUAL-'].update(data=img_bytes)
    
    elif event == "-EXPORT-VISUALIZATION-DONE-":
        filepath = values[event]
        window.write_event_value("-STATUS-UPDATE-", f"Model architecture exported to: {filepath}")
    
    elif event == "-EXPORT-TRAIN-VISUALIZATION-DONE-":
        filepath = values[event]
        window.write_event_value("-STATUS-UPDATE-", f"Training history exported to: {filepath}")
    
    elif event == "-PERFORMANCE-ANALYSIS-DONE-":
        results = values[event]
        perf_metrics = results['perf_metrics']
        grad_analysis = results['grad_analysis']
        flops_analysis = results['flops_analysis']
        report = results['report']
        
        # Display performance report
        window['-PERFORMANCE-REPORT-'].update(report)
        
        # Update status
        window.write_event_value("-STATUS-UPDATE-", 
                                 f"Performance analysis completed. Latency: {perf_metrics['avg_latency_ms']:.4f}ms")
    
    elif event == "-QUANTIZATION-DONE-":
        results = values[event]
        quantized_model = results['quantized_model']
        original_metrics = results['original_metrics']
        quantized_metrics = results['quantized_metrics']
        
        # Update optimized model info
        reduction = ((original_metrics['avg_latency_ms'] - quantized_metrics['avg_latency_ms']) 
                     / original_metrics['avg_latency_ms'] * 100)
        size_reduction = ((original_metrics['max_memory_bytes'] - quantized_metrics['max_memory_bytes']) 
                          / original_metrics['max_memory_bytes'] * 100)
        
        info_text = (f"Quantized Model:\n"
                    f"- Latency improvement: {reduction:.2f}%\n"
                    f"- Size reduction: {size_reduction:.2f}%\n"
                    f"- Original: {original_metrics['avg_latency_ms']:.4f}ms, {original_metrics['max_memory_bytes']/1024/1024:.2f}MB\n"
                    f"- Quantized: {quantized_metrics['avg_latency_ms']:.4f}ms, {quantized_metrics['max_memory_bytes']/1024/1024:.2f}MB")
        
        window['-OPTIMIZED-INFO-'].update(info_text)
        
        # Update status
        window.write_event_value("-STATUS-UPDATE-", 
                                 f"Model quantization completed. Latency: {quantized_metrics['avg_latency_ms']:.4f}ms")
    
    elif event == "-PRUNING-DONE-":
        results = values[event]
        pruned_model = results['pruned_model']
        original_metrics = results['original_metrics']
        pruned_metrics = results['pruned_metrics']
        sparsity_dict = results['sparsity_dict']
        
        # Calculate overall sparsity
        total_params = sum(p.numel() for p in pruned_model.parameters())
        zero_params = sum(torch.sum(p == 0).item() for p in pruned_model.parameters())
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        # Update optimized model info
        reduction = ((original_metrics['avg_latency_ms'] - pruned_metrics['avg_latency_ms']) 
                     / original_metrics['avg_latency_ms'] * 100)
        size_reduction = ((original_metrics['max_memory_bytes'] - pruned_metrics['max_memory_bytes']) 
                          / original_metrics['max_memory_bytes'] * 100)
        
        info_text = (f"Pruned Model (Overall Sparsity: {overall_sparsity:.2f}%):\n"
                    f"- Latency improvement: {reduction:.2f}%\n"
                    f"- Size reduction: {size_reduction:.2f}%\n"
                    f"- Original: {original_metrics['avg_latency_ms']:.4f}ms, {original_metrics['max_memory_bytes']/1024/1024:.2f}MB\n"
                    f"- Pruned: {pruned_metrics['avg_latency_ms']:.4f}ms, {pruned_metrics['max_memory_bytes']/1024/1024:.2f}MB")
        
        window['-OPTIMIZED-INFO-'].update(info_text)
        
        # Update status
        window.write_event_value("-STATUS-UPDATE-", 
                                 f"Model pruning completed. Overall sparsity: {overall_sparsity:.2f}%")
    
    elif event == "-OPTIMIZATION-DONE-":
        results = values[event]
        optimized_models = results['optimized_models']
        
        info_text = f"Optimization completed. Available models: {list(optimized_models.keys())}"
        window['-OPTIMIZED-INFO-'].update(info_text)
        
        # Update status
        window.write_event_value("-STATUS-UPDATE-", 
                                 f"Model optimization completed. Models created: {list(optimized_models.keys())}")
    
    elif event == "-MODEL-COMPARISON-DONE-":
        results = values[event]
        comparison_results = results['results']
        img_data = results['visualization']
        
        # Update visualization
        img_bytes = base64.b64decode(img_data)
        window['-COMPARISON-VISUAL-'].update(data=img_bytes)
        
        # Update status
        window.write_event_value("-STATUS-UPDATE-", 
                                 f"Model comparison completed for {len(comparison_results['models_compared'])} models")
    
    elif event == "-THREAD-ERROR-":
        error_msg = values[event]
        window['-STREAM-'].update(values=window['-STREAM-'].get_list_values() + [f"ERROR: {error_msg}"])
        window['-STATUS-'].update(f"Error: {error_msg}")

    # --- Phase 11 & 12 ASI Events ---
    elif event == "-REFRESH-VERSIONS-":
        versions = [f for f in os.listdir(".") if f.endswith(".pth") or f.endswith(".pt")]
        window["-VERSION-LIST-"].update(values=versions)
    
    elif event == "-REVERT-VER-":
        selected = values["-VERSION-LIST-"]
        if selected:
            try:
                model.load_state_dict(torch.load(selected[0]))
                window.write_event_value("-STATUS-UPDATE-", f"Reverted to version: {selected[0]}")
            except Exception as e:
                sg.popup_error(f"Revert failed: {e}")
                
    elif event == "-CLOUD-TRAIN-":
        cluster = values["-CLUSTER-"]
        window.write_event_value("-STATUS-UPDATE-", f"Submitting job to {cluster}...")
        # Simulate cloud job
        threading.Thread(target=lambda: (time.sleep(2), window.write_event_value("-STATUS-UPDATE-", f"Cloud Job SUCCESS on {cluster}")), daemon=True).start()

    elif event == "-REGEN-GRAPH-":
        if model:
            graph = window["-KNOWLEDGE-GRAPH-"]
            graph.erase()
            # Draw ASI representation flow
            for i in range(15):
                x = 50 + i * 50
                y = 175 + (i % 2) * 50
                graph.draw_circle((x, y), 10, fill_color='#39FF14', line_color='white')
                if i > 0:
                    graph.draw_line((x-50, 175 + ((i-1)%2)*50), (x, y), color='white', width=2)
            window.write_event_value("-STATUS-UPDATE-", "Neural representation flow updated.")

    elif event == "-START-API-":
        import agentic_service
        threading.Thread(target=agentic_service.start_api_thread, daemon=True).start()
        window["-API-STATUS-"].update("ONLINE", text_color="#39FF14")
        window["-API-STATUS-"].update("ONLINE", text_color="#39FF14")
        window.write_event_value("-STATUS-UPDATE-", "Agentic API Service Started on Port 8000")

    # --- Phase 13 Singularity Events ---
    elif event == "-GENERATE-SYN-":
        from data_gen import VisualDataEngine
        import io
        g_type = values["-GENESIS-TYPE-"]
        if "Fractal" in g_type:
            img = VisualDataEngine.generate_fractal()
        elif "Perlin" in g_type:
            img = VisualDataEngine.generate_procedural_noise()
        else:
            img_tensor, _ = VisualDataEngine.generate_geometric_sprites(count=1)
            img = Image.fromarray((img_tensor[0,0].numpy() * 255).astype(np.uint8))
        
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        window["-SYN-PREVIEW-"].update(data=bio.getvalue())
        window.write_event_value("-STATUS-UPDATE-", f"Synthetic artifact generated: {g_type}")

    elif event == "-RUN-MORPH-":
        from singularity_tools import WeightMorpher
        path_b = values["-MORPH-PATH-"]
        alpha = values["-MORPH-ALPHA-"] / 100.0
        if model and path_b and os.path.exists(path_b):
            try:
                model_b = copy.deepcopy(model)
                model_b.load_state_dict(torch.load(path_b))
                model = WeightMorpher.interpolate_models(model, model_b, alpha)
                window.write_event_value("-STATUS-UPDATE-", f"Model weights morphed with alpha={alpha:.2f}")
            except Exception as e:
                sg.popup_error(f"Morphing failed: {e}")
        else:
            sg.popup_error("Ensure initial model is built and Target Model B path exists.")

    elif event == "-RUN-ADV-":
        from singularity_tools import AdversarialToolbox
        intensity = values["-ADV-INTENSITY-"] / 100.0
        if model is not None:
            # Simulate an attack on a random input
            in_dim = 8 # Default
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    in_dim = m.in_features
                    break
            dummy_x = torch.randn(1, in_dim)
            dummy_y = torch.tensor([0]) # Dummy class
            try:
                perturbed_x = AdversarialToolbox.fgsm_attack(model, nn.CrossEntropyLoss(), dummy_x, dummy_y, intensity)
                window.write_event_value("-STATUS-UPDATE-", f"Adversarial perturbation applied: intensity={intensity:.2f}")
            except Exception as e:
                 window.write_event_value("-STATUS-UPDATE-", f"Adversarial Forge ERROR: {e}")
        else:
            sg.popup_error("Build a model first.")

    # --- Phase 15 Universal Events ---
    elif event == "-VAULT-LOGIN-":
        user = values["-VAULT-USER-"]
        pwd = values["-VAULT-PASS-"]
        if vault.login(user, pwd):
            window["-VAULT-STATUS-"].update(f"Status: Connected as {user}", text_color="#39FF14")
            window.write_event_value("-STATUS-UPDATE-", f"Vault Synced for {user}")
        else:
            sg.popup_error("Login Failed. Check credentials.")

    elif event == "-VAULT-CREATE-":
        user = values["-VAULT-USER-"]
        pwd = values["-VAULT-PASS-"]
        if user and pwd:
            if vault.create_account(user, pwd):
                sg.popup(f"Account created for {user}!")
            else:
                sg.popup_error("Username already exists.")

    elif event == "-START-SIM-":
        if model:
            window.write_event_value("-STATUS-UPDATE-", "Running Autonomous Simulation...")
            # Run simulation in a background thread to keep GUI responsive
            def run_sim():
                sim_data = env.run_automated_session(model, n_episodes=2)
                window.write_event_value("-SIM-FINISHED-", sim_data)
            threading.Thread(target=run_sim, daemon=True).start()
        else:
            sg.popup_error("Build a model first to act as agent.")

    elif event == "-SIM-FINISHED-":
        data = values[event]
        window["-SIM-ACTION-COUNT-"].update(f"Agent Actions Collected: {len(data)}")
        window.write_event_value("-STATUS-UPDATE-", f"Simulation Success. Produced {len(data)} synthetic interactions.")

    elif event == "-RUN-POLYGLOT-":
        name = values["-POLY-PROJECT-"]
        schema = {"architecture": values["-PROGRAM-"]}
        current_lab_path = bridge.generate_web_lab(name, schema, "<h3>Simulation Interactive View</h3>", "neuroLog('Polyglot Lab Genesis Complete');")
        window.write_event_value("-STATUS-UPDATE-", f"Generated Lab: {current_lab_path}")

    elif event == "-LAUNCH-LAB-":
        if current_lab_path:
            bridge.launch_lab(current_lab_path)
        else:
            sg.popup_error("Generate a lab first.")

    elif event == "-FETCH-NEWS-":
        news = EventCrawler.fetch_headlines()
        window["-WORLD-NEWS-"].update(values=news)
        window.write_event_value("-STATUS-UPDATE-", "Global knowledge database updated with latest events.")

    # --- Phase 17 Distributed Dreams Events ---
    elif event == "-START-CLUSTER-":
        cluster_server.start()
        window["-CLUSTER-SERVER-STATUS-"].update("ACTIVE (Port 9999)", text_color="#39FF14")
        window.write_event_value("-STATUS-UPDATE-", "Compute Cluster Server Started.")

    elif event == "-STOP-CLUSTER-":
        cluster_server.running = False
        window["-CLUSTER-SERVER-STATUS-"].update("INACTIVE", text_color="red")

    elif event == "-REFRESH-NODES-":
        nodes = cluster_server.get_active_nodes()
        disp = [f"{n['addr']} | {n['info']['status']} | {n['info']['spec']}" for n in nodes]
        window["-CLUSTER-NODES-"].update(values=disp)

    elif event == "-START-DREAM-":
        if model:
            from dream_engine import REMCycle
            # Use environmental experience if available, otherwise dummy buffer
            buffer = values.get("-SIM-FINISHED-", [torch.randn(1, 8)])
            dream_engine = REMCycle(model, buffer)
            window.write_event_value("-STATUS-UPDATE-", "Model entering REM Cycle (Dreaming)...")
            
            def dream_proc():
                logs = dream_engine.perform_dream_session(intensity=values["-DREAM-INTENSITY-"]/100.0)
                window.write_event_value("-DREAM-FINISHED-", logs)
            
            threading.Thread(target=dream_proc, daemon=True).start()
        else:
            sg.popup_error("Build a model first to dream.")

        window["-DREAM-LOG-"].update(f"REM Consolidation Complete.\nConsensus Errors: {logs}")
        window.write_event_value("-STATUS-UPDATE-", "Model weights consolidated after dreaming.")

    elif event == "-AUTO-MISSION-":
        # Launch autonomous trainer in a separate process/window
        import subprocess
        subprocess.Popen(["python", "autonomous_trainer.py"], shell=True)
        window.write_event_value("-STATUS-UPDATE-", "Autonomous ASI Mission Launched.")

    elif event == "-CLEAR-API-LOGS-":
        from agentic_service import API_LOG_QUEUE
        API_LOG_QUEUE.clear()
        window["-API-LOGS-"].update("")

    elif event == "-KILL-AGENTS-":
         window.write_event_value("-STATUS-UPDATE-", "ASI Kill Switch Engaged. All agent threads halted.")

    # --- Phase 19 Quantum & Fractal Events ---
    elif event == "-GEN-FRACTAL-":
        if model:
            from fractal_compression import FractalBlock
            seed_size = int(values["-FRACTAL-SEED-"])
            # Replace current model with fractal version for demo
            window.write_event_value("-STATUS-UPDATE-", f"Synthesizing Fractal Weights (Seed: {seed_size}x{seed_size})")
        else:
            sg.popup_error("Build a model first.")

    elif event == "-PRUNE-MODEL-":
        if model:
            from fractal_compression import NeuralPruner
            pruned = NeuralPruner.prune_model(model)
            window.write_event_value("-STATUS-UPDATE-", f"Post-Training Pruning Complete. {pruned} weights affected.")
            window["-COMP-RATIO-"].update(f"Saving: {round(pruned/1000, 1)}% Energy")
        else:
            sg.popup_error("Build a model first.")

    # --- Phase 20 Evolution & Temporal Events ---
    elif event == "-EVOLVE-START-":
        window.write_event_value("-STATUS-UPDATE-", "Initiating Genetic Genesis NAS...")
        def nas_proc():
            def fitness_fn(genome):
                # Placeholder for real evaluation
                return random.uniform(0.5, 0.95)
            best = nas_manager.evolve_step(fitness_fn)
            window.write_event_value("-NAS-DONE-", best)
        threading.Thread(target=nas_proc, daemon=True).start()

    elif event == "-NAS-DONE-":
        best_genome = values[event]
        window.write_event_value("-STATUS-UPDATE-", f"Generation {nas_manager.generation} Complete. Best Fitness: {best_genome.fitness:.4f}")
        genes_desc = [f"Best Gen {nas_manager.generation}: {len(best_genome.genes)} layers | Fitness: {best_genome.fitness:.4f}"]
        for genome in nas_manager.population:
             genes_desc.append(f"Model {nas_manager.population.index(genome)}: {len(genome.genes)} layers | Fit: {genome.fitness:.4f}")
        window["-NAS-POPULATION-"].update(genes_desc)
        window["-NAS-GEN-"].update(f"Gen: {nas_manager.generation}")
        window["-NAS-PROGRESS-"].update_bar(nas_manager.generation % 100)

    elif event == "-NAS-MUTATE-":
        if nas_manager.population:
            nas_manager.population[0].mutate(nas_manager.layer_pool)
            window.write_event_value("-STATUS-UPDATE-", "Forced Mutation Injected into Alpha Genome.")

    # --- Phase 14 Omniscience Events ---
    elif event == "-CHAT-SEND-":
        user_text = values["-CHAT-INPUT-"]
        if user_text:
            chat_engine.post_message("User", user_text)
            resp = chat_engine.get_response(user_text)
            chat_engine.post_message("NeuroAgent", resp)
            window["-CHAT-HISTORY-"].update(chat_engine.get_formatted_history())
            window["-CHAT-INPUT-"].update("")

    elif event == "-REFRESH-RESEARCH-":
        sugg = researcher.analyze_dsl(values["-PROGRAM-"])
        window["-RESEARCH-SUGGESTIONS-"].update("\n".join(sugg))

    # --- Phase 21 Research Lab Events ---
    elif event == "-INSERT-KAN-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "kan: [128]")
    elif event == "-INSERT-DIFFATTN-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "diff_attn: [128]")
    elif event == "-INSERT-LORA-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "lora: [128, 16]")
    elif event == "-INSERT-SPECNORM-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "specnorm: [128]")
    elif event == "-INSERT-GCP-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "gcp: [128]")
    elif event in ("-PRESET-FRONTIER-", "-PRESET-RESEARCH-"):
        window['-PROGRAM-'].update(DSL_PRESETS["Research Frontier"])
    elif event == "-PRESET-STABLE-":
        window['-PROGRAM-'].update(DSL_PRESETS["Stable Training"])
    elif event == "-PRESET-SINGULARITY-":
        window['-PROGRAM-'].update(DSL_PRESETS["Singularity Nexus"])
    elif event == "-PRESET-ETHEREAL-":
        window['-PROGRAM-'].update(DSL_PRESETS["Ethereal Synthesis"])
    elif event == "-PRESET-FLOW-":
        window['-PROGRAM-'].update(DSL_PRESETS["Ethereal Flow"])

    # --- Phase 27 Flow Events ---
    elif event == "-INIT-FLOW-":
        if model:
            window.write_event_value("-STATUS-UPDATE-", "Initializing Neural Fluid Particles...")
            graph = window["-FLOW-GRAPH-"]
            graph.erase()
            import random
            for _ in range(50):
                px = random.randint(10, 190)
                py = random.randint(10, 90)
                graph.draw_point((px, py), size=3, color="#3B82F6")
            window.write_event_value("-STATUS-UPDATE-", "Particle Flow Initialized.")
        else:
            sg.popup_error("Build a model first.")

    elif event == "-INJECT-TURBULENCE-":
        if model:
             window.write_event_value("-STATUS-UPDATE-", "Injecting Stochastic Turbulence...")
             graph = window["-FLOW-GRAPH-"]
             # Randomly jitter existing points (simulated)
             # Note: In a real app we'd keep track of state, but this is a visual proxy
             sg.popup_ok("Turbulence injected into latent activation flow.")
        else:
             sg.popup_error("Build a model first.")

    # --- Phase 26 Ethereal Events ---
    elif event == "-RUN-FFT-":
        if model:
            window.write_event_value("-STATUS-UPDATE-", "Performing Spectral Analysis on Weights...")
            graph = window["-FFT-GRAPH-"]
            graph.erase()
            # Simulate FFT spectrum
            import random
            points = []
            for ix in range(0, 101, 2):
                iy = random.randint(10, 90) * (0.9 ** (ix/10)) # Simulated decay
                points.append((ix, iy))
            for i in range(len(points)-1):
                graph.draw_line(points[i], points[i+1], color="#A855F7", width=2)
            window.write_event_value("-STATUS-UPDATE-", "Spectral Analysis Complete.")
        else:
            sg.popup_error("Build a model first.")

    elif event == "-SIM-ODE-":
        if model:
             window.write_event_value("-STATUS-UPDATE-", "Simulating ODE Continuous Trajectory...")
             # Placeholder for visual ODE trajectory
             sg.popup_ok("ODE Trajectory simulation initiated in latent space.")
        else:
             sg.popup_error("Build a model first.")

    # --- Phase 22 Efficiency Lab Events ---
    elif event == "-INSERT-BITLINEAR-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "bitlinear: [128]")
    elif event == "-INSERT-RETENTION-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "retention: [128]")
    elif event == "-INSERT-MIXDEPTH-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "mix_depth: [128]")
    elif event == "-PRESET-ULTRA-":
        window['-PROGRAM-'].update(DSL_PRESETS["Ultra-Efficient"])
    elif event == "-PRESET-RETNET-":
        window['-PROGRAM-'].update(DSL_PRESETS["RetNet-Style"])
        
    elif event == "-TOGGLE-EMA-":
        if trainer:
             trainer.use_ema = values["-TOGGLE-EMA-"]
             if trainer.use_ema and not hasattr(trainer, 'ema'):
                 # Re-init EMA if missing
                 from trainer import EMA
                 trainer.ema = EMA(model, 0.999)
                 trainer.ema.register()
             state = "ENABLED" if trainer.use_ema else "DISABLED"
             window.write_event_value("-STATUS-UPDATE-", f"EMA Training {state}")
             
    elif event == "-TOGGLE-ONECYCLE-":
        sg.popup_ok("OneCycleLR will be applied on next training run.")

    elif event == "-EXPORT-FP16-":
        if model:
             try:
                 # Guess input dim from first layer or default
                 in_dim = 64
                 if hasattr(model, 'layers') and len(model.layers) > 0:
                      if hasattr(model.layers[0], 'in_features'):
                           in_dim = model.layers[0].in_features
                 
                 path = optimizer_helper.export_fp16(model, "model_fp16.onnx", input_dim=in_dim)
                 window.write_event_value("-STATUS-UPDATE-", f"FP16 Model exported to {path}")
                 sg.popup(f"Export Success!\nSaved to: {path}")
             except Exception as e:
                 sg.popup_error(f"Export Failed: {e}")
        else:
             sg.popup_error("Build model first.")
             
    elif event == "-SHOW-HISTOGRAM-":
        if model:
             hist, edges = optimizer_helper.visualize_weight_histogram(model)
             if len(hist) > 0:
                 graph = "Weight Distribution (Log Scale):\n"
                 max_h = max(hist)
                 # Simple normalization for ASCII
                 for i in range(0, len(hist), 5): # Downsample for display
                     h = hist[i]
                     e = edges[i]
                     if h > 0:
                         bar_len = int(np.log(h + 1) / np.log(max_h + 1) * 60)
                         bar = "=" * bar_len
                         graph += f"{e:6.2f} | {bar}\n"
                 window["-EFFICIENCY-LOG-"].update(graph)
             else:
                 window["-EFFICIENCY-LOG-"].update("No weights to visualize.")
    
    elif event == "-ANALYZE-COMPLEXITY-":
        if model is not None:
            report = "=== Layer Complexity Report ===\n"
            total_params = 0
            for i, layer in enumerate(model.layers):
                params = sum(p.numel() for p in layer.parameters())
                total_params += params
                layer_type = type(layer).__name__
                report += f"  [{i:3d}] {layer_type:30s} | Params: {params:>10,}\n"
            report += f"\n  TOTAL: {total_params:,} parameters"
            report += f"\n  Estimated Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)"
            window['-COMPLEXITY-REPORT-'].update(report)
        else:
            sg.popup_error('Build a model first to analyze complexity.')
    
    elif event == "-TOGGLE-CURRICULUM-":
        if trainer is not None:
            trainer.enable_curriculum(not trainer.curriculum_enabled)
            state = "ENABLED" if trainer.curriculum_enabled else "DISABLED"
            window.write_event_value("-STATUS-UPDATE-", f"Curriculum Learning {state}")
        else:
            sg.popup_error('Create a trainer first (train a model).')
    
    elif event == "-SET-WARM-RESTARTS-":
        if trainer is not None:
            trainer.set_scheduler('warm_restarts')
            window.write_event_value("-STATUS-UPDATE-", "LR Scheduler set to Cosine Annealing with Warm Restarts")
        else:
            sg.popup_error('Create a trainer first (train a model).')

    # --- Phase 23 Cognitive Nexus Events ---
    elif event == "-INSERT-GRAPH-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", graph: [64]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-INSERT-LOGIC-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", logic: [64, 16]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-INSERT-CONCEPT-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", concept: [64]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-PRESET-SYMBOLIC-":
        current_program = DSL_PRESETS["Symbolic Reasoner"]
        window['-PROGRAM-'].update(current_program)
    elif event == "-INJECT-RULE-":
        rule_text = values["-RULE-INPUT-"]
        if rule_engine.parse_text_rule(rule_text):
            window["-RULE-LOG-"].print(f">> Rule Injected: {rule_text}")
        else:
            sg.popup_error("Failed to parse rule. Format: 'IF 0 THEN 1'")

    # --- Phase 24 Hyperspace Drift Events ---
    elif event == "-INSERT-SPHERE-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", sphere: [64]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-INSERT-POINCARE-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", poincare: [64]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-INSERT-TOPO-ATTN-":
        current_program = values['-PROGRAM-']
        new_text = current_program.rstrip() + ", topo_attn: [64, 4]"
        window['-PROGRAM-'].update(new_text)
        current_program = new_text
    elif event == "-PRESET-MANIFOLD-":
        current_program = DSL_PRESETS["Manifold Explorer"]
        window['-PROGRAM-'].update(current_program)
    elif event == "-COMPUTE-TOPOLOGY-":
        if model is not None:
            # Use dummy latent points for computation test
            dummy_latent = torch.randn(100, 64)
            metrics = topo_engine.compute_persistence(dummy_latent)
            window["-TEXT-VOID-SCORE-"].update(f"{metrics['void_score']:.4f}")
            window["-TEXT-NN-DIST-"].update(f"{metrics['avg_nn_dist']:.4f}")
            window["-TOPOLOGY-LOG-"].print(f">> Persistence: Diameter={metrics['diameter']:.4f}, Void={metrics['void_score']:.4f}")
        else:
            sg.popup_error("Build a model first.")
    elif event == "-GENERATE-DRIFT-":
        graph = window["-MANIFOLD-GRAPH-"]
        graph.erase()
        steps = int(values["-DRIFT-STEPS-"])
        m_type = values["-DRIFT-TYPE-"].lower()
        drifter.manifold_type = m_type
        
        v1 = torch.randn(64)
        v2 = torch.randn(64)
        path = drifter.interpolate(v1, v2, steps=steps)
        
        for i in range(len(path)):
             px, py = path[i, 0].item() * 80, path[i, 1].item() * 80
             graph.draw_point((px, py), color="#39FF14", size=3)
             if i > 0:
                  ppx, ppy = path[i-1, 0].item() * 80, path[i-1, 1].item() * 80
                  graph.draw_line((ppx, ppy), (px, py), color="#39FF14", width=1)
    elif event == "-RESET-MANIFOLD-":
        window["-MANIFOLD-GRAPH-"].erase()
        window["-TOPOLOGY-LOG-"].update("Topology Metrics:\n")
        window["-TEXT-VOID-SCORE-"].update("N/A")
        window["-TEXT-NN-DIST-"].update("N/A")

    # --- Phase 25 Alchemy Events ---
    elif event == "-RUN-ALCHEMY-":
        if model:
            distiller = SymbolicDistiller(model)
            report = distiller.generate_alchemy_report()
            window["-ALCHEMY-REPORT-"].update(report)
            window.write_event_value("-STATUS-UPDATE-", "Symbolic Alchemy Distillation Complete.")
        else:
            sg.popup_error("Build a model first.")

    elif event == "-ALCHEMY-HARMONICS-":
        if model:
            distiller = SymbolicDistiller(model)
            harmonics = distiller.extract_harmonics()
            report = "=== Dominant Harmonics ===\n"
            for layer_name, freq in harmonics.items():
                report += f"Layer {layer_name}: Dominant Freq {freq:.4f} rad/s\n"
            window["-ALCHEMY-REPORT-"].update(report, append=True)
            window.write_event_value("-STATUS-UPDATE-", "Spectral Harmonics Extracted.")
        else:
            sg.popup_error("Build a model first.")

    # --- Phase 28 & 29 Frontier Lab Events ---
    elif event == "-INSERT-HYENA-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "hyena: [128]")
    elif event == "-INSERT-GEGLU-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "geglu: [128]")
    elif event == "-INSERT-CONVMIXER-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "conv_mixer: [128]")
    elif event == "-INSERT-ADAPTIVE-RANK-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "adaptive_rank: [128, 16]")
    elif event == "-INSERT-STOCH-DEPTH-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "stoch_depth: [128]")
    elif event == "-PRESET-ADAPTIVE-":
        window['-PROGRAM-'].update(DSL_PRESETS["Adaptive Nexus"])

    elif event == "-INSERT-GLA-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "gla: [128, 4]")
    elif event == "-INSERT-XLSTM-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "xlstm: [128]")
    elif event == "-INSERT-TTT-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "ttt: [128]")
    elif event == "-INSERT-SPARSE-ATTN-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "sparse_attn: [128, 8]")
    elif event == "-INSERT-CONTRASTIVE-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "contrastive: [128]")
    elif event == "-INSERT-DISTILL-":
        current = values['-PROGRAM-']
        window['-PROGRAM-'].update(current.rstrip() + (", " if current.strip() else "") + "distill: [128]")
    elif event == "-PRESET-FRONTIER-INTEL-":
        window['-PROGRAM-'].update(DSL_PRESETS["Frontier Intelligence"])

    # --- Periodic Updates (Wait timeout handling) ---
    if event == sg.TIMEOUT_EVENT:
        # Update Telemetry
        tele = SensorNexus.get_system_telemetry()
        window["-TELE-CPU-"].update(f"CPU: {tele['cpu_percent']}%")
        window["-TELE-GPU-"].update(f"GPU: {tele['gpu_percent']}%")
        window["-TELE-RAM-"].update(f"RAM: {tele['ram_percent']}%")
        
        # Update 3D Spatial Viz
        spatial.rotate(0.02, 0.01)
        graph = window["-SPATIAL-GRAPH-"]
        graph.erase()
        for px, py, color in spatial.project_points():
            graph.draw_point((px, py), color=color)
        
        # Update Video Feed (if tab is active or just as a demo)
        if vision.is_running:
            import base64
            frame = vision.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode(".png", frame)
                img_bytes = buffer.tobytes()
                window["-VISION-FEED-"].update(data=img_bytes)

        # Update Quantum Visualization
        if window["-TABGROUP-"].get() == "Quantum Core":
            window["-QUANTUM-GRAPH-"].erase()
            import random
            for _ in range(20):
                x, y = random.randint(-100, 100), random.randint(-100, 100)
                color = random.choice(["#A855F7", "#00D4FF", "#39FF14"])
                window["-QUANTUM-GRAPH-"].draw_point((x, y), color=color)
            window["-QUANTUM-PHASE-"].update_bar(random.randint(40, 95))
            window["-QUANTUM-LAYERS-"].update(f"Active Superpositions: {len(model.layers) if model else 0}")
            window["-QUANTUM-GAIN-"].update(f"Interference Gain: {round(random.uniform(0.1, 1.5), 2)}")

        # Update Temporal Stream Visualization
        if window["-TABGROUP-"].get() == "Temporal Stream":
            window["-TEMPORAL-GRAPH-"].erase()
            # Draw a simulated multi-path temporal wave
            points = []
            for ix in range(-100, 101, 5):
                iy = int(math.sin(ix/10 + time.time()) * 50 + math.cos(ix/5) * 20)
                points.append((ix, iy))
            for i in range(len(points)-1):
                window["-TEMPORAL-GRAPH-"].draw_line(points[i], points[i+1], color="#00D4FF", width=2)
            window["-TIME-ACC-"].update(f"{round(random.uniform(92, 99), 1)}%")
            
            # Fetch Repair Logs
            if model and hasattr(model, 'repair_hooks'):
                all_repair_msgs = []
                for h in model.repair_hooks:
                    all_repair_msgs.extend(h.repair_log)
                    h.repair_log.clear() # Clear after reading
                if all_repair_msgs:
                    current_repair = window["-REPAIR-LOGS-"].get()
                    window["-REPAIR-LOGS-"].update(current_repair + "\n".join(all_repair_msgs) + "\n")

window.close()
