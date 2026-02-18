from pyparsing import Word, nums, Optional, Group, Suppress, ZeroOrMore, StringEnd, Literal, alphanums, Regex
import torch
import json
import os
from network import ModernMLP

# ==============================================================
# DSL PRESETS — Common architectures ready to go
# ==============================================================

DSL_PRESETS = {
    "Classifier (MLP)": "[128, 64], dropout: [0.2], [64, 32], [32, 10]",
    "Deep Classifier": "[256, 128], residual: [128], dropout: [0.3], [128, 64], [64, 10]",
    "AutoEncoder": "[784, 256], [256, 64], [64, 256], [256, 784]",
    "Transformer Block": "fractal: [256, 2], trans: [256], [256, 10]",
    "MoE Heavy": "[128, 256], moe: [256, 8], dropout: [0.1], [256, 10]",
    "Adaptive Creative": "[128, 256], moe: [256, 10, 2], mod: [256, 4, 0.4], gqa: [256, 8, 2], [256, 10]",
    "Attention Pipeline": "[64, 128], gqa: [128, 8, 2], residual: [128], [128, 10]",
    "Conv-LSTM Hybrid": "[64, 128], lstm: [128], [128, 10]",
    "MoE-Heavy": "moe: [128, 8, 2], moe: [128, 8, 2], [128, 10]",
    "Fractal Deep": "fractal: [64, 4], [64, 10]",
    "Omni-Model (SOTA)": "[16, 32], conv3d: [32, 3], conv1d: [32, 3], trans: [32], moe: [32, 4, 1], fractal: [32, 2], lstm: [32, 2], [32, 10]",
    "Kitchen Sink": "fractal: [256, 2], moe: [256, 8, 1], mod: [256, 4, 0.35], gqa: [256, 8, 2], residual: [256], conv1d: [256, 5], lstm: [256], dropout: [0.1], [256, 10]",
    "ASI Omni-Intelligence": "mamba: [256], hyper: [256], liquid: [256], trans: [256], moe: [256, 16, 2], [256, 10]",
}

PRESETS_FILE = "user_presets.json"

def load_presets():
    """Load built-in and user presets."""
    presets = DSL_PRESETS.copy()
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r") as f:
                user_presets = json.load(f)
                presets.update(user_presets)
        except Exception as e:
            print(f"Error loading presets: {e}")
    return presets

def save_preset(name, dsl_code):
    """Save a user preset."""
    user_presets = {}
    if os.path.exists(PRESETS_FILE):
         try:
            with open(PRESETS_FILE, "r") as f:
                user_presets = json.load(f)
         except: pass
         
    user_presets[name] = dsl_code
    with open(PRESETS_FILE, "w") as f:
        json.dump(user_presets, f, indent=2)

def parse_program(program):
    """Parse the NeuroDSL specification into a list of layer definitions."""
    clean_prog = "\n".join([line.split("#")[0] for line in program.splitlines()])
    
    number = Word(nums).setParseAction(lambda t: int(t[0]))
    float_number = Regex(r'\d+\.?\d*').setParseAction(lambda t: float(t[0]))
    comma = Suppress(",")
    colon = Suppress(":")
    open_bracket = Suppress("[")
    close_bracket = Suppress("]")
    
    linear_layer = Group(open_bracket + number + comma + number + close_bracket).set_parse_action(
        lambda t: {'type': 'linear', 'in': t[0][0], 'out': t[0][1]}
    )
    attn_layer = Group(Suppress("attn") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'attn', 'dim': t[0][0]}
    )
    gqa_layer = Group(Suppress("gqa") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'gqa', 'dim': t[0][0], 
            'heads': t[0][1] if len(t[0]) > 1 else 8, 
            'groups': t[0][2] if len(t[0]) > 2 else 2
        }
    )
    moe_layer = Group(Suppress("moe") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'moe', 'dim': t[0][0],
            'experts': t[0][1] if len(t[0]) > 1 else 8,
            'shared': t[0][2] if len(t[0]) > 2 else 0,
        }
    )
    trans_layer = Group(Suppress("trans") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'trans', 'dim': t[0][0]}
    )
    fractal_layer = Group(Suppress("fractal") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {'type': 'fractal', 'dim': t[0][0], 'depth': t[0][1] if len(t[0]) > 1 else 2}
    )
    dropout_layer = Group(Suppress("dropout") + colon + open_bracket + float_number + close_bracket).set_parse_action(
        lambda t: {'type': 'dropout', 'rate': t[0][0] if t[0][0] <= 1.0 else t[0][0] / 100.0}
    )
    residual_layer = Group(Suppress("residual") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {'type': 'residual', 'dim': t[0][0], 'expansion': t[0][1] if len(t[0]) > 1 else 4}
    )
    conv1d_layer = Group(Suppress("conv1d") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {'type': 'conv1d', 'dim': t[0][0], 'kernel': t[0][1] if len(t[0]) > 1 else 3}
    )
    lstm_layer = Group(Suppress("lstm") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {'type': 'lstm', 'dim': t[0][0], 'layers': t[0][1] if len(t[0]) > 1 else 1}
    )
    mod_layer = Group(Suppress("mod") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + float_number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'mod', 'dim': t[0][0], 'expansion': t[0][1] if len(t[0]) > 1 else 4,
            'threshold': (t[0][2] / 100.0) if (len(t[0]) > 2 and t[0][2] > 1.0) else (t[0][2] if len(t[0]) > 2 else 0.35)
        }
    )
    conv3d_layer = Group(Suppress("conv3d") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'conv3d', 'dim': t[0][0], 
            'kernel': t[0][1] if len(t[0]) > 1 else 3, 
            'groups': t[0][2] if len(t[0]) > 2 else 1
        }
    )
    # ASI v6.0
    mamba_layer = Group(Suppress("mamba") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'mamba', 'dim': t[0][0]}
    )
    liquid_layer = Group(Suppress("liquid") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'liquid', 'dim': t[0][0]}
    )
    hyper_layer = Group(Suppress("hyper") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'hyper', 'dim': t[0][0]}
    )

    layer = (mamba_layer | liquid_layer | hyper_layer | conv3d_layer | fractal_layer | moe_layer | gqa_layer | attn_layer | 
             trans_layer | dropout_layer | residual_layer | conv1d_layer | lstm_layer | mod_layer | linear_layer)
    
    nn_definition = Optional(Suppress("nn") + colon) + layer + ZeroOrMore(Optional(comma) + layer) + StringEnd()
    
    try:
        result = nn_definition.parse_string(clean_prog)
        return result.asList()
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

def validate_dsl(program):
    """Validate a DSL program and return a list of warnings/errors."""
    issues = []
    result = parse_program(program)
    if result is None:
        issues.append(("ERROR", "Failed to parse DSL. Check syntax."))
        return issues, None
    if len(result) == 0:
        issues.append(("ERROR", "No layers defined."))
        return issues, None
    
    prev_out = None
    for i, layer_def in enumerate(result):
        lt = layer_def['type']
        if lt == 'linear':
            if prev_out is not None and layer_def['in'] != prev_out:
                issues.append(("WARNING", f"Layer {i}: input {layer_def['in']} != previous output {prev_out}"))
            prev_out = layer_def['out']
        elif lt in ('fractal', 'moe', 'gqa', 'attn', 'trans', 'residual', 'conv1d', 'lstm', 'mod', 'conv3d', 'mamba', 'liquid', 'hyper'):
            dim = layer_def.get('dim')
            if prev_out is not None and dim != prev_out:
                issues.append(("WARNING", f"Layer {i} ({lt}): dim {dim} != previous output {prev_out}"))
            prev_out = dim
    return issues, result

def create_modern_nn(layer_defs):
    if not layer_defs: return None
    model = ModernMLP(layer_defs)
    if torch.cuda.is_available(): model = model.cuda()
    return model
