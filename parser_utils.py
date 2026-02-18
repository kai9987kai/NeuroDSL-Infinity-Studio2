from pyparsing import Word, nums, Optional, Group, Suppress, ZeroOrMore, StringEnd, Literal, alphanums, Regex, OneOrMore
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
    "ASI Omni-Intelligence": "mamba: [256], hyper: [256], liquid: [256], trans: [256], moe: [256, 16, 2], [256, 10]",
    "Quantum-Fractal ASI": "quantum: [128], fractal_synth: [64], [64, 10]",
    "Research Frontier": "kan: [128], diff_attn: [128], lora: [128, 16], [128, 10]",
    "Stable Training": "[128, 256], specnorm: [256], gcp: [256], residual: [256], [256, 10]",
    "Ultra-Efficient": "bitlinear: [128], bitlinear: [128], retention: [128], [128, 10]",
    "RetNet-Style": "[128, 256], retention: [256, 8], mix_depth: [256], retention: [256, 8], [256, 10]",
    "Symbolic Reasoner": "[128, 64], logic: [64, 16], graph: [64], concept: [64], [64, 10]",
    "Manifold Explorer": "[128, 64], sphere: [64], poincare: [64], topo_attn: [64, 4], [64, 10]",
    "Singularity Nexus": "[128, 64], holographic: [64], alchemy: [64], logic: [64, 16], [64, 10]",
    "Ethereal Synthesis": "[128, 64], hypernet: [64], node: [64], xfusion: [64], [64, 10]",
    "Ethereal Flow": "[128, 64], turbulence: [64], fluid: [64], fluid: [64], [64, 10]",
    "Frontier Intelligence": "gla: [128, 4], xlstm: [128], ttt: [128], sparse_attn: [128, 8], [128, 10]",
    "Adaptive Nexus": "hyena: [128], geglu: [128], conv_mixer: [128], stoch_depth: [128], [128, 10]",
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
    equals = Suppress("=")
    open_bracket = Suppress("[")
    close_bracket = Suppress("]")
    quoted_string = Regex(r'\"[^\"]*\"').setParseAction(lambda t: t[0].strip('"'))
    
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
    script_layer = Group(Suppress("script") + colon + open_bracket + quoted_string + close_bracket).set_parse_action(
        lambda t: {'type': 'script', 'code': t[0][0]}
    )
    # Universal v9.0
    diamond_layer = Group(Suppress("diamond") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'diamond', 'dim': t[0][0]}
    )
    # Ensemble v10.0
    highway_layer = Group(Suppress("highway") + colon + open_bracket + Group(OneOrMore(quoted_string + Optional(comma))) + Optional(comma + Suppress("mode") + equals + quoted_string) + close_bracket).set_parse_action(
        lambda t: {'type': 'highway', 'models': t[0][0], 'mode': t[0][1] if len(t[0]) > 1 else "average"}
    )
    ensemble_layer_kw = Group(Suppress("ensemble") + colon + open_bracket + quoted_string + close_bracket).set_parse_action(
        lambda t: {'type': 'ensemble', 'path': t[0][0]}
    )
    # Bio-inspired v11.0
    imagine_layer = Group(Suppress("imagine") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'imagine', 'dim': t[0][0]}
    )
    # v13.0 Quantum & Fractal Synth
    quantum_layer = Group(Suppress("quantum") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'quantum', 'dim': t[0][0]}
    )
    fractal_synth_layer = Group(Suppress("fractal_synth") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'fractal_synth', 'dim': t[0][0]}
    )
    # v14.0 Chrono & Genetic
    chrono_layer = Group(Suppress("chrono") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'chrono', 'dim': t[0][0]}
    )
    evolve_kw = Group(Suppress("evolve") + colon + open_bracket + Group(OneOrMore(quoted_string + Optional(comma))) + close_bracket).set_parse_action(
        lambda t: {'type': 'evolve', 'pool': t[0][0]}
    )
    # v15.0 Phase 21 Research Frontier
    kan_layer = Group(Suppress("kan") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'kan', 'dim': t[0][0],
            'grid': t[0][1] if len(t[0]) > 1 else 5,
            'order': t[0][2] if len(t[0]) > 2 else 3
        }
    )
    diff_attn_layer = Group(Suppress("diff_attn") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'diff_attn', 'dim': t[0][0],
            'heads': t[0][1] if len(t[0]) > 1 else 8
        }
    )
    lora_layer = Group(Suppress("lora") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + float_number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'lora', 'dim': t[0][0],
            'rank': t[0][1] if len(t[0]) > 1 else 16,
            'alpha': t[0][2] if len(t[0]) > 2 else 1.0
        }
    )
    specnorm_layer = Group(Suppress("specnorm") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'specnorm', 'dim': t[0][0],
            'expansion': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    gcp_layer = Group(Suppress("gcp") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'gcp', 'dim': t[0][0],
            'expansion': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    # v16.0 Phase 22 Efficiency Frontier
    bitlinear_layer = Group(Suppress("bitlinear") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'bitlinear', 'dim': t[0][0],
            'expansion': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    retention_layer = Group(Suppress("retention") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'retention', 'dim': t[0][0],
            'heads': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    mix_depth_layer = Group(Suppress("mix_depth") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + float_number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'mix_depth', 'dim': t[0][0],
            'expansion': t[0][1] if len(t[0]) > 1 else 4,
            'capacity': t[0][2] if len(t[0]) > 2 else 0.5
        }
    )
    # Phase 23 Cognitive Nexus
    graph_conv_layer = Group(Suppress("graph") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'graph_conv', 'dim': t[0][0],
            'expansion': t[0][1] if len(t[0]) > 1 else 1
        }
    )
    diff_logic_layer = Group(Suppress("logic") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'diff_logic', 'dim': t[0][0],
            'rules': t[0][1] if len(t[0]) > 1 else 16
        }
    )
    concept_layer = Group(Suppress("concept") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'concept', 'dim': t[0][0]}
    )
    # Phase 24 Hyperspace Drift
    sphere_layer = Group(Suppress("sphere") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'sphere', 'dim': t[0][0]}
    )
    poincare_layer = Group(Suppress("poincare") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'poincare', 'dim': t[0][0]}
    )
    topo_attn_layer = Group(Suppress("topo_attn") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'topo_attn', 'dim': t[0][0],
            'heads': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    # Phase 25 Singularity Nexus
    holographic_layer = Group(Suppress("holographic") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'holographic', 'dim': t[0][0]}
    )
    alchemy_layer = Group(Suppress("alchemy") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'alchemy', 'dim': t[0][0]}
    )
    # Phase 26 Ethereal Synthesis
    xfusion_layer = Group(Suppress("xfusion") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'xfusion', 'dim': t[0][0]}
    )
    node_layer = Group(Suppress("node") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'node', 'dim': t[0][0]}
    )
    hypernet_layer = Group(Suppress("hypernet") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'hypernet', 'dim': t[0][0]}
    )
    # Phase 27 Ethereal Flow
    fluid_layer = Group(Suppress("fluid") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'fluid', 'dim': t[0][0]}
    )
    turbulence_layer = Group(Suppress("turbulence") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'turbulence', 'dim': t[0][0]}
    )
    # Phase 28 Frontier Intelligence
    gla_layer = Group(Suppress("gla") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'gla', 'dim': t[0][0],
            'heads': t[0][1] if len(t[0]) > 1 else 4
        }
    )
    xlstm_layer = Group(Suppress("xlstm") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'xlstm', 'dim': t[0][0]}
    )
    ttt_layer = Group(Suppress("ttt") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'ttt', 'dim': t[0][0]}
    )
    contrastive_layer = Group(Suppress("contrastive") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'contrastive', 'dim': t[0][0]}
    )
    sparse_attn_layer = Group(Suppress("sparse_attn") + colon + open_bracket + number + Optional(comma + number) + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'sparse_attn', 'dim': t[0][0],
            'k': t[0][1] if len(t[0]) > 1 else 8,
            'heads': t[0][2] if len(t[0]) > 2 else 4
        }
    )
    distill_layer = Group(Suppress("distill") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'distill', 'dim': t[0][0]}
    )
    # Phase 29 Adaptive Nexus
    hyena_layer = Group(Suppress("hyena") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'hyena', 'dim': t[0][0]}
    )
    geglu_layer = Group(Suppress("geglu") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'geglu', 'dim': t[0][0]}
    )
    conv_mixer_layer = Group(Suppress("conv_mixer") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'conv_mixer', 'dim': t[0][0]}
    )
    adaptive_rank_layer = Group(Suppress("adaptive_rank") + colon + open_bracket + number + Optional(comma + number) + close_bracket).set_parse_action(
        lambda t: {
            'type': 'adaptive_rank', 'dim': t[0][0],
            'rank': t[0][1] if len(t[0]) > 1 else 16
        }
    )
    stoch_depth_layer = Group(Suppress("stoch_depth") + colon + open_bracket + number + close_bracket).set_parse_action(
        lambda t: {'type': 'stoch_depth', 'dim': t[0][0]}
    )

    layer = (hyena_layer | geglu_layer | conv_mixer_layer | adaptive_rank_layer | stoch_depth_layer | gla_layer | xlstm_layer | ttt_layer | contrastive_layer | sparse_attn_layer | distill_layer | fluid_layer | turbulence_layer | xfusion_layer | node_layer | hypernet_layer | holographic_layer | alchemy_layer | graph_conv_layer | diff_logic_layer | concept_layer | sphere_layer | poincare_layer | topo_attn_layer | bitlinear_layer | retention_layer | mix_depth_layer | kan_layer | diff_attn_layer | lora_layer | specnorm_layer | gcp_layer | mamba_layer | liquid_layer | hyper_layer | script_layer | diamond_layer | highway_layer | ensemble_layer_kw | imagine_layer | quantum_layer | fractal_synth_layer | chrono_layer | evolve_kw | conv3d_layer | fractal_layer | moe_layer | gqa_layer | attn_layer | 
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
            prev_out = layer_def['out']
        elif lt in ('hyena', 'geglu', 'conv_mixer', 'adaptive_rank', 'stoch_depth', 'gla', 'xlstm', 'ttt', 'contrastive', 'sparse_attn', 'distill', 'fluid', 'turbulence', 'xfusion', 'node', 'hypernet', 'holographic', 'alchemy', 'fractal', 'fractal_synth', 'quantum', 'moe', 'gqa', 'attn', 'trans', 'residual', 'conv1d', 'lstm', 'mod', 'conv3d', 'mamba', 'liquid', 'hyper', 'diamond', 'imagine', 'kan', 'diff_attn', 'lora', 'specnorm', 'gcp', 'bitlinear', 'retention', 'mix_depth', 'graph_conv', 'diff_logic', 'concept', 'sphere', 'poincare', 'topo_attn'):
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
