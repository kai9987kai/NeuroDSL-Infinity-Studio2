import random
import re

class NASLite:
    def __init__(self):
        self.layer_types = [
            " [128, 128]", 
            " [256, 256]", 
            " [64, 64]",
            " trans: [128]", 
            " trans: [256]",
            " residual: [128, 4]",
            " residual: [256, 4]",
            " dropout: [0.2]",
            " dropout: [0.5]"
        ]

    def parse_dimensions(self, dsl_code):
        # Extract the input dimension from the first layer
        # Assumes format like "[784, 128], ..."
        match = re.search(r"\[(\d+),\s*(\d+)\]", dsl_code)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def mutate_dsl(self, dsl_code):
        """
        Randomly mutates the DSL architecture.
        Strategies:
        1. Change width of a hidden layer (naive string replacement for now, improved later).
        2. Add a layer (maintaining connectivity is tricky with Regex, so we'll append compatible blocks).
        3. Remove a layer.
        """
        parts = [p.strip() for p in dsl_code.split(",")]
        
        # Strategy: Randomly insert a block in the middle
        if len(parts) > 2 and random.random() < 0.6:
            insert_idx = random.randint(1, len(parts)-1)
            
            # Context-aware insertion? 
            # For simplicity, we assume we can insert "trans: [D]" blocks if D matches.
            # But fixing dimensions is hard without full parsing.
            # Let's try a safer mutation:
            # "Residualize": Wrap a standard layer in residual if dimensions match?
            
            # EASIER STRATEGY: Global Width Scaling
            # Find all numbers like 128, 256 and multiply/divide them?
            # Or just replace standard hidden sizes.
            
            mutation_type = random.choice(["scale_width", "add_dropout", "swap_activation"])
            
            if mutation_type == "scale_width":
                # Find a common dimension and change it
                if "128" in dsl_code:
                    return dsl_code.replace("128", random.choice(["64", "256"]))
                if "256" in dsl_code:
                    return dsl_code.replace("256", random.choice(["128", "512"]))
                    
            elif mutation_type == "add_dropout":
                # Add dropout after a random layer
                idx = random.randint(0, len(parts)-2)
                parts.insert(idx+1, "dropout: [0.2]")
                return ", ".join(parts)
                
            elif mutation_type == "swap_activation":
                # Not strictly in DSL yet, but we could add activation args if supported.
                # Since DSL is fixed, let's try replacing "trans" with "residual"
                return dsl_code.replace("trans:", "residual:")
                
        return dsl_code 

    def generate_random_architecture(self, input_dim, output_dim, depth=3):
        """
        Generates a random valid NeuroDSL string.
        """
        layers = []
        current_dim = input_dim
        
        for _ in range(depth):
            next_dim = random.choice([64, 128, 256, 512])
            layers.append(f"[{current_dim}, {next_dim}]")
            
            # Optional fancy block
            if random.random() < 0.5:
                # Must accept next_dim and output next_dim
                block = random.choice(["trans", "residual", "moe"])
                if block == "trans":
                    layers.append(f"trans: [{next_dim}]")
                elif block == "residual":
                    layers.append(f"residual: [{next_dim}, 4]")
                elif block == "moe":
                    layers.append(f"moe: [{next_dim}, 4, 0]")
                    
            # Dropout
            if random.random() < 0.3:
                layers.append("dropout: [0.3]")
                
            current_dim = next_dim
            
        # Final layer
        layers.append(f"[{current_dim}, {output_dim}]")
        
        return ", ".join(layers)
