import random
import copy
import torch
import torch.nn as nn

class NeuralGenome:
    """
    Bio-inspired representation of a NeuroDSL architecture.
    """
    def __init__(self, genes=None):
        # Genes = list of layer definitions (dicts)
        self.genes = genes or []
        self.fitness = 0.0

    def mutate(self, layer_pool):
        """Randomly alters a layer or inserts a new one."""
        if not self.genes: return
        idx = random.randint(0, len(self.genes) - 1)
        mutation_type = random.choice(["change", "insert", "delete"])
        
        if mutation_type == "change":
            self.genes[idx] = random.choice(layer_pool)
        elif mutation_type == "insert":
            self.genes.insert(idx, random.choice(layer_pool))
        elif mutation_type == "delete" and len(self.genes) > 2:
            self.genes.pop(idx)

    def crossover(self, other):
        """Combines genes with another genome."""
        cut = random.randint(1, min(len(self.genes), len(other.genes)) - 1)
        new_genes = self.genes[:cut] + other.genes[cut:]
        return NeuralGenome(new_genes)

class EvolutionManager:
    """
    Manages a population of models and selects for fitness.
    """
    def __init__(self, population_size=10):
        self.population = [NeuralGenome() for _ in range(population_size)]
        self.generation = 0
        self.layer_pool = [
            {'type': 'mamba', 'dim': 128},
            {'type': 'moe', 'dim': 128, 'experts': 8},
            {'type': 'liquid', 'dim': 128},
            {'type': 'quantum', 'dim': 128},
            {'type': 'fractal_synth', 'dim': 128}
        ]

    def evolve_step(self, fitness_callback):
        """Runs one generation of evolution."""
        # 1. Evaluate
        for genome in self.population:
            genome.fitness = fitness_callback(genome)
            
        # 2. Select (Elite survival)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elites = self.population[:2]
        
        # 3. Repopulate via mutation/crossover
        new_pop = elites
        while len(new_pop) < len(self.population):
            if random.random() > 0.5:
                parent = random.choice(elites)
                child = copy.deepcopy(parent)
                child.mutate(self.layer_pool)
                new_pop.append(child)
            else:
                p1, p2 = random.sample(elites, 2)
                new_pop.append(p1.crossover(p2))
                
        self.population = new_pop
        self.generation += 1
        return self.population[0] # Return the best genome
