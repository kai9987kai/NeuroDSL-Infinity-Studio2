import unittest
from genetic_engine import NeuralGenome, EvolutionManager

class TestGeneticNAS(unittest.TestCase):
    def test_mutation(self):
        genome = NeuralGenome([{'type': 'mamba', 'dim': 128}, {'type': 'linear', 'in': 128, 'out': 10}])
        original_genes = list(genome.genes)
        layer_pool = [{'type': 'liquid', 'dim': 128}]
        
        # Mutate until something changes
        changed = False
        for _ in range(100):
            genome.mutate(layer_pool)
            if genome.genes != original_genes:
                changed = True
                break
        self.assertTrue(changed)

    def test_evolution_step(self):
        manager = EvolutionManager(population_size=4)
        def fitness_fn(g):
            return len(g.genes) * 0.1 # Fitness proportional to complexity for test
            
        best = manager.evolve_step(fitness_fn)
        self.assertEqual(manager.generation, 1)
        self.assertGreater(len(manager.population), 0)

if __name__ == "__main__":
    unittest.main()
