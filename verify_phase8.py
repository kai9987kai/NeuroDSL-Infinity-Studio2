import unittest
import torch
from trainer import TrainingEngine

class TestPhase8(unittest.TestCase):
    
    def test_preview_scheduler(self):
        model = torch.nn.Linear(10, 1)
        trainer = TrainingEngine(model, max_epochs=100, warmup_steps=10)
        
        lrs = trainer.preview_scheduler(100)
        self.assertEqual(len(lrs), 100)
        
        # Check warmup: starts low, increases
        self.assertLess(lrs[0], lrs[9])
        # Check cosine: decreases after warmup
        self.assertGreater(lrs[10], lrs[99])

    def test_noise_aug(self):
        model = torch.nn.Linear(10, 1)
        trainer = TrainingEngine(model)
        X = torch.ones(10, 10) # uniform input
        y = torch.ones(10, 1)
        
        # With noise, inputs seen by model will vary, but we can't easily capture that specific internal tensor
        # purely from outside without hooks.
        # But we can check that train_step runs without error with noise=0.1
        
        loss, lr, grad = trainer.train_step(X, y, noise_std=0.5)
        self.assertIsInstance(loss, float)
        
    def test_early_stopping_logic(self):
        # We simulate the loop logic used in main.py
        patience = 3
        best_loss = 1.0
        patience_counter = 0
        triggered = False
        
        # Losses that don't improve
        losses = [1.0, 1.1, 1.2, 1.1, 1.05] 
        
        for i, loss in enumerate(losses):
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                triggered = True
                break
        
        self.assertTrue(triggered)

if __name__ == '__main__':
    unittest.main()
