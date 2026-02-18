import torch
import unittest
import time
import socket
from distributed_compute import ComputeServer, ComputeClient
from dream_engine import REMCycle, ImagineLayer

class TestPhase17DistributedDreams(unittest.TestCase):
    
    def test_compute_cluster_registration(self):
        server = ComputeServer(port=12345)
        server.start()
        
        client = ComputeClient('localhost', server_port=12345)
        client.register({'gpu': 'TestGPU'})
        
        time.sleep(0.5) # Give time for socket handle
        nodes = server.get_active_nodes()
        server.running = False # Clean up
        
        self.assertGreater(len(nodes), 0)
        self.assertEqual(nodes[0]['info']['spec']['gpu'], 'TestGPU')

    def test_dream_consolidation(self):
        # Linear model that should favor high confidence (low entropy)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4)
        )
        # Experience buffer
        buffer = [torch.randn(1, 8) for _ in range(5)]
        
        dreamer = REMCycle(model, buffer)
        init_logs = dreamer.perform_dream_session(intensity=0.01, cycles=5)
        
        self.assertEqual(len(init_logs), 5)
        # Ensure model is still functional
        out = model(torch.randn(1, 8))
        self.assertEqual(out.shape, (1, 4))

    def test_imagine_layer(self):
        layer = ImagineLayer(dim=8)
        x = torch.ones(1, 8)
        
        # In training mode, it should be an identity
        layer.train()
        out_train = layer(x)
        self.assertTrue(torch.allclose(x, out_train))
        
        # In eval mode, it should add noise (imagination)
        layer.eval()
        out_eval = layer(x)
        self.assertFalse(torch.allclose(x, out_eval))

if __name__ == "__main__":
    unittest.main()
