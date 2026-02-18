import unittest
import os
import shutil
from unittest.mock import MagicMock, patch
import nas_lite
import app_builder

class TestPhase5(unittest.TestCase):
    def setUp(self):
        self.nas = nas_lite.NASLite()

    def test_nas_parsing(self):
        in_dim, out_dim = self.nas.parse_dimensions("[784, 128], [128, 10]")
        self.assertEqual(in_dim, 784)
        self.assertEqual(out_dim, 128)

    def test_nas_mutation(self):
        dsl = "[784, 128], [128, 10]"
        # Mutate multiple times to ensure something changes
        mutated = dsl
        changed = False
        for _ in range(10):
            mutated = self.nas.mutate_dsl(dsl)
            if mutated != dsl:
                changed = True
                break
        self.assertTrue(changed, "Mutation should eventually change the DSL")
        print(f"Original: {dsl}")
        print(f"Mutated:  {mutated}")

    def test_app_builder_script_gen(self):
        dsl = "[784, 10]"
        model_path = "test_model.pth"
        output_path = "test_app.py"
        
        path = app_builder.generate_inference_script(dsl, model_path, output_path)
        self.assertTrue(os.path.exists(path))
        
        with open(path, "r") as f:
            content = f.read()
            
        self.assertIn('DSL_CODE = "[784, 10]"', content)
        self.assertIn('MODEL_PATH = "test_model.pth"', content)
        self.assertIn('import FreeSimpleGUI as sg', content)
        
        os.remove(path)

    @patch('subprocess.check_call')
    def test_build_exe_call(self, mock_subprocess):
        # Mock subprocess to avoid actually running pyinstaller (time consuming/environment dependent)
        success, path = app_builder.build_exe("test_app.py", "model.pth")
        
        self.assertTrue(success)
        self.assertIn("dist/NeuroDSL_App.exe", path)
        mock_subprocess.assert_called()

if __name__ == '__main__':
    unittest.main()
