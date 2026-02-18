import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from parser_utils import create_modern_nn, parse_program


class TestChampionPackage(unittest.TestCase):
    def test_champion_package_generation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dsl_path = tmp / "toy.dsl"
            model_path = tmp / "toy.pth"
            out_dir = tmp / "pkg"

            dsl = "[4, 8], [8, 2]"
            dsl_path.write_text(dsl, encoding="utf-8")

            defs = parse_program(dsl)
            model = create_modern_nn(defs)
            torch.save(model.state_dict(), model_path)

            cmd = [
                "python",
                "omni_cli.py",
                "champion-package",
                "--model",
                str(model_path),
                "--dsl-file",
                str(dsl_path),
                "--output-dir",
                str(out_dir),
                "--name",
                "ToyChampion",
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

            self.assertTrue((out_dir / "champion_model.pth").exists())
            self.assertTrue((out_dir / "champion_model.dsl").exists())
            self.assertTrue((out_dir / "champion_interact.py").exists())
            self.assertTrue((out_dir / "champion_manifest.json").exists())
            self.assertTrue((out_dir / "run_champion.bat").exists())
            self.assertTrue((out_dir / "README_champion.txt").exists())


if __name__ == "__main__":
    unittest.main()
