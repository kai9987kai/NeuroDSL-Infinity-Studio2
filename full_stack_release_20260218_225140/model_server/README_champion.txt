Champion Package: NeuroDSL Champion v2

Run with Python:
  C:\Users\kai99\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe champion_interact.py --mode repl

One-shot inference:
  C:\Users\kai99\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe champion_interact.py --mode infer --input "0.1,0.2,..."

Batch inference:
  C:\Users\kai99\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe champion_interact.py --mode batch --input-csv inputs.csv --output-csv preds.csv

Serve HTTP:
  C:\Users\kai99\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe champion_interact.py --mode serve --host 127.0.0.1 --port 8092
