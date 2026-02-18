# NeuroDSL Champion V2 Web Console

This is a standalone static webpage for interacting with the local champion model server.

## 1) Start model server

Use either script or EXE:

```powershell
outputs/champion_release_v2/dist/NeuroDSL_Champion_v2.exe --mode serve --host 127.0.0.1 --port 8092
```

Or:

```powershell
python outputs/champion_release_v2/champion_interact.py --mode serve --host 127.0.0.1 --port 8092
```

## 2) Open the webpage

Serve this folder and open the URL:

```powershell
cd web_champion_v2
python -m http.server 8787
```

Then open:

`http://127.0.0.1:8787`

## 3) Use

- Check health.
- Run inference with 32-value vectors.
- Run quick benchmark.
