"""Generate multi-language connectors for NeuroDSL local APIs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def detect_data_libraries() -> dict:
    probes = {
        "numpy": "numpy",
        "pandas": "pandas",
        "polars": "polars",
        "pyarrow": "pyarrow",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "torch": "torch",
        "tensorflow": "tensorflow",
        "opencv": "cv2",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }
    out = {}
    for label, mod in probes.items():
        out[label] = importlib.util.find_spec(mod) is not None
    return out


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def scaffold_polyglot_connectors(
    out_dir: str = "outputs/polyglot_bridge",
    base_url: str = "http://127.0.0.1:8090",
) -> dict:
    root = Path(out_dir)
    files = []

    py_client = f"""import json
import urllib.request

BASE = "{base_url}"

def infer(inputs):
    payload = json.dumps({{"inputs": inputs}}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/session/infer",
        data=payload,
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))

if __name__ == "__main__":
    print(infer([[0.1] * 16]))
"""
    files.append(_write(root / "python" / "client.py", py_client))

    js_client = f"""const BASE = "{base_url}";

export async function infer(inputs) {{
  const r = await fetch(`${{BASE}}/session/infer`, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ inputs }}),
  }});
  return await r.json();
}}

if (import.meta.main) {{
  infer([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
    .then(console.log)
    .catch(console.error);
}}
"""
    files.append(_write(root / "javascript" / "client.mjs", js_client))

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NeuroDSL Bridge</title>
  <link rel="stylesheet" href="./styles.css" />
</head>
<body>
  <main class="shell">
    <h1>NeuroDSL Agent Bridge</h1>
    <p>Connect HTML/CSS/JS clients to the local model API.</p>
    <textarea id="inp">[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]</textarea>
    <button id="run">Infer</button>
    <pre id="out"></pre>
  </main>
  <script src="./app.js" type="module"></script>
</body>
</html>
"""
    css = """:root {
  --bg: #f2f4f8;
  --card: #ffffff;
  --ink: #14181f;
  --accent: #0a84ff;
  --line: #d5dbe3;
}
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, sans-serif;
  background: radial-gradient(circle at top right, #dae6ff, var(--bg) 40%);
  color: var(--ink);
}
.shell {
  max-width: 720px;
  margin: 24px auto;
  padding: 20px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
}
textarea {
  width: 100%;
  min-height: 120px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}
button {
  margin-top: 12px;
  border: 0;
  border-radius: 8px;
  padding: 10px 14px;
  color: #fff;
  background: var(--accent);
  cursor: pointer;
}
pre {
  white-space: pre-wrap;
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f7f9fc;
}
"""
    app_js = f"""const BASE = "{base_url}";
const inp = document.getElementById("inp");
const out = document.getElementById("out");
document.getElementById("run").addEventListener("click", async () => {{
  try {{
    const inputs = JSON.parse(inp.value);
    const r = await fetch(`${{BASE}}/session/infer`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ inputs }}),
    }});
    const payload = await r.json();
    out.textContent = JSON.stringify(payload, null, 2);
  }} catch (err) {{
    out.textContent = String(err);
  }}
}});
"""
    files.append(_write(root / "web" / "index.html", html))
    files.append(_write(root / "web" / "styles.css", css))
    files.append(_write(root / "web" / "app.js", app_js))

    go_client = f"""package main

import (
  "bytes"
  "encoding/json"
  "fmt"
  "net/http"
)

func main() {{
  payload := map[string]interface{{}}{{"inputs": [][]float64{{{{0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}}}}}}
  b, _ := json.Marshal(payload)
  resp, err := http.Post("{base_url}/session/infer", "application/json", bytes.NewReader(b))
  if err != nil {{
    panic(err)
  }}
  defer resp.Body.Close()
  var out map[string]interface{{}}
  json.NewDecoder(resp.Body).Decode(&out)
  fmt.Println(out)
}}
"""
    files.append(_write(root / "go" / "client.go", go_client))

    java_client = f"""import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class NeuroClient {{
  public static void main(String[] args) throws Exception {{
    var client = HttpClient.newHttpClient();
    var body = "{{\\"inputs\\":[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]}}";
    var req = HttpRequest.newBuilder()
      .uri(URI.create("{base_url}/session/infer"))
      .header("Content-Type", "application/json")
      .POST(HttpRequest.BodyPublishers.ofString(body))
      .build();
    var resp = client.send(req, HttpResponse.BodyHandlers.ofString());
    System.out.println(resp.body());
  }}
}}
"""
    files.append(_write(root / "java" / "NeuroClient.java", java_client))

    cs_client = f"""using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

class NeuroClient {{
  static async Task Main() {{
    using var client = new HttpClient();
    var body = "{{\\"inputs\\":[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]}}";
    var content = new StringContent(body, Encoding.UTF8, "application/json");
    var resp = await client.PostAsync("{base_url}/session/infer", content);
    Console.WriteLine(await resp.Content.ReadAsStringAsync());
  }}
}}
"""
    files.append(_write(root / "csharp" / "NeuroClient.cs", cs_client))

    manifest = {
        "base_url": base_url,
        "files": files,
        "data_libraries": detect_data_libraries(),
    }
    files.append(_write(root / "manifest.json", json.dumps(manifest, indent=2)))
    manifest["manifest_path"] = str(root / "manifest.json")
    return manifest
