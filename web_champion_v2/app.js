const EXPECTED_DIM = 32;

const apiUrlEl = document.getElementById("apiUrl");
const healthBtn = document.getElementById("healthBtn");
const healthStatusEl = document.getElementById("healthStatus");
const vectorInputEl = document.getElementById("vectorInput");
const randomBtn = document.getElementById("randomBtn");
const sampleBtn = document.getElementById("sampleBtn");
const inferBtn = document.getElementById("inferBtn");
const inputErrorEl = document.getElementById("inputError");
const outputBarsEl = document.getElementById("outputBars");
const jsonOutEl = document.getElementById("jsonOut");
const benchmarkBtn = document.getElementById("benchmarkBtn");
const benchOutEl = document.getElementById("benchOut");
const logBoxEl = document.getElementById("logBox");

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  logBoxEl.textContent += `[${ts}] ${msg}\n`;
  logBoxEl.scrollTop = logBoxEl.scrollHeight;
}

function parseVector(text) {
  const values = text
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .map((v) => Number(v));
  if (values.some((v) => Number.isNaN(v))) {
    throw new Error("Input contains non-numeric values.");
  }
  if (values.length !== EXPECTED_DIM) {
    throw new Error(`Expected ${EXPECTED_DIM} values, got ${values.length}.`);
  }
  return values;
}

function randomVector() {
  const vals = Array.from({ length: EXPECTED_DIM }, () => (Math.random() * 2 - 1).toFixed(4));
  return vals.join(", ");
}

function sampleVector() {
  return Array.from({ length: EXPECTED_DIM }, () => "0.1").join(", ");
}

function setHealth(ok, text) {
  healthStatusEl.textContent = text;
  healthStatusEl.style.color = ok ? "#1fe7b7" : "#ff8282";
}

function renderOutputBars(output) {
  outputBarsEl.innerHTML = "";
  const absMax = Math.max(...output.map((v) => Math.abs(v)), 1e-6);
  output.forEach((value, idx) => {
    const pct = Math.min(100, Math.max(2, (Math.abs(value) / absMax) * 100));
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div>out_${idx}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div>${value.toFixed(6)}</div>
    `;
    outputBarsEl.appendChild(row);
  });
}

async function checkHealth() {
  const base = apiUrlEl.value.replace(/\/+$/, "");
  const url = `${base}/health`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setHealth(true, `Online (${data.device || "?"})`);
    log(`Health OK: ${JSON.stringify(data)}`);
  } catch (err) {
    setHealth(false, "Offline");
    log(`Health failed: ${err.message}`);
  }
}

async function runInference() {
  inputErrorEl.textContent = "";
  const base = apiUrlEl.value.replace(/\/+$/, "");
  const url = `${base}/infer`;
  let vector;
  try {
    vector = parseVector(vectorInputEl.value);
  } catch (err) {
    inputErrorEl.textContent = err.message;
    return;
  }

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: vector }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const data = await res.json();
    const output = (data.outputs && data.outputs[0]) || [];
    if (!Array.isArray(output)) {
      throw new Error("Invalid response format.");
    }
    renderOutputBars(output);
    jsonOutEl.textContent = JSON.stringify(data, null, 2);
    log(`Inference OK (${output.length} outputs).`);
  } catch (err) {
    log(`Inference failed: ${err.message}`);
  }
}

async function runBenchmark() {
  const base = apiUrlEl.value.replace(/\/+$/, "");
  const url = `${base}/infer`;
  const vector = parseVector(vectorInputEl.value);
  const runs = 20;
  const t0 = performance.now();
  for (let i = 0; i < runs; i += 1) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: vector }),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    await res.json();
  }
  const dt = (performance.now() - t0) / runs;
  benchOutEl.textContent = `${dt.toFixed(2)} ms/request`;
  log(`Benchmark done: ${dt.toFixed(2)} ms/request over ${runs} requests.`);
}

healthBtn.addEventListener("click", checkHealth);
randomBtn.addEventListener("click", () => {
  vectorInputEl.value = randomVector();
});
sampleBtn.addEventListener("click", () => {
  vectorInputEl.value = sampleVector();
});
inferBtn.addEventListener("click", runInference);
benchmarkBtn.addEventListener("click", async () => {
  try {
    await runBenchmark();
  } catch (err) {
    log(`Benchmark failed: ${err.message}`);
  }
});

vectorInputEl.value = sampleVector();
log("Console ready.");
