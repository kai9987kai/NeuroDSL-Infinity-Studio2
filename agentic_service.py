from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neuro_sdk import NeuroLab
import torch
from visual_tool import NeuralVisualizer
from vault_service import VaultService
import os
from typing import List, Optional, Dict, Any

app = FastAPI(title="NeuroDSL Agentic API", version="1.0.0 ASI")
sdk = NeuroLab()
API_LOG_QUEUE = []

def log_api(msg: str):
    import time
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    API_LOG_QUEUE.append(entry)
    if len(API_LOG_QUEUE) > 100:
        API_LOG_QUEUE.pop(0)

class DSLRequest(BaseModel):
    dsl_code: str
    preset_name: Optional[str] = None

class TrainRequest(BaseModel):
    epochs: int = 10
    lr: float = 0.001

class VisualRequest(BaseModel):
    params: Dict[str, Any]

class KnowledgeRequest(BaseModel):
    query: str
    lang: str = "en"

@app.get("/")
async def root():
    return {"status": "ONLINE", "mode": "ASI-PROMPT", "device": "CUDA" if torch.cuda.is_available() else "CPU"}

@app.post("/build")
async def build_model(req: DSLRequest):
    log_api(f"BUILD called: {req.dsl_code[:50]}...")
    try:
        model = sdk.build(req.dsl_code)
        params = sum(p.numel() for p in model.parameters())
        return {"status": "SUCCESS", "parameters": params, "layers": len(sdk.trainer.model.layers)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/auto")
async def train_auto(req: TrainRequest):
    # Generates dummy data and trains
    try:
        if not sdk.model:
             raise HTTPException(status_code=400, detail="Build model first.")
        
        # Simulate data
        X = torch.randn(100, sdk.model.layers[0].in_features if hasattr(sdk.model.layers[0], 'in_features') else 16)
        y = torch.randn(100, 10) 
        
        loss = sdk.train(X, y, epochs=req.epochs, lr=req.lr)
        return {"status": "FINISHED", "final_loss": loss}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize")
async def interact_visualizer(req: VisualRequest):
    """Callable data tool for the Lab Visualizer."""
    log_api(f"VISUALIZE called with params: {req.params}")
    viz = NeuralVisualizer()
    return viz.process_state(req.params)

@app.post("/knowledge/query")
async def query_knowledge(req: KnowledgeRequest):
    """Enables AI to query the Global Multilingual Dictionary."""
    vault = VaultService()
    import sqlite3
    conn = sqlite3.connect(vault.db_path)
    c = conn.cursor()
    c.execute("SELECT phrase, language, context FROM global_knowledge WHERE phrase LIKE ? AND language = ?", 
              (f"%{req.query}%", req.lang))
    results = c.fetchall()
    conn.close()
    return [{"phrase": r[0], "lang": r[1], "context": r[2]} for r in results]

@app.post("/dream/start")
async def start_dream_cycle():
    """Triggers an autonomous REM cycle for the current model."""
    if not sdk.model:
        raise HTTPException(status_code=400, detail="Build a model first.")
    from dream_engine import REMCycle
    # Simulated experience buffer for API-driven dreaming
    buffer = [torch.randn(1, 8) for _ in range(10)]
    dreamer = REMCycle(sdk.model, buffer)
    logs = dreamer.perform_dream_session()
    return {"status": "SUCCESS", "consensus_entropy": logs}

@app.get("/manifesto")
async def get_manifesto():
    """Returns documentation for AI Agents to use this tool."""
    return {
        "agent_instructions": "You are a Neuro-Architect. Use /build to define architectures, /visualize to get data feedback, and /dream to optimize weights.",
        "dsl_syntax": "fractal: [dim, depth], moe: [dim, experts], mamba: [dim], liquid: [dim], hyper: [dim], imagine: [dim], diamond: [dim]",
        "tools": {
            "visualizer": "/visualize",
            "dream_engine": "/dream/start",
            "global_dictionary": "/knowledge/query"
        },
        "api_v": "8.0 (Phase 18)"
    }

def start_api_thread():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    start_api_thread()
