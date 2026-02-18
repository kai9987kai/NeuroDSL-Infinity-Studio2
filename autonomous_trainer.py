import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def autonomous_run():
    print("Starting Autonomous ASI Training Loop...")
    
    # 1. Build Model via API
    dsl = "mamba:[8], imagine:[8], linear:[8, 4]"
    print(f"Building Model: {dsl}")
    requests.post(f"{BASE_URL}/build", json={"dsl_code": dsl})
    
    # 2. Get Visual Insight
    print("Querying Visualizer Tool for initial stability...")
    viz_resp = requests.post(f"{BASE_URL}/visualize", json={"params": {"sampleRange": 75, "sampleSelect": "mode2"}})
    insights = viz_resp.json().get("ai_insights", {})
    print(f"Visual Insights: {insights}")
    
    # 3. Decision Logic
    lr = 0.001
    if not insights.get("is_stable", True):
        print("System unstable! Reducing learning rate for safety.")
        lr = 0.0001
    
    # 4. Global Knowledge Context
    print("Querying Global Knowledge for 'Stochastic Thinking' context...")
    kb_resp = requests.post(f"{BASE_URL}/knowledge/query", json={"query": "Thinking", "lang": "en"})
    print(f"Knowledge Context: {kb_resp.json()}")

    # 5. Execute Training
    print(f"Executing Training with LR={lr}...")
    train_resp = requests.post(f"{BASE_URL}/train/auto", json={"epochs": 5, "lr": lr})
    print(f"Final Loss: {train_resp.json().get('final_loss')}")

    # 6. Optimize via Dream
    print("Initiating REM Consolidation Cycle...")
    dream_resp = requests.post(f"{BASE_URL}/dream/start")
    print(f"Dream Entropy Logs: {dream_resp.json().get('consensus_entropy')}")

if __name__ == "__main__":
    # Ensure API is running before this
    try:
        autonomous_run()
    except Exception as e:
        print(f"Error: Could not connect to API. Start agentic_service.py first. Details: {e}")
