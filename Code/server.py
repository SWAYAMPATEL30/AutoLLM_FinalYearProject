import os
import sys
import uuid
import threading
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Ensure Code directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autollm.orchestrator import AutoLLMOrchestrator
from autollm.intent.schemas import PipelineProgress, StepStatus

app = FastAPI(title="AutoLLM Builder Web Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job state store
jobs_state: Dict[str, Dict[str, Any]] = {}
loaded_models: Dict[str, Any] = {}

class FindDatasetsRequest(BaseModel):
    prompt: str

class BuildRequest(BaseModel):
    prompt: str
    dataset_id: Optional[str] = None
    training_mode: Optional[str] = "qwen_lora"

class ChatRequest(BaseModel):
    job_id: str
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autollm", "web", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AutoLLM Builder Dashboard</h1><p>UI loading error.</p>"

@app.post("/api/find-datasets")
def find_datasets(req: FindDatasetsRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    orchestrator = AutoLLMOrchestrator(base_jobs_dir=JOBS_DIR, force_cpu=True)
    spec = orchestrator.parser.parse(req.prompt)
    datasets = orchestrator.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)

    return {
        "domain": spec.domain,
        "task_type": spec.task_type.value,
        "query": spec.dataset_query,
        "datasets": [d.model_dump() for d in datasets]
    }

def run_job_in_background(job_id: str, prompt: str, dataset_id: Optional[str] = None, training_mode: Optional[str] = "qwen_lora"):
    orchestrator = AutoLLMOrchestrator(base_jobs_dir=JOBS_DIR, force_cpu=True)

    def on_progress(p: PipelineProgress):
        if job_id in jobs_state:
            jobs_state[job_id]["progress"] = p.model_dump()

    try:
        result = orchestrator.run(
            prompt,
            chosen_dataset_id=dataset_id,
            training_mode=training_mode,
            on_progress=on_progress
        )
        jobs_state[job_id]["result"] = result
        jobs_state[job_id]["status"] = "completed"
    except Exception as e:
        jobs_state[job_id]["status"] = "failed"
        jobs_state[job_id]["error"] = str(e)
        if "progress" in jobs_state[job_id]:
            jobs_state[job_id]["progress"]["status"] = StepStatus.FAILED.value
            jobs_state[job_id]["progress"]["message"] = f"Failed: {str(e)}"
            jobs_state[job_id]["progress"]["logs"].append(f"ERROR: {str(e)}")

@app.post("/api/build")
def start_build(req: BuildRequest, background_tasks: BackgroundTasks):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs_state[job_id] = {
        "job_id": job_id,
        "prompt": req.prompt,
        "dataset_id": req.dataset_id,
        "training_mode": req.training_mode or "qwen_lora",
        "status": "running",
        "progress": {
            "job_id": job_id,
            "step": "initializing",
            "percentage": 2.0,
            "status": "running",
            "message": "Initializing build pipeline...",
            "logs": ["AutoLLM Agent initialized on CPU."]
        }
    }

    background_tasks.add_task(run_job_in_background, job_id, req.prompt, req.dataset_id, req.training_mode or "qwen_lora")
    return {"job_id": job_id, "status": "started"}

def ensure_job_loaded(job_id: str):
    if job_id not in jobs_state:
        job_dir = os.path.join(JOBS_DIR, job_id)
        export_dir = os.path.join(job_dir, "export")
        if os.path.exists(export_dir) and os.path.exists(os.path.join(export_dir, "model.pt")):
            metrics = {}
            metrics_path = os.path.join(export_dir, "eval_metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    pass
            jobs_state[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "progress": {
                    "job_id": job_id,
                    "step": "completed",
                    "percentage": 100.0,
                    "status": "completed",
                    "message": "Model ready for use."
                },
                "result": {
                    "job_id": job_id,
                    "status": "success",
                    "spec": {"domain": "medical"},
                    "package": {"model_dir": export_dir},
                    "metrics": metrics
                }
            }

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    ensure_job_loaded(job_id)
    if job_id not in jobs_state:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_state[job_id]

@app.post("/api/chat")
def test_model(req: ChatRequest):
    from autollm.engine.inference import get_inference_engine

    job_id = req.job_id
    ensure_job_loaded(job_id)
    if job_id not in jobs_state or jobs_state[job_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job has not completed yet.")

    job_dir = os.path.join(JOBS_DIR, job_id)
    domain = jobs_state[job_id].get("result", {}).get("spec", {}).get("domain", "general")

    if job_id not in loaded_models:
        loaded_models[job_id] = get_inference_engine(job_dir, domain=domain)

    engine = loaded_models[job_id]
    completion = engine.generate_response(
        user_query=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )

    return {
        "job_id": job_id,
        "prompt": req.prompt,
        "completion": completion,
        "full_text": completion
    }

@app.get("/api/download/{job_id}")
def download_package(job_id: str):
    if job_id not in jobs_state or jobs_state[job_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not found or not completed.")

    zip_file = os.path.join(JOBS_DIR, job_id, f"autollm_model_{job_id}.zip")
    if not os.path.exists(zip_file):
        raise HTTPException(status_code=404, detail="Model archive missing.")

    return FileResponse(
        zip_file,
        media_type="application/zip",
        filename=f"autollm_{job_id}.zip"
    )

@app.get("/api/dataset/{job_id}")
def get_job_dataset(job_id: str):
    job_dir = os.path.join(JOBS_DIR, job_id)
    data_file = os.path.join(job_dir, "data", "train.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "export", "knowledge.jsonl")
    if not os.path.exists(data_file):
        raise HTTPException(status_code=404, detail="Dataset not found for this job.")

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass

    dataset_name = "Auto-Curated Web Dataset"
    if job_id in jobs_state and "result" in jobs_state[job_id]:
        dataset_name = jobs_state[job_id]["result"].get("dataset", {}).get("name", dataset_name)

    return {
        "job_id": job_id,
        "dataset_name": dataset_name,
        "total_samples": len(samples),
        "preview_samples": samples[:10]
    }

@app.get("/api/dataset-download/{job_id}")
def download_dataset_file(job_id: str):
    job_dir = os.path.join(JOBS_DIR, job_id)
    data_file = os.path.join(job_dir, "data", "train.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "export", "knowledge.jsonl")
    if not os.path.exists(data_file):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return FileResponse(
        data_file,
        media_type="application/jsonl",
        filename=f"dataset_{job_id}.jsonl"
    )

if __name__ == "__main__":
    import uvicorn
    print("AutoLLM Builder Web Dashboard running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
