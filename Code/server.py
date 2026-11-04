import os
import sys
import uuid
import json
import time
import psutil
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure Code directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autollm.orchestrator import AutoLLMOrchestrator
from autollm.intent.schemas import PipelineProgress, StepStatus
from autollm.engine.design_preview import ArchitecturePlanner
from autollm.engine.inference import get_inference_engine, DomainInferenceEngine, QwenInferenceEngine
from autollm.security import sanitize_job_id, safe_join_paths, sanitize_user_prompt, validate_inference_params, validate_uploaded_file

app = FastAPI(title="AutoLLM Builder Web Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs"))
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job state store
jobs_state: Dict[str, Dict[str, Any]] = {}
loaded_models: Dict[str, Any] = {}

class FindDatasetsRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)

class BuildRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    dataset_id: Optional[str] = None
    training_mode: Optional[str] = "qwen_lora"
    curation_mode: Optional[str] = "web_search"
    document_text: Optional[str] = None

class ChatRequest(BaseModel):
    job_id: str = Field(..., max_length=100)
    prompt: str = Field(..., max_length=2000)
    max_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.01, le=2.0)

class CompareRequest(BaseModel):
    job_id: str = Field(..., max_length=100)
    query: str = Field(..., max_length=2000)
    max_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.01, le=2.0)

class DesignPreviewRequest(BaseModel):
    prompt: Optional[str] = ""
    target_size_mb: float = 1024.0
    domain: str = "general"
    context_window: int = 512

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autollm", "web", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AutoLLM Builder Dashboard</h1><p>UI loading error.</p>"

@app.get("/api/system-metrics")
def get_system_metrics():
    vm = psutil.virtual_memory()
    return {
        "cpu_cores": os.cpu_count() or 4,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
        "used_ram_percent": vm.percent,
        "device": "CPU (Multi-threaded)"
    }

@app.post("/api/find-datasets")
def find_datasets(req: FindDatasetsRequest):
    clean_prompt = sanitize_user_prompt(req.prompt)
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    orchestrator = AutoLLMOrchestrator(base_jobs_dir=JOBS_DIR, force_cpu=True)
    spec = orchestrator.parser.parse(clean_prompt)
    datasets = orchestrator.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)

    return {
        "domain": spec.domain,
        "task_type": spec.task_type.value,
        "query": spec.dataset_query,
        "route": spec.route.value,
        "route_explanation": spec.route_explanation,
        "datasets": [d.model_dump() for d in datasets]
    }

@app.post("/api/design-preview")
def preview_architecture(req: DesignPreviewRequest):
    target_mb = max(10.0, min(float(req.target_size_mb), 16384.0))
    domain = (req.domain or "general").strip().lower()
    report = ArchitecturePlanner.design_architecture(
        target_size_mb=target_mb,
        domain=domain,
        context_window=req.context_window,
        vocab_size=32000
    )
    return report.model_dump()

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    OWASP A03/A04 Secure File Upload Endpoint.
    Accepts .txt, .md, .json, or .csv documents, validates MIME, size, and encoding,
    and returns sanitized text for RAG indexing.
    """
    try:
        content_bytes = await file.read()
        clean_text = validate_uploaded_file(file.filename or "document.txt", content_bytes, max_size_mb=5.0)
        word_count = len(clean_text.split())
        char_count = len(clean_text)
        return {
            "status": "success",
            "filename": file.filename,
            "word_count": word_count,
            "char_count": char_count,
            "text": clean_text
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

def run_job_in_background(
    job_id: str,
    prompt: str,
    dataset_id: Optional[str] = None,
    training_mode: Optional[str] = None,
    curation_mode: Optional[str] = "web_search",
    document_text: Optional[str] = None
):
    orchestrator = AutoLLMOrchestrator(base_jobs_dir=JOBS_DIR, force_cpu=True)

    def on_progress(p: PipelineProgress):
        if job_id in jobs_state:
            jobs_state[job_id]["progress"] = p.model_dump()

    try:
        result = orchestrator.run(
            prompt=prompt,
            job_id=job_id,
            chosen_dataset_id=dataset_id,
            training_mode=training_mode,
            curation_mode=curation_mode,
            document_text=document_text,
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
    clean_prompt = sanitize_user_prompt(req.prompt)
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs_state[job_id] = {
        "job_id": job_id,
        "prompt": clean_prompt,
        "dataset_id": req.dataset_id,
        "training_mode": req.training_mode,
        "curation_mode": req.curation_mode or "web_search",
        "status": "running",
        "progress": {
            "job_id": job_id,
            "step": "initializing",
            "percentage": 2.0,
            "status": "running",
            "message": "Initializing build pipeline across 4-Way Decision Router...",
            "logs": ["AutoLLM Builder Agent initialized on CPU (SOLID/OWASP Pipeline Active)."]
        }
    }

    background_tasks.add_task(
        run_job_in_background,
        job_id,
        clean_prompt,
        req.dataset_id,
        req.training_mode,
        req.curation_mode or "web_search",
        req.document_text
    )
    return {"job_id": job_id, "status": "started"}

def ensure_job_loaded(job_id: str):
    clean_id = sanitize_job_id(job_id)
    if clean_id not in jobs_state:
        job_dir = safe_join_paths(JOBS_DIR, clean_id)
        export_dir = os.path.join(job_dir, "export")
        has_artifacts = (
            os.path.exists(os.path.join(job_dir, "adapter")) or 
            os.path.exists(os.path.join(export_dir, "adapter")) or
            os.path.exists(os.path.join(job_dir, "model.pt")) or 
            os.path.exists(os.path.join(job_dir, "model_quantized.pt")) or 
            os.path.exists(os.path.join(export_dir, "model_quantized.pt")) or
            os.path.exists(os.path.join(job_dir, "vector_store.json")) or
            os.path.exists(os.path.join(job_dir, "recipe.yaml"))
        )
        if has_artifacts:
            metrics = {}
            for loc in [os.path.join(job_dir, "eval_metrics.json"), os.path.join(export_dir, "eval_metrics.json")]:
                if os.path.exists(loc):
                    try:
                        with open(loc, "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                            break
                    except Exception:
                        pass

            quant_metrics = {}
            for loc in [os.path.join(job_dir, "quantization_metrics.json"), os.path.join(export_dir, "quantization_metrics.json")]:
                if os.path.exists(loc):
                    try:
                        with open(loc, "r", encoding="utf-8") as f:
                            quant_metrics = json.load(f)
                            break
                    except Exception:
                        pass

            jobs_state[clean_id] = {
                "job_id": clean_id,
                "status": "completed",
                "progress": {
                    "job_id": clean_id,
                    "step": "completed",
                    "percentage": 100.0,
                    "status": "completed",
                    "message": "Model ready for use."
                },
                "result": {
                    "job_id": clean_id,
                    "status": "success",
                    "spec": {"domain": "general", "base_model": "Qwen 2.5 LoRA / MiniGPT"},
                    "package": {
                        "model_dir": export_dir if os.path.exists(export_dir) else job_dir,
                        "quantization_metrics": quant_metrics
                    },
                    "metrics": metrics
                }
            }

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    clean_id = sanitize_job_id(job_id)
    ensure_job_loaded(clean_id)
    if clean_id not in jobs_state:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_state[clean_id]

@app.post("/api/chat")
def test_model(req: ChatRequest):
    clean_id = sanitize_job_id(req.job_id)
    ensure_job_loaded(clean_id)
    if clean_id not in jobs_state or jobs_state[clean_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job has not completed yet.")

    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    domain = jobs_state[clean_id].get("result", {}).get("spec", {}).get("domain", "general")

    if clean_id not in loaded_models:
        loaded_models[clean_id] = get_inference_engine(job_dir, domain=domain)

    engine = loaded_models[clean_id]
    temp, tokens = validate_inference_params(req.temperature, req.max_tokens)
    clean_prompt = sanitize_user_prompt(req.prompt)

    t0 = time.time()
    try:
        completion = engine.generate_response(
            user_query=clean_prompt,
            max_tokens=tokens,
            temperature=temp
        )
    except Exception as e:
        completion = f"Inference notice: {str(e)}"
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "job_id": clean_id,
        "prompt": clean_prompt,
        "completion": completion,
        "full_text": completion,
        "latency_ms": latency_ms,
        "engine_type": engine.__class__.__name__
    }

@app.post("/api/compare")
def compare_models(req: CompareRequest):
    clean_id = sanitize_job_id(req.job_id)
    ensure_job_loaded(clean_id)
    if clean_id not in jobs_state or jobs_state[clean_id].get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job has not completed yet.")

    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    domain = jobs_state[clean_id].get("result", {}).get("spec", {}).get("domain", "general")

    # Baseline Engine (Untrained / General fallback)
    baseline_engine = DomainInferenceEngine(domain=domain, dataset_path=None)

    # Fine-Tuned / Adapted Engine
    if clean_id not in loaded_models:
        loaded_models[clean_id] = get_inference_engine(job_dir, domain=domain)
    finetuned_engine = loaded_models[clean_id]

    temp, tokens = validate_inference_params(req.temperature, req.max_tokens)
    clean_query = sanitize_user_prompt(req.query)

    t_base = time.time()
    try:
        base_out = baseline_engine.generate_response(clean_query, max_tokens=tokens, temperature=temp)
    except Exception as e:
        base_out = f"Baseline error: {str(e)}"
    base_latency_ms = round((time.time() - t_base) * 1000, 1)

    t_ft = time.time()
    try:
        ft_out = finetuned_engine.generate_response(clean_query, max_tokens=tokens, temperature=temp)
    except Exception as e:
        ft_out = f"Adapted model error: {str(e)}"
    ft_latency_ms = round((time.time() - t_ft) * 1000, 1)

    return {
        "job_id": clean_id,
        "query": clean_query,
        "baseline_output": base_out,
        "finetuned_output": ft_out,
        "baseline_latency_ms": base_latency_ms,
        "finetuned_latency_ms": ft_latency_ms,
        "engine_type": finetuned_engine.__class__.__name__
    }

@app.get("/api/download/{job_id}")
def download_package(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    zip_file = os.path.join(job_dir, f"autollm_model_{clean_id}.zip")
    if not os.path.exists(zip_file):
        raise HTTPException(status_code=404, detail="Model archive missing.")

    return FileResponse(
        zip_file,
        media_type="application/zip",
        filename=f"autollm_{clean_id}.zip"
    )

@app.get("/api/dataset/{job_id}")
def get_job_dataset(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    data_file = os.path.join(job_dir, "data", "train.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "export", "knowledge.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "knowledge.jsonl")
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
    if clean_id in jobs_state and "result" in jobs_state[clean_id]:
        dataset_name = jobs_state[clean_id]["result"].get("dataset", {}).get("name", dataset_name)

    return {
        "job_id": clean_id,
        "dataset_name": dataset_name,
        "total_samples": len(samples),
        "preview_samples": samples[:10]
    }

@app.get("/api/dataset-download/{job_id}")
def download_dataset_file(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    data_file = os.path.join(job_dir, "data", "train.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "export", "knowledge.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(job_dir, "knowledge.jsonl")
    if not os.path.exists(data_file):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return FileResponse(
        data_file,
        media_type="application/jsonl",
        filename=f"dataset_{clean_id}.jsonl"
    )

@app.get("/api/recipe/{job_id}")
def get_recipe_yaml(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    recipe_file = os.path.join(job_dir, "recipe.yaml")
    if not os.path.exists(recipe_file):
        recipe_file = os.path.join(job_dir, "export", "recipe.yaml")
    if not os.path.exists(recipe_file):
        raise HTTPException(status_code=404, detail="Recipe file not found.")

    with open(recipe_file, "r", encoding="utf-8") as f:
        content = f.read()
    return {"job_id": clean_id, "recipe_yaml": content}

@app.get("/api/recipe-download/{job_id}")
def download_recipe_yaml(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    recipe_file = os.path.join(job_dir, "recipe.yaml")
    if not os.path.exists(recipe_file):
        recipe_file = os.path.join(job_dir, "export", "recipe.yaml")
    if not os.path.exists(recipe_file):
        raise HTTPException(status_code=404, detail="Recipe file not found.")

    return FileResponse(
        recipe_file,
        media_type="application/x-yaml",
        filename=f"recipe_{clean_id}.yaml"
    )

@app.get("/api/eval-report/{job_id}")
def get_eval_report(job_id: str):
    clean_id = sanitize_job_id(job_id)
    job_dir = safe_join_paths(JOBS_DIR, clean_id)
    report_file = os.path.join(job_dir, "eval_report.json")
    if not os.path.exists(report_file):
        report_file = os.path.join(job_dir, "export", "eval_report.json")
    if not os.path.exists(report_file):
        if clean_id in jobs_state and "result" in jobs_state[clean_id]:
            report = jobs_state[clean_id]["result"].get("eval_report", {})
            if report:
                return report
        raise HTTPException(status_code=404, detail="Evaluation report not found.")

    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/synthesize-preview")
def preview_synthetic_dataset(req: FindDatasetsRequest):
    from autollm.datasets.synthesizer import DatasetSynthesizer
    from autollm.intent.parser import IntentParser

    clean_prompt = sanitize_user_prompt(req.prompt)
    parser = IntentParser(force_cpu=True)
    spec = parser.parse(clean_prompt)
    system_prompt = DatasetSynthesizer.generate_system_prompt(spec.domain, spec.task_type.value, clean_prompt)
    generators = DatasetSynthesizer._get_domain_seed_generators(spec.domain, clean_prompt, spec.task_type)

    preview_items = [g(i).model_dump() for i, g in enumerate(generators[:6])]
    return {
        "domain": spec.domain,
        "task_type": spec.task_type.value,
        "system_prompt": system_prompt,
        "samples": preview_items,
        "sample_pairs": preview_items # For backwards compatibility with older clients
    }

if __name__ == "__main__":
    import uvicorn
    print("AutoLLM Builder Web Dashboard running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
