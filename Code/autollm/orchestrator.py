import os
import time
import uuid
from typing import Callable, Optional, Dict, Any

from autollm.intent.schemas import (
    PipelineSpec,
    PipelineProgress,
    StepStatus,
    DeploymentPackage,
    RouteType
)
from autollm.intent.parser import IntentParser
from autollm.datasets.searcher import DatasetSearcher
from autollm.engine.generator import CodeGenerator
from autollm.engine.runner import ScriptRunner
from autollm.deploy.package_builder import PackageBuilder

class AutoLLMOrchestrator:
    """
    Master Agent Orchestrator:
    Prompt -> Intent Understanding -> Dataset Acquisition -> Code Generation -> Training Execution -> Deployable Package.
    """

    def __init__(self, base_jobs_dir: str = "jobs", force_cpu: bool = True):
        self.base_jobs_dir = base_jobs_dir
        self.force_cpu = force_cpu
        self.parser = IntentParser(force_cpu=force_cpu)
        self.searcher = DatasetSearcher(cache_dir=os.path.join(base_jobs_dir, "cache"))
        os.makedirs(self.base_jobs_dir, exist_ok=True)

    def run(
        self,
        prompt: str,
        chosen_dataset_id: Optional[str] = None,
        training_mode: Optional[str] = "qwen_lora",
        on_progress: Optional[Callable[[PipelineProgress], None]] = None
    ) -> Dict[str, Any]:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job_dir = os.path.abspath(os.path.join(self.base_jobs_dir, job_id))
        os.makedirs(job_dir, exist_ok=True)

        progress = PipelineProgress(
            job_id=job_id,
            step="intent_parsing",
            percentage=5.0,
            status=StepStatus.RUNNING,
            message=f"Analyzing user prompt: '{prompt}'"
        )
        if on_progress:
            on_progress(progress)

        # 1. Parse Intent
        spec: PipelineSpec = self.parser.parse(prompt)
        if training_mode:
            spec.training_mode = training_mode
            if training_mode == "qwen_lora":
                spec.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
                spec.route = RouteType.PEFT_FINETUNE
            else:
                spec.base_model = None
                spec.route = RouteType.FROM_SCRATCH

        progress.logs.append(f"Model Engine: {'Qwen 2.5 (0.5B) LoRA Fine-Tuning' if spec.training_mode == 'qwen_lora' else 'Scratch Transformer (CPU)'}")
        progress.logs.append(f"Intent extracted: Domain={spec.domain}, Task={spec.task_type.value}, Target Size={spec.target_size_mb}MB, Device={spec.target_device.value}")
        progress.logs.append(f"Feasibility: {spec.hardware_estimate.feasibility_notes}")
        progress.step = "dataset_acquisition"
        progress.percentage = 20.0
        progress.message = f"Sourcing and curating {spec.domain} dataset from web..."
        if on_progress:
            on_progress(progress)

        # 2. Acquire & Format Dataset
        data_dir = os.path.join(job_dir, "data")
        if chosen_dataset_id:
            progress.logs.append(f"User selected dataset: '{chosen_dataset_id}'. Fetching from web...")
            dataset_spec = self.searcher.acquire_chosen_dataset(chosen_dataset_id, spec.domain, data_dir)
        else:
            top_3 = self.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)
            best_choice = top_3[0].id if top_3 else spec.domain
            progress.logs.append(f"Auto-selected top ranked web dataset: '{best_choice}'")
            dataset_spec = self.searcher.acquire_chosen_dataset(best_choice, spec.domain, data_dir)

        progress.logs.append(f"Acquired dataset '{dataset_spec.name}': {dataset_spec.train_samples} train / {dataset_spec.val_samples} val samples.")
        progress.step = "code_generation"
        progress.percentage = 35.0
        progress.message = "Writing custom PyTorch training code tailored for CPU..."
        if on_progress:
            on_progress(progress)

        # 3. Generate Training Script
        train_script = os.path.join(job_dir, "train_run.py")
        CodeGenerator.generate_training_script(
            spec=spec,
            dataset_spec=dataset_spec,
            output_script_path=train_script,
            job_dir=job_dir
        )
        progress.logs.append(f"Generated standalone training script: {os.path.basename(train_script)}")
        progress.step = "training_execution"
        progress.percentage = 45.0
        progress.message = "Running model training on CPU..."
        if on_progress:
            on_progress(progress)

        # 4. Execute Training Runner
        def handle_event(evt: Dict[str, Any]):
            evt_type = evt.get("type")
            if evt_type == "status":
                progress.logs.append(evt.get("message", ""))
                progress.message = evt.get("message", "")
            elif evt_type == "progress":
                progress.epoch = evt.get("epoch")
                progress.total_epochs = evt.get("total_epochs")
                progress.current_loss = evt.get("loss")
                pct_base = 45.0
                pct_span = 40.0
                progress.percentage = round(pct_base + (evt.get("percentage", 0) / 100.0) * pct_span, 1)
                progress.message = f"Training Epoch {progress.epoch}/{progress.total_epochs} (Loss: {progress.current_loss})"
            if on_progress:
                on_progress(progress)

        def handle_log(log_line: str):
            progress.logs.append(log_line)
            if on_progress:
                on_progress(progress)

        metrics = ScriptRunner.execute(
            script_path=train_script,
            cwd=job_dir,
            on_event=handle_event,
            on_log=handle_log
        )

        # 5. Build Deployment Package
        progress.step = "packaging_deployment"
        progress.percentage = 90.0
        progress.message = "Generating zero-setup FastAPI server & interactive playground..."
        if on_progress:
            on_progress(progress)

        pkg: DeploymentPackage = PackageBuilder.build_package(job_id=job_id, job_dir=job_dir)
        progress.logs.append(f"Model package built successfully! Size: {pkg.model_size_mb} MB.")
        progress.logs.append(f"Artifact location: {pkg.model_dir}")

        progress.step = "completed"
        progress.percentage = 100.0
        progress.status = StepStatus.COMPLETED
        progress.message = "Your domain model is fully trained, packaged, and ready for immediate use!"
        progress.model_artifact_path = pkg.model_dir
        if on_progress:
            on_progress(progress)

        return {
            "job_id": job_id,
            "status": "success",
            "spec": spec.model_dump(),
            "dataset": dataset_spec.model_dump(),
            "package": pkg.model_dump(),
            "metrics": metrics,
            "logs": progress.logs
        }
