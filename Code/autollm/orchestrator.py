import os
import time
import uuid
import json
from typing import Callable, Optional, Dict, Any, List

from autollm.intent.schemas import (
    PipelineSpec,
    PipelineProgress,
    StepStatus,
    DeploymentPackage,
    RouteType,
    DataCurationMode,
    EvaluationReport
)
from autollm.intent.parser import IntentParser
from autollm.datasets.searcher import DatasetSearcher
from autollm.datasets.synthesizer import DatasetSynthesizer
from autollm.engine.generator import CodeGenerator
from autollm.engine.runner import ScriptRunner
from autollm.engine.recipe import RecipeBuilder
from autollm.engine.evaluator import ModelEvaluator
from autollm.engine.inference import get_inference_engine, DomainInferenceEngine, QwenInferenceEngine
from autollm.deploy.package_builder import PackageBuilder

class AutoLLMOrchestrator:
    """
    Master Agent Orchestrator (Oumi & Distil Aligned Architecture):
    1. Intent & Spec Extraction
    2. Data Curation (Web Discovery, Autonomous Synthesis, or Augmentation)
    3. Transparent Recipe Compilation (recipe.yaml)
    4. Pre-Training Baseline Evaluation (Failure Mode Discovery)
    5. Training & Fine-Tuning Execution (CPU-optimized LoRA / MiniGPT)
    6. Post-Training Evaluation & Failure Mode Analysis (Before vs After)
    7. Zero-Setup Serving & Packaging
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
        job_id: Optional[str] = None,
        chosen_dataset_id: Optional[str] = None,
        training_mode: Optional[str] = "qwen_lora",
        curation_mode: Optional[str] = "web_search",
        on_progress: Optional[Callable[[PipelineProgress], None]] = None
    ) -> Dict[str, Any]:
        if not job_id:
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

        # 1. Parse Intent & Extract System Prompt
        spec: PipelineSpec = self.parser.parse(prompt)
        if training_mode:
            spec.training_mode = training_mode
            if training_mode == "qwen_lora":
                spec.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
                spec.route = RouteType.PEFT_FINETUNE
            else:
                spec.base_model = None
                spec.route = RouteType.FROM_SCRATCH

        if curation_mode:
            try:
                spec.curation_mode = DataCurationMode(curation_mode)
            except Exception:
                spec.curation_mode = DataCurationMode.WEB_SEARCH

        spec.system_prompt = DatasetSynthesizer.generate_system_prompt(spec.domain, spec.task_type.value, prompt)

        progress.logs.append(f"Model Engine: {'Qwen 2.5 (0.5B) LoRA Fine-Tuning' if spec.training_mode == 'qwen_lora' else 'Scratch Transformer (CPU)'}")
        progress.logs.append(f"Intent extracted: Domain={spec.domain}, Task={spec.task_type.value}, Target Size={spec.target_size_mb}MB, Device={spec.target_device.value}")
        progress.logs.append(f"System Prompt: \"{spec.system_prompt}\"")
        progress.logs.append(f"Feasibility: {spec.hardware_estimate.feasibility_notes}")

        progress.step = "dataset_acquisition"
        progress.percentage = 18.0
        progress.message = f"Curating dataset via {spec.curation_mode.value.replace('_', ' ').title()}..."
        if on_progress:
            on_progress(progress)

        # 2. Acquire or Synthesize Dataset
        data_dir = os.path.join(job_dir, "data")
        if spec.curation_mode == DataCurationMode.SYNTHETIC:
            progress.logs.append(f"[Oumi/Distil Loop] Autonomous Synthesis: Generating domain instruction pairs & test cases...")
            dataset_spec = DatasetSynthesizer.synthesize_dataset(
                prompt=prompt,
                domain=spec.domain,
                task_type=spec.task_type,
                output_dir=data_dir,
                target_samples=35
            )
        elif spec.curation_mode == DataCurationMode.AUGMENTED:
            progress.logs.append(f"[Oumi Loop] Dataset Augmentation: Synthesizing baseline and expanding coverage...")
            base_syn = DatasetSynthesizer.synthesize_dataset(prompt, spec.domain, spec.task_type, data_dir, target_samples=20)
            train_items = []
            with open(os.path.join(data_dir, "train.jsonl"), "r", encoding="utf-8") as f:
                train_items = [json.loads(line) for line in f if line.strip()]
            dataset_spec = DatasetSynthesizer.augment_dataset(train_items, data_dir, factor=2)
        else:
            # Web Search Mode (Hugging Face)
            if chosen_dataset_id:
                progress.logs.append(f"User selected dataset: '{chosen_dataset_id}'. Fetching from web...")
                dataset_spec = self.searcher.acquire_chosen_dataset(chosen_dataset_id, spec.domain, data_dir)
            else:
                top_3 = self.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)
                best_choice = top_3[0].id if top_3 else spec.domain
                progress.logs.append(f"Auto-selected top ranked web dataset: '{best_choice}'")
                dataset_spec = self.searcher.acquire_chosen_dataset(best_choice, spec.domain, data_dir)

        progress.logs.append(f"Acquired dataset '{dataset_spec.name}': {dataset_spec.train_samples} train / {dataset_spec.val_samples} val samples.")

        # 3. Transparent Recipe Compilation (recipe.yaml)
        recipe_path = os.path.join(job_dir, "recipe.yaml")
        RecipeBuilder.generate_recipe(spec=spec, dataset_spec=dataset_spec, output_yaml_path=recipe_path)
        progress.logs.append(f"[Oumi Recipe] Compiled transparent, reproducible recipe: recipe.yaml")

        progress.step = "code_generation"
        progress.percentage = 32.0
        progress.message = "Writing custom PyTorch training code tailored for CPU..."
        if on_progress:
            on_progress(progress)

        # 4. Generate Training Script
        train_script = os.path.join(job_dir, "train_run.py")
        CodeGenerator.generate_training_script(
            spec=spec,
            dataset_spec=dataset_spec,
            output_script_path=train_script,
            job_dir=job_dir
        )
        progress.logs.append(f"Generated standalone training script: {os.path.basename(train_script)}")

        # 5. Pre-Training Baseline Evaluation (Oumi Loop Step #1)
        progress.logs.append(f"[Oumi Loop] Initializing Pre-Training Baseline Evaluation...")
        progress.step = "training_execution"
        progress.percentage = 40.0
        progress.message = "Running model training on CPU..."
        if on_progress:
            on_progress(progress)

        # 6. Execute Training Runner
        def handle_event(evt: Dict[str, Any]):
            evt_type = evt.get("type")
            if evt_type == "status":
                progress.logs.append(evt.get("message", ""))
                progress.message = evt.get("message", "")
            elif evt_type == "progress":
                progress.epoch = evt.get("epoch")
                progress.total_epochs = evt.get("total_epochs")
                progress.current_loss = evt.get("loss")
                pct_base = 42.0
                pct_span = 38.0
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

        # Update recipe.yaml with final metrics
        RecipeBuilder.generate_recipe(spec=spec, dataset_spec=dataset_spec, output_yaml_path=recipe_path, eval_metrics=metrics)

        # 7. Post-Training Evaluation & Failure Mode Analysis (Oumi Loop Step #4)
        progress.step = "evaluation_analysis"
        progress.percentage = 84.0
        progress.message = "Running automated post-training evaluation & failure mode analysis..."
        if on_progress:
            on_progress(progress)

        eval_report_dict = {}
        try:
            # Load test samples
            test_samples = []
            test_file = os.path.join(data_dir, "test.jsonl")
            if not os.path.exists(test_file):
                test_file = os.path.join(data_dir, "val.jsonl")
            if os.path.exists(test_file):
                with open(test_file, "r", encoding="utf-8") as f:
                    test_samples = [json.loads(line) for line in f if line.strip()][:5]

            # Build baseline engine and fine-tuned engine
            if spec.training_mode == "qwen_lora":
                baseline_engine = QwenInferenceEngine(adapter_dir="")
            else:
                baseline_engine = DomainInferenceEngine(domain=spec.domain, dataset_path=None)

            finetuned_engine = get_inference_engine(job_dir, domain=spec.domain)

            eval_report: EvaluationReport = ModelEvaluator.evaluate_pipeline(
                baseline_engine=baseline_engine,
                finetuned_engine=finetuned_engine,
                test_samples=test_samples,
                domain=spec.domain
            )
            eval_report_dict = eval_report.model_dump()

            eval_report_path = os.path.join(job_dir, "eval_report.json")
            with open(eval_report_path, "w", encoding="utf-8") as f:
                json.dump(eval_report_dict, f, indent=2)

            progress.logs.append(f"[Oumi Evaluation] {eval_report.summary}")
            progress.logs.append(f"[Oumi Evaluation] Baseline: {eval_report.baseline_accuracy_pct}% -> Fine-Tuned: {eval_report.finetuned_accuracy_pct}% (+{eval_report.improvement_delta_pct}% improvement)")
            progress.logs.append(f"[Oumi Evaluation] Failure patterns reduced by: {eval_report.failure_reduction_rate_pct}% | Latency: {eval_report.avg_latency_ms} ms/token")
        except Exception as e:
            progress.logs.append(f"Warning during evaluation loop: {e}")

        # 8. Build Deployment Package
        progress.step = "packaging_deployment"
        progress.percentage = 94.0
        progress.message = "Generating zero-setup FastAPI server & interactive playground..."
        if on_progress:
            on_progress(progress)

        pkg: DeploymentPackage = PackageBuilder.build_package(job_id=job_id, job_dir=job_dir)
        progress.logs.append(f"Model package built successfully! Size: {pkg.model_size_mb} MB.")
        progress.logs.append(f"Artifact location: {pkg.model_dir}")

        progress.step = "completed"
        progress.percentage = 100.0
        progress.status = StepStatus.COMPLETED
        progress.message = "Your domain model is fully trained, evaluated, and ready for immediate use!"
        progress.model_artifact_path = pkg.model_dir
        if on_progress:
            on_progress(progress)

        return {
            "job_id": job_id,
            "status": "success",
            "spec": spec.model_dump(),
            "dataset": dataset_spec.model_dump(),
            "recipe_path": recipe_path,
            "eval_report": eval_report_dict,
            "package": pkg.model_dump(),
            "metrics": metrics,
            "logs": progress.logs
        }
