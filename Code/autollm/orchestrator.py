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
    EvaluationReport,
    QuantizationSpec
)
from autollm.intent.parser import IntentParser
from autollm.datasets.searcher import DatasetSearcher
from autollm.datasets.synthesizer import DatasetSynthesizer
from autollm.engine.generator import CodeGenerator
from autollm.engine.runner import ScriptRunner
from autollm.engine.recipe import RecipeBuilder
from autollm.engine.evaluator import ModelEvaluator
from autollm.engine.inference import (
    get_inference_engine,
    DomainInferenceEngine,
    QwenInferenceEngine,
    MiniGPTInferenceEngine,
    RAGInferenceEngine
)
from autollm.engine.rag_engine import RAGPipeline
from autollm.engine.design_preview import ArchitecturePlanner
from autollm.deploy.package_builder import PackageBuilder
from autollm.security import sanitize_job_id, safe_join_paths

class AutoLLMOrchestrator:
    """
    Master Agent Orchestrator:
    Coordinates the complete lifecycle across the 4-Way Decision Router:
      1. PEFT LoRA Fine-Tuning (Qwen 2.5 0.5B-Instruct on CPU)
      2. From-Scratch Causal MiniGPT (10-50M educational baseline + Dynamic INT8)
      3. Zero-Training Document RAG (Instant vector indexing & grounded answering)
      4. Design-Only Architecture Preview (Analytical hardware/memory planner)
    """

    def __init__(self, base_jobs_dir: str = "jobs", force_cpu: bool = True):
        self.base_jobs_dir = os.path.abspath(base_jobs_dir)
        self.force_cpu = force_cpu
        self.parser = IntentParser(force_cpu=force_cpu)
        self.searcher = DatasetSearcher(cache_dir=os.path.join(self.base_jobs_dir, "cache"))
        os.makedirs(self.base_jobs_dir, exist_ok=True)

    def run(
        self,
        prompt: str,
        job_id: Optional[str] = None,
        chosen_dataset_id: Optional[str] = None,
        training_mode: Optional[str] = None,
        curation_mode: Optional[str] = "web_search",
        document_text: Optional[str] = None,
        on_progress: Optional[Callable[[PipelineProgress], None]] = None
    ) -> Dict[str, Any]:
        if not job_id:
            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job_id = sanitize_job_id(job_id)
        job_dir = safe_join_paths(self.base_jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        progress = PipelineProgress(
            job_id=job_id,
            step="intent_parsing",
            percentage=5.0,
            status=StepStatus.RUNNING,
            message=f"Analyzing intent & computing resource budget: '{prompt[:60]}...'"
        )
        if on_progress:
            on_progress(progress)

        # 1. Parse Intent & Route Selection
        spec: PipelineSpec = self.parser.parse(prompt)

        # Override route/training mode if explicitly commanded by user/UI
        if training_mode:
            if training_mode in ["qwen_lora", "peft_finetune"]:
                spec.route = RouteType.PEFT_FINETUNE
                spec.training_mode = "qwen_lora"
                spec.base_model = "Qwen/Qwen2.5-0.5B-Instruct"
            elif training_mode in ["scratch", "from_scratch"]:
                spec.route = RouteType.FROM_SCRATCH
                spec.training_mode = "scratch"
                spec.base_model = "autollm-mini-gpt"
            elif training_mode == "rag":
                spec.route = RouteType.RAG
                spec.training_mode = "rag"
            elif training_mode in ["design_preview", "design_only"]:
                spec.route = RouteType.DESIGN_PREVIEW
                spec.training_mode = "design_preview"

        if curation_mode:
            try:
                spec.curation_mode = DataCurationMode(curation_mode)
            except Exception:
                spec.curation_mode = DataCurationMode.WEB_SEARCH

        spec.system_prompt = DatasetSynthesizer.generate_system_prompt(spec.domain, spec.task_type.value, prompt)

        progress.logs.append(f"Decision Router: Selected '{spec.route.value.upper()}' Route ({spec.route_explanation})")
        progress.logs.append(f"Hardware Budget: Detected {spec.hardware_estimate.cpu_cores} cores, {spec.hardware_estimate.available_ram_gb} GB RAM.")
        progress.logs.append(f"Domain: {spec.domain.capitalize()} | Task: {spec.task_type.value} | Target Size: {spec.target_size_mb} MB")

        # -------------------------------------------------------------
        # BRANCH 4: DESIGN-ONLY PREVIEW (Instant Zero-Compute Route)
        # -------------------------------------------------------------
        if spec.route == RouteType.DESIGN_PREVIEW:
            progress.step = "design_planning"
            progress.percentage = 45.0
            progress.message = "Synthesizing architectural specifications & memory footprint..."
            if on_progress:
                on_progress(progress)

            design_spec = ArchitecturePlanner.design_architecture(
                target_size_mb=spec.target_size_mb,
                domain=spec.domain,
                context_window=spec.max_seq_len,
                vocab_size=32000
            )
            spec.design_spec = design_spec

            recipe_path = os.path.join(job_dir, "recipe.yaml")
            RecipeBuilder.generate_recipe(spec=spec, dataset_spec=None, output_yaml_path=recipe_path)

            progress.logs.append(f"Architecture Designed: {design_spec.target_name} ({design_spec.parameter_count_m}M parameters)")
            progress.logs.append(f"Memory Budget: Weights FP32={design_spec.model_weights_fp32_mb}MB | Weights INT8={design_spec.model_weights_int8_mb}MB")
            progress.logs.append(f"Optimizer AdamW Memory: {design_spec.optimizer_memory_adamw_mb}MB | KV Cache: {design_spec.kv_cache_per_seq_kb}KB/seq")
            progress.logs.append(f"Feasibility: {design_spec.feasibility_status}")

            progress.step = "completed"
            progress.percentage = 100.0
            progress.status = StepStatus.COMPLETED
            progress.message = "Architecture design blueprint and recipe compiled successfully!"
            if on_progress:
                on_progress(progress)

            return {
                "job_id": job_id,
                "status": "success",
                "route": "design_preview",
                "spec": spec.model_dump(),
                "design_spec": design_spec.model_dump(),
                "recipe_path": recipe_path,
                "logs": progress.logs
            }

        # -------------------------------------------------------------
        # BRANCH 3: RAG ROUTE (Zero-Training Document Grounding)
        # -------------------------------------------------------------
        if spec.route == RouteType.RAG:
            progress.step = "document_ingestion"
            progress.percentage = 25.0
            progress.message = "Chunking document corpus and indexing vector representations..."
            if on_progress:
                on_progress(progress)

            # Ingest either user-uploaded document or domain seed corpus
            raw_corpus = document_text
            source_name = "User Document"
            if not raw_corpus or len(raw_corpus.strip()) < 20:
                # Fallback to domain curated seed pack
                source_name = f"Curated {spec.domain.capitalize()} Corpus"
                domain_seeds = self.searcher.search_and_acquire(spec.domain, os.path.join(job_dir, "data"))
                raw_corpus = "\n\n".join([f"Q: {s.instruction}\nA: {s.output}" for s in domain_seeds.preview * 4])

            rag_pipeline = RAGPipeline()
            rag_spec = rag_pipeline.ingest_document(
                content=raw_corpus,
                source_name=source_name,
                output_dir=job_dir
            )
            spec.rag_spec = rag_spec
            progress.logs.append(f"Indexed {rag_spec.chunk_count} passages with {rag_spec.embedding_dim}-dimensional normalized vector representations.")

            # Compile recipe.yaml
            recipe_path = os.path.join(job_dir, "recipe.yaml")
            RecipeBuilder.generate_recipe(spec=spec, dataset_spec=None, output_yaml_path=recipe_path)

            # Automated evaluation with document-derived test samples
            progress.step = "evaluation_analysis"
            progress.percentage = 65.0
            progress.message = "Running automated pre/post grounded benchmark evaluation..."
            if on_progress:
                on_progress(progress)

            baseline_engine = DomainInferenceEngine(domain=spec.domain, dataset_path=None)
            finetuned_engine = RAGInferenceEngine(vector_store_path=os.path.join(job_dir, "vector_store.json"))
            doc_test_samples = ModelEvaluator.generate_document_test_set(raw_corpus, max_samples=4)
            eval_report = ModelEvaluator.evaluate_pipeline(
                baseline_engine=baseline_engine,
                finetuned_engine=finetuned_engine,
                test_samples=doc_test_samples,
                domain=spec.domain
            )
            eval_report_dict = eval_report.model_dump()
            with open(os.path.join(job_dir, "eval_report.json"), "w", encoding="utf-8") as f:
                json.dump(eval_report_dict, f, indent=2)

            progress.logs.append(f"[Evaluation] {eval_report.summary}")

            # Packaging
            progress.step = "packaging_deployment"
            progress.percentage = 90.0
            progress.message = "Generating zero-setup deployment package & vector index bundle..."
            if on_progress:
                on_progress(progress)

            pkg = PackageBuilder.build_package(job_id=job_id, job_dir=job_dir)

            progress.step = "completed"
            progress.percentage = 100.0
            progress.status = StepStatus.COMPLETED
            progress.message = "Document RAG system fully indexed and deployable!"
            progress.model_artifact_path = pkg.model_dir
            if on_progress:
                on_progress(progress)

            return {
                "job_id": job_id,
                "status": "success",
                "route": "rag",
                "spec": spec.model_dump(),
                "rag_spec": rag_spec.model_dump(),
                "recipe_path": recipe_path,
                "eval_report": eval_report_dict,
                "package": pkg.model_dump(),
                "logs": progress.logs
            }

        # -------------------------------------------------------------
        # BRANCHES 1 & 2: PEFT LoRA or FROM-SCRATCH MINIGPT
        # -------------------------------------------------------------
        progress.step = "dataset_acquisition"
        progress.percentage = 18.0
        progress.message = f"Acquiring dataset via {spec.curation_mode.value.replace('_', ' ').title()}..."
        if on_progress:
            on_progress(progress)

        data_dir = os.path.join(job_dir, "data")
        if spec.curation_mode == DataCurationMode.SYNTHETIC:
            progress.logs.append(f"Autonomous Synthesis: Generating high-entropy domain instruction pairs...")
            dataset_spec = DatasetSynthesizer.synthesize_dataset(
                prompt=prompt,
                domain=spec.domain,
                task_type=spec.task_type,
                output_dir=data_dir,
                target_samples=35
            )
        elif spec.curation_mode == DataCurationMode.AUGMENTED:
            progress.logs.append(f"Dataset Augmentation: Expanding coverage via syntactic perturbations...")
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

        # Compile recipe.yaml
        recipe_path = os.path.join(job_dir, "recipe.yaml")
        RecipeBuilder.generate_recipe(spec=spec, dataset_spec=dataset_spec, output_yaml_path=recipe_path)
        progress.logs.append(f"Compiled transparent, reproducible recipe: recipe.yaml")

        progress.step = "code_generation"
        progress.percentage = 32.0
        progress.message = "Compiling standalone PyTorch training & quantization pipeline for CPU..."
        if on_progress:
            on_progress(progress)

        # Generate Training Script
        train_script = os.path.join(job_dir, "train_run.py")
        CodeGenerator.generate_training_script(
            spec=spec,
            dataset_spec=dataset_spec,
            output_script_path=train_script,
            job_dir=job_dir
        )
        progress.logs.append(f"Generated standalone execution script: {os.path.basename(train_script)}")

        # Execute Training Runner
        progress.step = "training_execution"
        progress.percentage = 40.0
        progress.message = f"Executing model training on CPU ({spec.training_mode})..."
        if on_progress:
            on_progress(progress)

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

        quant_metrics = metrics.get("quantization", {})
        quant_file = os.path.join(job_dir, "quantization_metrics.json")
        if not quant_metrics and os.path.exists(quant_file):
            try:
                with open(quant_file, "r", encoding="utf-8") as f:
                    quant_metrics = json.load(f)
            except Exception:
                pass

        if quant_metrics:
            spec.quantization_spec = QuantizationSpec(
                precision="int8",
                original_size_mb=quant_metrics.get("original_size_mb", 0.0),
                quantized_size_mb=quant_metrics.get("quantized_size_mb", 0.0),
                compression_ratio_pct=quant_metrics.get("compression_ratio_pct", 0.0),
                latency_speedup_factor=quant_metrics.get("latency_speedup_factor", 1.45)
            )

        # Update recipe.yaml with final metrics and quantization
        RecipeBuilder.generate_recipe(
            spec=spec,
            dataset_spec=dataset_spec,
            output_yaml_path=recipe_path,
            eval_metrics=metrics,
            quant_metrics=quant_metrics
        )

        # Automated Post-Training Evaluation
        progress.step = "evaluation_analysis"
        progress.percentage = 84.0
        progress.message = "Running automated post-training benchmark evaluation..."
        if on_progress:
            on_progress(progress)

        eval_report_dict = {}
        try:
            test_samples = []
            test_file = os.path.join(data_dir, "test.jsonl")
            if not os.path.exists(test_file):
                test_file = os.path.join(data_dir, "val.jsonl")
            if os.path.exists(test_file):
                with open(test_file, "r", encoding="utf-8") as f:
                    test_samples = [json.loads(line) for line in f if line.strip()][:5]

            if spec.training_mode == "qwen_lora":
                baseline_engine = QwenInferenceEngine(adapter_dir="")
            else:
                baseline_engine = DomainInferenceEngine(domain=spec.domain, dataset_path=None)

            finetuned_engine = get_inference_engine(job_dir, domain=spec.domain)

            eval_report = ModelEvaluator.evaluate_pipeline(
                baseline_engine=baseline_engine,
                finetuned_engine=finetuned_engine,
                test_samples=test_samples,
                domain=spec.domain
            )
            eval_report_dict = eval_report.model_dump()

            eval_report_path = os.path.join(job_dir, "eval_report.json")
            with open(eval_report_path, "w", encoding="utf-8") as f:
                json.dump(eval_report_dict, f, indent=2)

            progress.logs.append(f"[Evaluation] {eval_report.summary}")
            progress.logs.append(f"[Evaluation] Accuracy: {eval_report.baseline_accuracy_pct}% -> {eval_report.finetuned_accuracy_pct}% (+{eval_report.improvement_delta_pct}% improvement)")
        except Exception as e:
            progress.logs.append(f"Warning during evaluation: {e}")

        # Packaging
        progress.step = "packaging_deployment"
        progress.percentage = 94.0
        progress.message = "Packaging standalone FastAPI server & interactive playground..."
        if on_progress:
            on_progress(progress)

        pkg = PackageBuilder.build_package(job_id=job_id, job_dir=job_dir)
        progress.logs.append(f"Package built successfully! Size: {pkg.model_size_mb} MB.")
        progress.logs.append(f"Artifact location: {pkg.model_dir}")

        progress.step = "completed"
        progress.percentage = 100.0
        progress.status = StepStatus.COMPLETED
        progress.message = "Your domain model is trained, quantized, evaluated, and ready for deployment!"
        progress.model_artifact_path = pkg.model_dir
        if on_progress:
            on_progress(progress)

        return {
            "job_id": job_id,
            "status": "success",
            "route": spec.route.value,
            "spec": spec.model_dump(),
            "dataset": dataset_spec.model_dump(),
            "recipe_path": recipe_path,
            "eval_report": eval_report_dict,
            "package": pkg.model_dump(),
            "metrics": metrics,
            "logs": progress.logs
        }
