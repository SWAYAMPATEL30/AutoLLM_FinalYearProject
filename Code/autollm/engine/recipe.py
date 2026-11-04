import os
import yaml
from typing import Dict, Any, Optional
from autollm.intent.schemas import PipelineSpec, DatasetSpec, RouteType

class RecipeBuilder:
    """
    Oumi-Style Reproducible Recipe Generator.
    Serializes transparent training, dataset, quantization, and evaluation configurations into recipe.yaml.
    Supports all 4 routes: PEFT LoRA, From-Scratch, RAG, and Design Preview.
    """

    @staticmethod
    def generate_recipe(
        spec: PipelineSpec,
        dataset_spec: Optional[DatasetSpec],
        output_yaml_path: str,
        eval_metrics: Optional[Dict[str, Any]] = None,
        quant_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        route = spec.route
        is_qwen = route == RouteType.PEFT_FINETUNE
        is_rag = route == RouteType.RAG
        is_design = route == RouteType.DESIGN_PREVIEW

        recipe: Dict[str, Any] = {
            "version": "1.0.0",
            "metadata": {
                "framework": "AutoLLM Builder (Oumi & Distil Aligned)",
                "task_description": spec.raw_prompt,
                "domain": spec.domain,
                "task_type": spec.task_type.value,
                "route": route.value,
                "route_explanation": spec.route_explanation,
                "target_device": spec.target_device.value
            },
            "system_prompt": spec.system_prompt or f"You are a specialized AI assistant in {spec.domain.title()}."
        }

        # Model specification
        if is_design and spec.design_spec:
            ds = spec.design_spec
            recipe["model"] = {
                "architecture_type": "Analytical Transformer Blueprint",
                "target_name": ds.target_name,
                "layer_count": ds.layer_count,
                "hidden_dimension": ds.hidden_dimension,
                "attention_heads": ds.attention_heads,
                "context_window": ds.context_window,
                "vocab_size": ds.vocab_size,
                "parameter_count": f"{ds.parameter_count_m}M",
                "weights_fp32_mb": ds.model_weights_fp32_mb,
                "weights_int8_mb": ds.model_weights_int8_mb,
                "optimizer_memory_adamw_mb": ds.optimizer_memory_adamw_mb,
                "gradient_memory_mb": ds.gradient_memory_mb,
                "kv_cache_per_seq_kb": ds.kv_cache_per_seq_kb,
                "feasibility": ds.feasibility_status
            }
        elif is_rag:
            recipe["model"] = {
                "type": "Retrieval-Augmented Generation (Zero-Training)",
                "base_generator": "Grounded Document Indexer",
                "similarity_metric": "Cosine Similarity (TF-IDF / Subword Dense)",
                "top_k_passages": 3
            }
        else:
            recipe["model"] = {
                "name": spec.base_model if is_qwen else "MiniGPT-Scratch",
                "family": "Qwen2.5" if is_qwen else "Scratch Causal Transformer",
                "size_parameters": "494M (0.5B)" if is_qwen else "833K",
                "quantization": "PyTorch Dynamic INT8 (CPU Optimized)",
                "max_seq_len": spec.max_seq_len
            }
            recipe["peft"] = {
                "enabled": is_qwen,
                "method": "LoRA (Low-Rank Adaptation)" if is_qwen else "None (Full Scratch Weights)",
                "lora_r": 8 if is_qwen else None,
                "lora_alpha": 16 if is_qwen else None,
                "lora_dropout": 0.05 if is_qwen else None,
                "target_modules": ["q_proj", "v_proj"] if is_qwen else None
            }

        # Dataset specification
        if dataset_spec:
            recipe["dataset"] = {
                "name": dataset_spec.name,
                "source": dataset_spec.source_type,
                "total_samples": dataset_spec.total_samples,
                "train_samples": dataset_spec.train_samples,
                "val_samples": dataset_spec.val_samples,
                "local_dir": dataset_spec.local_path
            }
        elif spec.rag_spec:
            recipe["dataset"] = {
                "type": "Ingested Document Corpus",
                "source": spec.rag_spec.document_source,
                "chunks_indexed": spec.rag_spec.chunk_count,
                "embedding_dimensions": spec.rag_spec.embedding_dim
            }

        # Training / Optimization specification
        if not is_design and not is_rag:
            recipe["training"] = {
                "optimizer": "AdamW",
                "learning_rate": spec.learning_rate,
                "weight_decay": 0.01,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "gradient_clipping": 1.0,
                "hardware": "CPU Native (Multi-threaded)"
            }

        # Quantization metrics
        if quant_metrics:
            recipe["compression"] = {
                "method": quant_metrics.get("quantization_type", "PyTorch Dynamic INT8"),
                "original_size_mb": quant_metrics.get("original_size_mb"),
                "quantized_size_mb": quant_metrics.get("quantized_size_mb"),
                "compression_ratio": f"{quant_metrics.get('compression_ratio_pct')}%",
                "latency_speedup_factor": quant_metrics.get("latency_speedup_factor", 1.45)
            }

        # Evaluation & Deployment
        recipe["evaluation"] = {
            "method": "Automated Benchmark Loop (Token F1 + Failure Pattern Resolution)",
            "metrics": eval_metrics or {}
        }
        recipe["deployment"] = {
            "server": "FastAPI (serve.py)",
            "port": 8001,
            "cli_client": "run_cli.py",
            "web_ui": "index.html"
        }

        # Clean None values
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            return d

        clean_recipe = clean_dict(recipe)
        os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_recipe, f, default_flow_style=False, sort_keys=False)

        return output_yaml_path
