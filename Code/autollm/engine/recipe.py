import os
import yaml
from typing import Dict, Any, Optional
from autollm.intent.schemas import PipelineSpec, DatasetSpec

class RecipeBuilder:
    """
    Oumi-Style Reproducible Recipe Generator.
    Serializes transparent training, dataset, and evaluation configurations into recipe.yaml.
    """

    @staticmethod
    def generate_recipe(
        spec: PipelineSpec,
        dataset_spec: DatasetSpec,
        output_yaml_path: str,
        eval_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        is_qwen = spec.training_mode == "qwen_lora"

        recipe: Dict[str, Any] = {
            "version": "1.0.0",
            "metadata": {
                "framework": "AutoLLM Builder (Oumi & Distil Aligned)",
                "task_description": spec.raw_prompt,
                "domain": spec.domain,
                "task_type": spec.task_type.value,
                "target_device": spec.target_device.value
            },
            "system_prompt": spec.system_prompt or f"You are a specialized AI assistant in {spec.domain.title()}.",
            "model": {
                "name": spec.base_model if is_qwen else "MiniGPT-Scratch",
                "family": "Qwen2.5" if is_qwen else "Scratch Causal Transformer",
                "size_parameters": "494M (0.5B)" if is_qwen else "833K",
                "quantization": "FP32 / CPU-Optimized",
                "max_seq_len": spec.max_seq_len
            },
            "peft": {
                "enabled": is_qwen,
                "method": "LoRA (Low-Rank Adaptation)" if is_qwen else "None (Full Scratch Weights)",
                "lora_r": 8 if is_qwen else None,
                "lora_alpha": 16 if is_qwen else None,
                "lora_dropout": 0.05 if is_qwen else None,
                "target_modules": ["q_proj", "v_proj"] if is_qwen else None
            },
            "dataset": {
                "name": dataset_spec.name,
                "source": dataset_spec.source_type,
                "total_samples": dataset_spec.total_samples,
                "train_samples": dataset_spec.train_samples,
                "val_samples": dataset_spec.val_samples,
                "local_dir": dataset_spec.local_path
            },
            "training": {
                "optimizer": "AdamW",
                "learning_rate": spec.learning_rate,
                "weight_decay": 0.01,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "gradient_clipping": 1.0,
                "hardware": "CPU (Multi-threaded)"
            },
            "evaluation": {
                "method": "Pre/Post Automated Benchmark Loop",
                "metrics": eval_metrics or {}
            },
            "deployment": {
                "server": "FastAPI (serve.py)",
                "port": 8001,
                "cli_client": "run_cli.py",
                "web_ui": "index.html"
            }
        }

        # Clean None values
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v is not None}
            return d

        clean_recipe = clean_dict(recipe)

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_recipe, f, default_flow_style=False, sort_keys=False)

        return output_yaml_path
