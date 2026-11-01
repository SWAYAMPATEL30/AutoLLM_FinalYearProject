import re
import os
import json
from typing import Optional
from autollm.intent.schemas import (
    PipelineSpec,
    RouteType,
    TaskType,
    DeviceType,
)
from autollm.estimator.hardware_calc import HardwareEstimator

class IntentParser:
    """
    Extracts structured ML specifications from free-text user prompts.
    Includes zero-dependency heuristic parser and optional LLM-assisted refinement.
    """

    def __init__(self, force_cpu: bool = True):
        self.estimator = HardwareEstimator(force_cpu=force_cpu)

    def parse(self, prompt: str) -> PipelineSpec:
        text = prompt.lower().strip()

        # 1. Domain Detection
        domain = "general"
        domain_keywords = {
            "agriculture": ["agriculture", "farming", "farm", "farmer", "crop", "crops", "plant", "plants", "tree", "trees", "lemon", "avocado", "botany", "fruit", "gardening", "soil", "fertilizer", "pest"],
            "medical": ["medical", "medicine", "health", "doctor", "clinical", "hospital", "patient", "symptoms", "physician", "illness"],
            "legal": ["legal", "contract", "law", "compliance", "clause", "statute", "court", "attorney"],
            "code": ["code", "python", "programming", "javascript", "developer", "sql", "coding", "bug"],
            "finance": ["finance", "banking", "investment", "stock", "crypto", "economic", "financial"],
            "creative": ["creative", "story", "shakespeare", "poetry", "novel", "fiction", "character"],
            "customer_service": ["faq", "customer", "support", "helpdesk", "ticket", "service"],
            "agriculture": ["agriculture", "plant", "tree", "lemon", "avocado", "crop", "farm", "botany", "fruit", "gardening"],
            "science": ["physics", "biology", "chemistry", "science", "math", "equation"]
        }
        for dom, keywords in domain_keywords.items():
            if any(k in text for k in keywords):
                domain = dom
                break

        # 2. Task Type Detection
        task_type = TaskType.INSTRUCTION_QA
        if any(w in text for w in ["chat", "chatbot", "assistant", "conversational"]):
            task_type = TaskType.CHAT
        elif any(w in text for w in ["classify", "classification", "sentiment", "categorize"]):
            task_type = TaskType.CLASSIFICATION
        elif any(w in text for w in ["summarize", "summary", "summarization"]):
            task_type = TaskType.SUMMARIZATION
        elif any(w in text for w in ["poem", "story", "generate text", "creative", "shakespeare"]):
            task_type = TaskType.CREATIVE

        # 3. Route Selection
        # On CPU-only environments, FROM_SCRATCH mini-transformer provides immediate,
        # smooth end-to-end training without CUDA driver dependencies.
        route = RouteType.FROM_SCRATCH
        if any(w in text for w in ["rag", "retrieval", "vector", "search documents", "pdf"]):
            route = RouteType.RAG
        elif any(w in text for w in ["design only", "preview", "plan only", "architecture spec"]):
            route = RouteType.DESIGN_PREVIEW
        elif any(w in text for w in ["fine-tune", "finetune", "lora", "qlora", "peft"]) and not any(w in text for w in ["scratch", "from scratch"]):
            route = RouteType.PEFT_FINETUNE

        # 4. Target Size Extraction (e.g. "under 2GB", "500MB", "1.5 GB")
        target_size_mb = 1024.0 # default 1GB
        size_gb_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:gb|gigabytes?)', text)
        size_mb_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mb|megabytes?)', text)

        if size_gb_match:
            target_size_mb = float(size_gb_match.group(1)) * 1024.0
        elif size_mb_match:
            target_size_mb = float(size_mb_match.group(1))

        # 5. Language Detection
        language = "English"
        if "hindi" in text:
            language = "Hindi"
        elif "spanish" in text:
            language = "Spanish"
        elif "multilingual" in text:
            language = "Multilingual"

        # 6. Dataset Query
        dataset_query = f"{domain} {task_type.value}".strip()

        # 7. Hardware Feasibility
        hw_estimate = self.estimator.inspect_system()

        # Hyperparameter tuning for fast CPU convergence
        epochs = 3
        batch_size = 4
        lr = 5e-4
        if "fast" in text or "quick" in text:
            epochs = 2
        elif "deep" in text or "thorough" in text:
            epochs = 5

        base_model = "autollm-mini-gpt" if route == RouteType.FROM_SCRATCH else "HuggingFaceTB/SmolLM-135M"

        spec = PipelineSpec(
            raw_prompt=prompt,
            domain=domain,
            task_type=task_type,
            route=route,
            target_device=DeviceType.CPU,
            target_size_mb=target_size_mb,
            language=language,
            base_model=base_model,
            dataset_query=dataset_query,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_len=256,
            hardware_estimate=hw_estimate
        )

        return spec
