import re
import os
import json
from typing import Optional
from autollm.intent.schemas import (
    PipelineSpec,
    RouteType,
    TaskType,
    DeviceType,
    DataCurationMode,
)
from autollm.estimator.hardware_calc import HardwareEstimator
from autollm.security import sanitize_user_prompt

class IntentParser:
    """
    Extracts structured ML specifications from free-text user prompts.
    Provides intelligent 4-way route classification:
      1. PEFT_FINETUNE: Domain adaptation of Qwen 2.5 foundation models via LoRA.
      2. FROM_SCRATCH: Training compact Causal Transformer baseline from random weights.
      3. RAG: Zero-training document vector indexing and grounded question answering.
      4. DESIGN_PREVIEW: Instant hardware/memory budgeting and architecture blueprint without compute.
    """

    def __init__(self, force_cpu: bool = True):
        self.estimator = HardwareEstimator(force_cpu=force_cpu)

    def parse(self, prompt: str) -> PipelineSpec:
        clean_raw = sanitize_user_prompt(prompt)
        text = clean_raw.lower().strip()

        # 1. Domain Detection (Maximum keyword frequency with word boundaries)
        domain = "general"
        domain_keywords = {
            "agriculture": ["agriculture", "agricultural", "plant", "plants", "tree", "trees", "lemon", "avocado", "crop", "crops", "farm", "farmer", "farming", "botany", "fruit", "gardening", "soil", "fertilizer", "pest"],
            "medical": ["medical", "medicine", "health", "doctor", "clinical", "hospital", "patient", "symptoms", "physician", "illness", "disease"],
            "legal": ["legal", "contract", "law", "compliance", "clause", "statute", "court", "attorney", "nda"],
            "code": ["code", "python", "programming", "javascript", "developer", "sql", "coding", "bug", "software"],
            "finance": ["finance", "banking", "investment", "stock", "crypto", "economic", "financial", "fraud", "transaction"],
            "creative": ["creative", "story", "shakespeare", "poetry", "novel", "fiction", "character", "poem", "verse"],
            "customer_service": ["faq", "customer", "support", "helpdesk", "ticket", "service"],
            "science": ["physics", "biology", "chemistry", "science", "math", "equation"]
        }
        max_hits = 0
        for dom, keywords in domain_keywords.items():
            hits = sum(1 for k in keywords if re.search(r'\b' + re.escape(k) + r'\b', text))
            if hits > max_hits:
                max_hits = hits
                domain = dom

        # 2. Task Type Detection
        task_type = TaskType.INSTRUCTION_QA
        if any(w in text for w in ["chat", "chatbot", "assistant", "conversational"]):
            task_type = TaskType.CHAT
        elif any(w in text for w in ["classify", "classification", "sentiment", "categorize", "triage", "detect"]):
            task_type = TaskType.CLASSIFICATION
        elif any(w in text for w in ["summarize", "summary", "summarization"]):
            task_type = TaskType.SUMMARIZATION
        elif any(w in text for w in ["poem", "story", "generate text", "creative", "shakespeare", "verse"]):
            task_type = TaskType.CREATIVE

        # 3. 4-Way Decision Router Classification
        route = RouteType.PEFT_FINETUNE
        route_explanation = ""

        # Branch 4: Design-Only Mode
        if any(w in text for w in ["design only", "preview", "plan only", "architecture spec", "estimate memory", "how many parameters", "hardware plan", "blueprint"]):
            route = RouteType.DESIGN_PREVIEW
            route_explanation = "User requested architectural planning and memory budgeting without running compute."
        
        # Branch 3: RAG Route
        elif any(w in text for w in ["rag", "retrieval", "vector", "search documents", "pdf", "policy", "handbook", "grounded on file", "from my document", "knowledge base"]):
            route = RouteType.RAG
            route_explanation = "User requested grounded factual Q&A over document collections; routed to zero-training RAG index."
        
        # Branch 2: From-Scratch MiniGPT
        elif any(w in text for w in ["scratch", "from scratch", "from zero", "toy", "tiny transformer", "educational", "shakespeare"]) and not any(w in text for w in ["fine-tune", "finetune", "lora", "peft"]):
            route = RouteType.FROM_SCRATCH
            route_explanation = "User requested a lightweight from-scratch causal transformer baseline for educational or narrow domain pretraining."
        
        # Branch 1: PEFT LoRA (Default production path)
        else:
            route = RouteType.PEFT_FINETUNE
            route_explanation = "User requested high-intelligence domain adaptation; routed to Qwen 2.5 (0.5B-Instruct) LoRA fine-tuning on CPU."

        # 4. Target Size Extraction (e.g. "under 2GB", "500MB", "1.5 GB")
        target_size_mb = 2048.0 # default 2GB
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

        # Hyperparameters based on prompt cues
        epochs = 2
        batch_size = 4
        lr = 2e-4 if route == RouteType.PEFT_FINETUNE else 5e-4
        if "fast" in text or "quick" in text:
            epochs = 1
        elif "deep" in text or "thorough" in text:
            epochs = 3

        training_mode = "qwen_lora" if route == RouteType.PEFT_FINETUNE else ("scratch" if route == RouteType.FROM_SCRATCH else route.value)
        base_model = "Qwen/Qwen2.5-0.5B-Instruct" if route == RouteType.PEFT_FINETUNE else ("autollm-mini-gpt" if route == RouteType.FROM_SCRATCH else None)

        spec = PipelineSpec(
            raw_prompt=clean_raw,
            domain=domain,
            task_type=task_type,
            route=route,
            route_explanation=route_explanation,
            target_device=DeviceType.CPU,
            target_size_mb=target_size_mb,
            language=language,
            base_model=base_model,
            training_mode=training_mode,
            dataset_query=dataset_query,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_len=256,
            hardware_estimate=hw_estimate
        )

        return spec
