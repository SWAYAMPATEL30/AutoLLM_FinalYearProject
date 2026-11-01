import os
import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from autollm.intent.schemas import InstructionSample, DatasetSpec, TaskType

class DatasetSynthesizer:
    """
    Autonomous Synthetic Data Generation and Augmentation Engine.
    Inspired by Oumi's data synthesis loop and gpt-llm-trainer.
    Generates tailored, high-entropy instruction pairs + system prompts from user intent.
    """

    @staticmethod
    def generate_system_prompt(domain: str, task_type: str, prompt: str) -> str:
        domain_clean = domain.replace("_", " ").title()
        if "classif" in task_type.lower():
            return (
                f"You are an expert AI classifier specialized in {domain_clean}. "
                f"Your task is to accurately analyze input queries, identify intents or categories, "
                f"and provide clear, structured classifications with supporting rationale."
            )
        elif "summar" in task_type.lower():
            return (
                f"You are a professional summarization assistant for {domain_clean}. "
                f"Distill complex inputs into concise, accurate, and actionable key points."
            )
        elif "code" in domain.lower() or "program" in prompt.lower():
            return (
                "You are an expert software engineer and code assistant. "
                "Provide clean, idiomatically correct, secure code with step-by-step explanations."
            )
        else:
            return (
                f"You are a helpful, authoritative domain expert in {domain_clean}. "
                f"Answer user inquiries with high precision, verified domain facts, "
                f"and direct, professional assistance aligned with: '{prompt}'."
            )

    @staticmethod
    def synthesize_dataset(
        prompt: str,
        domain: str,
        task_type: TaskType,
        output_dir: str,
        target_samples: int = 40
    ) -> DatasetSpec:
        os.makedirs(output_dir, exist_ok=True)
        system_prompt = DatasetSynthesizer.generate_system_prompt(domain, task_type.value, prompt)

        domain_seeds = DatasetSynthesizer._get_domain_seed_generators(domain, prompt, task_type)

        generated_samples: List[InstructionSample] = []
        for i in range(target_samples):
            generator = random.choice(domain_seeds)
            sample = generator(i)
            generated_samples.append(sample)

        # Shuffle
        random.seed(42)
        random.shuffle(generated_samples)

        # Split into Train (70%), Val (15%), Test (15%) for Oumi evaluation loop
        split_train = int(len(generated_samples) * 0.70)
        split_val = int(len(generated_samples) * 0.85)

        train_set = generated_samples[:split_train]
        val_set = generated_samples[split_train:split_val]
        test_set = generated_samples[split_val:]

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")
        test_path = os.path.join(output_dir, "test.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_set:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_set:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        with open(test_path, "w", encoding="utf-8") as f:
            for s in test_set:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        return DatasetSpec(
            name=f"Synthetic {domain.title()} Instruction Set ({len(generated_samples)} samples)",
            domain=domain,
            source_type="autonomous_synthetic_engine",
            total_samples=len(generated_samples),
            train_samples=len(train_set),
            val_samples=len(val_set),
            local_path=output_dir,
            preview=train_set[:5]
        )

    @staticmethod
    def augment_dataset(existing_samples: List[Dict[str, str]], output_dir: str, factor: int = 2) -> DatasetSpec:
        """
        Expands dataset coverage with syntactic variations and perturbations (Oumi augmentation).
        """
        os.makedirs(output_dir, exist_ok=True)
        augmented: List[InstructionSample] = []

        variation_prefixes = [
            "Could you explain: ",
            "Please provide details on: ",
            "What should I know regarding: ",
            "Guide me through: ",
            "I need an explanation for: "
        ]

        for s in existing_samples:
            inst = s.get("instruction", "")
            out = s.get("output", "")
            inp = s.get("input", "")
            augmented.append(InstructionSample(instruction=inst, input=inp, output=out))

            # Generate variations
            if factor > 1:
                clean_inst = re.sub(r'^(what is|how do i|please explain|tell me about)\s*', '', inst, flags=re.IGNORECASE)
                var_inst = random.choice(variation_prefixes) + clean_inst.strip()
                augmented.append(InstructionSample(instruction=var_inst, input=inp, output=out))

        split_train = int(len(augmented) * 0.8)
        train_set = augmented[:split_train]
        val_set = augmented[split_train:]

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_set:
                f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for item in val_set:
                f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

        return DatasetSpec(
            name=f"Augmented Dataset ({len(augmented)} samples)",
            domain="custom",
            source_type="augmented_curation",
            total_samples=len(augmented),
            train_samples=len(train_set),
            val_samples=len(val_set),
            local_path=output_dir,
            preview=train_set[:5]
        )

    @staticmethod
    def _get_domain_seed_generators(domain: str, prompt: str, task_type: TaskType):
        p = prompt.lower()
        d = domain.lower()

        if "fraud" in p or "finance" in d or "banking" in p:
            return [
                lambda i: InstructionSample(
                    instruction=f"Evaluate transaction ID #{1000 + i}: Credit card charge of ${500 + i * 85} made from IP located in Nigeria while user card was issued in Chicago 2 hours earlier.",
                    output="Classification: HIGH RISK (Likely Fraud). Rationale: Velocity discrepancy (impossible travel time between card issue country and transaction IP location) combined with high-value round denomination."
                ),
                lambda i: InstructionSample(
                    instruction=f"Analyze transaction: Regular grocery purchase of ${35 + i * 3} at local supermarket within cardholder's home postal code.",
                    output="Classification: LOW RISK (Legitimate). Rationale: Consistent with cardholder's baseline transaction history, familiar merchant category code, and matching geographic location."
                ),
                lambda i: InstructionSample(
                    instruction=f"What are key indicators of account takeover (ATO) fraud in online banking?",
                    output="Key ATO indicators include: 1) Password or MFA resets immediately followed by large balance transfers, 2) Sudden addition of unfamiliar payees or beneficiaries, 3) Login sessions originating from known proxy/VPN IP ranges, and 4) Changes to contact email or phone numbers prior to withdrawal requests."
                )
            ]

        elif "support" in p or "ticket" in p or "triage" in p:
            return [
                lambda i: InstructionSample(
                    instruction=f"Triage and categorize this customer ticket #{200 + i}: 'My checkout page keeps giving error 500 when processing Visa cards, customers cannot complete purchases!'",
                    output="Category: Payment Gateway Failure | Priority: P0 (CRITICAL) | Routing: SRE & Payments Team | Action: Escalate immediately as production checkout is blocked."
                ),
                lambda i: InstructionSample(
                    instruction=f"Triage customer inquiry: 'Where can I download my invoice for the previous month's subscription?'",
                    output="Category: Billing / Account Management | Priority: P3 (LOW) | Routing: Automated Support / Tier-1 | Action: Direct customer to Settings -> Billing -> Invoices tab with self-service download guide."
                ),
                lambda i: InstructionSample(
                    instruction=f"Classify support ticket: 'The app crashes when I attempt to upload a profile picture exceeding 10MB on Android 14.'",
                    output="Category: Mobile App Bug | Priority: P2 (MEDIUM) | Routing: Mobile Android Dev Team | Action: Replicate memory allocation crash on large image payloads; confirm file size validation modal."
                )
            ]

        elif "agri" in d or "crop" in p or "farm" in p:
            return [
                lambda i: InstructionSample(
                    instruction=f"What is the average lifespan and productive harvesting window of a lemon tree?",
                    output="Lemon trees have an average lifespan of 50 to 100 years under proper soil drainage and care. Commercial fruit production typically begins in year 3 to 5 and peaks between years 10 and 30."
                ),
                lambda i: InstructionSample(
                    instruction=f"Why does a 10-foot tall avocado tree fail to bear fruit?",
                    output="Common causes include: 1) Grown from seed (avocados take 8-15 years to mature from seed, vs 3-4 years for grafted varieties), 2) Inadequate pollination between Type A and Type B flowering trees, 3) Excessive nitrogen fertilizer causing vegetative growth over fruiting, or 4) Frost damage to early flower buds."
                ),
                lambda i: InstructionSample(
                    instruction=f"How can a farmer identify and treat early blight on tomato foliage?",
                    output="Early blight (Alternaria solani) presents as concentric brown rings or 'target-board' lesions on lower, mature leaves, accompanied by yellow halos. Treatment: Prune infected bottom leaves, avoid overhead sprinkler watering, apply copper-based organic fungicides, and implement crop rotation."
                )
            ]

        elif "legal" in d or "contract" in p:
            return [
                lambda i: InstructionSample(
                    instruction="Explain the purpose and essential components of a Non-Disclosure Agreement (NDA).",
                    output="An NDA protects proprietary and confidential information. Essential components include: 1) Definition of confidential material, 2) Exclusions from confidentiality (e.g., publicly known data), 3) Obligations of the receiving party, 4) Term duration (typically 2-5 years), and 5) Remedies for breach."
                ),
                lambda i: InstructionSample(
                    instruction="What does an Indemnification Clause stipulate in commercial software licensing?",
                    output="An indemnification clause specifies that one party (the indemnitor) will defend, hold harmless, and compensate the other party (the indemnitee) for losses, damages, or legal expenses arising from third-party intellectual property infringement claims or contractual breaches."
                )
            ]

        elif "code" in d or "python" in p or "program" in p:
            return [
                lambda i: InstructionSample(
                    instruction="Write a Python function to safely debounce rapid events with a configurable time window.",
                    output="```python\nimport time\nfrom functools import wraps\n\ndef debounce(wait_seconds: float):\n    def decorator(fn):\n        last_called = 0.0\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n            nonlocal last_called\n            now = time.time()\n            if now - last_called >= wait_seconds:\n                last_called = now\n                return fn(*args, **kwargs)\n        return wrapper\n    return decorator\n```"
                ),
                lambda i: InstructionSample(
                    instruction="Explain the difference between threading and multiprocessing in Python.",
                    output="Threading uses light-weight threads within a single process sharing memory, but is constrained by Python's Global Interpreter Lock (GIL) for CPU-bound tasks (ideal for I/O operations). Multiprocessing spawns separate OS processes with dedicated Python interpreters and memory spaces, achieving true parallelism across CPU cores for compute-heavy workloads."
                )
            ]

        else: # General fallback
            return [
                lambda i: InstructionSample(
                    instruction=f"Provide a clear, domain-specific explanation for: '{prompt}' (Concept #{i+1})",
                    output=f"In the context of {domain.title()}, this entails applying verified domain principles: establishing baseline facts, verifying requirements, eliminating edge-case ambiguities, and providing direct, professional guidance."
                ),
                lambda i: InstructionSample(
                    instruction=f"What are best practices when addressing challenges in {domain.title()}?",
                    output=f"Best practices in {domain.title()} include: 1) Rigorous systematic analysis, 2) Adherence to verified standards and procedures, 3) Proactive risk mitigation, and 4) Continuous iterative evaluation and validation against measurable outcomes."
                )
            ]
