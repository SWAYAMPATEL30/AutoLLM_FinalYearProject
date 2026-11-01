import json
import os
from typing import List, Dict, Any, Tuple
from autollm.intent.schemas import InstructionSample

class DatasetFormatter:
    """
    Normalizes diverse dataset schemas into uniform instruction-tuning JSONL format.
    Standard format: {"instruction": str, "input": str, "output": str}
    """

    @staticmethod
    def normalize_sample(raw: Dict[str, Any]) -> InstructionSample:
        # Check standard fields
        instruction = raw.get("instruction") or raw.get("prompt") or raw.get("question") or raw.get("input_text") or ""
        input_data = raw.get("input") or raw.get("context") or ""
        output = raw.get("output") or raw.get("response") or raw.get("answer") or raw.get("completion") or raw.get("target_text") or ""

        # If it's a raw text corpus (like Shakespeare/poetry)
        if not instruction and not output and "text" in raw:
            full_text = raw["text"].strip()
            # Split into prompt & continuation
            parts = full_text.split("\n", 1)
            instruction = f"Continue the following passage: {parts[0][:100]}"
            output = parts[1][:500] if len(parts) > 1 else parts[0]

        return InstructionSample(
            instruction=str(instruction).strip(),
            input=str(input_data).strip(),
            output=str(output).strip()
        )

    @classmethod
    def save_splits(
        cls,
        samples: List[InstructionSample],
        output_dir: str,
        val_ratio: float = 0.1
    ) -> Tuple[str, str, int, int]:
        os.makedirs(output_dir, exist_ok=True)
        split_idx = int(len(samples) * (1.0 - val_ratio))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:] if split_idx < len(samples) else samples[:max(1, len(samples)//5)]

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

        return train_path, val_path, len(train_samples), len(val_samples)
