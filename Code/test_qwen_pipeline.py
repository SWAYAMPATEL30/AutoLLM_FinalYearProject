import os
import sys
import shutil

# Guarantee Code root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autollm.orchestrator import AutoLLMOrchestrator
from autollm.engine.inference import get_inference_engine

def main():
    test_jobs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen_jobs")
    os.makedirs(test_jobs_dir, exist_ok=True)

    orchestrator = AutoLLMOrchestrator(base_jobs_dir=test_jobs_dir, force_cpu=True)

    prompt = "Create an agriculture farming expert model that diagnoses crop diseases and gives farming advice"

    print("--- 1. Testing Dataset Discovery ---")
    spec = orchestrator.parser.parse(prompt)
    print(f"Parsed domain: {spec.domain}")

    top_3 = orchestrator.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)
    print("Top 3 datasets found:")
    for i, d in enumerate(top_3):
        print(f" [{i+1}] {d.name} ({d.id}) - {d.downloads} downloads")

    chosen = top_3[0].id
    print(f"\n--- 2. Running Orchestrator with Qwen LoRA on dataset: {chosen} ---")

    def on_progress(p):
        print(f"[{p.percentage}%] {p.step}: {p.message}")

    result = orchestrator.run(
        prompt=prompt,
        chosen_dataset_id=chosen,
        training_mode="qwen_lora",
        on_progress=on_progress
    )

    job_id = result["job_id"]
    job_dir = os.path.join(test_jobs_dir, job_id)
    print(f"\nTraining complete! Job ID: {job_id}")

    print("\n--- 3. Testing Interactive Qwen LoRA Inference ---")
    engine = get_inference_engine(job_dir, domain=spec.domain)
    print(f"Engine type: {engine.__class__.__name__}")

    test_queries = [
        "What is the average lifespan of a lemon tree?",
        "Why is my avocado tree not producing fruit?"
    ]

    for q in test_queries:
        print(f"\nUser > {q}")
        reply = engine.generate_response(q, max_tokens=100)
        print(f"Qwen Model > {reply}")

if __name__ == "__main__":
    main()
