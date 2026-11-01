import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.orchestrator import AutoLLMOrchestrator
from autollm.intent.schemas import PipelineProgress

last_msg = ""
def cli_progress(p: PipelineProgress):
    global last_msg
    bar_len = 20
    filled = int(bar_len * (p.percentage / 100.0))
    bar = "=" * filled + "-" * (bar_len - filled)
    msg = p.message[:50]
    line = f"\r[{bar}] {p.percentage:>5.1f}% | {p.step:<22} | {msg:<50}"
    print(line, end="", flush=True)
    if p.status.value in ["completed", "failed"]:
        print()

def main():
    parser = argparse.ArgumentParser(description="AutoLLM Builder: Prompt-Driven SLM Compiler (CPU Optimized)")
    parser.add_argument("prompt", type=str, help="Prompt describing your model requirement")
    parser.add_argument("--jobs-dir", type=str, default="jobs", help="Directory where models are saved")

    args = parser.parse_args()

    print("=" * 75)
    print("AutoLLM Builder — Prompt-Driven Small Language Model Generation Framework")
    print("Execution Engine: CPU Optimized (Standard Laptop / Desktop)")
    print("=" * 75)
    print(f"User Prompt: '{args.prompt}'\n")

    orchestrator = AutoLLMOrchestrator(base_jobs_dir=args.jobs_dir, force_cpu=True)
    spec = orchestrator.parser.parse(args.prompt)

    print(f"[*] Detected Domain: {spec.domain.upper()} | Task: {spec.task_type.value}")
    print("[*] Searching web & Hugging Face Hub for top 3 domain datasets...\n")

    top_datasets = orchestrator.searcher.find_top_3_datasets(query=spec.dataset_query or "", domain=spec.domain)

    print("-" * 75)
    print("TOP 3 RECOMMENDED DATASETS FOUND:")
    print("-" * 75)
    for idx, ds in enumerate(top_datasets, 1):
        rec_tag = " [RECOMMENDED]" if ds.is_recommended else ""
        print(f"[{idx}] {ds.name} ({ds.id}){rec_tag}")
        print(f"    Source: {ds.source} | Downloads: {ds.downloads:,} | Likes: {ds.likes}")
        print(f"    Description: {ds.description}")
        if ds.sample_preview:
            print(f"    Preview: {ds.sample_preview[:90]}...")
        print()

    # User Selection
    chosen_id = top_datasets[0].id
    try:
        choice_input = input("Select dataset [1-3] (Press Enter for #1 Recommended): ").strip()
        if choice_input in ["1", "2", "3"]:
            chosen_id = top_datasets[int(choice_input) - 1].id
    except (EOFError, KeyboardInterrupt):
        pass

    print(f"\n[*] Using dataset: {chosen_id}")
    print("[*] Commencing pipeline execution on CPU...\n")

    result = orchestrator.run(args.prompt, chosen_dataset_id=chosen_id, on_progress=cli_progress)

    print("\n" + "=" * 75)
    print("[SUCCESS] MODEL GENERATION COMPLETE!")
    print(f"Job ID: {result['job_id']}")
    print(f"Trained Parameters: {result['metrics'].get('total_params', 'N/A'):,}")
    print(f"Validation Loss: {result['metrics'].get('val_loss', 'N/A')}")
    print(f"Model Artifacts: {result['package']['model_dir']}")
    print(f"Interactive UI: {result['package']['interactive_ui_path']}")
    print(f"To launch the inference server:")
    print(f"   cd {result['package']['model_dir']} && python serve.py")
    print(f"To launch the interactive terminal chat:")
    print(f"   cd {result['package']['model_dir']} && python run_cli.py")
    print("=" * 75)

if __name__ == "__main__":
    main()
