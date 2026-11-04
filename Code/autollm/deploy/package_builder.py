import os
import shutil
import zipfile
import json
from autollm.intent.schemas import DeploymentPackage
from autollm.deploy.server_template import (
    get_fastapi_server_code,
    get_cli_chat_code,
    get_interactive_ui_html
)

class PackageBuilder:
    """
    Builds a standalone, zero-setup deployment bundle for the trained model or RAG index.
    """

    @staticmethod
    def build_package(job_id: str, job_dir: str) -> DeploymentPackage:
        export_dir = os.path.join(job_dir, "export")
        os.makedirs(export_dir, exist_ok=True)

        model_src = os.path.join(job_dir, "model.pt")
        quant_src = os.path.join(job_dir, "model_quantized.pt")
        tok_src = os.path.join(job_dir, "tokenizer.json")
        metrics_src = os.path.join(job_dir, "eval_metrics.json")
        quant_metrics_src = os.path.join(job_dir, "quantization_metrics.json")
        vec_src = os.path.join(job_dir, "vector_store.json")

        model_dest = os.path.join(export_dir, "model.pt")
        quant_dest = os.path.join(export_dir, "model_quantized.pt")
        tok_dest = os.path.join(export_dir, "tokenizer.json")
        metrics_dest = os.path.join(export_dir, "eval_metrics.json")
        quant_metrics_dest = os.path.join(export_dir, "quantization_metrics.json")
        vec_dest = os.path.join(export_dir, "vector_store.json")

        data_src = os.path.join(job_dir, "data", "train.jsonl")
        data_dest = os.path.join(export_dir, "knowledge.jsonl")

        adapter_src = os.path.join(job_dir, "adapter")
        adapter_dest = os.path.join(export_dir, "adapter")

        recipe_src = os.path.join(job_dir, "recipe.yaml")
        recipe_dest = os.path.join(export_dir, "recipe.yaml")
        eval_report_src = os.path.join(job_dir, "eval_report.json")
        eval_report_dest = os.path.join(export_dir, "eval_report.json")

        if os.path.exists(adapter_src):
            if os.path.exists(adapter_dest):
                shutil.rmtree(adapter_dest)
            shutil.copytree(adapter_src, adapter_dest)

        if os.path.exists(model_src):
            shutil.copy2(model_src, model_dest)
        if os.path.exists(quant_src):
            shutil.copy2(quant_src, quant_dest)
        if os.path.exists(tok_src):
            shutil.copy2(tok_src, tok_dest)
        if os.path.exists(metrics_src):
            shutil.copy2(metrics_src, metrics_dest)
        if os.path.exists(quant_metrics_src):
            shutil.copy2(quant_metrics_src, quant_metrics_dest)
        if os.path.exists(vec_src):
            shutil.copy2(vec_src, vec_dest)
        if os.path.exists(data_src):
            shutil.copy2(data_src, data_dest)
        if os.path.exists(recipe_src):
            shutil.copy2(recipe_src, recipe_dest)
        if os.path.exists(eval_report_src):
            shutil.copy2(eval_report_src, eval_report_dest)

        # 1. Write serve.py
        serve_py = os.path.join(export_dir, "serve.py")
        with open(serve_py, "w", encoding="utf-8") as f:
            f.write(get_fastapi_server_code())

        # 2. Write run_cli.py
        cli_py = os.path.join(export_dir, "run_cli.py")
        with open(cli_py, "w", encoding="utf-8") as f:
            f.write(get_cli_chat_code())

        # 3. Write index.html
        ui_html = os.path.join(export_dir, "index.html")
        with open(ui_html, "w", encoding="utf-8") as f:
            f.write(get_interactive_ui_html())

        # 4. Write start_server.bat for Windows one-click execution
        bat_file = os.path.join(export_dir, "start_server.bat")
        with open(bat_file, "w", encoding="utf-8") as f:
            f.write("@echo off\necho Starting AutoLLM Model Inference Server on CPU...\npython serve.py\npause\n")

        # 5. Read metrics & quantization
        metrics = {}
        if os.path.exists(metrics_dest):
            with open(metrics_dest, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        quant_metrics = {}
        if os.path.exists(quant_metrics_dest):
            with open(quant_metrics_dest, "r", encoding="utf-8") as f:
                quant_metrics = json.load(f)

        # Determine effective model size
        model_size_mb = 0.0
        if os.path.exists(quant_dest):
            model_size_mb = round(os.path.getsize(quant_dest) / (1024 * 1024), 2)
        elif os.path.exists(model_dest):
            model_size_mb = round(os.path.getsize(model_dest) / (1024 * 1024), 2)
        elif os.path.exists(adapter_dest):
            for root, _, files in os.walk(adapter_dest):
                for f in files:
                    model_size_mb += os.path.getsize(os.path.join(root, f))
            model_size_mb = round(model_size_mb / (1024 * 1024), 2)
        elif os.path.exists(vec_dest):
            model_size_mb = round(os.path.getsize(vec_dest) / (1024 * 1024), 2)

        # 6. Create ZIP archive
        zip_path = os.path.join(job_dir, f"autollm_model_{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)

        eval_report_data = None
        if os.path.exists(eval_report_dest):
            try:
                with open(eval_report_dest, "r", encoding="utf-8") as f:
                    eval_report_data = json.load(f)
            except Exception:
                pass

        return DeploymentPackage(
            job_id=job_id,
            model_dir=export_dir,
            model_weights_path=quant_dest if os.path.exists(quant_dest) else model_dest,
            quantized_weights_path=quant_dest if os.path.exists(quant_dest) else None,
            tokenizer_path=tok_dest,
            serve_script_path=serve_py,
            interactive_ui_path=ui_html,
            recipe_yaml_path=recipe_dest if os.path.exists(recipe_dest) else None,
            vector_store_path=vec_dest if os.path.exists(vec_dest) else None,
            model_size_mb=model_size_mb,
            eval_metrics=metrics,
            quantization_metrics=quant_metrics,
            eval_report=eval_report_data
        )
