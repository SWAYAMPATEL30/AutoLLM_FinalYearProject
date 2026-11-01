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
    Builds a standalone, zero-setup deployment bundle for the trained model.
    """

    @staticmethod
    def build_package(job_id: str, job_dir: str) -> DeploymentPackage:
        export_dir = os.path.join(job_dir, "export")
        os.makedirs(export_dir, exist_ok=True)

        model_src = os.path.join(job_dir, "model.pt")
        tok_src = os.path.join(job_dir, "tokenizer.json")
        metrics_src = os.path.join(job_dir, "eval_metrics.json")

        model_dest = os.path.join(export_dir, "model.pt")
        tok_dest = os.path.join(export_dir, "tokenizer.json")
        metrics_dest = os.path.join(export_dir, "eval_metrics.json")

        data_src = os.path.join(job_dir, "data", "train.jsonl")
        data_dest = os.path.join(export_dir, "knowledge.jsonl")

        adapter_src = os.path.join(job_dir, "adapter")
        adapter_dest = os.path.join(export_dir, "adapter")

        if os.path.exists(adapter_src):
            if os.path.exists(adapter_dest):
                shutil.rmtree(adapter_dest)
            shutil.copytree(adapter_src, adapter_dest)

        if os.path.exists(model_src):
            shutil.copy2(model_src, model_dest)
        if os.path.exists(tok_src):
            shutil.copy2(tok_src, tok_dest)
        if os.path.exists(metrics_src):
            shutil.copy2(metrics_src, metrics_dest)
        if os.path.exists(data_src):
            shutil.copy2(data_src, data_dest)

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

        # 5. Read metrics
        metrics = {}
        if os.path.exists(metrics_dest):
            with open(metrics_dest, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        model_size_mb = round(os.path.getsize(model_dest) / (1024 * 1024), 2) if os.path.exists(model_dest) else 0.0

        # 6. Create ZIP archive
        zip_path = os.path.join(job_dir, f"autollm_model_{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)

        return DeploymentPackage(
            job_id=job_id,
            model_dir=export_dir,
            model_weights_path=model_dest,
            tokenizer_path=tok_dest,
            serve_script_path=serve_py,
            interactive_ui_path=ui_html,
            model_size_mb=model_size_mb,
            eval_metrics=metrics
        )
