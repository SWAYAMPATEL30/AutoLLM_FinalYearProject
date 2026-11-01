import subprocess
import sys
import os
import json
from typing import Callable, Optional, Dict, Any

class ScriptRunner:
    """
    Subprocess execution manager that runs generated training jobs
    and streams structured events and logs in real-time.
    """

    @staticmethod
    def execute(
        script_path: str,
        cwd: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_log: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure project root is in PYTHONPATH
        project_root = os.path.abspath(os.path.join(cwd, "..", ".."))
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [sys.executable, script_path]

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        final_metrics = {}
        error_logs = []

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                clean_line = line.strip()
                if not clean_line:
                    continue

                if clean_line.startswith("AUTOLLM_EVENT:"):
                    json_str = clean_line[len("AUTOLLM_EVENT:"):].strip()
                    try:
                        event_data = json.loads(json_str)
                        if event_data.get("type") == "completed":
                            final_metrics = event_data.get("metrics", {})
                        if on_event:
                            on_event(event_data)
                    except json.JSONDecodeError:
                        if on_log:
                            on_log(clean_line)
                else:
                    if on_log:
                        on_log(clean_line)
                    error_logs.append(clean_line)

        process.wait()
        return_code = process.returncode

        if return_code != 0:
            err_msg = "\n".join(error_logs[-10:])
            raise RuntimeError(f"Training script failed with exit code {return_code}:\n{err_msg}")

        return final_metrics
