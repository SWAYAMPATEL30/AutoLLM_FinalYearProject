import os
import psutil
import torch
from autollm.intent.schemas import HardwareEstimate, DeviceType

class HardwareEstimator:
    """
    Analytical Feasibility and Resource Budget Calculator.
    Specifically configured for CPU-only training and inference efficiency.
    """

    def __init__(self, force_cpu: bool = True):
        self.force_cpu = force_cpu

    def inspect_system(self) -> HardwareEstimate:
        vm = psutil.virtual_memory()
        avail_ram_gb = round(vm.available / (1024 ** 3), 2)
        cpu_cores = os.cpu_count() or 4

        device = DeviceType.CPU
        cuda_vram = 0.0

        if not self.force_cpu and torch.cuda.is_available():
            device = DeviceType.CUDA
            cuda_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

        # For CPU execution, tailor the parameter budget to avoid memory swapping
        if device == DeviceType.CPU:
            if avail_ram_gb > 8:
                max_params_m = 45.0
                est_time = 2.5
            elif avail_ram_gb > 4:
                max_params_m = 25.0
                est_time = 1.8
            else:
                max_params_m = 10.0
                est_time = 1.0
            rec_quant = "PyTorch CPU Native (FP32) / Dynamic INT8"
            notes = (
                f"CPU Engine Active. Detected {cpu_cores} cores and {avail_ram_gb} GB RAM. "
                f"Allocating lightweight mini-architecture (~{max_params_m}M params) "
                f"for fast, smooth training on CPU without throttling."
            )
        else:
            max_params_m = 500.0
            est_time = 3.0
            rec_quant = "4-bit QLoRA / GGUF Q4_K_M"
            notes = f"CUDA GPU Active ({cuda_vram} GB VRAM). Hardware accelerated."

        return HardwareEstimate(
            device=device,
            available_ram_gb=avail_ram_gb,
            cpu_cores=cpu_cores,
            estimated_vram_gb=cuda_vram,
            recommended_max_params_m=max_params_m,
            estimated_train_time_min=est_time,
            recommended_quantization=rec_quant,
            is_feasible=True,
            feasibility_notes=notes,
        )

    def estimate_model_footprint(self, param_count_m: float, precision: str = "fp32") -> float:
        """
        Returns estimated disk/RAM footprint in MB.
        """
        bytes_per_param = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.55
        }.get(precision.lower(), 4.0)

        total_bytes = (param_count_m * 1_000_000) * bytes_per_param
        return round(total_bytes / (1024 * 1024), 2)
