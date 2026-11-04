import os
import psutil
from typing import Dict, Any
from autollm.intent.schemas import DesignReportSpec, HardwareEstimate

class ArchitecturePlanner:
    """
    Branch 4 in Decision Router: Design-Only Architecture Preview.
    Provides rigorous analytical hardware budgeting, network dimensioning,
    and memory footprint calculations without running training compute.
    """

    @staticmethod
    def design_architecture(
        target_size_mb: float = 1024.0,
        domain: str = "general",
        context_window: int = 512,
        vocab_size: int = 32000
    ) -> DesignReportSpec:
        vm = psutil.virtual_memory()
        avail_ram_gb = round(vm.available / (1024 ** 3), 2)
        cpu_cores = os.cpu_count() or 4

        # Target parameters based on target size constraint
        # Each FP32 parameter takes 4 bytes; INT8 takes 1 byte.
        # Max parameters = target_size_mb * 1024 * 1024 / 4
        max_params_from_size = (target_size_mb * 1024 * 1024) / 4.0
        
        # Determine architectural configuration tiers
        if max_params_from_size > 800_000_000:
            # 1B class
            hidden_dim = 2048
            num_layers = 16
            num_heads = 16
            name = "AutoLLM-Transformer-1B"
        elif max_params_from_size > 300_000_000:
            # 500M class (Qwen 0.5B equivalent)
            hidden_dim = 1024
            num_layers = 24
            num_heads = 16
            name = "AutoLLM-Transformer-0.5B"
        elif max_params_from_size > 100_000_000:
            # 135M class (SmolLM equivalent)
            hidden_dim = 576
            num_layers = 30
            num_heads = 9
            name = "AutoLLM-Transformer-135M"
        elif max_params_from_size > 20_000_000:
            # 25M class (SLM on-device)
            hidden_dim = 384
            num_layers = 12
            num_heads = 6
            name = "AutoLLM-Transformer-25M"
        else:
            # MiniGPT CPU Class
            hidden_dim = 128
            num_layers = 4
            num_heads = 4
            name = "AutoLLM-MiniGPT-CPU"

        # Calculate exact parameter count:
        # Embedding: vocab_size * hidden_dim
        # Attention per layer: 4 * (hidden_dim^2)
        # MLP per layer: 2 * 4 * (hidden_dim^2) = 8 * (hidden_dim^2)
        # LayerNorms: 2 * 2 * hidden_dim per layer
        # Output LM Head (tied or untied): vocab_size * hidden_dim
        params_per_block = 12 * (hidden_dim ** 2) + 4 * hidden_dim
        total_transformer_params = (num_layers * params_per_block) + (2 * vocab_size * hidden_dim)
        param_count_m = round(total_transformer_params / 1_000_000, 2)

        # Memory calculations (MB)
        weights_fp32_mb = round((total_transformer_params * 4) / (1024 * 1024), 2)
        weights_int8_mb = round((total_transformer_params * 1) / (1024 * 1024), 2)
        optimizer_adamw_mb = round((total_transformer_params * 8) / (1024 * 1024), 2) # 2 states (m, v) * 4 bytes
        gradients_mb = round((total_transformer_params * 4) / (1024 * 1024), 2)
        
        # Total training memory needed = weights + grads + optimizer states + activations
        activation_buffer_mb = round(weights_fp32_mb * 0.35, 2)
        total_training_ram_mb = weights_fp32_mb + optimizer_adamw_mb + gradients_mb + activation_buffer_mb

        # KV Cache per sequence (KB):
        # 2 * 2 * num_layers * num_heads * (hidden_dim / num_heads) * context_window * 2 bytes (FP16)
        head_dim = hidden_dim // num_heads
        kv_cache_bytes = 4 * num_layers * num_heads * head_dim * context_window * 2
        kv_cache_per_seq_kb = round(kv_cache_bytes / 1024, 2)

        # Batch size recommendation
        avail_ram_mb = avail_ram_gb * 1024
        if total_training_ram_mb < avail_ram_mb * 0.6:
            rec_batch = 8 if total_training_ram_mb < avail_ram_mb * 0.3 else 4
            status = f"FEASIBLE ON CURRENT CPU: Model requires {round(total_training_ram_mb/1024, 2)} GB RAM. System has {avail_ram_gb} GB free."
        else:
            rec_batch = 1
            status = f"HIGH RESOURCE USAGE: Training requires {round(total_training_ram_mb/1024, 2)} GB RAM. Recommend INT8 Quantized LoRA or cloud GPU."

        notes = (
            f"Designed for {domain.title()} domain. Architecture: {num_layers} layers, {hidden_dim} hidden dimension, "
            f"{num_heads} attention heads, {context_window} context tokens. "
            f"Weights compress from {weights_fp32_mb} MB (FP32) to {weights_int8_mb} MB (INT8)."
        )

        return DesignReportSpec(
            target_name=name,
            layer_count=num_layers,
            hidden_dimension=hidden_dim,
            attention_heads=num_heads,
            context_window=context_window,
            vocab_size=vocab_size,
            parameter_count_m=param_count_m,
            model_weights_fp32_mb=weights_fp32_mb,
            model_weights_int8_mb=weights_int8_mb,
            optimizer_memory_adamw_mb=optimizer_adamw_mb,
            gradient_memory_mb=gradients_mb,
            kv_cache_per_seq_kb=kv_cache_per_seq_kb,
            recommended_batch_size=rec_batch,
            feasibility_status=status,
            architectural_notes=notes
        )
