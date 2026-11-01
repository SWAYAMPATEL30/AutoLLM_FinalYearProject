from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class RouteType(str, Enum):
    FROM_SCRATCH = "from_scratch"
    PEFT_FINETUNE = "peft_finetune"
    RAG = "rag"
    DESIGN_PREVIEW = "design_preview"

class TaskType(str, Enum):
    INSTRUCTION_QA = "instruction-qa"
    CHAT = "chat"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    GENERAL = "general"

class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

class HardwareEstimate(BaseModel):
    device: DeviceType = DeviceType.CPU
    available_ram_gb: float
    cpu_cores: int
    estimated_vram_gb: float = 0.0
    recommended_max_params_m: float
    estimated_train_time_min: float
    recommended_quantization: str = "FP32 / INT8 (CPU)"
    is_feasible: bool = True
    feasibility_notes: str = ""

class PipelineSpec(BaseModel):
    raw_prompt: str
    domain: str = "general"
    task_type: TaskType = TaskType.INSTRUCTION_QA
    route: RouteType = RouteType.PEFT_FINETUNE
    target_device: DeviceType = DeviceType.CPU
    target_size_mb: float = 2048.0
    language: str = "English"
    base_model: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct"
    training_mode: str = "qwen_lora"
    dataset_query: Optional[str] = None
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_seq_len: int = 256
    hardware_estimate: Optional[HardwareEstimate] = None

class InstructionSample(BaseModel):
    instruction: str
    input: str = ""
    output: str

class DatasetOption(BaseModel):
    id: str
    name: str
    source: str = "Hugging Face Hub"
    description: str
    downloads: int = 0
    likes: int = 0
    estimated_samples: str = "1,000+ QA pairs"
    sample_preview: str = ""
    is_recommended: bool = False

class DatasetSpec(BaseModel):
    name: str
    domain: str
    source_type: str = "curated_domain | hf_hub"
    total_samples: int
    train_samples: int
    val_samples: int
    local_path: str
    preview: List[InstructionSample] = Field(default_factory=list)

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class PipelineProgress(BaseModel):
    job_id: str
    step: str = "init"
    percentage: float = 0.0
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_loss: Optional[float] = None
    logs: List[str] = Field(default_factory=list)
    model_artifact_path: Optional[str] = None
    api_url: Optional[str] = None

class DeploymentPackage(BaseModel):
    job_id: str
    model_dir: str
    model_weights_path: str
    tokenizer_path: str
    serve_script_path: str
    interactive_ui_path: str
    model_size_mb: float
    eval_metrics: Dict[str, Any] = Field(default_factory=dict)
