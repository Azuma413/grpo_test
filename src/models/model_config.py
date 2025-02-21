from dataclasses import dataclass
from typing import List, Dict, Any
from unsloth import is_bfloat16_supported
from vllm import SamplingParams

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    max_seq_length: int = 512
    lora_rank: int = 16
    model_name: str = "unsloth/Phi-4"
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.6
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["gate_proj", "up_proj", "down_proj"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format.
        
        Returns:
            Dict: Configuration as dictionary.
        """
        return {
            "max_seq_length": self.max_seq_length,
            "lora_rank": self.lora_rank,
            "model_name": self.model_name,
            "load_in_4bit": self.load_in_4bit,
            "fast_inference": self.fast_inference,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "target_modules": self.target_modules,
        }

@dataclass
class SamplingConfig:
    """Configuration for text generation sampling."""
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024
    
    def to_params(self) -> SamplingParams:
        """Convert config to vllm SamplingParams.
        
        Returns:
            SamplingParams: Configured sampling parameters.
        """
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

@dataclass
class TrainingConfig:
    """Configuration for GRPO training."""
    use_vllm: bool = True
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    bf16: bool = is_bfloat16_supported()
    fp16: bool = not is_bfloat16_supported()
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6
    max_prompt_length: int = 256
    max_completion_length: int = 200
    max_steps: int = 100
    save_steps: int = 250
    max_grad_norm: float = 0.1
    report_to: str = "wandb"
    output_dir: str = "outputs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format.
        
        Returns:
            Dict: Configuration as dictionary.
        """
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }
