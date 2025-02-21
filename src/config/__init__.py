from pathlib import Path
from typing import Dict, Any
import os

from ..models import ModelConfig, TrainingConfig, SamplingConfig

def get_default_model_config() -> ModelConfig:
    """Get default model configuration.
    
    Returns:
        ModelConfig: Default configuration for model initialization.
    """
    return ModelConfig(
        max_seq_length=512,
        lora_rank=16,
        model_name="unsloth/Phi-4",
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )

def get_default_training_config() -> TrainingConfig:
    """Get default training configuration.
    
    Returns:
        TrainingConfig: Default configuration for GRPO training.
    """
    return TrainingConfig(
        output_dir=str(Path(os.getcwd()) / "outputs"),
    )

def get_default_sampling_config() -> SamplingConfig:
    """Get default sampling configuration.
    
    Returns:
        SamplingConfig: Default configuration for text generation.
    """
    return SamplingConfig(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )

# For backward compatibility
def get_model_config() -> Dict[str, Any]:
    """Get model configuration dictionary (legacy format).
    
    Returns:
        Dict[str, Any]: Model configuration dictionary.
    """
    return get_default_model_config().to_dict()

def get_training_config(lora_rank: int) -> Dict[str, Any]:
    """Get training configuration dictionary (legacy format).
    
    Args:
        lora_rank: LoRA rank parameter.
        
    Returns:
        Dict[str, Any]: Training configuration dictionary.
    """
    config = get_default_training_config()
    config.lora_rank = lora_rank
    return config.to_dict()

def get_sampling_params() -> Dict[str, Any]:
    """Get sampling parameters dictionary (legacy format).
    
    Returns:
        Dict[str, Any]: Sampling configuration dictionary.
    """
    return get_default_sampling_config().to_params()
