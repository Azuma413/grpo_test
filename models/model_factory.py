from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from unsloth import FastLanguageModel, PatchFastRL

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    model_name: str
    max_seq_length: int
    lora_rank: int
    load_in_4bit: bool
    fast_inference: bool
    gpu_memory_utilization: float
    target_modules: List[str]
    random_seed: int = 3407

class ModelFactory:
    """Factory for creating and configuring models."""

    @staticmethod
    def create_model(
        config: ModelConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Create and configure a model with its tokenizer.
        
        Args:
            config: Model configuration.
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: Configured model and tokenizer.
        """
        # Patch FastLanguageModel for GRPO
        PatchFastRL("GRPO", FastLanguageModel)
        
        # Initialize model and tokenizer with forced memory settings
        import os
        os.environ["VLLM_FORCED_GPU_MEMORY"] = str(config.gpu_memory_utilization)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            fast_inference=config.fast_inference,
            max_lora_rank=config.lora_rank,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_exllama=True,  # Prevent memory optimizations that might override settings
        )
        
        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=config.target_modules,
            lora_alpha=config.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=config.random_seed,
        )
        
        return model, tokenizer

    @staticmethod
    def save_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        path: str,
        save_full_model: bool = False
    ):
        """Save model and tokenizer.
        
        Args:
            model: The model to save.
            tokenizer: The tokenizer to save.
            path: Path to save to.
            save_full_model: Whether to save the full model or just LoRA weights.
        """
        if save_full_model:
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
        else:
            model.save_lora(path)

    @staticmethod
    def load_model(
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        path: str,
        is_lora: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load a saved model.
        
        Args:
            base_model: Base model to apply LoRA weights to.
            tokenizer: Tokenizer to use.
            path: Path to load from.
            is_lora: Whether loading LoRA weights or full model.
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: Loaded model and tokenizer.
        """
        if is_lora:
            base_model.load_lora(path)
            return base_model, tokenizer
        else:
            model = PreTrainedModel.from_pretrained(path)
            tokenizer = PreTrainedTokenizerBase.from_pretrained(path)
            return model, tokenizer

    @staticmethod
    def get_sampling_params(
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> dict:
        """Get generation sampling parameters.
        
        Args:
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to return.
            
        Returns:
            dict: Sampling parameters.
        """
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
        }
