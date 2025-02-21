from typing import Tuple, Optional, Dict, Any
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb

from .model_config import ModelConfig, TrainingConfig, SamplingConfig

class ModelFactory:
    """Factory class for creating and configuring language models."""
    
    @staticmethod
    def create_model(
        config: ModelConfig
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Create and configure a language model.
        
        Args:
            config: Model configuration.
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: Configured model and tokenizer.
        """
        # Initialize model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            fast_inference=config.fast_inference,
            max_lora_rank=config.lora_rank,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )

        return model, tokenizer

    @staticmethod
    def add_lora(
        model: PreTrainedModel,
        config: ModelConfig
    ) -> PreTrainedModel:
        """Add LoRA adapters to model.
        
        Args:
            model: Base model.
            config: Model configuration.
            
        Returns:
            PreTrainedModel: Model with LoRA adapters.
        """
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=config.target_modules,
            lora_alpha=config.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return model
    
    @staticmethod
    def load_lora(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        lora_path: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load saved LoRA weights.
        
        Args:
            model: Base model.
            tokenizer: Model tokenizer.
            lora_path: Path to saved LoRA weights.
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: Model with loaded weights
                and tokenizer.
        """
        model = FastLanguageModel.from_pretrained(
            model,
            tokenizer,
            lora_path,
        )
        return model, tokenizer
    
    @staticmethod
    def save_lora(model: PreTrainedModel, save_path: str) -> None:
        """Save LoRA weights.
        
        Args:
            model: Model with LoRA adapters.
            save_path: Path to save weights.
        """
        model.save_lora(save_path)

    @staticmethod
    def init_wandb(
        project_name: str,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
        **kwargs
    ) -> None:
        """Initialize Weights & Biases logging.
        
        Args:
            project_name: W&B project name.
            model_config: Model configuration.
            training_config: Optional training configuration.
            **kwargs: Additional W&B configuration.
        """
        config = {
            **model_config.to_dict(),
            **(training_config.to_dict() if training_config else {}),
            **kwargs
        }
        wandb.init(project=project_name, config=config)
        
    @staticmethod
    def generate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text: str,
        sampling_config: Optional[SamplingConfig] = None,
        system_prompt: Optional[str] = None,
        lora_path: Optional[str] = None,
    ) -> str:
        """Generate text using the model.
        
        Args:
            model: Language model.
            tokenizer: Model tokenizer.
            text: Input text prompt.
            sampling_config: Generation parameters.
            system_prompt: Optional system prompt to prepend.
            lora_path: Optional path to LoRA weights to use.
            
        Returns:
            str: Generated text.
        """
        # Format input with chat template
        if system_prompt:
            input_text = tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ], tokenize=False, add_generation_prompt=True)
        else:
            input_text = tokenizer.apply_chat_template([
                {"role": "user", "content": text},
            ], tokenize=False, add_generation_prompt=True)

        # Setup sampling parameters
        if sampling_config is None:
            sampling_config = SamplingConfig()
            
        # Handle LoRA if specified
        lora_request = None
        if lora_path:
            lora_request = model.load_lora(lora_path)

        # Generate text
        output = model.fast_generate(
            [input_text] if not lora_path else input_text,
            sampling_params=sampling_config.to_params(),
            lora_request=lora_request,
        )[0].outputs[0].text

        return output
