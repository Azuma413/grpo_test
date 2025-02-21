"""
Main script for GRPO-based shogi model training.
"""
import wandb
from src.config import (
    get_default_model_config,
    get_default_training_config,
    get_default_sampling_config,
)
from src.data import ShogiDataset
from src.models import ModelFactory
from src.rewards import RewardFunctions
from src.shogi import YaneuraOuEngine
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

def main():
    """Main training routine."""
    # Get configurations
    model_config = get_default_model_config()
    training_config = get_default_training_config()
    
    # Initialize wandb
    ModelFactory.init_wandb(
        project_name="grpo-training",
        model_config=model_config,
        training_config=training_config,
    )
    
    try:
        # Initialize shogi engine
        engine = YaneuraOuEngine()
        if not engine.start():
            raise RuntimeError("Failed to start YaneuraOu engine")
        
        max_seq_length = 512 # Can increase for longer reasoning traces
        lora_rank = 16 # Larger rank = smarter, but slower

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Phi-4",
            max_seq_length = max_seq_length,
            load_in_4bit = True, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = lora_rank,
            gpu_memory_utilization = 0.6, # Reduce if out of memory
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["gate_proj", "up_proj", "down_proj",],
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
            random_state = 3407,
        )   

        training_args = GRPOConfig(
            use_vllm = True, # use vLLM for fast inference!
            learning_rate = 5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "paged_adamw_8bit",
            logging_steps = 1,
            bf16 = is_bfloat16_supported(),
            fp16 = not is_bfloat16_supported(),
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 6, # Decrease if out of memory
            max_prompt_length = 256,
            max_completion_length = 200,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = 100,
            save_steps = 250,
            max_grad_norm = 0.1,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",
        )
        # Load dataset
        dataset = ShogiDataset()
        dataset.load_jsonl("datasets/training_data.jsonl")
        reward_functions = RewardFunctions()
        # Initialize reward functions
        reward_funcs = [
            reward_functions.xml_reward,
            reward_functions.strict_format_reward,
            reward_functions.soft_format_reward,
            reward_functions.soft_shogi_format_reward,
            reward_functions.strict_shogi_reward,
            reward_functions.evaluation_reward
        ]
        
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = reward_funcs,
            args = training_args,
            train_dataset = dataset.get_training_examples(),
        )
        
        # Run training
        trainer.train()
        
        # Save final model
        model.save_lora("grpo_saved_lora")
    finally:
        # Cleanup
        engine.close()
        wandb.finish()

if __name__ == "__main__":
    main()
