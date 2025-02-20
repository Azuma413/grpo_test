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
from src.rewards import (
    XMLCountReward,
    StrictFormatReward,
    SoftFormatReward,
    EvaluationReward,
    SoftShogiFormatReward,
    StrictShogiFormatReward,
)
from src.shogi import YaneuraOuEngine
from src.trainer import GRPOTrainer

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
            
        # Initialize model and tokenizer
        model, tokenizer = ModelFactory.create_model(model_config)
        model = ModelFactory.add_lora(model, model_config)
        
        # Load dataset
        dataset = ShogiDataset()
        dataset.load_jsonl("datasets/training_data.jsonl")
        
        # Initialize reward functions
        reward_funcs = [
            XMLCountReward(),              # XML structure check
            SoftFormatReward(),           # Basic format check
            StrictFormatReward(),         # Strict format check
            EvaluationReward(),           # Engine-based evaluation
            SoftShogiFormatReward(),      # Basic shogi move format
            StrictShogiFormatReward(),    # Strict shogi move format
        ]
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            training_config=training_config,
            train_dataset=dataset.get_training_examples(),
        )
        
        # Run training
        trainer.train()
        
        # Save final model
        ModelFactory.save_lora(model, "grpo_saved_lora")
        
        # Test example
        test_board = """
| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |
| 　 | 飛 | 　 | 　 | 　 | 　 | 　 | 角 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 角 | 　 | 　 | 　 | 　 | 　 | 飛 | 　 |
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |

持ち駒：なし
"""
        # Generate moves with and without LoRA
        print("\nTesting model outputs:")
        print("-" * 50)
        
        # Without LoRA
        output = ModelFactory.generate(
            model,
            tokenizer,
            test_board,
            sampling_config=get_default_sampling_config(),
        )
        print("Without LoRA:")
        print(output)
        print("-" * 50)
        
        # With LoRA
        output = ModelFactory.generate(
            model,
            tokenizer,
            test_board,
            sampling_config=get_default_sampling_config(),
            lora_path="grpo_saved_lora",
        )
        print("With LoRA:")
        print(output)
        print("-" * 50)
        
    finally:
        # Cleanup
        engine.close()
        wandb.finish()

if __name__ == "__main__":
    main()
