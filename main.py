from typing import Optional
import torch

from models.model_factory import ModelFactory, ModelConfig
from trainer.online_trainer import OnlineGRPOTrainer
from game.engine_interface import YaneuraOuEngineWrapper
from utils.logging import setup_logging, log_training_config
from utils.reward_functions import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    evaluation_reward_func,
    soft_shogi_format_reward_func,
    strict_shogi_format_reward_func,
    DEFAULT_REWARD_WEIGHTS
)
from config import get_model_config, get_training_config
from data_utils import SYSTEM_PROMPT
from shogi_engine import YaneuraOuEngine

def create_trainer(
    model_config: ModelConfig,
    system_prompt: str,
    engine: YaneuraOuEngineWrapper,
    learning_rate: float = 5e-6,
    device: Optional[str] = None,
) -> OnlineGRPOTrainer:
    """Create and configure the trainer.
    
    Args:
        model_config: Model configuration.
        system_prompt: System prompt for the model.
        engine: Shogi engine interface.
        learning_rate: Learning rate for optimization.
        device: Device to use for training.
        
    Returns:
        OnlineGRPOTrainer: Configured trainer instance.
    """
    # Create model and tokenizer
    model, tokenizer = ModelFactory.create_model(model_config)
    
    # Configure reward functions with weights
    reward_funcs = [
        xmlcount_reward_func,              # XML structure check
        soft_format_reward_func,           # Basic format check
        strict_format_reward_func,         # Strict format check
        evaluation_reward_func,            # Engine evaluation reward
        soft_shogi_format_reward_func,     # Basic shogi format check
        strict_shogi_format_reward_func,   # Strict shogi format check
    ]
    
    # Create trainer
    trainer = OnlineGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        reward_funcs=reward_funcs,
        system_prompt=system_prompt,
        learning_rate=learning_rate,
        device=device,
        beta=0.1,  # KL divergence weight
        num_generations=4,  # Number of candidates per move
        max_completion_length=64,
    )
    
    return trainer

def train(
    trainer: OnlineGRPOTrainer,
    num_games: int,
    logger,
    save_path: Optional[str] = "grpo_saved_lora"
):
    """Train the model through self-play.
    
    Args:
        trainer: The trainer instance.
        num_games: Number of games to play.
        logger: Logger instance for metrics.
        save_path: Path to save model weights (optional).
    """
    for game_idx in range(num_games):
        try:
            game_reward, moves = trainer.train_game()
            logger.log({
                "game": game_idx,
                "game_reward": game_reward,
                "num_moves": len(moves)
            })
        except Exception as e:
            print(f"Error in game {game_idx}: {e}")
            continue
    
    if save_path:
        ModelFactory.save_model(trainer.model, trainer.tokenizer, save_path)

def test_model(
    model_config: ModelConfig,
    question: str,
    use_lora: bool = False,
    lora_path: Optional[str] = "grpo_saved_lora"
):
    """Test the model on a given question.
    
    Args:
        model_config: Model configuration.
        question: Input question/position.
        use_lora: Whether to use trained LoRA weights.
        lora_path: Path to LoRA weights if use_lora is True.
    """
    # Create base model and tokenizer
    model, tokenizer = ModelFactory.create_model(model_config)
    
    # Apply LoRA weights if requested
    if use_lora:
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        lora_request = model.load_lora(lora_path)
    else:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        lora_request = None

    # Generate response
    sampling_params = ModelFactory.get_sampling_params()
    output = model.fast_generate(
        [text] if not use_lora else text,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text

    print(f"{'With LoRA:' if use_lora else 'Without LoRA:'}")
    print(output)
    print("-" * 50)

def main():
    # Get configurations
    model_config = get_model_config()
    training_config = get_training_config()
    
    # Update and log configuration
    config = {**model_config, **training_config}
    config = log_training_config(config)
    
    # Initialize logger
    logger = setup_logging(
        project_name="grpo-online-training",
        config=config,
    )
    
    # Initialize engine
    engine = YaneuraOuEngine()
    if not engine.start():
        raise RuntimeError("Failed to start YaneuraOu engine")
    engine_wrapper = YaneuraOuEngineWrapper(engine.process)
    
    try:
        # Create and train model
        trainer = create_trainer(
            model_config=ModelConfig(**model_config),
            system_prompt=SYSTEM_PROMPT,
            engine=engine_wrapper,
            learning_rate=training_config["learning_rate"],
        )
        
        train(
            trainer=trainer,
            num_games=training_config["num_games"],
            logger=logger,
        )
        
        # Test example
        test_question = """
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
        test_model(ModelConfig(**model_config), test_question, use_lora=False)
        test_model(ModelConfig(**model_config), test_question, use_lora=True)
        
    finally:
        logger.finish()
        engine_wrapper.close()

if __name__ == "__main__":
    main()
