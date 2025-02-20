from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

import wandb
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig

from ..models import TrainingConfig
from ..rewards.base import BaseReward

class BaseTrainer(ABC):
    """Base class for model trainers."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_funcs: List[BaseReward],
        training_config: TrainingConfig,
        train_dataset: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            tokenizer: Model tokenizer.
            reward_funcs: List of reward functions.
            training_config: Training configuration.
            train_dataset: Training examples.
            output_dir: Directory to save outputs (default: training_config.output_dir).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.output_dir = output_dir or training_config.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize training state
        self.current_step = 0
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0
        
    @abstractmethod
    def train(self) -> None:
        """Run training loop."""
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate current model.
        
        Returns:
            float: Average reward across evaluation examples.
        """
        pass
    
    def save_checkpoint(self, step: int, reward: float) -> None:
        """Save training checkpoint.
        
        Args:
            step: Current training step.
            reward: Current reward value.
        """
        checkpoint_dir = Path(self.output_dir) / f"checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "step": step,
            "reward": reward,
            "best_reward": self.best_reward,
            "early_stopping_counter": self.early_stopping_counter,
        }
        wandb.save(os.path.join(checkpoint_dir, "trainer_state.pt"))
        
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory.
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        self.model = self.model.from_pretrained(checkpoint_dir)
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_dir)
        
        # Load training state
        state = wandb.restore(
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            run_path=wandb.run.path
        )
        self.current_step = state["step"]
        self.best_reward = state["best_reward"]
        self.early_stopping_counter = state["early_stopping_counter"]
        
    def _should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved.
        
        Args:
            step: Current training step.
            
        Returns:
            bool: True if checkpoint should be saved.
        """
        return step > 0 and step % self.training_config.save_steps == 0
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to tracking system.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for logging.
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
            
    def _get_grpo_config(self) -> GRPOConfig:
        """Create GRPO training configuration.
        
        Returns:
            GRPOConfig: Configured GRPO training arguments.
        """
        return GRPOConfig(
            **self.training_config.to_dict()
        )
