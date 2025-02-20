from typing import List, Dict, Any, Optional
import torch
from trl import GRPOTrainer as TRLGRPOTrainer

from .trainer_base import BaseTrainer
from ..models import TrainingConfig

class GRPOTrainer(BaseTrainer):
    """Trainer for GRPO (Gradient Reward Policy Optimization)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize GRPO trainer."""
        super().__init__(*args, **kwargs)
        
        # Initialize TRL trainer
        self.trl_trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_funcs,
            args=self._get_grpo_config(),
            train_dataset=self.train_dataset,
        )
        
        # Configure early stopping
        self.patience = getattr(self.training_config, "early_stopping_patience", 3)
        self.min_delta = getattr(self.training_config, "early_stopping_delta", 0.01)
        
    def train(self) -> None:
        """Run GRPO training loop.
        
        This implements the main training loop with:
        - Gradient accumulation
        - Learning rate scheduling
        - Gradient clipping
        - Checkpointing
        - Early stopping
        - Metric logging
        """
        try:
            # Training loop
            while self.current_step < self.training_config.max_steps:
                # Train step
                metrics = self.trl_trainer.train()
                self.current_step += 1
                
                # Log metrics
                self._log_metrics(metrics, step=self.current_step)
                
                # Evaluate and save checkpoint
                if self._should_save(self.current_step):
                    reward = self.evaluate()
                    self._log_metrics(
                        {"eval/reward": reward},
                        step=self.current_step
                    )
                    
                    # Save checkpoint if best so far
                    if reward > self.best_reward + self.min_delta:
                        self.best_reward = reward
                        self.early_stopping_counter = 0
                        self.save_checkpoint(self.current_step, reward)
                    else:
                        self.early_stopping_counter += 1
                        
                    # Early stopping check
                    if self.early_stopping_counter >= self.patience:
                        print(f"Early stopping triggered at step {self.current_step}")
                        break
                        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            # Save final checkpoint
            final_reward = self.evaluate()
            self.save_checkpoint(self.current_step, final_reward)
            
    def evaluate(self) -> float:
        """Evaluate current model.
        
        Generates responses for evaluation examples and calculates
        average reward across all reward functions.
        
        Returns:
            float: Average reward value.
        """
        # Get evaluation examples
        eval_examples = self.train_dataset  # Using training data for now
        
        # Generate responses
        all_rewards = []
        for example in eval_examples:
            responses = self.trl_trainer.generate(
                example["prompt"],
                num_return_sequences=1
            )
            
            # Calculate rewards from each reward function
            example_rewards = []
            for reward_func in self.reward_funcs:
                rewards = reward_func(
                    [responses],
                    **example.get("metadata", {})  # Pass any additional data
                )
                example_rewards.extend(rewards)
                
            # Average rewards for this example
            all_rewards.append(sum(example_rewards) / len(example_rewards))
            
        # Return average reward across all examples
        return sum(all_rewards) / len(all_rewards)
