import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base_trainer import BaseTrainer

class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization (GRPO) trainer implementation."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_funcs: List[Callable],
        beta: float = 0.1,
        num_generations: int = 4,
        max_completion_length: int = 128,
        learning_rate: float = 5e-6,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(model, tokenizer, learning_rate, device)
        self.reward_funcs = reward_funcs
        self.beta = beta
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        
        # Create reference model for KL divergence computation
        self.ref_model = type(model)(model.config).eval().to(device)
        self.ref_model.load_state_dict(model.state_dict())

    def compute_rewards(self, prompts: List[str], completions: List[str], **kwargs) -> torch.Tensor:
        """Compute rewards for each completion using multiple reward functions.
        
        Args:
            prompts: List of input prompts.
            completions: List of generated completions.
            **kwargs: Additional arguments passed to reward functions.
            
        Returns:
            torch.Tensor: Tensor of rewards of shape (batch_size,).
        """
        all_rewards = []
        for func in self.reward_funcs:
            if func.__name__ == 'evaluation_reward_func':
                r = torch.tensor(func(completions, kwargs.get('current_position'), kwargs.get('engine')))
            elif func.__name__ == 'strict_shogi_format_reward_func':
                r = torch.tensor(func(completions, kwargs.get('current_position'), kwargs.get('engine')))
            else:
                r = torch.tensor(func(completions))
            all_rewards.append(r)
        
        # Combine rewards (simple average for now, could be weighted)
        rewards = torch.stack(all_rewards).mean(dim=0)
        return rewards.to(self.device)

    def compute_kl_divergence(
        self,
        model_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between model and reference model distributions."""
        model_probs = F.softmax(model_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        per_token_kl = F.kl_div(
            model_probs.log(),
            ref_probs,
            reduction='none'
        ).sum(-1)
        
        # Mask out padding tokens
        return (per_token_kl * attention_mask).sum() / attention_mask.sum()

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using group-wise reward normalization.
        
        Args:
            rewards: Tensor of rewards of shape (batch_size,).
            
        Returns:
            torch.Tensor: Normalized advantages of shape (batch_size,).
        """
        # Reshape rewards to (num_groups, num_generations)
        grouped_rewards = rewards.view(-1, self.num_generations)
        
        # Compute mean and std per group
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        
        # Normalize rewards within each group
        advantages = (grouped_rewards - group_means) / (group_stds + 1e-8)
        
        # Reshape back to (batch_size,)
        return advantages.reshape(-1)

    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """Compute the GRPO loss for a batch of inputs."""
        # Get model outputs
        model_outputs = self.model(**inputs)
        model_logits = model_outputs.logits
        
        # Get reference model outputs for KL divergence
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(
            model_logits,
            ref_logits,
            inputs['attention_mask']
        )
        
        # Compute policy loss
        advantages = self.compute_advantages(inputs['rewards'])
        policy_loss = -(model_outputs.log_probs * advantages.unsqueeze(-1)).mean()
        
        # Combine losses
        total_loss = policy_loss + self.beta * kl_div
        return total_loss

    def train_step(self, prompt: str, completion: str, reward: float, **kwargs):
        """Perform a single training step.
        
        Args:
            prompt: Input prompt.
            completion: Generated completion.
            reward: Computed reward.
            **kwargs: Additional arguments.
            
        Returns:
            float: The loss value for this step.
        """
        # Prepare inputs
        inputs = self.tokenizer(
            prompt,
            completion,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_completion_length
        ).to(self.device)
        
        # Add rewards to inputs
        inputs['rewards'] = torch.tensor([reward], device=self.device)
        
        # Compute loss
        loss = self.compute_loss(inputs)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update reference model periodically (here we update every step)
        self.ref_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
