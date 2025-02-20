from abc import ABC, abstractmethod
import torch
from typing import List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        learning_rate: float = 5e-6,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @abstractmethod
    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """Compute the loss for a batch of inputs.

        Args:
            inputs: Dictionary containing the model inputs.

        Returns:
            torch.Tensor: The computed loss.
        """
        pass

    @abstractmethod
    def compute_rewards(self, *args, **kwargs) -> Union[float, List[float]]:
        """Compute rewards for given outputs.

        Returns:
            Union[float, List[float]]: The computed reward(s).
        """
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Perform a single training step."""
        pass

    def save(self, path: str):
        """Save the model and training state.

        Args:
            path: Path to save the model and training state.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load the model and training state.

        Args:
            path: Path to load the model and training state from.
        """
        self.model.from_pretrained(path)
        self.tokenizer.from_pretrained(path)
