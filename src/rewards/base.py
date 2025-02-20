from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseReward(ABC):
    """Base class for all reward functions used in GRPO training.
    
    All reward functions should inherit from this class and implement
    the calculate method.
    """
    
    @abstractmethod
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate rewards for a batch of model completions.
        
        Args:
            completions: List of model completions, where each completion is a
                dictionary containing the model's output.
            **kwargs: Additional arguments that may be required by specific
                reward functions (e.g., current_position for evaluation rewards).
        
        Returns:
            List[float]: A list of reward values for each completion.
        """
        pass

    def __call__(self, completions: List[Dict], **kwargs) -> List[float]:
        """Convenience method to make reward functions callable.
        
        Args:
            completions: List of model completions.
            **kwargs: Additional arguments passed to calculate().
        
        Returns:
            List[float]: Reward values for each completion.
        """
        return self.calculate(completions, **kwargs)

    @staticmethod
    def extract_xml_answer(text: str) -> str:
        """Extract content between <answer> tags from text.
        
        Args:
            text: Input text containing XML-style answer tags.
        
        Returns:
            str: Content between <answer> tags, or empty string if not found.
        """
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except IndexError:
            return ""
