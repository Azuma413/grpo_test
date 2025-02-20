import math
from typing import List, Dict

from .base import BaseReward

class EvaluationReward(BaseReward):
    """Reward function based on YaneuraOu engine's position evaluation."""
    
    def __init__(self, normalization_factor: float = 1000.0):
        """Initialize the evaluation reward function.
        
        Args:
            normalization_factor: Value used to normalize engine evaluation scores.
                Default is 1000.0, which normalizes typical centipawn values to
                a reasonable range when using tanh.
        """
        self.normalization_factor = normalization_factor
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on engine evaluation of positions after moves.
        
        Args:
            completions: List of model completions.
            **kwargs:
                current_position: Current board position (required)
                engine: YaneuraOu engine instance (required)
            
        Returns:
            List[float]: Normalized evaluation scores in range [-1.0, 1.0].
                - Positive values indicate good moves
                - Negative values indicate bad moves
                - Values close to 0 indicate neutral moves
        
        Raises:
            ValueError: If required kwargs are missing.
        """
        if "current_position" not in kwargs or "engine" not in kwargs:
            raise ValueError("current_position and engine are required")
        
        responses = [self.extract_xml_answer(completion[0]["content"]) 
                    for completion in completions]
        rewards = []
        
        for response in responses:
            try:
                # Get evaluation from engine
                evaluation = kwargs["engine"].get_position_evaluation(
                    kwargs["current_position"], 
                    response
                )
                # Normalize to [-1, 1] range using tanh
                normalized_reward = math.tanh(evaluation / self.normalization_factor)
                rewards.append(normalized_reward)
            except Exception as e:
                # Invalid moves get minimum reward
                rewards.append(-1.0)
        
        return rewards
