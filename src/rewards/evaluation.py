import math
import re
from typing import List, Dict, Optional

from .base import BaseReward
from src.shogi.utils import move_to_usi

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
        
        engine = kwargs["engine"]
        current_position = kwargs["current_position"]
        rewards = []
        
        for completion in completions:
            try:
                # Extract move from completion
                move = self._extract_move(completion[0]["content"])
                
                # Set current position in engine
                engine.get_current_sfen(current_position)
                
                # Make the move and get evaluation
                if move:
                    # Update position with the move
                    if engine._update_position_sfen(move):
                        # Get evaluation for the resulting position
                        evaluation = engine.get_position_evaluation()
                        
                        # Normalize to [-1, 1] range using tanh
                        normalized_reward = math.tanh(evaluation / self.normalization_factor)
                        rewards.append(normalized_reward)
                    else:
                        # Failed to update position
                        rewards.append(-1.0)
                else:
                    # Invalid move format
                    rewards.append(-1.0)
            except Exception as e:
                # Invalid moves get minimum reward
                rewards.append(-1.0)
        
        return rewards
    
    def _extract_move(self, content: str) -> Optional[str]:
        """Extract move from completion content.
        
        Args:
            content: Model completion content
            
        Returns:
            Optional[str]: Extracted move in USI format, or None if invalid
        """
        try:
            # First try to extract from XML format
            move = self.extract_xml_answer(content)
            if move:
                return move_to_usi(move)
            
            # Fallback to direct content parsing
            move = content.strip()
            if len(move) >= 3:  # Basic validation for Japanese notation
                return move_to_usi(move)
            return None
        except Exception:
            return None
            
    def extract_xml_answer(self, content: str) -> Optional[str]:
        """Extract move from XML-formatted content.
        
        Args:
            content: XML-formatted content containing move
            
        Returns:
            Optional[str]: Extracted move in Japanese notation, or None if invalid
        """
        try:
            # Look for move between XML tags
            # Support both <move> and <answer> tags for flexibility
            pattern = r'<(move|answer)>(.*?)</\1>'
            match = re.search(pattern, content)
            if match:
                return match.group(2).strip()
            return None
        except Exception:
            return None
