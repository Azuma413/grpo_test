import re
from typing import List, Dict

from .base import BaseReward

class XMLCountReward(BaseReward):
    """Reward function that checks XML tag structure and placement."""
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on XML tag count and placement.
        
        Args:
            completions: List of model completions.
            **kwargs: Not used by this reward function.
            
        Returns:
            List[float]: Rewards in range [0.0, 0.5] based on XML structure.
        """
        contents = [completion[0]["content"] for completion in completions]
        return [self._count_xml(c) for c in contents]
    
    def _count_xml(self, text: str) -> float:
        """Count and score XML tags in text.
        
        Rewards:
            - 0.125 for correct <reasoning> opening tag with newline
            - 0.125 for correct </reasoning> closing tag with newlines
            - 0.125 for correct <answer> opening tag with newline
            - 0.125 for correct </answer> closing tag with newline
            - Small penalties for content after final tag
        """
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            # Penalize content after the final tag
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            # Additional penalty for content after final tag
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return max(0.0, count)  # Ensure non-negative reward

class StrictFormatReward(BaseReward):
    """Reward function that enforces strict format compliance."""
    
    def __init__(self):
        self._pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on strict format compliance.
        
        Args:
            completions: List of model completions.
            **kwargs: Not used by this reward function.
            
        Returns:
            List[float]: 0.5 for perfect format, 0.0 otherwise.
        """
        responses = [completion[0]["content"] for completion in completions]
        return [0.5 if re.match(self._pattern, r, re.DOTALL) else 0.0 
                for r in responses]

class SoftFormatReward(BaseReward):
    """Reward function that allows more flexible format compliance."""
    
    def __init__(self):
        self._pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on more lenient format requirements.
        
        Args:
            completions: List of model completions.
            **kwargs: Not used by this reward function.
            
        Returns:
            List[float]: 0.5 for acceptable format, 0.0 otherwise.
        """
        responses = [completion[0]["content"] for completion in completions]
        return [0.5 if re.match(self._pattern, r, re.DOTALL) else 0.0 
                for r in responses]

class SoftShogiFormatReward(BaseReward):
    """Reward function that checks basic shogi move format."""
    
    def __init__(self):
        # Matches format: [１-９7-9][一二三四五六七八九123456789][歩香桂銀金角飛玉と馬龍]
        self._pattern = r"^[１-９7-9][一二三四五六七八九123456789][歩香桂銀金角飛玉と馬龍]$"
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on basic shogi move format.
        
        Args:
            completions: List of model completions.
            **kwargs: Not used by this reward function.
            
        Returns:
            List[float]: 0.5 for correct format, 0.0 otherwise.
        """
        responses = [self.extract_xml_answer(completion[0]["content"]) 
                    for completion in completions]
        return [0.5 if re.match(self._pattern, r) else 0.0 
                for r in responses]

class StrictShogiFormatReward(BaseReward):
    """Reward function that strictly validates shogi moves."""
    
    def __init__(self):
        self._number_pattern = r"^[１-９]"  # 1-9 in kanji
        self._position_pattern = r"[一二三四五六七八九]"  # Position in kanji
        self._piece_pattern = r"[歩香桂銀金角飛玉と馬龍]$"  # Piece types
    
    def calculate(self, completions: List[Dict], **kwargs) -> List[float]:
        """Calculate reward based on strict shogi move validation.
        
        Args:
            completions: List of model completions.
            **kwargs:
                current_position: Current board position (required)
                engine: Shogi engine instance (required)
            
        Returns:
            List[float]: 1.0 for legal moves, 0.5 for valid format, 0.0 otherwise.
        """
        if "current_position" not in kwargs or "engine" not in kwargs:
            raise ValueError("current_position and engine are required")
        
        responses = [self.extract_xml_answer(completion[0]["content"]) 
                    for completion in completions]
        rewards = []
        
        for response in responses:
            # Check basic format first
            if (re.match(self._number_pattern, response) and
                re.search(self._position_pattern, response) and
                re.search(self._piece_pattern, response)):
                reward = 0.5
                # Validate move with engine
                if kwargs["engine"].is_legal_move(kwargs["current_position"], response):
                    reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)
        
        return rewards
