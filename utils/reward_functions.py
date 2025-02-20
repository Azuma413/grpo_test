import re
from typing import List, Union, Callable
from game.engine_interface import EngineInterface

# Type alias for reward functions
RewardFunction = Callable[[List[str], ...], List[float]]

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text)
    if not match:
        raise ValueError("No answer tag found in text")
    return match.group(1).strip()

def xmlcount_reward_func(completions: List[Union[str, dict]]) -> List[float]:
    """Reward based on proper XML format."""
    rewards = []
    for completion in completions:
        content = completion["content"] if isinstance(completion, dict) else completion
        try:
            _ = extract_xml_answer(content)
            rewards.append(1.0)
        except ValueError:
            rewards.append(0.0)
    return rewards

def soft_format_reward_func(completions: List[Union[str, dict]]) -> List[float]:
    """Basic format check reward."""
    pattern = r'[１-９][一二三四五六七八九]'
    rewards = []
    for completion in completions:
        content = completion["content"] if isinstance(completion, dict) else completion
        try:
            answer = extract_xml_answer(content)
            # Check for basic format (e.g., ７六)
            if re.search(pattern, answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards

def strict_format_reward_func(completions: List[Union[str, dict]]) -> List[float]:
    """Strict format check reward."""
    pattern = r'^[１-９][一二三四五六七八九](歩|角|飛|香|桂|銀|金|玉|と|成香|成桂|成銀|馬|龍)?(成|不成)?$'
    rewards = []
    for completion in completions:
        content = completion["content"] if isinstance(completion, dict) else completion
        try:
            answer = extract_xml_answer(content)
            if re.match(pattern, answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards

def evaluation_reward_func(
    completions: List[Union[str, dict]],
    current_position: str,
    engine: EngineInterface
) -> List[float]:
    """Reward based on engine evaluation."""
    rewards = []
    for completion in completions:
        try:
            content = completion["content"] if isinstance(completion, dict) else completion
            move = extract_xml_answer(content)
            
            # Set position and get evaluation
            engine.set_position(current_position)
            engine.go_bestmove(1000)  # 1 second evaluation
            best_move = engine.get_bestmove()
            
            # Compare with best move
            if best_move and move == best_move:
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # Partial reward for valid move
        except Exception:
            rewards.append(0.0)
    return rewards

def soft_shogi_format_reward_func(completions: List[Union[str, dict]]) -> List[float]:
    """Relaxed shogi move format check."""
    valid_pieces = "歩|角|飛|香|桂|銀|金|玉|と|成香|成桂|成銀|馬|龍"
    pattern = f"[１-９][一二三四五六七八九]({valid_pieces})?"
    rewards = []
    for completion in completions:
        content = completion["content"] if isinstance(completion, dict) else completion
        try:
            answer = extract_xml_answer(content)
            if re.search(pattern, answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards

def strict_shogi_format_reward_func(
    completions: List[Union[str, dict]],
    current_position: str,
    engine: EngineInterface
) -> List[float]:
    """Strict shogi move validation using engine."""
    rewards = []
    for completion in completions:
        try:
            content = completion["content"] if isinstance(completion, dict) else completion
            move = extract_xml_answer(content)
            
            # Validate move with engine
            engine.set_position(current_position)
            valid_moves = engine.get_legal_moves()
            
            if move in valid_moves:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# Create a dictionary mapping reward function names to their default weights
DEFAULT_REWARD_WEIGHTS = {
    "xmlcount_reward_func": 1.0,
    "soft_format_reward_func": 1.0,
    "strict_format_reward_func": 1.0,
    "evaluation_reward_func": 2.0,
    "soft_shogi_format_reward_func": 1.0,
    "strict_shogi_format_reward_func": 2.0,
}
