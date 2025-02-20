from .base import BaseReward
from .format import (
    XMLCountReward,
    StrictFormatReward,
    SoftFormatReward,
    SoftShogiFormatReward,
    StrictShogiFormatReward,
)
from .evaluation import EvaluationReward

# Create default instances for backward compatibility
xmlcount_reward_func = XMLCountReward()
strict_format_reward_func = StrictFormatReward()
soft_format_reward_func = SoftFormatReward()
soft_shogi_format_reward_func = SoftShogiFormatReward()
strict_shogi_format_reward_func = StrictShogiFormatReward()
evaluation_reward_func = EvaluationReward()

__all__ = [
    # Classes
    "BaseReward",
    "XMLCountReward",
    "StrictFormatReward", 
    "SoftFormatReward",
    "SoftShogiFormatReward",
    "StrictShogiFormatReward",
    "EvaluationReward",
    # Function-style instances (for backward compatibility)
    "xmlcount_reward_func",
    "strict_format_reward_func",
    "soft_format_reward_func",
    "soft_shogi_format_reward_func", 
    "strict_shogi_format_reward_func",
    "evaluation_reward_func",
]
