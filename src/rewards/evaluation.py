import math
import re
from typing import List, Dict, Optional, Union
from src.shogi.utils import move_to_usi
from src.shogi.engine import YaneuraOuEngine

class RewardFunctions():
    """Reward function based on YaneuraOu engine's position evaluation."""
    
    def __init__(self, engine_path: str = "/mnt/e/SourceCode/app/yaneuraou/YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe", normalization_factor: float = 100.0):
        """Initialize rewards with YaneuraOu engine.
        
        Args:
            engine_path: Path to YaneuraOu engine executable
            normalization_factor: Value used to normalize engine evaluation scores
        """
        self.engine = YaneuraOuEngine(engine_path=engine_path)
        if not self.engine.start():
            raise RuntimeError("Failed to start YaneuraOu engine")
            
        self.normalization_factor = normalization_factor
        self._number_pattern = r"^[１-９]"  # 1-9 in kanji
        self._position_pattern = r"[一二三四五六七八九]"  # Position in kanji
        self._piece_pattern = r"[歩香桂銀金角飛玉と馬龍]$"  # Piece types
    
    def xml_reward(self, prompts, completions: List[Dict], **kwargs) -> List[float]:
        """
        出力がXML形式になっているかどうかを評価する
        0~0.5
        """
        contents = [completion[0]["content"] for completion in completions]
        return [self._validate_xml_structure(c) for c in contents]

    def strict_format_reward(self, prompts, completions: List[Dict], **kwargs) -> List[float]:
        """
        出力が指定された形式になっているかどうかを厳しく評価する
        0~0.5
        """
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def soft_format_reward(self, prompts, completions: List[Dict], **kwargs) -> List[float]:
        """
        出力が指定された形式になっているかどうかをゆるく評価する
        0~0.5
        """
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

    def soft_shogi_format_reward(self, prompts, completions: List[Dict], **kwargs) -> List[float]:
        """
        将棋の指し手の形式になっているかどうかをゆるく評価する
        0~1
        """
        pattern = r"^[１-９7-9][一二三四五六七八九123456789][歩香桂銀金角飛玉と馬龍]$"
        rewards = []
        for completion in completions:
            try:
                response = self._extract_xml_answer(completion[0]["content"])
                reward = 1.0 if re.match(pattern, response) else 0.0
            except (TypeError, AttributeError):
                reward = 0.0
            rewards.append(reward)
        return rewards

    def strict_shogi_reward(self, prompts, completions: List[Dict], answer: List[str], **kwargs) -> List[float]:
        """
        将棋の指し手が正しいかどうかを厳しく評価する
        0~1
        """
        # 指し手のリストを取得
        responses = [self._extract_xml_answer(completion[0]["content"]) for completion in completions]
        rewards = []
        
        for response, sfen in zip(responses, answer):
            # Check basic format first
            try:
                if (re.match(self._number_pattern, response) and
                    re.search(self._position_pattern, response) and
                    re.search(self._piece_pattern, response)):
                    reward = 0.5
                    # responseをUSI形式に変換
                    move = move_to_usi(response)
                    # Validate move with engine
                    if self.engine.is_legal_move(sfen, move):
                        reward = 1.0
                else:
                    reward = 0.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
        
        return rewards

    def evaluation_reward(self, prompts, completions: List[Dict], answer: List[str], **kwargs) -> List[float]:
        """
        将棋の指し手の評価値を取得する
        0~
        """
        rewards = []
        
        for completion, sfen in zip(completions, answer):
            try:
                # Extract move from completion
                response = self._extract_move(completion[0]["content"])
                print("指し手: ", response)
                move = move_to_usi(response)
                # sfenの局面でmoveの評価値を取得
                score = self.engine.get_move_score(move, sfen)
                # Normalize score
                reward = (score + self.normalization_factor) / (2 * self.normalization_factor)
                rewards.append(reward)
            except Exception:
                # Invalid moves get minimum reward
                rewards.append(0.0)
        return rewards
    
    def _validate_xml_structure(self, text: str) -> float:
        """Count and score XML tags in text.
        Returns:
            float: Score from 0.0 to 0.5 based on XML structure
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
        return max(0.0, count) # 最大で0.5まで
    
    def _extract_move(self, content: str) -> Optional[str]:
        """Extract move from completion content.
        
        Args:
            content: Model completion content
            
        Returns:
            Optional[str]: Extracted move in USI format, or None if invalid
        """
        try:
            # First try to extract from XML format
            move = self._extract_xml_answer(content)
            if move:
                return move_to_usi(move)
            
            # Fallback to direct content parsing
            move = content.strip()
            if len(move) >= 3:  # Basic validation for Japanese notation
                return move_to_usi(move)
            return None
        except Exception:
            return None
            
    def _extract_xml_answer(self, content: str) -> Optional[str]:
        """
        <answer>タグ内の指し手を抽出する
        """
        try:
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
            return None
        except Exception:
            return None

    def __del__(self):
        """Close engine when object is destroyed."""
        if hasattr(self, 'engine'):
            self.engine.close()
