from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .engine_interface import EngineInterface
from utils.reward_functions import extract_xml_answer

@dataclass
class Position:
    """Represents a shogi board position."""
    markdown: str  # Markdown format board representation
    sfen: str     # SFEN format
    
    @classmethod
    def initial(cls) -> 'Position':
        """Create initial game position."""
        from shogi_utils import create_initial_board, markdown_to_sfen
        markdown = create_initial_board()
        sfen = markdown_to_sfen(markdown)
        return cls(markdown=markdown, sfen=sfen)

class ShogiGame:
    """Manages a shogi game session."""
    
    def __init__(
        self,
        engine: EngineInterface,
        max_moves: int = 100,
        mate_time_limit: int = 3000,
        cpu_think_time: int = 5000
    ):
        self.engine = engine
        self.max_moves = max_moves
        self.mate_time_limit = mate_time_limit
        self.cpu_think_time = cpu_think_time
        self.reset()
        
    def reset(self):
        """Reset the game to initial position."""
        self.position = Position.initial()
        self.moves: List[str] = []
        self.repetition_count: Dict[str, int] = {self.position.markdown: 1}
        
    def _update_position(self, move: str, is_cpu: bool = False) -> None:
        """Update the game position after a move.
        
        Args:
            move: Move in USI format.
            is_cpu: Whether the move is from the CPU.
        """
        from shogi_utils import sfen_to_markdown
        
        # Update SFEN
        self.position.sfen = f"{self.position.sfen} {move}"
        
        # Update markdown representation
        self.position.markdown = sfen_to_markdown(self.position.sfen)
        
        # Update move history and repetition count
        self.moves.append(move)
        self.repetition_count[self.position.markdown] = (
            self.repetition_count.get(self.position.markdown, 0) + 1
        )
        
    def make_move(self, move: str) -> Tuple[bool, float, str]:
        """Make a move and return game status.
        
        Args:
            move: Move in USI or natural language format.
            
        Returns:
            Tuple[bool, float, str]: (is_game_over, reward, reason)
        """
        from shogi_utils import move_to_usi
        
        # Convert to USI format if needed
        if not move.islower():  # Heuristic for natural language move
            usi = move_to_usi(move)
        else:
            usi = move
            
        # Update position
        self._update_position(usi)
        
        # Check for repetition
        if self.repetition_count[self.position.markdown] >= 4:
            return True, 0.0, "千日手により引き分け"
            
        # Check for maximum moves
        if len(self.moves) > self.max_moves:
            return True, 0.0, "長手数による引き分け"
            
        # Check for mate
        self.engine.set_position(self.position.sfen)
        self.engine.go_mate(self.mate_time_limit)
        if self.engine.get_mate_result():
            return True, 1.0, "詰みを発見"
            
        return False, 0.0, ""
        
    def get_cpu_move(self) -> Optional[Tuple[str, bool, float, str]]:
        """Get move from CPU engine.
        
        Returns:
            Optional[Tuple[str, bool, float, str]]: (move, is_game_over, reward, reason)
            Returns None if CPU resigns.
        """
        # Get CPU move
        self.engine.set_position(self.position.sfen)
        self.engine.go_bestmove(self.cpu_think_time)
        move = self.engine.get_bestmove()
        
        if not move:  # CPU resigns
            return None
            
        # Update position with CPU move
        self._update_position(move, is_cpu=True)
        
        # Check for repetition
        if self.repetition_count[self.position.markdown] >= 4:
            return move, True, 0.0, "千日手により引き分け"
            
        # Check for maximum moves
        if len(self.moves) > self.max_moves:
            return move, True, 0.0, "長手数による引き分け"
            
        # Check for mate
        self.engine.set_position(self.position.sfen)
        self.engine.go_mate(self.mate_time_limit)
        if self.engine.get_mate_result():
            return move, True, -1.0, "CPU勝利（詰み）"
            
        return move, False, 0.0, ""

def process_shogi_board(markdown_board: str, move: str) -> Tuple[str, str]:
    """Convert board and move to SFEN/USI format.
    
    Args:
        markdown_board: Board in markdown format.
        move: Move in natural language format.
        
    Returns:
        Tuple[str, str]: (SFEN format board, USI format move)
    """
    from shogi_utils import markdown_to_sfen, move_to_usi
    sfen = markdown_to_sfen(markdown_board)
    usi = move_to_usi(move)
    return sfen, usi

def create_initial_board() -> str:
    """Create initial board position in markdown format."""
    from shogi_utils import create_initial_board as create_board
    return create_board()
