from abc import ABC, abstractmethod
import time
from typing import Optional

class EngineInterface(ABC):
    """Abstract interface for chess engines."""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the engine process.
        
        Returns:
            bool: True if engine started successfully.
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the engine process."""
        pass
    
    @abstractmethod
    def set_position(self, sfen: str):
        """Set the current position.
        
        Args:
            sfen: Position in SFEN format.
        """
        pass
    
    @abstractmethod
    def go_mate(self, time_ms: int) -> None:
        """Search for mate.
        
        Args:
            time_ms: Time limit in milliseconds.
        """
        pass
    
    @abstractmethod
    def go_bestmove(self, time_ms: int) -> None:
        """Search for best move.
        
        Args:
            time_ms: Time limit in milliseconds.
        """
        pass
    
    @abstractmethod
    def get_bestmove(self) -> Optional[str]:
        """Get the best move found by the engine.
        
        Returns:
            str: Best move in USI format, or None if no move found.
        """
        pass
    
    @abstractmethod
    def get_mate_result(self) -> bool:
        """Get whether a mate was found.
        
        Returns:
            bool: True if mate was found.
        """
        pass
    
    @abstractmethod
    def sfen_to_markdown(self, sfen: str) -> str:
        """Convert SFEN position to markdown board representation.
        
        Args:
            sfen: Position in SFEN format.
            
        Returns:
            str: Board in markdown format.
        """
        pass

class YaneuraOuEngineWrapper(EngineInterface):
    """Wrapper for YaneuraOu shogi engine."""
    
    def __init__(self, process=None):
        self.process = process
        
    def start(self) -> bool:
        """Start YaneuraOu engine process.
        
        Returns:
            bool: True if engine started successfully.
        """
        # The actual engine process should be injected in the constructor
        return self.process is not None
        
    def close(self):
        """Close the engine process."""
        if self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
            finally:
                self.process = None
                
    def set_position(self, sfen: str):
        """Set the current position.
        
        Args:
            sfen: Position in SFEN format.
        """
        if "moves" in sfen:
            position, moves = sfen.split(" moves ")
            self._send_command(f"position sfen {position} moves {moves}")
        else:
            self._send_command(f"position sfen {sfen}")
            
    def go_mate(self, time_ms: int):
        """Search for mate.
        
        Args:
            time_ms: Time limit in milliseconds.
        """
        self._send_command(f"go mate {time_ms}")
        
    def go_bestmove(self, time_ms: int):
        """Search for best move.
        
        Args:
            time_ms: Time limit in milliseconds.
        """
        self._send_command(f"go bestmove byoyomi {time_ms}")
        
    def get_bestmove(self) -> Optional[str]:
        """Get the best move found by the engine.
        
        Returns:
            str: Best move in USI format, or None if no move found.
        """
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                move = line.split()[1]
                return None if move == "resign" else move
        return None
        
    def get_mate_result(self) -> bool:
        """Get whether a mate was found.
        
        Returns:
            bool: True if mate was found.
        """
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            line = self.process.stdout.readline().strip()
            if "checkmate" in line:
                return True
        return False
        
    def sfen_to_markdown(self, sfen: str) -> str:
        """Convert SFEN position to markdown board representation.
        
        Args:
            sfen: Position in SFEN format.
            
        Returns:
            str: Board in markdown format.
        """
        from shogi_utils import sfen_to_markdown
        return sfen_to_markdown(sfen)
        
    def _send_command(self, cmd: str):
        """Send a command to the engine.
        
        Args:
            cmd: Command string.
        """
        if self.process and self.process.poll() is None:
            self.process.stdin.write(f"{cmd}\n".encode('utf-8'))
            self.process.stdin.flush()
