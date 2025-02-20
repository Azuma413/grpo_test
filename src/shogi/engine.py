import subprocess
import time
from typing import List, Optional

class YaneuraOuEngine:
    """Interface for the YaneuraOu shogi engine."""
    
    def __init__(self, engine_path: str = "E:/SourceCode/app/yaneuraou/YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe"):
        """Initialize YaneuraOu engine interface.
        
        Args:
            engine_path: Path to YaneuraOu executable.
        """
        self.engine_path = engine_path
        self.process: Optional[subprocess.Popen] = None
        self.position = "startpos"

    def start(self) -> bool:
        """Start and initialize the engine in USI mode.
        
        Returns:
            bool: True if engine started successfully, False otherwise.
        """
        try:
            print(1)
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffering
            )
            print(2)
            
            # Initialize USI mode
            if not self._initialize_usi():
                print(3)
                return False
            print(4)
            # Start new game
            self._send_command("usinewgame")
            print(5)
            return True
            
        except Exception as e:
            print(f"Engine start error: {e}")
            return False

    def _initialize_usi(self, timeout: int = 10) -> bool:
        """Initialize the USI protocol.
        
        Args:
            timeout: Maximum time to wait for engine responses.
            
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        print("a")
        # Send USI command and wait for usiok
        self._send_command("usi")
        print("b")
        
        # Consume all initial messages until usiok
        if not self._wait_for_response("usiok", timeout):
            print("c")
            return False
        print("d")
        
        # Give engine some time to initialize
        time.sleep(0.1)
        
        # Send isready command and wait for readyok
        self._send_command("isready")
        print("e")
        if not self._wait_for_response("readyok", timeout):
            print("f")
            return False
        print("g")
        return True

    def _send_command(self, command: str):
        """Send a command to the engine.
        
        Args:
            command: USI command to send.
        """
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(f"{command}\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error sending command: {e}")

    def _wait_for_response(self, expected: str, timeout: int = 10) -> bool:
        """Wait for an expected response from the engine.
        
        Args:
            expected: Expected response string.
            timeout: Maximum time to wait in seconds.
            
        Returns:
            bool: True if expected response received, False if timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.process or self.process.poll() is not None:
                print("Engine process is not running")
                return False
                
            try:
                text = self.process.stdout.readline().strip()
                print(f"Engine response: {text}")
                if text == expected:
                    return True
            except Exception as e:
                print(f"Error reading response: {e}")
                return False
        print(f"Timeout waiting for {expected}")
        return False

    def set_position(self, sfen: str):
        """Set the current board position.
        
        Args:
            sfen: Position in SFEN format or special 'startpos' command.
        """
        self.position = sfen
        self._send_command(f"position {sfen}")

    def get_position_evaluation(self, position: str, move: str, search_depth: int = 10) -> float:
        """Get engine's evaluation after a move.
        
        Args:
            position: Current position in SFEN format.
            move: Move to evaluate in USI format.
            search_depth: Search depth for evaluation.
            
        Returns:
            float: Position evaluation in centipawns, or ±inf for mate.
        """
        self.set_position(position)
        self._send_command(f"go mate {move}")
        
        evaluation = 0.0
        start_time = time.time()
        while time.time() - start_time < 5:  # 5 second timeout
            line = self.process.stdout.readline().strip()
            if line.startswith("info score cp"):
                evaluation = float(line.split()[3])
                break
            elif line.startswith("info score mate"):
                # Return ±inf for mate positions
                evaluation = float('inf') if int(line.split()[3]) > 0 else float('-inf')
                break
        
        return evaluation

    def get_legal_moves(self, timeout: int = 2) -> List[str]:
        """Get list of legal moves for current position.
        
        Args:
            timeout: Maximum time to wait for move list.
            
        Returns:
            List[str]: Legal moves in USI format.
        """
        self._send_command("go movelist")
        
        legal_moves = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
            if line.startswith("info string"):
                moves = line.split()[2:]
                legal_moves.extend(moves)
        
        return legal_moves

    def is_legal_move(self, position: str, move: str) -> bool:
        """Check if a move is legal in the given position.
        
        Args:
            position: Position in SFEN format.
            move: Move to check in USI format.
            
        Returns:
            bool: True if move is legal, False otherwise.
        """
        self.set_position(position)
        legal_moves = self.get_legal_moves()
        return move in legal_moves

    def close(self):
        """Shutdown the engine."""
        if self.process:
            self._send_command("quit")
            self.process.wait()
