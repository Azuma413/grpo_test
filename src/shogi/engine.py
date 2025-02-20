import subprocess
import time
from typing import List, Optional

class YaneuraOuEngine:
    """Interface for the YaneuraOu shogi engine."""
    
    def __init__(self, engine_path: str = "/mnt/e/SourceCode/app/yaneuraou/YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe", think_time_ms: int = 1000):
        """Initialize YaneuraOu engine interface.
        
        Args:
            engine_path: Path to YaneuraOu executable.
            think_time_ms: Thinking time limit per move in milliseconds.
        """
        self.engine_path = engine_path
        self.think_time_ms = think_time_ms
        self.process: Optional[subprocess.Popen] = None
        self.position = "startpos"

    def start(self) -> bool:
        """Start and initialize the engine in USI mode.
        
        Returns:
            bool: True if engine started successfully, False otherwise.
        """
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffering
            )
            # Set eval directory using Windows path
            current_dir = "/".join(self.engine_path.split("/")[:-1])
            if current_dir.startswith("/mnt/"):
                # Convert WSL path to Windows path
                drive = current_dir.split("/")[2]
                windows_path = current_dir.replace(f"/mnt/{drive}", f"{drive.upper()}:")
                windows_path = windows_path + "/eval"
                windows_path = windows_path.replace("/", "\\")
                self._send_command(f"setoption name EvalDir value {windows_path}")
            
            # Initialize USI mode
            if not self._initialize_usi():
                return False
            # Start new game
            self._send_command("usinewgame")
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
        # Send USI command and wait for usiok
        self._send_command("usi")
        # Consume all initial messages until usiok
        if not self._wait_for_response("usiok", timeout):
            return False
        # Give engine some time to initialize
        time.sleep(0.1)
        
        # Send isready command and wait for readyok
        self._send_command("isready")
        if not self._wait_for_response("readyok", timeout):
            return False
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
                if self.process.stdout.readline().strip() == expected:
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

    def get_position_evaluation(self, timeout: int = 5) -> float:
        """Get engine's evaluation for current position.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            float: Position evaluation in centipawns, or ±inf for mate.
        """
        self._send_command(f"go movetime {self.think_time_ms}")
        
        evaluation = 0.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
            elif line.startswith("info") and ("score cp" in line or "score mate" in line):
                # Extract score from the line
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    if parts[score_idx + 1] == "cp":
                        evaluation = float(parts[score_idx + 2])
                    elif parts[score_idx + 1] == "mate":
                        mate_in = int(parts[score_idx + 2])
                        evaluation = float('inf') if mate_in > 0 else float('-inf')
                except (ValueError, IndexError):
                    continue
        
        return evaluation

    def get_legal_moves(self, timeout: int = 2) -> List[str]:
        """Get list of legal moves for current position using multiple quick searches.
        
        Args:
            timeout: Maximum time to wait for engine response.
            
        Returns:
            List[str]: Legal moves in USI format.
        """
        legal_moves = []
        excluded_moves = set()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Construct go command excluding known moves
            exclude_str = " searchmoves " + " ".join(legal_moves) if legal_moves else ""
            self._send_command(f"go movetime {self.think_time_ms}{exclude_str}")
            
            found_new_move = False
            while True:
                line = self.process.stdout.readline().strip()
                if not line:
                    continue
                    
                if line.startswith("bestmove"):
                    move = line.split()[1]
                    if move != "none" and move not in excluded_moves:
                        legal_moves.append(move)
                        excluded_moves.add(move)
                        found_new_move = True
                    break
                elif line.startswith("info") and "pv" in line:
                    # Extract moves from principal variation
                    try:
                        pv_index = line.index("pv")
                        moves = line.split()[pv_index + 1:]
                        for move in moves:
                            if move not in excluded_moves:
                                legal_moves.append(move)
                                excluded_moves.add(move)
                                found_new_move = True
                    except (ValueError, IndexError):
                        continue
            
            if not found_new_move:
                break
                
        return legal_moves

    def is_legal_move(self, move: str) -> bool:
        """Check if a move is legal in the current position.
        
        Args:
            move: Move to check in USI format.
            
        Returns:
            bool: True if move is legal, False otherwise.
        """
        return move in self.get_legal_moves()

    def close(self):
        """Shutdown the engine."""
        if self.process:
            self._send_command("quit")
            self.process.wait()
