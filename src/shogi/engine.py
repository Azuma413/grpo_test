import subprocess
import time
from typing import List, Optional
import fcntl
import os

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
        self._position_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        self._move_count = 1

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
            # Set non-blocking mode for stdout
            fd = self.process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
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
        self.position = sfen if sfen.startswith("position ") else f"position {sfen}"
        self._send_command(self.position)
        # Wait for engine to process position
        time.sleep(0.1)
        # Send isready to ensure position is processed
        self._send_command("isready")
        self._wait_for_response("readyok", timeout=1)
        # Get current SFEN after position is set
        self._update_position_sfen()

    def _update_position_sfen(self) -> None:
        """Update the internal SFEN representation based on the current position."""
        # If position is startpos with moves, extract moves and calculate SFEN
        if self.position.startswith("position startpos"):
            if "moves" in self.position:
                moves = self.position.split("moves ")[1].split()
                self._move_count = len(moves) + 1
            else:
                self._move_count = 1
            # Keep initial SFEN for startpos
            self._position_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        elif "sfen" in self.position:
            # For direct SFEN input, use it as is
            self._position_sfen = self.position.split("sfen ")[1].split(" moves ")[0]
            if " moves " in self.position:
                moves = self.position.split(" moves ")[1].split()
                self._move_count = len(moves) + 1
            else:
                self._move_count = 1
        else:
            # Default to startpos if no valid format is found
            self._move_count = 1
            self._position_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    def get_current_sfen(self) -> str:
        """Get the SFEN representation of the current position.
        
        Returns:
            str: Current position in SFEN format.
        """
        return self._position_sfen

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

    def get_legal_moves(self, timeout: int = 1) -> List[str]:
        """Get list of legal moves for current position using the moves command.
        
        Args:
            timeout: Maximum time to wait for engine response in seconds.
            
        Returns:
            List[str]: Legal moves in USI format.
        """
        if not self.process or self.process.poll() is not None:
            return []

        print(f"Current position: {self.position}")  # Debug line

        # Send stop command to ensure no ongoing search
        self._send_command("stop")
        time.sleep(0.1)
        
        # Re-synchronize position state
        self._send_command(self.position)
        time.sleep(0.1)
        self._send_command("isready")
        if not self._wait_for_response("readyok", timeout=1):
            print("Warning: Engine did not respond with readyok")
            return []  # Return empty list if engine is not responding
        
        # Clear any pending output
        try:
            while self.process.stdout.readline().strip():
                continue
        except:
            pass
            
        legal_moves = set()
        start_time = time.time()
        
        # Send moves command to get all legal moves
        self._send_command("moves")
        while time.time() - start_time < timeout:
            try:
                line = self.process.stdout.readline().strip()
                if not line:
                    continue
                print("line: ", line)
                # Split the moves line into individual moves
                if not line.startswith("info") and not line.startswith("bestmove"):
                    moves = line.split()
                    for move in moves:
                        if move != "none":
                            legal_moves.add(move)
            except BlockingIOError:
                if legal_moves:  # If we have moves, we can return
                    break
                continue  # Otherwise keep trying to read
            except Exception as e:
                print(f"Error reading legal moves: {e}")
                break
        print("legal_moves: ", legal_moves)
        return list(legal_moves)

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
