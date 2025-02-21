import subprocess
import time
from typing import List, Optional, Dict
import fcntl
import os
from threading import Thread
from queue import Queue, Empty
from datetime import datetime

class MessageQueue:
    def __init__(self):
        self.queue = Queue()
        self._last_error = None
    
    def wait_for_type(self, msg_type: str, timeout: float) -> Optional[Dict]:
        """タイムアウトをより厳密に管理"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msg = self.queue.get(timeout=min(0.05, timeout/10))
                if msg['type'] == msg_type:
                    return msg
                if msg['type'] == 'error':
                    self._last_error = msg['content']
            except Empty:
                continue
        return None

    def put_message(self, message: Dict):
        """メッセージをキューに追加"""
        self.queue.put(message)

    def clear(self):
        """キューをクリア"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def get_last_error(self) -> Optional[str]:
        """最後のエラーメッセージを取得"""
        return self._last_error

class OutputMonitor:
    def __init__(self, process: subprocess.Popen, message_queue: MessageQueue):
        self.process = process
        self.message_queue = message_queue
        self.log_file = open('log.txt', 'w', encoding='utf-8')
        self.running = False
        self.thread = None

    def start(self):
        """モニタリングスレッドを開始"""
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.daemon = True  # メインスレッド終了時に自動終了
        self.thread.start()

    def stop(self):
        """モニタリングスレッドを停止"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.log_file:
            self.log_file.close()

    def _monitor(self):
        """エンジンの出力を監視"""
        while self.running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    continue
                    
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                line_str = line.strip()
                self.log_file.write(f"[{timestamp}] {line_str}\n")
                self.log_file.flush()
                
                # メッセージをパースしてキューに格納
                if line_str.startswith("info string Error!"):
                    self.message_queue.put_message({
                        'type': 'error',
                        'content': line_str
                    })
                elif line_str.startswith("info"):
                    self.message_queue.put_message({
                        'type': 'info',
                        'content': line_str
                    })
                elif line_str.startswith("bestmove"):
                    self.message_queue.put_message({
                        'type': 'bestmove',
                        'content': line_str
                    })
                elif line_str.startswith("position"):
                    self.message_queue.put_message({
                        'type': 'position',
                        'content': line_str
                    })
                elif line_str.startswith("moves"):
                    self.message_queue.put_message({
                        'type': 'info',
                        'content': line_str
                    })
                elif line_str == "readyok":
                    self.message_queue.put_message({
                        'type': 'readyok',
                        'content': line_str
                    })
                elif line_str == "usiok":
                    self.message_queue.put_message({
                        'type': 'usiok',
                        'content': line_str
                    })
            except Exception as e:
                self.message_queue.put_message({
                    'type': 'error',
                    'content': str(e)
                })

class YaneuraOuEngine:
    """Interface for the YaneuraOu shogi engine."""
    
    def __init__(self, engine_path: str = "/mnt/e/SourceCode/app/yaneuraou/YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe", think_time_ms: int = 1000):
        self.engine_path = engine_path
        self.think_time_ms = think_time_ms
        self.process: Optional[subprocess.Popen] = None
        self.position = "startpos"
        self._position_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        self._moves: List[str] = []
        self.message_queue = MessageQueue()
        self.output_monitor = None
        self.default_timeout = 0.5  # 500ミリ秒
        
    def _update_position_sfen(self, move: str) -> bool:
        """Update internal SFEN state after a move"""
        try:
            # Clear any pending messages
            self.message_queue.clear()
            
            # Add move to list and update position
            self._moves.append(move)
            new_position = f"position startpos moves {' '.join(self._moves)}"
            self.position = new_position
            
            # Send new position to engine and wait for sync
            print(f"[Engine] Updating position after move {move}")
            self._send_command(new_position)
            self._send_command("isready")
            if not self.message_queue.wait_for_type('readyok', timeout=1.0):
                print("[Engine] Failed to sync after position update")
                return False
            
            # Basic SFEN position adjustment (not complete, but tracks color changes)
            if 'b' in self._position_sfen:
                self._position_sfen = self._position_sfen.replace('b', 'w')
            else:
                self._position_sfen = self._position_sfen.replace('w', 'b')
                
            return True
        except Exception as e:
            print(f"[Engine] Error updating position: {e}")
            return False

    def start(self) -> bool:
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            # Set non-blocking mode for stdout
            fd = self.process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            # 出力モニターを開始
            self.output_monitor = OutputMonitor(self.process, self.message_queue)
            self.output_monitor.start()

            # Get engine directory
            engine_dir = "/".join(self.engine_path.split("/")[:-1])
            if engine_dir.startswith("/mnt/"):
                drive = engine_dir.split("/")[2]
                windows_dir = engine_dir.replace(f"/mnt/{drive}", f"{drive.upper()}:")
                
                # Set eval directory
                eval_dir = windows_dir + "\\eval"
                print(f"Setting eval directory to: {eval_dir}")
                self._send_command(f"setoption name EvalDir value {eval_dir}")
                
                # Set book directory
                book_dir = windows_dir + "\\book"
                print(f"Setting book directory to: {book_dir}")
                self._send_command(f"setoption name BookDir value {book_dir}")

            # Initialize USI protocol first
            if not self._initialize_usi():
                return False

            # Configure engine options
            self._send_command("setoption name USI_Hash value 4096")
            self._send_command("setoption name MinimumThinkingTime value 100")
            
            # Wait for engine to process options
            self._send_command("isready")
            if not self.message_queue.wait_for_type('readyok', timeout=1.0):
                print("Engine not responding to isready")
                return False
            self._send_command("usinewgame")
            return True
            
        except Exception as e:
            print(f"Engine start error: {e}")
            return False

    def _initialize_usi(self, timeout: float = 1.0) -> bool:
        self._send_command("usi")
        if not self.message_queue.wait_for_type("usiok", timeout):
            return False
        
        self._send_command("isready")
        if not self.message_queue.wait_for_type("readyok", timeout):
            return False
        return True

    def _send_command(self, command: str) -> bool:
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(f"{command}\n")
                self.process.stdin.flush()
                return True
            except Exception as e:
                print(f"Error sending command: {e}")
                return False
        return False

    def get_current_sfen(self, sfen: str, timeout: float = 0.5) -> str:
        """Get current position in SFEN format and update engine state"""
        # Clear any pending messages
        self.message_queue.clear()
        
        # Update internal position state
        self.position = sfen if sfen.startswith("position ") else f"position {sfen}"
        
        # Send position command and wait for synchronization
        self._send_command(self.position)
        self._send_command("isready")
        if not self.message_queue.wait_for_type('readyok', timeout=0.2):
            return self._position_sfen
        
        # Request SFEN directly
        self._send_command("position")
        msg = self.message_queue.wait_for_type('position', timeout=0.2)
        
        if msg and 'sfen' in msg['content']:
            sfen_parts = msg['content'].split("sfen ", 1)
            if len(sfen_parts) > 1:
                self._position_sfen = sfen_parts[1].strip()
        
        return self._position_sfen

    def get_best_move(self, timeout: float = 1.0) -> str:
        """Get the best move for the current position.
        
        Args:
            timeout: Maximum time to wait for engine response.
            
        Returns:
            str: Best move in USI format, or "none" if no legal moves exist.
        """
        if not self.process or self.process.poll() is not None:
            print("[Engine] Process not available")
            return "none"

        # Make sure we're in a valid state and clear any pending messages
        self.message_queue.clear()
        print("[Engine] Ensuring engine is ready...")
        self._send_command("isready")
        
        # Give engine more time to get ready
        if not self.message_queue.wait_for_type('readyok', timeout=1.0):
            print("[Engine] Not ready before search")
            # Try one more time
            self._send_command("isready")
            if not self.message_queue.wait_for_type('readyok', timeout=1.0):
                print("[Engine] Engine failed to respond twice")
                return "none"
        
        # Confirm current position is set
        print(f"[Engine] Current position: {self.position}")
        self._send_command(self.position)
        self._send_command("isready")
        if not self.message_queue.wait_for_type('readyok', timeout=0.5):
            print("[Engine] Failed to confirm position")
            return "none"

        # Clear messages again before starting search
        self.message_queue.clear()
        print(f"[Engine] Starting search with movetime {self.think_time_ms}ms")
        self._send_command(f"go movetime {self.think_time_ms}")
        
        # Wait for bestmove
        bestmove_msg = self.message_queue.wait_for_type('bestmove', timeout=timeout)
        if not bestmove_msg:
            print("[Engine] No bestmove received within timeout")
            return "none"
        
        # Parse the best move
        try:
            move = bestmove_msg['content'].split()[1]
            print(f"[Engine] Found move: {move}")
        except (IndexError, KeyError):
            print("[Engine] Failed to parse bestmove response")
            return "none"
        
        # Force resync after getting move
        for _ in range(3):  # Try up to 3 times
            print("[Engine] Resyncing after move...")
            self.message_queue.clear()
            self._send_command("isready")
            if self.message_queue.wait_for_type('readyok', timeout=1.0):
                # One final position check
                self._send_command(self.position)
                self._send_command("isready")
                if self.message_queue.wait_for_type('readyok', timeout=1.0):
                    print(f"[Engine] Successfully resynced after move: {move}")
                    return move
            time.sleep(0.1)  # Small delay between attempts
        
        print("[Engine] Failed to resync after move")
        return "none"

    def get_position_evaluation(self, timeout: float = 0.5) -> float:
        self._send_command(f"go movetime {self.think_time_ms}")
        
        evaluation = 0.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.message_queue.wait_for_type('info', timeout=0.1)
            if not msg:
                continue
                
            line = msg['content']
            if "score cp" in line or "score mate" in line:
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
            
            if self.message_queue.wait_for_type('bestmove', timeout=0.1):
                break
        
        return evaluation

    def close(self):
        if self.output_monitor:
            self.output_monitor.stop()
        if self.process:
            self._send_command("quit")
            self.process.wait()
