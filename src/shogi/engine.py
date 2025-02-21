import subprocess
import time
from typing import List, Optional, Dict
import fcntl
import os
from threading import Thread
from queue import Queue, Empty
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler

# ロガーの設定
logger = logging.getLogger('YaneuraOu')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('engine.log', maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class MessageQueue:
    def __init__(self):
        self.queue = Queue()
        self._last_error = None
        self._processed_messages = []  # 処理済みメッセージを保持
    
    def wait_for_type(self, msg_type: str, timeout: float) -> Optional[Dict]:
        """タイムアウトをより厳密に管理し、エラー処理を強化"""
        start_time = time.time()
        messages = []  # 処理中に受け取ったメッセージを保持
        
        while time.time() - start_time < timeout:
            try:
                msg = self.queue.get(timeout=min(0.1, timeout/5))  # タイムアウト時間を調整
                messages.append(msg)
                
                if msg['type'] == msg_type:
                    self._processed_messages.extend(messages)  # 処理済みメッセージを保存
                    return msg
                if msg['type'] == 'error':
                    self._last_error = msg['content']
                    logger.error(f"Error received: {msg['content']}")
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in wait_for_type: {e}")
                self._last_error = str(e)
        
        # タイムアウト時のログ
        logger.warning(f"Timeout waiting for message type: {msg_type}")
        return None

    def put_message(self, message: Dict):
        """メッセージをキューに追加し、ログに記録"""
        self.queue.put(message)
        logger.debug(f"Message added to queue: {message}")

    def clear(self):
        """キューをクリアし、処理済みメッセージを保持"""
        cleared_messages = []
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                cleared_messages.append(msg)
            except Empty:
                break
        
        # 重要なメッセージは保持
        important_types = {'error', 'bestmove', 'position'}
        important_messages = [msg for msg in cleared_messages if msg['type'] in important_types]
        self._processed_messages.extend(important_messages)
        logger.debug(f"Queue cleared. Retained {len(important_messages)} important messages")

    def get_last_error(self) -> Optional[str]:
        """最後のエラーメッセージを取得"""
        return self._last_error

    def get_processed_messages(self) -> List[Dict]:
        """処理済みメッセージを取得"""
        return self._processed_messages.copy()

class OutputMonitor:
    def __init__(self, process: subprocess.Popen, message_queue: MessageQueue):
        self.process = process
        self.message_queue = message_queue
        self.running = False
        self.thread = None
        self._last_activity = time.time()
        self._error_count = 0
        self.MAX_ERRORS = 10
        self.ERROR_RESET_TIME = 60  # 60秒後にエラーカウントをリセット

    def start(self):
        """モニタリングスレッドを開始"""
        if self.thread and self.thread.is_alive():
            logger.warning("Monitor thread already running")
            return
            
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Output monitor started")

    def stop(self):
        """モニタリングスレッドを停止"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=5.0)  # 5秒でタイムアウト
                if self.thread.is_alive():
                    logger.warning("Monitor thread failed to stop gracefully")
            except Exception as e:
                logger.error(f"Error stopping monitor thread: {e}")

    def _monitor(self):
        """エンジンの出力を監視"""
        while self.running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    # プロセスが終了していないか確認
                    if self.process.poll() is not None:
                        logger.error("Engine process terminated unexpectedly")
                        break
                    continue

                self._last_activity = time.time()
                line_str = line.strip()
                
                # エラーカウントのリセットチェック
                if time.time() - self._last_activity > self.ERROR_RESET_TIME:
                    self._error_count = 0
                
                # メッセージをパースしてキューに格納
                try:
                    self._parse_and_queue_message(line_str)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    self._handle_error(e)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self._handle_error(e)
                
        logger.info("Monitor thread stopped")

    def _parse_and_queue_message(self, line_str: str):
        """メッセージをパースしてキューに追加"""
        message = None
        
        if line_str.startswith("info string Error!"):
            message = {'type': 'error', 'content': line_str}
            logger.error(f"Engine error: {line_str}")
        elif line_str.startswith("info"):
            message = {'type': 'info', 'content': line_str}
        elif line_str.startswith("bestmove"):
            message = {'type': 'bestmove', 'content': line_str}
            logger.info(f"Best move found: {line_str}")
        elif line_str.startswith("position"):
            message = {'type': 'position', 'content': line_str}
        elif line_str.startswith("moves"):
            message = {'type': 'info', 'content': line_str}
        elif line_str == "readyok":
            message = {'type': 'readyok', 'content': line_str}
        elif line_str == "usiok":
            message = {'type': 'usiok', 'content': line_str}
        elif line_str.startswith("sfen"):
            message = {'type': 'info', 'content': line_str}
        # Handle board output from 'd' command
        elif "sfen" in line_str and not line_str.startswith("info"):
            message = {'type': 'info', 'content': line_str}
            
        if message:
            self.message_queue.put_message(message)

    def _handle_error(self, error: Exception):
        """エラー処理とリカバリーロジック"""
        self._error_count += 1
        if self._error_count >= self.MAX_ERRORS:
            logger.critical(f"Too many errors ({self._error_count}), stopping monitor")
            self.running = False
        
        self.message_queue.put_message({
            'type': 'error',
            'content': str(error)
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
        self.default_timeout = 2.0  # タイムアウトを2秒に延長
        self._last_sync_time = time.time()
        self._sync_interval = 5.0  # 5秒ごとに同期チェック
        self._command_retries = 3  # コマンドの再試行回数
        
    def check_ready(self, timeout: float = 1.0) -> bool:
        """Check if engine is ready and wait for confirmation.
        
        Args:
            timeout: Maximum time to wait for readyok response.
            
        Returns:
            bool: True if engine confirmed ready, False otherwise.
        """
        # 定期的な同期チェック
        current_time = time.time()
        if current_time - self._last_sync_time > self._sync_interval:
            logger.info("Performing periodic sync check")
            self._last_sync_time = current_time
            
            # エンジンの状態を確認
            if not self.process or self.process.poll() is not None:
                logger.error("Engine process not available during sync check")
                return False
        
        # Clear any pending messages
        self.message_queue.clear()
        
        # Send isready command with retries
        for attempt in range(self._command_retries):
            if attempt > 0:
                logger.warning(f"Retrying isready command (attempt {attempt + 1})")
                time.sleep(0.1 * attempt)  # 徐々に待ち時間を増やす
                
            if self._send_command("isready"):
                # Wait for readyok response
                if self.message_queue.wait_for_type('readyok', timeout=timeout):
                    return True
                    
        logger.error("Failed to get readyok response after all retries")
        return False

    def _update_position_sfen(self, move: str) -> bool:
        """Update internal SFEN state after a move with improved error handling"""
        logger.info(f"Updating position after move: {move}")
        
        try:
            # Clear any pending messages
            self.message_queue.clear()
            
            # Create new position with the move
            new_moves = self._moves + [move]
            new_position = f"position startpos moves {' '.join(new_moves)}"
            
            # Send position and wait for sync with retries
            for attempt in range(self._command_retries):
                if attempt > 0:
                    logger.warning(f"Retrying position update (attempt {attempt + 1})")
                    time.sleep(0.1 * attempt)
                
                if not self._send_command(new_position):
                    continue
                    
                if not self.check_ready(timeout=self.default_timeout):
                    logger.error("Failed to sync after position update")
                    continue
                
                # Get updated SFEN from engine
                if not self._send_command("position"):
                    continue
                    
                msg = self.message_queue.wait_for_type('position', timeout=self.default_timeout)
                if not msg:
                    logger.error("Failed to get updated position")
                    continue
                    
                # Update internal state
                self._moves = new_moves
                self.position = new_position
                if 'sfen' in msg['content']:
                    sfen_parts = msg['content'].split("sfen ", 1)
                    if len(sfen_parts) > 1:
                        self._position_sfen = sfen_parts[1].strip()
                        logger.info(f"Successfully updated position to: {self._position_sfen}")
                        return True
                        
            logger.error("Failed to update position after all retries")
            return False
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
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
            if not self.check_ready():
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
        
        return self.check_ready(timeout)

    def _send_command(self, command: str) -> bool:
        """Send command to engine with error handling and logging"""
        if not self.process:
            logger.error("No engine process available")
            return False
            
        if self.process.poll() is not None:
            logger.error("Engine process has terminated")
            return False
            
        try:
            logger.debug(f"Sending command: {command}")
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
            return True
        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            return False

    def get_current_sfen(self, sfen: str, timeout: float = 1.0) -> str:
        """Get current position in SFEN format and update engine state"""
        logger.info(f"Getting current SFEN for position: {sfen}")
        
        # Clear any pending messages
        self.message_queue.clear()
        
        # Update internal position state
        self.position = sfen if sfen.startswith("position ") else f"position {sfen}"
        
        # Send position command and wait for synchronization with retries
        for attempt in range(self._command_retries):
            if attempt > 0:
                logger.warning(f"Retrying position update (attempt {attempt + 1})")
                time.sleep(0.1 * attempt)
                
            if not self._send_command(self.position):
                continue
                
            # Request SFEN using 'd' command
            if self._send_command("d"):
                # Wait for all lines of output from 'd' command
                sfen_found = False
                start_time = time.time()
                while time.time() - start_time < timeout:
                    msg = self.message_queue.wait_for_type('info', timeout=0.1)
                    if not msg:
                        continue
                        
                    content = msg['content']
                    if 'sfen' in content:
                        sfen_parts = content.split("sfen ", 1)
                        if len(sfen_parts) > 1:
                            self._position_sfen = sfen_parts[1].strip()
                            logger.info(f"Found SFEN: {self._position_sfen}")
                            sfen_found = True
                            break
                
                if sfen_found:
                    return self._position_sfen
                            
        logger.error("Failed to get current SFEN after all retries")
        return self._position_sfen

    def get_best_move(self, timeout: float = 2.0) -> str:
        """Get the best move for the current position with improved error handling and retries.
        
        Args:
            timeout: Maximum time to wait for engine response.
            
        Returns:
            str: Best move in USI format, or "none" if no legal moves exist.
        """
        logger.info("Getting best move")
        
        if not self.process or self.process.poll() is not None:
            logger.error("Engine process not available")
            return "none"
        
        # 盤面情報を更新
        # self._send_command(f"position sfen {self.position}")
            
        # 探索開始
        self.message_queue.clear()
        logger.info(f"Starting search with movetime {self.think_time_ms}ms")
        if not self._send_command(f"go movetime {self.think_time_ms}"):
            return "none"
        
        # bestmoveを待つ
        bestmove_msg = self.message_queue.wait_for_type('bestmove', timeout=max(timeout, self.think_time_ms/1000 + 0.5))
        if not bestmove_msg:
            logger.warning("No bestmove received within timeout")
            return "none"
        
        # 指し手をパース
        try:
            move = bestmove_msg['content'].split()[1]
            logger.info(f"Found move: {move}")
            
            # 最終同期チェック
            if self.check_ready(timeout=timeout):
                if self._send_command(self.position) and self.check_ready(timeout=timeout):
                    logger.info(f"Successfully validated move: {move}")
                    return move
        except (IndexError, KeyError) as e:
            logger.error(f"Failed to parse bestmove response: {e}")
            return "none"

    def get_position_evaluation(self, timeout: float = 1.0) -> float:
        """局面の評価値を取得（改善版）"""
        logger.info("Getting position evaluation")
        
        if not self.check_ready(timeout=timeout):
            logger.error("Engine not ready for evaluation")
            return 0.0
            
        if not self._send_command(f"go movetime {self.think_time_ms}"):
            logger.error("Failed to send evaluation command")
            return 0.0
        
        evaluation = 0.0
        best_evaluation = None
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.message_queue.wait_for_type('info', timeout=0.1)
            if not msg:
                continue
                
            line = msg['content']
            if "score cp" in line or "score mate" in line:
                try:
                    parts = line.split()
                    score_idx = parts.index("score")
                    if parts[score_idx + 1] == "cp":
                        current_eval = float(parts[score_idx + 2])
                        # より深い探索の評価値を優先
                        if "depth" in line and (best_evaluation is None or "depth" not in line):
                            best_evaluation = current_eval
                            evaluation = current_eval
                    elif parts[score_idx + 1] == "mate":
                        mate_in = int(parts[score_idx + 2])
                        evaluation = float('inf') if mate_in > 0 else float('-inf')
                        best_evaluation = evaluation  # メイト発見時は即座に確定
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing evaluation: {e}")
                    continue
            
            if self.message_queue.wait_for_type('bestmove', timeout=0.1):
                break
        
        logger.info(f"Final evaluation: {evaluation}")
        return evaluation

    def close(self):
        """エンジンを安全に終了"""
        logger.info("Closing engine")
        try:
            if self.output_monitor:
                self.output_monitor.stop()
                logger.info("Output monitor stopped")
                
            if self.process:
                # 終了コマンドを送信
                if self._send_command("quit"):
                    # 正常終了を待機
                    try:
                        exit_code = self.process.wait(timeout=5.0)
                        logger.info(f"Engine process terminated with exit code: {exit_code}")
                    except subprocess.TimeoutExpired:
                        logger.warning("Engine process did not terminate gracefully, forcing termination")
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            logger.error("Failed to terminate engine process, killing")
                            self.process.kill()
                            self.process.wait()
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
            # 最後の手段としてプロセスを強制終了
            if self.process:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception as e2:
                    logger.critical(f"Failed to kill engine process: {e2}")
