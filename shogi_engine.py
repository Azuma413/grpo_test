import subprocess
import time

class YaneuraOuEngine:
    def __init__(self, engine_path: str = "/mnt/e/SourceCode/app/yaneuraou/YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe"):
        """
        やねうら王エンジンの初期化
        Args:
            engine_path: やねうら王実行ファイルへのパス
        """
        self.engine_path = engine_path
        self.process = None
        self.position = "startpos"

    def start(self):
        """エンジンを起動し、USIモードで初期化"""
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # USIコマンドを送信
            self._send_command("usi")
            # "usiok"を待つ
            self._wait_for_response("usiok")
            
            # エンジンの準備
            self._send_command("isready")
            # "readyok"を待つ
            self._wait_for_response("readyok")
            
            # 新規ゲーム
            self._send_command("usinewgame")
            
            return True
        except Exception as e:
            print(f"Engine start error: {e}")
            return False

    def _send_command(self, command: str):
        """
        エンジンにコマンドを送信
        Args:
            command: USIコマンド
        """
        if self.process and self.process.poll() is None:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()

    def _wait_for_response(self, expected: str, timeout: int = 10) -> bool:
        """
        期待する応答を待つ
        Args:
            expected: 期待する文字列
            timeout: タイムアウト秒数
        Returns:
            bool: 期待する応答があったかどうか
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.stdout.readline().strip() == expected:
                return True
        return False

    def set_position(self, sfen: str):
        """
        局面をセット
        Args:
            sfen: SFEN形式の局面
        """
        self.position = sfen
        self._send_command(f"position {sfen}")

    def get_position_evaluation(self, position: str, move: str) -> float:
        """
        指定された局面で指し手を指した後の評価値を取得
        Args:
            position: SFEN形式の局面
            move: USI形式の指し手
        Returns:
            float: 評価値
        """
        self.set_position(position)
        self._send_command(f"go mate {move}")
        
        evaluation = 0.0
        start_time = time.time()
        while time.time() - start_time < 5:  # 5秒タイムアウト
            line = self.process.stdout.readline().strip()
            if line.startswith("info score cp"):
                evaluation = float(line.split()[3])
                break
            elif line.startswith("info score mate"):
                # 詰みの場合は大きな値を返す
                evaluation = float('inf') if int(line.split()[3]) > 0 else float('-inf')
                break
        
        return evaluation

    def is_legal_move(self, position: str, move: str) -> bool:
        """
        指し手が合法手かどうかを判定
        Args:
            position: SFEN形式の局面
            move: USI形式の指し手
        Returns:
            bool: 合法手であればTrue
        """
        self.set_position(position)
        self._send_command("go movelist")
        
        # 合法手リストを取得
        legal_moves = []
        start_time = time.time()
        while time.time() - start_time < 2:  # 2秒タイムアウト
            line = self.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
            if line.startswith("info string"):
                moves = line.split()[2:]
                legal_moves.extend(moves)
        
        return move in legal_moves

    def close(self):
        """エンジンを終了"""
        if self.process:
            self._send_command("quit")
            self.process.wait()
