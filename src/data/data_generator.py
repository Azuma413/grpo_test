import random
import json
import csv
import os
from typing import List, Dict, Any
from ..shogi import YaneuraOuEngine, sfen_to_markdown

class ShogiDataGenerator:
    """Generate training data for shogi position learning."""
    
    def __init__(self, engine: YaneuraOuEngine, evaluation_threshold: float = 1000.0):
        """Initialize the data generator.
        
        Args:
            engine: Initialized YaneuraOu engine instance.
            evaluation_threshold: Position evaluation threshold for game termination.
                Games end when abs(evaluation) > threshold.
        """
        self.engine = engine
        self.evaluation_threshold = evaluation_threshold

    def generate_data(self, num_games: int = 100) -> List[List[Dict[str, Any]]]:
        """Generate training data from multiple games.
        
        Args:
            num_games: Number of games to generate.
            
        Returns:
            List[List[Dict]]: List of games, where each game is a list of positions.
                Each position is a dictionary containing:
                - sfen: Position in SFEN format
                - hands: Captured pieces
                - move_number: Move number in the game
        """
        all_positions = []
        for _ in range(num_games):
            game_positions = self._generate_game_positions()
            all_positions.append(game_positions)
            print(f"Generated game with {len(game_positions)} positions")
        
        total_positions = sum(len(game_pos) for game_pos in all_positions)
        print(f"Generated total of {total_positions} positions from {num_games} games")
        
        return all_positions

    def _generate_game_positions(self) -> List[Dict[str, Any]]:
        """Generate all positions from a single game.
        
        Returns:
            List[Dict]: List of positions from the game.
        """
        positions = []
        current_position = "startpos"
        move_number = 0
        
        while not self._is_game_over(current_position):
            # Save current position
            if current_position == "startpos":
                sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"
                hands = "なし"
            else:
                sfen = current_position.replace("position sfen ", "") if current_position.startswith("position sfen ") else current_position
                hands = sfen.split()[-2] if len(sfen.split()) >= 4 else "-"
                hands = "なし" if hands == "-" else hands
            
            positions.append({
                "sfen": sfen,
                "hands": hands,
                "move_number": move_number
            })
            
            # Get next move
            legal_moves = self._get_legal_moves(current_position)
            if not legal_moves:
                break
                
            next_move = random.choice(legal_moves)
            if current_position == "startpos":
                current_position = f"position startpos moves {next_move}"
            else:
                current_position = f"{current_position} {next_move}"
            
            move_number += 1
        
        return positions

    def _is_game_over(self, position: str) -> bool:
        """Check if the game should end.
        
        Args:
            position: Current position in SFEN format.
            
        Returns:
            bool: True if game should end, False to continue.
        """
        legal_moves = self._get_legal_moves(position)
        if not legal_moves:
            return True
            
        # Check position evaluation
        self.engine.set_position(position)
        self.engine._send_command("go depth 10")
        
        while True:
            line = self.engine.process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
            if line.startswith("info score cp"):
                try:
                    score = float(line.split()[3])
                    if abs(score) > self.evaluation_threshold:
                        return True
                except (IndexError, ValueError):
                    pass
        
        return False

    def _get_legal_moves(self, position: str) -> List[str]:
        """Get list of legal moves for a position.
        
        Args:
            position: Position in SFEN format.
            
        Returns:
            List[str]: List of legal moves in USI format.
        """
        self.engine.set_position(position)
        return self.engine.get_legal_moves()

    def save_positions_csv(self, positions: List[List[Dict]], filepath: str = "datasets/positions.csv"):
        """Save generated positions to CSV file.
        
        Args:
            positions: List of games with their positions.
            filepath: Output CSV file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['sfen', 'hands', 'game_id', 'move_number'])
            writer.writeheader()
            for game_id, game_positions in enumerate(positions):
                for pos in game_positions:
                    writer.writerow({
                        'sfen': pos['sfen'],
                        'hands': pos['hands'],
                        'game_id': game_id,
                        'move_number': pos['move_number']
                    })

    def convert_to_jsonl(self, csv_path: str = "datasets/positions.csv", 
                        output_path: str = "datasets/training_data.jsonl"):
        """Convert CSV positions to JSONL training data.
        
        Args:
            csv_path: Input CSV file with positions.
            output_path: Output JSONL file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile, \
             open(output_path, 'w', encoding='utf-8') as jsonlfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                markdown_board = sfen_to_markdown(row['sfen'], row['hands'])
                entry = {
                    "prompt": f"次の局面で指すべき手を考えてください。\n\n{markdown_board}",
                    "response": ""
                }
                json.dump(entry, jsonlfile, ensure_ascii=False)
                jsonlfile.write('\n')
