import random
import json
import csv
import os
from typing import List, Dict, Any
from ..shogi import YaneuraOuEngine, sfen_to_markdown

class ShogiDataGenerator:
    """Generate training data for shogi position learning."""
    
    def __init__(self, engine: YaneuraOuEngine):
        """Initialize the data generator.
        
        Args:
            engine: Initialized YaneuraOu engine instance.
        """
        self.engine = engine

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
        for game_num in range(num_games):
            game_positions = self._generate_game_positions()
            if game_positions:  # Only add if positions were generated
                all_positions.append(game_positions)
                print(f"Game {game_num + 1}: Generated {len(game_positions)} positions")
            else:
                print(f"Game {game_num + 1}: Failed to generate positions")
        
        total_positions = sum(len(game_pos) for game_pos in all_positions)
        print(f"\nGenerated total of {total_positions} positions from {len(all_positions)} successful games")
        
        return all_positions # これが良くない

    def _generate_game_positions(self) -> List[Dict[str, Any]]:
        """Generate all positions from a single game.
        
        Returns:
            List[Dict]: List of positions from the game.
        """
        positions = []
        moves = []
        move_number = 0
        
        # Set initial position
        initial_position = "position startpos"
        self.engine._send_command(initial_position)
        self.engine._send_command("isready")
        if not self.engine.message_queue.wait_for_type('readyok', timeout=1.0):
            print("Failed to set initial position")
            return []
        
        # Get initial SFEN and store position
        current_sfen = self.engine.get_current_sfen(initial_position)
        positions.append({
            "sfen": current_sfen,
            "hands": "なし",
            "move_number": move_number,
            "previous_move": ""
        })
        
        while True:
            next_move = self.engine.get_best_move(timeout=2.0) # 最善手を取得
            if next_move == "none": # 最善手がない場合は詰み
                print("Game over")
                break
            moves.append(next_move)
            next_position = f"position startpos moves {' '.join(moves)}"
            current_sfen = self.engine.get_current_sfen(next_position)
            if not current_sfen: # SFENの取得に失敗
                print("Failed to get updated SFEN")
                break
            # 持ち駒の取得
            hands = "なし"  # Default value
            if " w " in current_sfen:
                hands = current_sfen.split(" w ")[1].split(" ")[0]
            elif " b " in current_sfen:
                hands = current_sfen.split(" b ")[1].split(" ")[0]
            if hands == "-":
                hands = "なし"
            positions.append({
                "sfen": current_sfen,
                "hands": hands,
                "move_number": move_number,
                "previous_move": next_move
            })
            print(f"Move {move_number}: {next_move}")
            move_number += 1
            if move_number >= 10: # 10手までで終了
                break
        
        return positions

    def _is_game_over(self, position: str) -> bool:
        """Check if the game should end.
        
        Args:
            position: Current position in SFEN format.
            
        Returns:
            bool: True if game should end, False to continue.
        """
        self.engine.get_current_sfen(position)  # This updates engine's position state
        return self.engine.get_best_move() == "none"

    def save_positions_csv(self, positions: List[List[Dict]], filepath: str = "datasets/positions.csv"):
        """Save generated positions to CSV file.
        
        Args:
            positions: List of games with their positions.
            filepath: Output CSV file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['sfen', 'hands', 'game_id', 'move_number', 'previous_move'])
            writer.writeheader()
            for game_id, game_positions in enumerate(positions):
                for pos in game_positions:
                    writer.writerow({
                        'sfen': pos['sfen'],
                        'hands': pos['hands'],
                        'game_id': game_id,
                        'move_number': pos['move_number'],
                        'previous_move': pos.get('previous_move', '')
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
