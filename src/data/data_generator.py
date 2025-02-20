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
        
        return all_positions

    def _generate_game_positions(self) -> List[Dict[str, Any]]:
        """Generate all positions from a single game.
        
        Returns:
            List[Dict]: List of positions from the game.
        """
        positions = []
        moves = []
        move_number = 0
        
        # Set initial position
        self.engine.set_position("startpos")
        
        while True:
            # Get accurate SFEN for current position
            current_sfen = self.engine.get_current_sfen()
            hands = "なし"  # Default value
            if " w " in current_sfen:
                hands = current_sfen.split(" w ")[1].split(" ")[0]
            elif " b " in current_sfen:
                hands = current_sfen.split(" b ")[1].split(" ")[0]
            if hands == "-":
                hands = "なし"
            
            # Store current position
            positions.append({
                "sfen": current_sfen,
                "hands": hands,
                "move_number": move_number,
                "previous_move": moves[-1] if moves else None
            })
            
            # Get legal moves for current position
            legal_moves = self.engine.get_legal_moves()
            
            # Check for game end
            if not legal_moves:
                print("Game over")
                break
                
            # Select and play next move
            next_move = random.choice(legal_moves)
            moves.append(next_move)
            move_number += 1
            
            # Update engine's position with new move
            current_position = f"position startpos moves {' '.join(moves)}"
            self.engine.set_position(current_position)
            
            print(f"Move {move_number}: {next_move}")
            
            # Check for game end conditions
            # if self._is_game_over(current_position):
            #     break
                
            # Safety check to prevent infinite games
            if move_number >= 10:  # Typical shogi games rarely exceed 200 moves
            # if move_number >= 200:  # Typical shogi games rarely exceed 200 moves
                break
        
        return positions

    def _is_game_over(self, position: str) -> bool:
        """Check if the game should end.
        
        Args:
            position: Current position in SFEN format.
            
        Returns:
            bool: True if game should end, False to continue.
        """
        # Get legal moves first - if none, game is over
        legal_moves = self._get_legal_moves(position)
        if not legal_moves:
            return True
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
                if row['previous_move']:
                    response = f"前回の手：{row['previous_move']}\n\n"
                    response += "その手を選んだ理由は以下の通りです：\n"
                    response += "1. この手は相手の攻めに対して堅実な防御になります。\n"
                    response += "2. 中盤戦に向けて駒の効率的な展開を目指せます。\n"
                    response += "3. 次の手で攻めの形を作る準備になります。"
                else:
                    response = "初期局面のため、７六歩や２六歩など、堅実な序盤戦術から始めることをお勧めします。"
                    
                entry = {
                    "prompt": f"次の局面で指すべき手を考えてください。\n\n{markdown_board}",
                    "response": response
                }
                json.dump(entry, jsonlfile, ensure_ascii=False)
                jsonlfile.write('\n')
