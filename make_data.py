"""
Script to generate shogi training data.
"""
import random
from pathlib import Path

from src.shogi import YaneuraOuEngine
from src.data import ShogiDataGenerator

def main(num_games: int = 100, seed: int = 42):
    """Generate training data from shogi games.
    
    Args:
        num_games: Number of games to generate.
        seed: Random seed for reproducibility.
    """
    # Set random seed
    random.seed(seed)
    
    # Initialize shogi engine
    engine = YaneuraOuEngine()
    if not engine.start():
        raise RuntimeError("Failed to start YaneuraOu engine")
    
    try:
        # Initialize data generator
        generator = ShogiDataGenerator(engine)
        
        # Generate positions from games
        positions = generator.generate_data(num_games=num_games)
        
        # Save raw positions to CSV
        generator.save_positions_csv(
            positions,
            filepath="datasets/positions.csv"
        )
        
        # Convert to JSONL training format
        generator.convert_to_jsonl(
            csv_path="datasets/positions.csv",
            output_path="datasets/training_data.jsonl"
        )
        
        # Print summary
        total_positions = sum(len(game_pos) for game_pos in positions)
        print(f"\nGenerated {total_positions} positions from {num_games} games")
        print(f"Data saved to:")
        print(f"- Raw positions: datasets/positions.csv")
        print(f"- Training data: datasets/training_data.jsonl")
        
    finally:
        engine.close()

if __name__ == "__main__":
    # Create datasets directory if it doesn't exist
    Path("datasets").mkdir(exist_ok=True)
    main()
