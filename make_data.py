"""
Script to generate shogi training data.
"""
import random
from pathlib import Path

from src.shogi import YaneuraOuEngine
from src.data import ShogiDataGenerator

def main(data_size: int = 1000, seed: int = 42, think_time_ms: int = 1000):
    """Generate training data from shogi games.
    
    Args:
        data_size: Number of games to generate.
        seed: Random seed for reproducibility.
        think_time_ms: Thinking time limit per move in milliseconds.
    """
    # Set random seed
    random.seed(seed)
    
    # Initialize shogi engine with thinking time limit
    try:
        engine = YaneuraOuEngine(think_time_ms=think_time_ms)
        if not engine.start():
            raise RuntimeError("Failed to start YaneuraOu engine")
    except Exception as e:
        print(f"Error starting engine: {str(e)}")
        raise RuntimeError(f"Failed to start YaneuraOu engine: {str(e)}")
    
    try:
        # Initialize data generator
        generator = ShogiDataGenerator(engine)
        
        # Generate positions from games
        positions = generator.generate_data(data_size=data_size)
        
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
        print(f"Data saved to:")
        print(f"- Raw positions: datasets/positions.csv")
        print(f"- Training data: datasets/training_data.jsonl")
        
    finally:
        engine.close()

if __name__ == "__main__":
    # Create datasets directory if it doesn't exist
    Path("datasets").mkdir(exist_ok=True)
    main(data_size=1000, seed=42, think_time_ms=100)  # Default 1 second thinking time
