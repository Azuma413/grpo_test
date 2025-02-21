import json
from typing import List, Dict, Optional
from pathlib import Path

class ShogiDataset:
    """Dataset class for shogi training data."""
    
    def __init__(self):
        """Initialize dataset.
        
        Args:
            system_prompt: Optional system prompt to prepend to each example.
        """
        self.system_prompt = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
        self.data: List[Dict] = []
        
    def load_jsonl(self, path: str) -> None:
        """Load dataset from JSONL file.
        
        Args:
            path: Path to JSONL file containing training examples.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def save_jsonl(self, path: str) -> None:
        """Save dataset to JSONL file.
        
        Args:
            path: Output path for JSONL file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in self.data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

    def add_example(self, prompt: str, response: str = "") -> None:
        """Add a single example to the dataset.
        
        Args:
            prompt: Input prompt/question.
            response: Optional model response/answer.
        """
        self.data.append({
            "prompt": prompt,
            "response": response
        })
        
    def get_training_examples(self) -> List[Dict]:
        """Get examples in training format.
        
        Returns:
            List[Dict]: Examples formatted for training, including system prompt
                if specified.
        """
        examples = []
        for item in self.data:
            example = {
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": item["prompt"]}
                ],
                "answer": item["response"]
            }
            examples.append(example)
        return examples

    def __len__(self) -> int:
        """Get number of examples in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        return self.data[idx]