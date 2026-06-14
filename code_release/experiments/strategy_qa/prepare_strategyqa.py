import json
import requests
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def prepare_strategyqa(subset_size=30):
    """Downloads StrategyQA and prepares a subset for testing."""
    url = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"
    save_path = os.path.join(os.path.dirname(__file__), "strategyqa_full.json")
    subset_path = os.path.join(os.path.dirname(__file__), "strategyqa_subset.json")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        print(f"Downloading StrategyQA from {url}...")
        try:
            response = requests.get(url, timeout=10)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            # Create a mock dataset for testing
            print("Creating mock StrategyQA dataset...")
            data = [
                {
                    "question": "Is a square a type of rectangle?",
                    "answer": True,
                    "decomposition": ["Are squares and rectangles both quadrilaterals?"]
                },
                {
                    "question": "Would a python be hurt by a porcupine's quill?",
                    "answer": True,
                    "decomposition": ["Do porcupine quills hurt when they touch skin?"]
                },
                {
                    "question": "Can you hum a full song without taking a breath?",
                    "answer": True,
                    "decomposition": ["Can humans hold their breath while humming?"]
                }
            ]
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Mock data saved to {save_path}")
    
    with open(save_path, 'r') as f:
        data = json.load(f)
    
    # Filter for questions with clear answers
    random.seed(42)
    subset = random.sample(data, min(subset_size, len(data)))
    
    with open(subset_path, 'w') as f:
        json.dump(subset, f, indent=2)
    print(f"Subset of {len(subset)} questions saved to {subset_path}.")

if __name__ == "__main__":
    prepare_strategyqa()
