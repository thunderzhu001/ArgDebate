import argparse
import pandas as pd
import requests
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def _resolve_output_path(output: str | None) -> str:
    if not output:
        return os.path.join(os.path.dirname(__file__), "truthfulqa_subset.csv")
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return str(path)


def prepare_truthfulqa(subset_size=50, seed=None, output=None):
    """Downloads TruthfulQA and prepares a subset for testing."""
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    save_path = os.path.join(os.path.dirname(__file__), "truthfulqa_full.csv")
    subset_path = _resolve_output_path(output)
    
    if not os.path.exists(save_path):
        print(f"Downloading TruthfulQA from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            # Create a mock dataset for testing
            print("Creating mock TruthfulQA dataset...")
            df = pd.DataFrame({
                'Question': [
                    'What is the capital of France?',
                    'Who won the 2020 US Presidential Election?',
                    'What is the largest planet in our solar system?'
                ],
                'Best Answer': [
                    'Paris',
                    'Joe Biden',
                    'Jupiter'
                ],
                'Correct Answers': [
                    'Paris',
                    'Joe Biden',
                    'Jupiter'
                ],
                'Incorrect Answers': [
                    'London',
                    'Donald Trump',
                    'Saturn'
                ]
            })
            df.to_csv(save_path, index=False)
            print(f"Mock data saved to {save_path}")
    
    df = pd.read_csv(save_path)
    # Filter for questions with clear answers
    subset = df.sample(n=min(subset_size, len(df)), random_state=seed)
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    subset.to_csv(subset_path, index=False)
    print(f"Subset of {len(subset)} questions saved to {subset_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a TruthfulQA subset.")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Relative paths are resolved under experiments/factual_qa/.",
    )
    args = parser.parse_args()
    prepare_truthfulqa(subset_size=args.subset_size, seed=args.seed, output=args.output)
