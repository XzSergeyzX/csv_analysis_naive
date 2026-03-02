import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main(input_path: str, train_path: str, test_path: str, test_size: float, seed: int):
    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"]
    )

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train={len(train_df)} to {train_path}")
    print(f"Saved test ={len(test_df)} to {test_path}")
    print("Test label distribution:", test_df["label"].value_counts().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train_out", required=True)
    p.add_argument("--test_out", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args.input, args.train_out, args.test_out, args.test_size, args.seed)