import argparse
from pathlib import Path
import yaml
import pandas as pd
import joblib


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, input_path: str, output_path: str):
    cfg = load_config(config_path)
    threshold = float(cfg.get("threshold", 0.5))
    model_path = cfg["model_path"]

    model = joblib.load(model_path)

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    proba = model.predict_proba(df["text"].astype(str))
    # модель — Pipeline, последний шаг clf
    classes = list(model.named_steps["clf"].classes_)
    spam_idx = classes.index("spam")
    spam_proba = proba[:, spam_idx]

    pred = ["spam" if p >= threshold else "ham" for p in spam_proba]

    out = df.copy()
    out["spam_proba"] = spam_proba
    out["prediction"] = pred

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path} (threshold={threshold})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.config, args.input, args.output)