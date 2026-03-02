import argparse
from pathlib import Path
import pandas as pd


def main(input_path: str, output_path: str):
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # формат: label \t text
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts[0].strip(), parts[1].strip()
            rows.append({"label": label, "text": text})

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved: {out} | rows={len(df)} | labels={df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.input, args.output)