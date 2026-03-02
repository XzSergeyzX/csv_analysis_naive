import argparse
import yaml
import pandas as pd
import joblib


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path, input_path, output_path):
    config = load_config(config_path)
    model_path = config["model_path"]

    model = joblib.load(model_path)

    df = pd.read_csv(input_path)
    preds = model.predict(df)

    df["prediction"] = preds
    df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.config, args.input, args.output)