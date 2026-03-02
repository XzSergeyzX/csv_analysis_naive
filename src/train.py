import argparse
from pathlib import Path
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    df = pd.read_csv(cfg["processed_path"])
    text_col = cfg["text_col"]
    target_col = cfg["target"]

    X = df[text_col].astype(str)
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    # после pipeline.fit(...)
    proba = pipeline.predict_proba(X_test)
    # столбец вероятности класса "spam"
    spam_idx = list(pipeline.named_steps["clf"].classes_).index("spam")
    spam_proba = proba[:, spam_idx]

    threshold = float(cfg.get("threshold", 0.5))
    preds = ["spam" if p >= threshold else "ham" for p in spam_proba]

    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()
    print(f"threshold={threshold}")
    print(cm)
    print(f"FP={fp} FN={fn} TP={tp} TN={tn}")
    print()
    print(classification_report(y_test, preds, digits=4))

    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)