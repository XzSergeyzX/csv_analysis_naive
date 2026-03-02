import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_threshold_zero_fp(spam_proba: np.ndarray, y_true: np.ndarray) -> float:
    ham_mask = (y_true == "ham")
    thresholds = np.unique(spam_proba)
    thresholds.sort()

    # минимальный порог, при котором FP=0
    for t in thresholds:
        pred_spam = spam_proba >= t
        fp = int(np.sum(pred_spam & ham_mask))
        if fp == 0:
            return float(t)

    # фолбэк: никого не помечать как spam
    return 1.0


def proba_and_pred(model: Pipeline, texts: pd.Series, threshold: float):
    proba = model.predict_proba(texts.astype(str))
    classes = list(model.named_steps["clf"].classes_)
    spam_idx = classes.index("spam")
    spam_proba = proba[:, spam_idx]
    pred = np.where(spam_proba >= threshold, "spam", "ham")
    return spam_proba, pred


def main(config_path: str):
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))

    train_df = pd.read_csv(cfg["train_path"])
    test_df = pd.read_csv(cfg["test_path"])

    text_col = cfg["text_col"]
    target_col = cfg["target"]

    X_train_full = train_df[text_col].astype(str)
    y_train_full = train_df[target_col].astype(str).to_numpy()

    X_test = test_df[text_col].astype(str)
    y_test = test_df[target_col].astype(str).to_numpy()

    # внутренняя валидация (test не трогаем)
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 1) обучили на sub
    model.fit(X_sub, y_sub)

    # 2) threshold: либо фикс из конфига, либо zero-fp на val
    if "threshold" in cfg:
        threshold = float(cfg["threshold"])
        print(f"Using fixed threshold from config: {threshold:.6f}")
    else:
        val_proba, _ = proba_and_pred(model, X_val, threshold=0.0)
        threshold = pick_threshold_zero_fp(val_proba, y_val)
        print(f"Chosen threshold (zero FP on val): {threshold:.6f}")
        print(f"Recommended to add to config.yaml: threshold: {threshold:.6f}")

    # 3) финально переобучаем на ВСЁМ train
    model.fit(X_train_full, y_train_full)

    # 4) оценка на test
    _, test_pred = proba_and_pred(model, X_test, threshold=threshold)

    cm = confusion_matrix(y_test, test_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()

    fp_per_1000 = (fp / max(1, (fp + tn))) * 1000.0
    recall_spam = tp / max(1, (tp + fn))

    print("TEST confusion matrix labels=['ham','spam']:")
    print(cm)
    print(f"TEST FP={fp} FN={fn} TP={tp} TN={tn} | FP/1000={fp_per_1000:.3f} | recall_spam={recall_spam:.4f}")
    print()
    print(classification_report(y_test, test_pred, digits=4))

    # save model
    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)