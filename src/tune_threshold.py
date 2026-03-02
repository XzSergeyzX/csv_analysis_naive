import argparse
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    df = pd.read_csv(cfg["processed_path"])
    X = df[cfg["text_col"]].astype(str)
    y = df[cfg["target"]].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_val)
    classes = list(pipe.named_steps["clf"].classes_)
    spam_idx = classes.index("spam")
    spam_proba = proba[:, spam_idx]

    y_val = y_val.to_numpy()
    ham_mask = (y_val == "ham")
    spam_mask = (y_val == "spam")

    # Цель: FP <= 1 на 1000 ham
    max_fp_per_1000 = float(cfg.get("max_fp_per_1000", 1.0))
    max_fpr = max_fp_per_1000 / 1000.0

    # Пороги берём из уникальных вероятностей (точнее, чем шаг 0.01)
    thresholds = np.unique(spam_proba)
    thresholds.sort()

    best = None
    rows = []

    for t in thresholds:
        pred_spam = spam_proba >= t

        fp = int(np.sum(pred_spam & ham_mask))
        tn = int(np.sum((~pred_spam) & ham_mask))
        fn = int(np.sum((~pred_spam) & spam_mask))
        tp = int(np.sum(pred_spam & spam_mask))

        fpr = fp / max(1, (fp + tn))
        tpr = tp / max(1, (tp + fn))  # recall(spam)

        rows.append((t, fp, tn, fn, tp, fpr, tpr))

        if fpr <= max_fpr:
            best = (t, fp, tn, fn, tp, fpr, tpr)
            break  # минимальный порог, который укладывается в FP-лимит

    if best is None:
        # если совсем не нашли — берём максимальный порог (минимальный FP)
        best = rows[-1]

    t, fp, tn, fn, tp, fpr, tpr = best
    fp_per_1000 = fpr * 1000.0

    print(f"Constraint: FP/1000 <= {max_fp_per_1000}")
    print(f"Selected threshold: {t:.6f}")
    print(f"Val: FP={fp} TN={tn} FN={fn} TP={tp} | FP/1000={fp_per_1000:.3f} | recall_spam={tpr:.4f}")

    # (опционально) сохраним таблицу для графиков
    out = pd.DataFrame(rows, columns=["threshold", "fp", "tn", "fn", "tp", "fpr", "recall_spam"])
    out.to_csv("data/out/threshold_scan.csv", index=False)
    print("Saved scan to data/out/threshold_scan.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)