import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data_processor import build_id_map, data_processor
from src.model_utils import (
    LABEL2ID,
    initialize_experiment,
    save_json,
    save_metrics,
    save_sklearn_model,
)


MODEL_NAME = "lr_tfidf"


def cut_text(text):
    import jieba

    return " ".join(jieba.lcut(str(text)))


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            min_df=2,
            ngram_range=(1, 2)
        )),
        ("lr", LogisticRegression(
            max_iter=10000,
            solver="saga",
            random_state=42
        ))
    ])


def build_param_grid(small_grid=False):
    if small_grid:
        return {
            "tfidf__max_features": [5000],
            "tfidf__ngram_range": [(1, 2)],
            "lr__C": [1],
            "lr__max_iter": [10000],
        }

    return {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "lr__C": [0.01, 0.1, 1, 10, 100],
        "lr__max_iter": [10000, 50000],
    }


def sample_train_data(train_data, sample_size):
    if sample_size is None or sample_size >= len(train_data):
        return train_data

    label_count = train_data["label"].nunique()
    per_label = max(1, sample_size // label_count)

    sampled_parts = []
    for _, group in train_data.groupby("label", sort=False):
        sampled_parts.append(
            group.sample(n=min(len(group), per_label), random_state=42)
        )

    sampled = pd.concat(sampled_parts, axis=0)

    remaining = sample_size - len(sampled)
    if remaining > 0:
        rest = train_data.drop(sampled.index)
        extra = rest.sample(n=min(remaining, len(rest)), random_state=42)
        sampled = pd.concat([sampled, extra], axis=0)

    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def get_cv_fold_count(labels, requested_cv=5):
    min_class_count = labels.value_counts().min()
    if min_class_count < 2:
        raise ValueError("Each class needs at least 2 samples for GridSearchCV.")
    return min(requested_cv, int(min_class_count))


def validate_data():
    train_data, val_data, test_data = data_processor()
    for name, df in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        missing_cols = {"text", "label"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name}_data is missing columns: {missing_cols}")

        unknown_labels = sorted(set(df["label"]) - set(LABEL2ID))
        if unknown_labels:
            raise ValueError(f"{name}_data has unknown labels: {unknown_labels}")

    print("Data check passed.")
    print(f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


def train(sample_size=None, small_grid=False):
    train_data, val_data, test_data = data_processor()

    train_data = sample_train_data(train_data, sample_size)

    train_label = build_id_map(train_data["label"])
    test_label = build_id_map(test_data["label"])
    cv = get_cv_fold_count(train_data["label"], requested_cv=5)

    train_text = train_data["text"].apply(cut_text)
    test_text = test_data["text"].apply(cut_text)

    config = {
        "model_name": MODEL_NAME,
        "model_type": "sklearn_pipeline",
        "vectorizer": "TfidfVectorizer",
        "classifier": "LogisticRegression",
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "sample_size": sample_size,
        "cv": cv,
        "random_state": 42,
        "param_grid": build_param_grid(small_grid=small_grid),
        "note": "LR is saved as .pkl because it is not a PyTorch model.",
    }

    _, params_dir = initialize_experiment(MODEL_NAME, config)

    grid_lg = GridSearchCV(
        build_pipeline(),
        param_grid=build_param_grid(small_grid=small_grid),
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid_lg.fit(train_text, train_label)

    best_lg = grid_lg.best_estimator_
    test_pred = best_lg.predict(test_text)

    metrics = {
        "test_accuracy": accuracy_score(test_label, test_pred),
        "test_macro_f1": f1_score(test_label, test_pred, average="macro"),
        "best_cv_score": grid_lg.best_score_,
    }

    model_path = save_sklearn_model(MODEL_NAME, best_lg, filename="model.pkl")
    save_json(grid_lg.best_params_, params_dir / "best_params.json")
    save_metrics(MODEL_NAME, metrics)

    print(f"Saved LR model to: {model_path}")
    print(f"Saved LR parameters to: {params_dir}")
    return metrics


def parse_lr_args(args=None):
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--small-grid", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    return parser.parse_args(args)


def main(args=None):
    args = parse_lr_args(args)

    if args.check_data:
        validate_data()
        return None

    return train(sample_size=args.sample_size, small_grid=args.small_grid)


if __name__ == "__main__":
    main()
