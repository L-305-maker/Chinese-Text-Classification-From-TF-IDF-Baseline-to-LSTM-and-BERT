import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.multiclass import OneVsRestClassifier
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
            max_features=20000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("lr", OneVsRestClassifier(
            LogisticRegression(
                C=12,
                max_iter=3000,
                solver="liblinear",
                tol=1e-4,
                random_state=42
            ),
            n_jobs=1
        ))
    ])


def build_param_grid(small_grid=False):
    if small_grid:
        return {
            "tfidf__max_features": [20000],
            "tfidf__min_df": [2],
            "tfidf__max_df": [0.95],
            "tfidf__ngram_range": [(1, 3)],
            "tfidf__sublinear_tf": [True],
            "lr__estimator__C": [12],
            "lr__estimator__max_iter": [3000],
            "lr__estimator__tol": [1e-4],
        }

    return {
        "tfidf__max_features": [20000, 50000],
        "tfidf__min_df": [2],
        "tfidf__max_df": [0.95],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__sublinear_tf": [True],
        "lr__estimator__C": [8, 12, 16],
        "lr__estimator__max_iter": [3000],
        "lr__estimator__tol": [1e-4],
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


def score_model(model, text, labels):
    pred = model.predict(text)
    return {
        "accuracy": accuracy_score(labels, pred),
        "macro_f1": f1_score(labels, pred, average="macro"),
    }


def find_best_model(train_text, train_label, val_text, val_label, small_grid=False):
    best_params = None
    best_metrics = None
    search_results = []
    param_candidates = list(ParameterGrid(build_param_grid(small_grid=small_grid)))

    for candidate_idx, params in enumerate(param_candidates, start=1):
        model = build_pipeline()
        model.set_params(**params)
        model.fit(train_text, train_label)

        val_metrics = score_model(model, val_text, val_label)
        result = {
            "candidate": candidate_idx,
            "params": params,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        search_results.append(result)

        print(
            f"[{candidate_idx}/{len(param_candidates)}] "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}, "
            f"params={params}"
        )

        if (
            best_metrics is None
            or val_metrics["macro_f1"] > best_metrics["macro_f1"]
            or (
                val_metrics["macro_f1"] == best_metrics["macro_f1"]
                and val_metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_params = params
            best_metrics = val_metrics

    return best_params, best_metrics, search_results


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
    val_label = build_id_map(val_data["label"])
    test_label = build_id_map(test_data["label"])

    train_text = train_data["text"].apply(cut_text)
    val_text = val_data["text"].apply(cut_text)
    test_text = test_data["text"].apply(cut_text)

    config = {
        "model_name": MODEL_NAME,
        "model_type": "sklearn_pipeline",
        "vectorizer": "TfidfVectorizer",
        "classifier": "OneVsRestClassifier(LogisticRegression)",
        "text_preprocessing": "jieba.lcut joined by spaces before TF-IDF",
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "sample_size": sample_size,
        "search_strategy": "validation_set",
        "selection_metric": "val_macro_f1",
        "random_state": 42,
        "param_grid": build_param_grid(small_grid=small_grid),
        "note": "LR is saved as .pkl because it is not a PyTorch model.",
    }

    _, params_dir = initialize_experiment(MODEL_NAME, config)

    best_params, best_val_metrics, search_results = find_best_model(
        train_text,
        train_label,
        val_text,
        val_label,
        small_grid=small_grid,
    )

    final_train_text = pd.concat([train_text, val_text], axis=0).reset_index(drop=True)
    final_train_label = train_label + val_label

    best_lg = build_pipeline()
    best_lg.set_params(**best_params)
    best_lg.fit(final_train_text, final_train_label)
    test_pred = best_lg.predict(test_text)

    metrics = {
        "test_accuracy": accuracy_score(test_label, test_pred),
        "test_macro_f1": f1_score(test_label, test_pred, average="macro"),
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_val_f1": best_val_metrics["macro_f1"],
    }

    model_path = save_sklearn_model(MODEL_NAME, best_lg, filename="model.pkl")
    save_json(best_params, params_dir / "best_params.json")
    save_json(search_results, params_dir / "validation_search_results.json")
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
