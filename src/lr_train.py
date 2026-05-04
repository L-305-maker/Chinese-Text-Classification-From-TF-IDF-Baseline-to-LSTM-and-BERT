from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import os
import json
import joblib

from data_process import data_processor, build_id_map


def cut_text(text):
    return " ".join(jieba.lcut(str(text)))


test_data, train_data, val_data = data_processor()

train_label = build_id_map(train_data["label"])
test_label = build_id_map(test_data["label"])

train_text = train_data["text"].apply(cut_text)
test_text = test_data["text"].apply(cut_text)

pipe = Pipeline([
    ("TF_IDF", TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2)
    )),
    ("lg", LogisticRegression(
        max_iter=10000,
        solver="saga",
        random_state=42
    ))
])

param = {
    "TF_IDF__max_features": [5000, 10000],
    "TF_IDF__ngram_range": [(1, 2), (1, 3)],
    "lg__C": [0.01, 0.1, 1, 10, 100],
    "lg__max_iter": [10000, 50000]
}

grid_lg = GridSearchCV(
    pipe,
    param_grid=param,
    cv=5,
    n_jobs=1,
    verbose=1
)

grid_lg.fit(train_text, train_label)

best_lg = grid_lg.best_estimator_

test_pred = best_lg.predict(test_text)

os.makedirs("models", exist_ok=True)
os.makedirs("parameters", exist_ok=True)

joblib.dump(best_lg, "models/log_tfidf_model.pkl")

with open("parameters/best_lg_params.json", "w", encoding="utf-8") as f:
    json.dump(grid_lg.best_params_, f, ensure_ascii=False, indent=4)