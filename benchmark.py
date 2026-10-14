
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(
    BASE_DIR / "data" / "student-mat.csv",
    sep=";"
)

# 2. Define target and exactly four features
X = df[["sex", "age", "studytime", "absences"]].copy()
y = (df["G3"] >= 10).astype(int)

# 3. Create a training set and untouched test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        ["age", "studytime", "absences"]
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        ["sex"]
    )
])

# 5. Candidate models
models = {
    "Majority baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight=None,
        random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        random_state=42
    )
}

# 6. Train and compare on the same split
results = []

for name, classifier in models.items():

    # HistGradientBoosting requires dense input
    if name == "Gradient Boosting":
        from sklearn.pipeline import FunctionTransformer

        dense_preprocessor = ColumnTransformer([
            (
                "num",
                StandardScaler(),
                ["age", "studytime", "absences"]
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                ["sex"]
            )
        ])

        pipeline = Pipeline([
            ("preprocessing", dense_preprocessor),
            ("model", classifier)
        ])
    else:
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", classifier)
        ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append((name, accuracy, pipeline))

    print(f"\n{name}: {accuracy:.2%}")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

# 7. Rank the candidates by test accuracy for initial exploration
print("\n--- MODEL COMPARISON ---")

for name, accuracy, _ in sorted(
    results,
    key=lambda result: result[1],
    reverse=True
):
    print(f"{name}: {accuracy:.2%}")
