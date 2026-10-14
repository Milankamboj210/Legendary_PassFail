
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

tf.keras.utils.set_random_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load data and define target
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

# 1 = Pass, 0 = Fail
df["Pass"] = (df["G3"] >= 10).astype(int)

# Exactly four features: no G1, G2, or G3 inputs
X = df[["sex", "age", "studytime", "absences"]].copy()
y = df["Pass"].copy()

print("Features:", X.columns.tolist())
print("Dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. Create train, validation, and test sets
# --------------------------------------------------

# Reserve the test set for final evaluation only
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the remaining training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)


# --------------------------------------------------
# 4. Preprocess the four features
# --------------------------------------------------

numerical_cols = ["age", "studytime", "absences"]
categorical_cols = ["sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# Fit preprocessing only on the training subset
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)
X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)
X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_val_np = y_val.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.int32)

input_features = X_train_processed.shape[1]

print("\nTraining shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)


# --------------------------------------------------
# 5. Build ANN
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_features,)),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 6. Train ANN
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

history = model.fit(
    X_train_processed,
    y_train_np,
    epochs=150,
    batch_size=16,
    validation_data=(X_val_processed, y_val_np),
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 7. Select threshold using validation data
# --------------------------------------------------

val_probabilities = model.predict(
    X_val_processed, verbose=0
).ravel()

thresholds = np.arange(0.20, 0.801, 0.01)

threshold_results = []

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = accuracy_score(y_val, val_predictions)

    threshold_results.append((score, threshold))

# Prefer a threshold closest to 0.50 when accuracy ties
best_val_accuracy, best_threshold = max(
    threshold_results,
    key=lambda item: (item[0], -abs(item[1] - 0.50))
)

print("\nBest validation accuracy:",
      f"{best_val_accuracy:.2%}")
print("Selected threshold:", round(best_threshold, 2))


# --------------------------------------------------
# 8. Final test evaluation
# --------------------------------------------------

test_probabilities = model.predict(
    X_test_processed, verbose=0
).ravel()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test_np, test_predictions
)

print("\nFinal test accuracy:",
      f"{test_accuracy:.2%}")

# Compare with a majority-class baseline
majority_class = int(y_train_full.mode().iloc[0])
baseline_predictions = np.full(
    len(y_test_np), majority_class
)

baseline_accuracy = accuracy_score(
    y_test_np, baseline_predictions
)

print("Majority-class baseline accuracy:",
      f"{baseline_accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_np,
    test_predictions,
    labels=[0, 1]
))

print("\nClassification Report:")
print(classification_report(
    y_test_np,
    test_predictions,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    zero_division=0
))


# --------------------------------------------------
# 9. Plot learning curves
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Loss")
plt.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. Save matching artifacts
# --------------------------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)

model.save(MODEL_DIR / "model.keras")

joblib.dump(
    float(best_threshold),
    MODEL_DIR / "threshold.pkl"
)

print("\nModel saved:", MODEL_DIR / "model.keras")
print("Preprocessor saved:", MODEL_DIR / "preprocessor.pkl")
print("Threshold saved:", MODEL_DIR / "threshold.pkl")
print("Model input shape:", model.input_shape)
print("Expected original features:", X.columns.tolist())
