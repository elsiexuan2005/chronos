from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent 
DATA_PATH = ROOT / "data" / "activity_training.csv"
MODEL_DIR = ROOT / "ml"

gameplay_features = [
    "difficulty",
    "duration_seconds",
    "wrong_attempts",
]

vision_features = [
    "face_visibility_ratio",
    "pitch_coverage_ratio",
    "mean_pitch_deviation",
    "pitch_variability",
    "head_tilt_ratio",
]

df = pd.read_csv(DATA_PATH)

# Use whole sessions for testing, not individual random rows.
# Hold out the last session of each activity.
test_sessions = set()

for activity in ["mental", "paper", "calculator"]:
    sessions = df.loc[
        df["activity"] == activity, "session_id"
    ].drop_duplicates().tolist()

    if len(sessions) < 2:
        raise ValueError(f"Need at least two sessions for {activity}")

    test_sessions.add(sessions[-1])

train = df[~df["session_id"].isin(test_sessions)].copy()
test = df[df["session_id"].isin(test_sessions)].copy()

print("Training records:", len(train))
print("Testing records:", len(test))


def train_model(name, features, filename):
    # X = information the model can see
    X_train = train[features].fillna(-1)
    X_test = test[features].fillna(-1)

    # y = correct activity labels
    y_train = train["activity"]
    y_test = test["activity"]

    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42
    )

    # Learn from training examples
    model.fit(X_train, y_train)

    # Predict activities for held-out questions
    predictions = model.predict(X_test)

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, predictions))

    labels = ["mental", "paper", "calculator"]

    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    ))



    labels = ["mental", "paper", "calculator"]
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / filename

    joblib.dump(
        {"model": model, "features": features},
        model_path
    )

    print("Saved model:", model_path)


# MODEL A
train_model(
    "Gameplay only",
    gameplay_features,
    "model_a.joblib"
)

# MODEL B
train_model(
    "Gameplay + Vision",
    gameplay_features + vision_features,
    "model_b.joblib"
)