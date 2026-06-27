from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")


def generate_synthetic_data(n_samples=5000, random_state=42):
    rng = np.random.default_rng(random_state)
    temperature = rng.normal(82, 7, n_samples)
    vibration = rng.normal(0.55, 0.12, n_samples)
    pressure = rng.normal(75, 6, n_samples)
    humidity = rng.normal(45, 8, n_samples)
    operating_cycles = rng.uniform(50, 1800, n_samples)

    failure_prob = 1 / (
        1
        + np.exp(
            -(
                -6.8
                + 0.095 * (temperature - 70)
                + 6.0 * (vibration - 0.4)
                + 0.06 * (pressure - 70)
                + 0.05 * (humidity - 40)
                + 0.003 * operating_cycles
            )
        )
    )
    failure_prob = np.clip(failure_prob + 0.08 * np.maximum((temperature - 85) / 20, 0), 0.01, 0.95)
    failure = rng.random(n_samples) < failure_prob

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "humidity": humidity,
            "operating_cycles": operating_cycles,
            "failure": failure.astype(int),
        }
    )
    return df


def engineer_features(df):
    df = df.copy()
    df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"] + 1e-6)
    df["vibration_temp_interaction"] = df["vibration"] * df["temperature"]
    df["high_load"] = ((df["temperature"] > 85) & (df["vibration"] > 0.6)).astype(int)
    return df


def train_models(df):
    df = engineer_features(df)
    features = [
        "temperature",
        "vibration",
        "pressure",
        "humidity",
        "operating_cycles",
        "temp_pressure_ratio",
        "vibration_temp_interaction",
        "high_load",
    ]
    X = df[features]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds), 4),
                "recall": round(recall_score(y_test, preds), 4),
                "f1_score": round(f1_score(y_test, preds), 4),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            }
        )

    return pd.DataFrame(results)


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "predictive_maintenance_synthetic.csv"

    df = generate_synthetic_data()
    df.to_csv(csv_path, index=False)
    metrics_df = train_models(df)
    print(metrics_df)


if __name__ == "__main__":
    main()
