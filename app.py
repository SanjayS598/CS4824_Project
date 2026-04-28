from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


DATA_PATH = Path("data/heart_failure_clinical_records_5000.csv")


def train_test_split_from_scratch(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    test_count = int(np.ceil(X.shape[0] * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_standardizer(train_df, columns):
    means = train_df[columns].mean()
    stds = train_df[columns].std(ddof=0).replace(0, 1.0)
    return means, stds


def apply_standardizer(frame, columns, means, stds):
    transformed = frame.copy()
    transformed[columns] = (transformed[columns] - means) / stds
    return transformed


class LogisticRegression:
    def __init__(self, learning_rate=0.03, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_output)
            dw = np.dot(X.T, (predictions - y)) / n_samples
            db = np.mean(predictions - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)


def load_model_artifacts():
    df = pd.read_csv(DATA_PATH)
    clean_df = df.drop_duplicates().copy()
    if clean_df.isnull().sum().sum() > 0:
        clean_df = clean_df.dropna().reset_index(drop=True)

    feature_columns = [column for column in clean_df.columns if column != "DEATH_EVENT"]
    continuous_features = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
    ]

    X_df = clean_df[feature_columns].copy()
    y = clean_df["DEATH_EVENT"].to_numpy(dtype=float)

    X_train_raw, _, y_train, _ = train_test_split_from_scratch(
        X_df.to_numpy(dtype=float),
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_df = pd.DataFrame(X_train_raw, columns=feature_columns)
    train_means, train_stds = fit_standardizer(X_train_df, continuous_features)
    X_train_processed = apply_standardizer(
        X_train_df, continuous_features, train_means, train_stds
    )

    model = LogisticRegression(learning_rate=0.03, epochs=5000)
    model.fit(X_train_processed.to_numpy(dtype=float), y_train)

    return {
        "feature_columns": feature_columns,
        "continuous_features": continuous_features,
        "train_means": train_means,
        "train_stds": train_stds,
        "model": model,
        "raw_rows": len(df),
        "clean_rows": len(clean_df),
    }


MODEL_ARTIFACTS = load_model_artifacts()

DEFAULT_PATIENT = {
    "age": 75.0,
    "anaemia": 0,
    "creatinine_phosphokinase": 250.0,
    "diabetes": 1,
    "ejection_fraction": 35.0,
    "high_blood_pressure": 1,
    "platelets": 250000.0,
    "serum_creatinine": 1.3,
    "serum_sodium": 137.0,
    "sex": 1,
    "smoking": 0,
    "time": 120.0,
}


def prepare_single_patient(patient_dict):
    patient_df = pd.DataFrame([patient_dict], columns=MODEL_ARTIFACTS["feature_columns"])
    patient_df = apply_standardizer(
        patient_df,
        MODEL_ARTIFACTS["continuous_features"],
        MODEL_ARTIFACTS["train_means"],
        MODEL_ARTIFACTS["train_stds"],
    )
    return patient_df.to_numpy(dtype=float)


def predict_patient_risk(patient_dict, threshold=0.5):
    patient_array = prepare_single_patient(patient_dict)
    probability = float(MODEL_ARTIFACTS["model"].predict_proba(patient_array)[0])
    prediction = int(probability >= threshold)
    return probability, prediction


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    patient = DEFAULT_PATIENT.copy()
    probability = None
    prediction = None
    error = None

    if request.method == "POST":
        try:
            patient = {
                "age": float(request.form["age"]),
                "anaemia": int(request.form["anaemia"]),
                "creatinine_phosphokinase": float(request.form["creatinine_phosphokinase"]),
                "diabetes": int(request.form["diabetes"]),
                "ejection_fraction": float(request.form["ejection_fraction"]),
                "high_blood_pressure": int(request.form["high_blood_pressure"]),
                "platelets": float(request.form["platelets"]),
                "serum_creatinine": float(request.form["serum_creatinine"]),
                "serum_sodium": float(request.form["serum_sodium"]),
                "sex": int(request.form["sex"]),
                "smoking": int(request.form["smoking"]),
                "time": float(request.form["time"]),
            }
            probability, prediction = predict_patient_risk(patient)
        except (KeyError, ValueError):
            error = "Please enter valid numeric values for every field."

    return render_template(
        "index.html",
        patient=patient,
        probability=probability,
        prediction=prediction,
        error=error,
        raw_rows=MODEL_ARTIFACTS["raw_rows"],
        clean_rows=MODEL_ARTIFACTS["clean_rows"],
    )


if __name__ == "__main__":
    app.run(debug=True)
