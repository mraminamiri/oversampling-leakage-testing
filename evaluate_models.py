"""Fit the seven study models and save their predictions and test scores."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipelines import (
    load_data,
    make_buggy_adasyn,
    make_buggy_ros,
    make_buggy_smote,
    make_reference_pipeline,
    original_split,
)

ROOT = Path(__file__).resolve().parent


def evaluate_models(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "results"
    X, y = load_data()
    X_train, X_test, y_train, y_test = original_split(X, y)
    if not X_train.index.intersection(X_test.index).empty:
        raise ValueError("The original training and test sets overlap")

    models = {}
    for method, name in [
        ("baseline", "Baseline"),
        ("ros", "Correct ROS"),
        ("smote", "Correct SMOTE"),
        ("adasyn", "Correct ADASYN"),
    ]:
        model = make_reference_pipeline(method)
        model.fit(X_train, y_train)
        models[name] = model

    for name, factory in [
        ("Buggy ROS", make_buggy_ros),
        ("Buggy SMOTE", make_buggy_smote),
        ("Buggy ADASYN", make_buggy_adasyn),
    ]:
        model, _ = factory(X, y, X_train, y_train)
        models[name] = model

    metric_rows = []
    prediction_rows = []
    confusion_rows = []
    for name, model in models.items():
        if model.classes_.tolist() != [0, 1]:
            raise ValueError("Expected class 1 to be the positive class")
        probability = model.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        if not np.isfinite(probability).all():
            raise ValueError(f"Non-finite predictions from {name}")

        metric_rows.append({
            "Scenario": name,
            "Ground truth": "LEAKY" if name.startswith("Buggy") else "CLEAN",
            "Test samples": len(y_test),
            "Accuracy": accuracy_score(y_test, predicted),
            "Balanced accuracy": balanced_accuracy_score(y_test, predicted),
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "Recall": recall_score(y_test, predicted, zero_division=0),
            "F1": f1_score(y_test, predicted, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probability),
            "Average precision (AP)": average_precision_score(y_test, probability),
        })
        for source_id, actual, prediction, score in zip(
            X_test.index, y_test, predicted, probability
        ):
            prediction_rows.append({
                "Scenario": name,
                "source_id": int(source_id),
                "y_true": int(actual),
                "y_pred": int(prediction),
                "probability_class_1": float(score),
            })
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        confusion_rows.append({
            "Scenario": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    metrics = pd.DataFrame(metric_rows).set_index("Scenario")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "pilot_model_performance.csv")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nF1, precision and recall refer to Outcome=1; threshold=0.5.")
    print("Buggy-model scores use contaminated Test records; they are descriptive.")
    return metrics


if __name__ == "__main__":
    evaluate_models()
