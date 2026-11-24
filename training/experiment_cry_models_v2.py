
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "training" / "cry_train.csv"
TEST_PATH = ROOT / "training" / "cry_test.csv"

RESULT_DIR = ROOT / "training" / "cry_experiments_v2"
MODEL_DIR = ROOT / "models" / "cry_experiments_v2"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 3

METADATA_COLUMNS = {
    "filename",
    "label",
    "age_group",
    "gender",
    "mel_file",
    "app_instance_id",
}

def main():
    print("\n========== LOADING DATA ==========")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if "app_instance_id" not in train or "app_instance_id" not in test:
        raise ValueError("Missing app_instance_id column.")

    if train["label"].isna().any() or test["label"].isna().any():
        raise ValueError("Missing labels detected.")

    feature_columns = [
        c for c in train.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(train[c])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    missing = sorted(set(feature_columns) - set(test.columns))
    if missing:
        raise ValueError(f"Test data missing features: {missing}")

    X_train = train[feature_columns].astype(float)
    X_test = test[feature_columns].astype(float)

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Test features contain NaN or infinity.")

    y_train = train["label"].astype(str)
    y_test = test["label"].astype(str)

    groups = train["app_instance_id"].astype(str)

    overlap = set(groups) & set(
        test["app_instance_id"].astype(str)
    )

    if overlap:
        raise ValueError(
            f"Train/test group leakage detected: {len(overlap)} groups"
        )

    classes = sorted(y_train.unique())

    if set(y_test.unique()) != set(classes):
        raise ValueError("Training and test classes do not match.")

    group_counts = (
        pd.DataFrame({"label": y_train, "group": groups})
        .groupby("label")["group"]
        .nunique()
    )

    print("Training recordings:", len(train))
    print("Testing recordings:", len(test))
    print("Features:", len(feature_columns))
    print("Classes:", classes)
    print("\nIndependent training groups per class:")
    print(group_counts.to_string())

    if group_counts.min() < N_SPLITS:
        raise ValueError(
            "At least one class has fewer independent groups "
            "than the number of CV folds."
        )

    models = {
        "Random Forest": Pipeline([
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(
                n_estimators=250,
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "Extra Trees": Pipeline([
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", ExtraTreesClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "Balanced Random Forest": Pipeline([
            ("model", BalancedRandomForestClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "SVM RBF": Pipeline([
            ("scale", StandardScaler()),
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    print("\n========== GROUP CROSS-VALIDATION ==========")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    oof_predictions = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        fold_scores = []
        oof = np.empty(len(train), dtype=object)
        oof[:] = None

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train, groups), start=1
        ):
            X_fold = X_train.iloc[train_idx]
            y_fold = y_train.iloc[train_idx]

            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            if set(y_fold.unique()) != set(classes):
                raise ValueError(
                    f"{model_name}, fold {fold}: "
                    "training fold is missing a class."
                )

            model.fit(X_fold, y_fold)
            predictions = model.predict(X_val)

            oof[val_idx] = predictions

            ba = balanced_accuracy_score(y_val, predictions)
            mf1 = f1_score(
                y_val, predictions, average="macro", zero_division=0
            )
            acc = accuracy_score(y_val, predictions)

            fold_scores.append({
                "fold": fold,
                "accuracy": acc,
                "balanced_accuracy": ba,
                "macro_f1": mf1,
            })

            print(
                f"  Fold {fold}: "
                f"Accuracy={acc:.4f}, "
                f"Balanced Accuracy={ba:.4f}, "
                f"Macro F1={mf1:.4f}"
            )

        if any(value is None for value in oof):
            raise ValueError("Some training rows lack OOF predictions.")

        oof = np.asarray(oof, dtype=str)
        oof_predictions[model_name] = oof

        mean_ba = np.mean([
            item["balanced_accuracy"] for item in fold_scores
        ])
        mean_f1 = np.mean([
            item["macro_f1"] for item in fold_scores
        ])
        mean_acc = np.mean([
            item["accuracy"] for item in fold_scores
        ])

        results.append({
            "model": model_name,
            "mean_fold_accuracy": mean_acc,
            "mean_fold_balanced_accuracy": mean_ba,
            "mean_fold_macro_f1": mean_f1,
        })

        pd.DataFrame(fold_scores).to_csv(
            RESULT_DIR / (
                model_name.lower().replace(" ", "_")
                .replace("-", "_") + "_folds.csv"
            ),
            index=False,
        )

        pd.DataFrame({
            "actual": y_train,
            "predicted": oof,
            "group": groups,
        }).to_csv(
            RESULT_DIR / (
                model_name.lower().replace(" ", "_")
                .replace("-", "_") + "_oof.csv"
            ),
            index=False,
        )

    comparison = pd.DataFrame(results).sort_values(
        ["mean_fold_balanced_accuracy", "mean_fold_macro_f1"],
        ascending=False,
    )

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_model = models[best_name]

    print("\nSelected candidate:", best_name)
    print("Fitting candidate on full training data...")

    best_model.fit(X_train, y_train)

    print("\n========== HELD-OUT TEST EVALUATION ==========")

    test_predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_ba = balanced_accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(
        y_test, test_predictions, average="macro", zero_division=0
    )

    print(f"Accuracy:          {test_accuracy:.4f}")
    print(f"Balanced accuracy: {test_ba:.4f}")
    print(f"Macro F1:          {test_f1:.4f}")

    report = classification_report(
        y_test,
        test_predictions,
        labels=classes,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    pd.DataFrame(
        confusion_matrix(y_test, test_predictions, labels=classes),
        index=classes,
        columns=classes,
    ).to_csv(RESULT_DIR / "test_confusion_matrix.csv")

    pd.DataFrame({
        "actual": y_test,
        "predicted": test_predictions,
    }).to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
    )

    (RESULT_DIR / "test_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "classes": classes,
        "model_name": best_name,
    }

    candidate_path = MODEL_DIR / "best_cry_candidate.joblib"
    joblib.dump(bundle, candidate_path)

    summary = {
        "selected_model": best_name,
        "feature_count": len(feature_columns),
        "training_recordings": len(train),
        "test_recordings": len(test),
        "cv_balanced_accuracy": float(
            comparison.iloc[0]["mean_fold_balanced_accuracy"]
        ),
        "cv_macro_f1": float(
            comparison.iloc[0]["mean_fold_macro_f1"]
        ),
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_ba),
        "test_macro_f1": float(test_f1),
        "candidate_model": str(candidate_path),
        "production_model_modified": False,
    }

    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("Candidate saved:", candidate_path)
    print("Results saved:", RESULT_DIR)
    print("Production model was NOT modified.")


if __name__ == "__main__":
    main()
