"""Train isolated Phase 7 cry experiments without touching production data/models."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import sklearn
import imblearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "training" / "cry_features.csv"
METADATA_PATH = ROOT / "training" / "processed_cry_metadata_clean.csv"
OLD_TEST_PATH = ROOT / "training" / "cry_test.csv"
AUDIT_PATH = ROOT / "training" / "reports" / "cry_dataset_audit.md"
OUTPUT_DIR = ROOT / "models" / "experiments" / "cry_phase7"
REPORT_PATH = ROOT / "training" / "reports" / "cry_model_comparison_phase7.md"

SEED = 14
VALIDATION_SPLIT_SEED = 100014
MODEL_SEED = 42
LABELS = [
    "belly_pain", "burping", "cold_hot", "discomfort",
    "hungry", "lonely", "scared", "tired",
]

# These seven manually identified keys are excluded as a whole, including
# format copies, pending human adjudication of the cleaned folder/code conflict.
QUESTIONABLE_KEYS = {
    "999bf14b-e417-4b44-b746-9253f81efe38-1430844958178",
    "999bf14b-e417-4b44-b746-9253f81efe38-1430844979531",
    "999bf14b-e417-4b44-b746-9253f81efe38-1430844994519",
    "999bf14b-e417-4b44-b746-9253f81efe38-1430845018393",
    "999bf14b-e417-4b44-b746-9253f81efe38-1430845035663",
    "d6cda191-4962-4308-9a36-46d5648a95ed-1431086262776",
    "f258a8e2-92cd-4843-9da4-80ed26ec8e10-1436944996156",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_key(filename: str) -> tuple[str, str]:
    """Return documented app UUID + timestamp key and app UUID separately."""
    parts = Path(str(filename)).stem.rsplit("-", 5)
    if len(parts) != 6:
        return "", ""
    app_id, timestamp = parts[0].lower(), parts[1]
    if not timestamp.isdigit():
        return "", ""
    return f"{app_id}-{timestamp}", app_id


def resolve_project_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def audio_is_usable(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing_processed_audio"
    try:
        info = sf.info(path)
        if info.frames <= 0:
            return False, "zero_frame_audio"
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
        if audio.size == 0 or not np.isfinite(audio).all():
            return False, "empty_or_nonfinite_audio"
        if float(np.max(np.abs(audio))) == 0.0:
            return False, "silent_audio"
        return True, ""
    except Exception as exc:  # malformed WAV or decoder failure
        return False, f"audio_decode_error:{type(exc).__name__}"


def source_rank(row: pd.Series) -> tuple[int, int, str]:
    """Prefer native 16 kHz CAF, then lossless WAV, then compressed 3GP."""
    fmt = str(row["format"]).lower()
    quality = {".caf": 3, ".wav": 2, ".3gp": 1}.get(fmt, 0)
    rate = {".caf": 16000, ".wav": 8000, ".3gp": 8000}.get(fmt, 0)
    return quality, rate, str(row["original_filename"]).lower()


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    feature_df = pd.read_csv(FEATURES_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)
    old_test_df = pd.read_csv(OLD_TEST_PATH)

    required_feature = {"filename", "label"}
    required_metadata = {
        "processed_filename", "original_filename", "original_file_path",
        "processed_file_path", "format", "reason_code", "label",
    }
    if not required_feature.issubset(feature_df.columns):
        raise ValueError("Cry features are missing required filename/label columns.")
    if not required_metadata.issubset(metadata_df.columns):
        raise ValueError("Processed metadata is missing required provenance columns.")
    if feature_df["filename"].duplicated().any():
        raise ValueError("Feature filenames are not unique; refusing an ambiguous join.")
    if metadata_df["processed_filename"].duplicated().any():
        raise ValueError("Processed metadata filenames are not unique.")

    data = feature_df.merge(
        metadata_df,
        left_on="filename",
        right_on="processed_filename",
        how="left",
        suffixes=("_feature", "_metadata"),
        validate="one_to_one",
        indicator=True,
    )
    if not data["_merge"].eq("both").all():
        raise ValueError("Feature rows do not all have one source metadata row.")
    if not data["label_feature"].astype(str).eq(data["label_metadata"].astype(str)).all():
        raise ValueError("Feature and metadata labels differ; review before model fitting.")
    data["label"] = data["label_feature"].astype(str)
    keys = data["original_filename"].map(recording_key)
    data["recording_key"] = [item[0] for item in keys]
    data["app_instance_id"] = [item[1] for item in keys]
    data["original_path"] = data["original_file_path"].map(resolve_project_path)
    data["processed_path"] = data["processed_file_path"].map(resolve_project_path)

    if (data["recording_key"] == "").any():
        raise ValueError("Parsed metadata contains an invalid recording-key filename.")
    label_sets = data.groupby("recording_key")["label"].agg(lambda s: set(s))
    conflicting_keys = set(label_sets[label_sets.map(len) > 1].index)
    old_test_keys = {
        recording_key(name)[0]
        for name in old_test_df["filename"].astype(str)
        if recording_key(name)[0]
    }
    old_test_app_ids = {
        recording_key(name)[1]
        for name in old_test_df["filename"].astype(str)
        if recording_key(name)[1]
    }

    # Cache audio validation by processed path. This checks actual audio content,
    # not amplitude-based class inference; it only rejects empty/invalid files.
    audio_validation = {}
    for path in data["processed_path"].drop_duplicates():
        audio_validation[path] = audio_is_usable(path)

    reasons: dict[int, str] = {}
    candidates = []
    for key, group in data.groupby("recording_key", sort=True):
        label_values = set(group["label"])
        if key in conflicting_keys:
            for idx in group.index:
                reasons[idx] = "conflicting_labels_within_recording_key"
            continue
        label = next(iter(label_values)) if label_values else ""
        if not label or label in {"nan", "None"}:
            for idx in group.index:
                reasons[idx] = "missing_label"
            continue
        if label not in LABELS:
            for idx in group.index:
                reasons[idx] = "not_in_eight_class_taxonomy"
            continue
        if key in QUESTIONABLE_KEYS:
            for idx in group.index:
                reasons[idx] = "manual_label_review_required"
            continue

        usable_rows = []
        for idx, row in group.iterrows():
            ok, reason = audio_validation[row["processed_path"]]
            if ok:
                usable_rows.append(idx)
            else:
                reasons[idx] = reason
        if not usable_rows:
            continue

        usable_group = group.loc[usable_rows].copy()
        chosen_idx = sorted(usable_rows, key=lambda idx: source_rank(group.loc[idx]))[0]
        candidates.append(data.loc[chosen_idx].copy())
        for idx in usable_rows:
            if idx != chosen_idx:
                reasons[idx] = "format_copy_not_selected"

    chosen = pd.DataFrame(candidates).reset_index(drop=True)
    # Remove every recording key from the previous test set from the Phase 7
    # experiment; the new holdout is drawn only from non-legacy-test keys.
    legacy_mask = (
        chosen["recording_key"].isin(old_test_keys)
        | chosen["app_instance_id"].isin(old_test_app_ids)
    )
    legacy_candidate_keys = int(chosen.loc[legacy_mask, "recording_key"].nunique())
    legacy_candidate_apps = int(chosen.loc[legacy_mask, "app_instance_id"].nunique())
    for _, row in chosen[legacy_mask].iterrows():
        original_index = int(data.index[data["filename"] == row["filename"]][0])
        reasons[original_index] = "previously_evaluated_test_key_excluded"
    chosen = chosen.loc[~legacy_mask].reset_index(drop=True)

    # Exact-content duplicates across distinct keys cannot be independent items.
    chosen["content_sha256"] = chosen["original_path"].map(file_sha256)
    hash_groups = chosen.groupby("content_sha256", sort=False)
    drop_indices = set()
    for _, group in hash_groups:
        if len(group) <= 1:
            continue
        if group["label"].nunique() > 1:
            drop_indices.update(group.index)
            for idx in group.index:
                reasons[int(idx)] = "conflicting_labels_for_identical_audio"
            continue
        ordered = sorted(group.index, key=lambda idx: source_rank(chosen.loc[idx]))
        for idx in ordered[1:]:
            drop_indices.add(idx)
            original_index = int(data.index[data["filename"] == chosen.loc[idx, "filename"]][0])
            reasons[original_index] = "exact_audio_duplicate_not_selected"
    if drop_indices:
        chosen = chosen.drop(index=drop_indices).reset_index(drop=True)

    feature_columns = [
        col for col in feature_df.columns
        if col not in {"filename", "label", "age_group", "gender", "mel_file"}
        and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    if len(feature_columns) != 91:
        raise ValueError(f"Expected 91 existing audio features, found {len(feature_columns)}.")
    if not np.isfinite(chosen[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Selected feature rows contain non-finite numeric values.")

    # Make an auditable exclusion log from all original feature rows.
    exclusion_rows = []
    for idx, row in data.iterrows():
        reason = reasons.get(idx)
        if reason:
            exclusion_rows.append({
                "feature_filename": row["filename"],
                "original_filename": row["original_filename"],
                "recording_key": row["recording_key"],
                "label": row["label"],
                "reason": reason,
            })
    exclusion_log = pd.DataFrame(exclusion_rows)

    preparation = {
        "feature_rows": int(len(feature_df)),
        "metadata_rows": int(len(metadata_df)),
        "legacy_test_recording_keys_excluded": legacy_candidate_keys,
        "legacy_test_app_instance_groups_excluded": legacy_candidate_apps,
        "questionable_keys_excluded": int(len(QUESTIONABLE_KEYS)),
        "conflicting_label_keys_found": int(len(conflicting_keys)),
        "selected_unique_recordings": int(len(chosen)),
        "audio_validation_status_counts": {
            status: int(sum(1 for ok, why in audio_validation.values() if ("valid" if ok else why) == status))
            for status in sorted({"valid" if ok else why for ok, why in audio_validation.values()})
        },
        "exclusion_reason_counts": exclusion_log["reason"].value_counts().to_dict()
        if not exclusion_log.empty else {},
        "class_counts_selected": chosen["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
    }
    return chosen, exclusion_log, feature_columns, preparation


def create_group_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # App-installation grouping is stricter than recording-key grouping and
    # keeps all known recordings from one installation in one split.
    app_groups = np.array(sorted(data["app_instance_id"].unique()))
    train_groups, remainder_groups = train_test_split(
        app_groups, test_size=0.30, random_state=SEED, shuffle=True
    )
    validation_groups, test_groups = train_test_split(
        remainder_groups, test_size=0.50, random_state=VALIDATION_SPLIT_SEED, shuffle=True
    )
    group_to_split = {key: "train" for key in train_groups}
    group_to_split.update({key: "validation" for key in validation_groups})
    group_to_split.update({key: "test" for key in test_groups})
    data = data.copy()
    data["split"] = data["app_instance_id"].map(group_to_split)
    if data["split"].isna().any():
        raise ValueError("Some recordings were not assigned to a split.")

    split_groups = {
        name: set(data.loc[data["split"] == name, "app_instance_id"])
        for name in ("train", "validation", "test")
    }
    if (split_groups["train"] & split_groups["validation"]
            or split_groups["train"] & split_groups["test"]
            or split_groups["validation"] & split_groups["test"]):
        raise ValueError("App-installation group leakage detected.")
    key_sets = {
        name: set(data.loc[data["split"] == name, "recording_key"])
        for name in ("train", "validation", "test")
    }
    if key_sets["train"] & key_sets["validation"] or key_sets["train"] & key_sets["test"] or key_sets["validation"] & key_sets["test"]:
        raise ValueError("Recording-key leakage detected.")

    missing = {
        name: sorted(set(LABELS) - set(data.loc[data["split"] == name, "label"]))
        for name in ("train", "validation", "test")
    }
    if any(missing.values()):
        raise ValueError(
            "Cannot create a grouped split containing all classes in every split: "
            + json.dumps(missing)
        )
    counts = {
        name: {
            "recordings": int((data["split"] == name).sum()),
            "app_installation_groups": int(data.loc[data["split"] == name, "app_instance_id"].nunique()),
            "class_counts": data.loc[data["split"] == name, "label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        }
        for name in ("train", "validation", "test")
    }
    info = {
        "split_seed": SEED,
        "validation_split_seed": VALIDATION_SPLIT_SEED,
        "grouping": "app-instance UUID (keeps recording-key copies together); participant independence is unknown",
        "overlap_recording_keys": 0,
        "overlap_app_instance_ids": 0,
        "splits": counts,
        "missing_classes": missing,
    }
    return data, info


def build_candidates():
    from imblearn.ensemble import BalancedRandomForestClassifier

    families = {
        "class_weighted_rbf_svc": [
            {"C": c, "gamma": "scale"} for c in (0.1, 1.0, 10.0)
        ],
        "random_forest_unweighted": [
            {"min_samples_leaf": leaf} for leaf in (1, 2, 4)
        ],
        "random_forest_class_weighted": [
            {"min_samples_leaf": leaf} for leaf in (1, 2, 4)
        ],
        "balanced_random_forest": [
            {"min_samples_leaf": leaf} for leaf in (1, 2, 4)
        ],
        "logistic_regression_balanced": [
            {"C": c} for c in (0.01, 0.1, 1.0, 10.0)
        ],
    }

    def estimator(name: str, params: dict):
        if name == "class_weighted_rbf_svc":
            # Classification config matches the saved incumbent. The SVC
            # probability option stays at its false default: its internal sample
            # CV is not group aware, and this run does not fit a confidence model.
            return make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", class_weight="balanced", random_state=MODEL_SEED, **params),
            )
        if name == "random_forest_unweighted":
            return RandomForestClassifier(
                n_estimators=300, max_features="sqrt", class_weight=None,
                random_state=MODEL_SEED, n_jobs=-1, **params,
            )
        if name == "random_forest_class_weighted":
            return RandomForestClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                random_state=MODEL_SEED, n_jobs=-1, **params,
            )
        if name == "balanced_random_forest":
            return BalancedRandomForestClassifier(
                n_estimators=300, max_features="sqrt", sampling_strategy="all",
                replacement=False, random_state=MODEL_SEED, n_jobs=-1, **params,
            )
        if name == "logistic_regression_balanced":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", max_iter=5000, solver="lbfgs",
                    random_state=MODEL_SEED, **params,
                ),
            )
        raise KeyError(name)

    return families, estimator


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def aligned_probabilities(model, X: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        return None
    raw = model.predict_proba(X)
    model_classes = list(model.classes_)
    aligned = np.zeros((len(X), len(LABELS)), dtype=float)
    for column, label in enumerate(model_classes):
        if label in LABELS:
            aligned[:, LABELS.index(label)] = raw[:, column]
    return aligned


def calibration_diagnostics(model, X: pd.DataFrame, y_true) -> dict:
    proba = aligned_probabilities(model, X)
    if proba is None:
        return {"available": False, "reason": "SVC probability estimation disabled; no calibration fitted."}
    truth = np.zeros_like(proba)
    label_index = {label: idx for idx, label in enumerate(LABELS)}
    for row, label in enumerate(y_true):
        truth[row, label_index[label]] = 1.0
    try:
        ll = float(log_loss(y_true, proba, labels=LABELS))
    except ValueError:
        ll = None
    return {
        "available": True,
        "validation_multiclass_brier_raw": float(np.mean(np.sum((proba - truth) ** 2, axis=1))),
        "validation_log_loss_raw": ll,
        "note": "Raw validation diagnostics only; no post-hoc calibration was fitted and these are not correctness guarantees.",
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns]
    def render(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(rows)


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Phase 7 artifacts: {OUTPUT_DIR}")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT_PATH}")
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"Required Phase 6 audit is missing: {AUDIT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir()
    (OUTPUT_DIR / "confusion_matrices").mkdir()
    (OUTPUT_DIR / "per_class_reports").mkdir()

    data, exclusion_log, feature_columns, prep_info = load_and_prepare()
    split_data, split_info = create_group_splits(data)
    split_columns = [
        "split", "recording_key", "app_instance_id", "label", "filename",
        "original_filename", "original_file_path", "processed_file_path",
        "format", "content_sha256",
    ]
    split_data[split_columns].to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)
    exclusion_log.to_csv(OUTPUT_DIR / "exclusion_log.csv", index=False)

    split_counts = split_info["splits"]
    if min(split_counts[name]["recordings"] for name in split_counts) <= 0:
        raise ValueError("A split has no recordings.")

    train = split_data[split_data["split"] == "train"]
    validation = split_data[split_data["split"] == "validation"]
    test = split_data[split_data["split"] == "test"]
    development = split_data[split_data["split"].isin(["train", "validation"])]
    X_train, y_train = train[feature_columns], train["label"].astype(str)
    X_val, y_val = validation[feature_columns], validation["label"].astype(str)
    X_test, y_test = test[feature_columns], test["label"].astype(str)

    families, make_estimator = build_candidates()
    fixed_estimator_settings = {
        "class_weighted_rbf_svc": {
            "pipeline": ["StandardScaler", "SVC"], "kernel": "rbf",
            "class_weight": "balanced", "gamma": "scale", "probability": False,
            "random_state": MODEL_SEED,
        },
        "random_forest_unweighted": {
            "estimator": "RandomForestClassifier", "n_estimators": 300,
            "max_features": "sqrt", "class_weight": None,
            "random_state": MODEL_SEED, "n_jobs": -1,
        },
        "random_forest_class_weighted": {
            "estimator": "RandomForestClassifier", "n_estimators": 300,
            "max_features": "sqrt", "class_weight": "balanced",
            "random_state": MODEL_SEED, "n_jobs": -1,
        },
        "balanced_random_forest": {
            "estimator": "BalancedRandomForestClassifier", "n_estimators": 300,
            "max_features": "sqrt", "sampling_strategy": "all",
            "replacement": False, "random_state": MODEL_SEED, "n_jobs": -1,
        },
        "logistic_regression_balanced": {
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "class_weight": "balanced", "solver": "lbfgs", "max_iter": 5000,
            "random_state": MODEL_SEED,
        },
    }
    validation_rows = []
    selected_configs = {}
    validation_models = {}
    for family, grid in families.items():
        candidates = []
        for params in grid:
            model = make_estimator(family, params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            result = metrics(y_val, pred)
            row = {
                "model_family": family,
                "parameters": params,
                **result,
            }
            validation_rows.append(row)
            candidates.append((result["macro_f1"], result["balanced_accuracy"], params, model))
        _, _, best_params, best_model = max(candidates, key=lambda item: (item[0], item[1]))
        selected_configs[family] = best_params
        validation_models[family] = best_model

    validation_df = pd.DataFrame([
        {**row, "parameters": json.dumps(row["parameters"], sort_keys=True)}
        for row in validation_rows
    ])
    validation_df.to_csv(OUTPUT_DIR / "validation_hyperparameter_results.csv", index=False)
    selected_df = validation_df.copy()
    selected_df["params_dict"] = selected_df["parameters"].map(json.loads)
    selected_df = selected_df[
        selected_df.apply(lambda row: row["params_dict"] == selected_configs[row["model_family"]], axis=1)
    ]
    selected_df.drop(columns="params_dict").to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv", index=False
    )

    # The test partition has not been scored or used above. Fit each selected
    # family/config on train+validation, then make a single final test evaluation.
    test_rows = []
    test_predictions = pd.DataFrame({
        "recording_key": test["recording_key"].to_numpy(),
        "app_instance_id": test["app_instance_id"].to_numpy(),
        "true_label": y_test.to_numpy(),
    })
    model_details = {}
    per_class_details = {}
    confusion_details = {}
    for family, params in selected_configs.items():
        model = make_estimator(family, params)
        model.fit(development[feature_columns], development["label"].astype(str))
        pred = model.predict(X_test)
        result = metrics(y_test, pred)
        test_rows.append({"model_family": family, "parameters": json.dumps(params, sort_keys=True), **result,
                          "tired_recall": float(recall_score(y_test, pred, labels=["tired"], average="macro", zero_division=0)),
                          "hungry_recall": float(recall_score(y_test, pred, labels=["hungry"], average="macro", zero_division=0))})
        test_predictions[f"{family}_prediction"] = pred

        report = classification_report(
            y_test, pred, labels=LABELS, output_dict=True, zero_division=0
        )
        per_class = pd.DataFrame([
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        ])
        per_class.to_csv(OUTPUT_DIR / "per_class_reports" / f"{family}.csv", index=False)
        per_class_details[family] = per_class
        matrix = confusion_matrix(y_test, pred, labels=LABELS)
        matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
        matrix_df.to_csv(OUTPUT_DIR / "confusion_matrices" / f"{family}.csv", index_label="actual")
        confusion_details[family] = matrix_df

        val_diagnostics = calibration_diagnostics(
            validation_models[family], X_val, y_val.to_numpy()
        )
        model_details[family] = {
            "parameters": params,
            "validation_metrics": selected_df.loc[selected_df.model_family == family].drop(columns="params_dict").iloc[0].drop(labels=["parameters"]).to_dict(),
            "validation_probability_diagnostics": val_diagnostics,
            "test_metrics": result,
            "tired_recall": float(per_class.loc[per_class["class"] == "tired", "recall"].iloc[0]),
            "hungry_recall": float(per_class.loc[per_class["class"] == "hungry", "recall"].iloc[0]),
        }
        package = {
            "model": model,
            "model_name": f"phase7_{family}",
            "feature_columns": feature_columns,
            "classes": LABELS,
            "configuration": params,
            "random_seed": MODEL_SEED,
            "split_seed": SEED,
            "grouping_unit": "app-instance UUID; recording-key copies are contained within groups",
            "calibrated": False,
            "not_for_production": True,
        }
        joblib.dump(package, OUTPUT_DIR / "models" / f"{family}.joblib")

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTPUT_DIR / "test_model_comparison.csv", index=False)
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    (OUTPUT_DIR / "model_configurations.json").write_text(
        json.dumps({"fixed_estimator_settings": fixed_estimator_settings,
                    "parameter_grids": families, "selected_by_validation": selected_configs,
                    "selection_metric": "validation macro F1; balanced accuracy breaks ties",
                    "model_seed": MODEL_SEED}, indent=2), encoding="utf-8"
    )

    validation_chosen = selected_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    test_macro = test_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    test_balanced = test_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    best_validation = validation_chosen.iloc[0]["model_family"]
    best_test_macro = test_macro.iloc[0]["model_family"]
    best_test_balanced = test_balanced.iloc[0]["model_family"]

    excluded_reasons = exclusion_log["reason"].value_counts().to_dict()
    metadata = {
        "phase": 7,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": True,
        "production_artifact_modified": False,
        "application_modified": False,
        "cough_data_or_pipeline_modified": False,
        "source_feature_file": str(FEATURES_PATH.relative_to(ROOT)),
        "source_metadata_file": str(METADATA_PATH.relative_to(ROOT)),
        "legacy_test_manifest_used_for_exclusion_only": str(OLD_TEST_PATH.relative_to(ROOT)),
        "package_versions": {
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (FEATURES_PATH, METADATA_PATH, OLD_TEST_PATH, AUDIT_PATH)
        },
        "feature_columns": feature_columns,
        "preparation": prep_info,
        "excluded_feature_rows_by_reason": excluded_reasons,
        "split": split_info,
        "selected_hyperparameters": selected_configs,
        "fixed_estimator_settings": fixed_estimator_settings,
        "validation_model_comparison": selected_df.drop(columns="params_dict").to_dict(orient="records"),
        "test_model_comparison": test_rows,
        "best_validation_macro_f1": best_validation,
        "highest_test_macro_f1_descriptive_only": best_test_macro,
        "highest_test_balanced_accuracy_descriptive_only": best_test_balanced,
        "calibration_applied": False,
        "test_used_for_model_selection": False,
        "patient_level_independence_guaranteed": False,
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: float(value) if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )

    report_lines = [
        "# Phase 7 Experimental Cry Model Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "No experimental artifact was deployed. The production model and application were not modified.",
        "",
        "## Dataset and exclusions",
        "",
        f"- Feature rows inspected: {prep_info['feature_rows']}",
        f"- Unique recording keys used: {len(split_data)}",
        f"- Legacy test recording keys excluded from Phase 7: {prep_info['legacy_test_recording_keys_excluded']}",
        f"- Legacy test app-installation groups excluded: {prep_info['legacy_test_app_instance_groups_excluded']}",
        f"- Seven manual-review keys excluded: {prep_info['questionable_keys_excluded']}",
        f"- Exact one-row-per-key selection and quality exclusions: `{json.dumps(excluded_reasons, sort_keys=True)}`",
        "- For each remaining recording key, one feature row was selected by source quality: native 16 kHz CAF first, then 8 kHz WAV, then 8 kHz 3GP. Copies with conflicting labels or exact duplicate payloads would be excluded; none remain in the selected set.",
        "- Two additional malformed `.caf.caf` audio files from the raw directory have no feature/metadata rows and were not considered or labeled.",
        "- Labels are contributor-submitted suspected cry reasons, not verified clinical ground truth.",
        "- Participant IDs are unavailable. Splits group by app-installation UUID (and thus keep every recording key from an installation together), but cannot guarantee patient-level independence.",
        "",
        "The existing `cry_test.csv` recording keys were excluded entirely. The Phase 7 test was not used in model selection or hyperparameter tuning. However, these remaining recordings were present in historical development/training data; this is a new experimental split, not an external prospective test set.",
        "",
        "## Grouped split",
        "",
        f"- Split seed: {SEED}; second-stage validation/test seed: {VALIDATION_SPLIT_SEED}.",
        "- Grouping unit: app-installation UUID; recording keys and app UUIDs have zero overlap across splits.",
    ]
    for name in ("train", "validation", "test"):
        entry = split_counts[name]
        report_lines.append(
            f"- {name.title()}: {entry['recordings']} recordings, {entry['app_installation_groups']} app-installation groups."
        )
        report_lines.append(f"  - Class counts: `{json.dumps(entry['class_counts'], sort_keys=True)}`")
    report_lines += [
        "",
        "Only four `cold_hot` recording keys remain after required exclusions (two in train, one in validation, one in test). Test recall for that class is a one-record observation and is not a reliable estimate. The other minority class metrics are also based on small supports.",
        "",
        "## Validation comparison and configuration selection",
        "",
        "Each model family used training-only fitting and validation macro F1 for configuration selection; validation balanced accuracy broke ties. No test scores were consulted before the configurations were fixed.",
        "",
        markdown_table(validation_chosen, ["model_family", "parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]),
        "",
        "## Final test comparison",
        "",
        "After selection, each chosen configuration was refit on train plus validation and evaluated once on the new test partition. The test table is descriptive; the highest test scores were not used to select a model.",
        "",
        markdown_table(test_df.sort_values("model_family"), ["model_family", "parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "tired_recall", "hungry_recall"]),
        "",
        f"- Highest validation macro F1: **{best_validation}**.",
        f"- Highest test macro F1 (descriptive only): **{best_test_macro}**.",
        f"- Highest test balanced accuracy (descriptive only): **{best_test_balanced}**.",
        "",
        "### Per-class test metrics and confusion matrices",
    ]
    for family in families:
        report_lines += [
            "",
            f"#### {family}",
            "",
            markdown_table(per_class_details[family], ["class", "precision", "recall", "f1", "support"]),
            "",
            "Actual labels are rows; predicted labels are columns.",
            "",
            markdown_table(confusion_details[family].rename_axis("actual").reset_index(), ["actual"] + LABELS),
        ]

    report_lines += [
        "",
        "## Raw probability diagnostics (validation only)",
        "",
        "No calibrator or confidence threshold was fitted. Raw model probabilities are not correctness guarantees. Validation diagnostics use the validation partition after it was also used to choose hyperparameters, so these figures are exploratory and optimistic; the test set was not used for calibration.",
        "",
        "| Model | Validation probability diagnostics |",
        "|---|---|",
    ]
    for family, details in model_details.items():
        diag = details["validation_probability_diagnostics"]
        if diag.get("available"):
            cell = f"Brier={diag['validation_multiclass_brier_raw']:.4f}; log-loss={diag['validation_log_loss_raw']:.4f} (raw, uncalibrated)"
        else:
            cell = diag.get("reason", "Unavailable")
        report_lines.append(f"| {family} | {cell} |")
    report_lines += [
        "",
        "The validation set has only one `cold_hot` recording and limited examples for other minority classes. It is not large enough for defensible per-class calibration. A separate group-held-out calibration set and a new external test cohort are needed before confidence or threshold claims.",
        "",
        "## Imbalance and interpretation",
        "",
        "Class weighting changes error costs; Balanced Random Forest changes each tree's class sampling. Neither creates independent minority information. Compare per-class recall, macro F1, balanced accuracy, and the confusion matrices alongside accuracy. A higher score for one rare class can coincide with more false positives or lower recall elsewhere.",
        "",
        "These experiments use the existing 91 aggregate acoustic features. The five-second preprocessing window, user-tag label uncertainty, small minority supports, app-installation proxy groups, and the lack of patient identifiers remain material limits. No result establishes clinical reliability.",
        "",
        "## Experimental artifacts",
        "",
        "- `models/experiments/cry_phase7/models/*.joblib` — Phase 7-only models fit on train+validation.",
        "- `models/experiments/cry_phase7/split_assignments.csv` — exact selected source files, recording keys, app-installation groups, labels, and split names.",
        "- `models/experiments/cry_phase7/exclusion_log.csv` — row-level exclusion reasons.",
        "- `models/experiments/cry_phase7/model_configurations.json` and `training_metadata.json` — grids, selected settings, seeds, source hashes, metrics, and grouping details.",
        "- `models/experiments/cry_phase7/validation_hyperparameter_results.csv` and `validation_model_comparison.csv`.",
        "- `models/experiments/cry_phase7/test_model_comparison.csv`, `test_predictions.csv`, `per_class_reports/`, and `confusion_matrices/`.",
        "- This report: `training/reports/cry_model_comparison_phase7.md`.",
        "",
        "The models are experimental candidates only. Further work should prioritize manual label review, more independent minority-class recordings, and a prospectively held-out participant cohort before considering any application change.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "selected_unique_recordings": len(split_data),
        "split": split_info,
        "selected_configs": selected_configs,
        "validation_best_macro_f1": best_validation,
        "test_best_macro_f1_descriptive_only": best_test_macro,
        "test_best_balanced_accuracy_descriptive_only": best_test_balanced,
        "artifacts": str(OUTPUT_DIR),
        "report": str(REPORT_PATH),
    }, indent=2, default=lambda value: float(value) if isinstance(value, np.generic) else str(value)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Phase 7 experiment aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
