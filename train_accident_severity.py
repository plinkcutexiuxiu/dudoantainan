"""Train severity classifiers on UK STATS19 collision data.

This estimates severity conditional on a recorded collision, not whether one
will occur or real-world crash risk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

SEED = 42
DATA_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-2023.csv"
)
TARGET_ALIASES = ("accident_severity", "collision_severity", "severity")
SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}

# Only pre-outcome context fields are allowed. IDs, coordinates, casualty/vehicle
# counts, and other post-outcome fields are intentionally excluded.
FEATURE_ALIASES = {
    "weather_conditions": ("weather_conditions", "weather"),
    "road_surface_conditions": ("road_surface_conditions", "road_surface"),
    "light_conditions": ("light_conditions", "light"),
    "speed_limit": ("speed_limit", "speed limit"),
    "day_of_week": ("day_of_week", "day of week"),
    "urban_or_rural_area": ("urban_or_rural_area", "urban_or_rural", "area_type"),
    "road_type": ("road_type",),
    "junction_control": ("junction_control",),
    "first_road_class": ("first_road_class",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Use a local collision CSV instead of downloading.")
    parser.add_argument(
        "--sample-size", type=int, default=60_000,
        help="Stratified laptop-friendly sample size; use 0 for all rows (default: 60000).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in columns}
    return next((lookup[a.casefold()] for a in aliases if a.casefold() in lookup), None)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    raw = raw.copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    target_column = find_column(list(raw.columns), TARGET_ALIASES)
    if target_column is None:
        raise ValueError(f"Missing accident_severity/severity target. Columns: {list(raw.columns)}")

    target = pd.to_numeric(raw[target_column], errors="coerce")
    valid = target.isin(SEVERITY_LABELS)
    raw = raw.loc[valid].copy()
    raw["__target__"] = target.loc[valid].astype(int)
    selected: dict[str, str] = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        source = find_column(list(raw.columns), aliases)
        if source is not None and source != target_column:
            selected[canonical] = source

    time_column = find_column(list(raw.columns), ("time", "accident_time"))
    if time_column is not None:
        hour = pd.to_datetime(raw[time_column].astype("string"), format="%H:%M", errors="coerce").dt.hour
        raw["time_of_day"] = pd.cut(
            hour, bins=[-1, 5, 9, 15, 19, 23],
            labels=["Night", "Morning", "Day", "Evening", "Late night"],
        ).astype("string")
        selected["time_of_day"] = "time_of_day"

    if len(selected) < 3:
        raise ValueError(f"Need at least 3 supported context features; found {list(selected)}")
    x = pd.DataFrame({name: raw[source] for name, source in selected.items()}, index=raw.index)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    selected = {name: source for name, source in selected.items() if name in x.columns}
    for column in x.columns:
        if column == "speed_limit":
            x[column] = pd.to_numeric(x[column], errors="coerce")
        else:
            x[column] = x[column].astype("string").fillna("Unknown")
    y = raw.loc[x.index, "__target__"].astype(int)
    if set(y.unique()) != set(SEVERITY_LABELS):
        raise ValueError("All classes are required: 1=Fatal, 2=Serious, 3=Slight.")
    return x, y, selected


def download_data(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "SafeRoute-student-project/1.0"})
    print(f"Downloading official DfT data to {destination} ...")
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def make_pipeline(estimator, categorical: list[str], numeric: list[str]) -> Pipeline:
    transformers = []
    if categorical:
        categorical_steps = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_steps, categorical))
    if numeric:
        numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_steps, numeric))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", estimator),
    ])


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 0.5 or args.folds < 2 or args.sample_size < 0:
        raise ValueError("Use 0 < test-size < 0.5, folds >= 2, sample-size >= 0.")
    project_dir = Path(__file__).resolve().parent
    csv_path = args.csv or (project_dir / "data" / "dft-collision-2023.csv")
    if not csv_path.exists():
        if args.csv:
            raise FileNotFoundError(csv_path)
        download_data(csv_path)
    x, y, selected = prepare_data(pd.read_csv(csv_path, low_memory=False))
    print(f"Rows: {len(x):,}; features: {list(x.columns)}")
    print("Source columns:", selected)

    if args.sample_size and len(x) > args.sample_size:
        x, _, y, _ = train_test_split(x, y, train_size=args.sample_size, random_state=SEED, stratify=y)
        print(f"Using stratified sample of {len(x):,}; use --sample-size 0 for all rows.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = y.map(SEVERITY_LABELS).value_counts().reindex(["Fatal", "Serious", "Slight"], fill_value=0)
    print("\nClass counts:\n", counts.to_string())
    ax = counts.plot.bar(color=["#b8443c", "#e2a443", "#168780"])
    ax.set(title="Class distribution", xlabel="Accident severity", ylabel="Records")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(args.output_dir / "class_distribution.png", dpi=160)
    plt.close()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    if y_train.value_counts().min() < args.folds:
        raise ValueError("The smallest training class has fewer samples than CV folds.")
    categorical = [column for column in x if column != "speed_limit"]
    numeric = [column for column in x if column == "speed_limit"]
    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=12, min_samples_leaf=5,
            class_weight="balanced", ccp_alpha=0.0, random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
    }
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    pipelines, cv_rows = {}, []
    print(f"\n{args.folds}-fold CV on training data only ...")
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, categorical, numeric)
        scores = cross_validate(
            pipeline, x_train, y_train, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=1, error_score="raise",
        )
        pipelines[name] = pipeline
        cv_rows.append({
            "Model": name,
            "CV macro-F1 mean": scores["test_macro_f1"].mean(),
            "CV macro-F1 std": scores["test_macro_f1"].std(ddof=1),
            "CV balanced accuracy mean": scores["test_balanced_accuracy"].mean(),
            "CV balanced accuracy std": scores["test_balanced_accuracy"].std(ddof=1),
            "Train macro-F1 mean": scores["train_macro_f1"].mean(),
        })
    cv_table = pd.DataFrame(cv_rows).set_index("Model")
    candidates = ["Decision Tree", "Random Forest"]
    best_name = max(candidates, key=lambda name: cv_table.loc[name, "CV macro-F1 mean"])

    test_rows, predictions = [], {}
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        predictions[name] = predicted
        test_rows.append({
            "Model": name,
            "Test accuracy": accuracy_score(y_test, predicted),
            "Test balanced accuracy": balanced_accuracy_score(y_test, predicted),
            "Test macro precision": precision_score(y_test, predicted, average="macro", zero_division=0),
            "Test macro recall": recall_score(y_test, predicted, average="macro", zero_division=0),
            "Test macro-F1": f1_score(y_test, predicted, average="macro", zero_division=0),
            "Test weighted-F1": f1_score(y_test, predicted, average="weighted", zero_division=0),
        })
    results = cv_table.join(pd.DataFrame(test_rows).set_index("Model"))
    results.to_csv(args.output_dir / "model_comparison.csv")
    print("\nCV and final test comparison:\n", results.round(4).to_string())
    print(f"\nSelected by CV macro-F1: {best_name}")

    best, predicted = pipelines[best_name], predictions[best_name]
    order, names = [1, 2, 3], [SEVERITY_LABELS[i] for i in [1, 2, 3]]
    report = classification_report(
        y_test, predicted, labels=order, target_names=names, zero_division=0, output_dict=True
    )
    print("\nClassification report:\n", classification_report(
        y_test, predicted, labels=order, target_names=names, zero_division=0
    ))
    with (args.output_dir / "classification_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    matrix = confusion_matrix(y_test, predicted, labels=order)
    fig, axis = plt.subplots(figsize=(6.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title=f"Confusion matrix: {best_name}", xlabel="Predicted", ylabel="Actual",
             xticks=range(3), xticklabels=names, yticks=range(3), yticklabels=names)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Records")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    try:
        probabilities = best.predict_proba(x_test)
        auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=order)
        print(f"Macro one-vs-rest ROC-AUC: {auc:.4f}")
    except (AttributeError, ValueError) as error:
        print(f"ROC-AUC unavailable: {error}")

    # Post-hoc test-set explanation only; do not use it to tune the model.
    importance = permutation_importance(
        best, x_test, y_test, scoring="f1_macro", n_repeats=3, random_state=SEED, n_jobs=-1
    )
    importance_table = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance mean": importance.importances_mean,
        "Importance std": importance.importances_std,
    }).sort_values("Importance mean", ascending=False)
    importance_table.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    top = importance_table.head(12).sort_values("Importance mean")
    ax = top.plot.barh(x="Feature", y="Importance mean", xerr="Importance std",
                       legend=False, color="#168780", figsize=(8, 5.5))
    ax.set(title="Permutation importance on held-out test", xlabel="Macro-F1 decrease", ylabel="")
    plt.tight_layout()
    plt.savefig(args.output_dir / "permutation_importance.png", dpi=160)
    plt.close()

    errors = x_test.copy()
    errors["actual"] = y_test.map(SEVERITY_LABELS)
    errors["predicted"] = pd.Series(predicted, index=x_test.index).map(SEVERITY_LABELS)
    errors["is_error"] = errors["actual"] != errors["predicted"]
    errors.loc[errors["is_error"]].head(100).to_csv(args.output_dir / "misclassified_examples.csv", index=False)
    print(f"\nTest errors: {int(errors['is_error'].sum()):,}/{len(errors):,}")
    print("Error rate by actual class:\n", errors.groupby("actual", observed=False)["is_error"].agg(
        sample_count="size", error_rate="mean"
    ).round(3).to_string())
    for feature in ("weather_conditions", "road_surface_conditions", "light_conditions", "time_of_day"):
        if feature in errors:
            groups = errors.groupby(feature, dropna=False, observed=False)["is_error"].agg(
                sample_count="size", error_rate="mean"
            )
            groups = groups[groups["sample_count"] >= 30].sort_values("error_rate", ascending=False)
            if not groups.empty:
                print(f"\nHighest error-rate {feature} groups (n >= 30):\n", groups.head(8).round(3).to_string())

    gap = cv_table.loc[best_name, "Train macro-F1 mean"] - cv_table.loc[best_name, "CV macro-F1 mean"]
    print(f"\nTrain-CV macro-F1 gap: {gap:.4f}")
    if gap > 0.10:
        print("Large positive gap suggests possible overfitting; confirm on another year/region.")
    elif cv_table.loc[best_name, "CV macro-F1 mean"] < 0.45:
        print("Low CV score may indicate underfitting or weak features; inspect errors and coding.")
    else:
        print("Compare train and CV scores; a smaller gap is more consistent with generalization.")
    try:
        import joblib
        joblib.dump(best, args.output_dir / "best_model.joblib")
    except ImportError:
        print("Install joblib to export the selected model.")
    print(f"\nArtifacts saved to {args.output_dir.resolve()}")
    print("Use generated CV mean/std and final test metrics in the report; do not invent results.")


if __name__ == "__main__":
    main()