"""Train severity models and generate SafeRoute analytics from the Kaggle-linked US Accidents sample."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import gdown
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
KAGGLE_PAGE = "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"
SAMPLE_DRIVE_ID = "1U3u8QYzLjnEaSurtZfSAS_oh9AT2Mn8X"
SEVERITY_LABELS = {
    1: "Level 1 (lowest traffic impact)",
    2: "Level 2",
    3: "Level 3",
    4: "Level 4 (highest traffic impact)",
}
DATA_COLUMNS = [
    "Severity", "Start_Time", "Start_Lat", "Start_Lng", "City", "County", "State",
    "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)",
    "Precipitation(in)", "Weather_Condition", "Sunrise_Sunset", "Amenity", "Crossing",
    "Junction", "Stop", "Traffic_Signal", "Traffic_Calming",
]
CATEGORICAL_FEATURES = [
    "weather_condition", "sunrise_sunset", "state", "day_of_week",
    "amenity", "crossing", "junction", "stop", "traffic_signal", "traffic_calming",
]
NUMERIC_FEATURES = [
    "hour", "temperature_f", "humidity_pct", "visibility_mi",
    "wind_speed_mph", "precipitation_in",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Use an extracted US Accidents CSV.")
    parser.add_argument("--sample-size", type=int, default=60_000,
                        help="Stratified training sample size; 0 uses all available rows.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def dataset_path(project_dir: Path, override: Path | None) -> Path:
    if override:
        if not override.exists():
            raise FileNotFoundError(override)
        return override
    csv_path = project_dir / "data" / "us-accidents-sampled" / "US_Accidents_March23_sampled_500k.csv"
    if csv_path.exists():
        return csv_path
    data_dir = project_dir / "data" / "us-accidents-sampled"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = project_dir / "data" / "us-accidents-sampled.zip"
    if not archive_path.exists():
        gdown.download(id=SAMPLE_DRIVE_ID, output=str(archive_path), quiet=False)
    with zipfile.ZipFile(archive_path) as archive:
        csv_member = next((name for name in archive.namelist() if name.endswith(".csv") and Path(name).name == name), None)
        if csv_member is None:
            raise ValueError("The Kaggle-linked archive contains no root-level CSV file.")
        archive.extract(csv_member, data_dir)
    return data_dir / csv_member


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = sorted(set(DATA_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"US Accidents CSV is missing columns: {missing_columns}")
    target = pd.to_numeric(raw["Severity"], errors="coerce")
    valid = target.isin(SEVERITY_LABELS)
    raw = raw.loc[valid]
    y = target.loc[valid].astype(int)
    start = pd.to_datetime(raw["Start_Time"], errors="coerce", utc=True)

    x = pd.DataFrame(index=raw.index)
    x["weather_condition"] = raw["Weather_Condition"]
    x["sunrise_sunset"] = raw["Sunrise_Sunset"]
    x["state"] = raw["State"]
    x["day_of_week"] = start.dt.day_name()
    x["hour"] = start.dt.hour
    numeric_sources = {
        "temperature_f": "Temperature(F)",
        "humidity_pct": "Humidity(%)",
        "visibility_mi": "Visibility(mi)",
        "wind_speed_mph": "Wind_Speed(mph)",
        "precipitation_in": "Precipitation(in)",
    }
    for feature, source in numeric_sources.items():
        x[feature] = pd.to_numeric(raw[source], errors="coerce")
    boolean_sources = {
        "amenity": "Amenity", "crossing": "Crossing", "junction": "Junction",
        "stop": "Stop", "traffic_signal": "Traffic_Signal", "traffic_calming": "Traffic_Calming",
    }
    for feature, source in boolean_sources.items():
        x[feature] = raw[source].map({True: "Yes", False: "No"}).fillna("Unknown")
    for column in CATEGORICAL_FEATURES:
        x[column] = x[column].astype("string").fillna("Unknown")
    x = x.replace([np.inf, -np.inf], np.nan)
    present_classes = set(y.unique())
    if present_classes != set(SEVERITY_LABELS):
        raise ValueError(f"Expected severity classes 1–4; found {sorted(present_classes)}.")
    return x[FEATURE_COLUMNS], y


def make_pipeline(estimator) -> Pipeline:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessing = ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def create_analytics(raw: pd.DataFrame, output_dir: Path) -> dict:
    start = pd.to_datetime(raw["Start_Time"], errors="coerce", utc=True)
    enriched = raw.assign(
        month=start.dt.strftime("%Y-%m"),
        hour=start.dt.hour,
        weather=raw["Weather_Condition"].fillna("Unknown"),
    )
    class_counts = raw["Severity"].value_counts().sort_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#f7f9fc")
    palette = {1: "#177e89", 2: "#2a9d8f", 3: "#e9c46a", 4: "#e76f51"}
    axes[0, 0].bar([str(SEVERITY_LABELS[int(k)]) for k in class_counts.index], class_counts.values,
                   color=[palette[int(k)] for k in class_counts.index])
    axes[0, 0].set_title("Severity mix · 500k sample")
    axes[0, 0].tick_params(axis="x", labelrotation=15)
    monthly = enriched.groupby(["month", "Severity"], observed=False).size().unstack(fill_value=0)
    monthly.plot(ax=axes[0, 1], color=[palette[int(k)] for k in monthly.columns], linewidth=1.8)
    axes[0, 1].set_title("Incidents by month")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=35)
    hourly = enriched.groupby(["hour", "Severity"], observed=False).size().unstack(fill_value=0)
    hourly.plot(ax=axes[1, 0], color=[palette[int(k)] for k in hourly.columns], linewidth=1.8)
    axes[1, 0].set_title("Start time by hour")
    axes[1, 0].set_xlabel("Hour of day")
    states = raw["State"].value_counts().head(10).sort_values()
    axes[1, 1].barh(states.index, states.values, color="#168780")
    axes[1, 1].set_title("Top states by incident count")
    for axis in axes.flat:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    fig.tight_layout(pad=2.2)
    fig.savefig(output_dir / "accident_analytics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    profile = {
        "source": "US Accidents (2016–2023), Kaggle author-linked 500,000-row sample",
        "kaggle_url": KAGGLE_PAGE,
        "license": "CC BY-NC-SA 4.0; academic/non-commercial use only",
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "missing_values": int(raw[DATA_COLUMNS].isna().sum().sum()),
        "states": int(raw["State"].nunique()),
        "cities": int(raw["City"].nunique()),
        "date_start": str(start.min().date()),
        "date_end": str(start.max().date()),
        "classes": {str(SEVERITY_LABELS[int(code)]): int(count) for code, count in class_counts.items()},
        "top_weather": raw["Weather_Condition"].value_counts().head(10).to_dict(),
        "top_states": raw["State"].value_counts().head(10).to_dict(),
    }
    with (output_dir / "dataset_profile.json").open("w", encoding="utf-8") as output:
        json.dump(profile, output, ensure_ascii=False, indent=2)
    create_accident_map(raw, output_dir / "accident_map.html")
    return profile


def create_accident_map(raw: pd.DataFrame, destination: Path) -> None:
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    locations = raw[["Start_Lat", "Start_Lng", "Severity", "City", "State"]].dropna(
        subset=["Start_Lat", "Start_Lng"]
    )
    heat_sample = locations.sample(n=min(7000, len(locations)), random_state=SEED)
    center = [float(locations["Start_Lat"].median()), float(locations["Start_Lng"].median())]
    map_view = folium.Map(location=center, zoom_start=4, tiles="OpenStreetMap", control_scale=True)
    heat_layer = folium.FeatureGroup(name="Mật độ sự cố")
    HeatMap(heat_sample[["Start_Lat", "Start_Lng"]].values.tolist(), radius=8, blur=10).add_to(heat_layer)
    heat_layer.add_to(map_view)
    cluster = MarkerCluster(name="Điểm mẫu theo severity", disableClusteringAtZoom=8)
    for severity, group in locations.groupby("Severity", observed=False):
        points = group.sample(n=min(65, len(group)), random_state=SEED)
        color = {1: "#177e89", 2: "#2a9d8f", 3: "#e9c46a", 4: "#e76f51"}[int(severity)]
        for row in points.itertuples(index=False):
            popup = f"Severity {int(severity)} · {row.City or 'Unknown'}, {row.State or ''}"
            folium.CircleMarker(
                location=[row.Start_Lat, row.Start_Lng], radius=4, color=color,
                fill=True, fill_opacity=0.78, popup=popup,
            ).add_to(cluster)
    cluster.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.save(destination)


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 0.5 or args.folds < 2 or args.sample_size < 0:
        raise ValueError("Use 0 < test-size < 0.5, folds >= 2, and sample-size >= 0.")
    project_dir = Path(__file__).resolve().parent
    csv_path = dataset_path(project_dir, args.csv)
    raw = pd.read_csv(csv_path, usecols=DATA_COLUMNS, low_memory=False)
    x, y = prepare_data(raw)
    print(f"Rows: {len(x):,}; features: {list(x.columns)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = create_analytics(raw, args.output_dir)
    print(f"Dataset: {profile['source']}; states={profile['states']}; date={profile['date_start']}–{profile['date_end']}")

    if args.sample_size and len(x) > args.sample_size:
        x, _, y, _ = train_test_split(x, y, train_size=args.sample_size, random_state=SEED, stratify=y)
        print(f"Using a stratified sample of {len(x):,}; set --sample-size 0 for all rows.")
    counts = y.map(SEVERITY_LABELS).value_counts().reindex(list(SEVERITY_LABELS.values()), fill_value=0)
    print("\nSample class counts:\n", counts.to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    if y_train.value_counts().min() < args.folds:
        raise ValueError("The smallest training class has fewer samples than CV folds.")
    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=14, min_samples_leaf=8,
            class_weight="balanced", ccp_alpha=0.0, random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_features="sqrt", min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
    }
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    pipelines, cv_rows = {}, []
    print(f"\n{args.folds}-fold cross-validation on training data only ...")
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
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
    order = [1, 2, 3, 4]
    names = [SEVERITY_LABELS[label] for label in order]
    report = classification_report(y_test, predicted, labels=order, target_names=names,
                                   zero_division=0, output_dict=True)
    print("\nClassification report:\n", classification_report(
        y_test, predicted, labels=order, target_names=names, zero_division=0
    ))
    with (args.output_dir / "classification_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    matrix = confusion_matrix(y_test, predicted, labels=order)
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="YlOrRd")
    axis.set(title=f"Confusion matrix · {best_name}", xlabel="Predicted", ylabel="Actual",
             xticks=range(4), xticklabels=[f"Level {label}" for label in order],
             yticks=range(4), yticklabels=[f"Level {label}" for label in order])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, label="Incidents")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    try:
        probabilities = best.predict_proba(x_test)
        auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=order)
        print(f"Macro one-vs-rest ROC-AUC: {auc:.4f}")
    except (AttributeError, ValueError) as error:
        print(f"ROC-AUC unavailable: {error}")

    importance_sample = x_test.sample(n=min(3000, len(x_test)), random_state=SEED)
    importance = permutation_importance(
        best, importance_sample, y_test.loc[importance_sample.index], scoring="f1_macro",
        n_repeats=3, random_state=SEED, n_jobs=-1,
    )
    importance_table = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance mean": importance.importances_mean,
        "Importance std": importance.importances_std,
    }).sort_values("Importance mean", ascending=False)
    importance_table.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    top = importance_table.head(12).sort_values("Importance mean")
    axis = top.plot.barh(x="Feature", y="Importance mean", xerr="Importance std",
                         legend=False, color="#168780", figsize=(8, 6))
    axis.set(title="Permutation importance on held-out test sample", xlabel="Macro-F1 decrease", ylabel="")
    plt.tight_layout()
    plt.savefig(args.output_dir / "permutation_importance.png", dpi=170, bbox_inches="tight")
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
    for feature in ("weather_condition", "state", "sunrise_sunset", "day_of_week"):
        groups = errors.groupby(feature, dropna=False, observed=False)["is_error"].agg(
            sample_count="size", error_rate="mean"
        )
        groups = groups[groups["sample_count"] >= 30].sort_values("error_rate", ascending=False)
        if not groups.empty:
            print(f"\nHighest error-rate {feature} groups (n >= 30):\n", groups.head(8).round(3).to_string())
    gap = cv_table.loc[best_name, "Train macro-F1 mean"] - cv_table.loc[best_name, "CV macro-F1 mean"]
    print(f"\nTrain-CV macro-F1 gap: {gap:.4f}")
    if gap > 0.10:
        print("Large positive gap suggests possible overfitting; validate on a later period or unseen states.")
    joblib_path = args.output_dir / "best_model.joblib"
    import joblib
    joblib.dump(best, joblib_path)
    print(f"\nArtifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()