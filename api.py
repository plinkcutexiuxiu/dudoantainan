"""Local SafeRoute API and static demo routes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "best_model.joblib"
RESULTS_PATH = PROJECT_DIR / "artifacts" / "model_comparison.csv"
CLASS_REPORT_PATH = PROJECT_DIR / "artifacts" / "classification_report.json"
DATA_PATH = PROJECT_DIR / "data" / "dft-collision-2023.csv"
KAGGLE_DATA_PATH = PROJECT_DIR / "data" / "us-accidents-sampled" / "US_Accidents_March23_sampled_500k.csv"
PROFILE_PATH = PROJECT_DIR / "artifacts" / "dataset_profile.json"
MAP_PATH = PROJECT_DIR / "artifacts" / "accident_map.html"
LABELS = {
    1: "Level 1 · thấp nhất",
    2: "Level 2",
    3: "Level 3",
    4: "Level 4 · cao nhất",
}
PLOT_FILES = {"class_distribution.png", "accident_analytics.png", "confusion_matrix.png", "permutation_importance.png"}

app = FastAPI(title="SafeRoute API", version="1.0.0")


class PredictionInput(BaseModel):
    weather_condition: str = Field(min_length=1, max_length=80)
    sunrise_sunset: Literal["Day", "Night", "Unknown"]
    state: str = Field(min_length=2, max_length=80)
    day_of_week: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Unknown"]
    amenity: Literal["Yes", "No", "Unknown"]
    crossing: Literal["Yes", "No", "Unknown"]
    junction: Literal["Yes", "No", "Unknown"]
    stop: Literal["Yes", "No", "Unknown"]
    traffic_signal: Literal["Yes", "No", "Unknown"]
    traffic_calming: Literal["Yes", "No", "Unknown"]
    hour: int = Field(ge=0, le=23)
    temperature_f: float = Field(ge=-100, le=150)
    humidity_pct: float = Field(ge=0, le=100)
    visibility_mi: float = Field(ge=0, le=100)
    wind_speed_mph: float = Field(ge=0, le=200)
    precipitation_in: float = Field(ge=0, le=50)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa có model; hãy chạy train_accident_severity.py.")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Không tải được model: {error}") from error


@app.get("/api/health")
def health() -> dict:
    model = load_model()
    return {"status": "ready", "model": type(model.named_steps["model"]).__name__}


@app.get("/api/dataset")
def dataset_summary() -> dict:
    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Chưa có dataset profile; hãy chạy trainer Kaggle.")
    import json

    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


@app.get("/api/evaluation")
def evaluation() -> dict:
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Chưa có kết quả đánh giá.")
    results = pd.read_csv(RESULTS_PATH).astype(object)
    results = results.where(pd.notna(results), None)
    return {"models": results.to_dict(orient="records")}


@app.get("/api/classification-report")
def classification_report_data() -> dict:
    if not CLASS_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo theo lớp; hãy chạy lại huấn luyện.")
    import json

    return json.loads(CLASS_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/api/analytics/map")
def accident_map() -> FileResponse:
    if not MAP_PATH.exists():
        raise HTTPException(status_code=404, detail="Chưa có bản đồ; hãy chạy trainer Kaggle.")
    return FileResponse(MAP_PATH, media_type="text/html")


@app.get("/api/artifacts/{filename}")
def artifact(filename: str) -> FileResponse:
    if filename not in PLOT_FILES:
        raise HTTPException(status_code=404, detail="Không tìm thấy biểu đồ.")
    path = PROJECT_DIR / "artifacts" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có biểu đồ; hãy chạy huấn luyện.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict")
def predict(payload: PredictionInput) -> dict:
    model = load_model()
    frame = pd.DataFrame([payload.model_dump()])
    expected = list(getattr(model, "feature_names_in_", frame.columns))
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Thiếu đặc trưng: {missing}")
    frame = frame[expected]
    predicted = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    scores = {
        LABELS.get(int(code), str(code)): round(float(probability), 4)
        for code, probability in zip(model.classes_, probabilities)
    }
    return {
        "severity": predicted,
        "label": LABELS.get(predicted, "Không xác định"),
        "top_score": round(float(max(probabilities)), 4),
        "class_scores": scores,
        "model": type(model.named_steps["model"]).__name__,
        "scope": "Mức độ ảnh hưởng giao thông của một sự cố đã được ghi nhận; không phải xác suất sự cố xảy ra.",
    }


@app.get("/")
def demo_entry() -> Response:
    if os.getenv("SAFEROUTE_PUBLIC_DEMO") == "1":
        return RedirectResponse("/dashboard.html?demo=1&public=1")
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/dashboard.html")
def dashboard() -> FileResponse:
    return FileResponse(PROJECT_DIR / "dashboard.html")


@app.get("/index.html")
def login_page() -> FileResponse:
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/supabase-client.js")
def supabase_client() -> FileResponse:
    client_path = PROJECT_DIR / "supabase-client.js"
    if not client_path.exists():
        raise HTTPException(status_code=404, detail="Supabase client chưa được cấu hình.")
    return FileResponse(client_path, media_type="text/javascript")