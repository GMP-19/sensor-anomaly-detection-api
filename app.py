import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).parent
api_model = joblib.load(PROJECT_DIR / "isolation_forest_model.joblib")
metadata = json.loads((PROJECT_DIR / "metadata.json").read_text())
FEATURES = metadata["feature_columns"]

app = FastAPI(
    title="Sensor Anomaly Detection API",
    description="Detect unusual sensor observations using an Isolation Forest model trained on NASA C-MAPSS FD001.",
    version="1.0.0"
)

class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[float]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(FEATURES)
    }

@app.post("/predict")
def predict(data: SensorInput):
    if len(data.values) != len(FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FEATURES)} feature values."
        )

    row = pd.DataFrame([data.values], columns=FEATURES)
    prediction = int(api_model.predict(row)[0])
    score = float(api_model.decision_function(row)[0])

    return {
        "prediction": "anomaly" if prediction == -1 else "normal",
        "prediction_code": prediction,
        "anomaly_score": score,
        "features_used": len(FEATURES)
    }
