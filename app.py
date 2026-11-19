import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path

MODEL_PATH = Path("models/maintenance_bundle.joblib")
app = FastAPI(title="Predictive Maintenance API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class MachineReading(BaseModel):
    Type: str = Field("M", description="L, M, or H")
    air_temperature: float = Field(300.0, alias="Air temperature [K]")
    process_temperature: float = Field(310.0, alias="Process temperature [K]")
    rotational_speed: float = Field(1500.0, alias="Rotational speed [rpm]")
    torque: float = Field(40.0, alias="Torque [Nm]")
    tool_wear: float = Field(100.0, alias="Tool wear [min]")

    def to_frame(self, features):
        values = {
            "Type": self.Type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear
        }
        return pd.DataFrame([{c: values[c] for c in features}])

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running", "model_loaded": bundle is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(reading: MachineReading):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model missing. Run python train.py first.")
    try:
        X = reading.to_frame(bundle["features"])
        model = bundle["failure_model"]
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        rul = max(0.0, float(bundle["rul_model"].predict(X)[0]))
        status = "High Risk" if probability >= 0.5 else ("Monitor" if probability >= 0.2 else "Healthy")
        return {
            "failure_prediction": prediction,
            "failure_probability": round(probability, 4),
            "risk_status": status,
            "estimated_tool_life_remaining_minutes": round(rul, 2),
            "classifier": bundle["best_classifier"],
            "note": "Remaining life is a tool-wear proxy, not ground-truth machine RUL."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    import json
    path = Path("models/metrics.json")
    return json.loads(path.read_text()) if path.exists() else {"message": "Train models first."}
