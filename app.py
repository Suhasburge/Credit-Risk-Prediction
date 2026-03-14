from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import os
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Credit Risk Prediction API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- GLOBAL MODEL ----------------
pipeline = None


@app.on_event("startup")
def load_model():
    """
    This runs when container starts.
    Prevents crash loops in Docker/Kubernetes.
    """
    global pipeline
    print("Loading model into memory...")
    pipeline = PredictionPipeline()
    print("Model loaded successfully!")


# ---------------- DEBUG HANDLER ----------------
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------- INPUT SCHEMA ----------------
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


# ---------------- LOGGING ----------------
PRED_LOG = "artifacts/predictions/predictions.jsonl"
os.makedirs(os.path.dirname(PRED_LOG), exist_ok=True)


def save_prediction(data, prediction):
    record = {
        "timestamp": datetime.now().isoformat(),
        "input": data,
        "prediction": int(prediction)
    }
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
def predict(data: LoanApplication):

    global pipeline

    if pipeline is None:
        return {"error": "Model not loaded"}

    # Convert request to dict
    input_data = data.dict()

    # Convert dict to DataFrame (required by preprocessing pipeline)
    df = pd.DataFrame([input_data])

    # Predict
    result = pipeline.predict(df)

    prediction = int(result[0]) if hasattr(result, "__iter__") else int(result)

    # Save prediction log
    save_prediction(input_data, prediction)

    return {"prediction": prediction}