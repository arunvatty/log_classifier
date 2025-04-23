from fastapi import APIRouter
from pydantic import BaseModel
from app.predict import predict

router = APIRouter()

class LogInput(BaseModel):
    log: str

class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    encoded_length: int = None

@router.post("/predict")
def predict_log(input: LogInput):
    result = predict(input.log)
    
    # Make sure the prediction is either "Important" or "Not Important"
    # This standardizes the API response format
    if result["prediction"] == "Normal":
        result["prediction"] = "Not Important"
        
    return result