from fastapi import APIRouter
from pydantic import BaseModel
from app.predict import predict

router = APIRouter()

class LogInput(BaseModel):
    log: str

@router.post("/predict")
def predict_log(input: LogInput):
    result = predict(input.log)
    return {"prediction": result}
