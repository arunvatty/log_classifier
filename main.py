from fastapi import FastAPI
from app.api import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Log Classifier API is running", "endpoints": ["/predict"]}