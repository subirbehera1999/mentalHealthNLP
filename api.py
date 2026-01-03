# import required libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from utils.text_cleaner import TextCleanerTFIDF
from typing import List


# App initialization 
app = FastAPI(
    title="Mental Health Text Classification API",
    description="Predicts Anxiety, Depression, Normal, or Suicidal from text",
    version="1.0.0"
)

# load model 
model = joblib.load("model/model.pkl")

# Request Schema
class TextRequest(BaseModel):
    text: str


# api check 
@app.get("/")
def health_check():
    return {"status": "API is running"}

# prediction endpoint
@app.post("/predict")
def predict_text(request: TextRequest):
    # Convert input to DataFrame
    input_df = pd.DataFrame({"text": [request.text]})
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}

# prediction endpoint for multiple records
@app.post("/predict-batch")
def predict_text_batch(request: List[TextRequest]):
    texts = [item.text for item in request]
    # Convert input to DataFrame
    input_df = pd.DataFrame({"text": texts})
    predictions = model.predict(input_df)
    return {"prediction": list(predictions)}
