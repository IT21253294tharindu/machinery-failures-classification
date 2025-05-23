from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Load the trained model and label encoder
with open("./models/machine_failure_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("./models/machine_failure_label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Input schema
class SensorData(BaseModel):
    voltage: float
    current: float
    temperature: float
    vibration: float

app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Machine Failure Classifier is running!"}

# Prediction endpoint
@app.post("/predict")
def predict_failure(data: SensorData):
    try:
        # Prepare input for model
        features = np.array([[data.voltage, data.current, data.temperature, data.vibration]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        return {
            "prediction": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

        
# Run the app with: uvicorn main:app --reload