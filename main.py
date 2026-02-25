from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load ML model
model = joblib.load("model.pkl")

# Request body model
class UserInput(BaseModel):
    name: str
    age: int
    height: float
    weight: float

@app.get("/")
def home():
    return {"message": "PCOS Risk Prediction API is running"}

@app.post("/predict")
def predict(data: UserInput):
    height_m = data.height / 100
    bmi = data.weight / (height_m ** 2)

    input_data = np.array([[data.age, bmi]])
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        risk = "LOW"
    elif prediction == 1:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "name": data.name,
        "bmi": round(bmi, 2),
        "risk": risk
    }