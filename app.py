# Install: pip install fastapi uvicorn pydantic joblib scikit-learn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Load the Best Model
try:
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError("best_model.pkl not found. Run mlflow_tracking.py first.")

# 2. Define the Request Body Structure
# Corresponds to the four features of the Iris dataset
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 3. Initialize FastAPI
app = FastAPI(title="Iris Classifier API")

# 4. Define the Prediction Endpoint
@app.post("/predict/")
async def predict_iris(features: IrisFeatures):
    # Convert Pydantic model to a DataFrame row for the scikit-learn model
    data = features.model_dump()
    X_new = pd.DataFrame([data]) 
    
    # Make prediction
    prediction_proba = model.predict_proba(X_new)[0].tolist()
    prediction_label = int(model.predict(X_new)[0])
    
    # Map label to class name (for Iris dataset)
    target_names = ['setosa', 'versicolor', 'virginica']
    predicted_class = target_names[prediction_label]

    return {
        "model_used": type(model).__name__,
        "prediction_label": prediction_label,
        "predicted_class": predicted_class,
        "probabilities": {name: round(proba, 4) for name, proba in zip(target_names, prediction_proba)}
    }

# To run the API: Open terminal and type `uvicorn app:app --reload`