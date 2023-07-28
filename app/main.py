# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the pre-trained ML model
model = joblib.load("../model.joblib")

# Define the FastAPI app
app = FastAPI()

# Create a Pydantic model to validate the input data
class InputData(BaseModel):
    GRE_Score: float
    TOEFL_Score: float
    University_Rating: float
    SOP: float
    LOR: float
    CGPA: float
    Research: int


@app.get("/")
def read_root():
    return {"message": "Welcome to PandasPython World!"}

# Endpoint to make predictions
@app.post("/predict/")
def predict(input_data: InputData):
    try:
        
        data_dict = input_data.dict()
        data_dict = pd.DataFrame(input_data.dict(), index=[0])
        
        # Make predictions using the pre-trained model
        prediction = model.predict(data_dict)
        
        return {"prediction": prediction[0]}

    except Exception as e:
        # Return an error message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))
