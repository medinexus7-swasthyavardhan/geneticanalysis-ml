from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Genetic Analysis ML API",
    description="Prototype API for SNP-based trait prediction",
    version="1.0.0"
)

# --------------------------------------------------
# Load models
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {}

for trait in [
    "height",
    "strength",
    "power",
    "endurance",
    "auditory",
    "neuromotor"
]:
    path = os.path.join(BASE_DIR, f"{trait}_snp_model.pkl")

    loaded = joblib.load(path)

    # Height model was saved as the raw sklearn model.
    # The other five were saved as dictionaries.
    if isinstance(loaded, dict):
        models[trait] = loaded["model"]
    else:
        models[trait] = loaded


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class SNPInput(BaseModel):

    # Height
    HEIGHT_SNP_01: int
    HEIGHT_SNP_02: int
    HEIGHT_SNP_03: int
    HEIGHT_SNP_04: int
    HEIGHT_SNP_05: int
    HEIGHT_SNP_06: int

    # Strength
    STRENGTH_SNP_01: int
    STRENGTH_SNP_02: int
    STRENGTH_SNP_03: int
    STRENGTH_SNP_04: int
    STRENGTH_SNP_05: int
    STRENGTH_SNP_06: int

    # Power
    POWER_SNP_01: int
    POWER_SNP_02: int
    POWER_SNP_03: int
    POWER_SNP_04: int

    # Endurance
    ENDURANCE_SNP_01: int
    ENDURANCE_SNP_02: int
    ENDURANCE_SNP_03: int
    ENDURANCE_SNP_04: int
    ENDURANCE_SNP_05: int
    ENDURANCE_SNP_06: int

    # Auditory
    AUDITORY_SNP_01: int
    AUDITORY_SNP_02: int
    AUDITORY_SNP_03: int
    AUDITORY_SNP_04: int

    # Neuromotor
    NEUROMOTOR_SNP_01: int
    NEUROMOTOR_SNP_02: int
    NEUROMOTOR_SNP_03: int
    NEUROMOTOR_SNP_04: int


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: SNPInput):

    values = data.model_dump()

    predictions = {}

    # Feature lists must match training order
    feature_map = {
        "height": [
            "HEIGHT_SNP_01", "HEIGHT_SNP_02", "HEIGHT_SNP_03",
            "HEIGHT_SNP_04", "HEIGHT_SNP_05", "HEIGHT_SNP_06"
        ],

        "strength": [
            "STRENGTH_SNP_01", "STRENGTH_SNP_02", "STRENGTH_SNP_03",
            "STRENGTH_SNP_04", "STRENGTH_SNP_05", "STRENGTH_SNP_06"
        ],

        "power": [
            "POWER_SNP_01", "POWER_SNP_02",
            "POWER_SNP_03", "POWER_SNP_04"
        ],

        "endurance": [
            "ENDURANCE_SNP_01", "ENDURANCE_SNP_02", "ENDURANCE_SNP_03",
            "ENDURANCE_SNP_04", "ENDURANCE_SNP_05", "ENDURANCE_SNP_06"
        ],

        "auditory": [
            "AUDITORY_SNP_01", "AUDITORY_SNP_02",
            "AUDITORY_SNP_03", "AUDITORY_SNP_04"
        ],

        "neuromotor": [
            "NEUROMOTOR_SNP_01", "NEUROMOTOR_SNP_02",
            "NEUROMOTOR_SNP_03", "NEUROMOTOR_SNP_04"
        ]
    }

    for trait, features in feature_map.items():

        input_values = [[values[f] for f in features]]

        prediction = models[trait].predict(input_values)[0]

        predictions[f"{trait}_score"] = round(float(prediction), 2)

    return {
        "status": "success",
        "predictions": predictions
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Genetic Analysis ML API is running",
        "models": list(models.keys())
    }