import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("swasthya_vardhan_genetic_prototype_10000.csv")


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

CONFIG = {
    "Height": {
        "model": "height_snp_model.pkl",
        "target": "height_score",
        "snps": [
            "HEIGHT_SNP_01", "HEIGHT_SNP_02", "HEIGHT_SNP_03",
            "HEIGHT_SNP_04", "HEIGHT_SNP_05", "HEIGHT_SNP_06"
        ]
    },

    "Strength": {
        "model": "strength_snp_model.pkl",
        "target": "strength_score",
        "snps": [
            "STRENGTH_SNP_01", "STRENGTH_SNP_02", "STRENGTH_SNP_03",
            "STRENGTH_SNP_04", "STRENGTH_SNP_05", "STRENGTH_SNP_06"
        ]
    },

    "Power": {
        "model": "power_snp_model.pkl",
        "target": "power_score",
        "snps": [
            "POWER_SNP_01", "POWER_SNP_02",
            "POWER_SNP_03", "POWER_SNP_04"
        ]
    },

    "Endurance": {
        "model": "endurance_snp_model.pkl",
        "target": "endurance_score",
        "snps": [
            "ENDURANCE_SNP_01", "ENDURANCE_SNP_02", "ENDURANCE_SNP_03",
            "ENDURANCE_SNP_04", "ENDURANCE_SNP_05", "ENDURANCE_SNP_06"
        ]
    },

    "Auditory": {
        "model": "auditory_snp_model.pkl",
        "target": "auditory_score",
        "snps": [
            "AUDITORY_SNP_01", "AUDITORY_SNP_02",
            "AUDITORY_SNP_03", "AUDITORY_SNP_04"
        ]
    },

    "Neuromotor": {
        "model": "neuromotor_snp_model.pkl",
        "target": "neuromotor_score",
        "snps": [
            "NEUROMOTOR_SNP_01", "NEUROMOTOR_SNP_02",
            "NEUROMOTOR_SNP_03", "NEUROMOTOR_SNP_04"
        ]
    }
}


# --------------------------------------------------
# Evaluate each model
# --------------------------------------------------

results = []

for trait, config in CONFIG.items():

    model_data = joblib.load(config["model"])

    # Saved files contain a dictionary
    if isinstance(model_data, dict):
        if "model" in model_data:
            model = model_data["model"]
        else:
            model = next(
                value for value in model_data.values()
                if hasattr(value, "predict")
            )
    else:
        model = model_data

    X = df[config["snps"]]
    y = df[config["target"]]

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    results.append({
        "Trait": trait,
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    })


# --------------------------------------------------
# Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("GENETIC ANALYSIS ML MODEL EVALUATION")
print("=" * 60)

print(results_df.to_string(index=False))

print("=" * 60)