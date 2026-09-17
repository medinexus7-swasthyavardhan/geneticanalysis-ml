import joblib
import numpy as np

# Load trained models
def load_model(filename):
    obj = joblib.load(filename)

    # Our saved .pkl files contain a dictionary
    if isinstance(obj, dict):
        if "model" in obj:
            return obj["model"]

        # Find the estimator automatically
        for value in obj.values():
            if hasattr(value, "predict"):
                return value

    # If the file itself is already a model
    if hasattr(obj, "predict"):
        return obj

    raise TypeError(f"Could not find a trained model in {filename}")


MODELS = {
    "height": load_model("height_snp_model.pkl"),
    "strength": load_model("strength_snp_model.pkl"),
    "power": load_model("power_snp_model.pkl"),
    "endurance": load_model("endurance_snp_model.pkl"),
    "auditory": load_model("auditory_snp_model.pkl"),
    "neuromotor": load_model("neuromotor_snp_model.pkl"),
}

# SNP columns required by each model
SNP_COLUMNS = {
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
    ],
}


def predict_all_traits(genotype_data):
    """
    Predict all six phenotype-associated scores.

    genotype_data:
        Dictionary containing SNP names and genotype values.

    Returns:
        Dictionary containing predictions for all six traits.
    """

    predictions = {}

    for trait, model in MODELS.items():

        columns = SNP_COLUMNS[trait]

        # Check that all required SNPs are present
        missing = [col for col in columns if col not in genotype_data]

        if missing:
            raise ValueError(
                f"Missing SNPs for {trait}: {missing}"
            )

        # Preserve the exact training-column order
        X = np.array([[genotype_data[col] for col in columns]])

        prediction = model.predict(X)[0]

        predictions[trait] = round(float(prediction), 2)

    return predictions


if __name__ == "__main__":

    # Example genotype input
    example = {
        "HEIGHT_SNP_01": 1,
        "HEIGHT_SNP_02": 0,
        "HEIGHT_SNP_03": 0,
        "HEIGHT_SNP_04": 0,
        "HEIGHT_SNP_05": 1,
        "HEIGHT_SNP_06": 0,

        "STRENGTH_SNP_01": 1,
        "STRENGTH_SNP_02": 0,
        "STRENGTH_SNP_03": 1,
        "STRENGTH_SNP_04": 0,
        "STRENGTH_SNP_05": 1,
        "STRENGTH_SNP_06": 0,

        "POWER_SNP_01": 1,
        "POWER_SNP_02": 0,
        "POWER_SNP_03": 1,
        "POWER_SNP_04": 0,

        "ENDURANCE_SNP_01": 0,
        "ENDURANCE_SNP_02": 1,
        "ENDURANCE_SNP_03": 0,
        "ENDURANCE_SNP_04": 1,
        "ENDURANCE_SNP_05": 0,
        "ENDURANCE_SNP_06": 1,

        "AUDITORY_SNP_01": 1,
        "AUDITORY_SNP_02": 0,
        "AUDITORY_SNP_03": 1,
        "AUDITORY_SNP_04": 0,

        "NEUROMOTOR_SNP_01": 0,
        "NEUROMOTOR_SNP_02": 1,
        "NEUROMOTOR_SNP_03": 1,
        "NEUROMOTOR_SNP_04": 0,
    }

    result = predict_all_traits(example)

    print("\nPredicted phenotype-associated scores:")
    for trait, score in result.items():
        print(f"{trait.capitalize():12s}: {score}")