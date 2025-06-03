import os
import requests
import numpy as np
from tensorflow.keras.models import load_model

# Correct raw URL to your model file on GitHub
MODEL_URL = "https://github.com/aidrabd/hybrid_model/raw/main/aidimin.h5"
MODEL_PATH = "aidimin.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GitHub...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print("Model already downloaded.")

def main():
    # Download the model file if not exists
    download_model()

    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    # Prepare dummy input data (change shape according to your model)
    sample_input = np.random.random((1, 224, 224, 3))  # Example for image model

    # Run prediction
    print("Running prediction...")
    predictions = model.predict(sample_input)
    print("Prediction output:")
    print(predictions)

if __name__ == "__main__":
    main()