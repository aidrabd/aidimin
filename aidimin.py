import os
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('hybrid_model/aidimin.h5')

# Allowed file extensions
extensions = ('.fa', '.fasta', '.txt')

# Function to preprocess file content into model input
def preprocess_file(filepath):
    # Example: read file, convert characters to numbers (dummy example)
    with open(filepath, 'r') as f:
        content = f.read().strip()

    # Replace below with your actual preprocessing logic
    # For demo, create a fixed-size dummy input array
    # e.g. model expects (1, 224, 224, 3) shape, so we create random data
    dummy_input = np.random.random((1, 224, 224, 3))
    return dummy_input

def main():
    # Get current directory (where this script runs)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # List all files with allowed extensions
    input_files = [f for f in os.listdir(current_dir) if f.endswith(extensions)]

    if not input_files:
        print("No input files with .fa, .fasta, .txt found in current directory.")
        return

    for filename in input_files:
        filepath = os.path.join(current_dir, filename)
        print(f"Processing file: {filename}")

        # Preprocess file into model input
        model_input = preprocess_file(filepath)

        # Predict
        prediction = model.predict(model_input)

        # Print prediction output (customize as needed)
        print(f"Prediction for {filename}:")
        print(prediction)
        print("-" * 40)

if __name__ == "__main__":
    main()