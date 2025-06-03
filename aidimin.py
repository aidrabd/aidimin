import os
import sys
from tensorflow import keras
from keras.layers import InputLayer
import numpy as np

# Custom InputLayer to ignore 'batch_shape' argument during deserialization
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

def load_model_custom(model_path):
    custom_objects = {'InputLayer': CustomInputLayer}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def find_sequence_file(exclude_files):
    for f in os.listdir('.'):
        if os.path.isfile(f) and f not in exclude_files:
            if f.endswith(('.fasta', '.fa', '.txt')):
                return f
    return None

def read_sequence_file(filepath):
    # Basic read: concatenate all lines ignoring lines starting with '>' (fasta header)
    seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith('>'):  # fasta header
                continue
            seq.append(line.upper())
    return ''.join(seq)

def encode_sequence(seq, max_len=1000):
    # Simple integer encoding: A=0, C=1, G=2, T=3, others=4
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    encoded = [mapping.get(base, 4) for base in seq]
    # Pad or truncate
    if len(encoded) < max_len:
        encoded += [4] * (max_len - len(encoded))  # pad with 4 (unknown)
    else:
        encoded = encoded[:max_len]
    return np.array(encoded)

def main():
    MODEL_PATH = os.path.join('hybrid_model', 'aidimin.h5')
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading model...")
    model = load_model_custom(MODEL_PATH)
    print("Model loaded successfully!")
    model.summary()

    exclude = {os.path.basename(__file__), 'hybrid_model'}

    input_file = find_sequence_file(exclude)
    if input_file is None:
        print("No .fasta, .fa, or .txt input file found in current directory.")
        sys.exit(1)

    print(f"Found input file: {input_file}")

    try:
        sequence = read_sequence_file(input_file)
        print(f"Read sequence length: {len(sequence)}")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    # Adjust max_len and input shape to your model's expected input
    max_len = 1000
    encoded_seq = encode_sequence(sequence, max_len)

    # Reshape for model input, e.g. (batch_size, sequence_length), or add channels if needed
    # Adjust shape according to your model input, here assuming (1, max_len)
    input_data = np.expand_dims(encoded_seq, axis=0)

    print("Running prediction...")
    predictions = model.predict(input_data)

    print("Prediction result:")
    print(predictions)

if __name__ == '__main__':
    main()