import os
import glob
import numpy as np
from keras.models import load_model
from keras.utils import Sequence
import csv

class SequenceGenerator(Sequence):
    def __init__(self, sequences, batch_size=32, max_length=500):
        self.sequences = sequences
        self.batch_size = batch_size
        self.max_length = max_length
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}

    def one_hot_encode(self, seq):
        one_hot = np.zeros((self.max_length, len(self.amino_acids)), dtype=int)
        for i, aa in enumerate(seq):
            if aa in self.aa_to_index and i < self.max_length:
                one_hot[i, self.aa_to_index[aa]] = 1
        return one_hot

    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_sequences = self.sequences[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.one_hot_encode(seq) for seq in batch_sequences])
        return X

def load_fasta_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ''
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

def save_predictions_csv(sequences, y_pred_labels, y_pred_probs, output_dir, index_to_label, filename_prefix):
    filename = os.path.join(output_dir, f"{filename_prefix}_predictions.csv")
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence', 'Predicted Label', 'Predicted Probability'])
        for seq, pred_lbl, pred_prob in zip(sequences, y_pred_labels, y_pred_probs):
            label_str = index_to_label[pred_lbl]
            if label_str == "Control":
                label_str = "Not Immune Evasion Protein"
            writer.writerow([seq, label_str, float(max(pred_prob))])
    print(f"Predictions saved to {filename}")

def save_sequences_by_label(sequences, y_pred_labels, index_to_label, output_dir):
    label_to_seqs = {}
    for seq, pred_idx in zip(sequences, y_pred_labels):
        label = index_to_label[pred_idx]
        if label == "Control":
            label = "Not Immune Evasion Protein"
        if label not in label_to_seqs:
            label_to_seqs[label] = []
        label_to_seqs[label].append(seq)

    for label, seqs in label_to_seqs.items():
        safe_label = label.replace(" ", "_")
        filename = os.path.join(output_dir, f"{safe_label}.fasta")
        with open(filename, 'a') as f:
            for i, seq in enumerate(seqs, 1):
                header = f">{label}_seq{i}"
                f.write(header + "\n")
                f.write(seq + "\n")
        print(f"Saved {len(seqs)} sequences to {filename}")

def main():
    model_path = "hybrid_model/aidimin.h5"
    max_length = 500
    batch_size = 32
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Define fasta file extensions to look for
    fasta_extensions = ['*.fasta', '*.fa', '*.txt', '*.FASTA', '*.FA', '*.TXT']

    # Find all fasta files in current directory (exclude model and output folders)
    all_pred_files = []
    for ext in fasta_extensions:
        all_pred_files.extend(glob.glob(ext))

    # Filter out files inside 'hybrid_model' or 'Output' folders (if any)
    all_pred_files = [f for f in all_pred_files if os.path.isfile(f) and not (f.startswith('hybrid_model' + os.sep) or f.startswith('Output' + os.sep))]

    if not all_pred_files:
        print("No fasta/sequence files found in current directory for prediction.")
        return

    print(f"Found {len(all_pred_files)} sequence files for prediction: {all_pred_files}")

    # You must know your class labels in the order the model was trained
    # Replace this dict with your actual classes and order
    index_to_label = {0: "Control", 1: "Immune Evasion Protein"}  # Example, adjust accordingly

    for pred_file in all_pred_files:
        print(f"\nProcessing predictions for file: {pred_file}")
        sequences = load_fasta_sequences(pred_file)
        if not sequences:
            print(f"No sequences found in {pred_file}, skipping.")
            continue

        pred_gen = SequenceGenerator(sequences, batch_size=batch_size, max_length=max_length)
        preds_prob = model.predict(pred_gen, verbose=1)
        preds = np.argmax(preds_prob, axis=1)

        # Print summary predictions
        for i, (seq, pred_idx, prob) in enumerate(zip(sequences, preds, preds_prob)):
            label_str = index_to_label.get(pred_idx, "Unknown")
            if label_str == "Control":
                label_str = "Not Immune Evasion Protein"
            print(f"Seq#{i+1} (first 30 aa): {seq[:30]}... -> Predicted: {label_str} (prob={max(prob):.4f})")

        # Save predictions CSV
        filename_prefix = os.path.splitext(os.path.basename(pred_file))[0]
        save_predictions_csv(sequences, preds, preds_prob, output_dir, index_to_label, filename_prefix)

        # Save sequences grouped by predicted label
        save_sequences_by_label(sequences, preds, index_to_label, output_dir)

    print("\nAll predictions complete. Results saved in 'Output/' folder.")

if __name__ == "__main__":
    main()