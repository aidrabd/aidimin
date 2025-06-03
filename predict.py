#!/usr/bin/env python3
"""
AIDIMIN Prediction Tool
Standalone script for predicting protein sequences using pre-trained model
"""

import os
import sys
import csv
import argparse
import numpy as np
from keras.models import load_model
from keras.utils import Sequence
import glob

# Try to import configuration, fall back to defaults if not available
try:
    from config import CLASS_LABELS, AMINO_ACIDS, FASTA_EXTENSIONS, DEFAULT_OUTPUT_DIR
except ImportError:
    CLASS_LABELS = [
        "Control", "InfA HA", "InfA M2", "InfA NA", "InfA NP", "InfA NS1", 
        "InfA NS2", "InfA PA-X", "InfA PB1-F2", "InfA PB2", "InfB HA", 
        "InfB M2", "InfB NA", "InfB NP", "InfB NS1", "InfB NS2", "InfB PB2"
    ]
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    FASTA_EXTENSIONS = ['*.fasta', '*.fa', '*.txt', '*.FASTA', '*.FA', '*.TXT']
    DEFAULT_OUTPUT_DIR = "predictions"


class SequenceGenerator(Sequence):
    def __init__(self, sequences, batch_size=32, max_length=500):
        self.sequences = sequences
        self.batch_size = batch_size
        self.max_length = max_length
        self.amino_acids = AMINO_ACIDS
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
    """Load sequences from FASTA file"""
    sequences = []
    sequence_headers = []
    
    try:
        with open(file_path, 'r') as file:
            sequence = ''
            header = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence_headers.append(header)
                    header = line[1:]  # Remove '>' character
                    sequence = ''
                else:
                    sequence += line
            if sequence:  # Don't forget the last sequence
                sequences.append(sequence)
                sequence_headers.append(header)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return [], []
    
    return sequences, sequence_headers


def save_predictions_csv(sequences, headers, y_pred_labels, y_pred_probs, output_file, index_to_label):
    """Save predictions to CSV file"""
    try:
        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Header', 'Sequence', 'Predicted_Label', 'Confidence'])
            
            for header, seq, pred_lbl, pred_prob in zip(headers, sequences, y_pred_labels, y_pred_probs):
                label_str = index_to_label[pred_lbl]
                # Map 'Control' to 'Not Immune Evasion Protein'
                if label_str == "Control":
                    label_str = "Not Immune Evasion Protein"
                confidence = float(max(pred_prob))
                writer.writerow([header, seq[:50] + "..." if len(seq) > 50 else seq, label_str, confidence])
        
        print(f"✓ Predictions saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        return False


def save_sequences_by_label(sequences, headers, y_pred_labels, index_to_label, output_dir, input_filename):
    """Save sequences grouped by predicted label into separate FASTA files"""
    label_to_seqs = {}
    
    for header, seq, pred_idx in zip(headers, sequences, y_pred_labels):
        label = index_to_label[pred_idx]
        if label == "Control":
            label = "Not_Immune_Evasion_Protein"
        else:
            label = label.replace(" ", "_")  # Replace spaces with underscores
        
        if label not in label_to_seqs:
            label_to_seqs[label] = []
        label_to_seqs[label].append((header, seq))
    
    base_filename = os.path.splitext(input_filename)[0]
    
    for label, seq_data in label_to_seqs.items():
        filename = os.path.join(output_dir, f"{base_filename}_{label}.fasta")
        try:
            with open(filename, 'w') as f:
                for i, (header, seq) in enumerate(seq_data, 1):
                    f.write(f">{header}_pred_{label}\n")
                    f.write(seq + "\n")
            print(f"✓ Saved {len(seq_data)} sequences to: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")


def get_class_labels(num_classes):
    """Get class labels based on number of classes detected from model"""
    
    # Return mapping based on actual number of classes
    if num_classes <= len(CLASS_LABELS):
        return {i: CLASS_LABELS[i] for i in range(num_classes)}
    else:
        # If more classes than expected, use generic names for extras
        labels = {i: CLASS_LABELS[i] for i in range(len(CLASS_LABELS))}
        for i in range(len(CLASS_LABELS), num_classes):
            labels[i] = f"Class_{i}"
        return labels


def predict_single_file(model_path, input_file, output_dir, batch_size=32, verbose=True):
    """Predict sequences from a single FASTA file"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False
    
    # Load model
    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        if verbose:
            print("✓ Model loaded successfully")
        
        # Automatically detect number of classes from model output shape
        num_classes = model.output_shape[-1]
        if verbose:
            print(f"✓ Detected {num_classes} classes from model")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    # Load sequences
    sequences, headers = load_fasta_sequences(input_file)
    if not sequences:
        print(f"No sequences found in {input_file}")
        return False
    
    print(f"✓ Loaded {len(sequences)} sequences from {input_file}")
    
    # Create prediction generator
    pred_gen = SequenceGenerator(sequences, batch_size=batch_size, max_length=500)
    
    # Make predictions
    try:
        print("Making predictions...")
        preds_prob = model.predict(pred_gen, verbose=1 if verbose else 0)
        preds = np.argmax(preds_prob, axis=1)
        print("✓ Predictions completed")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return False
    
    # Get class labels based on model output
    index_to_label = get_class_labels(num_classes)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    csv_output = os.path.join(output_dir, f"{base_filename}_predictions.csv")
    
    # Save predictions to CSV
    save_predictions_csv(sequences, headers, preds, preds_prob, csv_output, index_to_label)
    
    # Save sequences grouped by prediction
    save_sequences_by_label(sequences, headers, preds, index_to_label, output_dir, os.path.basename(input_file))
    
    # Print summary
    if verbose:
        print(f"\nPrediction Summary for {input_file}:")
        for i, (header, seq, pred_idx, prob) in enumerate(zip(headers[:10], sequences[:10], preds[:10], preds_prob[:10])):  # Show first 10
            label_str = index_to_label[pred_idx]
            if label_str == "Control":
                label_str = "Not Immune Evasion Protein"
            print(f"  {i+1}. {header[:30]}... -> {label_str} (confidence: {max(prob):.4f})")
        
        if len(sequences) > 10:
            print(f"  ... and {len(sequences) - 10} more sequences")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AIDIMIN: Predict protein sequences using pre-trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -f sequences.fasta
  python predict.py -f input.fa -o results/
  python predict.py --auto
  python predict.py --help
        """
    )
    
    parser.add_argument('-f', '--file', type=str,
                       help='Input FASTA file (.fa, .fasta, .txt)')
    
    parser.add_argument('-m', '--model', type=str, default='aidimin.h5',
                       help='Path to trained model file (default: aidimin.h5)')
    
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory (default: predictions)')
    
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    
    parser.add_argument('--auto', action='store_true',
                       help='Automatically predict all FASTA files in current directory')
    
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AIDIMIN - Protein Sequence Prediction Tool")
    print("="*60)
    
    verbose = not args.quiet
    
    if args.auto:
        # Automatically find and predict all FASTA files
        all_files = []
        for ext in FASTA_EXTENSIONS:
            all_files.extend(glob.glob(ext))
        
        # Filter out model and output directories
        all_files = [f for f in all_files if os.path.isfile(f) and 
                    not (f.startswith('model' + os.sep) or f.startswith(args.output + os.sep))]
        
        if not all_files:
            print("No FASTA files found in current directory.")
            return
        
        print(f"Found {len(all_files)} FASTA files: {', '.join(all_files)}")
        
        success_count = 0
        for file in all_files:
            print(f"\n{'-'*40}")
            print(f"Processing: {file}")
            print(f"{'-'*40}")
            if predict_single_file(args.model, file, args.output, args.batch_size, verbose):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{len(all_files)} files processed successfully")
        
    elif args.file:
        # Predict single file
        if not os.path.exists(args.file):
            print(f"Error: Input file '{args.file}' not found.")
            return
        
        success = predict_single_file(args.model, args.file, args.output, args.batch_size, verbose)
        if success:
            print(f"\n{'='*60}")
            print("Prediction completed successfully!")
        else:
            print("Prediction failed.")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()