# AIDIMIN - Protein Sequence Prediction Tool

**AIDIMIN** (Aiding the Identification of Immune Evasion Proteins of Influenza A and B Viruses) is a deep learning tool for classifying protein sequences using a hybrid CNN-RNN model.

## Features

- 🧬 Predicts protein sequences from FASTA files
- 🤖 Uses pre-trained hybrid CNN-LSTM deep learning model
- 📁 Supports multiple file formats: `.fa`, `.fasta`, `.txt`
- 📊 Generates CSV reports with confidence scores
- 🗂️ Automatically groups predictions by class
- 🔄 Batch processing for multiple files

## Installation

### Prerequisites

- Ubuntu/Linux terminal

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/aidimin.git
cd aidimin

# Make prediction script executable
chmod +x predict.py
```

First, make sure you have conda installed:

```bash
1. Install  Miniconda (if not installed)

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

2. Activate conda base environment

conda init

Then, restart your terminal or run:

source ~/.bashrc

After that, activate the base environment with:

conda activate
```

Second, make sure you have Python specific version installed:

```bash
conda create -n py312 python=3.12.9
conda activate py312
python --version
```

Third, make sure you have specific Tensorflow, Keras, numpy, scikit-learn versions installed:

```bash

tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
scikit-learn>=1.0.0

conda install -c conda-forge "tensorflow>=2.8.0"
conda install -c conda-forge "keras>=2.8.0"
conda install -c conda-forge "numpy>=1.21.0"
conda install -c conda-forge "scikit-learn>=1.0.0"
```

## Usage

### Basic Usage

```bash
# Activate Python 3.12
conda activate py312

# Predict a single FASTA file
python predict.py -f your_sequences.fasta

# Predict with custom output directory
python predict.py -f sequences.fa -o results/

# Auto-detect and predict all FASTA files in current directory
python predict.py --auto
```

### Advanced Usage

```bash
# Custom model path and batch size
python predict.py -f input.fasta -m path/to/model.h5 -b 64

# Quiet mode (less verbose output)
python predict.py -f sequences.fasta --quiet

# Help
python predict.py --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --file` | Input FASTA file path | - |
| `-m, --model` | Path to model file | `aidimin.h5` |
| `-o, --output` | Output directory | `predictions` |
| `-b, --batch-size` | Batch size for prediction | `32` |
| `--auto` | Auto-predict all FASTA files | `False` |
| `-q, --quiet` | Suppress verbose output | `False` |

## Input Format

The tool accepts FASTA files with standard format:

```
>sequence_header_1
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY
>sequence_header_2
MKTIIALSYILCLVFAQKLPGFGDSIEAQCGTSVNVHSSLRDILV
```

**Supported file extensions:** `.fasta`, `.fa`, `.txt`, `.FASTA`, `.FA`, `.TXT`

## Output

The tool generates several output files:

### 1. CSV Predictions Report
- **File:** `{input_filename}_predictions.csv`
- **Contains:** Sequence headers, truncated sequences, predicted labels, confidence scores

### 2. Grouped FASTA Files
- **Files:** `{input_filename}_{predicted_class}.fasta`
- **Contains:** Sequences grouped by their predicted classification

### Example Output Structure
```
predictions/
├── sample_predictions.csv
├── sample_Not_Immune_Evasion_Protein.fasta
└── sample_Immune_Evasion_Protein.fasta
```

## Model Requirements

- **Model file:** `aidimin.h5` (should be in the same directory as predict.py)
- **Input shape:** (500, 20) - sequences up to 500 amino acids
- **Amino acids:** Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY)

## Examples

### Example 1: Single File Prediction
```bash
# Place your FASTA file in the current directory
python predict.py -f my_proteins.fasta

# Check results
ls predictions/
# Output: my_proteins_predictions.csv, my_proteins_Class1.fasta, etc.
```

### Example 2: Batch Processing
```bash
# Place multiple FASTA files in current directory
ls *.fasta
# sample1.fasta  sample2.fasta  sample3.fasta

python predict.py --auto

# All files will be processed automatically
```

### Example 3: Custom Configuration
```bash
python predict.py \
  -f large_dataset.fasta \
  -m models/custom_model.h5 \
  -o custom_results/ \
  -b 128
```

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: Model file 'aidimin.h5' not found.
   ```
   **Solution:** Ensure the model file is in the same directory or specify the correct path with `-m`

2. **No sequences found**
   ```
   No sequences found in input.fasta
   ```
   **Solution:** Check FASTA file format and ensure sequences are properly formatted

3. **Memory issues with large files**
   **Solution:** Reduce batch size using `-b 16` or `-b 8`

### GPU Support

If you have CUDA-compatible GPU and want to use GPU acceleration:

```bash
# Install TensorFlow GPU version
pip install tensorflow-gpu>=2.8.0
```

## Performance

- **Speed:** ~1000-5000 sequences per minute (CPU)
- **Memory:** ~2-4GB RAM for typical datasets
- **Accuracy:** Depends on model training (refer to training metrics)

## Model Information

- **Architecture:** Hybrid CNN-LSTM
- **Input:** One-hot encoded amino acid sequences
- **Max length:** 500 amino acids
- **Classes:** 17 protein classes including:
  - Control (Non-immune evasion proteins)
  - Influenza A proteins: HA, M2, NA, NP, NS1, NS2, PA-X, PB1-F2, PB2
  - Influenza B proteins: HA, M2, NA, NP, NS1, NS2, PB2

### Protein Classifications

| Class | Description |
|-------|-------------|
| Control | Non-immune evasion proteins |
| InfA HA | Influenza A Hemagglutinin |
| InfA M2 | Influenza A Matrix protein 2 |
| InfA NA | Influenza A Neuraminidase |
| InfA NP | Influenza A Nucleoprotein |
| InfA NS1 | Influenza A Non-structural protein 1 |
| InfA NS2 | Influenza A Non-structural protein 2 |
| InfA PA-X | Influenza A PA-X protein |
| InfA PB1-F2 | Influenza A PB1-F2 protein |
| InfA PB2 | Influenza A Polymerase basic protein 2 |
| InfB HA | Influenza B Hemagglutinin |
| InfB M2 | Influenza B Matrix protein 2 |
| InfB NA | Influenza B Neuraminidase |
| InfB NP | Influenza B Nucleoprotein |
| InfB NS1 | Influenza B Non-structural protein 1 |
| InfB NS2 | Influenza B Non-structural protein 2 |
| InfB PB2 | Influenza B Polymerase basic protein 2 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use AIDIMIN in your research, please cite:

```
[Your paper citation here]
```

## Contact

- **Author:** Jion Hossen
- **Email:** mdzeon19034@gmail.com
- **GitHub:** https://github.com/aidrabd/aidimin

## Changelog

### v1.0.0
- Initial release
- Basic prediction functionality
- CSV and FASTA output support
- Batch processing capability
