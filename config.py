# AIDIMIN Configuration File
# This file contains the class labels and other configuration settings

# Class labels based on your training data
# The order should match the order used during training
CLASS_LABELS = [
    "Control",          # Non-immune evasion proteins
    "InfA HA",         # Influenza A Hemagglutinin  
    "InfA M2",         # Influenza A Matrix protein 2
    "InfA NA",         # Influenza A Neuraminidase
    "InfA NP",         # Influenza A Nucleoprotein
    "InfA NS1",        # Influenza A Non-structural protein 1
    "InfA NS2",        # Influenza A Non-structural protein 2
    "InfA PA-X",       # Influenza A PA-X protein
    "InfA PB1-F2",     # Influenza A PB1-F2 protein
    "InfA PB2",        # Influenza A Polymerase basic protein 2
    "InfB HA",         # Influenza B Hemagglutinin
    "InfB M2",         # Influenza B Matrix protein 2
    "InfB NA",         # Influenza B Neuraminidase
    "InfB NP",         # Influenza B Nucleoprotein
    "InfB NS1",        # Influenza B Non-structural protein 1
    "InfB NS2",        # Influenza B Non-structural protein 2
    "InfB PB2"         # Influenza B Polymerase basic protein 2
]

# Model parameters
DEFAULT_MODEL_PATH = "aidimin.h5"
MAX_SEQUENCE_LENGTH = 500
BATCH_SIZE = 32

# Amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Output settings
DEFAULT_OUTPUT_DIR = "predictions"
CSV_COLUMNS = ['Header', 'Sequence', 'Predicted_Label', 'Confidence']

# File extensions for FASTA files
FASTA_EXTENSIONS = ['*.fasta', '*.fa', '*.txt', '*.FASTA', '*.FA', '*.TXT']

# Confidence threshold for predictions (optional)
CONFIDENCE_THRESHOLD = 0.5

# Protein class descriptions (for documentation)
CLASS_DESCRIPTIONS = {
    "Control": "Non-immune evasion proteins",
    "InfA HA": "Influenza A Hemagglutinin - Surface glycoprotein",
    "InfA M2": "Influenza A Matrix protein 2 - Ion channel protein",
    "InfA NA": "Influenza A Neuraminidase - Surface enzyme",
    "InfA NP": "Influenza A Nucleoprotein - RNA binding protein",
    "InfA NS1": "Influenza A Non-structural protein 1 - Immune antagonist",
    "InfA NS2": "Influenza A Non-structural protein 2 - Nuclear export protein",
    "InfA PA-X": "Influenza A PA-X protein - Viral endonuclease",
    "InfA PB1-F2": "Influenza A PB1-F2 protein - Virulence factor",
    "InfA PB2": "Influenza A Polymerase basic protein 2 - RNA polymerase subunit",
    "InfB HA": "Influenza B Hemagglutinin - Surface glycoprotein",
    "InfB M2": "Influenza B Matrix protein 2 - Ion channel protein",
    "InfB NA": "Influenza B Neuraminidase - Surface enzyme",
    "InfB NP": "Influenza B Nucleoprotein - RNA binding protein",
    "InfB NS1": "Influenza B Non-structural protein 1 - Immune antagonist",
    "InfB NS2": "Influenza B Non-structural protein 2 - Nuclear export protein",
    "InfB PB2": "Influenza B Polymerase basic protein 2 - RNA polymerase subunit"
}