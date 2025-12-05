"""
Canonical Amino Acid Vocabulary for Energy-Based ProteinMPNN

This module defines the CANONICAL amino acid vocabulary ordering that MUST be used
throughout the entire hybrid codebase. This ordering matches the ProteinMPNN standard
and is critical for data compatibility.

WARNING: Changing this ordering will make all existing checkpoints and data incompatible.
Any file that defines its own amino acid alphabet MUST be updated to import from here.

The ordering ARNDCQEGHILKMFPSTWYV is the ProteinMPNN standard and is used by:
- All ProteinMPNN training data
- Streaming dataset preprocessing 
- Checkpoint files
- Model embeddings

DO NOT USE alphabetical ordering (ACDEFGHIKLMNPQRSTVWY) - this causes data corruption.
"""

from typing import Dict, List, Set
import warnings


# CANONICAL amino acid vocabulary - ProteinMPNN standard ordering
# This MUST match the ordering used by streaming_dataset.py and all ProteinMPNN data
AMINO_ACID_ALPHABET: str = "ARNDCQEGHILKMFPSTWYV"

# Standard mappings derived from canonical alphabet
AMINO_ACID_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACID_ALPHABET)}
IDX_TO_AMINO_ACID: Dict[int, str] = {i: aa for i, aa in enumerate(AMINO_ACID_ALPHABET)}

# Vocabulary size constant
VOCAB_SIZE: int = len(AMINO_ACID_ALPHABET)

# Set for fast membership testing
AMINO_ACID_SET: Set[str] = set(AMINO_ACID_ALPHABET)

# Explicit mapping for clarity (matches streaming_dataset.py:1273-1275)
PROTEINMPNN_STANDARD_MAPPING: Dict[str, int] = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# DEPRECATED: Alphabetical ordering that was incorrectly used in some files
DEPRECATED_ALPHABETICAL: str = "ACDEFGHIKLMNPQRSTVWY"


def validate_alphabet_consistency(alphabet: str, context: str = "") -> None:
    """
    Validate that an alphabet string matches the canonical ProteinMPNN ordering.
    
    Args:
        alphabet: Amino acid alphabet string to check
        context: Description of where this alphabet is used (for error messages)
        
    Raises:
        ValueError: If alphabet doesn't match canonical ordering
        
    Example:
        >>> validate_alphabet_consistency("ACDEFGHIKLMNPQRSTVWY", "train_energy.py")
        ValueError: Alphabet mismatch in train_energy.py: found ACDEFGHIKLMNPQRSTVWY, expected ARNDCQEGHILKMFPSTWYV
    """
    if alphabet != AMINO_ACID_ALPHABET:
        raise ValueError(
            f"Alphabet mismatch{' in ' + context if context else ''}: "
            f"found {alphabet}, expected {AMINO_ACID_ALPHABET}. "
            f"Using wrong alphabet causes data corruption!"
        )


def validate_mapping_consistency(aa_to_idx: Dict[str, int], context: str = "") -> None:
    """
    Validate that an amino acid to index mapping matches canonical ordering.
    
    Args:
        aa_to_idx: Amino acid to index mapping to check
        context: Description of where this mapping is used (for error messages)
        
    Raises:
        ValueError: If mapping doesn't match canonical ordering
    """
    if aa_to_idx != AMINO_ACID_TO_IDX:
        raise ValueError(
            f"Mapping mismatch{' in ' + context if context else ''}: "
            f"found {aa_to_idx}, expected {AMINO_ACID_TO_IDX}. "
            f"Using wrong mapping causes data corruption!"
        )


def warn_deprecated_alphabet(alphabet: str, context: str = "") -> None:
    """
    Warn if deprecated alphabetical ordering is detected.
    
    Args:
        alphabet: Amino acid alphabet string to check
        context: Description of where this alphabet is used
    """
    if alphabet == DEPRECATED_ALPHABETICAL:
        warnings.warn(
            f"CRITICAL: Deprecated alphabetical ordering detected"
            f"{' in ' + context if context else ''}! "
            f"Found {DEPRECATED_ALPHABETICAL}, should be {AMINO_ACID_ALPHABET}. "
            f"This causes data corruption - update to use vocab.AMINO_ACID_ALPHABET",
            UserWarning,
            stacklevel=2
        )


def is_valid_amino_acid(aa: str) -> bool:
    """
    Check if a character is a valid amino acid in canonical vocabulary.
    
    Args:
        aa: Single character amino acid code
        
    Returns:
        True if aa is in canonical vocabulary, False otherwise
    """
    return aa.upper() in AMINO_ACID_SET


def get_canonical_encoding() -> Dict[str, int]:
    """
    Get the canonical amino acid to index mapping.
    
    Returns:
        Dictionary mapping amino acid characters to indices
    """
    return AMINO_ACID_TO_IDX.copy()


def get_canonical_alphabet() -> str:
    """
    Get the canonical amino acid alphabet string.
    
    Returns:
        Canonical alphabet in ProteinMPNN standard ordering
    """
    return AMINO_ACID_ALPHABET
