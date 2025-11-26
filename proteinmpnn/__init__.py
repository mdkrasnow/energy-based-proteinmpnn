"""
ProteinMPNN Package

This package provides the ProteinMPNN neural network architecture and utilities
for protein sequence design based on structure.
"""

# Import main classes for easier access
try:
    from .protein_mpnn_utils import ProteinMPNN, ProteinFeatures, CA_ProteinFeatures, EncLayer
    from .protein_mpnn_utils import gather_nodes, parse_fasta
except ImportError:
    # If imports fail, the module will still be importable
    # but classes will need to be imported directly from protein_mpnn_utils
    pass