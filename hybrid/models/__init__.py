"""
Model Components for Hybrid Energy-Based ProteinMPNN

This module contains the core neural network architectures:
- ProteinMPNN backbone encoder wrapper
- Continuous sequence representation with Gumbel-Softmax
- Energy head for stability prediction
"""

# Model imports - now with actual ProteinMPNN integration
from .mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from .sequence_repr import ContinuousSequenceRepr
from .energy_head import EnergyHead

# Main hybrid model with actual ProteinMPNN integration (no more placeholders!)
from .energy_model import EnergyBasedProteinMPNN, DeterministicStructuralEncoder, DeterministicSequenceEmbedding

__all__ = [
    "EnergyBasedProteinMPNN",
    "ProteinMPNNBackboneEncoder", 
    "load_pretrained_encoder",
    "ContinuousSequenceRepr",
    "EnergyHead",
    "DeterministicStructuralEncoder",
    "DeterministicSequenceEmbedding"
]