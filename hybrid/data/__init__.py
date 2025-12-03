"""
Data Loading and Processing for Hybrid Energy-Based ProteinMPNN

This module handles:
- Stability dataset creation and management
- Training data pipeline with positive/negative pairs
- Data augmentation and hard negative mining
- Structure dataset integration
- Streaming data processing for large-scale training
- PDB structure management and caching
"""

# Core dataset imports
from .stability_dataset import StabilityDataset

# Streaming data infrastructure (skeleton implementations for development)
from .streaming_dataset import StreamingProteinDataset, ProteinDataSource, LocalPDBSource, RemotePDBSource
from .pdb_cache import PDBCache, PDBDownloader
from .pdb_manager import PDBManager, PDBListManager, PDBMetadata, StructureValidator

__all__ = [
    # Core datasets
    "StabilityDataset",
    # Streaming infrastructure
    "StreamingProteinDataset",
    "ProteinDataSource", 
    "LocalPDBSource",
    "RemotePDBSource",
    "PDBCache",
    "PDBDownloader", 
    "PDBManager",
    "PDBListManager", 
    "PDBMetadata",
    "StructureValidator",
]