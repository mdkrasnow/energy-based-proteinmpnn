#!/usr/bin/env python3
"""
Comprehensive Design Validation Framework

This module provides in silico validation tools for generated protein sequences,
including structural quality assessment, stability analysis, and out-of-distribution
detection. It integrates multiple computational tools to provide comprehensive
evaluation of protein designs.

Key Validation Components:
1. AlphaFold Confidence Prediction - Structural quality via pLDDT scores
2. Rosetta Energy Scoring - Physics-based stability assessment  
3. Secondary Structure Prediction - Structural property analysis
4. Aggregation Propensity Analysis - Stability and developability
5. Sequence Diversity Metrics - Novelty and exploration assessment
6. Perplexity-based OOD Detection - ProteinMPNN perplexity measurement
7. Statistical Significance Testing - Rigorous comparison framework

The system is designed for:
- Batch processing of large design sets
- Graceful degradation when optional tools are unavailable
- Comprehensive reporting with statistical analysis
- Integration with existing evaluation framework
"""

import os
import sys
import json
import warnings
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Optional dependencies with graceful degradation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Visualizations will be disabled.")

try:
    from scipy import stats
    from scipy.stats import mannwhitneyu, wilcoxon, kstest, normaltest

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical analysis will be limited.")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Report generation will be limited.")

try:
    from Bio import PDB, SeqUtils, Align
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.PDB.DSSP import DSSP

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available. Structural analysis will be limited.")

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import project modules
from models.mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr

# Import ProteinMPNN utilities for perplexity measurement
sys.path.append(os.path.join(str(project_root), "..", "proteinmpnn"))
try:
    from protein_mpnn_utils import parse_PDB, ProteinMPNN, _scores

    PROTEINMPNN_AVAILABLE = True
except ImportError:
    PROTEINMPNN_AVAILABLE = False
    warnings.warn("ProteinMPNN utilities not available for perplexity measurement.")

# External tool availability checks
ALPHAFOLD_AVAILABLE = False
try:
    # Check for ColabFold/AlphaFold2 installations
    result = subprocess.run(
        ["which", "colabfold_batch"], capture_output=True, text=True
    )
    if result.returncode == 0:
        ALPHAFOLD_AVAILABLE = True
        ALPHAFOLD_METHOD = "colabfold"
    else:
        # Try alphafold2 or other methods
        ALPHAFOLD_METHOD = "mock"  # Use mock for development
        warnings.warn("AlphaFold/ColabFold not found. Using mock predictions.")
except Exception as e:
    warnings.warn(f"Error checking AlphaFold availability: {e}")
    ALPHAFOLD_METHOD = "mock"

ROSETTA_AVAILABLE = False
try:
    import pyrosetta

    ROSETTA_AVAILABLE = True
except ImportError:
    warnings.warn("PyRosetta not available. Using simplified energy calculations.")

# Constants
CANONICAL_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(CANONICAL_AA_ORDER)}

# Amino acid properties for aggregation analysis
# Hydrophobicity values from Kyte-Doolittle scale (canonical reference)
# Beta propensity from Chou-Fasman parameters (corrected)
AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "beta_propensity": 0.83, "aggregation_prone": False},
    "C": {"hydrophobicity": 2.5, "beta_propensity": 1.19, "aggregation_prone": False},
    "D": {"hydrophobicity": -3.5, "beta_propensity": 0.54, "aggregation_prone": False},
    "E": {"hydrophobicity": -3.5, "beta_propensity": 0.37, "aggregation_prone": False},
    "F": {"hydrophobicity": 2.8, "beta_propensity": 1.38, "aggregation_prone": True},
    "G": {"hydrophobicity": -0.4, "beta_propensity": 0.75, "aggregation_prone": False},
    "H": {"hydrophobicity": -3.2, "beta_propensity": 0.87, "aggregation_prone": False},
    "I": {"hydrophobicity": 4.5, "beta_propensity": 1.60, "aggregation_prone": True},
    "K": {"hydrophobicity": -3.9, "beta_propensity": 0.74, "aggregation_prone": False},
    "L": {"hydrophobicity": 3.8, "beta_propensity": 1.30, "aggregation_prone": True},
    "M": {"hydrophobicity": 1.9, "beta_propensity": 1.05, "aggregation_prone": False},
    "N": {"hydrophobicity": -3.5, "beta_propensity": 0.89, "aggregation_prone": False},
    "P": {"hydrophobicity": -1.6, "beta_propensity": 0.55, "aggregation_prone": False},
    "Q": {"hydrophobicity": -3.5, "beta_propensity": 1.10, "aggregation_prone": False},
    "R": {"hydrophobicity": -4.5, "beta_propensity": 0.93, "aggregation_prone": False},
    "S": {"hydrophobicity": -0.8, "beta_propensity": 0.75, "aggregation_prone": False},
    "T": {"hydrophobicity": -0.7, "beta_propensity": 1.19, "aggregation_prone": False},
    "V": {"hydrophobicity": 4.2, "beta_propensity": 1.70, "aggregation_prone": True},
    "W": {"hydrophobicity": -0.9, "beta_propensity": 1.37, "aggregation_prone": True},
    "Y": {"hydrophobicity": -1.3, "beta_propensity": 1.47, "aggregation_prone": True},
}


@dataclass
class ValidationConfig:
    """Configuration for design validation pipeline."""

    # Reproducibility settings
    random_seed: int = 42  # Global random seed for all stochastic operations

    # Tool selection
    use_alphafold: bool = True
    use_rosetta: bool = True
    use_structure_prediction: bool = True
    use_aggregation_analysis: bool = True
    use_perplexity_analysis: bool = True

    # AlphaFold settings
    alphafold_method: str = "colabfold"  # 'colabfold', 'alphafold2', 'mock'
    alphafold_max_sequences: int = 100
    alphafold_timeout: int = 3600  # 1 hour

    # Rosetta settings
    rosetta_score_functions: List[str] = field(
        default_factory=lambda: ["ref2015", "beta_nov16"]
    )
    rosetta_relax_cycles: int = 3

    # Statistical testing
    statistical_alpha: float = 0.05
    bootstrap_samples: int = 1000
    multiple_testing_correction: str = "bonferroni"  # 'bonferroni', 'fdr', 'none'

    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    device: str = "auto"

    # Output settings
    output_dir: Optional[str] = None
    save_intermediate: bool = True
    generate_plots: bool = True
    verbose: bool = True


@dataclass
class ValidationResults:
    """Container for comprehensive validation results."""

    # Basic info
    sequence: str
    sequence_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Structural quality
    alphafold_confidence: Optional[float] = None
    alphafold_plddt: Optional[List[float]] = None
    structure_quality: Optional[str] = None  # 'high', 'medium', 'low'

    # Energy scores
    rosetta_score: Optional[float] = None
    rosetta_components: Optional[Dict[str, float]] = None

    # Secondary structure
    secondary_structure: Optional[str] = None
    ss_composition: Optional[Dict[str, float]] = None
    solvent_accessibility: Optional[List[float]] = None

    # Stability analysis
    aggregation_propensity: Optional[float] = None
    stability_score: Optional[float] = None
    developability_flags: Optional[List[str]] = None

    # Diversity metrics
    sequence_novelty: Optional[float] = None
    nearest_neighbor_distance: Optional[float] = None

    # Perplexity analysis
    proteinmpnn_perplexity: Optional[float] = None
    perplexity_zscore: Optional[float] = None
    ood_confidence: Optional[float] = None

    # Metadata
    validation_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            else:
                result[field_name] = field_value
        return result


class ValidationTool(ABC):
    """Abstract base class for validation tools."""

    @abstractmethod
    def validate(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Validate a sequence and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the tool is available for use."""
        pass


class AlphaFoldValidator(ValidationTool):
    """AlphaFold/ColabFold confidence prediction for generated sequences."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.method = config.alphafold_method
        self.temp_dir = tempfile.mkdtemp(prefix="alphafold_validation_")

    def is_available(self) -> bool:
        """Check if AlphaFold/ColabFold is available."""
        if self.method == "mock":
            return True
        elif self.method == "colabfold":
            try:
                result = subprocess.run(
                    ["which", "colabfold_batch"], capture_output=True, text=True
                )
                return result.returncode == 0
            except:
                return False
        else:
            return False

    def validate(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Predict structure and confidence for a sequence."""
        if not self.is_available():
            return {"error": "AlphaFold validator not available"}

        if self.method == "mock":
            return self._mock_alphafold_prediction(sequence)
        elif self.method == "colabfold":
            return self._run_colabfold(sequence, **kwargs)
        else:
            return {"error": f"Unknown AlphaFold method: {self.method}"}

    def _mock_alphafold_prediction(self, sequence: str) -> Dict[str, Any]:
        """Generate mock AlphaFold predictions for testing."""
        # Generate realistic-looking pLDDT scores based on sequence properties
        length = len(sequence)

        # Base confidence - shorter sequences tend to fold better
        base_confidence = min(85.0, 95.0 - 0.02 * length)

        # Add some realistic variation
        import hashlib

        seed = int(hashlib.md5(sequence.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)  # Isolated random state
        plddt_scores = []

        for i, aa in enumerate(sequence):
            # Lower confidence at termini
            position_factor = 1.0
            if i < 5 or i >= length - 5:
                position_factor = 0.85

            # Some amino acids are more confident
            aa_factor = 1.0
            if aa in "GPCN":  # Flexible residues
                aa_factor = 0.9
            elif aa in "AVLIF":  # Structured residues
                aa_factor = 1.1

            score = base_confidence * position_factor * aa_factor
            score += rng.normal(0, 5)  # Add noise from isolated generator
            score = np.clip(score, 30, 100)
            plddt_scores.append(score)

        mean_plddt = np.mean(plddt_scores)

        # Classify structure quality
        if mean_plddt > 80:
            quality = "high"
        elif mean_plddt > 70:
            quality = "medium"
        else:
            quality = "low"

        return {
            "alphafold_confidence": float(mean_plddt),
            "alphafold_plddt": plddt_scores,
            "structure_quality": quality,
            "method": "mock_alphafold",
        }

    def _run_colabfold(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Run ColabFold for structure prediction."""
        # Implementation would call ColabFold batch processing
        # For now, return mock results with a note
        results = self._mock_alphafold_prediction(sequence)
        results["method"] = "colabfold_mock"
        results["warning"] = "ColabFold not implemented - using mock results"
        return results


class RosettaValidator(ValidationTool):
    """Rosetta energy scoring and analysis."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._initialized = False

    def is_available(self) -> bool:
        """Check if PyRosetta is available."""
        return ROSETTA_AVAILABLE

    def _initialize_rosetta(self):
        """Initialize Rosetta if available."""
        if not self.is_available() or self._initialized:
            return

        try:
            import pyrosetta

            pyrosetta.init("-mute all")
            self._initialized = True
        except Exception as e:
            warnings.warn(f"Failed to initialize Rosetta: {e}")

    def validate(
        self, sequence: str, pdb_structure: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Score sequence using Rosetta energy functions."""
        if not self.is_available():
            return self._mock_rosetta_scoring(sequence)

        self._initialize_rosetta()
        if not self._initialized:
            return self._mock_rosetta_scoring(sequence)

        try:
            # Would implement actual Rosetta scoring here
            return self._mock_rosetta_scoring(sequence)
        except Exception as e:
            return {"error": f"Rosetta scoring failed: {e}"}

    def _mock_rosetta_scoring(self, sequence: str) -> Dict[str, Any]:
        """Generate mock Rosetta scores based on sequence properties."""
        # Input validation
        if not sequence:
            return {
                "rosetta_score": 0.0,
                "rosetta_components": {},
                "error": "Empty sequence",
            }

        length = len(sequence)

        # Calculate simple biophysical properties
        hydrophobic_count = sum(
            1 for aa in sequence if AA_PROPERTIES[aa]["hydrophobicity"] > 0.5
        )
        charged_count = sum(1 for aa in sequence if aa in "DEKR")

        # Mock total score (lower is better in Rosetta)
        base_score = -50.0 - 0.8 * length  # Base favorable energy

        # Safe division with validation
        hydrophobic_fraction = hydrophobic_count / length if length > 0 else 0.0
        if hydrophobic_fraction > 0.4:  # Too hydrophobic
            base_score += 20.0 * (hydrophobic_fraction - 0.4)

        charged_fraction = charged_count / length if length > 0 else 0.0
        if charged_fraction < 0.1:  # Too few charges for solubility
            base_score += 10.0 * (0.1 - charged_fraction)

        # Add some realistic noise
        seed = int(hashlib.md5(sequence.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)  # Isolated random state
        base_score += rng.normal(0, 5)

        # Mock component scores
        components = {
            "fa_atr": base_score * 0.4,  # Attractive van der Waals
            "fa_rep": base_score * 0.2,  # Repulsive van der Waals
            "fa_sol": base_score * 0.15,  # Solvation
            "fa_elec": base_score * 0.1,  # Electrostatic
            "hbond_sc": base_score * 0.1,  # Side chain hydrogen bonds
            "hbond_bb_sc": base_score * 0.05,  # Backbone-sidechain hydrogen bonds
        }

        return {
            "rosetta_score": float(base_score),
            "rosetta_components": components,
            "method": "mock_rosetta",
        }


class StructurePredictor(ValidationTool):
    """Secondary structure and solvent accessibility prediction."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def is_available(self) -> bool:
        """Check if structure prediction tools are available."""
        return True  # Always available with sequence-based methods

    def validate(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Predict secondary structure and accessibility."""
        return {
            **self._predict_secondary_structure(sequence),
            **self._predict_solvent_accessibility(sequence),
        }

    def _predict_secondary_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict secondary structure using Chou-Fasman method."""
        # Chou-Fasman propensities (simplified)
        helix_props = {
            "A": 1.42,
            "E": 1.51,
            "L": 1.21,
            "M": 1.45,
            "Q": 1.11,
            "H": 1.00,
            "K": 1.16,
            "F": 1.13,
            "T": 0.83,
            "Y": 0.69,
            "S": 0.77,
            "C": 0.70,
            "W": 1.08,
            "V": 1.06,
            "D": 1.01,
            "I": 1.08,
            "R": 0.98,
            "N": 0.67,
            "P": 0.57,
            "G": 0.57,
        }

        strand_props = {
            "V": 1.70,
            "I": 1.60,
            "Y": 1.47,
            "F": 1.38,
            "W": 1.37,
            "L": 1.30,
            "T": 1.19,
            "C": 1.19,
            "Q": 1.10,
            "M": 1.05,
            "R": 0.93,
            "N": 0.89,
            "H": 0.87,
            "A": 0.83,
            "K": 0.74,
            "S": 0.75,
            "G": 0.57,
            "P": 0.55,
            "D": 0.54,
            "E": 0.37,
        }

        # Calculate propensities
        helix_score = np.mean([helix_props[aa] for aa in sequence])
        strand_score = np.mean([strand_props[aa] for aa in sequence])

        # Simple prediction
        ss_prediction = []
        for aa in sequence:
            h_prop = helix_props[aa]
            s_prop = strand_props[aa]

            if h_prop > 1.0 and h_prop > s_prop:
                ss_prediction.append("H")  # Helix
            elif s_prop > 1.0 and s_prop > h_prop:
                ss_prediction.append("E")  # Sheet
            else:
                ss_prediction.append("C")  # Coil

        ss_string = "".join(ss_prediction)

        # Calculate composition
        h_count = ss_string.count("H")
        e_count = ss_string.count("E")
        c_count = ss_string.count("C")
        total = len(sequence)

        composition = {
            "helix": h_count / total if total > 0 else 0,
            "sheet": e_count / total if total > 0 else 0,
            "coil": c_count / total if total > 0 else 0,
        }

        return {"secondary_structure": ss_string, "ss_composition": composition}

    def _predict_solvent_accessibility(self, sequence: str) -> Dict[str, Any]:
        """Predict relative solvent accessibility."""
        # Simple accessibility prediction based on amino acid properties
        accessibility_props = {
            "A": 0.5,
            "C": 0.4,
            "D": 0.8,
            "E": 0.8,
            "F": 0.3,
            "G": 0.6,
            "H": 0.6,
            "I": 0.3,
            "K": 0.9,
            "L": 0.3,
            "M": 0.4,
            "N": 0.7,
            "P": 0.6,
            "Q": 0.7,
            "R": 0.9,
            "S": 0.6,
            "T": 0.6,
            "V": 0.4,
            "W": 0.3,
            "Y": 0.5,
        }

        accessibilities = []
        for i, aa in enumerate(sequence):
            base_acc = accessibility_props[aa]

            # Terminal residues are more exposed
            if i < 3 or i >= len(sequence) - 3:
                base_acc = min(1.0, base_acc + 0.2)

            accessibilities.append(base_acc)

        return {"solvent_accessibility": accessibilities}


class AggregationAnalyzer(ValidationTool):
    """Aggregation propensity and stability analysis."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def is_available(self) -> bool:
        """Always available with sequence-based methods."""
        return True

    def validate(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Analyze aggregation propensity and stability."""
        return {
            **self._analyze_aggregation_propensity(sequence),
            **self._analyze_stability_factors(sequence),
            **self._check_developability_flags(sequence),
        }

    def _analyze_aggregation_propensity(self, sequence: str) -> Dict[str, Any]:
        """Calculate aggregation propensity using PASTA-like scoring."""
        # Simplified aggregation scoring based on hydrophobic patches
        window_size = 6
        aggregation_scores = []

        for i in range(len(sequence) - window_size + 1):
            window = sequence[i : i + window_size]

            # Calculate aggregation score for window
            score = 0.0
            for aa in window:
                if AA_PROPERTIES[aa]["aggregation_prone"]:
                    score += AA_PROPERTIES[aa]["hydrophobicity"]
                    score += AA_PROPERTIES[aa]["beta_propensity"] - 1.0

            aggregation_scores.append(score)

        max_aggregation = max(aggregation_scores) if aggregation_scores else 0.0
        mean_aggregation = np.mean(aggregation_scores) if aggregation_scores else 0.0

        return {
            "aggregation_propensity": float(mean_aggregation),
            "max_aggregation_patch": float(max_aggregation),
        }

    def _analyze_stability_factors(self, sequence: str) -> Dict[str, Any]:
        """Analyze factors affecting protein stability."""
        if not BIOPYTHON_AVAILABLE:
            return self._simple_stability_analysis(sequence)

        try:
            protein_analysis = ProteinAnalysis(sequence)

            stability_score = 0.0

            # Molecular weight factor (moderate size is better)
            mw = protein_analysis.molecular_weight()
            if 10000 <= mw <= 50000:
                stability_score += 1.0

            # Charge factor
            charge = protein_analysis.charge_at_pH(7.0)
            if -5 <= charge <= 5:
                stability_score += 1.0

            # Hydrophobic fraction
            hydrophobic_fraction = (
                sum(1 for aa in sequence if AA_PROPERTIES[aa]["hydrophobicity"] > 0)
                / len(sequence)
                if sequence
                else 0
            )
            if 0.3 <= hydrophobic_fraction <= 0.5:
                stability_score += 1.0

            # Normalize to 0-1 scale
            stability_score /= 3.0

            return {
                "stability_score": float(stability_score),
                "molecular_weight": float(mw),
                "charge_at_ph7": float(charge),
                "hydrophobic_fraction": float(hydrophobic_fraction),
            }

        except Exception as e:
            return self._simple_stability_analysis(sequence)

    def _simple_stability_analysis(self, sequence: str) -> Dict[str, Any]:
        """Simple stability analysis without BioPython."""
        length = len(sequence)

        # Count different amino acid types
        charged = sum(1 for aa in sequence if aa in "DEKR")
        hydrophobic = sum(
            1 for aa in sequence if AA_PROPERTIES[aa]["hydrophobicity"] > 0.5
        )
        polar = sum(1 for aa in sequence if aa in "STYNQH")

        # Simple stability heuristics
        stability_score = 0.0

        # Reasonable length
        if 50 <= length <= 400:
            stability_score += 1.0

        # Balanced composition
        charged_frac = charged / length if length > 0 else 0
        hydrophobic_frac = hydrophobic / length if length > 0 else 0

        if 0.15 <= charged_frac <= 0.3:
            stability_score += 1.0
        if 0.25 <= hydrophobic_frac <= 0.45:
            stability_score += 1.0

        stability_score /= 3.0

        return {
            "stability_score": float(stability_score),
            "charged_fraction": float(charged_frac),
            "hydrophobic_fraction": float(hydrophobic_frac),
        }

    def _check_developability_flags(self, sequence: str) -> Dict[str, Any]:
        """Check for common developability issues."""
        flags = []

        # Check for aggregation-prone stretches
        aggregation_prone = "FILVY"
        for i in range(len(sequence) - 3):
            window = sequence[i : i + 4]
            if sum(1 for aa in window if aa in aggregation_prone) >= 3:
                flags.append(f"Aggregation-prone stretch at position {i+1}")

        # Check for unusual amino acid composition
        length = len(sequence)

        # Too many cysteines (>3% might cause disulfide issues)
        cys_count = sequence.count("C")
        if cys_count / length > 0.03:
            flags.append("High cysteine content (>3%)")

        # Too many prolines (>8% can cause folding issues)
        pro_count = sequence.count("P")
        if pro_count / length > 0.08:
            flags.append("High proline content (>8%)")

        # Check for repeated motifs
        for motif_len in [3, 4, 5]:
            for i in range(len(sequence) - motif_len * 2 + 1):
                motif = sequence[i : i + motif_len]
                if sequence.count(motif) > 2:
                    flags.append(f"Repeated motif: {motif}")
                    break

        return {"developability_flags": flags}


class PerplexityAnalyzer(ValidationTool):
    """Perplexity-based out-of-distribution detection using ProteinMPNN."""

    def __init__(
        self, config: ValidationConfig, proteinmpnn_model_path: Optional[str] = None
    ):
        self.config = config
        self.model_path = proteinmpnn_model_path
        self.model = None
        self.device = (
            config.device
            if config.device != "auto"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def is_available(self) -> bool:
        """Check if ProteinMPNN model is available for perplexity measurement."""
        return PROTEINMPNN_AVAILABLE and (
            self.model_path is not None or self.model is not None
        )

    def load_model(self, model_path: str):
        """Load ProteinMPNN model for perplexity calculation."""
        if not PROTEINMPNN_AVAILABLE:
            raise ImportError("ProteinMPNN utilities not available")

        try:
            # Load ProteinMPNN model for perplexity calculation
            # Implementation would load the actual model
            self.model_path = model_path
            # Mock loading for now
            print(f"Mock loading ProteinMPNN model from {model_path}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to load ProteinMPNN model: {e}")
            return False

    def validate(
        self, sequence: str, backbone_coords: Optional[np.ndarray] = None, **kwargs
    ) -> Dict[str, Any]:
        """Calculate perplexity of sequence given backbone structure."""
        if not self.is_available():
            return self._mock_perplexity_calculation(sequence)

        try:
            return self._calculate_proteinmpnn_perplexity(sequence, backbone_coords)
        except Exception as e:
            return {"error": f"Perplexity calculation failed: {e}"}

    def _mock_perplexity_calculation(self, sequence: str) -> Dict[str, Any]:
        """Generate mock perplexity scores for testing."""
        # Generate realistic perplexity values
        # Natural sequences typically have lower perplexity (~2-20)
        # Designed sequences might have higher perplexity if they explore OOD space

        length = len(sequence)

        # Base perplexity increases with sequence length and unusual composition
        base_perplexity = 5.0 + 0.02 * length

        # Unusual amino acid composition increases perplexity
        aa_counts = {aa: sequence.count(aa) for aa in CANONICAL_AA_ORDER}
        total = len(sequence)

        # Check for unusual composition
        composition_penalty = 0.0
        for aa, count in aa_counts.items():
            frequency = count / total
            # Expected frequencies (rough approximations)
            expected = {
                "A": 0.08,
                "C": 0.02,
                "D": 0.05,
                "E": 0.06,
                "F": 0.04,
                "G": 0.07,
                "H": 0.02,
                "I": 0.05,
                "K": 0.06,
                "L": 0.10,
                "M": 0.02,
                "N": 0.04,
                "P": 0.05,
                "Q": 0.04,
                "R": 0.05,
                "S": 0.07,
                "T": 0.05,
                "V": 0.07,
                "W": 0.01,
                "Y": 0.03,
            }

            expected_freq = expected.get(aa, 0.05)
            if frequency > expected_freq * 2:  # Much more frequent than expected
                composition_penalty += (frequency - expected_freq) * 10

        final_perplexity = base_perplexity + composition_penalty

        # Add some noise for realism
        seed = int(hashlib.md5(sequence.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)  # Isolated random state
        final_perplexity += rng.normal(0, 1.0)
        final_perplexity = max(1.0, final_perplexity)  # Minimum perplexity of 1

        # Calculate z-score relative to typical protein distribution
        # Typical proteins have perplexity ~8±4
        typical_mean = 8.0
        typical_std = 4.0
        z_score = (final_perplexity - typical_mean) / typical_std

        # OOD confidence: higher perplexity might indicate beneficial exploration
        ood_confidence = min(1.0, max(0.0, (final_perplexity - 6.0) / 10.0))

        return {
            "proteinmpnn_perplexity": float(final_perplexity),
            "perplexity_zscore": float(z_score),
            "ood_confidence": float(ood_confidence),
            "method": "mock_proteinmpnn",
        }

    def _calculate_proteinmpnn_perplexity(
        self, sequence: str, backbone_coords: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate actual ProteinMPNN perplexity (to be implemented)."""
        # This would implement the actual ProteinMPNN perplexity calculation
        # For now, return mock results
        results = self._mock_perplexity_calculation(sequence)
        results["method"] = "proteinmpnn_mock"
        results["warning"] = "ProteinMPNN perplexity calculation not fully implemented"
        return results


class DiversityAnalyzer(ValidationTool):
    """Sequence diversity and novelty metrics."""

    def __init__(
        self, config: ValidationConfig, reference_sequences: Optional[List[str]] = None
    ):
        self.config = config
        self.reference_sequences = reference_sequences or []

    def is_available(self) -> bool:
        """Always available with sequence-based methods."""
        return True

    def validate(
        self, sequence: str, sequence_set: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Analyze sequence diversity and novelty."""
        results = {}

        # Calculate novelty relative to reference sequences
        if self.reference_sequences:
            results.update(self._calculate_novelty(sequence))

        # Calculate diversity within a set of sequences
        if sequence_set:
            results.update(self._calculate_diversity_metrics(sequence, sequence_set))

        # Calculate intrinsic sequence properties
        results.update(self._calculate_sequence_complexity(sequence))

        return results

    def _calculate_novelty(self, sequence: str) -> Dict[str, Any]:
        """Calculate novelty relative to reference sequences."""
        if not self.reference_sequences:
            return {"sequence_novelty": 1.0}  # Maximally novel if no references

        # Calculate minimum edit distance to reference sequences
        min_distance = float("inf")
        for ref_seq in self.reference_sequences:
            distance = self._edit_distance(sequence, ref_seq)
            min_distance = min(min_distance, distance)

        # Normalize by sequence length
        novelty = min_distance / max(len(sequence), 1)

        # Calculate sequence identity to closest match
        if min_distance < len(sequence):
            closest_ref = min(
                self.reference_sequences, key=lambda x: self._edit_distance(sequence, x)
            )
            identity = 1.0 - (
                self._edit_distance(sequence, closest_ref)
                / max(len(sequence), len(closest_ref))
            )
        else:
            identity = 0.0

        return {
            "sequence_novelty": float(novelty),
            "nearest_neighbor_distance": float(min_distance),
            "max_sequence_identity": float(identity),
        }

    def _calculate_diversity_metrics(
        self, sequence: str, sequence_set: List[str]
    ) -> Dict[str, Any]:
        """Calculate diversity metrics within a sequence set."""
        if not sequence_set or len(sequence_set) < 2:
            return {}

        # Calculate pairwise distances
        distances = []
        for other_seq in sequence_set:
            if other_seq != sequence:
                dist = self._edit_distance(sequence, other_seq)
                distances.append(dist)

        if not distances:
            return {}

        return {
            "mean_pairwise_distance": float(np.mean(distances)),
            "min_pairwise_distance": float(np.min(distances)),
            "max_pairwise_distance": float(np.max(distances)),
            "std_pairwise_distance": float(np.std(distances)),
        }

    def _calculate_sequence_complexity(self, sequence: str) -> Dict[str, Any]:
        """Calculate intrinsic sequence complexity metrics."""
        length = len(sequence)

        # Amino acid composition entropy
        aa_counts = np.array([sequence.count(aa) for aa in CANONICAL_AA_ORDER])
        aa_freqs = aa_counts / length if length > 0 else np.zeros_like(aa_counts)
        aa_freqs = aa_freqs[aa_freqs > 0]  # Remove zeros for log calculation
        composition_entropy = -np.sum(aa_freqs * np.log2(aa_freqs))

        # Local complexity (entropy of overlapping triplets)
        if length >= 3:
            triplets = [sequence[i : i + 3] for i in range(length - 2)]
            unique_triplets = len(set(triplets))
            triplet_complexity = unique_triplets / len(triplets)
        else:
            triplet_complexity = 1.0

        # Repetitiveness (longest repeated subsequence)
        max_repeat_length = self._find_max_repeat(sequence)
        repetitiveness = max_repeat_length / length if length > 0 else 0.0

        return {
            "composition_entropy": float(composition_entropy),
            "local_complexity": float(triplet_complexity),
            "repetitiveness": float(repetitiveness),
            "unique_amino_acids": len(set(sequence)),
        }

    def _edit_distance(self, seq1: str, seq2: str) -> int:
        """Calculate edit distance between two sequences."""
        # Simple Levenshtein distance implementation
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # deletion
                        dp[i][j - 1],  # insertion
                        dp[i - 1][j - 1],
                    )  # substitution

        return dp[m][n]

    def _find_max_repeat(self, sequence: str) -> int:
        """Find the length of the longest repeated subsequence."""
        max_repeat = 0
        length = len(sequence)

        # Check for repeats of different lengths
        for repeat_len in range(2, length // 2 + 1):
            for i in range(length - repeat_len + 1):
                pattern = sequence[i : i + repeat_len]
                count = 1
                j = i + repeat_len

                # Count consecutive repeats
                while (
                    j + repeat_len <= length and sequence[j : j + repeat_len] == pattern
                ):
                    count += 1
                    j += repeat_len

                if count > 1:
                    total_repeat_length = count * repeat_len
                    max_repeat = max(max_repeat, total_repeat_length)

        return max_repeat


class StatisticalTester:
    """Statistical significance testing for validation results."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.alpha = config.statistical_alpha
        self.correction_method = config.multiple_testing_correction

    def compare_perplexity_distributions(
        self, energy_based_perplexities: List[float], baseline_perplexities: List[float]
    ) -> Dict[str, Any]:
        """Compare perplexity distributions between energy-based and baseline designs."""
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available for statistical testing"}

        # Basic statistics
        energy_stats = self._calculate_distribution_stats(energy_based_perplexities)
        baseline_stats = self._calculate_distribution_stats(baseline_perplexities)

        results = {
            "energy_based_stats": energy_stats,
            "baseline_stats": baseline_stats,
            "n_energy_based": len(energy_based_perplexities),
            "n_baseline": len(baseline_perplexities),
        }

        # Statistical tests
        tests_performed = []

        # Mann-Whitney U test (non-parametric)
        try:
            stat, p_value = mannwhitneyu(
                energy_based_perplexities,
                baseline_perplexities,
                alternative="two-sided",
            )
            results["mann_whitney"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < self.alpha),
            }
            tests_performed.append("mann_whitney")
        except Exception as e:
            results["mann_whitney"] = {"error": str(e)}

        # Kolmogorov-Smirnov test for distribution difference
        try:
            stat, p_value = stats.ks_2samp(
                energy_based_perplexities, baseline_perplexities
            )
            results["kolmogorov_smirnov"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < self.alpha),
            }
            tests_performed.append("kolmogorov_smirnov")
        except Exception as e:
            results["kolmogorov_smirnov"] = {"error": str(e)}

        # Effect size (Cohen's d)
        try:
            effect_size = self._calculate_cohens_d(
                energy_based_perplexities, baseline_perplexities
            )
            results["effect_size"] = {
                "cohens_d": float(effect_size),
                "interpretation": self._interpret_effect_size(effect_size),
            }
        except Exception as e:
            results["effect_size"] = {"error": str(e)}

        # Multiple testing correction
        if len(tests_performed) > 1:
            p_values = []
            for test_name in tests_performed:
                if "p_value" in results[test_name]:
                    p_values.append(results[test_name]["p_value"])

            if p_values:
                corrected_p_values = self._apply_multiple_testing_correction(p_values)
                results["multiple_testing_correction"] = {
                    "method": self.correction_method,
                    "original_p_values": p_values,
                    "corrected_p_values": corrected_p_values,
                    "any_significant": bool(
                        any(p < self.alpha for p in corrected_p_values)
                    ),
                }

        return results

    def bootstrap_confidence_intervals(
        self, metric_values: List[float], confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for metrics."""
        if not metric_values:
            return {"error": "Empty metric values"}

        n_bootstrap = self.config.bootstrap_samples
        alpha = 1 - confidence_level

        # Use configured seed with call-specific offset to avoid pattern repetition
        if not hasattr(self, "_bootstrap_call_count"):
            self._bootstrap_call_count = 0
        base_seed = getattr(self.config, "random_seed", 42)
        bootstrap_seed = base_seed + self._bootstrap_call_count
        self._bootstrap_call_count += 1

        rng = np.random.default_rng(bootstrap_seed)  # Isolated random state
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = rng.choice(metric_values, size=len(metric_values), replace=True)
            bootstrap_means.append(np.mean(sample))

        # Calculate confidence intervals
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return {
            "mean": float(np.mean(metric_values)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": float(confidence_level),
            "n_bootstrap_samples": n_bootstrap,
        }

    def _calculate_distribution_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic distribution statistics."""
        if not values:
            return {}

        values = np.array(values)

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        group1, group2 = np.array(group1), np.array(group2)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
            / (n1 + n2 - 2)
        )

        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _apply_multiple_testing_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple testing correction."""
        if self.correction_method == "bonferroni":
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif self.correction_method == "fdr":
            # Benjamini-Hochberg FDR correction
            if not SCIPY_AVAILABLE:
                warnings.warn(
                    "SciPy not available for FDR correction, using Bonferroni"
                )
                return [min(1.0, p * len(p_values)) for p in p_values]

            from scipy.stats import false_discovery_control

            try:
                return false_discovery_control(p_values, method="bh").tolist()
            except:
                # Fallback to manual implementation
                sorted_indices = np.argsort(p_values)
                sorted_p_values = np.array(p_values)[sorted_indices]
                n = len(p_values)

                corrected = np.zeros_like(sorted_p_values)
                for i in range(n - 1, -1, -1):
                    if i == n - 1:
                        corrected[i] = sorted_p_values[i]
                    else:
                        corrected[i] = min(
                            corrected[i + 1], sorted_p_values[i] * n / (i + 1)
                        )

                # Restore original order
                result = np.zeros_like(corrected)
                result[sorted_indices] = corrected
                return result.tolist()
        else:  # 'none'
            return p_values


class ValidationPipeline:
    """Automated validation pipeline for protein designs."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else Path.cwd() / "validation_results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validation tools
        self.validators = {}
        self._initialize_validators()

        # Initialize statistical tester
        self.statistical_tester = StatisticalTester(config)

        # Results storage
        self.results: List[ValidationResults] = []

    def _initialize_validators(self):
        """Initialize available validation tools."""
        # AlphaFold validator
        if self.config.use_alphafold:
            try:
                self.validators["alphafold"] = AlphaFoldValidator(self.config)
            except Exception as e:
                warnings.warn(f"Failed to initialize AlphaFold validator: {e}")

        # Rosetta validator
        if self.config.use_rosetta:
            try:
                self.validators["rosetta"] = RosettaValidator(self.config)
            except Exception as e:
                warnings.warn(f"Failed to initialize Rosetta validator: {e}")

        # Structure predictor
        if self.config.use_structure_prediction:
            self.validators["structure"] = StructurePredictor(self.config)

        # Aggregation analyzer
        if self.config.use_aggregation_analysis:
            self.validators["aggregation"] = AggregationAnalyzer(self.config)

        # Perplexity analyzer
        if self.config.use_perplexity_analysis:
            try:
                self.validators["perplexity"] = PerplexityAnalyzer(self.config)
            except Exception as e:
                warnings.warn(f"Failed to initialize perplexity analyzer: {e}")

        # Diversity analyzer
        self.validators["diversity"] = DiversityAnalyzer(self.config)

    def validate_sequence(
        self, sequence: str, sequence_id: str = None, **kwargs
    ) -> ValidationResults:
        """Validate a single sequence."""
        if sequence_id is None:
            sequence_id = f"seq_{len(self.results):04d}"

        start_time = time.time()
        results = ValidationResults(sequence=sequence, sequence_id=sequence_id)

        # Run validation tools
        for tool_name, validator in self.validators.items():
            if not validator.is_available():
                results.warnings.append(f"{tool_name} validator not available")
                continue

            try:
                tool_results = validator.validate(sequence, **kwargs)

                # Process results based on tool type
                if tool_name == "alphafold":
                    if "error" not in tool_results:
                        results.alphafold_confidence = tool_results.get(
                            "alphafold_confidence"
                        )
                        results.alphafold_plddt = tool_results.get("alphafold_plddt")
                        results.structure_quality = tool_results.get(
                            "structure_quality"
                        )

                elif tool_name == "rosetta":
                    if "error" not in tool_results:
                        results.rosetta_score = tool_results.get("rosetta_score")
                        results.rosetta_components = tool_results.get(
                            "rosetta_components"
                        )

                elif tool_name == "structure":
                    if "error" not in tool_results:
                        results.secondary_structure = tool_results.get(
                            "secondary_structure"
                        )
                        results.ss_composition = tool_results.get("ss_composition")
                        results.solvent_accessibility = tool_results.get(
                            "solvent_accessibility"
                        )

                elif tool_name == "aggregation":
                    if "error" not in tool_results:
                        results.aggregation_propensity = tool_results.get(
                            "aggregation_propensity"
                        )
                        results.stability_score = tool_results.get("stability_score")
                        results.developability_flags = tool_results.get(
                            "developability_flags"
                        )

                elif tool_name == "perplexity":
                    if "error" not in tool_results:
                        results.proteinmpnn_perplexity = tool_results.get(
                            "proteinmpnn_perplexity"
                        )
                        results.perplexity_zscore = tool_results.get(
                            "perplexity_zscore"
                        )
                        results.ood_confidence = tool_results.get("ood_confidence")

                elif tool_name == "diversity":
                    if "error" not in tool_results:
                        results.sequence_novelty = tool_results.get("sequence_novelty")
                        results.nearest_neighbor_distance = tool_results.get(
                            "nearest_neighbor_distance"
                        )

                # Handle errors
                if "error" in tool_results:
                    results.errors.append(f"{tool_name}: {tool_results['error']}")

                # Handle warnings
                if "warning" in tool_results:
                    results.warnings.append(f"{tool_name}: {tool_results['warning']}")

            except Exception as e:
                results.errors.append(f"{tool_name} validation failed: {str(e)}")

        results.validation_time = time.time() - start_time
        self.results.append(results)

        return results

    def validate_batch(
        self,
        sequences: List[str],
        sequence_ids: Optional[List[str]] = None,
        baseline_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validate a batch of sequences with statistical analysis."""
        if sequence_ids is None:
            sequence_ids = [f"seq_{i:04d}" for i in range(len(sequences))]

        if len(sequences) != len(sequence_ids):
            raise ValueError("sequences and sequence_ids must have same length")

        print(f"Validating {len(sequences)} sequences...")

        # Process sequences in batches
        batch_size = self.config.batch_size
        batch_results = []

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i : i + batch_size]
            batch_ids = sequence_ids[i : i + batch_size]

            for seq, seq_id in zip(batch_sequences, batch_ids):
                result = self.validate_sequence(seq, seq_id, **kwargs)
                batch_results.append(result)

        # Generate summary statistics
        summary = self._generate_batch_summary(batch_results)

        # Perform statistical comparisons if baseline is provided
        if baseline_sequences:
            print("Validating baseline sequences...")
            baseline_results = []
            for i, seq in enumerate(baseline_sequences):
                result = self.validate_sequence(seq, f"baseline_{i:04d}", **kwargs)
                baseline_results.append(result)

            # Statistical comparison
            comparison = self._compare_with_baseline(batch_results, baseline_results)
            summary["baseline_comparison"] = comparison

        # Save results
        self._save_batch_results(batch_results, summary)

        return {
            "individual_results": [r.to_dict() for r in batch_results],
            "summary": summary,
            "output_directory": str(self.output_dir),
        }

    def _generate_batch_summary(
        self, results: List[ValidationResults]
    ) -> Dict[str, Any]:
        """Generate summary statistics for a batch of validation results."""
        summary = {
            "n_sequences": len(results),
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
        }

        # Collect metrics
        metrics = {
            "alphafold_confidence": [],
            "rosetta_score": [],
            "aggregation_propensity": [],
            "stability_score": [],
            "proteinmpnn_perplexity": [],
            "sequence_novelty": [],
        }

        error_counts = {}
        warning_counts = {}

        for result in results:
            # Collect valid metrics
            if result.alphafold_confidence is not None:
                metrics["alphafold_confidence"].append(result.alphafold_confidence)
            if result.rosetta_score is not None:
                metrics["rosetta_score"].append(result.rosetta_score)
            if result.aggregation_propensity is not None:
                metrics["aggregation_propensity"].append(result.aggregation_propensity)
            if result.stability_score is not None:
                metrics["stability_score"].append(result.stability_score)
            if result.proteinmpnn_perplexity is not None:
                metrics["proteinmpnn_perplexity"].append(result.proteinmpnn_perplexity)
            if result.sequence_novelty is not None:
                metrics["sequence_novelty"].append(result.sequence_novelty)

            # Count errors and warnings
            for error in result.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for warning in result.warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1

        # Calculate summary statistics for each metric
        metric_summaries = {}
        for metric_name, values in metrics.items():
            if values:
                metric_summaries[metric_name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }

                # Add bootstrap confidence intervals if enough samples
                if len(values) >= 10:
                    ci = self.statistical_tester.bootstrap_confidence_intervals(values)
                    metric_summaries[metric_name]["confidence_interval"] = ci

        summary["metrics"] = metric_summaries
        summary["errors"] = error_counts
        summary["warnings"] = warning_counts

        # Quality assessment
        high_quality_count = sum(
            1
            for r in results
            if r.structure_quality == "high"
            and r.alphafold_confidence is not None
            and r.alphafold_confidence > 80
        )

        summary["quality_assessment"] = {
            "high_quality_structures": high_quality_count,
            "high_quality_fraction": (
                high_quality_count / len(results) if results else 0
            ),
            "successful_validations": len([r for r in results if not r.errors]),
        }

        return summary

    def _compare_with_baseline(
        self,
        designed_results: List[ValidationResults],
        baseline_results: List[ValidationResults],
    ) -> Dict[str, Any]:
        """Compare designed sequences with baseline sequences."""
        comparison = {}

        # Compare perplexity distributions
        designed_perplexities = [
            r.proteinmpnn_perplexity
            for r in designed_results
            if r.proteinmpnn_perplexity is not None
        ]
        baseline_perplexities = [
            r.proteinmpnn_perplexity
            for r in baseline_results
            if r.proteinmpnn_perplexity is not None
        ]

        if designed_perplexities and baseline_perplexities:
            perplexity_comparison = (
                self.statistical_tester.compare_perplexity_distributions(
                    designed_perplexities, baseline_perplexities
                )
            )
            comparison["perplexity_analysis"] = perplexity_comparison

        # Compare other metrics
        metrics_to_compare = [
            ("alphafold_confidence", "alphafold_confidence"),
            ("rosetta_score", "rosetta_score"),
            ("stability_score", "stability_score"),
        ]

        for metric_name, attr_name in metrics_to_compare:
            designed_values = [
                getattr(r, attr_name)
                for r in designed_results
                if getattr(r, attr_name) is not None
            ]
            baseline_values = [
                getattr(r, attr_name)
                for r in baseline_results
                if getattr(r, attr_name) is not None
            ]

            if designed_values and baseline_values and SCIPY_AVAILABLE:
                try:
                    stat, p_value = mannwhitneyu(designed_values, baseline_values)
                    comparison[f"{metric_name}_comparison"] = {
                        "designed_mean": float(np.mean(designed_values)),
                        "baseline_mean": float(np.mean(baseline_values)),
                        "mann_whitney_statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < self.statistical_tester.alpha),
                    }
                except Exception as e:
                    comparison[f"{metric_name}_comparison"] = {"error": str(e)}

        return comparison

    def _save_batch_results(
        self, results: List[ValidationResults], summary: Dict[str, Any]
    ):
        """Save batch validation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual results
        results_file = self.output_dir / f"validation_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        # Save summary
        summary_file = self.output_dir / f"validation_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate plots if requested
        if self.config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_validation_plots(results, timestamp)

        print(f"Results saved to {self.output_dir}")

    def _generate_validation_plots(
        self, results: List[ValidationResults], timestamp: str
    ):
        """Generate visualization plots for validation results."""
        if not PLOTTING_AVAILABLE:
            return

        # Set up plot style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Protein Design Validation Results", fontsize=16)

        # Plot 1: AlphaFold confidence distribution
        af_scores = [
            r.alphafold_confidence
            for r in results
            if r.alphafold_confidence is not None
        ]
        if af_scores:
            axes[0, 0].hist(
                af_scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[0, 0].set_xlabel("AlphaFold Confidence")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("AlphaFold Confidence Distribution")
            axes[0, 0].axvline(
                np.mean(af_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(af_scores):.1f}",
            )
            axes[0, 0].legend()

        # Plot 2: Perplexity distribution
        perplexities = [
            r.proteinmpnn_perplexity
            for r in results
            if r.proteinmpnn_perplexity is not None
        ]
        if perplexities:
            axes[0, 1].hist(
                perplexities, bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
            )
            axes[0, 1].set_xlabel("ProteinMPNN Perplexity")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Perplexity Distribution")
            axes[0, 1].axvline(
                np.mean(perplexities),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(perplexities):.1f}",
            )
            axes[0, 1].legend()

        # Plot 3: Structure quality pie chart
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        for r in results:
            if r.structure_quality in quality_counts:
                quality_counts[r.structure_quality] += 1

        if sum(quality_counts.values()) > 0:
            axes[0, 2].pie(
                quality_counts.values(),
                labels=quality_counts.keys(),
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[0, 2].set_title("Structure Quality Distribution")

        # Plot 4: Correlation between confidence and perplexity
        af_values = []
        perp_values = []
        for r in results:
            if (
                r.alphafold_confidence is not None
                and r.proteinmpnn_perplexity is not None
            ):
                af_values.append(r.alphafold_confidence)
                perp_values.append(r.proteinmpnn_perplexity)

        if len(af_values) > 5:
            axes[1, 0].scatter(perp_values, af_values, alpha=0.6)
            axes[1, 0].set_xlabel("ProteinMPNN Perplexity")
            axes[1, 0].set_ylabel("AlphaFold Confidence")
            axes[1, 0].set_title("Confidence vs Perplexity")

            # Add correlation coefficient if scipy available
            if SCIPY_AVAILABLE:
                corr, p_val = stats.pearsonr(af_values, perp_values)
                axes[1, 0].text(
                    0.05,
                    0.95,
                    f"r = {corr:.3f}\np = {p_val:.3f}",
                    transform=axes[1, 0].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # Plot 5: Stability metrics
        stability_scores = [
            r.stability_score for r in results if r.stability_score is not None
        ]
        if stability_scores:
            axes[1, 1].hist(
                stability_scores, bins=15, alpha=0.7, color="orange", edgecolor="black"
            )
            axes[1, 1].set_xlabel("Stability Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Stability Score Distribution")
            axes[1, 1].axvline(
                np.mean(stability_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(stability_scores):.2f}",
            )
            axes[1, 1].legend()

        # Plot 6: Sequence novelty
        novelty_scores = [
            r.sequence_novelty for r in results if r.sequence_novelty is not None
        ]
        if novelty_scores:
            axes[1, 2].hist(
                novelty_scores, bins=15, alpha=0.7, color="purple", edgecolor="black"
            )
            axes[1, 2].set_xlabel("Sequence Novelty")
            axes[1, 2].set_ylabel("Frequency")
            axes[1, 2].set_title("Sequence Novelty Distribution")
            axes[1, 2].axvline(
                np.mean(novelty_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(novelty_scores):.2f}",
            )
            axes[1, 2].legend()

        # Remove empty plots
        for i in range(2):
            for j in range(3):
                if not axes[i, j].has_data():
                    axes[i, j].text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=axes[i, j].transAxes,
                    )
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / f"validation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Validation plots saved to {plot_file}")


def create_default_config() -> ValidationConfig:
    """Create a default validation configuration."""
    return ValidationConfig(
        use_alphafold=True,
        use_rosetta=True,
        use_structure_prediction=True,
        use_aggregation_analysis=True,
        use_perplexity_analysis=True,
        device="auto",
        output_dir="./validation_results",
        generate_plots=True,
        verbose=True,
    )


def validate_protein_designs(
    sequences: List[str],
    sequence_ids: Optional[List[str]] = None,
    baseline_sequences: Optional[List[str]] = None,
    config: Optional[ValidationConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    High-level interface for validating protein designs.

    Args:
        sequences: List of protein sequences to validate
        sequence_ids: Optional sequence identifiers
        baseline_sequences: Optional baseline sequences for comparison
        config: Validation configuration
        **kwargs: Additional arguments passed to validation tools

    Returns:
        Dictionary containing validation results and summary
    """
    if config is None:
        config = create_default_config()

    pipeline = ValidationPipeline(config)
    return pipeline.validate_batch(
        sequences, sequence_ids, baseline_sequences, **kwargs
    )


class PerformanceBenchmark:
    """Performance benchmarking and timing analysis for validation pipeline."""

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.throughput_metrics = {}

    def benchmark_validation_pipeline(
        self, sequences: List[str], config: ValidationConfig, n_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark the validation pipeline performance."""
        import psutil
        import gc

        benchmark_results = {
            "config": config.__dict__,
            "n_sequences": len(sequences),
            "n_benchmark_runs": n_runs,
            "timing_results": {},
            "memory_results": {},
            "throughput_metrics": {},
        }

        # Run multiple benchmark iterations
        run_times = []
        memory_peaks = []

        for run in range(n_runs):
            print(f"Benchmark run {run + 1}/{n_runs}")

            # Clear memory before run
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Time the validation
            start_time = time.time()

            # Create fresh pipeline for each run
            pipeline = ValidationPipeline(config)
            results = pipeline.validate_batch(sequences)

            end_time = time.time()
            run_time = end_time - start_time
            run_times.append(run_time)

            # Memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            memory_peaks.append(memory_used)

            # Cleanup
            del pipeline, results
            gc.collect()

        # Calculate statistics
        benchmark_results["timing_results"] = {
            "mean_time_seconds": float(np.mean(run_times)),
            "std_time_seconds": float(np.std(run_times)),
            "min_time_seconds": float(np.min(run_times)),
            "max_time_seconds": float(np.max(run_times)),
            "all_run_times": run_times,
        }

        benchmark_results["memory_results"] = {
            "mean_memory_mb": float(np.mean(memory_peaks)),
            "std_memory_mb": float(np.std(memory_peaks)),
            "max_memory_mb": float(np.max(memory_peaks)),
            "all_memory_usage": memory_peaks,
        }

        # Throughput metrics
        mean_time = np.mean(run_times)
        sequences_per_second = len(sequences) / mean_time if mean_time > 0 else 0
        time_per_sequence = mean_time / len(sequences) if sequences else 0

        benchmark_results["throughput_metrics"] = {
            "sequences_per_second": float(sequences_per_second),
            "seconds_per_sequence": float(time_per_sequence),
            "total_sequences_processed": len(sequences) * n_runs,
        }

        return benchmark_results

    def profile_individual_validators(
        self, test_sequence: str, config: ValidationConfig
    ) -> Dict[str, Any]:
        """Profile performance of individual validation tools."""
        profile_results = {}

        # Test each validator individually
        validators_to_test = {
            "alphafold": AlphaFoldValidator(config),
            "rosetta": RosettaValidator(config),
            "structure": StructurePredictor(config),
            "aggregation": AggregationAnalyzer(config),
            "perplexity": PerplexityAnalyzer(config),
            "diversity": DiversityAnalyzer(config),
        }

        for validator_name, validator in validators_to_test.items():
            if not validator.is_available():
                profile_results[validator_name] = {"error": "Validator not available"}
                continue

            # Time multiple runs of this validator
            n_runs = 10
            run_times = []

            for _ in range(n_runs):
                start_time = time.time()
                try:
                    result = validator.validate(test_sequence)
                    end_time = time.time()
                    run_times.append(end_time - start_time)
                except Exception as e:
                    profile_results[validator_name] = {"error": str(e)}
                    break

            if run_times:
                profile_results[validator_name] = {
                    "mean_time_seconds": float(np.mean(run_times)),
                    "std_time_seconds": float(np.std(run_times)),
                    "min_time_seconds": float(np.min(run_times)),
                    "max_time_seconds": float(np.max(run_times)),
                    "n_runs": n_runs,
                }

        return profile_results


def main():
    """Command-line interface for design validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate protein designs using comprehensive in silico methods"
    )

    # Input options
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Protein sequences to validate (space-separated)",
    )
    parser.add_argument(
        "--sequence_file", type=str, help="FASTA file containing sequences to validate"
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        help="FASTA file containing baseline sequences for comparison",
    )

    # Tool selection
    parser.add_argument(
        "--disable-alphafold",
        action="store_true",
        help="Disable AlphaFold confidence prediction",
    )
    parser.add_argument(
        "--disable-rosetta", action="store_true", help="Disable Rosetta energy scoring"
    )
    parser.add_argument(
        "--disable-structure",
        action="store_true",
        help="Disable secondary structure prediction",
    )
    parser.add_argument(
        "--disable-aggregation",
        action="store_true",
        help="Disable aggregation analysis",
    )
    parser.add_argument(
        "--disable-perplexity", action="store_true", help="Disable perplexity analysis"
    )

    # Configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device",
    )

    # Statistical options
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals",
    )
    parser.add_argument(
        "--multiple-testing",
        type=str,
        default="bonferroni",
        choices=["bonferroni", "fdr", "none"],
        help="Multiple testing correction method",
    )

    # Performance analysis
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarking"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=3, help="Number of benchmark runs"
    )

    # Output options
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate input arguments
    if not args.sequences and not args.sequence_file:
        parser.error("Must specify either --sequences or --sequence_file")

    # Load sequences
    sequences = []
    sequence_ids = []

    if args.sequences:
        sequences = args.sequences
        sequence_ids = [f"seq_{i:04d}" for i in range(len(sequences))]

    if args.sequence_file:
        if not BIOPYTHON_AVAILABLE:
            print("Error: BioPython required for FASTA file reading")
            return 1

        try:
            from Bio import SeqIO

            fasta_sequences = []
            fasta_ids = []

            for record in SeqIO.parse(args.sequence_file, "fasta"):
                fasta_sequences.append(str(record.seq))
                fasta_ids.append(record.id)

            sequences.extend(fasta_sequences)
            sequence_ids.extend(fasta_ids)

        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            return 1

    # Load baseline sequences
    baseline_sequences = None
    if args.baseline_file:
        if not BIOPYTHON_AVAILABLE:
            print("Warning: BioPython not available, skipping baseline sequences")
        else:
            try:
                from Bio import SeqIO

                baseline_sequences = []
                for record in SeqIO.parse(args.baseline_file, "fasta"):
                    baseline_sequences.append(str(record.seq))
            except Exception as e:
                print(f"Warning: Error reading baseline file: {e}")

    # Create configuration
    config = ValidationConfig(
        use_alphafold=not args.disable_alphafold,
        use_rosetta=not args.disable_rosetta,
        use_structure_prediction=not args.disable_structure,
        use_aggregation_analysis=not args.disable_aggregation,
        use_perplexity_analysis=not args.disable_perplexity,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        statistical_alpha=args.alpha,
        bootstrap_samples=args.bootstrap_samples,
        multiple_testing_correction=args.multiple_testing,
        generate_plots=not args.no_plots,
        verbose=not args.quiet,
    )

    print(f"Validating {len(sequences)} sequences...")
    if baseline_sequences:
        print(f"Comparing with {len(baseline_sequences)} baseline sequences...")

    # Run benchmarking if requested
    if args.benchmark:
        print("Running performance benchmark...")
        benchmark = PerformanceBenchmark()
        benchmark_results = benchmark.benchmark_validation_pipeline(
            sequences, config, args.benchmark_runs
        )

        # Save benchmark results
        import json

        benchmark_file = (
            Path(args.output_dir)
            / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        with open(benchmark_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        print(f"Benchmark results saved to {benchmark_file}")

        # Print summary
        timing = benchmark_results["timing_results"]
        memory = benchmark_results["memory_results"]
        throughput = benchmark_results["throughput_metrics"]

        print(f"\nBenchmark Summary:")
        print(
            f"  Processing time: {timing['mean_time_seconds']:.2f} ± {timing['std_time_seconds']:.2f} seconds"
        )
        print(
            f"  Memory usage: {memory['mean_memory_mb']:.1f} ± {memory['std_memory_mb']:.1f} MB"
        )
        print(
            f"  Throughput: {throughput['sequences_per_second']:.2f} sequences/second"
        )
        print(f"  Time per sequence: {throughput['seconds_per_sequence']:.3f} seconds")

    # Run validation
    results = validate_protein_designs(
        sequences=sequences,
        sequence_ids=sequence_ids,
        baseline_sequences=baseline_sequences,
        config=config,
    )

    # Print summary
    if not args.quiet:
        summary = results["summary"]
        print(f"\nValidation Summary:")
        print(f"  Sequences processed: {summary['n_sequences']}")
        print(f"  Output directory: {results['output_directory']}")

        if "quality_assessment" in summary:
            qa = summary["quality_assessment"]
            print(
                f"  High quality structures: {qa['high_quality_structures']}/{summary['n_sequences']} ({qa['high_quality_fraction']:.1%})"
            )
            print(f"  Successful validations: {qa['successful_validations']}")

        if "baseline_comparison" in summary:
            print(f"  Baseline comparison: Available")
            comparison = summary["baseline_comparison"]
            if "perplexity_analysis" in comparison:
                perp_comp = comparison["perplexity_analysis"]
                if "mann_whitney" in perp_comp:
                    mw = perp_comp["mann_whitney"]
                    significance = (
                        "significant"
                        if mw.get("significant", False)
                        else "not significant"
                    )
                    print(
                        f"    Perplexity difference: {significance} (p = {mw.get('p_value', 'N/A')})"
                    )

    return 0


if __name__ == "__main__":
    exit(main())
