"""
Validation Helper Utilities for ProteinMPNN Encoder Integration

This module provides reusable utilities for validating the hybrid ProteinMPNN encoder
integration, supporting the comprehensive Stage 1 validation strategy.

Key functions:
- PDB parsing and batch conversion
- Vocabulary consistency validation  
- Numerical stability checks
- Reproducibility verification utilities
- Reference comparison helpers
"""

import torch
import numpy as np
import random
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add proteinmpnn to path
project_root = Path(__file__).parent.parent.parent
proteinmpnn_path = project_root / "proteinmpnn"
sys.path.append(str(proteinmpnn_path))

try:
    from protein_mpnn_utils import parse_PDB, ProteinMPNN, gather_nodes
except ImportError as e:
    logger.error(f"Could not import ProteinMPNN utilities: {e}")
    raise


def setup_deterministic_environment(seed: int = 42) -> None:
    """
    Setup deterministic random state for reproducibility testing.
    
    Args:
        seed: Random seed for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def validate_vocabulary_consistency(encoder, reference_model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """
    Validate amino acid vocabulary ordering consistency across encoder components.
    
    Args:
        encoder: ProteinMPNN encoder instance
        reference_model: Optional reference ProteinMPNN model for comparison
        
    Returns:
        Dict with validation results and any identified issues
    """
    results = {
        "passed": True,
        "issues": [],
        "vocab_size": None,
        "aa_ordering_consistent": True
    }
    
    try:
        # Check encoder vocabulary size
        if hasattr(encoder.full_model, 'num_letters'):
            vocab_size = encoder.full_model.num_letters
            results["vocab_size"] = vocab_size
            
            # Standard ProteinMPNN uses 21 (20 AAs + X)
            if vocab_size != 21:
                results["passed"] = False
                results["issues"].append(f"Unexpected vocab size: {vocab_size}, expected 21")
        
        # Check decoder vocabulary consistency
        if hasattr(encoder.full_model, 'vocab'):
            decoder_vocab = encoder.full_model.vocab
            if decoder_vocab != vocab_size:
                results["passed"] = False
                results["issues"].append(f"Encoder/decoder vocab mismatch: {vocab_size} vs {decoder_vocab}")
        
        # If reference model provided, compare vocabularies
        if reference_model is not None:
            if hasattr(reference_model, 'num_letters'):
                ref_vocab = reference_model.num_letters
                if ref_vocab != vocab_size:
                    results["passed"] = False
                    results["issues"].append(f"Reference vocab mismatch: {vocab_size} vs {ref_vocab}")
    
    except Exception as e:
        results["passed"] = False
        results["issues"].append(f"Vocabulary validation error: {e}")
    
    return results


def check_numerical_stability(features: torch.Tensor, structure_name: str = "unknown") -> Dict[str, Any]:
    """
    Check numerical stability of feature tensors.
    
    Args:
        features: Feature tensor to validate [B, L, D]
        structure_name: Name of structure for error reporting
        
    Returns:
        Dict with stability check results
    """
    results = {
        "passed": True,
        "issues": [],
        "statistics": {}
    }
    
    try:
        # Check for NaN values
        nan_count = torch.isnan(features).sum().item()
        if nan_count > 0:
            results["passed"] = False
            results["issues"].append(f"Found {nan_count} NaN values in {structure_name}")
        
        # Check for Inf values
        inf_count = torch.isinf(features).sum().item()
        if inf_count > 0:
            results["passed"] = False
            results["issues"].append(f"Found {inf_count} Inf values in {structure_name}")
        
        # Check value ranges are reasonable
        feature_min = features.min().item()
        feature_max = features.max().item()
        feature_std = features.std().item()
        feature_mean = features.mean().item()
        
        results["statistics"] = {
            "min": feature_min,
            "max": feature_max, 
            "mean": feature_mean,
            "std": feature_std
        }
        
        # Reasonable range check (protein features should not be extreme)
        if feature_min < -100 or feature_max > 100:
            results["passed"] = False
            results["issues"].append(f"Features out of reasonable range [{feature_min:.2f}, {feature_max:.2f}] for {structure_name}")
        
        # Check that features are not constant
        if feature_std < 1e-6:
            results["passed"] = False
            results["issues"].append(f"Features appear constant (std={feature_std:.2e}) for {structure_name}")
        
        # Check for reasonable dynamic range
        if feature_std > 50:
            results["issues"].append(f"Large feature variance (std={feature_std:.2f}) for {structure_name} - may indicate instability")
    
    except Exception as e:
        results["passed"] = False
        results["issues"].append(f"Numerical stability check error: {e}")
    
    return results


def convert_parsed_pdb_to_batch(parsed_pdb: List, chain_id: Optional[str] = None, ca_only: bool = False) -> Dict[str, torch.Tensor]:
    """
    Convert ProteinMPNN parsed PDB format to encoder batch format.
    
    Args:
        parsed_pdb: Output from ProteinMPNN's parse_PDB function (list of dicts)
        chain_id: Specific chain to extract (if None, use first chain)
        ca_only: If True, extract only CA coordinates [B,L,3]; if False, extract N,CA,C,O [B,L,4,3]
        
    Returns:
        Batch dictionary with tensors for encoder input
    """
    try:
        # ProteinMPNN parse_PDB returns a list with one dict per structure
        if not parsed_pdb or not isinstance(parsed_pdb, list):
            raise ValueError("Invalid parsed PDB format - expected non-empty list")
        
        # Get the first (and typically only) structure
        structure_data = parsed_pdb[0]
        
        # Find available chains
        coord_keys = [k for k in structure_data.keys() if k.startswith('coords_chain_')]
        if not coord_keys:
            raise ValueError("No coordinate data found in parsed PDB")
        
        # Select chain
        if chain_id is None:
            # Use first available chain
            coords_key = coord_keys[0]
            chain_id = coords_key.replace('coords_chain_', '')
        else:
            coords_key = f'coords_chain_{chain_id}'
            if coords_key not in structure_data:
                available_chains = [k.replace('coords_chain_', '') for k in coord_keys]
                raise ValueError(f"Chain {chain_id} not found. Available chains: {available_chains}")
        
        # Extract coordinates
        chain_coords = structure_data[coords_key]
        
        if ca_only:
            # CA-only models expect only CA coordinates [L, 3]
            ca_key = f'CA_chain_{chain_id}'
            if ca_key not in chain_coords:
                raise ValueError(f"Missing CA coordinates for chain {chain_id}")
            
            ca_coords = chain_coords[ca_key]
            coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)  # [L, 3]
        else:
            # Full backbone models expect N, CA, C, O coordinates [L, 4, 3]
            atom_types = ['N', 'CA', 'C', 'O']
            coordinates = []
            
            for atom_type in atom_types:
                atom_key = f'{atom_type}_chain_{chain_id}'
                if atom_key not in chain_coords:
                    raise ValueError(f"Missing {atom_type} coordinates for chain {chain_id}")
                
                atom_coords = chain_coords[atom_key]
                coordinates.append(atom_coords)
            
            # Convert to tensor: [L, 4, 3] where L is sequence length
            coords_tensor = torch.tensor(coordinates, dtype=torch.float32)  # [4, L, 3]
            coords_tensor = coords_tensor.transpose(0, 1)  # [L, 4, 3]
        
        seq_length = coords_tensor.shape[0]
        
        # Create batch tensors
        batch = {
            'X': coords_tensor.unsqueeze(0),  # [1, L, 3] for CA-only or [1, L, 4, 3] for full backbone
            'mask': torch.ones(1, seq_length),  # [1, L]
            'residue_idx': torch.arange(seq_length).unsqueeze(0),  # [1, L]
            'chain_encoding_all': torch.zeros(1, seq_length)  # [1, L] - single chain
        }
        
        return batch
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert parsed PDB to batch: {e}")


def create_test_batch(seq_length: int, batch_size: int = 1, add_noise: bool = False) -> Dict[str, torch.Tensor]:
    """
    Create dummy batch for testing with realistic protein geometry.
    
    Args:
        seq_length: Number of residues
        batch_size: Batch size
        add_noise: Whether to add realistic coordinate noise
        
    Returns:
        Batch dictionary with dummy protein-like coordinates
    """
    # Create realistic protein backbone coordinates
    # Start with idealized backbone geometry
    coords = torch.zeros(batch_size, seq_length, 4, 3)
    
    for i in range(seq_length):
        # Idealized backbone coordinates with ~3.8Å CA-CA distance
        phi = i * 1.5  # Rough phi angle progression
        
        # N atom
        coords[:, i, 0, :] = torch.tensor([i * 3.8, 0.0, 0.0])
        # CA atom  
        coords[:, i, 1, :] = torch.tensor([i * 3.8 + 1.0, 0.5, 0.0]) 
        # C atom
        coords[:, i, 2, :] = torch.tensor([i * 3.8 + 2.0, 0.0, 0.0])
        # O atom
        coords[:, i, 3, :] = torch.tensor([i * 3.8 + 2.5, 1.0, 0.0])
    
    if add_noise:
        # Add realistic coordinate noise (±0.1Å)
        noise = torch.randn_like(coords) * 0.1
        coords += noise
    
    batch = {
        'X': coords,
        'mask': torch.ones(batch_size, seq_length),
        'residue_idx': torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1),
        'chain_encoding_all': torch.zeros(batch_size, seq_length)
    }
    
    return batch


def compare_encoder_outputs(
    encoder_features: torch.Tensor, 
    reference_features: torch.Tensor, 
    structure_name: str = "unknown",
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Compare encoder outputs against reference with detailed diagnostics.
    
    Args:
        encoder_features: Features from hybrid encoder [B, L, D]
        reference_features: Features from reference ProteinMPNN [B, L, D] 
        structure_name: Structure name for reporting
        tolerance: Relative tolerance for comparison
        
    Returns:
        Comparison results with diagnostics
    """
    results = {
        "passed": True,
        "issues": [],
        "statistics": {}
    }
    
    try:
        # Shape comparison
        if encoder_features.shape != reference_features.shape:
            results["passed"] = False
            results["issues"].append(
                f"Shape mismatch for {structure_name}: "
                f"{encoder_features.shape} vs {reference_features.shape}"
            )
            return results
        
        # Compute differences
        abs_diff = torch.abs(encoder_features - reference_features)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        
        # Relative difference
        reference_magnitude = torch.max(torch.abs(reference_features)).item()
        max_rel_diff = max_abs_diff / (reference_magnitude + 1e-8)
        mean_rel_diff = mean_abs_diff / (reference_magnitude + 1e-8)
        
        results["statistics"] = {
            "max_absolute_diff": max_abs_diff,
            "mean_absolute_diff": mean_abs_diff,
            "max_relative_diff": max_rel_diff,
            "mean_relative_diff": mean_rel_diff,
            "reference_magnitude": reference_magnitude
        }
        
        # Tolerance check
        if max_rel_diff > tolerance:
            results["passed"] = False
            results["issues"].append(
                f"Output mismatch for {structure_name}: "
                f"max_relative_diff={max_rel_diff:.2e} > tolerance={tolerance}"
            )
        
        # Additional diagnostics
        correlation = torch.corrcoef(
            torch.stack([encoder_features.flatten(), reference_features.flatten()])
        )[0, 1].item()
        
        results["statistics"]["correlation"] = correlation
        
        if correlation < 0.99:
            results["issues"].append(
                f"Low correlation for {structure_name}: {correlation:.4f} < 0.99"
            )
    
    except Exception as e:
        results["passed"] = False
        results["issues"].append(f"Output comparison error: {e}")
    
    return results


def verify_gradient_flow(
    encoder, 
    batch: Dict[str, torch.Tensor], 
    structure_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Verify gradient flow through encoder in frozen and unfrozen modes.
    
    Args:
        encoder: Encoder instance to test
        batch: Input batch for forward pass
        structure_name: Structure name for reporting
        
    Returns:
        Gradient flow validation results
    """
    results = {
        "frozen_mode": {"passed": True, "issues": []},
        "unfrozen_mode": {"passed": True, "issues": []}
    }
    
    try:
        # Test frozen mode
        encoder.eval()
        if hasattr(encoder, '_freeze_parameters'):
            encoder._freeze_parameters()
        
        features = encoder(batch)
        loss = features.sum()
        loss.backward()
        
        # Check frozen gradients
        gradient_found = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 1e-8:
                gradient_found = True
                results["frozen_mode"]["issues"].append(
                    f"Gradient found in frozen parameter {name}: {param.grad.abs().sum().item():.2e}"
                )
        
        if gradient_found:
            results["frozen_mode"]["passed"] = False
        
        # Clear gradients
        encoder.zero_grad()
        
        # Test unfrozen mode
        if hasattr(encoder, 'unfreeze_layers'):
            encoder.unfreeze_layers()
        
        features = encoder(batch)
        loss = features.sum()
        loss.backward()
        
        # Check unfrozen gradients
        gradient_norms = []
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    gradient_norms.append(grad_norm)
        
        if len(gradient_norms) == 0:
            results["unfrozen_mode"]["passed"] = False
            results["unfrozen_mode"]["issues"].append("No gradients found in unfrozen mode")
        elif max(gradient_norms) < 1e-6:
            results["unfrozen_mode"]["passed"] = False  
            results["unfrozen_mode"]["issues"].append(
                f"Gradient norms too small: max={max(gradient_norms):.2e}"
            )
        
        results["unfrozen_mode"]["gradient_stats"] = {
            "num_params_with_grads": len(gradient_norms),
            "max_gradient_norm": max(gradient_norms) if gradient_norms else 0.0,
            "mean_gradient_norm": np.mean(gradient_norms) if gradient_norms else 0.0
        }
    
    except Exception as e:
        results["frozen_mode"]["passed"] = False
        results["unfrozen_mode"]["passed"] = False
        results["error"] = str(e)
    
    return results


def save_validation_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save validation results to JSON file with formatted output.
    
    Args:
        results: Validation results dictionary
        output_path: Path to save results JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert torch tensors to lists for JSON serialization
    def serialize_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist() 
        elif isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_for_json(v) for v in obj]
        else:
            return obj
    
    serialized_results = serialize_for_json(results)
    
    with open(output_path, 'w') as f:
        json.dump(serialized_results, f, indent=2)
    
    logger.info(f"Validation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage and self-test
    print("Validation helpers loaded successfully")
    
    # Test deterministic setup
    setup_deterministic_environment(42)
    print("✓ Deterministic environment setup")
    
    # Test batch creation
    test_batch = create_test_batch(seq_length=10, add_noise=True)
    print(f"✓ Created test batch: {test_batch['X'].shape}")
    
    # Test numerical stability check
    stability_results = check_numerical_stability(
        torch.randn(1, 10, 128), 
        "test_structure"
    )
    print(f"✓ Numerical stability check: {stability_results['passed']}")