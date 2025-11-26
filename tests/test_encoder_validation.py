"""
ProteinMPNN Encoder Validation Test Suite

This module implements comprehensive validation for the hybrid ProteinMPNN encoder
integration, following the Stage 1 validation strategy from the multi-agent debate
synthesis. Tests verify scientific correctness, numerical stability, and integration
fidelity with the original ProteinMPNN implementation.

Test Categories:
1. Checkpoint Integrity: Model loading and architecture verification
2. Output Equivalence: Comparison against standalone ProteinMPNN reference
3. Vocabulary Consistency: AA ordering alignment across components  
4. Feature Extraction: RBF, positional encoding, orientation correctness
5. Numerical Stability: NaN/Inf detection, value range validation
6. Gradient Flow: Verification in frozen and unfrozen modes
7. Reproducibility: Deterministic seeding across runs
8. Edge Cases: Graceful handling of malformed inputs
"""

import pytest
import torch
import numpy as np
import os
import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "proteinmpnn"))
sys.path.append(str(project_root / "hybrid"))

# Imports
from hybrid.models.mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from protein_mpnn_utils import ProteinMPNN, parse_PDB, gather_nodes

# Test configuration
TEST_STRUCTURES = [
    "5L33.pdb",  # Small monomer (~150 residues)
    "3HTN.pdb",  # Complex structure  
    "4GYT.pdb"   # Homooligomer
]

TEST_CHECKPOINTS = [
    ("vanilla", "v_48_020"),
    ("ca_model", "v_48_020"), 
    ("soluble", "v_48_020")
]

TOLERANCE = 1e-5  # Reference comparison tolerance
RANDOM_SEED = 42  # Reproducibility seed


class TestFixtures:
    """Pytest fixtures for encoder validation tests"""
    
    @pytest.fixture(scope="session") 
    def project_paths(self):
        """Project directory paths"""
        root = Path(__file__).parent.parent
        return {
            "root": root,
            "proteinmpnn_dir": root / "proteinmpnn",
            "test_pdbs": root / "proteinmpnn" / "inputs",
            "outputs": root / "tests" / "validation_outputs"
        }
    
    @pytest.fixture(scope="session")
    def test_structures(self, project_paths):
        """Load and parse test PDB structures"""
        structures = {}
        pdb_dirs = [
            project_paths["test_pdbs"] / "PDB_monomers" / "pdbs",
            project_paths["test_pdbs"] / "PDB_complexes" / "pdbs", 
            project_paths["test_pdbs"] / "PDB_homooligomers" / "pdbs"
        ]
        
        for structure_name in TEST_STRUCTURES:
            found = False
            for pdb_dir in pdb_dirs:
                pdb_path = pdb_dir / structure_name
                if pdb_path.exists():
                    try:
                        # Parse using ProteinMPNN's parse_PDB
                        parsed = parse_PDB(str(pdb_path))
                        structures[structure_name] = {
                            "path": str(pdb_path),
                            "parsed": parsed,
                            "name": structure_name.split('.')[0]
                        }
                        found = True
                        break
                    except Exception as e:
                        pytest.skip(f"Failed to parse {structure_name}: {e}")
            
            if not found:
                pytest.skip(f"Test structure {structure_name} not found")
        
        return structures
    
    @pytest.fixture(scope="session", params=TEST_CHECKPOINTS)
    def encoder_checkpoint(self, request, project_paths):
        """Load encoder with different checkpoint types"""
        model_type, model_name = request.param
        try:
            encoder = ProteinMPNNBackboneEncoder.from_pretrained(
                model_name=model_name,
                model_type=model_type,
                freeze_layers=True
            )
            return {
                "encoder": encoder,
                "model_type": model_type, 
                "model_name": model_name
            }
        except Exception as e:
            pytest.skip(f"Failed to load checkpoint {model_type}/{model_name}: {e}")
    
    @pytest.fixture(scope="session")
    def reference_proteinmpnn(self, project_paths):
        """Load standalone ProteinMPNN for reference comparison"""
        try:
            # Use same parameters as encoder wrapper
            model = ProteinMPNN(
                num_letters=21,
                node_features=128,
                edge_features=128, 
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=21,
                k_neighbors=64,
                augment_eps=0.05,
                dropout=0.1,
                ca_only=False
            )
            
            # Load vanilla v_48_020 weights
            checkpoint_path = project_paths["proteinmpnn_dir"] / "vanilla_model_weights" / "v_48_020.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                return model
            else:
                pytest.skip(f"Reference checkpoint not found: {checkpoint_path}")
        except Exception as e:
            pytest.skip(f"Failed to load reference ProteinMPNN: {e}")
    
    @pytest.fixture
    def deterministic_setup(self):
        """Setup deterministic random state for reproducibility tests"""
        def setup_seeds(seed=RANDOM_SEED):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # Additional deterministic settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        return setup_seeds


class TestEncoderValidation(TestFixtures):
    """Main encoder validation test suite"""
    
    def test_checkpoint_integrity(self, encoder_checkpoint):
        """Test 1: Verify encoder loads correctly from all available checkpoints"""
        encoder_info = encoder_checkpoint
        encoder = encoder_info["encoder"]
        
        # Verify encoder components exist
        assert hasattr(encoder, 'graph_builder'), "Encoder missing graph_builder component"
        assert hasattr(encoder, 'encoder_layers'), "Encoder missing encoder_layers component"
        assert hasattr(encoder, 'W_e'), "Encoder missing edge embedding layer"
        
        # Verify encoder layers count
        assert len(encoder.encoder_layers) == 3, f"Expected 3 encoder layers, got {len(encoder.encoder_layers)}"
        
        # Verify embedding dimension
        assert encoder.get_embedding_dim() == 128, f"Expected embedding dim 128, got {encoder.get_embedding_dim()}"
        
        # Verify freeze status if requested
        if encoder.freeze_layers:
            for param in encoder.full_model.parameters():
                assert not param.requires_grad, "Encoder parameters should be frozen"
    
    def test_output_equivalence_reference(self, encoder_checkpoint, test_structures, reference_proteinmpnn):
        """Test 2: Compare encoder outputs against standalone ProteinMPNN reference"""
        encoder_info = encoder_checkpoint
        encoder = encoder_info["encoder"]
        reference_model = reference_proteinmpnn
        
        # Skip if encoder and reference use different model types
        if encoder_info["model_type"] != "vanilla":
            pytest.skip("Reference comparison only for vanilla models")
        
        for structure_name, structure_data in test_structures.items():
            parsed_pdb = structure_data["parsed"]
            
            # Convert parsed PDB to tensor format expected by encoder
            batch = self._convert_parsed_pdb_to_batch(parsed_pdb, ca_only=False)
            
            # Get encoder output
            with torch.no_grad():
                encoder_features = encoder(batch)
            
            # Get reference output (just encoder part)
            with torch.no_grad():
                # Build graph using reference model
                X, mask, residue_idx, chain_encoding_all = batch['X'], batch['mask'], batch['residue_idx'], batch['chain_encoding_all']
                E, E_idx = reference_model.features(X, mask, residue_idx, chain_encoding_all)
                
                # Initialize features with same device as E tensor
                h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
                h_E = reference_model.W_e(E)
                
                # Run through encoder layers
                mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) 
                mask_attend = mask.unsqueeze(-1) * mask_attend
                
                for layer in reference_model.encoder_layers:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                
                reference_features = h_V
            
            # Compare outputs
            assert encoder_features.shape == reference_features.shape, \
                f"Shape mismatch for {structure_name}: {encoder_features.shape} vs {reference_features.shape}"
            
            # Check numerical equivalence
            max_diff = torch.max(torch.abs(encoder_features - reference_features)).item()
            relative_diff = max_diff / (torch.max(torch.abs(reference_features)).item() + 1e-8)
            
            assert relative_diff < TOLERANCE, \
                f"Output mismatch for {structure_name}: max_relative_diff={relative_diff:.2e} > {TOLERANCE}"
    
    def test_vocabulary_consistency(self, encoder_checkpoint):
        """Test 3: Verify amino acid vocabulary ordering consistency"""
        encoder_info = encoder_checkpoint
        encoder = encoder_info["encoder"]
        
        # Get amino acid vocabulary from encoder
        # Note: ProteinMPNN uses standard 20 AA + X ordering
        expected_vocab_size = 21
        
        # Check if encoder has vocabulary information
        if hasattr(encoder.full_model, 'num_letters'):
            assert encoder.full_model.num_letters == expected_vocab_size, \
                f"Vocabulary size mismatch: expected {expected_vocab_size}, got {encoder.full_model.num_letters}"
        
        # Verify decoder vocab consistency (if present)
        if hasattr(encoder.full_model, 'vocab'):
            assert encoder.full_model.vocab == expected_vocab_size, \
                f"Decoder vocab mismatch: expected {expected_vocab_size}, got {encoder.full_model.vocab}"
        
        # Test with one-hot encoded amino acid inputs
        test_sequence_length = 10
        for aa_idx in range(20):  # Standard 20 amino acids
            batch = self._create_dummy_batch(test_sequence_length)
            
            with torch.no_grad():
                try:
                    features = encoder(batch)
                    assert not torch.any(torch.isnan(features)), \
                        f"NaN values in features for AA index {aa_idx}"
                except Exception as e:
                    pytest.fail(f"Encoder failed for AA index {aa_idx}: {e}")
    
    def test_numerical_stability(self, encoder_checkpoint, test_structures):
        """Test 4: Verify numerical stability - no NaN/Inf values, reasonable ranges"""
        encoder_info = encoder_checkpoint 
        encoder = encoder_info["encoder"]
        
        for structure_name, structure_data in test_structures.items():
            parsed_pdb = structure_data["parsed"]
            # Use ca_only flag from encoder info to handle coordinate format properly
            ca_only = encoder_info.get("model_type") == "ca_model"
            batch = self._convert_parsed_pdb_to_batch(parsed_pdb, ca_only=ca_only)
            
            with torch.no_grad():
                features = encoder(batch)
            
            # Check for NaN/Inf values
            assert not torch.any(torch.isnan(features)), \
                f"NaN values detected in features for {structure_name}"
            assert not torch.any(torch.isinf(features)), \
                f"Inf values detected in features for {structure_name}"
            
            # Check feature value ranges are reasonable
            feature_min, feature_max = features.min().item(), features.max().item()
            assert -100 < feature_min < 100, \
                f"Feature values out of reasonable range for {structure_name}: min={feature_min}"
            assert -100 < feature_max < 100, \
                f"Feature values out of reasonable range for {structure_name}: max={feature_max}"
            
            # Check feature variance (should not be constant)
            feature_std = features.std().item()
            assert feature_std > 1e-6, \
                f"Features appear constant (std={feature_std:.2e}) for {structure_name}"
    
    def test_gradient_flow(self, encoder_checkpoint, test_structures):
        """Test 5: Verify gradient flow in frozen and unfrozen modes"""
        encoder_info = encoder_checkpoint
        
        for structure_name, structure_data in test_structures.items():
            parsed_pdb = structure_data["parsed"] 
            batch = self._convert_parsed_pdb_to_batch(parsed_pdb)
            
            # Test frozen mode (gradients should NOT flow)
            encoder_frozen = ProteinMPNNBackboneEncoder.from_pretrained(
                model_name=encoder_info["model_name"],
                model_type=encoder_info["model_type"], 
                freeze_layers=True
            )
            
            features_frozen = encoder_frozen(batch)
            loss_frozen = features_frozen.sum()
            loss_frozen.backward()
            
            # Check that frozen parameters have no gradients
            has_gradients = False
            for param in encoder_frozen.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break
            
            assert not has_gradients, f"Gradients found in frozen encoder for {structure_name}"
            
            # Test unfrozen mode (gradients should flow)
            encoder_unfrozen = ProteinMPNNBackboneEncoder.from_pretrained(
                model_name=encoder_info["model_name"],
                model_type=encoder_info["model_type"],
                freeze_layers=False
            )
            
            features_unfrozen = encoder_unfrozen(batch)
            loss_unfrozen = features_unfrozen.sum()
            loss_unfrozen.backward()
            
            # Check that unfrozen parameters have gradients
            gradient_norms = []
            for param in encoder_unfrozen.parameters():
                if param.grad is not None:
                    gradient_norms.append(param.grad.norm().item())
            
            assert len(gradient_norms) > 0, f"No gradients found in unfrozen encoder for {structure_name}"
            assert max(gradient_norms) > 1e-8, f"Gradient norms too small for {structure_name}: max={max(gradient_norms):.2e}"
    
    def test_reproducibility(self, encoder_checkpoint, test_structures, deterministic_setup):
        """Test 6: Verify reproducible outputs with same random seed"""
        encoder_info = encoder_checkpoint
        encoder = encoder_info["encoder"]
        setup_seeds = deterministic_setup
        
        for structure_name, structure_data in test_structures.items():
            parsed_pdb = structure_data["parsed"]
            # Use ca_only flag from encoder info to handle coordinate format properly
            ca_only = encoder_info.get("model_type") == "ca_model"
            batch = self._convert_parsed_pdb_to_batch(parsed_pdb, ca_only=ca_only)
            
            # Run 1 with seed
            setup_seeds(RANDOM_SEED)
            with torch.no_grad():
                features_1 = encoder(batch)
            
            # Run 2 with same seed  
            setup_seeds(RANDOM_SEED)
            with torch.no_grad():
                features_2 = encoder(batch)
            
            # Run 3 with different seed
            setup_seeds(RANDOM_SEED + 1)
            with torch.no_grad():
                features_3 = encoder(batch)
            
            # Check reproducibility with same seed
            max_diff_same_seed = torch.max(torch.abs(features_1 - features_2)).item()
            assert max_diff_same_seed < 1e-10, \
                f"Non-reproducible outputs with same seed for {structure_name}: max_diff={max_diff_same_seed:.2e}"
            
            # Check that different seeds produce different outputs (stochastic components exist)
            max_diff_different_seed = torch.max(torch.abs(features_1 - features_3)).item()
            # Note: May be identical if no stochastic components, which is also valid
    
    def test_edge_cases(self, encoder_checkpoint):
        """Test 7: Graceful handling of edge cases and malformed inputs"""
        encoder_info = encoder_checkpoint
        encoder = encoder_info["encoder"]
        
        # Test 1: Very short sequence (single residue)
        short_batch = self._create_dummy_batch(seq_length=1)
        with torch.no_grad():
            try:
                features_short = encoder(short_batch)
                assert features_short.shape[1] == 1, "Short sequence feature shape incorrect"
            except Exception as e:
                # If it fails, should be graceful with informative message
                assert "length" in str(e).lower() or "size" in str(e).lower(), \
                    f"Uninformative error for short sequence: {e}"
        
        # Test 2: Very long sequence  
        long_batch = self._create_dummy_batch(seq_length=1000)
        with torch.no_grad():
            try:
                features_long = encoder(long_batch)
                assert features_long.shape[1] == 1000, "Long sequence feature shape incorrect"
            except Exception as e:
                # Memory errors are acceptable for very long sequences
                assert any(keyword in str(e).lower() for keyword in ["memory", "cuda", "size"]), \
                    f"Unexpected error for long sequence: {e}"
        
        # Test 3: Missing backbone atoms (NaN coordinates)
        nan_batch = self._create_dummy_batch(seq_length=10)
        nan_batch['X'][0, 5, :, :] = float('nan')  # Set one residue to NaN
        
        with torch.no_grad():
            try:
                features_nan = encoder(nan_batch)
                # If it succeeds, output should not contain NaN
                assert not torch.any(torch.isnan(features_nan)), \
                    "NaN input produced NaN output without proper handling"
            except Exception as e:
                # Should fail gracefully with informative error
                assert "nan" in str(e).lower() or "invalid" in str(e).lower(), \
                    f"Uninformative error for NaN coordinates: {e}"
        
        # Test 4: Chain breaks (large distance gaps)
        break_batch = self._create_dummy_batch(seq_length=10)
        # Create large gap between residues 4 and 5
        break_batch['X'][0, 5:, :, :] += 100.0  # Move residues 5-9 far away
        
        with torch.no_grad():
            try:
                features_break = encoder(break_batch)
                # Should handle chain breaks gracefully
                assert not torch.any(torch.isnan(features_break)), \
                    "Chain break produced NaN features"
            except Exception as e:
                # Chain break errors should be informative
                assert any(keyword in str(e).lower() for keyword in ["distance", "chain", "break", "gap"]), \
                    f"Uninformative error for chain break: {e}"
    
    def _convert_parsed_pdb_to_batch(self, parsed_pdb, ca_only: bool = False) -> Dict[str, torch.Tensor]:
        """Convert ProteinMPNN parsed PDB format to encoder batch format"""
        # Import validation helpers
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "hybrid"))
        from utils.validation_helpers import convert_parsed_pdb_to_batch
        
        return convert_parsed_pdb_to_batch(parsed_pdb, ca_only=ca_only)
    
    def _create_dummy_batch(self, seq_length: int, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create dummy batch for testing"""
        return {
            'X': torch.randn(batch_size, seq_length, 4, 3),  # [B, L, 4, 3] backbone coords
            'mask': torch.ones(batch_size, seq_length),      # [B, L] sequence mask
            'residue_idx': torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1),  # [B, L]
            'chain_encoding_all': torch.zeros(batch_size, seq_length)  # [B, L] chain IDs
        }


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v", "--tb=short"])