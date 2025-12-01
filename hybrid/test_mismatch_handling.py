"""
Unit Tests for Coordinate/Sequence Length Mismatch Handling

Tests the coordinate and sequence length mismatch handling logic in StabilityDataset,
including padding, truncation, threshold-based rejection, and edge cases.
"""

import torch
import numpy as np
import warnings
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Import the dataset class
import sys
sys.path.append(os.path.dirname(__file__))
from data.stability_dataset import StabilityDataset


class TestCoordSeqMismatchHandling(unittest.TestCase):
    """Test coordinate/sequence length mismatch handling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp(prefix="mismatch_test_")
        
        # Create a minimal PDB file (needed for dataset initialization)
        dummy_pdb = """HEADER    TEST PROTEIN                            01-JAN-00   TEST
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      20.154  18.000  11.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      21.000  18.000  12.000  1.00 20.00           C  
ATOM      4  O   ALA A   1      22.000  17.500  12.000  1.00 20.00           O  
END"""
        pdb_path = Path(self.test_dir) / "test.pdb"
        with open(pdb_path, 'w') as f:
            f.write(dummy_pdb)
        
        # Mock backbone encoder to avoid dependencies
        self.mock_encoder = Mock()
        self.mock_encoder.return_value = torch.randn(1, 10, 128)  # [batch, seq_len, hidden_dim]
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_dataset(self, max_coord_seq_mismatch_ratio=0.5, extract_backbone_features=True):
        """Helper to create dataset with mocked backbone encoder"""
        dataset = StabilityDataset(
            data_dir=self.test_dir,
            lazy_loading=True,
            extract_backbone_features=extract_backbone_features,
            max_coord_seq_mismatch_ratio=max_coord_seq_mismatch_ratio
        )
        
        # Mock the backbone encoder if feature extraction is enabled
        if extract_backbone_features:
            dataset.backbone_encoder = self.mock_encoder
        
        return dataset
    
    def _create_sample(self, sequence_length, coord_length, structure_file="test.pdb", chain_id="A"):
        """Helper to create sample with specified sequence and coordinate lengths"""
        # Create sequence
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequence = [amino_acids[i % 20] for i in range(sequence_length)]
        
        # Create coordinates [L, 4, 3] for backbone atoms N, CA, C, O
        coords = torch.randn(coord_length, 4, 3)
        
        return {
            'sequence': sequence,
            'coordinates': coords,
            'structure_file': structure_file,
            'chain_id': chain_id,
            'label': 1
        }
    
    def test_configurable_mismatch_threshold(self):
        """Test that the configurable mismatch threshold parameter works correctly"""
        # Test valid values
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=0.3)
        self.assertEqual(dataset.max_coord_seq_mismatch_ratio, 0.3)
        
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=0.8)
        self.assertEqual(dataset.max_coord_seq_mismatch_ratio, 0.8)
        
        # Test boundary values
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=0.0)
        self.assertEqual(dataset.max_coord_seq_mismatch_ratio, 0.0)
        
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=1.0)
        self.assertEqual(dataset.max_coord_seq_mismatch_ratio, 1.0)
        
        # Test invalid values raise ValueError
        with self.assertRaises(ValueError):
            self._create_dataset(max_coord_seq_mismatch_ratio=-0.1)
        
        with self.assertRaises(ValueError):
            self._create_dataset(max_coord_seq_mismatch_ratio=1.1)
    
    def test_perfect_match_no_warnings(self):
        """Test that perfect coordinate/sequence length match produces no warnings"""
        dataset = self._create_dataset()
        sample = self._create_sample(sequence_length=10, coord_length=10)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset._extract_backbone_features(sample)
            
            # Should not produce mismatch warnings
            mismatch_warnings = [warning for warning in w 
                               if "mismatch" in str(warning.message)]
            self.assertEqual(len(mismatch_warnings), 0)
            
            # Should return valid result
            self.assertIsNotNone(result)
    
    def test_truncation_coord_longer_than_sequence(self):
        """Test truncation when coordinate length > sequence length"""
        dataset = self._create_dataset()
        sample = self._create_sample(sequence_length=8, coord_length=12)  # 33% mismatch
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset._extract_backbone_features(sample)
            
            # Should produce mismatch warnings with sample identification
            mismatch_warnings = [warning for warning in w 
                               if "mismatch" in str(warning.message)]
            self.assertGreater(len(mismatch_warnings), 0)
            
            # Check that sample ID is included in warning
            has_sample_id = any("[test.pdb:A]" in str(warning.message) 
                              for warning in mismatch_warnings)
            self.assertTrue(has_sample_id)
            
            # Should have truncation warning
            truncation_warnings = [warning for warning in w 
                                 if "Truncated" in str(warning.message)]
            self.assertGreater(len(truncation_warnings), 0)
            
            # Should return valid result with expected shape
            self.assertIsNotNone(result)
            # Verify the mock encoder was called with expected sequence length
            self.mock_encoder.assert_called_once()
    
    def test_padding_sequence_longer_than_coord(self):
        """Test padding when sequence length > coordinate length"""
        dataset = self._create_dataset()
        sample = self._create_sample(sequence_length=12, coord_length=8)  # 33% mismatch
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset._extract_backbone_features(sample)
            
            # Should produce mismatch warnings
            mismatch_warnings = [warning for warning in w 
                               if "mismatch" in str(warning.message)]
            self.assertGreater(len(mismatch_warnings), 0)
            
            # Should have padding warning
            padding_warnings = [warning for warning in w 
                               if "Padded" in str(warning.message)]
            self.assertGreater(len(padding_warnings), 0)
            
            # Should return valid result
            self.assertIsNotNone(result)
    
    def test_large_mismatch_rejection(self):
        """Test that large mismatches are rejected"""
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=0.5)
        
        # Create sample with 60% mismatch (should be rejected)
        sample = self._create_sample(sequence_length=10, coord_length=25)  # 60% mismatch
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dataset._extract_backbone_features(sample)
            
            # Should be rejected (return None)
            self.assertIsNone(result)
            
            # Should produce rejection warning
            rejection_warnings = [warning for warning in w 
                                if "Rejecting sample" in str(warning.message)]
            self.assertGreater(len(rejection_warnings), 0)
    
    def test_configurable_threshold_rejection(self):
        """Test that configurable threshold affects rejection"""
        # Create sample with 40% mismatch
        sample = self._create_sample(sequence_length=10, coord_length=17)  # 41% mismatch
        
        # With default threshold (50%), should be accepted
        dataset_permissive = self._create_dataset(max_coord_seq_mismatch_ratio=0.5)
        result_accepted = dataset_permissive._extract_backbone_features(sample)
        self.assertIsNotNone(result_accepted)
        
        # With strict threshold (30%), should be rejected
        dataset_strict = self._create_dataset(max_coord_seq_mismatch_ratio=0.3)
        result_rejected = dataset_strict._extract_backbone_features(sample)
        self.assertIsNone(result_rejected)
    
    def test_no_coordinates_edge_case(self):
        """Test edge case with no coordinates available"""
        dataset = self._create_dataset()
        sample = self._create_sample(sequence_length=10, coord_length=0)
        
        result = dataset._extract_backbone_features(sample)
        self.assertIsNone(result)
    
    def test_empty_sequence_edge_case(self):
        """Test edge case with empty sequence"""
        dataset = self._create_dataset()
        sample = self._create_sample(sequence_length=0, coord_length=5)
        
        result = dataset._extract_backbone_features(sample)
        self.assertIsNone(result)
    
    def test_zero_mismatch_threshold(self):
        """Test that zero threshold rejects any mismatch"""
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=0.0)
        
        # Even 1 residue mismatch should be rejected
        sample = self._create_sample(sequence_length=10, coord_length=11)
        result = dataset._extract_backbone_features(sample)
        self.assertIsNone(result)
    
    def test_maximum_mismatch_threshold(self):
        """Test that maximum threshold (1.0) accepts any mismatch"""
        dataset = self._create_dataset(max_coord_seq_mismatch_ratio=1.0)
        
        # Very large mismatch should still be accepted  
        sample = self._create_sample(sequence_length=10, coord_length=50)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = dataset._extract_backbone_features(sample)
            
            # Should not be rejected due to threshold
            self.assertIsNotNone(result)
    
    def test_sample_identification_in_warnings(self):
        """Test that warnings include proper sample identification"""
        dataset = self._create_dataset()
        sample = self._create_sample(
            sequence_length=8, 
            coord_length=12,
            structure_file="/path/to/protein_1A2B.pdb",
            chain_id="B"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset._extract_backbone_features(sample)
            
            # Check that warnings contain the expected sample ID format
            mismatch_warnings = [warning for warning in w 
                               if "mismatch" in str(warning.message)]
            
            has_proper_id = any("[/path/to/protein_1A2B.pdb:B]" in str(warning.message) 
                              for warning in mismatch_warnings)
            self.assertTrue(has_proper_id)
    
    def test_missing_chain_id_sample_identification(self):
        """Test sample identification when chain_id is missing"""
        dataset = self._create_dataset()
        sample = self._create_sample(
            sequence_length=8,
            coord_length=12,
            structure_file="test_protein.pdb"
        )
        # Remove chain_id to test fallback
        del sample['chain_id']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset._extract_backbone_features(sample)
            
            # Should use structure file only when chain_id is missing
            mismatch_warnings = [warning for warning in w 
                               if "mismatch" in str(warning.message)]
            
            has_file_only_id = any("[test_protein.pdb]" in str(warning.message) 
                                 for warning in mismatch_warnings)
            self.assertTrue(has_file_only_id)


def run_mismatch_tests():
    """Run all coordinate/sequence mismatch handling tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_mismatch_tests()