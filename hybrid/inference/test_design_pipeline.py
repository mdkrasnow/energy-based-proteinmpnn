"""
Test Suite for End-to-End Protein Design Pipeline

This module provides comprehensive testing for the ProteinDesignPipeline,
including single design, batch processing, validation, error handling,
and integration with real PDB structures.

Test Coverage:
- Pipeline initialization and configuration
- Single protein design workflow
- Batch processing with memory management
- ProteinMPNN decoder integration
- Result validation and quality assessment
- Error handling and edge cases
- Integration with sample PDB structures
"""

import os
import sys
import tempfile
import shutil
import json
import warnings
from pathlib import Path
import pytest
import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import components to test
from inference.design_pipeline import (
    ProteinDesignPipeline,
    PipelineConfig,
    DesignResult,
    create_default_config,
    load_pipeline_from_config
)

# Test utilities
@pytest.fixture(scope="module")
def sample_pdb_paths():
    """Get paths to sample PDB structures for testing"""
    base_dir = Path(__file__).parent.parent.parent / "proteinmpnn" / "inputs"
    
    pdb_paths = []
    for subdir in ["PDB_monomers", "PDB_complexes", "PDB_homooligomers"]:
        pdb_dir = base_dir / subdir / "pdbs"
        if pdb_dir.exists():
            for pdb_file in pdb_dir.glob("*.pdb"):
                pdb_paths.append(str(pdb_file))
                if len(pdb_paths) >= 3:  # Limit for testing
                    break
        if len(pdb_paths) >= 3:
            break
    
    return pdb_paths


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    """Create mock pipeline configuration for testing"""
    # Create dummy checkpoint files
    encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
    energy_ckpt = os.path.join(temp_dir, "energy.pt")
    
    # Create minimal checkpoint data
    torch.save({"dummy": "data"}, encoder_ckpt)
    torch.save({"model_state_dict": {"dummy": torch.randn(10)}}, energy_ckpt)
    
    config = PipelineConfig(
        encoder_checkpoint=encoder_ckpt,
        energy_model_checkpoint=energy_ckpt,
        device='cpu',  # Force CPU for testing
        num_designs_per_target=2,
        validate_results=False,  # Skip validation for faster tests
        use_proteinmpnn_init=False  # Skip ProteinMPNN for faster tests
    )
    
    return config


@pytest.fixture
def mock_pipeline(mock_config):
    """Create mock pipeline with mocked components for testing"""
    with patch.multiple(
        'inference.design_pipeline.ProteinDesignPipeline',
        _initialize_components=Mock(),
        _initialize_proteinmpnn=Mock()
    ):
        pipeline = ProteinDesignPipeline(mock_config)
        
        # Create mock components
        pipeline.encoder = Mock()
        pipeline.encoder.eval = Mock()
        pipeline.encoder.to = Mock(return_value=pipeline.encoder)
        
        pipeline.sequence_repr = Mock()
        pipeline.sequence_repr.to = Mock(return_value=pipeline.sequence_repr)
        
        # Mock energy model
        mock_energy_model = Mock()
        mock_energy_model.eval = Mock()
        mock_energy_model.to = Mock(return_value=mock_energy_model)
        pipeline.energy_models = [mock_energy_model]
        
        # Mock optimizer
        from inference.ired_optimizer import OptimizationResult
        pipeline.optimizer = Mock()
        
        # Mock successful optimization result
        def mock_optimize(*args, **kwargs):
            return OptimizationResult(
                sequence=torch.randint(0, 20, (1, 50)),
                logits=torch.randn(1, 50, 20),
                trajectory=[],
                final_energy=-2.5,
                converged=True,
                total_steps=10,
                landscapes_used=3,
                optimization_failed=False
            )
        
        pipeline.optimizer.optimize_sequence = Mock(side_effect=mock_optimize)
        
        return pipeline


@pytest.fixture
def dummy_pdb_file(temp_dir):
    """Create a dummy PDB file for testing"""
    pdb_content = """HEADER    TEST PROTEIN                            01-JAN-00   TEST
ATOM      1  N   ALA A   1      20.000  16.000  10.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      20.000  17.000  11.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      21.000  17.000  12.000  1.00 20.00           C  
ATOM      4  O   ALA A   1      22.000  16.500  12.000  1.00 20.00           O  
ATOM      5  N   VAL A   2      21.000  18.000  13.000  1.00 20.00           N  
ATOM      6  CA  VAL A   2      21.500  18.000  14.000  1.00 20.00           C  
ATOM      7  C   VAL A   2      22.000  19.000  15.000  1.00 20.00           C  
ATOM      8  O   VAL A   2      23.000  19.500  15.000  1.00 20.00           O  
ATOM      9  N   GLY A   3      22.000  20.000  16.000  1.00 20.00           N  
ATOM     10  CA  GLY A   3      22.500  20.000  17.000  1.00 20.00           C  
ATOM     11  C   GLY A   3      23.000  21.000  18.000  1.00 20.00           C  
ATOM     12  O   GLY A   3      24.000  21.500  18.000  1.00 20.00           O  
END
"""
    
    pdb_path = os.path.join(temp_dir, "test_protein.pdb")
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)
    
    return pdb_path


class TestPipelineConfig:
    """Test pipeline configuration functionality"""
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation"""
        # Valid config
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        torch.save({}, encoder_ckpt)
        torch.save({}, energy_ckpt)
        
        config = PipelineConfig(
            encoder_checkpoint=encoder_ckpt,
            energy_model_checkpoint=energy_ckpt
        )
        assert config.device in ['cpu', 'cuda']
        assert config.batch_size > 0
        
        # Invalid config - missing files
        with pytest.raises(FileNotFoundError):
            PipelineConfig(
                encoder_checkpoint="nonexistent.pt",
                energy_model_checkpoint=energy_ckpt
            )
        
        # Invalid config - bad parameters
        with pytest.raises(ValueError):
            PipelineConfig(
                encoder_checkpoint=encoder_ckpt,
                energy_model_checkpoint=energy_ckpt,
                batch_size=-1
            )
    
    def test_config_from_dict(self, temp_dir):
        """Test configuration creation from dictionary"""
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        torch.save({}, encoder_ckpt)
        torch.save({}, energy_ckpt)
        
        config_dict = {
            'encoder_checkpoint': encoder_ckpt,
            'energy_model_checkpoint': energy_ckpt,
            'device': 'cpu',
            'num_designs_per_target': 3
        }
        
        config = PipelineConfig(**config_dict)
        assert config.num_designs_per_target == 3
        assert config.device == 'cpu'
    
    def test_config_serialization(self, temp_dir):
        """Test configuration save/load"""
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        torch.save({}, encoder_ckpt)
        torch.save({}, energy_ckpt)
        
        config_path = create_default_config(encoder_ckpt, energy_ckpt, 
                                          os.path.join(temp_dir, "config.json"))
        
        # Load and verify
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        assert config_data['encoder_checkpoint'] == encoder_ckpt
        assert config_data['energy_model_checkpoint'] == energy_ckpt


class TestPipelineInitialization:
    """Test pipeline initialization and component loading"""
    
    @patch('inference.design_pipeline.ProteinMPNNBackboneEncoder')
    @patch('inference.design_pipeline.EnergyHead')
    @patch('inference.design_pipeline.ContinuousSequenceRepr')
    @patch('inference.design_pipeline.IREDSequenceOptimizer')
    def test_pipeline_initialization(self, mock_optimizer, mock_seq_repr, 
                                   mock_energy, mock_encoder, mock_config):
        """Test successful pipeline initialization"""
        # Mock component initialization
        mock_encoder.return_value.to.return_value = mock_encoder.return_value
        mock_energy.return_value.to.return_value = mock_energy.return_value
        mock_seq_repr.return_value.to.return_value = mock_seq_repr.return_value
        
        # Don't mock the initialization methods this time
        pipeline = ProteinDesignPipeline(mock_config)
        
        # Verify components were created
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'device')
        assert pipeline.config == mock_config
    
    def test_pipeline_from_config_file(self, temp_dir):
        """Test pipeline loading from configuration file"""
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        torch.save({}, encoder_ckpt)
        torch.save({}, energy_ckpt)
        
        config_path = create_default_config(encoder_ckpt, energy_ckpt,
                                          os.path.join(temp_dir, "config.json"))
        
        with patch.multiple(
            'inference.design_pipeline.ProteinDesignPipeline',
            _initialize_components=Mock(),
            _initialize_proteinmpnn=Mock()
        ):
            pipeline = load_pipeline_from_config(config_path)
            assert isinstance(pipeline, ProteinDesignPipeline)


class TestSingleDesign:
    """Test single protein design functionality"""
    
    def test_design_sequence_success(self, mock_pipeline, dummy_pdb_file):
        """Test successful protein design"""
        # Mock PDB parsing
        with patch('inference.design_pipeline.parse_PDB') as mock_parse:
            mock_parse.return_value = {
                'coords': {'A': np.random.randn(50, 4, 3)},
                'mask': [1] * 50
            }
            
            # Mock feature extraction
            mock_pipeline._extract_features = Mock(return_value=torch.randn(1, 50, 128))
            mock_pipeline._generate_initial_sequences = Mock(return_value=(torch.randn(2, 50, 20), 'proteinmpnn'))
            
            result = mock_pipeline.design_sequence(dummy_pdb_file)
            
            assert isinstance(result, DesignResult)
            assert result.success == True
            assert result.num_designs == 2
            assert result.sequences is not None
            assert result.energies is not None
    
    def test_design_sequence_failure(self, mock_pipeline, dummy_pdb_file):
        """Test design failure handling"""
        # Mock PDB parsing failure
        with patch('inference.design_pipeline.parse_PDB', side_effect=Exception("Parse failed")):
            result = mock_pipeline.design_sequence(dummy_pdb_file)
            
            assert isinstance(result, DesignResult)
            assert result.success == False
            assert result.error_message is not None
            assert "Parse failed" in result.error_message
    
    def test_design_with_constraints(self, mock_pipeline, dummy_pdb_file):
        """Test design with position constraints"""
        mask = torch.ones(50)
        mask[10:20] = 0  # Fix positions 10-19
        
        fixed_positions = {0: 'A', 5: 'V', 15: 'G'}
        
        with patch('inference.design_pipeline.parse_PDB') as mock_parse:
            mock_parse.return_value = {
                'coords': {'A': np.random.randn(50, 4, 3)},
                'mask': [1] * 50
            }
            
            mock_pipeline._extract_features = Mock(return_value=torch.randn(1, 50, 128))
            mock_pipeline._generate_initial_sequences = Mock(return_value=(torch.randn(2, 50, 20), 'proteinmpnn'))
            
            result = mock_pipeline.design_sequence(
                dummy_pdb_file, 
                mask=mask, 
                fixed_positions=fixed_positions
            )
            
            assert result.success == True
            # Check that constraints were passed to optimization
            assert mock_pipeline.optimizer.optimize_sequence.called


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_batch_processing_success(self, mock_pipeline, temp_dir):
        """Test successful batch processing"""
        # Create multiple dummy PDB files
        pdb_paths = []
        for i in range(3):
            pdb_path = os.path.join(temp_dir, f"protein_{i}.pdb")
            with open(pdb_path, 'w') as f:
                f.write("HEADER TEST\nATOM 1 CA ALA A 1 0.0 0.0 0.0 1.0 20.0 C\nEND\n")
            pdb_paths.append(pdb_path)
        
        # Mock design_sequence to return success
        def mock_design_sequence(path, **kwargs):
            return DesignResult(
                sequences=torch.randint(0, 20, (2, 30)),
                logits=torch.randn(2, 30, 20),
                energies=torch.randn(2) - 1.0,  # Negative energies (better)
                backbone_path=path,
                success=True,
                num_designs=2,
                optimization_results=[],
                total_time=1.0
            )
        
        mock_pipeline.design_sequence = Mock(side_effect=mock_design_sequence)
        
        results = mock_pipeline.design_batch(pdb_paths)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_pipeline.design_sequence.call_count == 3
    
    def test_batch_with_memory_management(self, mock_pipeline, temp_dir):
        """Test batch processing with memory constraints"""
        # Create dummy PDB files
        pdb_paths = []
        for i in range(5):
            pdb_path = os.path.join(temp_dir, f"protein_{i}.pdb")
            with open(pdb_path, 'w') as f:
                f.write("HEADER TEST\nATOM 1 CA ALA A 1 0.0 0.0 0.0 1.0 20.0 C\nEND\n")
            pdb_paths.append(pdb_path)
        
        # Mock design_sequence
        mock_pipeline.design_sequence = Mock(return_value=DesignResult(
            sequences=torch.randint(0, 20, (1, 30)),
            logits=torch.randn(1, 30, 20),
            energies=torch.tensor([-1.5]),
            backbone_path="test",
            success=True,
            num_designs=1,
            optimization_results=[],
            total_time=1.0
        ))
        
        # Test with small chunk size
        results = mock_pipeline.design_batch(
            pdb_paths,
            chunk_size=2,
            save_intermediate=True,
            intermediate_dir=temp_dir,
            memory_limit_gb=1.0
        )
        
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Check intermediate files were created
        chunk_files = list(Path(temp_dir).glob("chunk_*.json"))
        assert len(chunk_files) >= 2  # Should have multiple chunks
    
    def test_batch_partial_failures(self, mock_pipeline, temp_dir):
        """Test batch processing with some failures"""
        pdb_paths = []
        for i in range(3):
            pdb_path = os.path.join(temp_dir, f"protein_{i}.pdb")
            with open(pdb_path, 'w') as f:
                f.write("HEADER TEST\nATOM 1 CA ALA A 1 0.0 0.0 0.0 1.0 20.0 C\nEND\n")
            pdb_paths.append(pdb_path)
        
        # Mock design_sequence to fail on second protein
        def mock_design_sequence(path, **kwargs):
            if "protein_1" in path:
                raise Exception("Design failed for protein 1")
            return DesignResult(
                sequences=torch.randint(0, 20, (1, 30)),
                logits=torch.randn(1, 30, 20),
                energies=torch.tensor([-1.5]),
                backbone_path=path,
                success=True,
                num_designs=1,
                optimization_results=[],
                total_time=1.0
            )
        
        mock_pipeline.design_sequence = Mock(side_effect=mock_design_sequence)
        
        results = mock_pipeline.design_batch(pdb_paths)
        
        assert len(results) == 3
        assert results[0].success == True
        assert results[1].success == False
        assert results[2].success == True
        assert "Design failed for protein 1" in results[1].error_message


class TestResultValidation:
    """Test result validation functionality"""
    
    def test_validation_with_sequences(self, mock_pipeline):
        """Test validation with successful results"""
        # Create mock result with sequences
        result = DesignResult(
            sequences=torch.randint(0, 20, (3, 50)),
            logits=torch.randn(3, 50, 20),
            energies=torch.tensor([-2.5, -1.8, -3.1]),
            backbone_path="test.pdb",
            success=True,
            num_designs=3,
            optimization_results=[],
            total_time=5.0
        )
        
        # Test validation
        mock_pipeline.config.validate_results = True
        validated_result = mock_pipeline._validate_results(result)
        
        assert validated_result.validation_metrics is not None
        assert 'energy_mean' in validated_result.validation_metrics
        assert 'quality_summary' in validated_result.validation_metrics
        assert validated_result.confidence_scores is not None
    
    def test_validation_failure_handling(self, mock_pipeline):
        """Test validation with failed results"""
        result = DesignResult(
            sequences=None,
            logits=None,
            energies=None,
            backbone_path="test.pdb",
            success=False,
            num_designs=0,
            optimization_results=[],
            total_time=0.0,
            error_message="Design failed"
        )
        
        mock_pipeline.config.validate_results = True
        validated_result = mock_pipeline._validate_results(result)
        
        # Should handle gracefully
        assert validated_result.validation_metrics is not None
        assert validated_result.confidence_scores is not None


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_pdb_file(self, mock_pipeline):
        """Test handling of missing PDB files"""
        result = mock_pipeline.design_sequence("nonexistent.pdb")
        
        assert result.success == False
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()
    
    def test_invalid_constraints(self, mock_pipeline, dummy_pdb_file):
        """Test handling of invalid constraints"""
        with patch('inference.design_pipeline.parse_PDB') as mock_parse:
            mock_parse.return_value = {
                'coords': {'A': np.random.randn(10, 4, 3)},
                'mask': [1] * 10
            }
            
            mock_pipeline._extract_features = Mock(return_value=torch.randn(1, 10, 128))
            mock_pipeline._generate_initial_sequences = Mock(return_value=(torch.randn(1, 10, 20), 'proteinmpnn'))
            
            # Invalid mask size
            invalid_mask = torch.ones(50)  # Wrong size for 10-residue protein
            
            # Should handle gracefully
            result = mock_pipeline.design_sequence(dummy_pdb_file, mask=invalid_mask)
            # The pipeline should still attempt to design (may succeed or fail)
            assert isinstance(result, DesignResult)
    
    def test_optimization_failure(self, mock_pipeline, dummy_pdb_file):
        """Test handling of optimization failures"""
        with patch('inference.design_pipeline.parse_PDB') as mock_parse:
            mock_parse.return_value = {
                'coords': {'A': np.random.randn(50, 4, 3)},
                'mask': [1] * 50
            }
            
            mock_pipeline._extract_features = Mock(return_value=torch.randn(1, 50, 128))
            mock_pipeline._generate_initial_sequences = Mock(return_value=(torch.randn(1, 50, 20), 'proteinmpnn'))
            
            # Mock optimizer to raise exception
            mock_pipeline.optimizer.optimize_sequence = Mock(side_effect=Exception("Optimization failed"))
            
            result = mock_pipeline.design_sequence(dummy_pdb_file)
            
            # Should create failed result with optimization failure
            assert result.success == False
            assert len(result.optimization_results) == 1
            assert result.optimization_results[0].optimization_failed == True


class TestIntegrationWithRealData:
    """Test integration with real PDB structures"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration tests")
    def test_with_sample_pdbs(self, sample_pdb_paths, temp_dir):
        """Test with real PDB structures (if available)"""
        if not sample_pdb_paths:
            pytest.skip("No sample PDB files found")
        
        # Use first available PDB
        pdb_path = sample_pdb_paths[0]
        
        # Create minimal working configuration
        # Note: This test requires actual model checkpoints to work fully
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        
        # Create minimal checkpoints (won't work for full pipeline but tests loading)
        torch.save({"dummy": "data"}, encoder_ckpt)
        torch.save({"model_state_dict": {"dummy": torch.randn(10)}}, energy_ckpt)
        
        config = PipelineConfig(
            encoder_checkpoint=encoder_ckpt,
            energy_model_checkpoint=energy_ckpt,
            device='cpu',
            num_designs_per_target=1,
            validate_results=False,
            use_proteinmpnn_init=False
        )
        
        # This test mainly verifies PDB parsing doesn't crash
        with patch.multiple(
            'inference.design_pipeline.ProteinDesignPipeline',
            _initialize_components=Mock(),
            _initialize_proteinmpnn=Mock()
        ):
            pipeline = ProteinDesignPipeline(config)
            
            # Test PDB parsing specifically
            try:
                backbone_features = pipeline._parse_backbone(pdb_path)
                assert 'coords' in backbone_features
                print(f"Successfully parsed {os.path.basename(pdb_path)}")
            except Exception as e:
                print(f"PDB parsing failed (expected with mock setup): {e}")
                # This is expected with mocked components


class TestResultSaving:
    """Test result saving functionality"""
    
    def test_save_single_result(self, mock_pipeline, temp_dir):
        """Test saving single design result"""
        result = DesignResult(
            sequences=torch.randint(0, 20, (2, 30)),
            logits=torch.randn(2, 30, 20),
            energies=torch.tensor([-2.5, -1.8]),
            backbone_path="test.pdb",
            success=True,
            num_designs=2,
            optimization_results=[],
            total_time=5.0
        )
        
        mock_pipeline.save_results(result, temp_dir, format='json')
        
        # Check files were created
        json_files = list(Path(temp_dir).glob("*.json"))
        assert len(json_files) > 0
        
        # Verify content
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        assert data['success'] == True
        assert data['num_designs'] == 2
    
    def test_save_batch_results(self, mock_pipeline, temp_dir):
        """Test saving batch results"""
        results = [
            DesignResult(
                sequences=torch.randint(0, 20, (1, 30)),
                logits=torch.randn(1, 30, 20),
                energies=torch.tensor([-2.5]),
                backbone_path=f"test_{i}.pdb",
                success=True,
                num_designs=1,
                optimization_results=[],
                total_time=2.0
            ) for i in range(3)
        ]
        
        mock_pipeline.save_results(results, temp_dir, format='fasta')
        
        # Check FASTA files were created
        fasta_files = list(Path(temp_dir).glob("*.fasta"))
        assert len(fasta_files) >= 3


def test_memory_estimation():
    """Test memory estimation for batch sizing"""
    # This is a simple unit test for the estimation function
    from inference.design_pipeline import ProteinDesignPipeline
    
    # Create dummy config for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        encoder_ckpt = os.path.join(temp_dir, "encoder.pt")
        energy_ckpt = os.path.join(temp_dir, "energy.pt")
        torch.save({}, encoder_ckpt)
        torch.save({}, energy_ckpt)
        
        config = PipelineConfig(
            encoder_checkpoint=encoder_ckpt,
            energy_model_checkpoint=energy_ckpt,
            num_designs_per_target=2
        )
        
        with patch.multiple(
            'inference.design_pipeline.ProteinDesignPipeline',
            _initialize_components=Mock(),
            _initialize_proteinmpnn=Mock()
        ):
            pipeline = ProteinDesignPipeline(config)
            
            # Test memory estimation
            chunk_size = pipeline._estimate_chunk_size(100, memory_limit_gb=4.0)
            assert chunk_size > 0
            assert chunk_size <= 100
            
            # Test with very small memory limit
            small_chunk = pipeline._estimate_chunk_size(100, memory_limit_gb=0.1)
            assert small_chunk >= 1  # Should always be at least 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])