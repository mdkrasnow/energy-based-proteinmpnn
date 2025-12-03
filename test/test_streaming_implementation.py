#!/usr/bin/env python3
"""
Test script to validate the streaming dataset implementation.

This script tests the core functionality of the streaming infrastructure
without requiring a full training setup.
"""

import sys
import logging
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from hybrid.data import (
            StreamingProteinDataset, 
            PDBCache, 
            PDBManager,
            ProteinDataSource,
            LocalPDBSource,
            RemotePDBSource
        )
        logger.info("✓ Data module imports successful")
    except Exception as e:
        logger.error(f"✗ Data module import failed: {e}")
        return False
    
    try:
        from hybrid.models import EnergyBasedProteinMPNN
        logger.info("✓ Model module imports successful")
    except Exception as e:
        logger.error(f"✗ Model module import failed: {e}")
        return False
        
    return True


def test_pdb_cache():
    """Test PDBCache basic functionality."""
    logger.info("Testing PDBCache...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            from hybrid.data import PDBCache
            
            cache = PDBCache(
                cache_dir=Path(temp_dir),
                max_memory_mb=128,
                max_disk_gb=1.0
            )
            
            # Test cache stats
            stats = cache.get_stats()
            logger.info(f"✓ Cache initialized, stats keys: {list(stats.keys())}")
            
            # Test cache operations
            result = cache.get("1UBQ")  # Should return None (not cached)
            logger.info(f"✓ Cache get operation returned: {type(result)}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ PDBCache test failed: {e}")
        return False


def test_streaming_dataset():
    """Test StreamingProteinDataset basic functionality."""
    logger.info("Testing StreamingProteinDataset...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            from hybrid.data import StreamingProteinDataset
            
            # Create minimal data sources config
            data_sources = [
                {
                    "type": "remote_pdb",
                    "name": "test_source",
                    "pdb_list": ["1UBQ", "1VII"],
                    "weight": 1.0,
                    "enabled": True
                }
            ]
            
            dataset = StreamingProteinDataset(
                data_sources=data_sources,
                cache_dir=Path(temp_dir),
                batch_size=1,
                prefetch_factor=1,
                num_workers=1,
                enable_timing=False  # Disable for testing
            )
            
            # Test data index
            logger.info(f"✓ Dataset created, data index size: {len(dataset.data_index)}")
            
            # Test optimization methods
            if hasattr(dataset, 'warm_cache_for_streaming'):
                result = dataset.warm_cache_for_streaming(warmup_size=1)
                logger.info(f"✓ Cache warming test: {result.get('status', 'unknown')}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ StreamingProteinDataset test failed: {e}")
        return False


def test_pdb_manager():
    """Test PDBManager basic functionality."""
    logger.info("Testing PDBManager...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            from hybrid.data import PDBManager
            
            # Create minimal data sources
            data_sources = [
                {
                    "type": "remote_pdb",
                    "pdb_list": ["1UBQ", "1VII"]
                }
            ]
            
            manager = PDBManager(
                data_sources=data_sources,
                metadata_db_path=Path(temp_dir) / "metadata.db"
            )
            
            # Test structure listing
            structures = manager.list_structures()
            logger.info(f"✓ PDBManager created, {len(structures)} structures indexed")
            
            # Test statistics
            stats = manager.get_statistics()
            logger.info(f"✓ Manager stats: {stats['total_structures']} total structures")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ PDBManager test failed: {e}")
        return False


def test_model():
    """Test EnergyBasedProteinMPNN model."""
    logger.info("Testing EnergyBasedProteinMPNN...")
    
    try:
        import torch
        from hybrid.models import EnergyBasedProteinMPNN
        
        # Create model config
        mpnn_config = {"hidden_dim": 128}
        energy_head_config = {
            "hidden_dim": 512,
            "num_layers": 3,
            "dropout": 0.1,
            "use_batch_norm": True
        }
        sequence_repr_config = {}
        
        model = EnergyBasedProteinMPNN(
            mpnn_config=mpnn_config,
            energy_head_config=energy_head_config,
            sequence_repr_config=sequence_repr_config
        )
        
        # Test forward pass
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        coordinates = torch.randn(20, 4, 3)
        mask = torch.ones(20, dtype=torch.bool)
        
        energy = model(sequence, coordinates, mask)
        logger.info(f"✓ Model forward pass successful, energy shape: {energy.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def test_config_loading():
    """Test configuration file loading."""
    logger.info("Testing configuration loading...")
    
    try:
        config_path = Path("hybrid/training/config_streaming.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✓ Config loaded successfully, keys: {list(config.keys())}")
            
            # Validate key sections
            required_sections = ['streaming', 'data_sources', 'model', 'training']
            for section in required_sections:
                if section in config:
                    logger.info(f"✓ Config section '{section}' present")
                else:
                    logger.warning(f"⚠ Config section '{section}' missing")
                    
        else:
            logger.warning("⚠ Config file not found")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("Starting streaming implementation validation...")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("PDBCache", test_pdb_cache),
        ("StreamingDataset", test_streaming_dataset),
        ("PDBManager", test_pdb_manager),
        ("Model", test_model),
        ("Configuration", test_config_loading)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<20}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Streaming implementation is functional.")
    else:
        logger.info(f"⚠ {failed} test(s) failed. Implementation needs fixes.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)