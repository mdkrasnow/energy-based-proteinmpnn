#!/usr/bin/env python3
"""
Test script for PDBListManager with RCSB API integration.

This script validates the PDBListManager implementation including:
- RCSB API integration
- Caching functionality 
- Fallback mechanisms
- Error handling
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/Users/mkrasnow/Desktop/energy-based-proteinmpnn')

from hybrid.data.pdb_manager import PDBListManager

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_pdb_list_manager_basic():
    """Test basic PDBListManager functionality."""
    print("=== Testing PDBListManager Basic Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize manager
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        # Test statistics (empty initially)
        stats = manager.get_statistics()
        print(f"Initial stats: {stats}")
        
        # Test fallback list
        print("\nTesting fallback list...")
        fallback_list = manager._get_fallback_list()
        print(f"Fallback list size: {len(fallback_list)}")
        print(f"Sample PDB IDs: {fallback_list[:10]}")
        
        # Validate PDB ID format
        print("\nTesting PDB ID validation...")
        test_ids = ["1UBQ", "invalid", "2ABC", "12345", "XYZ1", "4ABC"]
        valid_ids = manager._validate_pdb_ids(test_ids)
        print(f"Valid IDs from {test_ids}: {valid_ids}")
        
        return True

def test_cache_functionality():
    """Test caching functionality."""
    print("\n=== Testing Cache Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        # Test saving and loading cache
        test_pdb_list = ["1UBQ", "2CRO", "3HTN", "4YOW", "5L33"]
        
        print("Saving test PDB list to cache...")
        manager._save_cached_list(test_pdb_list)
        
        print("Loading cached PDB list...")
        cached_list = manager._load_cached_list(max_age_hours=1)
        
        print(f"Cached list: {cached_list}")
        print(f"Cache successful: {cached_list == test_pdb_list}")
        
        # Test cache statistics
        stats = manager.get_statistics()
        print(f"Cache stats: {stats}")
        
        return cached_list == test_pdb_list

def test_rcsb_query_construction():
    """Test RCSB query construction."""
    print("\n=== Testing RCSB Query Construction ===")
    
    manager = PDBListManager()
    
    # Build a test query
    query = manager._build_rcsb_query(
        max_resolution=3.5,
        min_length=20,
        max_length=500,
        experimental_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    )
    
    print("Generated RCSB query structure:")
    print(f"Query type: {query.get('query', {}).get('type')}")
    print(f"Logical operator: {query.get('query', {}).get('logical_operator')}")
    print(f"Number of filter nodes: {len(query.get('query', {}).get('nodes', []))}")
    print(f"Return type: {query.get('return_type')}")
    
    # Check query structure
    nodes = query.get('query', {}).get('nodes', [])
    filter_types = []
    for node in nodes:
        if 'parameters' in node:
            attr = node['parameters'].get('attribute', 'unknown')
            filter_types.append(attr)
    
    print(f"Filter attributes: {filter_types}")
    
    expected_filters = [
        "rcsb_entry_info.resolution_combined",
        "entity_poly.rcsb_sample_sequence_length",
        "exptl.method",
        "entity_poly.rcsb_entity_polymer_type"
    ]
    
    has_all_filters = all(any(ef in f for f in filter_types) for ef in ["resolution", "length", "method", "polymer_type"])
    print(f"Has all expected filter types: {has_all_filters}")
    
    return has_all_filters

def test_biological_query_sets():
    """Test biological query sets functionality."""
    print("\n=== Testing Biological Query Sets ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        print("Getting biological query sets (using fallback)...")
        try:
            # This will use fallback since we don't have network access in test
            query_sets = manager.get_biological_query_sets()
            
            print("Query set types and sizes:")
            for name, pdb_list in query_sets.items():
                print(f"  {name}: {len(pdb_list)} PDBs")
                print(f"    Sample: {pdb_list[:5]}")
                
            expected_sets = ["high_quality", "diverse_folds", "large_proteins", "small_proteins"]
            has_all_sets = all(s in query_sets for s in expected_sets)
            print(f"Has all expected query sets: {has_all_sets}")
            
            return has_all_sets
            
        except Exception as e:
            print(f"Error in biological query sets: {e}")
            return False

def test_filtered_pdb_list():
    """Test the main filtered PDB list functionality."""
    print("\n=== Testing Filtered PDB List (Fallback Mode) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        print("Getting filtered PDB list (will use fallback)...")
        try:
            # This should use fallback since we likely don't have network access
            pdb_list = manager.get_filtered_pdb_list(
                max_resolution=3.5,
                min_length=20,
                max_length=500,
                target_count=100,
                use_cache=False  # Force fresh request (will fall back to emergency list)
            )
            
            print(f"Retrieved {len(pdb_list)} PDB structures")
            print(f"Sample PDBs: {pdb_list[:10]}")
            
            # Validate PDB ID formats
            valid_format = all(len(pdb_id) == 4 and pdb_id[0].isdigit() for pdb_id in pdb_list[:10])
            print(f"PDB IDs have valid format: {valid_format}")
            
            # Test caching of results
            stats = manager.get_statistics()
            print(f"Manager statistics: {stats}")
            
            return len(pdb_list) > 0 and valid_format
            
        except Exception as e:
            print(f"Error in filtered PDB list: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_error_handling():
    """Test error handling and robustness."""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        # Test with invalid cache directory permissions (if possible)
        # This is a basic test for robustness
        
        print("Testing error handling scenarios...")
        
        # Test invalid PDB ID validation
        invalid_ids = ["", "INVALID", "12345", "abc", None, 123]
        try:
            valid_ids = manager._validate_pdb_ids(invalid_ids)
            print(f"Invalid ID handling successful: {len(valid_ids)} valid IDs from invalid input")
        except Exception as e:
            print(f"Error in PDB ID validation: {e}")
            return False
            
        # Test loading non-existent cache
        try:
            cached_list = manager._load_cached_list(max_age_hours=1)
            print(f"Non-existent cache handling: {cached_list is None}")
        except Exception as e:
            print(f"Error loading non-existent cache: {e}")
            return False
            
        return True

def main():
    """Main test function."""
    setup_logging()
    
    print("Testing PDBListManager with RCSB Integration")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_pdb_list_manager_basic),
        ("Cache Functionality", test_cache_functionality), 
        ("RCSB Query Construction", test_rcsb_query_construction),
        ("Biological Query Sets", test_biological_query_sets),
        ("Filtered PDB List", test_filtered_pdb_list),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'-'*20}")
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            results[test_name] = {"passed": result, "time": elapsed}
            status = "PASSED" if result else "FAILED"
            print(f"{test_name}: {status} ({elapsed:.2f}s)")
        except Exception as e:
            results[test_name] = {"passed": False, "error": str(e), "time": 0}
            print(f"{test_name}: FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed_tests = sum(1 for r in results.values() if r["passed"])
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        time_str = f"({result['time']:.2f}s)" if "time" in result else ""
        print(f"{test_name}: {status} {time_str}")
        if "error" in result:
            print(f"  Error: {result['error']}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! PDBListManager implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())