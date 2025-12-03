#!/usr/bin/env python3
"""
Real RCSB API integration test for PDBListManager.

This script tests the actual RCSB API integration with realistic
criteria matching the training requirements.
"""

import sys
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/Users/mkrasnow/Desktop/energy-based-proteinmpnn')

from hybrid.data.pdb_manager import PDBListManager

def test_real_rcsb_integration():
    """Test PDBListManager with real RCSB API call."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Real RCSB API Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize manager with realistic settings
        manager = PDBListManager(
            cache_dir=Path(temp_dir),
            search_timeout=30,
            max_retries=3,
            rate_limit_delay=0.5
        )
        
        # Test with training criteria: resolution ≤3.5Å, length 20-500 residues
        print("Fetching PDB structures with training criteria...")
        print("- Resolution ≤ 3.5Å")
        print("- Sequence length: 20-500 residues")
        print("- Methods: X-RAY DIFFRACTION, ELECTRON MICROSCOPY")
        print("- Target count: 100 structures")
        
        pdb_list = manager.get_filtered_pdb_list(
            max_resolution=3.5,
            min_length=20,
            max_length=500,
            experimental_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
            target_count=100,
            use_cache=False  # Force fresh API call
        )
        
        print(f"\nResults:")
        print(f"- Retrieved {len(pdb_list)} PDB structures")
        print(f"- Sample PDB IDs: {pdb_list[:10]}")
        
        # Validate PDB ID format
        valid_format = all(len(pdb_id) == 4 and pdb_id[0].isdigit() for pdb_id in pdb_list)
        print(f"- All PDB IDs have valid format: {valid_format}")
        
        # Test caching
        print(f"\nTesting cache functionality...")
        cached_list = manager.get_filtered_pdb_list(
            max_resolution=3.5,
            min_length=20,
            max_length=500,
            experimental_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
            target_count=50,
            use_cache=True  # Should use cache
        )
        
        print(f"- Retrieved {len(cached_list)} from cache")
        print(f"- Cache working: {len(cached_list) > 0}")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"\nManager statistics:")
        print(f"- Cache directory: {stats['cache_dir']}")
        print(f"- Cache exists: {stats['cache_exists']}")
        print(f"- Cached PDB count: {stats.get('cached_pdb_count', 0)}")
        if 'cache_timestamp' in stats:
            print(f"- Cache timestamp: {stats['cache_timestamp']}")
        
        # Test high-resolution filter (should return fewer results)
        print(f"\nTesting stricter filters...")
        high_res_list = manager.get_filtered_pdb_list(
            max_resolution=2.0,  # Stricter resolution
            min_length=50,
            max_length=200,
            target_count=50,
            use_cache=False
        )
        
        print(f"- High-resolution structures (≤2.0Å): {len(high_res_list)}")
        
        # Success criteria
        success = (
            len(pdb_list) >= 50 and  # Got substantial number of structures
            valid_format and         # All have valid PDB ID format
            len(cached_list) > 0 and # Caching works
            stats['cache_exists']    # Cache file created
        )
        
        print(f"\n=== Integration Test {'PASSED' if success else 'FAILED'} ===")
        return success

def test_biological_scenarios():
    """Test pre-defined biological query scenarios."""
    
    print("\n=== Testing Biological Query Scenarios ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = PDBListManager(cache_dir=Path(temp_dir))
        
        # Test different biological scenarios
        scenarios = [
            ("small_proteins", {"max_resolution": 2.5, "min_length": 20, "max_length": 150, "target_count": 10}),
            ("medium_proteins", {"max_resolution": 3.0, "min_length": 150, "max_length": 300, "target_count": 10}),
            ("large_proteins", {"max_resolution": 3.5, "min_length": 300, "max_length": 500, "target_count": 10}),
        ]
        
        results = {}
        for scenario_name, params in scenarios:
            print(f"\nTesting {scenario_name}...")
            pdb_list = manager.get_filtered_pdb_list(**params)
            results[scenario_name] = len(pdb_list)
            print(f"- Retrieved {len(pdb_list)} structures")
            print(f"- Sample: {pdb_list[:5]}")
        
        # Check that we get different sized result sets
        unique_counts = len(set(results.values()))
        print(f"\nScenario diversity: {unique_counts} different result counts")
        
        return all(count > 0 for count in results.values())

if __name__ == "__main__":
    print("Testing PDBListManager RCSB API Integration")
    print("=" * 50)
    
    try:
        # Test real integration
        integration_success = test_real_rcsb_integration()
        
        # Test biological scenarios  
        scenarios_success = test_biological_scenarios()
        
        overall_success = integration_success and scenarios_success
        
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"RCSB API Integration: {'PASS' if integration_success else 'FAIL'}")
        print(f"Biological Scenarios: {'PASS' if scenarios_success else 'FAIL'}")
        print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        if overall_success:
            print("\n🎉 PDBListManager successfully integrates with RCSB API!")
            print("✅ Ready for production use in training pipeline")
        else:
            print("\n❌ Some tests failed - review implementation")
            
        sys.exit(0 if overall_success else 1)
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)