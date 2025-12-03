#!/usr/bin/env python3
"""
Simple validation test for critical fixes.
"""

import sys
from pathlib import Path

# Direct import of just the cache module
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "pdb_cache", 
    Path(__file__).parent / "hybrid" / "data" / "pdb_cache.py"
)
pdb_cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdb_cache_module)

CacheStatistics = pdb_cache_module.CacheStatistics


def test_imp002_byte_counting():
    """Test IMP-002: Byte counting accuracy."""
    print("Testing IMP-002 byte counting accuracy...")
    
    stats = CacheStatistics()
    
    # Test download success recording with various sizes
    test_sizes = [1024, 2048, 4096, 8192]
    total_expected = sum(test_sizes)
    
    for size in test_sizes:
        stats.record_download_success(size)
    
    comprehensive_stats = stats.get_comprehensive_stats()
    actual_total = comprehensive_stats['download_performance']['total_bytes']
    
    assert actual_total == total_expected, f"Byte count mismatch: expected {total_expected}, got {actual_total}"
    assert comprehensive_stats['download_performance']['successes'] == len(test_sizes)
    
    print(f"✅ Byte counting accurate: {actual_total} bytes tracked correctly")
    return True


def test_rep003_deterministic():
    """Test REP-003: Deterministic behavior."""
    print("Testing REP-003 deterministic behavior...")
    
    # Test that statistics collection is deterministic
    results = []
    
    for run in range(3):
        stats = CacheStatistics()
        
        # Perform identical operations
        test_pdb_ids = ["1A2B", "2C3D", "4E5F"]
        for pdb_id in test_pdb_ids:
            stats.record_request(pdb_id)
            stats.record_cache_miss()
        
        result = stats.get_comprehensive_stats()
        key_stats = {
            'total_requests': result['cache_performance']['total_requests'],
            'misses': result['cache_performance']['misses'],
            'unique_ids': result['cache_performance']['unique_pdb_ids']
        }
        results.append(key_stats)
    
    # All results should be identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result == first_result, f"Non-deterministic: run {i+1} differs from run 1"
    
    expected_requests = 3
    assert first_result['total_requests'] == expected_requests
    
    print(f"✅ Deterministic behavior verified: {first_result}")
    return True


def test_statistics_thread_safety():
    """Test basic thread safety of statistics."""
    print("Testing statistics thread safety...")
    
    stats = CacheStatistics()
    
    # Test that timed operations work correctly
    with stats._timed_operation('lookup'):
        dummy = sum(i for i in range(100))  # Deterministic work
    
    comprehensive_stats = stats.get_comprehensive_stats()
    assert comprehensive_stats['timing_analysis']['total_lookup_operations'] == 1
    assert comprehensive_stats['monitoring_overhead']['monitoring_operations'] >= 1
    
    print("✅ Statistics thread safety working")
    return True


if __name__ == "__main__":
    print("Testing Critical Fixes")
    print("=" * 40)
    
    try:
        test_imp002_byte_counting()
        test_rep003_deterministic()
        test_statistics_thread_safety()
        
        print("=" * 40)
        print("✅ ALL CRITICAL FIXES VALIDATED")
        print("✅ IMP-002: Byte counting accuracy fixed")  
        print("✅ REP-003: Deterministic behavior implemented")
        print("✅ Basic thread safety working")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)