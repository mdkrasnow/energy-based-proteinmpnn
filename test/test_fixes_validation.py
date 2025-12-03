#!/usr/bin/env python3
"""
Validation test for the three critical fixes applied to PDB Cache.

This test validates:
1. ROB-001: Deadlock risk fixes in RLock usage under high concurrency
2. IMP-002: Data corruption fixes in download byte counting 
3. REP-003: Non-deterministic test data fixes
"""

import tempfile
import shutil
import time
import threading
from pathlib import Path
import sys

# Direct import of just the cache module to avoid package issues
sys.path.insert(0, str(Path(__file__).parent))

# Import specific classes to test fixes
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pdb_cache", 
    Path(__file__).parent / "hybrid" / "data" / "pdb_cache.py"
)
pdb_cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdb_cache_module)

PDBCache = pdb_cache_module.PDBCache
CacheStatistics = pdb_cache_module.CacheStatistics


def test_rob001_deadlock_prevention():
    """Test ROB-001: Verify deadlock prevention in concurrent access."""
    print("Testing ROB-001 deadlock prevention...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=10,
            max_disk_gb=0.01
        )
        
        # Simulate high concurrency that could trigger deadlock
        def worker(worker_id):
            for i in range(10):
                pdb_id = f"{worker_id}AB{i}"  # Valid format
                try:
                    # This should not deadlock with the fixes
                    cache.get(pdb_id)
                except Exception as e:
                    # Expected to fail since we don't have real URLs, but should not deadlock
                    pass
        
        threads = []
        for i in range(10):  # High number of threads
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # All threads should complete without deadlock
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=5.0)  # Should not take more than 5 seconds
            if thread.is_alive():
                raise Exception("Potential deadlock detected - thread did not complete")
        
        elapsed = time.time() - start_time
        print(f"Concurrent operations completed in {elapsed:.2f}s (no deadlock)")
        print("✓ ROB-001 deadlock prevention test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_imp002_byte_counting_accuracy():
    """Test IMP-002: Verify accurate byte counting without corruption."""
    print("Testing IMP-002 byte counting accuracy...")
    
    stats = CacheStatistics()
    
    # Test the download success recording
    test_sizes = [1024, 2048, 4096, 8192]
    total_expected = sum(test_sizes)
    
    for size in test_sizes:
        stats.record_download_success(size)
    
    comprehensive_stats = stats.get_comprehensive_stats()
    actual_total = comprehensive_stats['download_performance']['total_bytes']
    
    if actual_total != total_expected:
        raise Exception(f"Byte counting corrupted: expected {total_expected}, got {actual_total}")
    
    # Test byte counting consistency
    assert comprehensive_stats['download_performance']['successes'] == len(test_sizes)
    assert comprehensive_stats['download_performance']['attempts'] == 0  # No attempts recorded yet
    
    # Add download attempts
    for _ in test_sizes:
        stats.record_download_attempt()
    
    comprehensive_stats = stats.get_comprehensive_stats()
    success_rate = comprehensive_stats['download_performance']['success_rate']
    
    # Should be 100% success rate (4 successes / 4 attempts)
    if success_rate != 1.0:
        raise Exception(f"Success rate calculation corrupted: expected 1.0, got {success_rate}")
    
    print(f"Byte counting accuracy verified: {actual_total} bytes tracked correctly")
    print("✓ IMP-002 byte counting accuracy test passed")


def test_rep003_deterministic_behavior():
    """Test REP-003: Verify completely deterministic test behavior."""
    print("Testing REP-003 deterministic behavior...")
    
    # Fixed test data that should produce identical results every time
    test_pdb_ids = [
        "1A2B", "2C3D", "4E5F", "6G7H", "8I9J"  # Deterministic set
    ]
    
    # Run the same test multiple times and verify identical results
    results = []
    
    for run in range(3):  # Multiple runs
        temp_dir = tempfile.mkdtemp()
        try:
            cache = PDBCache(
                cache_dir=Path(temp_dir),
                max_memory_mb=10,
                max_disk_gb=0.01
            )
            
            # Perform identical operations
            for pdb_id in test_pdb_ids:
                cache.get(pdb_id)  # Will fail but record statistics
            
            stats = cache.get_stats()
            result = {
                'total_requests': stats['cache_performance']['total_requests'],
                'cache_hits': stats['cache_performance']['hits'],
                'cache_misses': stats['cache_performance']['misses'],
                'unique_ids': stats['cache_performance']['unique_pdb_ids']
            }
            results.append(result)
            
        finally:
            shutil.rmtree(temp_dir)
    
    # All results should be identical for deterministic behavior
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        if result != first_result:
            raise Exception(f"Non-deterministic behavior detected: run {i+1} differs from run 1")
    
    expected_requests = len(test_pdb_ids)
    if first_result['total_requests'] != expected_requests:
        raise Exception(f"Expected {expected_requests} requests, got {first_result['total_requests']}")
    
    print(f"Deterministic behavior verified across {len(results)} runs")
    print(f"Consistent results: {first_result}")
    print("✓ REP-003 deterministic behavior test passed")


def test_concurrent_statistics_accuracy():
    """Test that concurrent operations maintain accurate statistics."""
    print("Testing concurrent statistics accuracy...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=10,
            max_disk_gb=0.01
        )
        
        # Fixed deterministic PDB IDs for reproducible results
        test_pdb_ids = [
            "1A1B", "2B2C", "3C3D", "4D4E", "5E5F",
            "6F6G", "7G7H", "8H8I", "9I9J", "0J0K"
        ]
        
        def worker(worker_id, num_requests=10):
            for i in range(num_requests):
                pdb_idx = (worker_id * num_requests + i) % len(test_pdb_ids)
                pdb_id = test_pdb_ids[pdb_idx]
                cache.get(pdb_id)
        
        # Get baseline
        baseline_stats = cache.get_stats()
        baseline_requests = baseline_stats['cache_performance']['total_requests']
        
        # Start deterministic concurrent operations
        threads = []
        num_threads = 5
        requests_per_thread = 20
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i, requests_per_thread))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise Exception("Thread timeout - possible deadlock or performance issue")
        
        # Verify statistics accuracy
        final_stats = cache.get_stats()
        final_requests = final_stats['cache_performance']['total_requests']
        new_requests = final_requests - baseline_requests
        expected_requests = num_threads * requests_per_thread
        
        if new_requests != expected_requests:
            raise Exception(f"Statistics accuracy failed: expected {expected_requests} new requests, got {new_requests}")
        
        print(f"Concurrent statistics accurate: {new_requests}/{expected_requests} requests tracked correctly")
        print("✓ Concurrent statistics accuracy test passed")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Validating Critical Fixes for PDB Cache Performance Monitoring")
    print("=" * 70)
    
    try:
        test_rob001_deadlock_prevention()
        test_imp002_byte_counting_accuracy() 
        test_rep003_deterministic_behavior()
        test_concurrent_statistics_accuracy()
        
        print("=" * 70)
        print("✅ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY")
        print("✅ ROB-001: Deadlock prevention working")
        print("✅ IMP-002: Byte counting accuracy fixed")
        print("✅ REP-003: Deterministic behavior implemented")
        
    except Exception as e:
        print(f"❌ Fix validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)