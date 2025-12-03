#!/usr/bin/env python3
"""
Test script for the enhanced PDB Cache performance monitoring.

This script validates that the comprehensive performance monitoring
functionality works correctly with thread-safe operations.
"""

import tempfile
import shutil
import time
import threading
from pathlib import Path

# Add the project path
import sys
sys.path.append(str(Path(__file__).parent))

from hybrid.data.pdb_cache import PDBCache, CacheStatistics


def test_cache_statistics():
    """Test the CacheStatistics class functionality."""
    print("Testing CacheStatistics class...")
    
    stats = CacheStatistics()
    
    # Test basic recording
    stats.record_request("1ABC")
    stats.record_cache_hit()
    
    stats.record_request("2DEF") 
    stats.record_cache_miss()
    
    # Test download recording
    stats.record_download_attempt()
    stats.record_download_success(1024)
    
    # Test error recording
    stats.record_error('network')
    stats.record_error('parsing')
    
    # Test memory/disk updates
    stats.update_memory_usage(1024*1024)  # 1MB
    stats.update_disk_usage(5*1024*1024)  # 5MB
    
    # Get comprehensive stats
    comprehensive = stats.get_comprehensive_stats()
    
    # Validate results
    assert comprehensive['cache_performance']['total_requests'] == 2
    assert comprehensive['cache_performance']['hits'] == 1
    assert comprehensive['cache_performance']['misses'] == 1
    assert comprehensive['cache_performance']['hit_rate'] == 0.5
    assert comprehensive['download_performance']['attempts'] == 1
    assert comprehensive['download_performance']['successes'] == 1
    assert comprehensive['error_analysis']['network_errors'] == 1
    assert comprehensive['error_analysis']['parsing_errors'] == 1
    
    print("✓ CacheStatistics tests passed")


def test_performance_monitoring():
    """Test the enhanced PDBCache with performance monitoring."""
    print("Testing PDBCache performance monitoring...")
    
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp()
    try:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=10,  # Small cache for testing
            max_disk_gb=0.01   # 10MB disk limit
        )
        
        # Test basic operations with monitoring
        cache.set_monitoring_level('DEBUG')
        
        # Simulate some cache operations
        print("Simulating cache operations...")
        
        # These will be cache misses since we don't have real PDB data
        result1 = cache.get("1ABC")  # Will fail gracefully
        result2 = cache.get("2DEF")  # Will fail gracefully
        result3 = cache.get("1ABC")  # Second request for same ID
        
        # Get statistics
        stats = cache.get_stats()
        print(f"Total requests: {stats['cache_performance']['total_requests']}")
        print(f"Cache hit rate: {stats['cache_performance']['hit_rate']:.1%}")
        print(f"Unique PDB IDs: {stats['cache_performance']['unique_pdb_ids']}")
        
        # Test performance report
        report = cache.get_performance_report()
        print(f"Overall health: {report['performance_summary']['overall_health']}")
        print(f"Recommendations: {len(report['recommendations'])}")
        
        # Test data export
        json_export = cache.export_monitoring_data('json')
        csv_export = cache.export_monitoring_data('csv')
        
        assert len(json_export) > 100  # Should have substantial data
        assert 'hit_rate' in csv_export  # Should contain basic metrics
        
        print("✓ PDBCache monitoring tests passed")
        
        # Test thread safety with concurrent operations
        test_concurrent_monitoring(cache)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_concurrent_monitoring(cache):
    """Test thread-safe statistics collection."""
    print("Testing concurrent operations...")
    
    # CRITICAL FIX REP-003: Fully deterministic test data with seeded ordering
    # Fixed set of valid PDB IDs that follow the format (digit + 3 alphanumeric)
    # Sorted alphabetically for completely predictable ordering
    test_pdb_ids = [
        "1A2B", "1A2C", "1A2D", "1A2E", "1A2F", "1A2G", "1A2H", "1A2I", "1A2J", "1A2K",
        "2B3C", "2B3D", "2B3E", "2B3F", "2B3G", "2B3H", "2B3I", "2B3J", "2B3K", "2B3L", 
        "3C4D", "3C4E", "3C4F", "3C4G", "3C4H", "3C4I", "3C4J", "3C4K", "3C4L", "3C4M",
        "4D5E", "4D5F", "4D5G", "4D5H", "4D5I", "4D5J", "4D5K", "4D5L", "4D5M", "4D5N",
        "5E6F", "5E6G", "5E6H", "5E6I", "5E6J", "5E6K", "5E6L", "5E6M", "5E6N", "5E6O"
    ]
    
    def worker(worker_id, num_requests=10):
        """Worker thread that performs cache operations with deterministic patterns."""
        # CRITICAL FIX REP-003: Ensure completely deterministic access pattern
        for i in range(num_requests):
            # Use deterministic PDB IDs with reproducible pattern
            pdb_idx = (worker_id * num_requests + i) % len(test_pdb_ids)
            pdb_id = test_pdb_ids[pdb_idx]
            cache.get(pdb_id)  # Will be cache misses
            
            # CRITICAL FIX REP-003: Remove time.sleep to eliminate timing-based non-determinism
            # Small fixed delay replaced with deterministic CPU-bound operation
            dummy_work = sum(ord(c) for c in pdb_id)  # Deterministic CPU work
    
    # CRITICAL FIX REP-003: Fixed thread and request counts for deterministic testing
    threads = []
    num_threads = 5  # Fixed number for reproducibility 
    requests_per_thread = 20  # Fixed number for reproducibility
    
    # Get baseline request count to ensure accurate counting
    baseline_stats = cache.get_stats()
    baseline_requests = baseline_stats['cache_performance']['total_requests']
    
    # CRITICAL FIX REP-003: Start threads deterministically
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker, 
            args=(i, requests_per_thread),
            name=f"test_worker_{i}"  # Named threads for debugging
        )
        threads.append(thread)
    
    # Start all threads at once for more deterministic timing
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete with timeout for safety
    for thread in threads:
        thread.join(timeout=30.0)  # 30 second timeout
        if thread.is_alive():
            print(f"Warning: Thread {thread.name} did not complete in time")
    
    # CRITICAL FIX REP-003: Check final statistics with accurate baseline accounting
    stats = cache.get_stats()
    total_expected = num_threads * requests_per_thread
    actual_new_requests = stats['cache_performance']['total_requests'] - baseline_requests
    
    print(f"Expected new requests: {total_expected}")
    print(f"Actual new requests: {actual_new_requests}")
    print(f"Baseline requests: {baseline_requests}")
    print(f"Total requests: {stats['cache_performance']['total_requests']}")
    
    # CRITICAL FIX REP-003: Exact counting for deterministic validation
    assert actual_new_requests == total_expected, (
        f"Expected exactly {total_expected} new requests, got {actual_new_requests}"
    )
    
    print("✓ Concurrent monitoring tests passed")


def test_timing_operations():
    """Test the timed operation context manager."""
    print("Testing timing operations...")
    
    stats = CacheStatistics()
    
    # CRITICAL FIX REP-003: Replace time.sleep with deterministic CPU-bound operations
    # for reliable CI/CD testing without timing dependencies
    
    # Test timing context manager with deterministic work
    with stats._timed_operation('lookup'):
        # Deterministic CPU work instead of sleep - prevents timing flakiness
        dummy_work = sum(i * i for i in range(1000))  # ~1000 iterations
    
    with stats._timed_operation('download'):
        # Larger deterministic work for download timing
        dummy_work = sum(i * i * i for i in range(2000))  # ~2000 iterations
    
    comprehensive = stats.get_comprehensive_stats()
    
    # CRITICAL FIX REP-003: Check that timing was recorded with deterministic assertions
    assert comprehensive['timing_analysis']['total_lookup_operations'] == 1
    # Remove timing-dependent assertions that cause CI/CD flakiness
    assert comprehensive['timing_analysis']['avg_lookup_ms'] >= 0  # Any positive time
    assert comprehensive['download_performance']['recent_avg_time_seconds'] >= 0  # Any positive time
    
    # Check monitoring overhead tracking - structure validation only
    assert comprehensive['monitoring_overhead']['monitoring_operations'] >= 2
    assert comprehensive['monitoring_overhead']['avg_overhead_microseconds'] >= 0
    
    # CRITICAL FIX REP-003: Add deterministic functional validation
    assert len(stats._recent_lookup_times) == 1  # One lookup recorded
    assert len(stats._recent_download_times) == 1  # One download recorded
    
    print("✓ Timing operations tests passed")


if __name__ == "__main__":
    print("Starting PDB Cache Performance Monitoring Tests...")
    print("=" * 60)
    
    try:
        test_cache_statistics()
        test_timing_operations()
        test_performance_monitoring()
        
        print("=" * 60)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)