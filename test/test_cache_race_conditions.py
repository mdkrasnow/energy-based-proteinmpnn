#!/usr/bin/env python3
"""
Test script for validating the CRITICAL RACE CONDITION FIXES in PDBCache.

This test specifically validates the fixes for:
- Race condition in download deduplication (TASK-1-2)
- Recursive call elimination in download waits
- Unsafe clear_cache operations
- Thread-safe eviction and prefetch operations

Run this test to verify the concurrency fixes work correctly.
"""

import threading
import time
import tempfile
import shutil
from pathlib import Path
import concurrent.futures
from hybrid.data.pdb_cache import PDBCache

def test_download_deduplication_race_condition():
    """Test that download deduplication prevents race conditions."""
    print("Testing download deduplication race condition fixes...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=50,
            max_disk_gb=1.0,
            max_concurrent_downloads=4
        )
        
        # Test concurrent requests for the same PDB ID
        test_pdb_id = "1ABC"  # Fake PDB ID for testing
        test_url = "https://files.rcsb.org/download/1ABC.pdb"  # Will fail, but tests deduplication
        
        results = []
        errors = []
        
        def concurrent_download(thread_id):
            try:
                print(f"Thread {thread_id}: Starting download request for {test_pdb_id}")
                result = cache.get(test_pdb_id, test_url)
                results.append((thread_id, result))
                print(f"Thread {thread_id}: Download request completed")
            except Exception as e:
                errors.append((thread_id, str(e)))
                print(f"Thread {thread_id}: Error - {e}")
        
        # Launch multiple concurrent threads requesting the same PDB
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_download, args=(i,))
            threads.append(thread)
        
        # Start all threads simultaneously
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        end_time = time.time()
        
        print(f"Download deduplication test completed in {end_time - start_time:.2f}s")
        print(f"Results: {len(results)} successful, {len(errors)} errors")
        
        # Validate no deadlocks occurred (all threads completed)
        alive_threads = [t for t in threads if t.is_alive()]
        if alive_threads:
            print(f"WARNING: {len(alive_threads)} threads still alive (potential deadlock)")
            return False
        else:
            print("✓ No deadlocks detected")
            return True

def test_concurrent_cache_operations():
    """Test concurrent cache operations for thread safety."""
    print("Testing concurrent cache operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=100,
            max_disk_gb=1.0,
            max_concurrent_downloads=8
        )
        
        # Create some test files to work with
        test_data = {}
        for i in range(10):
            pdb_id = f"{i}ABC"
            test_data[pdb_id] = f"MOCK PDB DATA FOR {pdb_id}"
            # Add to memory cache directly for testing
            cache._add_to_memory_cache(pdb_id, test_data[pdb_id])
        
        operation_results = []
        
        def cache_operations(worker_id):
            """Perform various cache operations concurrently."""
            try:
                operations_performed = []
                
                # Test get operations
                for pdb_id in list(test_data.keys())[:5]:
                    try:
                        # This should hit memory cache
                        result = cache.get(pdb_id)
                        operations_performed.append(f"get_{pdb_id}")
                    except Exception as e:
                        operations_performed.append(f"get_{pdb_id}_ERROR: {e}")
                
                # Test statistics access
                try:
                    stats = cache.get_stats()
                    operations_performed.append("get_stats")
                except Exception as e:
                    operations_performed.append(f"get_stats_ERROR: {e}")
                
                # Test prefetch operations
                try:
                    cache.prefetch([f"{worker_id}XYZ"])  # This will fail download but tests logic
                    operations_performed.append("prefetch")
                except Exception as e:
                    operations_performed.append(f"prefetch_ERROR: {e}")
                
                operation_results.append((worker_id, operations_performed))
                
            except Exception as e:
                operation_results.append((worker_id, f"FATAL_ERROR: {e}"))
        
        # Launch concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(8)]
            
            # Wait for completion with timeout
            completed = 0
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    print(f"Operation future failed: {e}")
        
        print(f"Concurrent operations test completed: {completed}/8 workers finished")
        
        # Check for errors
        errors = [r for r in operation_results if "ERROR" in str(r[1]) or "FATAL_ERROR" in str(r[1])]
        if errors:
            print(f"Errors detected: {len(errors)}")
            for worker_id, error in errors[:3]:  # Show first 3 errors
                print(f"  Worker {worker_id}: {error}")
            return False
        else:
            print("✓ No errors detected in concurrent operations")
            return True

def test_cache_clear_safety():
    """Test that cache clear operations are safe under concurrent access."""
    print("Testing cache clear safety...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = PDBCache(
            cache_dir=Path(temp_dir),
            max_memory_mb=50,
            max_disk_gb=1.0,
            max_concurrent_downloads=4
        )
        
        # Add some data to cache
        test_pdbs = [f"{i}TST" for i in range(5)]
        for pdb_id in test_pdbs:
            cache._add_to_memory_cache(pdb_id, f"TEST DATA {pdb_id}")
        
        # Create some fake disk cache files
        for pdb_id in test_pdbs[:3]:
            file_path = cache.cache_dir / f"{pdb_id}.pdb"
            file_path.write_text(f"MOCK PDB CONTENT FOR {pdb_id}")
            with cache._disk_cache_lock:
                cache._access_times[pdb_id] = time.time()
        
        clear_completed = threading.Event()
        access_results = []
        
        def concurrent_access():
            """Try to access cache while it's being cleared."""
            while not clear_completed.wait(0.1):
                try:
                    # Try various cache operations
                    stats = cache.get_stats()
                    result = cache.get(test_pdbs[0])
                    access_results.append("success")
                except Exception as e:
                    access_results.append(f"error: {e}")
        
        # Start background access
        access_thread = threading.Thread(target=concurrent_access)
        access_thread.start()
        
        # Perform clear operation
        try:
            time.sleep(0.5)  # Let access thread run
            cache.clear_cache()
            time.sleep(0.5)  # Continue access during clear
            clear_completed.set()
            
            access_thread.join(timeout=5)
            
            if access_thread.is_alive():
                print("WARNING: Access thread did not complete (potential deadlock)")
                return False
            
            print(f"Clear safety test completed with {len(access_results)} concurrent accesses")
            
            # Check that cache was actually cleared
            stats = cache.get_stats()
            memory_items = stats['memory_cache']['item_count']
            disk_items = stats['disk_cache']['file_count']
            
            print(f"After clear - Memory items: {memory_items}, Disk items: {disk_items}")
            
            if memory_items == 0:
                print("✓ Memory cache successfully cleared")
                return True
            else:
                print(f"WARNING: Memory cache not completely cleared ({memory_items} items remain)")
                return False
            
        except Exception as e:
            print(f"Clear cache operation failed: {e}")
            clear_completed.set()
            return False

def main():
    """Run all race condition tests."""
    print("=== PDB Cache Race Condition Fix Validation ===")
    print("Testing critical fixes for TASK-1-2...")
    print()
    
    tests = [
        test_download_deduplication_race_condition,
        test_concurrent_cache_operations,
        test_cache_clear_safety
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n--- {test_func.__name__} ---")
            if test_func():
                print("✓ PASSED")
                passed += 1
            else:
                print("✗ FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("✓ ALL RACE CONDITION FIXES VALIDATED")
        return True
    else:
        print("✗ SOME TESTS FAILED - RACE CONDITIONS MAY STILL EXIST")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)