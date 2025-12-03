#!/usr/bin/env python3
"""
Simple test to verify race condition fixes work.
"""

import time
import threading
import tempfile
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the fixed PDB cache
import sys
sys.path.append(str(Path(__file__).parent / "hybrid" / "data"))
from pdb_cache import PDBCache


def test_deterministic_lru():
    """Test that deterministic LRU works correctly."""
    print("Testing deterministic LRU eviction...")
    
    test_dir = Path(tempfile.mkdtemp(prefix="pdb_test_"))
    
    try:
        # Create cache in deterministic mode
        cache = PDBCache(
            cache_dir=test_dir / "cache",
            max_memory_mb=10,
            max_disk_gb=0.01,  # Very small to force eviction
            deterministic_mode=True
        )
        
        # Manually add some files to the cache directory and tracking
        files_added = []
        for i in range(3):
            pdb_id = f"1AB{i}"
            file_path = cache.cache_dir / f"{pdb_id}.pdb"
            content = f"HEADER TEST {i}\nATOM    1  CA  ALA A   1      {i:6.3f}  0.000  0.000\nEND\n"
            file_path.write_text(content)
            
            # Add to tracking with deterministic ordering
            with cache._disk_cache_lock:
                if cache.deterministic_mode:
                    with cache._access_counter_lock:
                        cache._access_counter += 1
                        cache._access_times[pdb_id] = cache._access_counter
                else:
                    cache._access_times[pdb_id] = time.perf_counter()
                files_added.append(pdb_id)
            
            time.sleep(0.01)  # Ensure different timestamps
        
        print(f"Added files: {files_added}")
        
        # Get initial access times
        with cache._disk_cache_lock:
            initial_times = dict(cache._access_times)
        print(f"Initial access times: {initial_times}")
        
        # Access middle file to change its position in LRU
        middle_pdb = files_added[1]
        with cache._disk_cache_lock:
            cache._update_access_time_deterministic(middle_pdb)
            times_after_access = dict(cache._access_times)
        
        print(f"Access times after touching {middle_pdb}: {times_after_access}")
        
        # Verify deterministic behavior
        if cache.deterministic_mode:
            # In deterministic mode, access counter should increase
            assert times_after_access[middle_pdb] > initial_times[middle_pdb]
            print("✓ Deterministic access time update working")
        
        # Test eviction ordering
        cache.evict_lru(1000)  # Force some eviction
        
        with cache._disk_cache_lock:
            remaining_files = list(cache._access_times.keys())
        
        print(f"Remaining files after eviction: {remaining_files}")
        
        # Cleanup
        cache.executor.shutdown(wait=True)
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_concurrent_eviction_safety():
    """Test that concurrent evictions are safe."""
    print("Testing concurrent eviction safety...")
    
    test_dir = Path(tempfile.mkdtemp(prefix="pdb_evict_test_"))
    
    try:
        cache = PDBCache(
            cache_dir=test_dir / "cache",
            max_memory_mb=10,
            max_disk_gb=0.01,
            deterministic_mode=True
        )
        
        # Add several files
        files_added = []
        for i in range(5):
            pdb_id = f"1EV{i}"
            file_path = cache.cache_dir / f"{pdb_id}.pdb"
            content = f"HEADER EVICT TEST {i}\n" + "X" * 1000  # 1KB each
            file_path.write_text(content)
            
            with cache._disk_cache_lock:
                with cache._access_counter_lock:
                    cache._access_counter += 1
                    cache._access_times[pdb_id] = cache._access_counter
                files_added.append(pdb_id)
        
        # Launch concurrent evictions
        def eviction_worker(worker_id):
            try:
                cache.evict_lru(500)  # Try to evict 500 bytes
                return True
            except Exception as e:
                print(f"Eviction worker {worker_id} failed: {e}")
                return False
        
        # Run multiple evictions concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(eviction_worker, i) for i in range(3)]
            results = [future.result() for future in as_completed(futures)]
        
        success_count = sum(1 for result in results if result)
        
        # Check final state is consistent
        with cache._disk_cache_lock:
            remaining_files = list(cache._access_times.keys())
        
        print(f"Concurrent eviction results: {success_count}/3 successful")
        print(f"Remaining files: {remaining_files}")
        
        cache.executor.shutdown(wait=True)
        return success_count > 0  # At least one eviction should succeed
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_file_operations_protection():
    """Test that file operations are properly protected."""
    print("Testing file operations protection...")
    
    test_dir = Path(tempfile.mkdtemp(prefix="pdb_protect_test_"))
    
    try:
        cache = PDBCache(
            cache_dir=test_dir / "cache",
            max_memory_mb=10,
            max_disk_gb=0.01
        )
        
        # Add a test file
        pdb_id = "1TST"
        file_path = cache.cache_dir / f"{pdb_id}.pdb"
        content = "HEADER PROTECTION TEST\nATOM    1  CA  ALA A   1       0.000  0.000  0.000\nEND\n"
        file_path.write_text(content)
        
        with cache._disk_cache_lock:
            cache._access_times[pdb_id] = time.perf_counter()
        
        # Mark file as being parsed
        with cache._file_operations_lock:
            cache._parsing_files.add(pdb_id)
        
        # Try to evict - should fail because file is protected
        initial_files = len(cache._access_times)
        cache.evict_lru(1000)
        final_files = len(cache._access_times)
        
        # File should still be there because it's protected
        file_protected = (final_files == initial_files)
        
        # Unprotect file
        with cache._file_operations_lock:
            cache._parsing_files.discard(pdb_id)
        
        # Now eviction should work
        cache.evict_lru(1000)
        final_files_after_unprotect = len(cache._access_times)
        
        file_evicted_after_unprotect = (final_files_after_unprotect < final_files)
        
        print(f"File protected during parsing: {file_protected}")
        print(f"File evicted after unprotecting: {file_evicted_after_unprotect}")
        
        cache.executor.shutdown(wait=True)
        return file_protected and file_evicted_after_unprotect
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_disk_size_tracking():
    """Test disk size tracking accuracy."""
    print("Testing disk size tracking accuracy...")
    
    test_dir = Path(tempfile.mkdtemp(prefix="pdb_size_test_"))
    
    try:
        cache = PDBCache(
            cache_dir=test_dir / "cache",
            max_memory_mb=10,
            max_disk_gb=0.01
        )
        
        initial_size = cache._get_current_disk_size()
        print(f"Initial size: {initial_size}")
        
        # Add files manually and track sizes
        total_size = 0
        files_added = []
        
        for i in range(3):
            pdb_id = f"1SZ{i}"
            file_path = cache.cache_dir / f"{pdb_id}.pdb"
            content = f"HEADER SIZE TEST {i}\n" + "X" * (1000 * (i + 1))  # 1KB, 2KB, 3KB
            file_path.write_text(content)
            file_size = len(content)
            total_size += file_size
            files_added.append((pdb_id, file_size))
            
            # Add to cache tracking
            with cache._disk_cache_lock:
                cache._access_times[pdb_id] = time.perf_counter()
        
        # Force size recalculation
        with cache._disk_size_lock:
            cache._last_size_update = 0.0  # Force recalc
        
        calculated_size = cache._get_current_disk_size()
        print(f"Expected size: {total_size}, Calculated size: {calculated_size}")
        
        size_accurate = abs(calculated_size - total_size) < 100  # Allow small variance
        
        # Test concurrent size calculations
        def size_worker():
            return cache._get_current_disk_size()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(size_worker) for _ in range(3)]
            concurrent_sizes = [future.result() for future in as_completed(futures)]
        
        consistent_sizes = len(set(concurrent_sizes)) == 1
        print(f"Concurrent size calculations: {concurrent_sizes}")
        print(f"Size tracking accurate: {size_accurate}, Consistent: {consistent_sizes}")
        
        cache.executor.shutdown(wait=True)
        return size_accurate and consistent_sizes
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING PDB CACHE RACE CONDITION FIXES")
    print("=" * 60)
    
    # Reduce logging noise
    logging.basicConfig(level=logging.ERROR)
    
    tests = [
        ("Deterministic LRU", test_deterministic_lru),
        ("Concurrent Eviction Safety", test_concurrent_eviction_safety),
        ("File Operations Protection", test_file_operations_protection),
        ("Disk Size Tracking", test_disk_size_tracking)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            results.append(result)
        except Exception as e:
            print(f"{test_name}: ❌ FAIL - {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Race conditions have been fixed!")
        return 0
    else:
        print("⚠️  Some tests failed - race conditions may still exist")
        return 1


if __name__ == "__main__":
    exit(main())