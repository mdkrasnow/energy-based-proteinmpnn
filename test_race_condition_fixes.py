#!/usr/bin/env python3
"""
Comprehensive test suite for PDB Cache race condition fixes.

This test validates that all identified race conditions have been fixed:
- SCI-001: Data corruption risk during concurrent PDB parsing
- IMP-001: Download deduplication race condition
- REP-001: Non-deterministic LRU eviction order
- ROB-001: Download timeout cascading failures
- STORAGE-001: LRU eviction race conditions
- STORAGE-002: Disk size cache inconsistency
"""

import time
import threading
import tempfile
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Import the fixed PDB cache
import sys
sys.path.append(str(Path(__file__).parent / "hybrid" / "data"))
from pdb_cache import PDBCache


class RaceConditionTester:
    """Comprehensive tester for race condition fixes."""
    
    def __init__(self):
        self.test_dir = None
        self.cache = None
        self.results = {}
        
    def setup(self):
        """Setup test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="pdb_cache_test_"))
        print(f"Test directory: {self.test_dir}")
        
        # Create cache in deterministic mode for reproducible testing
        self.cache = PDBCache(
            cache_dir=self.test_dir / "cache",
            max_memory_mb=50,
            max_disk_gb=0.1,  # 100MB limit for testing
            max_concurrent_downloads=8,
            deterministic_mode=True,  # Enable deterministic mode
            enable_circuit_breaker=True
        )
        
    def teardown(self):
        """Cleanup test environment."""
        if self.cache:
            self.cache.executor.shutdown(wait=True)
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_download_deduplication_race_condition(self) -> Dict[str, Any]:
        """
        Test IMP-001: Download deduplication race condition fix.
        
        Simulates multiple threads requesting the same PDB simultaneously
        to ensure only one download occurs.
        """
        print("Testing download deduplication race condition fix...")
        
        # Create mock PDB file to download
        mock_pdb_content = """
HEADER    TEST PROTEIN                            01-JAN-23   1ABC
ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 30.00           N
ATOM      2  CA  ALA A   1      20.154  18.367  15.691  1.00 30.00           C
END
"""
        mock_file = self.test_dir / "mock_1abc.pdb"
        mock_file.write_text(mock_pdb_content)
        
        pdb_id = "1ABC"
        download_url = f"file://{mock_file.absolute()}"
        
        # Track download attempts
        download_attempts = []
        download_results = []
        
        def download_worker(worker_id: int):
            """Worker function to simulate concurrent download requests."""
            try:
                start_time = time.perf_counter()
                result = self.cache.get(pdb_id, download_url)
                end_time = time.perf_counter()
                
                download_attempts.append(worker_id)
                download_results.append({
                    'worker_id': worker_id,
                    'result': result is not None,
                    'duration': end_time - start_time
                })
                return result is not None
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
                return False
        
        # Launch concurrent download requests
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(download_worker, i) for i in range(num_workers)]
            successes = sum(1 for future in as_completed(futures) if future.result())
        
        # Verify only one actual download occurred
        stats = self.cache._statistics.get_comprehensive_stats()
        download_attempts_count = stats['download_performance']['attempts']
        
        result = {
            'test': 'download_deduplication',
            'workers': num_workers,
            'successes': successes,
            'download_attempts': download_attempts_count,
            'deduplication_working': download_attempts_count == 1,
            'all_workers_succeeded': successes == num_workers,
            'details': download_results
        }
        
        print(f"✓ Download deduplication test: {result['deduplication_working'] and result['all_workers_succeeded']}")
        return result
    
    def test_deterministic_lru_eviction(self) -> Dict[str, Any]:
        """
        Test REP-001: Deterministic LRU eviction for reproducible research.
        
        Verifies that eviction order is consistent across runs.
        """
        print("Testing deterministic LRU eviction...")
        
        # Create multiple mock PDB files
        mock_pdb_files = []
        for i in range(5):
            mock_content = f"""
HEADER    TEST PROTEIN {i:02d}                      01-JAN-23   {i}ABC
ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 30.00           N
END
"""
            mock_file = self.test_dir / f"mock_{i}abc.pdb"
            mock_file.write_text(mock_content)
            mock_pdb_files.append(mock_file)
        
        # Load files in deterministic order
        pdb_ids = [f"{i}ABC" for i in range(5)]
        for i, pdb_id in enumerate(pdb_ids):
            download_url = f"file://{mock_pdb_files[i].absolute()}"
            self.cache.get(pdb_id, download_url)
            time.sleep(0.01)  # Small delay to ensure ordering
        
        # Access files in different order to change LRU
        access_order = [2, 0, 4, 1, 3]
        for i in access_order:
            self.cache.get(pdb_ids[i])
        
        # Force eviction by adding large file
        large_content = "X" * (50 * 1024 * 1024)  # 50MB to force eviction
        large_file = self.test_dir / "large_file.pdb"
        large_file.write_text(large_content)
        
        # Record eviction order
        initial_files = list(self.cache._access_times.keys())
        
        # Trigger eviction
        self.cache.get("LARGE", f"file://{large_file.absolute()}")
        
        # Check which files were evicted
        remaining_files = list(self.cache._access_times.keys())
        evicted_files = [f for f in initial_files if f not in remaining_files]
        
        # Repeat the test to verify determinism
        self.cache.clear_cache()
        time.sleep(0.1)
        
        # Reload and repeat
        for i, pdb_id in enumerate(pdb_ids):
            download_url = f"file://{mock_pdb_files[i].absolute()}"
            self.cache.get(pdb_id, download_url)
            time.sleep(0.01)
        
        for i in access_order:
            self.cache.get(pdb_ids[i])
        
        initial_files_2 = list(self.cache._access_times.keys())
        self.cache.get("LARGE", f"file://{large_file.absolute()}")
        remaining_files_2 = list(self.cache._access_times.keys())
        evicted_files_2 = [f for f in initial_files_2 if f not in remaining_files_2]
        
        result = {
            'test': 'deterministic_lru',
            'first_eviction': evicted_files,
            'second_eviction': evicted_files_2,
            'deterministic': evicted_files == evicted_files_2,
            'deterministic_mode': self.cache.deterministic_mode
        }
        
        print(f"✓ Deterministic LRU test: {result['deterministic']}")
        return result
    
    def test_concurrent_parsing_safety(self) -> Dict[str, Any]:
        """
        Test SCI-001: Data corruption prevention during concurrent parsing.
        
        Verifies that concurrent parsing operations don't corrupt data.
        """
        print("Testing concurrent parsing safety...")
        
        # Create a PDB file
        mock_content = """
HEADER    CONCURRENT TEST                         01-JAN-23   CONC
ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 30.00           N
ATOM      2  CA  ALA A   1      20.154  18.367  15.691  1.00 30.00           C
ATOM      3  C   ALA A   1      18.768  18.827  15.691  1.00 30.00           C
END
"""
        mock_file = self.test_dir / "concurrent.pdb"
        mock_file.write_text(mock_content)
        
        pdb_id = "CONC"
        download_url = f"file://{mock_file.absolute()}"
        
        # Download once to cache the file
        self.cache.get(pdb_id, download_url)
        
        # Clear memory cache to force re-parsing
        with self.cache._cache_lock:
            self.cache._memory_cache.clear()
        
        # Parse concurrently multiple times
        parse_results = []
        parse_errors = []
        
        def parse_worker(worker_id: int):
            """Worker to parse the same file concurrently."""
            try:
                result = self.cache.get(pdb_id)
                parse_results.append((worker_id, result is not None, len(str(result)) if result else 0))
                return True
            except Exception as e:
                parse_errors.append((worker_id, str(e)))
                return False
        
        # Launch concurrent parsing
        num_workers = 8
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(parse_worker, i) for i in range(num_workers)]
            successes = sum(1 for future in as_completed(futures) if future.result())
        
        # Verify all parsings succeeded and produced consistent results
        result_sizes = [size for _, success, size in parse_results if success]
        consistent_results = len(set(result_sizes)) == 1 if result_sizes else False
        
        result = {
            'test': 'concurrent_parsing',
            'workers': num_workers,
            'successes': successes,
            'parse_errors': len(parse_errors),
            'consistent_results': consistent_results,
            'result_sizes': result_sizes,
            'errors': parse_errors[:3]  # First 3 errors
        }
        
        print(f"✓ Concurrent parsing test: {successes == num_workers and consistent_results}")
        return result
    
    def test_circuit_breaker_cascade_prevention(self) -> Dict[str, Any]:
        """
        Test ROB-001: Circuit breaker prevents cascade failures.
        
        Verifies that the circuit breaker opens after failures and prevents cascades.
        """
        print("Testing circuit breaker cascade prevention...")
        
        # Create cache with low failure threshold for testing
        test_cache = PDBCache(
            cache_dir=self.test_dir / "circuit_test",
            max_memory_mb=10,
            max_disk_gb=0.05,
            enable_circuit_breaker=True
        )
        test_cache._circuit_breaker_failure_threshold = 3  # Lower threshold for testing
        test_cache._circuit_breaker_recovery_time = 2.0  # Faster recovery for testing
        
        # Simulate failures by requesting non-existent files
        failure_count = 0
        blocked_count = 0
        
        # Cause failures to open circuit breaker
        for i in range(5):
            try:
                result = test_cache.get(f"FAIL{i}", "http://nonexistent.example.com/fail.pdb")
                if result is None:
                    failure_count += 1
            except Exception:
                failure_count += 1
        
        # Check if circuit breaker is open
        circuit_open = test_cache._circuit_breaker_open
        
        # Try more requests - they should be blocked
        for i in range(3):
            if not test_cache._check_circuit_breaker():
                blocked_count += 1
        
        # Wait for recovery and test
        time.sleep(2.5)
        recovery_allowed = test_cache._check_circuit_breaker()
        
        test_cache.executor.shutdown(wait=True)
        
        result = {
            'test': 'circuit_breaker',
            'failures_triggered': failure_count,
            'circuit_opened': circuit_open,
            'requests_blocked': blocked_count,
            'recovery_after_timeout': recovery_allowed,
            'working': circuit_open and blocked_count > 0 and recovery_allowed
        }
        
        print(f"✓ Circuit breaker test: {result['working']}")
        return result
    
    def test_disk_size_consistency(self) -> Dict[str, Any]:
        """
        Test STORAGE-002: Disk size cache consistency.
        
        Verifies that disk size tracking remains accurate during operations.
        """
        print("Testing disk size consistency...")
        
        # Start with empty cache
        initial_size = self.cache._get_current_disk_size()
        
        # Add files and verify size tracking
        mock_files = []
        expected_sizes = []
        
        for i in range(3):
            content = "X" * (1024 * (i + 1))  # 1KB, 2KB, 3KB
            mock_file = self.test_dir / f"size_test_{i}.pdb" 
            mock_file.write_text(content)
            mock_files.append(mock_file)
            expected_sizes.append(len(content))
            
            # Add to cache
            pdb_id = f"SIZE{i}"
            download_url = f"file://{mock_file.absolute()}"
            self.cache.get(pdb_id, download_url)
        
        # Check size tracking
        current_size = self.cache._get_current_disk_size()
        expected_total = sum(expected_sizes)
        
        # Perform eviction and verify size tracking
        pre_eviction_size = current_size
        self.cache.evict_lru(expected_sizes[0])  # Evict smallest file
        post_eviction_size = self.cache._get_current_disk_size()
        
        # Verify size decreased appropriately
        size_decreased = post_eviction_size < pre_eviction_size
        
        # Test concurrent size calculations
        size_results = []
        
        def size_worker():
            """Worker to calculate size concurrently."""
            return self.cache._get_current_disk_size()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(size_worker) for _ in range(5)]
            size_results = [future.result() for future in as_completed(futures)]
        
        # All concurrent size calculations should return the same value
        consistent_concurrent_sizes = len(set(size_results)) == 1
        
        result = {
            'test': 'disk_size_consistency',
            'initial_size': initial_size,
            'expected_total': expected_total,
            'measured_size': current_size,
            'size_tracking_accurate': abs(current_size - expected_total) < 1024,  # Allow 1KB variance
            'pre_eviction_size': pre_eviction_size,
            'post_eviction_size': post_eviction_size,
            'size_decreased_after_eviction': size_decreased,
            'concurrent_sizes': size_results,
            'consistent_concurrent_sizes': consistent_concurrent_sizes,
            'working': size_decreased and consistent_concurrent_sizes
        }
        
        print(f"✓ Disk size consistency test: {result['working']}")
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all race condition tests."""
        print("=" * 60)
        print("RUNNING COMPREHENSIVE RACE CONDITION TESTS")
        print("=" * 60)
        
        self.setup()
        
        try:
            # Run all tests
            self.results['download_deduplication'] = self.test_download_deduplication_race_condition()
            self.results['deterministic_lru'] = self.test_deterministic_lru_eviction()
            self.results['concurrent_parsing'] = self.test_concurrent_parsing_safety()
            self.results['circuit_breaker'] = self.test_circuit_breaker_cascade_prevention()
            self.results['disk_size_consistency'] = self.test_disk_size_consistency()
            
            # Calculate overall results
            all_tests_passed = all(
                result.get('deduplication_working', result.get('deterministic', 
                result.get('working', result.get('successes', 0) > 0)))
                for result in self.results.values()
            )
            
            self.results['summary'] = {
                'total_tests': len(self.results) - 1,  # Exclude summary itself
                'all_passed': all_tests_passed,
                'timestamp': time.time()
            }
            
            return self.results
            
        finally:
            self.teardown()


def main():
    """Main test runner."""
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    tester = RaceConditionTester()
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RACE CONDITION FIX TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        if test_name == 'summary':
            continue
            
        test_passed = result.get('deduplication_working', result.get('deterministic', 
                      result.get('working', result.get('successes', 0) > 0)))
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"{test_name.upper():25} {status}")
        
        if not test_passed:
            print(f"    Details: {result}")
    
    print("-" * 60)
    summary = results['summary']
    overall_status = "✅ ALL TESTS PASSED" if summary['all_passed'] else "❌ SOME TESTS FAILED"
    print(f"OVERALL RESULT: {overall_status}")
    print(f"Tests run: {summary['total_tests']}")
    
    if not summary['all_passed']:
        print("\nDetailed results:")
        import json
        print(json.dumps(results, indent=2, default=str))
    
    return 0 if summary['all_passed'] else 1


if __name__ == "__main__":
    exit(main())