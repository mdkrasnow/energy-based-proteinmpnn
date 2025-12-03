#!/usr/bin/env python3
"""
Test Suite for Storage Consistency Fixes in Task 1.3

This test suite validates the critical fixes for storage consistency issues:
- ROB-001: LRU eviction race conditions during file selection and execution
- ROB-002: Disk size cache inconsistency leads to incorrect eviction decisions  
- ROB-003: Storage monitoring lacks recovery mechanisms for corrupted states
- ROB-004: Eviction process may delete files currently being accessed
- ROB-005: No verification that evicted files are actually deleted

Tests are designed to stress test the storage management under concurrent operations.
"""

import os
import sys
import time
import tempfile
import threading
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid.data.pdb_cache import PDBCache


class StorageConsistencyTester:
    """Comprehensive tester for storage consistency fixes."""
    
    def __init__(self, test_dir: Path):
        """Initialize tester with temporary directory."""
        self.test_dir = test_dir
        self.cache_dir = test_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create cache with small limits for testing
        self.cache = PDBCache(
            cache_dir=self.cache_dir,
            max_memory_mb=10,  # 10MB memory limit
            max_disk_gb=0.1,   # 100MB disk limit
            target_free_bytes=20_000_000,  # 20MB free target
            deterministic_mode=True  # For reproducible testing
        )
        
        # Test data
        self.test_pdbs = [f"test_{i:04d}" for i in range(100)]
        self.results = {
            "test_name": "",
            "success": False,
            "issues_found": [],
            "performance_metrics": {},
            "validation_results": {}
        }
    
    def create_test_file(self, pdb_id: str, size_mb: float = 1.0) -> Path:
        """Create a test PDB file of specified size."""
        file_path = self.cache_dir / f"{pdb_id}.pdb"
        
        # Generate some PDB-like content
        content = f"HEADER    TEST PDB {pdb_id}\n"
        content += f"TITLE     SYNTHETIC TEST STRUCTURE {pdb_id}\n"
        
        # Add padding to reach desired size
        padding_size = int(size_mb * 1024 * 1024) - len(content.encode())
        if padding_size > 0:
            # Add realistic-looking PDB content
            for i in range(padding_size // 80):
                content += f"ATOM  {i:5d}  CA  ALA A{i:4d}    {i:8.3f}{i:8.3f}{i:8.3f}  1.00 20.00           C  \n"
        
        file_path.write_text(content)
        return file_path
    
    def test_rob_001_atomic_eviction_validation(self) -> Dict[str, Any]:
        """
        Test ROB-001: Atomic candidate validation during eviction.
        
        This test verifies that eviction candidates are validated immediately before
        deletion to prevent race conditions where files become protected after selection.
        """
        self.results["test_name"] = "ROB-001 Atomic Eviction Validation"
        
        try:
            # Create multiple test files
            for pdb_id in self.test_pdbs[:20]:
                self.create_test_file(pdb_id, 2.0)  # 2MB files
                # Add to cache tracking
                self.cache._update_disk_access_time(pdb_id)
            
            # Simulate concurrent access while triggering eviction
            def concurrent_access_worker(pdb_id: str):
                """Worker that continuously accesses a file."""
                access_count = 0
                try:
                    for _ in range(10):
                        if self.cache._acquire_file_access(pdb_id, "read"):
                            access_count += 1
                            time.sleep(0.1)  # Hold access briefly
                            self.cache._release_file_access(pdb_id, "read")
                        time.sleep(0.05)
                except Exception as e:
                    return {"pdb_id": pdb_id, "error": str(e), "access_count": access_count}
                return {"pdb_id": pdb_id, "access_count": access_count}
            
            # Start concurrent access threads for some files
            access_threads = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Start access threads for first 10 files
                for pdb_id in self.test_pdbs[:10]:
                    future = executor.submit(concurrent_access_worker, pdb_id)
                    access_threads.append((pdb_id, future))
                
                # Wait a moment for access to begin
                time.sleep(0.2)
                
                # Trigger aggressive eviction while files are being accessed
                initial_disk_size = self.cache._get_current_disk_size()
                eviction_start = time.perf_counter()
                
                # Request eviction of 30MB (should evict ~15 files)
                self.cache.evict_lru(30_000_000)
                
                eviction_time = time.perf_counter() - eviction_start
                final_disk_size = self.cache._get_current_disk_size()
                
                # Wait for access threads to complete
                access_results = []
                for pdb_id, future in access_threads:
                    try:
                        result = future.result(timeout=5.0)
                        access_results.append(result)
                    except Exception as e:
                        access_results.append({"pdb_id": pdb_id, "error": str(e)})
            
            # Validate results
            files_freed = initial_disk_size - final_disk_size
            validation_results = self.cache.validate_cache_consistency()
            
            # Check for critical issues
            issues = []
            if validation_results["overall_health"] == "critical":
                issues.append("Cache consistency validation failed")
            
            # Check that protected files were not evicted
            protected_files_evicted = 0
            for result in access_results:
                pdb_id = result["pdb_id"]
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                if "access_count" in result and result["access_count"] > 0:
                    if not file_path.exists():
                        protected_files_evicted += 1
                        issues.append(f"Protected file {pdb_id} was evicted during access")
            
            self.results.update({
                "success": len(issues) == 0,
                "issues_found": issues,
                "performance_metrics": {
                    "eviction_time_ms": eviction_time * 1000,
                    "bytes_freed": files_freed,
                    "protected_files_evicted": protected_files_evicted,
                    "access_thread_results": access_results
                },
                "validation_results": validation_results
            })
            
        except Exception as e:
            self.results.update({
                "success": False,
                "issues_found": [f"Test failed with exception: {e}"],
                "performance_metrics": {},
                "validation_results": {}
            })
        
        return self.results.copy()
    
    def test_rob_002_disk_size_auto_correction(self) -> Dict[str, Any]:
        """
        Test ROB-002: Disk size auto-correction mechanisms.
        
        This test verifies that disk size inconsistencies are detected and automatically
        corrected, preventing incorrect eviction decisions.
        """
        self.results["test_name"] = "ROB-002 Disk Size Auto-Correction"
        
        try:
            # Create initial test files
            for i, pdb_id in enumerate(self.test_pdbs[:15]):
                self.create_test_file(pdb_id, 1.5)  # 1.5MB files
                self.cache._update_disk_access_time(pdb_id)
            
            # Get initial state
            initial_tracked_size = self.cache._cached_disk_size
            initial_actual_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pdb"))
            
            # Simulate external file manipulation (corruption scenario)
            external_files = []
            for i in range(3):
                # Add external files not tracked by cache
                external_file = self.create_test_file(f"external_{i}", 2.0)
                external_files.append(external_file)
                
                # Modify size of some tracked files externally  
                if i < 2:
                    tracked_file = self.cache_dir / f"{self.test_pdbs[i]}.pdb"
                    if tracked_file.exists():
                        # Append data to make file larger
                        with open(tracked_file, "a") as f:
                            f.write("# External modification\n" * 1000)
            
            # Delete some files externally to create stale tracking entries
            deleted_files = []
            for i in range(2):
                file_to_delete = self.cache_dir / f"{self.test_pdbs[10 + i]}.pdb"
                if file_to_delete.exists():
                    file_to_delete.unlink()
                    deleted_files.append(self.test_pdbs[10 + i])
            
            # Force disk size recalculation and auto-recovery
            recovery_start = time.perf_counter()
            corrected_size = self.cache._get_current_disk_size()
            recovery_time = time.perf_counter() - recovery_start
            
            # Run comprehensive validation and recovery
            validation_start = time.perf_counter() 
            recovery_report = self.cache.validate_and_recover_storage_state()
            validation_time = time.perf_counter() - validation_start
            
            # Verify recovery effectiveness
            final_actual_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pdb"))
            size_accuracy = abs(corrected_size - final_actual_size) / max(final_actual_size, 1)
            
            issues = []
            if size_accuracy > 0.01:  # More than 1% error
                issues.append(f"Disk size accuracy poor: {size_accuracy:.3%} error")
            
            if recovery_report["storage_health"] not in ["excellent", "good"]:
                issues.append(f"Storage health poor after recovery: {recovery_report['storage_health']}")
            
            if len(recovery_report["issues_detected"]) > len(external_files) + len(deleted_files):
                issues.append(f"Unexpected issues detected: {recovery_report['issues_detected']}")
            
            self.results.update({
                "success": len(issues) == 0,
                "issues_found": issues,
                "performance_metrics": {
                    "recovery_time_ms": recovery_time * 1000,
                    "validation_time_ms": validation_time * 1000,
                    "initial_tracked_size": initial_tracked_size,
                    "corrected_size": corrected_size,
                    "final_actual_size": final_actual_size,
                    "size_accuracy": size_accuracy,
                    "external_files_added": len(external_files),
                    "files_deleted": len(deleted_files)
                },
                "validation_results": recovery_report
            })
            
        except Exception as e:
            self.results.update({
                "success": False,
                "issues_found": [f"Test failed with exception: {e}"],
                "performance_metrics": {},
                "validation_results": {}
            })
        
        return self.results.copy()
    
    def test_rob_003_comprehensive_auto_recovery(self) -> Dict[str, Any]:
        """
        Test ROB-003: Comprehensive auto-recovery mechanisms.
        
        This test verifies that the system can automatically recover from various
        corrupted states without manual intervention.
        """
        self.results["test_name"] = "ROB-003 Comprehensive Auto-Recovery"
        
        try:
            # Create comprehensive test scenario
            for pdb_id in self.test_pdbs[:25]:
                self.create_test_file(pdb_id, 1.0)
                self.cache._update_disk_access_time(pdb_id)
            
            # Simulate multiple types of corruption simultaneously
            corruption_start = time.perf_counter()
            
            # 1. Corrupt tracking data (stale entries)
            stale_entries = []
            with self.cache._disk_cache_lock:
                for pdb_id in self.test_pdbs[20:25]:
                    file_path = self.cache_dir / f"{pdb_id}.pdb"
                    if file_path.exists():
                        file_path.unlink()  # Delete file but leave in tracking
                        stale_entries.append(pdb_id)
            
            # 2. Create untracked files 
            untracked_files = []
            for i in range(5):
                untracked_id = f"untracked_{i}"
                self.create_test_file(untracked_id, 1.0)
                untracked_files.append(untracked_id)
            
            # 3. Simulate disk size corruption
            original_cached_size = self.cache._cached_disk_size
            with self.cache._disk_size_lock:
                self.cache._cached_disk_size = original_cached_size * 2  # Corrupt cached size
            
            # 4. Create file access leaks
            leaked_files = []
            for pdb_id in self.test_pdbs[:3]:
                if self.cache._acquire_file_access(pdb_id, "read"):
                    leaked_files.append(pdb_id)
                    # Intentionally don't release - simulate leak
            
            corruption_time = time.perf_counter() - corruption_start
            
            # Trigger comprehensive auto-recovery
            recovery_start = time.perf_counter()
            recovery_report = self.cache.validate_and_recover_storage_state()
            
            # Also clean up stale access references
            stale_cleaned = self.cache._cleanup_stale_access_references()
            
            # Force disk size recalculation
            corrected_size = self.cache._get_current_disk_size()
            recovery_time = time.perf_counter() - recovery_start
            
            # Validate recovery completeness
            post_recovery_validation = self.cache.validate_cache_consistency()
            
            # Check recovery effectiveness
            issues = []
            
            # Should have cleaned up stale entries
            if len(recovery_report["recovery_actions"]) == 0:
                issues.append("No recovery actions taken despite multiple corruption scenarios")
            
            # Storage health should be restored
            if recovery_report["storage_health"] == "critical":
                issues.append("Storage health still critical after recovery")
            
            # Consistency should be improved
            consistency_score = recovery_report["consistency_checks"]["consistency_percentage"]
            if consistency_score < 90:
                issues.append(f"Consistency not fully restored: {consistency_score:.1f}%")
            
            # Should have corrected disk size
            actual_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pdb"))
            size_error = abs(corrected_size - actual_size) / max(actual_size, 1)
            if size_error > 0.05:  # More than 5% error
                issues.append(f"Disk size not accurately corrected: {size_error:.1%} error")
            
            self.results.update({
                "success": len(issues) == 0,
                "issues_found": issues,
                "performance_metrics": {
                    "corruption_setup_time_ms": corruption_time * 1000,
                    "recovery_time_ms": recovery_time * 1000,
                    "stale_entries_created": len(stale_entries),
                    "untracked_files_created": len(untracked_files),
                    "leaked_access_references": len(leaked_files),
                    "stale_references_cleaned": stale_cleaned,
                    "recovery_actions_taken": len(recovery_report["recovery_actions"]),
                    "final_consistency_score": consistency_score
                },
                "validation_results": {
                    "recovery_report": recovery_report,
                    "post_recovery_validation": post_recovery_validation
                }
            })
            
        except Exception as e:
            self.results.update({
                "success": False,
                "issues_found": [f"Test failed with exception: {e}"],
                "performance_metrics": {},
                "validation_results": {}
            })
        
        return self.results.copy()
    
    def test_rob_004_file_access_protection(self) -> Dict[str, Any]:
        """
        Test ROB-004: Enhanced file access protection with reference counting.
        
        This test verifies that files being accessed are properly protected from
        eviction using robust reference counting.
        """
        self.results["test_name"] = "ROB-004 File Access Protection"
        
        try:
            # Create test files
            for pdb_id in self.test_pdbs[:30]:
                self.create_test_file(pdb_id, 2.0)  # 2MB files
                self.cache._update_disk_access_time(pdb_id)
            
            # Test reference counting accuracy
            test_pdb = self.test_pdbs[0]
            
            # Multiple acquisitions should increment reference count
            access_acquired = []
            for i in range(5):
                if self.cache._acquire_file_access(test_pdb, "read"):
                    access_acquired.append(i)
            
            status = self.cache._get_file_access_status(test_pdb)
            expected_ref_count = len(access_acquired)
            
            issues = []
            if status["reference_count"] != expected_ref_count:
                issues.append(f"Reference count incorrect: expected {expected_ref_count}, got {status['reference_count']}")
            
            # File should be protected
            if not status["is_protected"]:
                issues.append("File should be protected but is not")
            
            # Release half the references
            for i in range(len(access_acquired) // 2):
                self.cache._release_file_access(test_pdb, "read")
            
            status_after_partial_release = self.cache._get_file_access_status(test_pdb)
            remaining_refs = len(access_acquired) - len(access_acquired) // 2
            
            if status_after_partial_release["reference_count"] != remaining_refs:
                issues.append(f"Reference count after partial release incorrect: expected {remaining_refs}, got {status_after_partial_release['reference_count']}")
            
            # Test protection during eviction
            protection_test_files = self.test_pdbs[1:6]
            for pdb_id in protection_test_files:
                self.cache._acquire_file_access(pdb_id, "read")
            
            # Trigger aggressive eviction
            eviction_start = time.perf_counter()
            bytes_to_evict = 40_000_000  # Should require evicting most files
            self.cache.evict_lru(bytes_to_evict)
            eviction_time = time.perf_counter() - eviction_start
            
            # Check that protected files were not evicted
            protected_files_evicted = 0
            for pdb_id in protection_test_files:
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                if not file_path.exists():
                    protected_files_evicted += 1
                    issues.append(f"Protected file {pdb_id} was evicted")
            
            # Test stale reference cleanup
            stale_test_file = self.test_pdbs[10]
            self.cache._acquire_file_access(stale_test_file, "read")
            
            # Simulate stale reference by manually setting old timestamp
            with self.cache._reference_count_lock:
                self.cache._file_access_timestamps[stale_test_file] = time.perf_counter() - 400  # 400 seconds ago
            
            stale_cleaned = self.cache._cleanup_stale_access_references()
            
            if stale_cleaned == 0:
                issues.append("Stale reference cleanup did not detect stale reference")
            
            # Clean up remaining references
            for pdb_id in protection_test_files:
                self.cache._release_file_access(pdb_id, "read")
            
            # Release remaining references for test_pdb
            for i in range(remaining_refs):
                self.cache._release_file_access(test_pdb, "read")
            
            self.results.update({
                "success": len(issues) == 0,
                "issues_found": issues,
                "performance_metrics": {
                    "eviction_time_ms": eviction_time * 1000,
                    "protected_files_evicted": protected_files_evicted,
                    "stale_references_cleaned": stale_cleaned,
                    "initial_ref_count": expected_ref_count,
                    "ref_count_after_partial_release": status_after_partial_release["reference_count"]
                },
                "validation_results": {
                    "final_access_status": self.cache._get_file_access_status(test_pdb),
                    "cache_consistency": self.cache.validate_cache_consistency()
                }
            })
            
        except Exception as e:
            self.results.update({
                "success": False,
                "issues_found": [f"Test failed with exception: {e}"],
                "performance_metrics": {},
                "validation_results": {}
            })
        
        return self.results.copy()
    
    def test_rob_005_deletion_verification(self) -> Dict[str, Any]:
        """
        Test ROB-005: Deletion verification and error handling.
        
        This test verifies that eviction operations properly verify successful deletion
        and handle deletion failures appropriately.
        """
        self.results["test_name"] = "ROB-005 Deletion Verification"
        
        try:
            # Create test files
            for pdb_id in self.test_pdbs[:20]:
                self.create_test_file(pdb_id, 1.0)
                self.cache._update_disk_access_time(pdb_id)
            
            # Create some files with permission issues to test deletion failure handling
            protected_files = []
            if os.name != 'nt':  # Unix-like systems
                for i in range(3):
                    pdb_id = self.test_pdbs[15 + i]
                    file_path = self.cache_dir / f"{pdb_id}.pdb"
                    if file_path.exists():
                        # Remove write permission to simulate deletion failure
                        file_path.chmod(0o444)  # Read-only
                        protected_files.append(pdb_id)
            
            # Record initial state
            initial_files = set(f.stem for f in self.cache_dir.glob("*.pdb"))
            initial_disk_size = self.cache._get_current_disk_size()
            
            # Trigger eviction with deletion verification
            verification_start = time.perf_counter()
            bytes_to_evict = 15_000_000  # Should evict ~15 files
            
            self.cache.evict_lru(bytes_to_evict)
            
            verification_time = time.perf_counter() - verification_start
            
            # Check final state
            final_files = set(f.stem for f in self.cache_dir.glob("*.pdb"))
            final_disk_size = self.cache._get_current_disk_size()
            
            files_deleted = initial_files - final_files
            bytes_freed = initial_disk_size - final_disk_size
            
            # Validate deletion verification worked
            issues = []
            
            # Check that claimed deletions actually occurred
            with self.cache._disk_cache_lock:
                tracked_files = set(self.cache._access_times.keys())
            
            # Files removed from tracking should actually be deleted
            untracked_files = initial_files - tracked_files
            for pdb_id in untracked_files:
                if pdb_id in final_files:
                    issues.append(f"File {pdb_id} removed from tracking but still exists on disk")
            
            # Protected files should still exist if deletion failed
            protected_still_exists = 0
            for pdb_id in protected_files:
                if pdb_id in final_files:
                    protected_still_exists += 1
                    # Restore permissions for cleanup
                    file_path = self.cache_dir / f"{pdb_id}.pdb"
                    if file_path.exists():
                        file_path.chmod(0o644)
            
            # Check consistency between disk size tracking and actual files
            actual_final_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pdb"))
            size_tracking_error = abs(final_disk_size - actual_final_size) / max(actual_final_size, 1)
            
            if size_tracking_error > 0.01:  # More than 1% error
                issues.append(f"Disk size tracking inaccurate after eviction: {size_tracking_error:.3%} error")
            
            # Verify cache consistency
            consistency_validation = self.cache.validate_cache_consistency()
            if consistency_validation["overall_health"] not in ["excellent", "good"]:
                issues.append(f"Cache consistency poor after eviction: {consistency_validation['overall_health']}")
            
            # Check that eviction actually freed significant space
            if bytes_freed < bytes_to_evict * 0.5:  # At least 50% of requested space
                issues.append(f"Eviction freed insufficient space: {bytes_freed / (1024*1024):.1f}MB of {bytes_to_evict / (1024*1024):.1f}MB requested")
            
            self.results.update({
                "success": len(issues) == 0,
                "issues_found": issues,
                "performance_metrics": {
                    "verification_time_ms": verification_time * 1000,
                    "files_deleted": len(files_deleted),
                    "bytes_freed": bytes_freed,
                    "bytes_requested": bytes_to_evict,
                    "protected_files_created": len(protected_files),
                    "protected_files_still_exist": protected_still_exists,
                    "size_tracking_accuracy": 1.0 - size_tracking_error
                },
                "validation_results": consistency_validation
            })
            
        except Exception as e:
            self.results.update({
                "success": False,
                "issues_found": [f"Test failed with exception: {e}"],
                "performance_metrics": {},
                "validation_results": {}
            })
        
        return self.results.copy()
    
    def cleanup(self):
        """Clean up test resources."""
        try:
            # Release any remaining file access references
            if hasattr(self.cache, '_file_access_counts'):
                with self.cache._reference_count_lock:
                    for pdb_id in list(self.cache._file_access_counts.keys()):
                        count = self.cache._file_access_counts.get(pdb_id, 0)
                        for _ in range(count):
                            self.cache._release_file_access(pdb_id, "read")
            
            # Clear cache
            self.cache.clear_cache()
            
            # Remove test directory
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"Cleanup warning: {e}")


def run_storage_consistency_tests():
    """Run all storage consistency tests and generate comprehensive report."""
    
    print("="*80)
    print("STORAGE CONSISTENCY FIXES VALIDATION SUITE")
    print("Task 1.3: LRU Eviction and Storage Management")
    print("="*80)
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="storage_test_"))
    
    try:
        tester = StorageConsistencyTester(test_dir)
        
        # Define test suite
        tests = [
            ("ROB-001", tester.test_rob_001_atomic_eviction_validation),
            ("ROB-002", tester.test_rob_002_disk_size_auto_correction),
            ("ROB-003", tester.test_rob_003_comprehensive_auto_recovery),
            ("ROB-004", tester.test_rob_004_file_access_protection),
            ("ROB-005", tester.test_rob_005_deletion_verification),
        ]
        
        # Run tests
        test_results = []
        total_tests = len(tests)
        passed_tests = 0
        
        for test_id, test_func in tests:
            print(f"\nRunning {test_id}...")
            start_time = time.perf_counter()
            
            try:
                result = test_func()
                test_time = time.perf_counter() - start_time
                result["test_duration_ms"] = test_time * 1000
                
                if result["success"]:
                    passed_tests += 1
                    print(f"✓ {test_id} PASSED ({test_time:.3f}s)")
                else:
                    print(f"✗ {test_id} FAILED ({test_time:.3f}s)")
                    for issue in result["issues_found"]:
                        print(f"    - {issue}")
                
                test_results.append((test_id, result))
                
            except Exception as e:
                test_time = time.perf_counter() - start_time
                print(f"✗ {test_id} ERROR ({test_time:.3f}s): {e}")
                test_results.append((test_id, {
                    "test_name": test_id,
                    "success": False,
                    "issues_found": [f"Test crashed: {e}"],
                    "test_duration_ms": test_time * 1000
                }))
        
        # Generate summary report
        print("\n" + "="*80)
        print("STORAGE CONSISTENCY TEST SUMMARY")
        print("="*80)
        print(f"Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL STORAGE CONSISTENCY FIXES VALIDATED SUCCESSFULLY!")
            print("✓ ROB-001: Atomic eviction validation implemented")
            print("✓ ROB-002: Disk size auto-correction working")
            print("✓ ROB-003: Comprehensive auto-recovery operational")
            print("✓ ROB-004: File access protection with reference counting")
            print("✓ ROB-005: Deletion verification and error handling")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            for test_id, result in test_results:
                if not result["success"]:
                    print(f"\n{test_id} Issues:")
                    for issue in result["issues_found"]:
                        print(f"  - {issue}")
        
        # Detailed performance metrics
        print("\n" + "-"*50)
        print("PERFORMANCE METRICS")
        print("-"*50)
        
        for test_id, result in test_results:
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                print(f"\n{test_id}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if "time" in metric.lower():
                            print(f"  {metric}: {value:.2f}")
                        elif "bytes" in metric.lower():
                            print(f"  {metric}: {value / (1024*1024):.2f}MB")
                        else:
                            print(f"  {metric}: {value}")
                    else:
                        print(f"  {metric}: {value}")
        
        return passed_tests == total_tests
        
    finally:
        # Cleanup
        try:
            tester.cleanup()
        except:
            pass


if __name__ == "__main__":
    success = run_storage_consistency_tests()
    sys.exit(0 if success else 1)