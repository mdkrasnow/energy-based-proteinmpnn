#!/usr/bin/env python3
"""
Quick validation test for storage consistency fixes.
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hybrid.data.pdb_cache import PDBCache

def quick_validation_test():
    """Quick test to validate core storage consistency features."""
    
    print("Running Quick Storage Consistency Validation...")
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="storage_quick_test_"))
    cache_dir = test_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    try:
        # Create cache instance
        cache = PDBCache(
            cache_dir=cache_dir,
            max_memory_mb=5,
            max_disk_gb=0.05,  # 50MB
            deterministic_mode=True
        )
        
        # Test 1: Reference counting for file access protection
        print("✓ Testing file access protection...")
        test_pdb = "test_001"
        
        # Create test file
        test_file = cache_dir / f"{test_pdb}.pdb"
        test_file.write_text("HEADER    TEST PDB\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n" * 1000)
        
        # Test reference counting
        cache._update_disk_access_time(test_pdb)
        
        if cache._acquire_file_access(test_pdb, "read"):
            status = cache._get_file_access_status(test_pdb)
            assert status["reference_count"] == 1, f"Expected ref count 1, got {status['reference_count']}"
            assert status["is_protected"], "File should be protected"
            cache._release_file_access(test_pdb, "read")
            print("  ✓ Reference counting working correctly")
        
        # Test 2: Storage validation and auto-recovery
        print("✓ Testing storage validation and auto-recovery...")
        
        # Create some test files
        for i in range(5):
            pdb_id = f"test_{i:03d}"
            file_path = cache_dir / f"{pdb_id}.pdb"
            file_path.write_text(f"HEADER TEST PDB {pdb_id}\n" + "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n" * 500)
            cache._update_disk_access_time(pdb_id)
        
        # Corrupt state by deleting a file but leaving in tracking
        corrupt_file = cache_dir / "test_002.pdb"
        if corrupt_file.exists():
            corrupt_file.unlink()
        
        # Run validation and recovery
        recovery_report = cache.validate_and_recover_storage_state()
        
        assert recovery_report["storage_health"] in ["excellent", "good", "fair"], f"Poor storage health: {recovery_report['storage_health']}"
        assert len(recovery_report["recovery_actions"]) > 0, "Expected recovery actions to be taken"
        print("  ✓ Auto-recovery mechanisms working")
        
        # Test 3: Disk size consistency validation  
        print("✓ Testing disk size consistency...")
        
        initial_size = cache._get_current_disk_size()
        actual_size = sum(f.stat().st_size for f in cache_dir.glob("*.pdb"))
        
        size_error = abs(initial_size - actual_size) / max(actual_size, 1)
        assert size_error < 0.01, f"Disk size tracking inaccurate: {size_error:.3%} error"
        print("  ✓ Disk size tracking accurate")
        
        # Test 4: Eviction with validation
        print("✓ Testing eviction with candidate validation...")
        
        # Protect a file and ensure it's not evicted
        protected_pdb = "test_000"
        cache._acquire_file_access(protected_pdb, "read")
        
        # Trigger eviction
        bytes_to_evict = 50000  # Small amount
        cache.evict_lru(bytes_to_evict)
        
        # Protected file should still exist
        protected_file = cache_dir / f"{protected_pdb}.pdb"
        assert protected_file.exists(), "Protected file was evicted during access"
        
        cache._release_file_access(protected_pdb, "read")
        print("  ✓ File protection during eviction working")
        
        # Test 5: Stale reference cleanup
        print("✓ Testing stale reference cleanup...")
        
        stale_pdb = "test_001"
        cache._acquire_file_access(stale_pdb, "read")
        
        # Manually make reference stale
        with cache._reference_count_lock:
            cache._file_access_timestamps[stale_pdb] = 0  # Very old timestamp
        
        stale_cleaned = cache._cleanup_stale_access_references()
        assert stale_cleaned > 0, "Stale reference cleanup didn't work"
        print("  ✓ Stale reference cleanup working")
        
        print("\n🎉 ALL STORAGE CONSISTENCY FIXES VALIDATED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
        except:
            pass

if __name__ == "__main__":
    success = quick_validation_test()
    sys.exit(0 if success else 1)