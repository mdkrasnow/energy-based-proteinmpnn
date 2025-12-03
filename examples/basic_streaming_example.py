#!/usr/bin/env python3
"""
Basic Streaming Example for Energy-Based ProteinMPNN

This example demonstrates the basic usage of the streaming PDB system
for protein structure training with dynamic data loading.

Usage:
    python examples/basic_streaming_example.py [--config CONFIG_FILE] [--cache_dir CACHE_DIR]

Examples:
    # Basic usage with default settings
    python examples/basic_streaming_example.py

    # Custom configuration
    python examples/basic_streaming_example.py --config my_config.json --cache_dir ./my_cache

    # Development mode with small dataset
    python examples/basic_streaming_example.py --dev_mode

    # Harvard cluster deployment
    export USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
    export STREAMING_CACHE_DIR="$USER_SCRATCH/streaming_cache"
    python examples/basic_streaming_example.py --dev_mode

Environment Variables:
    Required for Harvard cluster deployment:
        USER                    - Harvard username (automatically set by SLURM)
        USER_SCRATCH            - Harvard lab scratch space: /n/netscratch/ydu_lab/Lab/$USER
        STREAMING_CACHE_DIR     - Cache directory: $USER_SCRATCH/streaming_cache
        TENSORBOARD_LOG_DIR     - TensorBoard logs: $USER_SCRATCH/logs/$(date +%Y%m%d_%H%M%S)
        CHECKPOINT_DIR          - Model checkpoints: $USER_SCRATCH/checkpoints
        SLURM_MAIL_USER        - Email notifications: your-email@harvard.edu
        PYTHONPATH             - Include project root: $PWD:$PWD/proteinmpnn:$PYTHONPATH
    
    Performance optimization (Harvard A100 cluster):
        OMP_NUM_THREADS         - CPU threads: 32 (match SLURM --cpus-per-task)
        MKL_NUM_THREADS         - Intel MKL threads: 32
        NUMBA_NUM_THREADS       - Numba JIT threads: 32
        TORCH_NUM_THREADS       - PyTorch CPU threads: 32
        CUDA_VISIBLE_DEVICES    - GPU selection: 0 (or $SLURM_LOCALID)
        PYTHONUNBUFFERED       - Real-time output: 1 (required for SLURM logs)
        CUDA_LAUNCH_BLOCKING    - GPU operation mode: 0 (async for performance)
        PYTORCH_CUDA_ALLOC_CONF - CUDA memory: "max_split_size_mb:128"
    
    Harvard cluster specific:
        SLURM_WORKING_DIR       - Working directory: $PWD
        TMPDIR                  - Temporary storage: /tmp
        SCRATCH_ROOT           - Lab scratch root: /n/netscratch/ydu_lab/Lab
        STREAMING_MAX_DISK_GB  - Cache size limit: 500 (Harvard quota)
        STREAMING_MAX_MEMORY_MB - Memory cache: 10240 (10GB for A100)
        MAX_CONCURRENT_DOWNLOADS - Network limit: 16 (cluster bandwidth)
    
    Development and debugging:
        STREAMING_DEBUG         - Enable debug logging: 0|1
        LOG_LEVEL              - Logging level: DEBUG|INFO|WARNING|ERROR
        STREAMING_FORCE_DOWNLOAD - Force cache rebuild: 0|1
        CACHE_STATS_INTERVAL   - Statistics frequency: 100 (batches)
        PROFILE_DATA_LOADING   - I/O performance profiling: 0|1
        STREAMING_VALIDATE_CACHE - Cache integrity checking: 1 (recommended)

Requirements:
    - PyTorch 1.13+
    - CUDA-capable GPU (optional but recommended)
    - Network access for PDB downloads
    - At least 2GB available disk space for cache
    - For Harvard cluster: access to /n/netscratch/ydu_lab/Lab/$USER
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add the repository root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import streaming components
from hybrid.data.streaming_dataset import StreamingProteinDataset
from hybrid.data.pdb_cache import PDBCache
from hybrid.data.pdb_manager import PDBListManager


def setup_logging():
    """Setup basic logging for the example."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Reduce noise from requests library
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def check_environment_variables():
    """Check and display relevant environment variables."""
    print("=== Environment Variables Check ===")
    
    # Required variables for Harvard cluster
    harvard_vars = {
        'USER_SCRATCH': 'Harvard lab scratch space',
        'STREAMING_CACHE_DIR': 'Streaming cache directory', 
        'SLURM_MAIL_USER': 'SLURM notification email'
    }
    
    # Performance variables
    performance_vars = {
        'OMP_NUM_THREADS': 'OpenMP thread count',
        'CUDA_VISIBLE_DEVICES': 'GPU device selection',
        'PYTHONUNBUFFERED': 'Python output buffering'
    }
    
    # Debug variables
    debug_vars = {
        'STREAMING_DEBUG': 'Streaming debug mode',
        'LOG_LEVEL': 'Logging verbosity',
        'STREAMING_FORCE_DOWNLOAD': 'Force cache re-download'
    }
    
    def check_var_group(var_group, group_name):
        print(f"\n{group_name}:")
        any_set = False
        for var, description in var_group.items():
            value = os.getenv(var)
            if value:
                print(f"  ✓ {var}: {value}")
                any_set = True
            else:
                print(f"  - {var}: (not set) - {description}")
        
        if not any_set and group_name == "Harvard Cluster Variables":
            print("  → Running in local development mode")
        return any_set
    
    harvard_configured = check_var_group(harvard_vars, "Harvard Cluster Variables")
    check_var_group(performance_vars, "Performance Variables")
    check_var_group(debug_vars, "Debug Variables")
    
    # Auto-detect cluster environment
    if os.path.exists("/n/netscratch/ydu_lab/Lab"):
        print(f"\n✓ Harvard cluster environment detected")
        if not harvard_configured:
            print("  → Consider setting USER_SCRATCH and STREAMING_CACHE_DIR")
            user = os.getenv('USER', 'your_username')
            print(f"  → Example: export USER_SCRATCH=/n/netscratch/ydu_lab/Lab/{user}")
    
    print()


def check_system_requirements():
    """Check if system meets basic requirements."""
    print("=== System Requirements Check ===")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        print("CUDA available: No (CPU-only mode)")
    
    # Check disk space
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    
    import shutil
    total, used, free = shutil.disk_usage(cache_dir)
    free_gb = free / (1024**3)
    print(f"Available disk space: {free_gb:.1f}GB")
    
    if free_gb < 2:
        print("WARNING: Less than 2GB disk space available")
    
    print()


def create_basic_config(cache_dir: Path, dev_mode: bool = False) -> Dict:
    """Create a basic configuration for streaming dataset."""
    
    if dev_mode:
        # Development configuration with smaller limits
        config = {
            "data_sources": [
                {
                    "type": "local_pdb",
                    "name": "local_dev_pdbs",
                    "data_dir": "proteinmpnn/inputs/PDB_monomers/pdbs",
                    "weight": 1.0,
                    "enabled": True
                }
            ],
            "streaming": {
                "enabled": True,
                "cache_dir": str(cache_dir),
                "max_memory_mb": 512,  # 512MB for development
                "max_disk_gb": 2.0,    # 2GB for development
                "num_workers": 2,      # Fewer workers for development
                "concurrent_downloads": 2,
                "prefetch_factor": 2
            },
            "data": {
                "batch_size": 4,       # Small batch size
                "negative_sampling_ratio": 0.5,
                "max_sequence_length": 200,  # Shorter sequences
                "min_sequence_length": 20
            }
        }
    else:
        # Production-like configuration
        config = {
            "data_sources": [
                {
                    "type": "local_pdb", 
                    "name": "local_pdbs",
                    "data_dir": "proteinmpnn/inputs/PDB_monomers/pdbs",
                    "weight": 0.3,
                    "enabled": True
                },
                {
                    "type": "remote_pdb",
                    "name": "rcsb_pdbs",
                    "base_url": "https://files.rcsb.org/download/",
                    "query_filters": {
                        "resolution_max": 3.0,
                        "sequence_length_min": 50,
                        "sequence_length_max": 500,
                        "method": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
                    },
                    "weight": 0.7,
                    "enabled": True,
                    "rate_limit_per_second": 10
                }
            ],
            "streaming": {
                "enabled": True,
                "cache_dir": str(cache_dir),
                "max_memory_mb": 2048,  # 2GB memory cache
                "max_disk_gb": 10.0,    # 10GB disk cache
                "num_workers": 4,
                "concurrent_downloads": 4,
                "prefetch_factor": 3
            },
            "data": {
                "batch_size": 8,
                "negative_sampling_ratio": 0.5,
                "max_sequence_length": 500,
                "min_sequence_length": 20
            }
        }
    
    return config


def example_1_basic_dataset_creation(config: Dict):
    """Example 1: Create and configure a basic streaming dataset."""
    print("=== Example 1: Basic Dataset Creation ===")
    
    try:
        # Extract configuration parameters
        cache_dir = Path(config["streaming"]["cache_dir"])
        data_sources = config["data_sources"]
        
        print(f"Cache directory: {cache_dir}")
        print(f"Data sources: {len(data_sources)}")
        
        # Initialize streaming dataset
        dataset = StreamingProteinDataset(
            data_sources=data_sources,
            cache_dir=cache_dir,
            batch_size=config["data"]["batch_size"],
            negative_sampling_ratio=config["data"]["negative_sampling_ratio"],
            max_sequence_length=config["data"]["max_sequence_length"],
            min_sequence_length=config["data"]["min_sequence_length"],
            num_workers=config["streaming"]["num_workers"],
            prefetch_factor=config["streaming"]["prefetch_factor"],
            seed=42  # For reproducibility
        )
        
        print("✓ Streaming dataset created successfully")
        print(f"  Configuration: {len(data_sources)} data sources")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Sequence length range: {config['data']['min_sequence_length']}-{config['data']['max_sequence_length']}")
        print()
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        raise


def example_2_iterate_samples(dataset: StreamingProteinDataset, num_samples: int = 10):
    """Example 2: Iterate through dataset samples and inspect data."""
    print(f"=== Example 2: Iterate Through {num_samples} Samples ===")
    
    try:
        sample_count = 0
        positive_count = 0
        negative_count = 0
        total_time = 0
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            start_time = time.time()
            
            # Extract sample information
            sequence = sample['sequence']
            coordinates = sample['coordinates']
            label = sample['label']
            length = sample['length']
            method = sample.get('method', 'unknown')
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Count positive vs negative samples
            if label == 1:
                positive_count += 1
                sample_type = "positive"
            else:
                negative_count += 1
                sample_type = f"negative ({method})"
            
            print(f"Sample {i+1:2d}: {sample_type:20s} | Length: {length:3d} | Time: {elapsed:.3f}s")
            
            # Validate sample structure
            assert isinstance(sequence, str), "Sequence should be string"
            assert len(sequence) == length, "Sequence length should match length field"
            assert coordinates.shape == (length, 4, 3), f"Coordinates shape should be [{length}, 4, 3]"
            assert label in [0, 1], "Label should be 0 or 1"
            
            sample_count += 1
        
        # Print summary statistics
        avg_time = total_time / sample_count if sample_count > 0 else 0
        print()
        print("Sample Statistics:")
        print(f"  Total samples: {sample_count}")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")
        print(f"  Average time per sample: {avg_time:.3f}s")
        print(f"  Samples per second: {1/avg_time:.1f}" if avg_time > 0 else "N/A")
        print()
        
    except Exception as e:
        print(f"✗ Error iterating samples: {e}")
        raise


def example_3_cache_management(config: Dict):
    """Example 3: Direct cache management and statistics."""
    print("=== Example 3: Cache Management ===")
    
    try:
        cache_dir = Path(config["streaming"]["cache_dir"])
        
        # Initialize cache manager
        cache = PDBCache(
            cache_dir=cache_dir / "pdb_cache",
            max_memory_mb=config["streaming"]["max_memory_mb"],
            max_disk_gb=config["streaming"]["max_disk_gb"],
            max_concurrent_downloads=config["streaming"]["concurrent_downloads"]
        )
        
        print("✓ Cache manager initialized")
        
        # Get initial statistics
        stats = cache.get_stats()
        print("Initial Cache Statistics:")
        print(f"  Memory usage: {stats['memory_cache']['size_mb']:.1f}MB / {stats['memory_cache']['max_size_mb']}MB")
        print(f"  Disk usage: {stats['disk_cache']['size_gb']:.2f}GB / {stats['disk_cache']['max_size_gb']}GB")
        print(f"  Cached files: {stats['disk_cache']['file_count']}")
        
        # Test cache functionality with a known PDB
        test_pdb_id = "1UBQ"  # Ubiquitin - small, common protein
        print(f"\nTesting cache with PDB ID: {test_pdb_id}")
        
        # Try to get PDB (will download if not cached)
        start_time = time.time()
        pdb_path = cache.get_pdb_path(test_pdb_id)
        elapsed = time.time() - start_time
        
        if pdb_path:
            print(f"✓ PDB retrieved: {pdb_path}")
            print(f"  Time taken: {elapsed:.2f}s")
            
            # Check if file exists
            if Path(pdb_path).exists():
                file_size = Path(pdb_path).stat().st_size
                print(f"  File size: {file_size:,} bytes")
            else:
                print("  ✗ File not found at reported path")
        else:
            print("✗ Failed to retrieve PDB")
        
        # Get updated statistics
        stats = cache.get_stats()
        print("\nUpdated Cache Statistics:")
        print(f"  Cache hit rate: {stats['detailed_stats']['hit_rate']:.1%}")
        print(f"  Total requests: {stats['detailed_stats']['total_requests']}")
        print(f"  Download attempts: {stats['detailed_stats']['download_attempts']}")
        print(f"  Download successes: {stats['detailed_stats']['download_successes']}")
        print()
        
    except Exception as e:
        print(f"✗ Error with cache management: {e}")
        import traceback
        traceback.print_exc()


def example_4_pdb_list_management():
    """Example 4: PDB list management with RCSB API."""
    print("=== Example 4: PDB List Management ===")
    
    try:
        # Test PDB list manager with small request
        with PDBListManager(cache_dir=Path("./cache")) as manager:
            print("✓ PDB List Manager initialized")
            
            # Get a small list of high-quality PDBs
            print("Requesting small list of high-quality PDBs...")
            start_time = time.time()
            
            pdb_ids = manager.get_filtered_pdb_list(
                max_resolution=2.0,     # High resolution only
                min_length=50,          # Reasonable size
                max_length=200,         # Not too large
                target_count=50,        # Small list for example
                use_cache=True          # Use cache if available
            )
            
            elapsed = time.time() - start_time
            
            if pdb_ids:
                print(f"✓ Retrieved {len(pdb_ids)} PDB IDs in {elapsed:.2f}s")
                print(f"  Examples: {pdb_ids[:5]}")
                
                # Get specialized query sets (small versions)
                print("Getting specialized query sets...")
                query_sets = {
                    "high_quality": manager.get_filtered_pdb_list(
                        max_resolution=1.5, min_length=50, max_length=200, target_count=20
                    ),
                    "small_proteins": manager.get_filtered_pdb_list(
                        max_resolution=2.5, min_length=20, max_length=100, target_count=20
                    )
                }
                
                for name, pdb_list in query_sets.items():
                    print(f"  {name}: {len(pdb_list)} structures")
                
            else:
                print("✗ No PDB IDs retrieved (possibly network issue)")
                print("  Using fallback list...")
                fallback_pdbs = ["1UBQ", "1VII", "2CRO", "1ROP", "1TEN"]
                print(f"  Fallback examples: {fallback_pdbs}")
        
        print()
        
    except Exception as e:
        print(f"✗ Error with PDB list management: {e}")
        print("  This is often due to network connectivity issues")
        print("  The system will use fallback PDB lists in production")


def example_5_negative_sampling_methods(dataset: StreamingProteinDataset):
    """Example 5: Explore different negative sampling methods."""
    print("=== Example 5: Negative Sampling Methods ===")
    
    try:
        print("Collecting samples with different negative sampling methods...")
        
        methods_seen = set()
        method_counts = {}
        sample_examples = {}
        
        # Collect samples until we see various methods
        for i, sample in enumerate(dataset):
            if i >= 100:  # Limit to avoid infinite loop
                break
                
            if sample['label'] == 0:  # Negative sample
                method = sample.get('method', 'unknown')
                methods_seen.add(method)
                method_counts[method] = method_counts.get(method, 0) + 1
                
                # Keep first example of each method
                if method not in sample_examples:
                    sample_examples[method] = {
                        'sequence_length': len(sample['sequence']),
                        'metadata': sample.get('metadata', {})
                    }
        
        print("Negative Sampling Methods Found:")
        for method, count in method_counts.items():
            example = sample_examples.get(method, {})
            seq_len = example.get('sequence_length', 'unknown')
            print(f"  {method:20s}: {count:2d} samples (example length: {seq_len})")
            
            # Print method-specific details
            metadata = example.get('metadata', {})
            if method == 'mutate_sequence':
                mutations = metadata.get('mutations', [])
                if mutations:
                    print(f"    Example mutations: {len(mutations)} positions")
            elif method == 'fragment_shuffle':
                fragments = metadata.get('fragments_details', [])
                if fragments:
                    print(f"    Example fragments: {len(fragments)} fragments shuffled")
            elif method == 'reverse_sequence':
                operations = metadata.get('reversal_operations', [])
                if operations:
                    print(f"    Example operations: {len(operations)} reversal operations")
        
        if not methods_seen:
            print("  No negative samples found in the first 100 samples")
            print("  Try increasing negative_sampling_ratio in configuration")
        
        print()
        
    except Exception as e:
        print(f"✗ Error exploring negative sampling: {e}")


def example_6_performance_monitoring(dataset: StreamingProteinDataset, config: Dict):
    """Example 6: Monitor performance and resource usage."""
    print("=== Example 6: Performance Monitoring ===")
    
    try:
        # Monitor for a short period
        monitor_samples = 20
        print(f"Monitoring performance for {monitor_samples} samples...")
        
        times = []
        memory_usage = []
        cache_hits = 0
        cache_misses = 0
        
        # Get initial cache stats
        cache_dir = Path(config["streaming"]["cache_dir"])
        cache = PDBCache(cache_dir / "pdb_cache")
        initial_stats = cache.get_stats()
        initial_hits = initial_stats['detailed_stats']['cache_hits']
        initial_misses = initial_stats['detailed_stats']['cache_misses']
        
        start_time = time.time()
        
        for i, sample in enumerate(dataset):
            if i >= monitor_samples:
                break
                
            sample_start = time.time()
            
            # Process sample (simulate some work)
            sequence = sample['sequence']
            coordinates = sample['coordinates']
            
            # Simulate some tensor operations
            if torch.cuda.is_available():
                coords_gpu = coordinates.cuda()
                # Simple computation to use GPU
                distances = torch.cdist(coords_gpu[:, 1], coords_gpu[:, 1])  # CA-CA distances
                coords_gpu = None  # Clear GPU memory
            
            sample_time = time.time() - sample_start
            times.append(sample_time)
            
            # Monitor memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_usage.append(gpu_memory)
        
        total_time = time.time() - start_time
        
        # Get final cache stats
        final_stats = cache.get_stats()
        final_hits = final_stats['detailed_stats']['cache_hits']
        final_misses = final_stats['detailed_stats']['cache_misses']
        
        cache_hits = final_hits - initial_hits
        cache_misses = final_misses - initial_misses
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        samples_per_sec = monitor_samples / total_time
        
        print("Performance Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per sample: {avg_time:.3f} ± {std_time:.3f}s")
        print(f"  Samples per second: {samples_per_sec:.1f}")
        print(f"  Cache performance:")
        print(f"    Cache hits: {cache_hits}")
        print(f"    Cache misses: {cache_misses}")
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            print(f"    Hit rate: {hit_rate:.1%}")
        
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            max_memory = np.max(memory_usage)
            print(f"  GPU memory usage:")
            print(f"    Average: {avg_memory:.2f}GB")
            print(f"    Peak: {max_memory:.2f}GB")
        
        print()
        
    except Exception as e:
        print(f"✗ Error monitoring performance: {e}")


def run_basic_example(config_file: Optional[str] = None, cache_dir: Optional[str] = None, 
                     dev_mode: bool = False):
    """Run the complete basic streaming example."""
    
    print("Energy-Based ProteinMPNN Streaming Example")
    print("=" * 50)
    print()
    
    # Setup
    setup_logging()
    check_environment_variables()
    check_system_requirements()
    
    # Determine cache directory
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = Path("./cache/streaming_example")
    
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_path}")
    
    # Load or create configuration
    if config_file and Path(config_file).exists():
        print(f"Loading configuration from: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Update cache directory
        config["streaming"]["cache_dir"] = str(cache_path)
    else:
        print("Creating basic configuration...")
        config = create_basic_config(cache_path, dev_mode=dev_mode)
    
    # Save configuration for reference
    config_path = cache_path / "example_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()
    
    try:
        # Example 1: Basic dataset creation
        dataset = example_1_basic_dataset_creation(config)
        
        # Example 2: Iterate through samples
        example_2_iterate_samples(dataset, num_samples=10)
        
        # Example 3: Cache management
        example_3_cache_management(config)
        
        # Example 4: PDB list management (may fail if no network)
        try:
            example_4_pdb_list_management()
        except Exception as e:
            print(f"PDB list management failed (likely network issue): {e}")
            print("This is normal in environments without internet access")
            print()
        
        # Example 5: Negative sampling methods
        example_5_negative_sampling_methods(dataset)
        
        # Example 6: Performance monitoring
        example_6_performance_monitoring(dataset, config)
        
        print("=" * 50)
        print("✓ All examples completed successfully!")
        print()
        print("Next Steps:")
        print("1. Review the generated configuration file")
        print("2. Explore the cache directory contents")
        print("3. Try modifying parameters and re-running")
        print("4. Check out the full user guide for advanced features")
        print(f"   Config: {config_path}")
        print(f"   Cache: {cache_path}")
        
    except Exception as e:
        print("=" * 50)
        print("✗ Example execution failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Basic streaming example for Energy-Based ProteinMPNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/basic_streaming_example.py
    python examples/basic_streaming_example.py --dev_mode
    python examples/basic_streaming_example.py --config my_config.json
    python examples/basic_streaming_example.py --cache_dir /path/to/cache
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--cache_dir", 
        type=str,
        help="Cache directory path (default: ./cache/streaming_example)"
    )
    
    parser.add_argument(
        "--dev_mode",
        action="store_true",
        help="Use development mode with smaller limits"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Set up minimal logging if quiet mode
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run the example
    return run_basic_example(
        config_file=args.config,
        cache_dir=args.cache_dir,
        dev_mode=args.dev_mode
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)