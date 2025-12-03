# Streaming PDB System User Guide

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Configuration Guide](#configuration-guide)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [Harvard Cluster Deployment](#harvard-cluster-deployment)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

## Overview

The Streaming PDB System enables large-scale protein structure training by dynamically downloading and caching PDB structures during training. This system was designed to handle datasets with 19,000+ protein structures while maintaining efficient memory usage and storage management.

### Key Features

- **Dynamic PDB Download**: Fetches PDB structures on-demand from RCSB
- **Intelligent Caching**: LRU-based cache with configurable size limits
- **Concurrent Processing**: Multi-threaded downloads with rate limiting
- **Memory Management**: Automatic eviction with memory pressure monitoring
- **Production Ready**: Robust error handling and recovery mechanisms
- **Harvard A100 Optimized**: Tuned for cluster deployment and 24-hour training

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streaming     │    │   PDB Cache     │    │   PDB Manager   │
│   Dataset       │────│   Manager       │────│   (RCSB API)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│   Background    │──────────────┘
                        │   Prefetch      │
                        └─────────────────┘
```

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA A100 80GB (recommended) or V100 32GB (minimum)
- **CPU**: 16+ cores for optimal I/O performance
- **RAM**: 250GB+ for large-scale training (64GB minimum)
- **Storage**: 100GB+ available space for caching

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 1.13+ with CUDA 11.8+
- **Dependencies**: See `requirements.txt`

### Network Requirements
- **Bandwidth**: 100Mbps+ for efficient PDB downloads
- **Latency**: Low latency to RCSB PDB servers
- **Firewall**: Allow HTTPS connections to `files.rcsb.org`

## Quick Start

### 1. Basic Setup

```bash
# Clone and install dependencies
git clone https://github.com/mdkrasnow/energy-based-proteinmpnn.git
cd energy-based-proteinmpnn
pip install -r requirements.txt

# Set environment variables
export STREAMING_CACHE_DIR="/path/to/cache"
export TENSORBOARD_LOG_DIR="/path/to/logs"
```

### 2. Simple Training Example

```python
from hybrid.data.streaming_dataset import StreamingProteinDataset
from hybrid.data.pdb_cache import PDBCache
from pathlib import Path

# Configure data sources
data_sources = [
    {
        "type": "remote_pdb",
        "name": "pdb_database", 
        "base_url": "https://files.rcsb.org/download/",
        "query_filters": {
            "resolution_max": 2.5,
            "sequence_length_min": 50,
            "sequence_length_max": 500
        }
    }
]

# Initialize streaming dataset
dataset = StreamingProteinDataset(
    data_sources=data_sources,
    cache_dir=Path("./cache"),
    batch_size=16,
    negative_sampling_ratio=0.5,
    max_sequence_length=500
)

# Use in DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=None)  # batch_size handled by dataset

# Start training
for batch in loader:
    # Each batch contains positive and negative protein samples
    sequence = batch['sequence']
    coordinates = batch['coordinates'] 
    labels = batch['label']
    # ... training logic
```

### 3. Harvard Cluster Quick Start

```bash
# Submit production job
sbatch train_hybrid_streaming.sh

# Monitor progress
squeue -u $USER
tail -f train_streaming_hybrid_*.out
```

## Configuration Guide

### Main Configuration File

The streaming system is configured through `hybrid/training/config_streaming.json`:

```json
{
    "streaming": {
        "enabled": true,
        "cache_dir": "${STREAMING_CACHE_DIR:-./cache/streaming}",
        "max_memory_mb": 5120,
        "max_disk_gb": 100,
        "prefetch_factor": 6,
        "num_workers": 8,
        "concurrent_downloads": 8
    },
    "data_sources": [
        {
            "type": "remote_pdb",
            "name": "pdb_database",
            "query_filters": {
                "resolution_max": 2.5,
                "sequence_length_min": 50,
                "sequence_length_max": 500
            },
            "weight": 1.0
        }
    ]
}
```

### Key Configuration Parameters

#### Streaming Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `max_memory_mb` | Memory cache size in MB | 5120 (5GB) |
| `max_disk_gb` | Disk cache size in GB | 100 |
| `num_workers` | Worker threads for I/O | 8 |
| `concurrent_downloads` | Simultaneous downloads | 8 |
| `prefetch_factor` | Batches to prefetch | 6 |

#### Data Source Configuration

```json
{
    "type": "remote_pdb",           // "local_pdb", "remote_pdb", "pdb_list"
    "name": "descriptive_name",
    "base_url": "https://files.rcsb.org/download/",
    "query_filters": {
        "resolution_max": 2.5,      // Maximum resolution in Å
        "sequence_length_min": 50,  // Minimum protein length
        "sequence_length_max": 500, // Maximum protein length
        "method": ["X-RAY DIFFRACTION"]
    },
    "weight": 1.0,                 // Sampling weight
    "enabled": true,
    "rate_limit_per_second": 20    // API rate limiting
}
```

### Environment Variables

#### Required Environment Variables

These variables must be set for production deployment:

```bash
# Required: User identification and paths
export USER="your_username"                          # Harvard username
export SLURM_MAIL_USER="your-email@harvard.edu"     # For job notifications

# Required: Storage paths (Harvard cluster specific)
export USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
export STREAMING_CACHE_DIR="$USER_SCRATCH/streaming_cache"
export TENSORBOARD_LOG_DIR="$USER_SCRATCH/logs"
export CHECKPOINT_DIR="$USER_SCRATCH/checkpoints"

# Required: Python and compute settings
export PYTHONPATH="$PWD:$PWD/proteinmpnn:$PYTHONPATH"
export PYTHONUNBUFFERED=1                           # Real-time output
export CUDA_VISIBLE_DEVICES=0                       # GPU assignment
```

#### Performance Optimization Variables

```bash
# CPU threading (match SLURM allocation)
export OMP_NUM_THREADS=32                           # OpenMP threads
export MKL_NUM_THREADS=32                           # Intel MKL threads  
export NUMBA_NUM_THREADS=32                         # Numba JIT threads
export TORCH_NUM_THREADS=32                         # PyTorch CPU threads

# GPU optimization
export CUDA_LAUNCH_BLOCKING=0                       # Async GPU operations
export CUDA_DEVICE_ORDER="PCI_BUS_ID"              # Consistent GPU ordering
export NCCL_DEBUG=INFO                             # Multi-GPU debugging

# Memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # CUDA memory management
export MALLOC_TRIM_THRESHOLD_=100000               # Reduce memory fragmentation
```

#### Optional Configuration Variables

```bash
# Debugging and monitoring
export STREAMING_DEBUG=1                           # Enable debug logging
export CACHE_STATS_INTERVAL=100                    # Cache statistics frequency
export PROFILE_DATA_LOADING=1                      # Profile I/O performance
export LOG_LEVEL="INFO"                            # Logging verbosity

# Network and I/O tuning  
export STREAMING_DOWNLOAD_TIMEOUT=600              # PDB download timeout (seconds)
export STREAMING_MAX_RETRIES=5                     # Download retry attempts
export STREAMING_RATE_LIMIT=10                     # Downloads per second
export RCSB_MIRROR_URL="https://files.rcsb.org"    # Alternative RCSB mirror

# Development overrides
export STREAMING_FORCE_DOWNLOAD=0                  # Force re-download cached files
export STREAMING_VALIDATE_CACHE=1                  # Validate cache integrity
export STREAMING_PREFETCH_DISABLED=0               # Disable background prefetch
```

#### Harvard Cluster Specific Variables

```bash
# Cluster integration and paths
export SLURM_WORKING_DIR="$PWD"                    # Working directory
export TMPDIR="/tmp"                               # Temporary directory  
export SCRATCH_ROOT="/n/netscratch/ydu_lab/Lab"    # Lab scratch space
export LAB_SHARED_DIR="/n/holylfs05/LABS/ydu_lab"  # Shared lab storage

# Module environment (set automatically by SLURM)
export PYTHON_VERSION="3.10.9"                    # Python module version
export CUDA_VERSION="12.2.0"                      # CUDA module version
export GCC_VERSION="12.2.0"                       # GCC module version
export SLURM_MODULE_SETUP_COMPLETE=1              # Module setup validation

# Resource limits and quotas (Harvard specific)
export STREAMING_MAX_DISK_GB=500                   # Maximum cache size (quota limit)
export STREAMING_MAX_MEMORY_MB=10240               # Maximum memory cache (10GB for A100)
export MAX_CONCURRENT_DOWNLOADS=16                 # Network bandwidth limits
export HARVARD_CLUSTER_MAX_JOBS=10                # Maximum concurrent jobs per user

# Job management and monitoring
export SLURM_SIGNAL_DELAY=300                     # Seconds before SIGKILL (5 minutes)
export CHECKPOINT_INTERVAL=3600                    # Checkpoint frequency (1 hour)
export BACKUP_INTERVAL=7200                        # Backup frequency (2 hours)
export LOG_RETENTION_DAYS=30                      # Log file retention period

# Harvard-specific network and security
export HTTP_PROXY=""                               # No proxy required
export HTTPS_PROXY=""                             # No proxy required
export NO_PROXY="localhost,127.0.0.1,*.harvard.edu,*.rc.fas.harvard.edu"
export HARVARD_FIREWALL_BYPASS=1                  # Cluster firewall configuration

# Production deployment metadata
export DEPLOYMENT_ENVIRONMENT="harvard_cluster"    # Environment identifier
export CLUSTER_NAME="cannon"                       # Harvard cluster name
export PARTITION_TYPE="gpu_requeue"               # Default partition
export NODE_TYPE="a100"                          # GPU node type
export LAB_BILLING_ACCOUNT="ydu_lab"             # Billing account

# Monitoring and alerting
export ENABLE_CLUSTER_MONITORING=1                # Enable cluster-specific monitoring
export ALERT_THRESHOLD_MEMORY_GB=350             # Memory usage alert threshold
export ALERT_THRESHOLD_DISK_GB=450               # Disk usage alert threshold
export PERFORMANCE_LOGGING_INTERVAL=300          # Performance metrics interval (seconds)

# Disaster recovery and backup
export BACKUP_ENABLED=1                          # Enable automatic backups
export BACKUP_DESTINATION="$LAB_SHARED_DIR/backups/$USER"  # Backup location
export RECOVERY_MODE=0                           # Recovery mode flag (set to 1 for recovery)
export CHECKPOINT_VALIDATION=1                   # Validate checkpoints on save

# Development vs production flags
export PRODUCTION_DEPLOYMENT=1                   # Production mode flag
export ENABLE_PROFILING=0                       # Disable profiling in production
export DEBUG_MODE=0                             # Disable debug mode in production
export STRICT_VALIDATION=1                      # Enable strict validation
```

#### Environment Setup Script

Create a setup script for consistent environment configuration:

```bash
#!/bin/bash
# File: setup_environment.sh

# Harvard cluster setup
setup_harvard_cluster() {
    # Load required modules
    module purge
    module load python/3.10.9-fasrc01
    module load cuda/12.2.0-fasrc01  
    module load gcc/12.2.0-fasrc01
    
    echo "✓ Modules loaded"
    module list
}

# Set all required variables
setup_environment_variables() {
    # Validate user is set
    if [[ -z "$USER" ]]; then
        echo "ERROR: USER environment variable not set"
        exit 1
    fi
    
    # Core paths
    export USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
    export STREAMING_CACHE_DIR="$USER_SCRATCH/streaming_cache"
    export TENSORBOARD_LOG_DIR="$USER_SCRATCH/logs/$(date +%Y%m%d_%H%M%S)"
    export CHECKPOINT_DIR="$USER_SCRATCH/checkpoints"
    
    # Create directories
    mkdir -p "$STREAMING_CACHE_DIR" "$TENSORBOARD_LOG_DIR" "$CHECKPOINT_DIR"
    
    # Python configuration
    export PYTHONPATH="$PWD:$PWD/proteinmpnn:$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    
    # Performance tuning based on allocated resources
    local cores=${SLURM_CPUS_PER_TASK:-32}
    export OMP_NUM_THREADS=$cores
    export MKL_NUM_THREADS=$cores
    export NUMBA_NUM_THREADS=$cores
    export TORCH_NUM_THREADS=$cores
    
    # GPU settings
    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
    export CUDA_LAUNCH_BLOCKING=0
    
    echo "✓ Environment variables set"
    echo "  Cache: $STREAMING_CACHE_DIR"
    echo "  Logs: $TENSORBOARD_LOG_DIR"
    echo "  CPU threads: $cores"
}

# Validate environment
validate_environment() {
    echo "=== Environment Validation ==="
    
    # Check required commands
    command -v python >/dev/null || { echo "ERROR: python not found"; exit 1; }
    command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
    
    # Check Python packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
    python -c "import hybrid.data.streaming_dataset" || exit 1
    
    # Check CUDA
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Check disk space
    echo "Disk space check:"
    df -h "$USER_SCRATCH"
    
    echo "✓ Environment validation complete"
}

# Main execution
main() {
    echo "Setting up Harvard cluster environment for streaming training..."
    
    setup_harvard_cluster
    setup_environment_variables
    validate_environment
    
    echo "Environment setup complete. Ready for training."
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

#### Usage Examples

**Production Job:**
```bash
# In your SLURM script
source setup_environment.sh
python hybrid/training/train_energy.py --config config_streaming_production.json
```

**Interactive Development:**
```bash
# Start interactive session
salloc -p gpu_test --constraint=a100 --gres=gpu:1 -c 16 -t 4:00:00 --mem=64G

# Setup environment
source setup_environment.sh
export STREAMING_DEBUG=1

# Run development training
python hybrid/training/train_energy.py --config config_streaming_dev.json
```

**Environment Verification:**
```bash
# Quick environment check
python -c "
import os
required_vars = ['STREAMING_CACHE_DIR', 'USER_SCRATCH', 'SLURM_MAIL_USER']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'Missing required variables: {missing}')
    exit(1)
print('✓ All required environment variables set')
"
```

## Usage Examples

### Example 1: Local Development

```python
# Development setup with small dataset
from hybrid.data.streaming_dataset import StreamingProteinDataset

dataset = StreamingProteinDataset(
    data_sources=[{
        "type": "local_pdb",
        "data_dir": "proteinmpnn/inputs/PDB_monomers/pdbs"
    }],
    cache_dir=Path("./dev_cache"),
    batch_size=4,  # Smaller for development
    max_sequence_length=200
)
```

### Example 2: Production Training

```python
# Production setup with multiple data sources
data_sources = [
    {
        "type": "local_pdb",
        "data_dir": "proteinmpnn/inputs/PDB_monomers/pdbs", 
        "weight": 0.3
    },
    {
        "type": "remote_pdb",
        "query_filters": {
            "resolution_max": 2.5,
            "sequence_length_min": 50,
            "sequence_length_max": 500
        },
        "weight": 0.7
    }
]

dataset = StreamingProteinDataset(
    data_sources=data_sources,
    cache_dir=Path("/n/netscratch/ydu_lab/Lab/mkrasnow/streaming_cache"),
    batch_size=16,
    prefetch_factor=6,
    num_workers=8,
    negative_sampling_ratio=0.5
)
```

### Example 3: Custom Negative Sampling

```python
# Advanced negative sampling configuration
dataset = StreamingProteinDataset(
    data_sources=data_sources,
    cache_dir=cache_dir,
    negative_sampling_ratio=0.5,
    augmentation_config={
        "coordinate_noise": {"std": 0.01},
        "sequence_masking": {"probability": 0.05}
    }
)

# Iterate through different negative sampling methods
for sample in dataset:
    method = sample.get('method', 'unknown')
    if method == 'mutate_sequence':
        mutations = sample['metadata']['mutations']
        print(f"Mutated {len(mutations)} positions")
    elif method == 'fragment_shuffle':
        fragments = sample['metadata']['fragments_details']
        print(f"Shuffled {len(fragments)} fragments")
```

### Example 4: PDB List Management

```python
from hybrid.data.pdb_manager import PDBListManager

# Get filtered PDB list for training
with PDBListManager(cache_dir=Path("./cache")) as manager:
    pdb_ids = manager.get_filtered_pdb_list(
        max_resolution=2.5,
        min_length=50,
        max_length=500,
        target_count=5000,
        use_cache=True
    )
    print(f"Retrieved {len(pdb_ids)} PDB structures")
    
    # Get specialized query sets
    query_sets = manager.get_biological_query_sets()
    high_quality = query_sets['high_quality']
    print(f"High-quality set: {len(high_quality)} structures")
```

### Example 5: Cache Management

```python
from hybrid.data.pdb_cache import PDBCache

# Initialize cache with custom settings
cache = PDBCache(
    cache_dir=Path("./cache"),
    max_memory_mb=2048,    # 2GB memory cache
    max_disk_gb=50,        # 50GB disk cache
    max_concurrent_downloads=16
)

# Check cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['detailed_stats']['hit_rate']:.2%}")
print(f"Disk usage: {stats['disk_cache']['size_gb']:.1f}GB")

# Manual cache management
cache.ensure_cache_space(bytes_needed=1_000_000_000)  # Ensure 1GB free
cache.clear_cache()  # Clear all cached data
```

## Performance Optimization

### A100 Optimized Settings

For Harvard A100 cluster deployment:

```json
{
    "training": {
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "num_workers": 8,
        "mixed_precision": true
    },
    "streaming": {
        "max_memory_mb": 5120,
        "max_disk_gb": 100, 
        "num_workers": 8,
        "concurrent_downloads": 8,
        "prefetch_factor": 6,
        "connection_pool_size": 32
    },
    "hardware_optimization": {
        "gpu_type": "a100_80gb",
        "memory_fraction": 0.9,
        "mixed_precision": "fp16",
        "tensor_cores": true,
        "pin_memory": true
    }
}
```

### Memory Tuning

Monitor and tune memory usage:

```bash
# Check memory usage during training
nvidia-smi -l 1

# Monitor cache statistics
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('cache_dir')
stats = cache.get_stats()
print(f'Memory: {stats[\"memory_cache\"][\"utilization_percent\"]:.1f}%')
print(f'Disk: {stats[\"disk_cache\"][\"utilization_percent\"]:.1f}%')
"
```

### I/O Optimization

Optimize for high-throughput I/O:

- **Concurrent Downloads**: Set to 8-16 for A100 systems
- **Prefetch Factor**: Use 4-6 for large memory systems
- **Worker Threads**: Match CPU core count (typically 8-16)
- **Connection Pooling**: Use 32+ for high-concurrency scenarios

## Harvard Cluster Deployment

### Cluster Environment Setup

```bash
# Load required modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

# Set cluster-specific paths
export SLURM_MAIL_USER="your-email@harvard.edu"
export USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
export STREAMING_CACHE_DIR="$USER_SCRATCH/streaming_cache"
export TENSORBOARD_LOG_DIR="$USER_SCRATCH/logs"
```

### SLURM Job Configuration

#### Production Job Template with Essential Parameters

```bash
#!/bin/bash
# Essential SLURM directives for Harvard cluster production deployment
#SBATCH -J train_streaming_hybrid                    # Job name
#SBATCH -p gpu_requeue                               # Partition (gpu_requeue recommended for production)
#SBATCH --constraint=a100                            # GPU constraint (A100 required)
#SBATCH --gres=gpu:a100:1                           # GPU resource allocation
#SBATCH -c 32                                        # CPU cores per task
#SBATCH -t 48:00:00                                  # Time limit (48 hours for production)
#SBATCH --mem=400G                                   # Memory allocation (400GB for A100 production)

# Job management and monitoring
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,REQUEUE
#SBATCH --mail-user=${SLURM_MAIL_USER}              # Email notifications
#SBATCH --requeue                                    # Auto-requeue on node failure
#SBATCH --signal=SIGUSR1@600                        # Graceful shutdown signal (10 min warning)
#SBATCH --job-name=stream-hybrid-${USER}            # User-specific job naming

# Output and error handling
#SBATCH --output=%j_train_streaming_hybrid_%A.out   # Standard output file
#SBATCH --error=%j_train_streaming_hybrid_%A.err    # Standard error file
#SBATCH --open-mode=append                          # Append to existing log files
#SBATCH --export=ALL                                # Export all environment variables

# Resource optimization for A100 production
#SBATCH --exclusive                                 # Exclusive node access (recommended)
#SBATCH --cpus-per-task=32                         # CPU cores (matches -c for clarity)
#SBATCH --ntasks-per-node=1                        # Single task per node
#SBATCH --nodes=1                                  # Single node allocation
#SBATCH --threads-per-core=1                       # Disable hyperthreading for performance

# Advanced resource management
#SBATCH --nice=0                                   # Normal priority
#SBATCH --qos=normal                              # Quality of service
#SBATCH --account=ydu_lab                         # Billing account (lab-specific)

# Performance and reliability optimizations
#SBATCH --mem-bind=verbose,local                  # Memory binding for NUMA optimization
#SBATCH --cpu-bind=verbose,cores                  # CPU binding for performance
#SBATCH --gpu-bind=verbose,closest_draining       # GPU binding optimization

# Production monitoring and checkpointing
#SBATCH --checkpoint=60                           # Checkpoint interval (minutes)
#SBATCH --checkpoint-dir=${CHECKPOINT_DIR}        # Checkpoint directory
#SBATCH --restart                                 # Enable restart from checkpoint

# Essential module loading with error checking
echo "Loading required modules..."
module purge  # Clear any conflicting modules
if ! module load python/3.10.9-fasrc01; then
    echo "FATAL: Failed to load Python module" >&2
    exit 1
fi
if ! module load cuda/12.2.0-fasrc01; then
    echo "FATAL: Failed to load CUDA module" >&2
    exit 1
fi
if ! module load gcc/12.2.0-fasrc01; then
    echo "FATAL: Failed to load GCC module" >&2
    exit 1
fi
echo "✓ All required modules loaded successfully"
module list

# Comprehensive environment verification and setup
echo "=== Production Environment Verification ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "User: $USER"

# GPU verification
echo
echo "GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "  Used: %s MB / %s MB (%.1f%%)\n", $1, $2, $1/$2*100}'
else
    echo "ERROR: nvidia-smi not available" >&2
    exit 1
fi
echo "================================================"

# Production environment variables with comprehensive validation
echo "Setting up production environment variables..."

# Critical path validation
export USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
if [[ ! -d "/n/netscratch/ydu_lab/Lab" ]]; then
    echo "FATAL: Lab scratch space not accessible" >&2
    exit 1
fi

export STREAMING_CACHE_DIR="$USER_SCRATCH/streaming_cache"
export TENSORBOARD_LOG_DIR="$USER_SCRATCH/logs/run_$(date +%Y%m%d_%H%M%S)"
export CHECKPOINT_DIR="$USER_SCRATCH/checkpoints"
export BACKUP_DIR="$USER_SCRATCH/backups/job_$SLURM_JOB_ID"

# Python environment
export PYTHONPATH="$PWD:$PWD/proteinmpnn:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Performance tuning (match allocated resources)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GPU optimization
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Harvard cluster specific settings
export DEPLOYMENT_ENVIRONMENT="harvard_cluster"
export CLUSTER_NAME="cannon"
export PRODUCTION_DEPLOYMENT=1
export STRICT_VALIDATION=1
export ENABLE_CLUSTER_MONITORING=1

# Create all required directories with verification
echo "Creating production directory structure..."
REQUIRED_DIRS=(
    "$STREAMING_CACHE_DIR"
    "$TENSORBOARD_LOG_DIR" 
    "$CHECKPOINT_DIR"
    "$BACKUP_DIR"
    "$USER_SCRATCH/logs"
    "$USER_SCRATCH/tmp"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
        echo "✓ Created: $dir"
        # Test write permission
        test_file="$dir/.write_test_$$"
        if touch "$test_file" 2>/dev/null; then
            rm -f "$test_file"
        else
            echo "ERROR: No write permission for $dir" >&2
            exit 1
        fi
    else
        echo "ERROR: Failed to create $dir" >&2
        exit 1
    fi
done

# Verify available disk space
echo
echo "Disk space verification:"
available_space=$(df -BG "$USER_SCRATCH" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space in user scratch: ${available_space}GB"
if [[ $available_space -lt 50 ]]; then
    echo "WARNING: Less than 50GB available space" >&2
    echo "Consider cleaning up old files or requesting quota increase" >&2
fi

# Production signal handling for graceful shutdown
production_cleanup() {
    echo
    echo "=========================================="
    echo "Graceful shutdown initiated at $(date)"
    echo "Signal received: $1"
    echo "=========================================="
    
    # Set cleanup flag
    CLEANUP_IN_PROGRESS=1
    
    # Save current state immediately
    if [[ -n $TRAINING_PID ]] && kill -0 $TRAINING_PID 2>/dev/null; then
        echo "Sending graceful shutdown signal to training process $TRAINING_PID"
        
        # Send SIGUSR1 first for graceful checkpoint saving
        kill -USR1 $TRAINING_PID 2>/dev/null
        echo "Waiting 300 seconds for graceful checkpoint save..."
        
        # Wait with timeout for graceful shutdown
        timeout=300
        while [[ $timeout -gt 0 ]] && kill -0 $TRAINING_PID 2>/dev/null; do
            sleep 5
            timeout=$((timeout - 5))
            echo "Waiting... ${timeout}s remaining"
        done
        
        # Force kill if still running
        if kill -0 $TRAINING_PID 2>/dev/null; then
            echo "Force terminating training process"
            kill -TERM $TRAINING_PID 2>/dev/null
            sleep 10
            kill -KILL $TRAINING_PID 2>/dev/null
        fi
    fi
    
    # Backup critical data
    echo "Creating emergency backup..."
    backup_emergency_data
    
    # Final status report
    production_completion_report
    
    echo "Graceful shutdown complete at $(date)"
    exit 130  # Signal termination exit code
}

# Emergency backup function
backup_emergency_data() {
    if [[ -d "$CHECKPOINT_DIR" ]]; then
        emergency_backup="$BACKUP_DIR/emergency_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$emergency_backup"
        
        echo "Backing up to: $emergency_backup"
        
        # Copy latest checkpoint
        latest_checkpoint=$(ls -1t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)
        if [[ -n "$latest_checkpoint" ]]; then
            cp "$latest_checkpoint" "$emergency_backup/latest_checkpoint.pt"
            echo "✓ Checkpoint saved"
        fi
        
        # Copy configuration
        cp config_streaming_production.json "$emergency_backup/" 2>/dev/null
        
        # Save environment info
        env | grep -E "(SLURM|STREAMING|CUDA)" > "$emergency_backup/environment.txt"
        
        echo "✓ Emergency backup complete"
    fi
}

# Set up signal handlers
trap 'production_cleanup SIGUSR1' SIGUSR1
trap 'production_cleanup SIGTERM' SIGTERM
trap 'production_cleanup SIGINT' SIGINT

# Validate Python environment and configuration
echo
echo "Validating production environment..."
python -c "
import sys, torch
sys.path.insert(0, '$PWD')

# Test imports
try:
    from hybrid.data.streaming_dataset import StreamingProteinDataset
    from hybrid.training.train_energy import main
    print('✓ All imports successful')
except ImportError as e:
    print(f'FATAL: Import error: {e}')
    sys.exit(1)

# Test CUDA
if not torch.cuda.is_available():
    print('FATAL: CUDA not available')
    sys.exit(1)
print(f'✓ CUDA available: {torch.cuda.device_count()} GPUs')

# Test configuration
import json, os
config_path = 'config_streaming_production.json'
if not os.path.exists(config_path):
    print(f'FATAL: Configuration file missing: {config_path}')
    sys.exit(1)
    
with open(config_path) as f:
    config = json.load(f)
    print(f'✓ Configuration loaded: {len(config.get(\"data_sources\", []))} data sources')
"

if [[ $? -ne 0 ]]; then
    echo "FATAL: Environment validation failed" >&2
    exit 1
fi

# Start production training with comprehensive monitoring
echo
echo "=========================================="
echo "Starting Production Training"
echo "Time: $(date)"
echo "Configuration: config_streaming_production.json"
echo "Cache: $STREAMING_CACHE_DIR"
echo "Logs: $TENSORBOARD_LOG_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "=========================================="

# Pre-training cache warm-up (if enabled)
if [[ "${ENABLE_CACHE_WARMUP:-0}" == "1" ]]; then
    echo "Performing cache warm-up..."
    python -c "
from hybrid.data.pdb_cache import PDBCache
from pathlib import Path
cache = PDBCache(Path('$STREAMING_CACHE_DIR'))
stats = cache.get_stats()
print(f'Cache ready: {stats[\"disk_cache\"][\"file_count\"]} files cached')
"
fi

# Start training with monitoring
python hybrid/training/train_energy.py \
    --config config_streaming_production.json \
    --streaming_mode \
    --device cuda \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --tensorboard_log_dir "$TENSORBOARD_LOG_DIR" \
    --production_mode \
    --enable_monitoring &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"

# Background monitoring
{
    sleep 60  # Initial delay
    while kill -0 $TRAINING_PID 2>/dev/null; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A,N/A")
        cache_size=$(du -sh "$STREAMING_CACHE_DIR" 2>/dev/null | cut -f1 || echo "N/A")
        
        echo "[$timestamp] GPU: ${gpu_util}% | Memory: $gpu_mem | Cache: $cache_size"
        sleep 300  # Monitor every 5 minutes
    done
} &
MONITOR_PID=$!

# Wait for training completion
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Kill monitoring process
kill $MONITOR_PID 2>/dev/null

echo "Training completed with exit code: $TRAINING_EXIT_CODE at $(date)"

# Production completion report
production_completion_report() {
    echo
    echo "=========================================="
    echo "Production Job Completion Report"
    echo "=========================================="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Completion time: $(date)"
    echo "Total runtime: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo
    
    echo "Resource Usage:"
    echo "  Final GPU memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "    %s MB / %s MB (%.1f%%)\n", $1, $2, $1/$2*100}'
    
    echo "  Cache statistics:"
    cache_size=$(du -sh "$STREAMING_CACHE_DIR" 2>/dev/null | cut -f1 || echo "N/A")
    echo "    Cache size: $cache_size"
    
    echo "  Checkpoint status:"
    if [[ -d "$CHECKPOINT_DIR" ]]; then
        checkpoint_count=$(ls -1 "$CHECKPOINT_DIR"/*.pt 2>/dev/null | wc -l)
        if [[ $checkpoint_count -gt 0 ]]; then
            latest_checkpoint=$(ls -1t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)
            checkpoint_size=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1 || echo "N/A")
            echo "    Checkpoints saved: $checkpoint_count"
            echo "    Total size: $checkpoint_size"
            echo "    Latest: $(basename "$latest_checkpoint")"
        else
            echo "    No checkpoints found"
        fi
    else
        echo "    Checkpoint directory missing"
    fi
    
    echo "  Log files:"
    log_size=$(du -sh "$TENSORBOARD_LOG_DIR" 2>/dev/null | cut -f1 || echo "N/A")
    echo "    Log directory size: $log_size"
    
    # Performance summary
    echo
    echo "Performance Summary:"
    if [[ -f "${TENSORBOARD_LOG_DIR}/performance_metrics.txt" ]]; then
        tail -5 "${TENSORBOARD_LOG_DIR}/performance_metrics.txt"
    else
        echo "  Performance metrics not available"
    fi
    
    echo "=========================================="
}

# Call completion report
production_completion_report

# Exit with training process exit code
exit $TRAINING_EXIT_CODE
```

#### Development Job Template

```bash
#!/bin/bash
#SBATCH -J dev_streaming_hybrid
#SBATCH -p gpu_test
#SBATCH --constraint=a100
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=FAIL
#SBATCH --output=dev_%j_%A.out
#SBATCH --error=dev_%j_%A.err

# Load modules
module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01

# Set development environment
export STREAMING_CACHE_DIR="$PWD/cache/dev_cache"
export TENSORBOARD_LOG_DIR="$PWD/logs/dev_$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=16

# Create directories
mkdir -p "$STREAMING_CACHE_DIR" "$TENSORBOARD_LOG_DIR"

# Run with development config
python hybrid/training/train_energy.py \
    --config config_streaming_dev.json \
    --streaming_mode \
    --device cuda \
    --max_epochs 5
```

#### Multi-Node Training Template

```bash
#!/bin/bash
#SBATCH -J multi_streaming_hybrid
#SBATCH -p gpu_requeue
#SBATCH --constraint=a100
#SBATCH --gres=gpu:a100:4
#SBATCH -N 2
#SBATCH -c 64
#SBATCH -t 72:00:00
#SBATCH --mem=800G
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --signal=SIGUSR1@600

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Load modules on all nodes
module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01

# Run distributed training
srun python hybrid/training/train_energy.py \
    --config config_streaming_production.json \
    --distributed \
    --streaming_mode
```

### Production Monitoring

Monitor production training:

```bash
# Check job status
squeue -u $USER

# Monitor training progress
tail -f train_streaming_hybrid_*.out

# Check cache usage
du -sh $STREAMING_CACHE_DIR

# Monitor GPU utilization
nvidia-smi -l 5
```

## Best Practices

### Development Workflow

1. **Start Small**: Test with local PDB files before remote
2. **Incremental Scaling**: Gradually increase batch size and cache
3. **Monitor Resources**: Track memory and disk usage continuously
4. **Validate Configurations**: Use config validation before production
5. **Test Graceful Shutdown**: Ensure checkpoint saving works correctly

### Production Deployment

1. **Pre-warm Cache**: Download common PDBs before training
2. **Monitor Disk Space**: Set up alerts for low disk space
3. **Use Environment Variables**: Never hardcode paths in configs
4. **Enable Email Notifications**: Configure SLURM email alerts
5. **Plan for Failures**: Test restart and recovery procedures

### Cache Management

1. **Size Appropriately**: Use 10-20% of available disk for cache
2. **Monitor Hit Rates**: Aim for >80% cache hit rates
3. **Regular Cleanup**: Clear old cache files periodically
4. **Backup Critical Data**: Cache is temporary, checkpoint regularly

### Error Handling

1. **Graceful Degradation**: Handle network failures gracefully
2. **Retry Logic**: Implement exponential backoff for downloads
3. **Validation**: Validate PDB files after download
4. **Logging**: Enable comprehensive logging for debugging

## API Reference

### StreamingProteinDataset

```python
class StreamingProteinDataset(IterableDataset):
    def __init__(
        self,
        data_sources: List[Dict[str, Any]],
        cache_dir: Path,
        batch_size: int = 32,
        prefetch_factor: int = 2,
        num_workers: int = 4,
        negative_sampling_ratio: float = 0.5,
        max_sequence_length: int = 500,
        min_sequence_length: int = 20,
        augmentation_config: Optional[Dict] = None,
        seed: Optional[int] = None
    )
```

**Parameters:**
- `data_sources`: List of data source configurations
- `cache_dir`: Directory for caching downloaded data
- `batch_size`: Batch size for prefetching
- `negative_sampling_ratio`: Ratio of negative to positive samples
- `max_sequence_length`: Maximum protein sequence length
- `augmentation_config`: Configuration for data augmentation

**Methods:**
- `__iter__()`: Iterate over dataset samples
- `validate_negative_sample(sample)`: Validate sample quality
- `get_statistics()`: Get dataset statistics

### PDBCache

```python
class PDBCache:
    def __init__(
        self,
        cache_dir: Path,
        max_memory_mb: int = 1024,
        max_disk_gb: float = 5.0,
        max_concurrent_downloads: int = 16
    )
```

**Methods:**
- `get(pdb_id, download_url=None)`: Get PDB data from cache
- `get_stats()`: Get comprehensive cache statistics
- `ensure_cache_space(bytes_needed)`: Ensure sufficient cache space
- `clear_cache()`: Clear all cached data

### PDBListManager

```python
class PDBListManager:
    def get_filtered_pdb_list(
        self,
        max_resolution: float = 3.5,
        min_length: int = 20,
        max_length: int = 500,
        experimental_methods: Optional[List[str]] = None,
        target_count: int = 5000,
        use_cache: bool = True
    ) -> List[str]
```

**Methods:**
- `get_filtered_pdb_list()`: Get filtered PDB list from RCSB
- `get_biological_query_sets()`: Get pre-defined query sets
- `get_statistics()`: Get manager statistics

### Configuration Validation

```python
from hybrid.training.validate_config import ConfigValidator

validator = ConfigValidator()
result = validator.validate_config('config.json')

if result.is_valid:
    print("Configuration valid!")
else:
    print("Errors:", result.errors)
```

---

## Summary

The Streaming PDB System provides a comprehensive solution for large-scale protein structure training. Key benefits include:

- **Scalability**: Handle 19,000+ structures efficiently
- **Performance**: A100-optimized for maximum throughput  
- **Reliability**: Production-ready with robust error handling
- **Flexibility**: Support for multiple data sources and sampling methods

For detailed troubleshooting and advanced usage, see the [Troubleshooting Guide](troubleshooting_guide.md) and [Performance Tuning Guide](hybrid/training/a100_performance_tuning_guide.md).