# Streaming PDB System Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [Error Messages and Solutions](#error-messages-and-solutions)
3. [Performance Issues](#performance-issues)
4. [Harvard Cluster Specific](#harvard-cluster-specific)
5. [Cache Problems](#cache-problems)
6. [Network and Download Issues](#network-and-download-issues)
7. [Memory and Resource Issues](#memory-and-resource-issues)
8. [Configuration Problems](#configuration-problems)
9. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
10. [Recovery Procedures](#recovery-procedures)
11. [Contact and Support](#contact-and-support)

## Common Issues

### 1. Training Fails to Start

**Symptoms:**
- Python import errors
- Configuration validation failures
- CUDA/GPU initialization errors

**Quick Diagnosis:**
```bash
# Check basic setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from hybrid.data.streaming_dataset import StreamingProteinDataset"

# Validate configuration
python hybrid/training/validate_config.py config_streaming.json -v
```

**Common Solutions:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Load required modules: `module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01`
- Check GPU availability: `nvidia-smi`
- Verify configuration file syntax and paths

### 2. Cache Directory Not Accessible

**Symptoms:**
- Permission denied errors
- Cache directory creation failures
- "No space left on device" errors

**Diagnosis:**
```bash
# Check cache directory permissions
ls -la $(dirname $STREAMING_CACHE_DIR)
df -h $STREAMING_CACHE_DIR

# Test cache directory creation
mkdir -p $STREAMING_CACHE_DIR/test && echo "Success" || echo "Failed"
```

**Solutions:**
- Create cache directory with proper permissions: `mkdir -p $STREAMING_CACHE_DIR`
- Check disk space: Use `df -h` and clear space if needed
- Verify environment variables: `echo $STREAMING_CACHE_DIR`
- For Harvard cluster: Ensure netscratch path is correct

### 3. Slow Training Performance

**Symptoms:**
- Low GPU utilization (<50%)
- Long iteration times
- High I/O wait times

**Quick Checks:**
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Check I/O performance
iotop -a -o

# Monitor cache hit rate
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('$STREAMING_CACHE_DIR')
stats = cache.get_stats()
print(f'Hit rate: {stats[\"detailed_stats\"][\"hit_rate\"]:.2%}')
"
```

## Error Messages and Solutions

### Configuration Errors

#### Error: "Configuration file not found"
```
FileNotFoundError: [Errno 2] No such file or directory: 'config_streaming.json'
```

**Solution:**
```bash
# Verify config file exists
ls -la config_streaming.json

# Use absolute path
python train_energy.py --config /full/path/to/config_streaming.json

# Check current directory
pwd
```

#### Error: "Invalid JSON in configuration"
```
json.JSONDecodeError: Expecting ',' delimiter: line 45 column 5
```

**Solution:**
```bash
# Validate JSON syntax
python -m json.tool config_streaming.json

# Common JSON errors:
# - Missing commas between elements
# - Trailing commas before closing braces
# - Unescaped quotes in strings
# - Comments (not allowed in JSON)
```

#### Error: "Cache directory uses hardcoded path"
```
WARNING: Cache directory uses hardcoded path '/fixed/path/cache' - consider using environment variables
```

**Solution:**
```json
// Instead of hardcoded path:
"cache_dir": "/n/netscratch/ydu_lab/Lab/mkrasnow/cache"

// Use environment variable:
"cache_dir": "${STREAMING_CACHE_DIR:-./cache/streaming}"
```

### Import and Dependency Errors

#### Error: "ProteinMPNN utilities not available"
```
ImportError: cannot import name 'parse_PDB' from 'protein_mpnn_utils'
```

**Solution:**
```bash
# Check ProteinMPNN path
ls -la proteinmpnn/protein_mpnn_utils.py

# Add to Python path if needed
export PYTHONPATH="$PWD/proteinmpnn:$PYTHONPATH"

# Or install proteinmpnn package
cd proteinmpnn && pip install -e .
```

#### Error: "CUDA version mismatch"
```
RuntimeError: CUDA runtime version 11.8 does not match the version with which PyTorch was compiled
```

**Solution:**
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Load compatible CUDA module
module load cuda/12.2.0-fasrc01

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Runtime Errors

#### Error: "Invalid PDB ID format"
```
ValueError: Invalid PDB ID format: ../../../etc/passwd
```

**Solution:**
This is a security validation error. PDB IDs must be exactly 4 characters (digit + 3 alphanumeric).

```python
# Valid PDB IDs:
"1ABC", "2XYZ", "3h4j"

# Invalid PDB IDs:
"../path", "abc", "12345", "1AB$"
```

#### Error: "No PDB IDs found in RCSB response"
```
WARNING: No PDB IDs found in RCSB response
```

**Solution:**
```bash
# Check network connectivity
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" > /dev/null && echo "Connected" || echo "Network issue"

# Check query filters in configuration
# Overly restrictive filters may return no results
```

#### Error: "Sequence length outside bounds"
```
ERROR: Sequence length 1500 outside bounds [20, 500]
```

**Solution:**
```json
// Adjust sequence length limits in configuration
{
    "data": {
        "max_sequence_length": 1000,  // Increase if needed
        "min_sequence_length": 20
    }
}
```

## Performance Issues

### Low GPU Utilization

**Symptoms:**
- GPU utilization <50%
- Training much slower than expected
- CPU bottlenecks visible in monitoring

**Diagnosis:**
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet -d 1

# Monitor CPU usage
htop

# Check data loading performance
python -c "
import time
from your_dataloader import dataloader
start = time.time()
for i, batch in enumerate(dataloader):
    if i >= 100: break
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f'Batch {i}: {elapsed:.2f}s')
"
```

**Solutions:**

1. **Increase Worker Threads:**
```json
{
    "training": {
        "num_workers": 16  // Increase from 8
    },
    "streaming": {
        "num_workers": 16
    }
}
```

2. **Optimize Batch Size:**
```json
{
    "training": {
        "batch_size": 32,  // Increase if memory allows
        "gradient_accumulation_steps": 2  // Adjust accordingly
    }
}
```

3. **Improve Prefetching:**
```json
{
    "streaming": {
        "prefetch_factor": 8,  // Increase prefetch
        "max_memory_mb": 8192  // More cache memory
    }
}
```

### High Memory Usage

**Symptoms:**
- Out of memory errors
- System becomes unresponsive
- Memory usage keeps growing

**Diagnosis:**
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head -20

# Check GPU memory
nvidia-smi

# Monitor cache memory usage
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('$STREAMING_CACHE_DIR')
stats = cache.get_stats()
print(f'Memory cache: {stats[\"memory_cache\"][\"size_mb\"]:.1f}MB')
print(f'Utilization: {stats[\"memory_cache\"][\"utilization_percent\"]:.1f}%')
"
```

**Solutions:**

1. **Reduce Cache Size:**
```json
{
    "streaming": {
        "max_memory_mb": 2048  // Reduce from 5120
    },
    "cache_config": {
        "pdb_cache": {
            "max_memory_mb": 1024  // Reduce cache
        }
    }
}
```

2. **Reduce Batch Size:**
```json
{
    "training": {
        "batch_size": 8,  // Reduce from 16
        "gradient_accumulation_steps": 8  // Maintain effective batch size
    }
}
```

3. **Enable Memory Monitoring:**
```json
{
    "debug": {
        "memory_profiling": true,
        "cache_statistics": true
    }
}
```

### Slow Download Performance

**Symptoms:**
- Long wait times for PDB downloads
- Low cache hit rates
- Network timeout errors

**Diagnosis:**
```bash
# Test download speed
time curl -o test.pdb "https://files.rcsb.org/download/1ABC.pdb"

# Check concurrent downloads
netstat -an | grep ':443' | wc -l
```

**Solutions:**

1. **Optimize Concurrent Downloads:**
```json
{
    "streaming": {
        "concurrent_downloads": 16,  // Increase from 8
        "connection_pool_size": 64,  // Increase pool size
        "download_timeout_seconds": 600  // Increase timeout
    }
}
```

2. **Pre-warm Cache:**
```python
# Pre-download common PDBs
from hybrid.data.pdb_manager import PDBListManager
manager = PDBListManager()
common_pdbs = manager.get_biological_query_sets()['high_quality']
# Cache these before training
```

## Harvard Cluster Specific

### Systematic Cluster Diagnostic Procedures

#### Complete Production Diagnostic Workflow

This section provides systematic procedures for diagnosing issues in Harvard cluster production deployment. Follow these procedures in order when encountering problems.

#### Pre-Job Cluster Health Check

Before submitting jobs, run this comprehensive diagnostic:

```bash
#!/bin/bash
# File: cluster_health_check.sh

# Cluster diagnostics script for Harvard streaming training
echo "=== Harvard Cluster Health Check ==="
echo "Timestamp: $(date)"
echo "User: $USER"
echo "Host: $(hostname)"
echo

# 1. Module and Environment Check
echo "1. Module Environment:"
echo "   Available Python modules:"
module avail python 2>&1 | head -5
echo "   Available CUDA modules:"
module avail cuda 2>&1 | head -5

echo "   Currently loaded modules:"
module list 2>&1 || echo "   No modules loaded"

# 2. Storage and Quota Check
echo
echo "2. Storage Diagnostics:"
echo "   Home directory usage:"
du -sh $HOME 2>/dev/null || echo "   Cannot access $HOME"

echo "   Netscratch lab space:"
if [[ -d "/n/netscratch/ydu_lab/Lab" ]]; then
    echo "   ✓ Lab netscratch accessible"
    df -h /n/netscratch/ydu_lab/Lab | tail -1
else
    echo "   ✗ Lab netscratch NOT accessible"
fi

echo "   User scratch space:"
USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
if [[ -d "$USER_SCRATCH" ]]; then
    echo "   ✓ User scratch exists: $USER_SCRATCH"
    du -sh "$USER_SCRATCH" 2>/dev/null || echo "   Cannot measure usage"
    
    # Check write permissions
    test_file="$USER_SCRATCH/.write_test_$$"
    if touch "$test_file" 2>/dev/null; then
        echo "   ✓ Write permission confirmed"
        rm -f "$test_file"
    else
        echo "   ✗ Write permission DENIED"
    fi
else
    echo "   ✗ User scratch NOT found"
    echo "   Creating user scratch directory..."
    mkdir -p "$USER_SCRATCH" && echo "   ✓ Created" || echo "   ✗ Creation failed"
fi

# 3. GPU and Hardware Check
echo
echo "3. Hardware Diagnostics:"
if command -v nvidia-smi >/dev/null; then
    echo "   GPU information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    
    echo "   GPU processes:"
    nvidia-smi pmon -c 1 2>/dev/null || echo "   No active GPU processes"
else
    echo "   ✗ nvidia-smi not available"
fi

echo "   System memory:"
free -h | grep -E "(Mem|Swap)"

echo "   CPU information:"
echo "   Cores: $(nproc)"
echo "   Load average: $(uptime | awk -F'load average:' '{print $2}')"

# 4. Network Connectivity Check  
echo
echo "4. Network Diagnostics:"
echo "   RCSB PDB connectivity:"
if curl -s --max-time 10 "https://files.rcsb.org/download/1UBQ.pdb" >/dev/null; then
    echo "   ✓ RCSB reachable"
else
    echo "   ✗ RCSB unreachable or slow"
fi

echo "   DNS resolution:"
if nslookup files.rcsb.org >/dev/null 2>&1; then
    echo "   ✓ DNS working"
else
    echo "   ✗ DNS issues detected"
fi

# 5. SLURM Queue Status
echo
echo "5. SLURM Cluster Status:"
echo "   Queue information:"
squeue -p gpu_requeue,gpu_test --format="%.8i %.4P %.8u %.8T %.4C %.6D %.5Q" | head -10

echo "   User's current jobs:"
squeue -u $USER --format="%.8i %.4P %.20j %.8T %.10M %.6D" || echo "   No current jobs"

echo "   Available GPU resources:"
sinfo -p gpu_requeue,gpu_test --format="%.10P %.5D %.5C %.7m %.10G %.6t" | head -10

# 6. Python Environment Check
echo
echo "6. Python Environment:"
if command -v python >/dev/null; then
    echo "   Python version: $(python --version)"
    echo "   Python path: $(which python)"
    
    echo "   Critical packages:"
    python -c "
import sys
packages = ['torch', 'numpy', 'pandas']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'   ✓ {pkg}: {version}')
    except ImportError:
        print(f'   ✗ {pkg}: NOT FOUND')
" 2>/dev/null
else
    echo "   ✗ Python not available"
fi

# 7. Streaming System Check
echo
echo "7. Streaming System Readiness:"
repo_root="$PWD"
if [[ -f "$repo_root/hybrid/data/streaming_dataset.py" ]]; then
    echo "   ✓ Streaming code found"
    
    # Test import
    if python -c "import sys; sys.path.insert(0, '$repo_root'); import hybrid.data.streaming_dataset" 2>/dev/null; then
        echo "   ✓ Streaming imports work"
    else
        echo "   ✗ Streaming import failed"
    fi
else
    echo "   ✗ Streaming code not found in $repo_root"
fi

echo
echo "=== Health Check Complete ==="
echo "Review any ✗ items before submitting jobs"
```

#### Systematic Production Issue Diagnosis

Follow this systematic approach when troubleshooting production issues:

##### Step 1: Environment and Resource Verification

```bash
#!/bin/bash
# File: systematic_diagnosis_step1.sh
# Step 1: Verify environment and resource availability

echo "=== Step 1: Environment and Resource Verification ==="
echo "Timestamp: $(date)"
echo

# 1.1 Environment Variables Check
echo "1.1 Critical Environment Variables:"
REQUIRED_VARS=(
    "USER" "SLURM_MAIL_USER" "USER_SCRATCH" 
    "STREAMING_CACHE_DIR" "TENSORBOARD_LOG_DIR" 
    "CHECKPOINT_DIR" "PYTHONPATH"
)

missing_vars=()
for var in "${REQUIRED_VARS[@]}"; do
    value=$(printenv "$var")
    if [[ -n "$value" ]]; then
        echo "  ✓ $var: $value"
    else
        echo "  ✗ $var: NOT SET"
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "  CRITICAL: Missing required variables: ${missing_vars[*]}"
    echo "  → Set these before proceeding"
    exit 1
fi

# 1.2 Module Environment
echo
echo "1.2 Module Environment:"
if module list 2>&1 | grep -q "python\|cuda"; then
    echo "  ✓ Required modules loaded:"
    module list 2>&1 | grep -E "(python|cuda|gcc)"
else
    echo "  ✗ Required modules not loaded"
    echo "  → Load with: module load python/3.10.9-fasrc01 cuda/12.2.0-fasrc01"
    exit 1
fi

# 1.3 Storage Accessibility
echo
echo "1.3 Storage Accessibility:"
for dir_var in USER_SCRATCH STREAMING_CACHE_DIR TENSORBOARD_LOG_DIR CHECKPOINT_DIR; do
    dir_path=$(printenv "$dir_var")
    if [[ -n "$dir_path" ]]; then
        if [[ -d "$dir_path" ]]; then
            echo "  ✓ $dir_var: $dir_path (exists)"
            # Test write permission
            test_file="$dir_path/.write_test_$$"
            if touch "$test_file" 2>/dev/null; then
                echo "    ✓ Write permission confirmed"
                rm -f "$test_file"
            else
                echo "    ✗ Write permission DENIED"
                exit 1
            fi
        else
            echo "  ⚠ $dir_var: $dir_path (creating...)"
            if mkdir -p "$dir_path" 2>/dev/null; then
                echo "    ✓ Created successfully"
            else
                echo "    ✗ Creation FAILED"
                exit 1
            fi
        fi
    fi
done

# 1.4 Disk Space Verification
echo
echo "1.4 Disk Space Verification:"
for dir_var in USER_SCRATCH STREAMING_CACHE_DIR; do
    dir_path=$(printenv "$dir_var")
    if [[ -d "$dir_path" ]]; then
        available=$(df -BG "$dir_path" | tail -1 | awk '{print $4}' | sed 's/G//')
        echo "  $dir_var: ${available}GB available"
        if [[ $available -lt 10 ]]; then
            echo "    ⚠ WARNING: Less than 10GB available"
        fi
    fi
done

echo "✓ Step 1 Complete: Environment verified"
```

##### Step 2: SLURM Queue and Resource Analysis

```bash
#!/bin/bash
# File: systematic_diagnosis_step2.sh  
# Step 2: SLURM queue status and resource availability

echo "=== Step 2: SLURM Queue and Resource Analysis ==="
echo

# 2.1 User Job Status
echo "2.1 Current Job Status:"
USER_JOBS=$(squeue -u $USER --format="%.8i %.4P %.20j %.8T %.10M %.6D %.15R" 2>/dev/null)
if [[ -n "$USER_JOBS" ]]; then
    echo "$USER_JOBS"
    
    # Check for problematic job states
    FAILED_JOBS=$(echo "$USER_JOBS" | grep -c "FAILED\|CANCELLED\|TIMEOUT")
    PENDING_JOBS=$(echo "$USER_JOBS" | grep -c "PENDING")
    
    if [[ $FAILED_JOBS -gt 0 ]]; then
        echo "  ⚠ $FAILED_JOBS failed jobs detected"
        echo "  → Run diagnosis step 3 for detailed failure analysis"
    fi
    
    if [[ $PENDING_JOBS -gt 0 ]]; then
        echo "  ⚠ $PENDING_JOBS jobs pending"
        echo "  → Check resource availability below"
    fi
else
    echo "  No current jobs"
fi

# 2.2 GPU Resource Availability
echo
echo "2.2 GPU Resource Availability:"
GPU_NODES=$(sinfo -p gpu_requeue,gpu_test --format="%.10P %.10R %.5D %.5C %.10G %.6t" --noheader)
if [[ -n "$GPU_NODES" ]]; then
    echo "Partition   Nodes      CPUs   GPUs       State"
    echo "$GPU_NODES" | while read line; do
        echo "$line"
    done
    
    # Count available A100 nodes
    IDLE_A100=$(echo "$GPU_NODES" | grep -c "a100.*idle")
    MIX_A100=$(echo "$GPU_NODES" | grep -c "a100.*mix")
    echo
    echo "  A100 Summary: $IDLE_A100 idle, $MIX_A100 partially used"
    
    if [[ $IDLE_A100 -eq 0 && $MIX_A100 -eq 0 ]]; then
        echo "  ⚠ No A100 nodes available - expect queue wait"
    fi
else
    echo "  ✗ Cannot query GPU node status"
fi

# 2.3 Queue Wait Time Estimation
echo
echo "2.3 Queue Analysis:"
QUEUE_PENDING=$(squeue -p gpu_requeue --states=PENDING --noheader | wc -l)
QUEUE_RUNNING=$(squeue -p gpu_requeue --states=RUNNING --noheader | wc -l)
echo "  gpu_requeue: $QUEUE_RUNNING running, $QUEUE_PENDING pending"

if [[ $QUEUE_PENDING -gt 50 ]]; then
    echo "  ⚠ High queue load - consider gpu_test for development"
elif [[ $QUEUE_PENDING -gt 20 ]]; then
    echo "  ⚠ Moderate queue load - expect delays"
else
    echo "  ✓ Normal queue load"
fi

# 2.4 Resource Usage History
echo
echo "2.4 Recent Resource Usage:"
RECENT_JOBS=$(sacct -u $USER -S $(date -d '24 hours ago' '+%Y-%m-%d') --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode --noheader 2>/dev/null | tail -5)
if [[ -n "$RECENT_JOBS" ]]; then
    echo "Recent Jobs (last 24h):"
    echo "JobID    JobName              State     Elapsed   MaxRSS    ExitCode"
    echo "$RECENT_JOBS"
else
    echo "  No recent job history available"
fi

echo "✓ Step 2 Complete: SLURM status analyzed"
```

##### Step 3: Application-Specific Diagnosis

```bash
#!/bin/bash
# File: systematic_diagnosis_step3.sh
# Step 3: Streaming application specific diagnostics

echo "=== Step 3: Application-Specific Diagnosis ==="
echo

# 3.1 Python Environment Validation
echo "3.1 Python Environment:"
if command -v python >/dev/null; then
    echo "  Python: $(python --version)"
    echo "  Location: $(which python)"
    
    # Test critical imports
    echo "  Testing critical imports..."
    python -c "
import sys, torch, numpy
print(f'  ✓ PyTorch: {torch.__version__}')
print(f'  ✓ NumPy: {numpy.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'    GPU {i}: {props.name}')
" 2>/dev/null || {
    echo "  ✗ Critical Python imports failed"
    echo "  → Check module loading and package installation"
    exit 1
}

    # Test streaming system imports
    echo "  Testing streaming system imports..."
    python -c "
import sys
sys.path.insert(0, '$PWD')
try:
    from hybrid.data.streaming_dataset import StreamingProteinDataset
    from hybrid.data.pdb_cache import PDBCache
    from hybrid.data.pdb_manager import PDBListManager
    print('  ✓ Streaming system imports successful')
except ImportError as e:
    print(f'  ✗ Streaming system import failed: {e}')
    sys.exit(1)
" || {
    echo "  → Check PYTHONPATH and code availability"
    exit 1
}
else
    echo "  ✗ Python not available"
    exit 1
fi

# 3.2 Configuration Validation
echo
echo "3.2 Configuration Validation:"
CONFIG_FILES=("config_streaming.json" "hybrid/training/config_streaming.json")
VALID_CONFIG=""

for config in "${CONFIG_FILES[@]}"; do
    if [[ -f "$config" ]]; then
        echo "  Found config: $config"
        if python -c "import json; json.load(open('$config'))" 2>/dev/null; then
            echo "  ✓ Valid JSON format"
            VALID_CONFIG="$config"
            break
        else
            echo "  ✗ Invalid JSON format"
        fi
    fi
done

if [[ -n "$VALID_CONFIG" ]]; then
    # Validate configuration content
    python -c "
import json, os
config = json.load(open('$VALID_CONFIG'))

# Check required sections
required = ['streaming', 'data_sources']
missing = [key for key in required if key not in config]
if missing:
    print(f'  ✗ Missing config sections: {missing}')
    exit(1)

# Check cache directory
cache_dir = config.get('streaming', {}).get('cache_dir', '')
if cache_dir:
    # Expand environment variables
    expanded = os.path.expandvars(cache_dir)
    if os.path.exists(os.path.dirname(expanded)):
        print(f'  ✓ Cache directory path valid: {expanded}')
    else:
        print(f'  ✗ Cache directory parent missing: {expanded}')

# Check data sources
sources = config.get('data_sources', [])
print(f'  ✓ Data sources configured: {len(sources)}')
for i, source in enumerate(sources):
    source_type = source.get('type', 'unknown')
    print(f'    Source {i+1}: {source_type}')

print('  ✓ Configuration validation passed')
" 2>/dev/null || {
    echo "  ✗ Configuration validation failed"
    echo "  → Fix configuration issues before proceeding"
    exit 1
}
else
    echo "  ✗ No valid configuration file found"
    echo "  → Create config_streaming.json before proceeding"
    exit 1
fi

# 3.3 Network Connectivity Test
echo
echo "3.3 Network Connectivity:"
echo "  Testing RCSB PDB access..."
if curl -s --max-time 10 "https://files.rcsb.org/download/1UBQ.pdb" >/dev/null; then
    echo "  ✓ RCSB PDB accessible"
    
    # Test download speed
    echo "  Testing download performance..."
    DOWNLOAD_TIME=$(time (curl -s "https://files.rcsb.org/download/1UBQ.pdb" >/dev/null) 2>&1 | grep real | awk '{print $2}')
    echo "  Download time for 1UBQ: $DOWNLOAD_TIME"
    
    # Test API access
    if curl -s --max-time 5 "https://search.rcsb.org/rcsbsearch/v2/query" >/dev/null; then
        echo "  ✓ RCSB API accessible"
    else
        echo "  ⚠ RCSB API slow/inaccessible (fallback will be used)"
    fi
else
    echo "  ⚠ RCSB PDB inaccessible (offline mode will be used)"
fi

# 3.4 Cache System Test
echo
echo "3.4 Cache System Test:"
CACHE_DIR="$STREAMING_CACHE_DIR"
if [[ -n "$CACHE_DIR" && -d "$CACHE_DIR" ]]; then
    echo "  Cache directory: $CACHE_DIR"
    
    # Test cache initialization
    python -c "
from pathlib import Path
sys.path.insert(0, '$PWD')
from hybrid.data.pdb_cache import PDBCache

cache = PDBCache(Path('$CACHE_DIR'))
stats = cache.get_stats()
print(f'  ✓ Cache initialized successfully')
print(f'  Memory cache: {stats[\"memory_cache\"][\"size_mb\"]:.1f}MB / {stats[\"memory_cache\"][\"max_size_mb\"]}MB')
print(f'  Disk cache: {stats[\"disk_cache\"][\"size_gb\"]:.2f}GB / {stats[\"disk_cache\"][\"max_size_gb\"]}GB')
print(f'  Cached files: {stats[\"disk_cache\"][\"file_count\"]}')
" 2>/dev/null || {
    echo "  ✗ Cache initialization failed"
    echo "  → Check cache directory permissions and disk space"
    exit 1
}
else
    echo "  ✗ Cache directory not set or missing"
    exit 1
fi

echo "✓ Step 3 Complete: Application diagnostics passed"
```

##### Step 4: Job Failure Analysis

```bash
#!/bin/bash
# File: systematic_diagnosis_step4.sh
# Step 4: Detailed job failure analysis (run when jobs fail)

echo "=== Step 4: Job Failure Analysis ==="
echo

if [[ -z "$1" ]]; then
    echo "Usage: $0 <job_id>"
    echo "Analyzing most recent failed job..."
    
    # Find most recent failed job
    RECENT_FAILED=$(sacct -u $USER --state=FAILED,CANCELLED,TIMEOUT --format=JobID --noheader | tail -1)
    if [[ -n "$RECENT_FAILED" ]]; then
        JOB_ID="$RECENT_FAILED"
        echo "Found recent failed job: $JOB_ID"
    else
        echo "No recent failed jobs found"
        exit 0
    fi
else
    JOB_ID="$1"
fi

echo "Analyzing job: $JOB_ID"
echo

# 4.1 Basic Job Information
echo "4.1 Job Information:"
scontrol show job $JOB_ID 2>/dev/null || {
    echo "Job not in current queue, checking accounting..."
    sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,TimeLimit,MaxRSS,ExitCode,NodeList
}

# 4.2 Resource Usage Analysis
echo
echo "4.2 Resource Usage:"
SACCT_OUTPUT=$(sacct -j $JOB_ID --format=JobID,MaxRSS,MaxVMSize,CPUTime,TotalCPU,UserCPU,SystemCPU --noheader)
echo "$SACCT_OUTPUT"

# Analyze memory usage
MAX_RSS=$(echo "$SACCT_OUTPUT" | awk '{print $2}' | head -1)
if [[ "$MAX_RSS" =~ ([0-9]+)([KMG]) ]]; then
    SIZE=${BASH_REMATCH[1]}
    UNIT=${BASH_REMATCH[2]}
    
    case $UNIT in
        G) RSS_GB=$SIZE ;;
        M) RSS_GB=$((SIZE / 1024)) ;;
        K) RSS_GB=$((SIZE / 1024 / 1024)) ;;
    esac
    
    echo "Memory analysis:"
    echo "  Peak memory usage: ${RSS_GB}GB"
    
    if [[ $RSS_GB -gt 350 ]]; then
        echo "  ⚠ Very high memory usage (>350GB) - consider reducing batch size"
    elif [[ $RSS_GB -gt 200 ]]; then
        echo "  ⚠ High memory usage (>200GB) - monitor for efficiency"
    else
        echo "  ✓ Normal memory usage"
    fi
fi

# 4.3 Exit Code Analysis
EXIT_CODE=$(sacct -j $JOB_ID --format=ExitCode --noheader | head -1 | cut -d: -f1)
echo
echo "4.3 Exit Code Analysis:"
echo "  Exit code: $EXIT_CODE"

case "$EXIT_CODE" in
    "0") echo "  ✓ Normal completion" ;;
    "1") echo "  ✗ General error - check application logs" ;;
    "125") echo "  ✗ Module/environment error" ;;
    "137") echo "  ✗ Killed (out of memory or SIGKILL)" ;;
    "139") echo "  ✗ Segmentation fault" ;;
    "143") echo "  ✗ Terminated (SIGTERM - likely timeout)" ;;
    *) echo "  ⚠ Unusual exit code - check documentation" ;;
esac

# 4.4 Log File Analysis
echo
echo "4.4 Log File Analysis:"
LOG_PATTERN="*${JOB_ID}*"
LOG_FILES=$(find . -name "$LOG_PATTERN" -type f 2>/dev/null)

if [[ -n "$LOG_FILES" ]]; then
    echo "  Log files found:"
    echo "$LOG_FILES" | while read log; do
        size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log")
        echo "    $log (${size} bytes)"
    done
    
    # Check for common error patterns
    echo
    echo "  Error pattern analysis:"
    
    ERRORS_FOUND=""
    if grep -l "CUDA out of memory\|RuntimeError.*memory" $LOG_FILES 2>/dev/null; then
        echo "    ✗ GPU memory exhaustion detected"
        ERRORS_FOUND="gpu_memory"
    fi
    
    if grep -l "No space left on device" $LOG_FILES 2>/dev/null; then
        echo "    ✗ Disk space exhaustion detected"
        ERRORS_FOUND="${ERRORS_FOUND} disk_space"
    fi
    
    if grep -l "Permission denied\|cannot access" $LOG_FILES 2>/dev/null; then
        echo "    ✗ Permission issues detected"
        ERRORS_FOUND="${ERRORS_FOUND} permissions"
    fi
    
    if grep -l "Connection.*timeout\|Network.*unreachable\|Download.*failed" $LOG_FILES 2>/dev/null; then
        echo "    ✗ Network connectivity issues detected"
        ERRORS_FOUND="${ERRORS_FOUND} network"
    fi
    
    if grep -l "ImportError\|ModuleNotFoundError" $LOG_FILES 2>/dev/null; then
        echo "    ✗ Python import errors detected"
        ERRORS_FOUND="${ERRORS_FOUND} imports"
    fi
    
    if [[ -z "$ERRORS_FOUND" ]]; then
        echo "    ✓ No common error patterns found"
        echo "    → Check logs manually for application-specific issues"
        
        # Show last few lines of error log
        ERROR_LOG=$(echo "$LOG_FILES" | grep -E "\.(err|error)" | head -1)
        if [[ -n "$ERROR_LOG" ]]; then
            echo
            echo "  Last 10 lines of error log:"
            tail -10 "$ERROR_LOG"
        fi
    fi
else
    echo "  ✗ No log files found for job $JOB_ID"
fi

echo
echo "✓ Step 4 Complete: Job failure analysis finished"
```

##### Step 5: Recovery and Mitigation

```bash
#!/bin/bash  
# File: systematic_diagnosis_step5.sh
# Step 5: Automated recovery and mitigation suggestions

echo "=== Step 5: Recovery and Mitigation ==="
echo

# 5.1 Checkpoint Recovery Assessment
echo "5.1 Checkpoint Recovery Assessment:"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$USER_SCRATCH/checkpoints}"

if [[ -d "$CHECKPOINT_DIR" ]]; then
    CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f 2>/dev/null)
    
    if [[ -n "$CHECKPOINTS" ]]; then
        echo "  ✓ Checkpoints found in $CHECKPOINT_DIR"
        
        # Find most recent checkpoint
        LATEST_CHECKPOINT=$(ls -1t $CHECKPOINT_DIR/*.pt 2>/dev/null | head -1)
        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            CHECKPOINT_AGE=$(( $(date +%s) - $(stat -f%m "$LATEST_CHECKPOINT" 2>/dev/null || stat -c%Y "$LATEST_CHECKPOINT") ))
            HOURS_OLD=$((CHECKPOINT_AGE / 3600))
            
            echo "  Latest checkpoint: $(basename $LATEST_CHECKPOINT)"
            echo "  Age: ${HOURS_OLD}h ago"
            
            # Test checkpoint integrity
            python -c "
import torch
try:
    checkpoint = torch.load('$LATEST_CHECKPOINT', map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f'  ✓ Checkpoint valid (epoch: {epoch})')
    
    # Check for required keys
    required_keys = ['model_state_dict', 'optimizer_state_dict']
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        print(f'  ⚠ Missing checkpoint keys: {missing}')
    else:
        print('  ✓ Checkpoint complete')
except Exception as e:
    print(f'  ✗ Checkpoint corrupted: {e}')
" 2>/dev/null
            
            if [[ $CHECKPOINT_AGE -lt 7200 ]]; then  # Less than 2 hours
                echo "  → RECOMMENDATION: Resume from checkpoint"
            else
                echo "  → RECOMMENDATION: Verify checkpoint before resuming"
            fi
        fi
    else
        echo "  ⚠ No checkpoints found"
        echo "  → RECOMMENDATION: Start fresh training"
    fi
else
    echo "  ⚠ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "  → RECOMMENDATION: Create checkpoint directory and enable checkpointing"
fi

# 5.2 Configuration Optimization Suggestions
echo
echo "5.2 Configuration Optimization:"

# Detect previous job failures and suggest fixes
RECENT_FAILURES=$(sacct -u $USER --state=FAILED,CANCELLED,TIMEOUT --format=JobID,ExitCode --noheader -S $(date -d '7 days ago' '+%Y-%m-%d') | wc -l)

if [[ $RECENT_FAILURES -gt 3 ]]; then
    echo "  ⚠ Multiple recent failures detected ($RECENT_FAILURES in last 7 days)"
    echo "  → RECOMMENDATION: Review and optimize configuration"
    
    # Suggest conservative settings
    echo
    echo "  Suggested conservative configuration adjustments:"
    echo "    # Reduce resource usage"
    echo "    \"batch_size\": 8,                    # Reduce from 16"
    echo "    \"gradient_accumulation_steps\": 8,   # Maintain effective batch size"
    echo "    \"max_memory_mb\": 2048,              # Reduce cache memory"
    echo "    \"num_workers\": 4,                   # Reduce I/O workers"
    echo "    \"concurrent_downloads\": 4,          # Reduce concurrent downloads"
    echo "    \"prefetch_factor\": 2                # Reduce prefetching"
    
elif [[ $RECENT_FAILURES -gt 0 ]]; then
    echo "  ⚠ Some recent failures detected ($RECENT_FAILURES in last 7 days)"
    echo "  → RECOMMENDATION: Monitor next run closely"
else
    echo "  ✓ No recent failures detected"
fi

# 5.3 Resource Allocation Suggestions
echo
echo "5.3 Resource Allocation Suggestions:"

# Check current resource availability
AVAILABLE_MEMORY=$(free -g | grep Mem | awk '{print $7}')
AVAILABLE_DISK=$(df -BG "$USER_SCRATCH" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')

echo "  Current availability:"
echo "    Memory: ${AVAILABLE_MEMORY}GB"
echo "    Disk: ${AVAILABLE_DISK}GB"

# Suggest resource optimization
if [[ $AVAILABLE_MEMORY -lt 50 ]]; then
    echo "  ⚠ Low memory availability"
    echo "  → RECOMMENDATION: Request memory-optimized node or reduce memory usage"
fi

if [[ $AVAILABLE_DISK -lt 20 ]]; then
    echo "  ⚠ Low disk space"
    echo "  → RECOMMENDATION: Clean cache or request additional storage"
    
    # Suggest cache cleanup
    echo
    echo "  Cache cleanup commands:"
    echo "    # Remove old cache files"
    echo "    find \$STREAMING_CACHE_DIR -name '*.pt' -mtime +7 -delete"
    echo "    # Clear temporary files"
    echo "    find \$STREAMING_CACHE_DIR -name '*.tmp' -delete"
fi

# 5.4 Job Submission Recommendations
echo
echo "5.4 Job Submission Recommendations:"

# Check queue status for recommendation
QUEUE_LENGTH=$(squeue -p gpu_requeue --states=PENDING --noheader | wc -l)

if [[ $QUEUE_LENGTH -gt 50 ]]; then
    echo "  ⚠ High queue load (${QUEUE_LENGTH} pending jobs)"
    echo "  → RECOMMENDATION: Use gpu_test partition for development"
    echo "  → OR: Submit during off-peak hours"
elif [[ $QUEUE_LENGTH -gt 20 ]]; then
    echo "  ⚠ Moderate queue load (${QUEUE_LENGTH} pending jobs)"
    echo "  → RECOMMENDATION: Expect delays"
else
    echo "  ✓ Normal queue load (${QUEUE_LENGTH} pending jobs)"
fi

# 5.5 Monitoring and Alerting Setup
echo
echo "5.5 Monitoring Setup Verification:"

if [[ -n "$SLURM_MAIL_USER" ]]; then
    echo "  ✓ Email notifications configured: $SLURM_MAIL_USER"
else
    echo "  ⚠ No email notifications configured"
    echo "  → RECOMMENDATION: Set SLURM_MAIL_USER for job status updates"
fi

# Check for monitoring scripts
if [[ -f "monitor_training.sh" ]]; then
    echo "  ✓ Monitoring script available"
else
    echo "  ⚠ No monitoring script found"
    echo "  → RECOMMENDATION: Set up training monitoring"
fi

echo
echo "=== Complete Diagnostic Summary ==="
echo "Follow the above recommendations to resolve issues."
echo "If problems persist, gather this diagnostic output for support requests."
echo "✓ Systematic diagnosis complete"
```

#### Cluster Resource Monitoring

```bash
#!/bin/bash
# File: monitor_cluster_resources.sh

# Real-time cluster resource monitoring
echo "Starting cluster resource monitoring..."
echo "Press Ctrl+C to stop"

while true; do
    clear
    echo "=== Cluster Resource Monitor - $(date) ==="
    echo
    
    # Current user jobs
    echo "Your Running Jobs:"
    squeue -u $USER --format="%.8i %.4P %.20j %.8T %.10M %.6D %.5Q %.15R" || echo "No jobs running"
    echo
    
    # GPU utilization on allocated nodes
    if [[ -n "$SLURM_JOB_NODELIST" ]]; then
        echo "GPU Status on Allocated Nodes ($SLURM_JOB_NODELIST):"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
        echo
    fi
    
    # Queue status
    echo "GPU Queue Summary:"
    squeue -p gpu_requeue,gpu_test --states=RUNNING,PENDING --format="%.4P %.5D %.6t" | tail -n +2 | sort | uniq -c
    echo
    
    # Available resources
    echo "Available GPU Nodes:"
    sinfo -p gpu_requeue,gpu_test -t idle,mix --format="%.10P %.10R %.5D %.5C %.7m %.10G"
    echo
    
    # Storage usage
    if [[ -d "/n/netscratch/ydu_lab/Lab/$USER" ]]; then
        echo "Storage Usage:"
        df -h /n/netscratch/ydu_lab/Lab | tail -1
        echo "Your usage: $(du -sh /n/netscratch/ydu_lab/Lab/$USER 2>/dev/null | cut -f1)"
    fi
    
    echo "Refreshing in 30 seconds..."
    sleep 30
done
```

### SLURM Job Issues

#### Error: "Job killed due to time limit"
```
slurmstepd: error: *** JOB 12345 ON node001 CANCELLED AT 2024-12-02T23:59:59 DUE TO TIME LIMIT ***
```

**Diagnosis Steps:**
1. Check job time allocation: `sacct -j $JOB_ID --format=JobID,Elapsed,Timelimit`
2. Review training progress to estimate remaining time
3. Check if graceful shutdown was triggered

**Solutions:**
```bash
# Request longer time limit
#SBATCH -t 48:00:00  # 48 hours for production
#SBATCH -t 72:00:00  # 72 hours for large datasets

# Enable graceful shutdown with sufficient notice
#SBATCH --signal=SIGUSR1@600  # 10 minutes warning

# Auto-restart job if needed
#SBATCH --requeue

# Monitor job progress
scontrol show job $SLURM_JOB_ID
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,Elapsed,TimeLimit,MaxRSS
```

#### Job Failure Diagnosis Workflow

```bash
#!/bin/bash
# File: diagnose_job_failure.sh

JOB_ID=${1:-$SLURM_JOB_ID}

if [[ -z "$JOB_ID" ]]; then
    echo "Usage: $0 <job_id>"
    echo "Or run from within a SLURM job (uses \$SLURM_JOB_ID)"
    exit 1
fi

echo "=== SLURM Job Failure Diagnosis ==="
echo "Job ID: $JOB_ID"
echo "Timestamp: $(date)"
echo

# Job details
echo "1. Job Information:"
scontrol show job $JOB_ID 2>/dev/null || {
    echo "Job $JOB_ID not found in current queue"
    echo "Checking accounting database..."
}

# Accounting information
echo
echo "2. Job History:"
sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,TimeLimit,MaxRSS,MaxVMSize,ExitCode,NodeList

# Resource usage
echo
echo "3. Resource Usage:"
sacct -j $JOB_ID --format=JobID,MaxRSS,MaxVMSize,CPUTime,TotalCPU,UserCPU,SystemCPU

# Check for common failure patterns
echo
echo "4. Failure Pattern Analysis:"

# Check for memory issues
MAX_RSS=$(sacct -j $JOB_ID --format=MaxRSS --noheader | head -1 | tr -d ' ')
if [[ "$MAX_RSS" =~ [0-9]+[KMG] ]]; then
    echo "   Memory usage: $MAX_RSS"
    
    # Convert to MB for comparison (rough)
    if [[ "$MAX_RSS" =~ ([0-9]+)G ]]; then
        RSS_MB=$((${BASH_REMATCH[1]} * 1024))
    elif [[ "$MAX_RSS" =~ ([0-9]+)M ]]; then
        RSS_MB=${BASH_REMATCH[1]}
    else
        RSS_MB=0
    fi
    
    if [[ $RSS_MB -gt 200000 ]]; then
        echo "   ⚠ High memory usage detected (>200GB)"
        echo "   Consider reducing batch size or memory cache"
    fi
else
    echo "   Memory usage: Unknown"
fi

# Check exit codes
EXIT_CODE=$(sacct -j $JOB_ID --format=ExitCode --noheader | head -1 | tr -d ' ' | cut -d: -f1)
case "$EXIT_CODE" in
    "0") echo "   ✓ Job completed normally" ;;
    "1") echo "   ✗ General error (check logs)" ;;
    "2") echo "   ✗ Misuse of shell builtins" ;;
    "125") echo "   ✗ Container/module error" ;;
    "126") echo "   ✗ Command cannot execute" ;;
    "127") echo "   ✗ Command not found" ;;
    "128+*") echo "   ✗ Signal termination ($(($EXIT_CODE - 128)))" ;;
    "") echo "   Status: Still running or unknown" ;;
    *) echo "   Exit code: $EXIT_CODE (check documentation)" ;;
esac

# Check log files
echo
echo "5. Log File Analysis:"
# Look for common log file patterns
LOG_PATTERN="*${JOB_ID}*"
LOG_FILES=$(find . -name "$LOG_PATTERN" -type f 2>/dev/null)

if [[ -n "$LOG_FILES" ]]; then
    echo "   Found log files:"
    echo "$LOG_FILES" | while read -r log_file; do
        echo "     $log_file ($(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file") bytes)"
    done
    
    echo
    echo "   Last 20 lines from error log:"
    ERROR_LOG=$(echo "$LOG_FILES" | grep -E "\.(err|error)" | head -1)
    if [[ -n "$ERROR_LOG" ]]; then
        tail -20 "$ERROR_LOG"
    else
        echo "   No error log found"
    fi
    
    echo
    echo "   Common error patterns:"
    if echo "$LOG_FILES" | xargs grep -l "CUDA out of memory" 2>/dev/null; then
        echo "   ✗ GPU memory exhaustion detected"
        echo "     Solution: Reduce batch_size or use gradient accumulation"
    fi
    
    if echo "$LOG_FILES" | xargs grep -l "No space left on device" 2>/dev/null; then
        echo "   ✗ Disk space exhaustion detected"
        echo "     Solution: Clean cache or increase disk allocation"
    fi
    
    if echo "$LOG_FILES" | xargs grep -l "Permission denied" 2>/dev/null; then
        echo "   ✗ Permission issues detected"
        echo "     Solution: Check file/directory permissions"
    fi
    
    if echo "$LOG_FILES" | xargs grep -l "Connection timed out\|Network is unreachable" 2>/dev/null; then
        echo "   ✗ Network connectivity issues detected"
        echo "     Solution: Check internet access and retry"
    fi
else
    echo "   No log files found for job $JOB_ID"
fi

# Recovery recommendations
echo
echo "6. Recovery Recommendations:"

# Check if checkpoint exists
USER_SCRATCH="/n/netscratch/ydu_lab/Lab/$USER"
CHECKPOINT_DIR="$USER_SCRATCH/checkpoints"

if [[ -d "$CHECKPOINT_DIR" ]] && [[ $(ls -1 "$CHECKPOINT_DIR"/*.pt 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "   ✓ Checkpoints found in $CHECKPOINT_DIR"
    LATEST_CHECKPOINT=$(ls -1t "$CHECKPOINT_DIR"/*.pt | head -1)
    CHECKPOINT_AGE=$(( $(date +%s) - $(stat -f%m "$LATEST_CHECKPOINT" 2>/dev/null || stat -c%Y "$LATEST_CHECKPOINT") ))
    echo "   Latest checkpoint: $LATEST_CHECKPOINT ($(($CHECKPOINT_AGE / 3600))h ago)"
    
    if [[ $CHECKPOINT_AGE -lt 7200 ]]; then  # Less than 2 hours old
        echo "   → Can resume from recent checkpoint"
    else
        echo "   → Checkpoint may be outdated, verify before resuming"
    fi
else
    echo "   ⚠ No checkpoints found - job will restart from beginning"
fi

# Suggest configuration adjustments
echo "   Configuration adjustments:"
if [[ "$EXIT_CODE" == "137" ]] || echo "$LOG_FILES" | xargs grep -l "out of memory" 2>/dev/null; then
    echo "   → Reduce batch_size from current value"
    echo "   → Consider using gradient accumulation"
    echo "   → Decrease cache memory limits"
fi

echo
echo "=== Diagnosis Complete ==="
echo "Review the above information and adjust job parameters before resubmitting"
```

#### Error: "Insufficient disk space in netscratch"
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check disk usage
df -h /n/netscratch/ydu_lab/Lab/$USER

# Clean up old files
find $USER_SCRATCH -name "*.tmp" -delete
find $USER_SCRATCH -name "job_*" -mtime +7 -exec rm -rf {} \;

# Request quota increase if needed
```

#### Error: "Cannot access netscratch"
```
Permission denied: '/n/netscratch/ydu_lab/Lab'
```

**Solution:**
```bash
# Check group membership
groups

# Verify lab access
ls -la /n/netscratch/ydu_lab/

# Contact cluster admin if access is missing
```

### Module Loading Issues

#### Error: "Module not found"
```
ModuleCmd_Load.c(208):ERROR:105: Unable to locate a modulefile for 'python/3.10.9-fasrc01'
```

**Solution:**
```bash
# Check available modules
module avail python
module avail cuda

# Load correct modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

# Add to your .bashrc for persistence
```

## Cache Problems

### Cache Corruption

**Symptoms:**
- Parsing errors for cached PDB files
- Inconsistent training results
- "Invalid PDB format" errors

**Diagnosis:**
```bash
# Check cache file integrity
python -c "
import torch
try:
    data = torch.load('cache/pdb_cache/1ABC.pt')
    print('Cache file OK')
except:
    print('Cache file corrupted')
"

# Check disk errors
dmesg | grep -i error
```

**Solution:**
```bash
# Clear corrupted cache
rm -rf $STREAMING_CACHE_DIR/*

# Enable cache validation
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('$STREAMING_CACHE_DIR')
# Validation is enabled by default
"
```

### Cache Eviction Issues

**Symptoms:**
- Frequent cache misses despite large cache
- "Cache size exceeded" warnings
- Poor cache hit rates

**Diagnosis:**
```bash
# Monitor cache statistics
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('$STREAMING_CACHE_DIR')
stats = cache.get_stats()
print(f'Hit rate: {stats[\"detailed_stats\"][\"hit_rate\"]:.2%}')
print(f'Evictions: {stats[\"detailed_stats\"][\"evictions_performed\"]}')
"
```

**Solution:**
```json
{
    "cache_config": {
        "pdb_cache": {
            "max_disk_gb": 200,  // Increase cache size
            "max_memory_mb": 8192,  // More memory cache
            "eviction_policy": "lru"  // Keep least-recently-used policy
        }
    }
}
```

## Network and Download Issues

### RCSB API Failures

**Symptoms:**
- HTTP 429 (rate limit) errors
- Connection timeout errors
- API returning empty results

**Diagnosis:**
```bash
# Test RCSB connectivity
curl -v "https://files.rcsb.org/download/1ABC.pdb"

# Check API rate limiting
python -c "
import requests
import time
for i in range(10):
    resp = requests.get('https://files.rcsb.org/download/1ABC.pdb')
    print(f'{i}: {resp.status_code}')
    time.sleep(1)
"
```

**Solution:**
```json
{
    "data_sources": [{
        "rate_limit_per_second": 10,  // Reduce rate limit
        "retry_attempts": 5,  // Increase retries
        "timeout_seconds": 300  // Increase timeout
    }]
}
```

### DNS Resolution Issues

**Symptoms:**
- "Name resolution failed" errors
- Intermittent connection failures

**Solution:**
```bash
# Test DNS resolution
nslookup files.rcsb.org

# Use alternative DNS if needed
export RESOLVE_CONF=/etc/resolv.conf

# Add to job script if persistent
echo "nameserver 8.8.8.8" >> /tmp/resolv.conf
export RESOLVE_CONF=/tmp/resolv.conf
```

## Memory and Resource Issues

### Out of Memory (OOM) Errors

**Symptoms:**
- "CUDA out of memory" errors
- System OOM killer activating
- Training process killed

**Immediate Actions:**
1. Reduce batch size
2. Clear GPU memory cache
3. Restart training with smaller configuration

**Diagnosis:**
```bash
# Check memory usage patterns
dmesg | grep -i "killed process"

# Monitor GPU memory
nvidia-smi -l 1

# Check system memory
free -h && vmstat 1 5
```

**Solutions:**

1. **GPU Memory:**
```json
{
    "training": {
        "batch_size": 8,  // Reduce from 16
        "gradient_accumulation_steps": 8
    },
    "hardware_optimization": {
        "mixed_precision": "fp16",  // Use half precision
        "memory_fraction": 0.8  // Reduce GPU memory usage
    }
}
```

2. **System Memory:**
```json
{
    "streaming": {
        "max_memory_mb": 1024,  // Reduce cache
        "num_workers": 4  // Reduce workers
    }
}
```

### Resource Contention

**Symptoms:**
- High I/O wait times
- CPU usage at 100%
- Network bandwidth saturation

**Solution:**
```json
{
    "streaming": {
        "concurrent_downloads": 4,  // Reduce concurrency
        "num_workers": 8,  // Match CPU cores
        "prefetch_factor": 2  // Reduce prefetching
    }
}
```

## Configuration Problems

### Invalid Parameter Values

**Error:** "Batch size must be positive integer"
```
ValueError: Invalid batch_size: 0
```

**Solution:**
```bash
# Validate configuration before training
python hybrid/training/validate_config.py config_streaming.json -v

# Common validation errors:
# - Batch size <= 0
# - Invalid file paths
# - Conflicting memory settings
# - Missing required fields
```

### Path Resolution Issues

**Error:** "Cache directory parent does not exist"
```
WARNING: Cache directory parent does not exist: /non/existent/path
```

**Solution:**
```bash
# Create parent directories
mkdir -p $(dirname $STREAMING_CACHE_DIR)

# Use relative paths for portability
export STREAMING_CACHE_DIR="./cache/streaming"

# Verify environment variable expansion
echo "Cache dir: $STREAMING_CACHE_DIR"
```

## Debugging Tools and Techniques

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in configuration:
{
    "debug": {
        "profile_data_loading": true,
        "validate_samples": true,
        "memory_profiling": true,
        "cache_statistics": true
    }
}
```

### Performance Profiling

```python
# Profile data loading
import time
from hybrid.data.streaming_dataset import StreamingProteinDataset

start_time = time.time()
for i, batch in enumerate(dataset):
    if i % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Batch {i}: {elapsed:.2f}s avg per batch")
    if i >= 1000:
        break
```

### Cache Analysis

```bash
# Analyze cache performance
python -c "
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache('$STREAMING_CACHE_DIR')
stats = cache.get_stats()

print('Cache Performance:')
print(f'  Hit rate: {stats[\"detailed_stats\"][\"hit_rate\"]:.2%}')
print(f'  Total requests: {stats[\"detailed_stats\"][\"total_requests\"]}')
print(f'  Cache hits: {stats[\"detailed_stats\"][\"cache_hits\"]}')
print(f'  Cache misses: {stats[\"detailed_stats\"][\"cache_misses\"]}')
print(f'  Download success rate: {stats[\"detailed_stats\"][\"download_successes\"] / max(stats[\"detailed_stats\"][\"download_attempts\"], 1):.2%}')

print('Memory Usage:')
print(f'  Memory cache: {stats[\"memory_cache\"][\"size_mb\"]:.1f}MB / {stats[\"memory_cache\"][\"max_size_mb\"]}MB')
print(f'  Disk cache: {stats[\"disk_cache\"][\"size_gb\"]:.1f}GB / {stats[\"disk_cache\"][\"max_size_gb\"]}GB')
"
```

### Network Diagnostics

```bash
# Test network performance
python -c "
import time
import requests

def test_download_speed():
    url = 'https://files.rcsb.org/download/1ABC.pdb'
    start = time.time()
    resp = requests.get(url)
    elapsed = time.time() - start
    size_mb = len(resp.content) / (1024*1024)
    speed_mbps = (size_mb * 8) / elapsed
    print(f'Download speed: {speed_mbps:.1f} Mbps')
    print(f'Latency: {elapsed:.2f}s for {size_mb:.2f}MB')

test_download_speed()
"
```

### System Resource Monitoring

```bash
# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% MEM=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}') CACHE=$(du -sh $STREAMING_CACHE_DIR 2>/dev/null | cut -f1)"
    sleep 30
done
EOF

chmod +x monitor_training.sh
./monitor_training.sh &
```

## Recovery Procedures

### Corrupted Training State

**Symptoms:**
- Training crashes on restart
- Inconsistent checkpoint files
- Model loading errors

**Recovery Steps:**

1. **Backup Current State:**
```bash
cp -r checkpoints checkpoints.backup.$(date +%Y%m%d_%H%M%S)
```

2. **Validate Checkpoint:**
```python
import torch
try:
    checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Keys: {list(checkpoint.keys())}")
    print("Checkpoint appears valid")
except Exception as e:
    print(f"Checkpoint corrupted: {e}")
```

3. **Rollback to Previous Checkpoint:**
```bash
# List available checkpoints
ls -la checkpoints/*.pt

# Use earlier checkpoint
cp checkpoints/checkpoint_epoch_50.pt checkpoints/latest.pt
```

### Cache Recovery

**Complete Cache Rebuild:**
```bash
# Backup cache metadata
cp $STREAMING_CACHE_DIR/cache_metadata.json $STREAMING_CACHE_DIR/cache_metadata.backup

# Clear corrupted cache
rm -rf $STREAMING_CACHE_DIR/*

# Restart training (cache will rebuild automatically)
```

**Partial Cache Recovery:**
```python
# Validate and repair cache
from hybrid.data.pdb_cache import PDBCache
cache = PDBCache(cache_dir)

# This will scan and rebuild cache index
stats = cache.get_stats()
print(f"Recovered {stats['disk_cache']['file_count']} cache files")
```

### Job Recovery After Node Failure

**SLURM Job Recovery:**
```bash
# Check if job was requeued
scontrol show job $SLURM_JOB_ID

# Manually requeue if needed
scontrol requeue $SLURM_JOB_ID

# Check for partial results
ls -la $SCRATCH_DIR/checkpoints/
```

**Data Recovery:**
```bash
# Rsync partial results
rsync -av $SCRATCH_DIR/checkpoints/ ./recovered_checkpoints/

# Verify checkpoint integrity
python -c "
import torch
import glob
for ckpt in glob.glob('recovered_checkpoints/*.pt'):
    try:
        data = torch.load(ckpt, map_location='cpu')
        print(f'{ckpt}: OK (epoch {data.get(\"epoch\", \"?\")})') 
    except Exception as e:
        print(f'{ckpt}: CORRUPTED ({e})')
"
```

### Emergency Shutdown Procedures

**Graceful Training Termination:**
```bash
# Send SIGTERM to training process
kill -TERM $TRAINING_PID

# Wait for checkpoint saving (up to 5 minutes)
timeout 300 tail --pid=$TRAINING_PID -f /dev/null

# Force kill if needed
kill -KILL $TRAINING_PID
```

**Data Preservation:**
```bash
# Save critical data immediately
rsync -av $STREAMING_CACHE_DIR/ $BACKUP_DIR/emergency_cache/
rsync -av checkpoints/ $BACKUP_DIR/emergency_checkpoints/

# Save configuration and logs
cp config_streaming.json $BACKUP_DIR/
cp *.out *.err $BACKUP_DIR/
```

## Contact and Support

### Internal Resources

1. **Documentation:**
   - User Guide: `documentation/streaming_user_guide.md`
   - Performance Guide: `hybrid/training/a100_performance_tuning_guide.md`
   - Configuration Reference: `hybrid/training/config_streaming.json`

2. **Example Scripts:**
   - Training Script: `train_hybrid_streaming.sh`
   - Validation Script: `hybrid/training/validate_config.py`
   - Test Scripts: `test_*.py`

### External Resources

1. **Harvard RC Support:**
   - Email: rchelp@harvard.edu
   - Documentation: https://docs.rc.fas.harvard.edu/
   - Office Hours: Check RC website for current schedule

2. **RCSB PDB Support:**
   - API Documentation: https://search.rcsb.org/index.html#search-api
   - Status Page: https://status.rcsb.org/
   - Contact: info@rcsb.org

### Creating Support Requests

**Include in Support Request:**

1. **Environment Information:**
```bash
# System information
uname -a
module list 2>&1
nvidia-smi
python --version

# Configuration
cat config_streaming.json

# Recent logs
tail -50 train_streaming_hybrid_*.err
```

2. **Error Details:**
   - Full error message and stack trace
   - Steps to reproduce the issue
   - Configuration file used
   - Recent changes made

3. **Resource Usage:**
```bash
# Current resource usage
free -h
df -h
nvidia-smi
squeue -u $USER
```

**For Performance Issues:**
   - Include cache statistics
   - GPU utilization logs
   - Training progress metrics
   - System monitoring data

**For Cache Issues:**
   - Cache directory contents
   - Cache statistics output
   - Network connectivity test results
   - Disk space information

---

## Quick Reference

### Emergency Commands
```bash
# Kill training immediately
kill -KILL $(pgrep -f "train_energy.py")

# Clear all cache
rm -rf $STREAMING_CACHE_DIR/*

# Check disk space
df -h $(dirname $STREAMING_CACHE_DIR)

# Test basic functionality
python -c "from hybrid.data.streaming_dataset import StreamingProteinDataset; print('Import OK')"

# Validate configuration
python hybrid/training/validate_config.py config_streaming.json
```

### Diagnostic Commands
```bash
# System health check
nvidia-smi && free -h && df -h

# Network test
curl -s https://files.rcsb.org/download/1ABC.pdb > /dev/null && echo "Network OK"

# Cache analysis
python -c "from hybrid.data.pdb_cache import PDBCache; print(PDBCache('$STREAMING_CACHE_DIR').get_stats())"
```

This troubleshooting guide covers the most common issues encountered with the streaming PDB system. For issues not covered here, please refer to the detailed logs and consider creating a support request with the information gathering templates provided above.