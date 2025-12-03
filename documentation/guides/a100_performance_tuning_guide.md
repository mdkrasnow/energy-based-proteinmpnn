# A100 Performance Tuning Guidelines for Energy-Based ProteinMPNN Streaming Training

This document provides comprehensive guidelines for optimizing streaming training performance on Harvard's A100 cluster (gpu_requeue partition).

## Table of Contents
1. [Hardware Overview](#hardware-overview)
2. [Optimal Configuration Parameters](#optimal-configuration-parameters)
3. [Memory Management](#memory-management)
4. [I/O and Streaming Optimization](#io-and-streaming-optimization)
5. [Training Optimization](#training-optimization)
6. [Production Deployment Best Practices](#production-deployment-best-practices)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Troubleshooting](#troubleshooting)
9. [Performance Benchmarks](#performance-benchmarks)
10. [24-Hour Unattended Training](#24-hour-unattended-training)

## Hardware Overview

### A100 80GB Specifications
- **GPU Memory**: 80GB HBM2e 
- **Memory Bandwidth**: 2TB/s
- **Compute Capability**: 8.0 (3rd gen Tensor Cores)
- **Peak FP16 Performance**: 312 TFLOPS (with sparsity)
- **Peak FP32 Performance**: 19.5 TFLOPS

### Harvard Cluster Environment
- **CPU Cores**: 16 cores per node (typically 2.5GHz)
- **System Memory**: Up to 1TB per node
- **Storage**: `/n/netscratch` high-performance parallel filesystem
- **Network**: High-speed InfiniBand interconnect

## Optimal Configuration Parameters

### Recommended Settings for Production

```json
{
  "training": {
    "batch_size": 16,              // Optimal for 80GB VRAM
    "gradient_accumulation_steps": 4,
    "num_workers": 8,              // Half the CPU cores for I/O
    "mixed_precision": true        // Essential for A100 performance
  },
  "streaming": {
    "max_memory_mb": 5120,         // 5GB cache for streaming
    "num_workers": 8,              // Match training workers
    "concurrent_downloads": 8,      // Avoid contention
    "prefetch_factor": 6,          // Higher prefetch for large cache
    "connection_pool_size": 32     // Handle many simultaneous requests
  },
  "hardware_optimization": {
    "gpu_type": "a100_80gb",
    "memory_fraction": 0.9,        // Use 90% of GPU memory
    "mixed_precision": "fp16",     // Leverage Tensor Cores
    "tensor_cores": true,          // Enable Tensor Core optimization
    "cudnn_benchmark": true,       // Optimize convolutions
    "pin_memory": true,            // Faster CPU-GPU transfers
    "non_blocking_transfer": true  // Overlap transfers with compute
  }
}
```

### Scaling Guidelines

| Batch Size | GPU Memory Usage | Effective Batch* | Throughput | Recommendations |
|------------|------------------|------------------|------------|-----------------|
| 8          | ~45GB           | 32               | Medium     | Conservative, safe |
| 16         | ~55GB           | 64               | **High**   | **Recommended** |
| 24         | ~65GB           | 96               | High       | Aggressive, monitor memory |
| 32         | ~75GB           | 128              | Lower      | Risk of OOM |

*With gradient_accumulation_steps=4

## Memory Management

### GPU Memory Optimization

1. **Use Mixed Precision (FP16)**
   ```python
   # Enables automatic mixed precision
   "mixed_precision": true
   "hardware_optimization": {
     "mixed_precision": "fp16",
     "tensor_cores": true
   }
   ```

2. **Memory Fraction Settings**
   ```json
   "memory_fraction": 0.9  // Reserve 8GB for system overhead
   ```

3. **Gradient Checkpointing** (if needed)
   ```json
   "training": {
     "gradient_checkpointing": true  // Trade compute for memory
   }
   ```

### System Memory Optimization

1. **Streaming Cache Configuration**
   ```json
   "streaming": {
     "max_memory_mb": 5120,        // 5GB for streaming cache
     "cache_dir": "/n/netscratch/ydu_lab/Lab/mkrasnow/streaming_cache"
   },
   "cache_config": {
     "pdb_cache": {
       "max_memory_mb": 4096,      // 4GB for PDB structures
       "compression": true,         // Save memory with compression
       "preload_popular": true     // Cache frequently used structures
     }
   }
   ```

2. **Memory Pinning**
   ```json
   "performance_cache": {
     "pin_memory": true,           // Faster CPU-GPU transfers
     "prefetch_to_device": true,   // Overlap data movement
     "async_loading": true         // Non-blocking I/O
   }
   ```

## I/O and Streaming Optimization

### Network and Download Optimization

1. **Connection Pool Management**
   ```json
   "streaming": {
     "connection_pool_size": 32,    // Handle many concurrent requests
     "download_timeout_seconds": 300,
     "retry_attempts": 3,
     "chunk_size_mb": 10           // Efficient download chunks
   }
   ```

2. **Rate Limiting**
   ```json
   "data_sources": [{
     "rate_limit_per_second": 20   // Avoid overwhelming RCSB API
   }]
   ```

### File System Optimization

1. **Use netscratch for all temporary data**
   ```bash
   # All paths should use netscratch
   /n/netscratch/ydu_lab/Lab/mkrasnow/streaming_cache
   /n/netscratch/ydu_lab/Lab/mkrasnow/streaming_logs
   ```

2. **Cache Warming Strategy**
   ```json
   "cache_config": {
     "pdb_cache": {
       "cache_warming_enabled": true,  // Pre-populate cache
       "eviction_policy": "lru"        // Keep frequently used data
     }
   }
   ```

### Data Loading Optimization

1. **Worker Configuration**
   ```json
   "training": {
     "num_workers": 8,             // Balance I/O and CPU usage
     "prefetch_factor": 4          // Maintain data pipeline
   }
   ```

2. **Async Loading**
   ```json
   "debug": {
     "profile_data_loading": true  // Monitor data loading performance
   }
   ```

## Training Optimization

### Learning Rate and Batch Size

1. **Batch Size Scaling**
   ```python
   # Scale learning rate with effective batch size
   base_lr = 1e-4
   effective_batch = batch_size * gradient_accumulation_steps
   scaled_lr = base_lr * sqrt(effective_batch / 16)
   ```

2. **Gradient Accumulation**
   ```json
   "training": {
     "gradient_accumulation_steps": 4,  // Effective batch size = 64
     "max_grad_norm": 1.0              // Prevent gradient explosion
   }
   ```

### Model Architecture Tuning

1. **Energy Head Sizing**
   ```json
   "model": {
     "energy_head": {
       "hidden_dim": 512,           // Balance capacity and speed
       "num_layers": 3,             // Optimal depth for A100
       "dropout": 0.1,              // Regularization
       "use_batch_norm": true       // Stable training
     }
   }
   ```

2. **Sequence Representation**
   ```json
   "sequence_repr": {
     "temperature_schedule": [1.0, 0.5, 0.1],  // Annealing schedule
     "min_temperature": 0.001,
     "max_temperature": 10.0
   }
   ```

### Loss Function Optimization

1. **Dynamic Loss Weighting**
   ```json
   "loss": {
     "dynamic_weighting": {
       "enabled": true,
       "adaptation_rate": 0.01,     // Adapt to training dynamics
       "min_weight": 0.1,
       "max_weight": 5.0
     }
   }
   ```

## Production Deployment Best Practices

### Harvard Cluster Integration

#### SLURM Configuration Optimization

1. **Resource Allocation Strategy**
   ```bash
   # Optimal SLURM parameters for A100 cluster
   #SBATCH -p gpu_requeue           # Dedicated A100 partition
   #SBATCH --constraint=a100        # Ensure A100 hardware
   #SBATCH --gres=gpu:1            # Single A100 GPU (80GB)
   #SBATCH -c 16                   # 16 CPU cores
   #SBATCH --mem=250G              # 250GB system memory
   #SBATCH -t 24:00:00             # 24-hour time limit
   ```

2. **Production-Ready SLURM Features**
   ```bash
   # Enhanced SLURM options for unattended training
   #SBATCH --requeue               # Auto-requeue on node failure
   #SBATCH --signal=SIGUSR1@90     # 90-second graceful shutdown
   #SBATCH --open-mode=append      # Resume logging on restart
   #SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
   ```

#### Network Storage Optimization

1. **Netscratch Usage Patterns**
   ```bash
   # Optimal directory structure
   BASE_DIR="/n/netscratch/ydu_lab/Lab/$USER"
   STREAMING_CACHE="$BASE_DIR/streaming_cache"     # Hot data cache
   CHECKPOINT_DIR="$BASE_DIR/checkpoints"          # Model snapshots
   LOG_DIR="$BASE_DIR/logs"                        # Monitoring data
   ARCHIVE_DIR="$BASE_DIR/completed_runs"          # Finished experiments
   ```

2. **Cache Management Strategy**
   ```json
   {
     "cache_config": {
       "pdb_cache": {
         "max_memory_mb": 5120,        // 5GB in-memory cache
         "max_disk_gb": 100,           // 100GB persistent cache
         "eviction_policy": "lru",     // Least recently used
         "cleanup_threshold": 0.9      // Clean when 90% full
       }
     }
   }
   ```

### Signal Handling and Graceful Shutdown

1. **SLURM Signal Integration**
   ```python
   import signal
   import torch
   
   def graceful_shutdown_handler(signum, frame):
       """Handle SLURM shutdown signals gracefully"""
       print(f"Received signal {signum}, initiating graceful shutdown...")
       
       # Save emergency checkpoint
       torch.save({
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'epoch': current_epoch,
           'step': current_step,
           'emergency_save': True
       }, 'emergency_checkpoint.pt')
       
       # Clean up resources
       cleanup_streaming_cache()
       sys.exit(0)
   
   # Register signal handler
   signal.signal(signal.SIGUSR1, graceful_shutdown_handler)
   ```

### Checkpoint Strategy for Long Training

1. **Multi-Level Checkpointing**
   ```json
   {
     "checkpointing": {
       "emergency_save_frequency": 300,      // Every 5 minutes
       "regular_save_frequency": 1800,       // Every 30 minutes
       "best_model_frequency": 3600,         // Every hour
       "archive_frequency": 21600            // Every 6 hours
     }
   }
   ```

2. **Checkpoint Recovery Logic**
   ```python
   def load_checkpoint_with_fallback(checkpoint_dir):
       """Load checkpoint with multiple fallback options"""
       checkpoint_files = [
           'latest.pt',
           'emergency_checkpoint.pt',
           'best_model.pt'
       ]
       
       for checkpoint_file in checkpoint_files:
           path = os.path.join(checkpoint_dir, checkpoint_file)
           if os.path.exists(path):
               try:
                   checkpoint = torch.load(path)
                   print(f"Loaded checkpoint: {checkpoint_file}")
                   return checkpoint
               except Exception as e:
                   print(f"Failed to load {checkpoint_file}: {e}")
                   continue
       
       print("No valid checkpoint found, starting from scratch")
       return None
   ```

### Resource Monitoring and Alerting

1. **Real-Time Resource Tracking**
   ```python
   import psutil
   import torch
   import nvidia_ml_py3 as nvml
   
   def monitor_resources():
       """Comprehensive resource monitoring"""
       nvml.nvmlInit()
       handle = nvml.nvmlDeviceGetHandleByIndex(0)
       
       metrics = {
           'timestamp': time.time(),
           'cpu_percent': psutil.cpu_percent(),
           'memory_percent': psutil.virtual_memory().percent,
           'disk_usage_percent': psutil.disk_usage('/').percent,
           'gpu_memory_used': torch.cuda.memory_allocated(0),
           'gpu_memory_cached': torch.cuda.memory_reserved(0),
           'gpu_utilization': nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
           'gpu_temperature': nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
       }
       
       return metrics
   ```

2. **Automated Performance Alerts**
   ```python
   def check_performance_thresholds(metrics):
       """Alert on performance issues"""
       alerts = []
       
       if metrics['gpu_utilization'] < 70:
           alerts.append(f"Low GPU utilization: {metrics['gpu_utilization']}%")
       
       if metrics['memory_percent'] > 95:
           alerts.append(f"High memory usage: {metrics['memory_percent']}%")
       
       if metrics['gpu_temperature'] > 85:
           alerts.append(f"High GPU temperature: {metrics['gpu_temperature']}°C")
       
       return alerts
   ```

### Failure Recovery and Resilience

1. **Automatic Restart Strategy**
   ```bash
   # In SLURM script - attempt automatic recovery
   RESTART_ATTEMPTS=3
   CURRENT_ATTEMPT=0
   
   while [ $CURRENT_ATTEMPT -lt $RESTART_ATTEMPTS ]; do
       echo "Training attempt $((CURRENT_ATTEMPT + 1))/$RESTART_ATTEMPTS"
       
       # Run training
       python hybrid/training/train_energy.py --config "$CONFIG" \
           --resume_from_checkpoint "$CHECKPOINT_DIR/latest.pt"
       
       TRAIN_EXIT=$?
       
       if [ $TRAIN_EXIT -eq 0 ]; then
           echo "Training completed successfully"
           break
       else
           echo "Training failed (exit code: $TRAIN_EXIT)"
           CURRENT_ATTEMPT=$((CURRENT_ATTEMPT + 1))
           
           # Wait before retry
           sleep 60
       fi
   done
   ```

2. **Data Integrity Verification**
   ```python
   def verify_checkpoint_integrity(checkpoint_path):
       """Verify checkpoint can be loaded and used"""
       try:
           checkpoint = torch.load(checkpoint_path, map_location='cpu')
           required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
           
           for key in required_keys:
               if key not in checkpoint:
                   return False, f"Missing key: {key}"
           
           # Test model loading
           model = create_model()
           model.load_state_dict(checkpoint['model_state_dict'])
           
           return True, "Checkpoint is valid"
           
       except Exception as e:
           return False, f"Checkpoint corrupted: {e}"
   ```

### Production Configuration Validation

1. **Pre-Training Validation**
   ```python
   def validate_production_config(config_path):
       """Comprehensive configuration validation"""
       with open(config_path, 'r') as f:
           config = json.load(f)
       
       validation_results = {
           'valid': True,
           'warnings': [],
           'errors': []
       }
       
       # Check hardware optimization settings
       if not config.get('hardware_optimization', {}).get('mixed_precision'):
           validation_results['warnings'].append("Mixed precision not enabled - performance impact expected")
       
       # Check batch size for A100
       batch_size = config.get('training', {}).get('batch_size', 0)
       if batch_size < 16:
           validation_results['warnings'].append(f"Batch size {batch_size} may underutilize A100")
       elif batch_size > 32:
           validation_results['warnings'].append(f"Batch size {batch_size} may cause OOM on A100")
       
       # Check streaming configuration
       cache_mb = config.get('streaming', {}).get('max_memory_mb', 0)
       if cache_mb < 4096:
           validation_results['errors'].append("Streaming cache too small for production")
       
       # Check paths exist and are writable
       cache_dir = config.get('streaming', {}).get('cache_dir')
       if not os.access(cache_dir, os.W_OK):
           validation_results['errors'].append(f"Cache directory not writable: {cache_dir}")
       
       if validation_results['errors']:
           validation_results['valid'] = False
       
       return validation_results
   ```

2. **Hardware Compatibility Check**
   ```python
   def verify_a100_compatibility():
       """Verify A100-specific features are available"""
       checks = {}
       
       # Check CUDA compute capability
       if torch.cuda.is_available():
           props = torch.cuda.get_device_properties(0)
           checks['device_name'] = props.name
           checks['compute_capability'] = f"{props.major}.{props.minor}"
           checks['memory_gb'] = props.total_memory / (1024**3)
           checks['tensor_cores'] = props.major >= 7
           checks['a100_features'] = 'A100' in props.name and props.major >= 8
       
       return checks
   ```

## Monitoring and Debugging

### Performance Monitoring

1. **System Resource Monitoring**
   ```json
   "monitoring": {
     "metrics": {
       "track_cache_stats": true,
       "track_download_stats": true,
       "track_memory_usage": true
     }
   }
   ```

2. **Training Metrics**
   ```json
   "training": {
     "log_frequency": 25,           // Frequent logging for monitoring
     "eval_frequency": 250,         // Regular evaluation
     "save_frequency": 50           // Regular checkpoints
   }
   ```

### Debug Configuration

1. **Enable Profiling**
   ```json
   "debug": {
     "profile_data_loading": true,
     "validate_samples": true,
     "memory_profiling": true,
     "cache_statistics": true
   }
   ```

2. **TensorBoard Integration**
   ```json
   "monitoring": {
     "tensorboard": {
       "enabled": true,
       "log_dir": "/n/netscratch/ydu_lab/Lab/mkrasnow/streaming_logs/tensorboard"
     }
   }
   ```

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `16 → 12 → 8`
2. Increase gradient accumulation: `4 → 6 → 8`
3. Enable gradient checkpointing
4. Check for memory leaks in data loading

#### 2. Slow Data Loading

**Symptoms:**
- GPU utilization < 80%
- High CPU wait time
- Slow iteration times

**Solutions:**
1. Increase data workers: `8 → 12`
2. Increase prefetch factor: `6 → 8`
3. Check network connectivity to netscratch
4. Enable cache warming

#### 3. Network Download Failures

**Symptoms:**
```
Connection timeout
HTTP 429 (rate limited)
```

**Solutions:**
1. Reduce concurrent downloads: `16 → 8`
2. Increase timeout: `300 → 600`
3. Enable retry with backoff
4. Check RCSB API status

#### 4. Cache Thrashing

**Symptoms:**
- Low cache hit rate
- Frequent disk I/O
- Inconsistent performance

**Solutions:**
1. Increase cache size: `5GB → 8GB`
2. Adjust eviction policy
3. Enable cache warming
4. Monitor cache statistics

### Performance Debugging Commands

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Monitor system resources  
htop

# Check disk I/O
iotop

# Monitor network
iftop

# TensorBoard for training metrics
tensorboard --logdir /n/netscratch/ydu_lab/Lab/mkrasnow/streaming_logs/tensorboard
```

## Performance Benchmarks

### Expected Performance Targets

| Metric | Target Value | Excellent | Good | Needs Improvement |
|--------|-------------|-----------|------|------------------|
| GPU Utilization | >85% | >90% | 80-90% | <80% |
| Memory Utilization | 70-90% | 80-90% | 70-80% | <70% or >95% |
| Samples/second | >50 | >100 | 50-100 | <50 |
| Cache Hit Rate | >80% | >90% | 80-90% | <80% |
| Network Bandwidth | >100MB/s | >500MB/s | 100-500MB/s | <100MB/s |

### Baseline Configuration Performance

**Standard Configuration:**
- Batch Size: 16
- Workers: 8  
- Mixed Precision: FP16
- Expected Throughput: ~80 samples/second

**Optimized Configuration:**
- All recommendations applied
- Expected Throughput: ~120 samples/second
- GPU Memory Usage: ~55GB (69% of 80GB)

### Scaling Analysis

**Single A100 Performance:**
- Optimal batch size: 16-24
- Peak throughput: ~120 samples/second
- Training time (100k steps): ~20 hours

**Memory vs Batch Size Trade-off:**
```
Batch 8:  45GB GPU, 40 samples/sec  (Conservative)
Batch 16: 55GB GPU, 80 samples/sec  (Recommended)
Batch 24: 65GB GPU, 100 samples/sec (Aggressive)
Batch 32: 75GB GPU, 95 samples/sec  (Risk OOM)
```

## Configuration Validation

Always validate your configuration before long training runs:

```bash
# Validate configuration
python hybrid/training/validate_config.py config_streaming.json --verbose

# Expected output for optimal configuration:
# Status: ✅ VALID
# Score: 100.0/100
# GPU Memory: 3600MB (4.4%)
# Effective Batch Size: 64
```

## Summary

For optimal A100 streaming training performance:

1. **Use batch_size=16** with gradient_accumulation_steps=4
2. **Enable mixed precision** (FP16) and Tensor Cores
3. **Allocate 5GB** for streaming cache in netscratch
4. **Use 8 workers** for both training and streaming
5. **Monitor GPU utilization** and adjust if <85%
6. **Validate configuration** before production runs
7. **Enable comprehensive monitoring** for debugging

Following these guidelines should achieve >85% GPU utilization and >80 samples/second throughput on Harvard's A100 cluster.

## 24-Hour Unattended Training

### Pre-Flight Checklist

Before launching a 24-hour production training run, complete this checklist:

#### Environment Verification
- [ ] **A100 Hardware Confirmed**: `nvidia-smi` shows A100 GPU
- [ ] **CUDA Version Compatible**: CUDA 12.x or compatible
- [ ] **Python Environment**: All dependencies installed
- [ ] **Netscratch Access**: Write permissions to `/n/netscratch/ydu_lab/Lab/$USER`
- [ ] **Disk Space**: >150GB available in netscratch
- [ ] **Network Connectivity**: Can access RCSB PDB API

#### Configuration Validation
- [ ] **Config File Valid**: `python hybrid/training/validate_config.py config_streaming.json`
- [ ] **Batch Size Optimal**: Set to 16 for A100 80GB
- [ ] **Mixed Precision Enabled**: FP16 training configured
- [ ] **Checkpointing Configured**: Save frequency ≤ 30 minutes
- [ ] **Monitoring Enabled**: TensorBoard and system monitoring active

#### Recovery Preparation
- [ ] **Signal Handlers**: SIGUSR1 graceful shutdown configured
- [ ] **Checkpoint Resume**: Test loading from `latest.pt`
- [ ] **Email Notifications**: SLURM mail alerts configured
- [ ] **Emergency Contacts**: Monitoring dashboard accessible

### Launch Procedure

1. **Submit Training Job**
   ```bash
   # Navigate to project directory
   cd /path/to/energy-based-proteinmpnn
   
   # Submit with production configuration
   sbatch train_hybrid_streaming.sh
   ```

2. **Monitor Job Status**
   ```bash
   # Check job status
   squeue -u $USER
   
   # View real-time logs
   tail -f train_streaming_hybrid_[JOB_ID].out
   
   # Check resource utilization
   ssh [compute_node] "nvidia-smi; htop"
   ```

3. **Track Progress Indicators**
   ```bash
   # Monitor checkpoints
   ls -la /n/netscratch/ydu_lab/Lab/$USER/streaming_results_[JOB_ID]/checkpoints/
   
   # Check training metrics
   tensorboard --logdir /n/netscratch/ydu_lab/Lab/$USER/streaming_logs/tensorboard
   ```

### Monitoring Dashboard

#### Key Metrics to Track

1. **Training Progress**
   - Steps/epoch completed
   - Loss convergence trends
   - Validation metrics
   - Checkpoint timestamps

2. **Resource Utilization**
   - GPU utilization (target: >85%)
   - GPU memory usage (target: 60-80%)
   - CPU usage (target: 60-80%)
   - Network I/O for streaming

3. **Data Pipeline Health**
   - Cache hit rate (target: >80%)
   - Download success rate (target: >95%)
   - Streaming throughput
   - Queue depths

#### Automated Monitoring Script

```bash
#!/bin/bash
# monitor_training.sh - Run this in a separate terminal

JOB_ID=$1
RESULTS_DIR="/n/netscratch/ydu_lab/Lab/$USER/streaming_results_${JOB_ID}"
MONITOR_LOG="${RESULTS_DIR}/logs/system_monitor.jsonl"

echo "Monitoring training job: $JOB_ID"
echo "Results directory: $RESULTS_DIR"

while true; do
    clear
    echo "=== Training Monitoring Dashboard ==="
    echo "Time: $(date)"
    echo "Job ID: $JOB_ID"
    echo ""
    
    # Check job status
    JOB_STATUS=$(squeue -h -j $JOB_ID -o "%T" 2>/dev/null || echo "NOT_FOUND")
    echo "SLURM Status: $JOB_STATUS"
    
    # Check checkpoint count
    if [ -d "${RESULTS_DIR}/checkpoints" ]; then
        CHECKPOINT_COUNT=$(find "${RESULTS_DIR}/checkpoints" -name "*.pt" | wc -l)
        LATEST_CHECKPOINT=$(find "${RESULTS_DIR}/checkpoints" -name "*.pt" -exec stat -c "%Y %n" {} \; 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | xargs basename)
        echo "Checkpoints: $CHECKPOINT_COUNT (Latest: $LATEST_CHECKPOINT)"
    fi
    
    # Show recent GPU utilization
    if [ -f "$MONITOR_LOG" ]; then
        echo ""
        echo "Recent GPU Metrics:"
        tail -5 "$MONITOR_LOG" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        gpu_util = data.get('gpu_utilization', 'N/A')
        gpu_mem = data.get('gpu_memory_allocated', 0) / 1e9
        print(f'  GPU: {gpu_util}% util, {gpu_mem:.1f}GB memory')
    except:
        continue
"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    sleep 30
done
```

### Common Issues and Recovery

#### Issue: Training Stalled
**Symptoms**: No new checkpoints for >2 hours
**Recovery**:
1. Check job status: `squeue -j $JOB_ID`
2. SSH to compute node: `ssh [node]`
3. Check GPU utilization: `nvidia-smi`
4. Review error logs: `tail -100 train_streaming_hybrid_$JOB_ID.err`
5. If needed, kill and restart with checkpoint resume

#### Issue: GPU Memory Error
**Symptoms**: CUDA OOM errors in logs
**Recovery**:
1. Reduce batch size in config: `16 → 12 → 8`
2. Increase gradient accumulation: `4 → 6 → 8`
3. Restart training with new config

#### Issue: Network/Download Failures
**Symptoms**: High rate of download failures
**Recovery**:
1. Check RCSB API status
2. Reduce concurrent downloads: `8 → 4`
3. Increase timeout: `300 → 600` seconds
4. Enable more aggressive retry policy

#### Issue: Disk Space Exhausted
**Symptoms**: Write errors to netscratch
**Recovery**:
1. Clean old cache: `find $CACHE_DIR -mtime +7 -delete`
2. Archive completed runs
3. Reduce cache size in config
4. Request additional quota if needed

### Post-Training Checklist

After 24-hour training completion:

- [ ] **Verify Completion**: Check final logs and exit status
- [ ] **Model Validation**: Load and test best_model.pt
- [ ] **Performance Review**: Analyze throughput and utilization
- [ ] **Results Archival**: Copy to permanent storage
- [ ] **Cleanup**: Remove large temporary files
- [ ] **Documentation**: Update training log with results

### Emergency Procedures

#### Graceful Job Termination
```bash
# Request graceful shutdown (90 seconds notice)
scancel -s USR1 $JOB_ID

# Force termination if needed
scancel $JOB_ID
```

#### Emergency Checkpoint Recovery
```bash
# Find most recent checkpoint
find /n/netscratch/ydu_lab/Lab/$USER -name "*.pt" -exec stat -c "%Y %n" {} \; | sort -n | tail -5

# Test checkpoint loading
python -c "
import torch
checkpoint = torch.load('path/to/checkpoint.pt', map_location='cpu')
print(f'Epoch: {checkpoint.get(\"epoch\", \"unknown\")}')
print(f'Step: {checkpoint.get(\"step\", \"unknown\")}')
print('Checkpoint appears valid')
"
```

This comprehensive guide ensures reliable 24-hour unattended training runs on Harvard's A100 cluster with proper monitoring, recovery procedures, and production-grade reliability.