#!/bin/bash
#SBATCH -J train_streaming_hybrid           # Job name
#SBATCH -p gpu_requeue                      # Use GPU requeue partition for production
#SBATCH --constraint=a100                   # Require A100 GPUs
#SBATCH --account=ydu_lab                   # Your lab account
#SBATCH --gres=gpu:1                        # 1 A100 GPU (80GB VRAM)
#SBATCH -c 16                               # 16 CPU cores (match streaming workers)
#SBATCH -t 24:00:00                         # 24 hours for production streaming training
#SBATCH --mem=250G                          # 250 GB RAM for large-scale streaming
#SBATCH -o train_streaming_hybrid_%j.out    # STDOUT file
#SBATCH -e train_streaming_hybrid_%j.err    # STDERR file
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT # Comprehensive email notifications
#SBATCH --mail-user=${SLURM_MAIL_USER}
#SBATCH --requeue                           # Allow requeuing on node failure
#SBATCH --signal=SIGUSR1@90                 # Send signal 90 seconds before time limit
#SBATCH --open-mode=append                  # Append to output files on restart

# ------------------------------------------------------------------------------
# Validate environment and configuration
# ------------------------------------------------------------------------------

# Check if SLURM_MAIL_USER is set, warn if not
if [ -z "${SLURM_MAIL_USER:-}" ]; then
    echo "⚠️  WARNING: SLURM_MAIL_USER environment variable not set"
    echo "   Email notifications will not be sent"
    echo "   Set SLURM_MAIL_USER=your-email@domain.com to receive notifications"
    echo ""
fi

echo "=============================================="
echo "  Streaming Hybrid ProteinMPNN Training Job"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# ------------------------------------------------------------------------------
# Setup signal handling for graceful shutdown
# ------------------------------------------------------------------------------

GRACEFUL_SHUTDOWN=false
TRAINING_PID=""

graceful_shutdown() {
    echo ""
    echo "⚠️  Received shutdown signal (90 seconds before time limit)"
    echo "Initiating graceful shutdown..."
    GRACEFUL_SHUTDOWN=true
    
    if [ -n "$TRAINING_PID" ]; then
        echo "Sending SIGTERM to training process (PID: $TRAINING_PID)"
        kill -TERM $TRAINING_PID 2>/dev/null || true
        
        # Wait up to 60 seconds for graceful shutdown
        for i in {1..60}; do
            if ! kill -0 $TRAINING_PID 2>/dev/null; then
                echo "Training process shut down gracefully"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 $TRAINING_PID 2>/dev/null; then
            echo "Force killing training process"
            kill -KILL $TRAINING_PID 2>/dev/null || true
        fi
        
        # Verify checkpoint saving after process termination
        echo "Verifying checkpoint integrity..."
        verify_checkpoint_save
    fi
    
    echo "Graceful shutdown completed"
    exit 0
}

verify_checkpoint_save() {
    # Verify that checkpoints were saved successfully with retry logic
    local checkpoint_dir="${CHECKPOINT_DIR:-}"
    local max_retries=3
    local retry_count=0
    
    if [ -z "$checkpoint_dir" ] || [ ! -d "$checkpoint_dir" ]; then
        echo "⚠️  Checkpoint directory not found: $checkpoint_dir"
        return 1
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Checkpoint verification attempt $((retry_count + 1))/$max_retries..."
        
        # Check if latest checkpoint exists and is recent
        local latest_checkpoint="$checkpoint_dir/latest.pt"
        if [ -f "$latest_checkpoint" ]; then
            local checkpoint_age=$(( $(date +%s) - $(stat -c %Y "$latest_checkpoint" 2>/dev/null || echo 0) ))
            
            # Checkpoint should be recent (within last 10 minutes)
            if [ $checkpoint_age -le 600 ]; then
                # Verify checkpoint integrity using Python
                python3 -c "
import torch
import sys
try:
    checkpoint = torch.load('$latest_checkpoint', map_location='cpu')
    if 'model_state_dict' in checkpoint and 'epoch' in checkpoint:
        print('✓ Checkpoint integrity verified')
        sys.exit(0)
    else:
        print('✗ Checkpoint structure invalid')
        sys.exit(1)
except Exception as e:
    print(f'✗ Checkpoint validation failed: {e}')
    sys.exit(1)
" 2>/dev/null
                
                if [ $? -eq 0 ]; then
                    echo "✓ Final checkpoint saved and verified successfully"
                    return 0
                else
                    echo "✗ Checkpoint integrity check failed"
                fi
            else
                echo "⚠️  Checkpoint is too old (${checkpoint_age}s), may not contain final state"
            fi
        else
            echo "⚠️  No latest checkpoint found"
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "Waiting 5 seconds before retry..."
            sleep 5
        fi
    done
    
    echo "❌ Failed to verify checkpoint save after $max_retries attempts"
    echo "⚠️  Training progress may not be fully preserved"
    return 1
}

# Trap the SIGUSR1 signal sent by SLURM
trap 'graceful_shutdown' SIGUSR1

# Additional error handling
set -euo pipefail  # Exit on any error, undefined var, or pipe failure

error_handler() {
    local exit_code=$?
    echo ""
    echo "❌ FATAL ERROR occurred (exit code: $exit_code)"
    echo "Location: Line $1 in script"
    echo "Command: $BASH_COMMAND"
    echo "Time: $(date)"
    
    # Clean up any running processes
    if [ -n "$TRAINING_PID" ]; then
        kill -TERM $TRAINING_PID 2>/dev/null || true
    fi
    
    # Stop monitoring
    if [ -n "${MONITOR_PID:-}" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    echo "Error handler cleanup completed"
    exit $exit_code
}

trap 'error_handler $LINENO' ERR

# ------------------------------------------------------------------------------
# 1. Configure Harvard netscratch paths for A100 cluster
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"
USER_SCRATCH="/n/netscratch/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${USER_SCRATCH}/streaming_train_${SLURM_JOB_ID}"
STREAMING_CACHE="${USER_SCRATCH}/streaming_cache"
STREAMING_LOGS="${USER_SCRATCH}/streaming_logs"

echo "User scratch root: $USER_SCRATCH"
echo "Job scratch dir:   $JOB_SCRATCH" 
echo "Streaming cache:   $STREAMING_CACHE"
echo "Streaming logs:    $STREAMING_LOGS"

# Create directories with proper permissions
mkdir -p "$USER_SCRATCH" || {
    echo "ERROR: Cannot create user scratch at $USER_SCRATCH"
    exit 1
}

mkdir -p "$JOB_SCRATCH" || {
    echo "ERROR: Cannot create job scratch at $JOB_SCRATCH"
    exit 1
}

mkdir -p "$STREAMING_CACHE" || {
    echo "ERROR: Cannot create streaming cache at $STREAMING_CACHE"
    exit 1
}

mkdir -p "$STREAMING_LOGS" || {
    echo "ERROR: Cannot create streaming logs at $STREAMING_LOGS"
    exit 1
}

cd "$JOB_SCRATCH" || {
    echo "ERROR: Cannot cd to job scratch"
    exit 1
}

echo "Working from: $(pwd)"

# ------------------------------------------------------------------------------
# Validate disk space requirements before proceeding
# ------------------------------------------------------------------------------

echo "Validating disk space requirements..."
validate_disk_space() {
    local required_gb=150
    local netscratch_root="/n/netscratch/${LAB_NAME}/Lab"
    
    if [ ! -d "$netscratch_root" ]; then
        echo "❌ ERROR: Netscratch root directory not accessible: $netscratch_root"
        exit 1
    fi
    
    # Get available space in GB
    local available_kb=$(df "$netscratch_root" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    echo "Disk space check for $netscratch_root:"
    echo "  Required: ${required_gb}GB"
    echo "  Available: ${available_gb}GB"
    
    if [ $available_gb -lt $required_gb ]; then
        echo ""
        echo "❌ INSUFFICIENT DISK SPACE"
        echo "   Required: ${required_gb}GB"
        echo "   Available: ${available_gb}GB"
        echo "   Shortfall: $((required_gb - available_gb))GB"
        echo ""
        echo "Training cannot proceed with insufficient disk space."
        echo "Please free up space in netscratch or contact system administrators."
        echo ""
        echo "Disk usage breakdown:"
        du -sh "$netscratch_root"/* 2>/dev/null | head -10
        exit 1
    fi
    
    echo "✓ Sufficient disk space available (${available_gb}GB >= ${required_gb}GB)"
    
    # Additional check for user quota if applicable
    quota -u "$USER" 2>/dev/null | grep -v "^Disk quotas for user" | grep "/n/netscratch" && {
        echo "⚠️  Note: User quotas are in effect. Monitor usage during training."
    } || true
}

validate_disk_space

# ------------------------------------------------------------------------------
# 2. Clone repository and setup code
# ------------------------------------------------------------------------------

REPO_URL="https://github.com/mdkrasnow/energy-based-proteinmpnn.git"
REPO_DIR="$JOB_SCRATCH/energy-based-proteinmpnn"

echo "Cloning latest codebase from $REPO_URL..."

if [ -d "$REPO_DIR" ]; then
    rm -rf "$REPO_DIR"
fi

git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository"
    exit 1
}

cd "$REPO_DIR" || {
    echo "ERROR: Cannot cd to repository"
    exit 1
}

echo "Repository cloned to: $REPO_DIR"

# ------------------------------------------------------------------------------
# 3. Setup Python environment and dependencies
# ------------------------------------------------------------------------------

echo "Setting up Python environment for A100..."

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

# Install dependencies optimized for A100
echo "Installing optimized dependencies..."
pip install --user -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    numpy pandas matplotlib seaborn \
    tqdm einops accelerate \
    tensorboard scikit-learn \
    biopython biotite mdanalysis \
    requests urllib3 \
    psutil gpustat nvidia-ml-py3

echo "Dependencies installed."

# Verify GPU setup
echo "Verifying A100 GPU configuration..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB memory')
        print(f'Compute capability: {props.major}.{props.minor}')
        print(f'TensorCore support: {props.major >= 7}')
        # Test tensor cores
        if props.major >= 7:
            x = torch.randn(32, 32, device=f'cuda:{i}', dtype=torch.half)
            y = torch.randn(32, 32, device=f'cuda:{i}', dtype=torch.half) 
            z = torch.mm(x, y)
            print(f'TensorCore test: Success - {z.shape}')
"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------------------
# 4. Prepare streaming configuration and validate setup
# ------------------------------------------------------------------------------

echo "Preparing streaming training configuration..."

STREAMING_CONFIG="$JOB_SCRATCH/config_streaming_production.json"
cp "$REPO_DIR/hybrid/training/config_streaming.json" "$STREAMING_CONFIG"

echo "Streaming config copied to: $STREAMING_CONFIG"

# Update paths in configuration to use netscratch
sed -i "s|/n/netscratch/ydu_lab/Lab/mkrasnow|${USER_SCRATCH}|g" "$STREAMING_CONFIG"

echo "Configuration paths updated for current user: $USER"

# Validate configuration
echo "Validating streaming configuration..."
python -c "
import json
try:
    with open('$STREAMING_CONFIG', 'r') as f:
        config = json.load(f)
    print('✓ Configuration is valid JSON')
    
    # Check critical parameters
    batch_size = config['training']['batch_size']
    workers = config['training']['num_workers']
    cache_mb = config['streaming']['max_memory_mb']
    
    print(f'✓ Batch size: {batch_size}')
    print(f'✓ Workers: {workers}')  
    print(f'✓ Cache memory: {cache_mb}MB')
    
    if batch_size != 16:
        print(f'WARNING: Batch size {batch_size} not optimized for A100')
    if cache_mb < 5120:
        print(f'WARNING: Cache memory {cache_mb}MB may be insufficient')
        
except Exception as e:
    print(f'✗ Configuration error: {e}')
    exit(1)
"

# ------------------------------------------------------------------------------
# 5. Pre-warm cache and verify streaming setup
# ------------------------------------------------------------------------------

echo "Pre-warming streaming cache system..."

# Test PDB list manager functionality
echo "Testing PDB streaming infrastructure..."
python -c "
import sys
sys.path.append('$REPO_DIR')

try:
    from hybrid.data.pdb_manager import PDBListManager
    
    manager = PDBListManager(cache_dir='$STREAMING_CACHE')
    print('✓ PDBListManager initialized')
    
    # Test with small sample
    pdb_ids = manager.get_pdb_list(max_structures=100, use_cache=True)
    print(f'✓ Retrieved {len(pdb_ids)} PDB IDs for testing')
    
    if len(pdb_ids) >= 10:
        print('✓ Streaming infrastructure ready')
    else:
        print('WARNING: Limited PDB IDs available')
        
except Exception as e:
    print(f'✗ Streaming test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# ------------------------------------------------------------------------------
# 6. Run streaming training with monitoring
# ------------------------------------------------------------------------------

echo "=============================================="
echo "Starting Production Streaming Training"
echo "=============================================="

CHECKPOINT_DIR="$JOB_SCRATCH/checkpoints"
LOG_DIR="$STREAMING_LOGS/job_${SLURM_JOB_ID}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Set up monitoring
echo "Setting up training monitoring..."

# Create monitoring script
MONITOR_SCRIPT="$JOB_SCRATCH/monitor_training.py"
cat > "$MONITOR_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import os
import time
import psutil
import json
import torch
from datetime import datetime

def monitor_system():
    """Monitor system resources during training"""
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    if torch.cuda.is_available():
        stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9
        stats['gpu_utilization'] = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'unknown'
    
    return stats

if __name__ == "__main__":
    log_file = os.environ.get('MONITOR_LOG', '/tmp/training_monitor.log')
    
    while True:
        try:
            stats = monitor_system()
            with open(log_file, 'a') as f:
                f.write(json.dumps(stats) + '\n')
            time.sleep(60)  # Monitor every minute
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(60)
EOF

# Start background monitoring
export MONITOR_LOG="$LOG_DIR/system_monitor.jsonl"
python "$MONITOR_SCRIPT" &
MONITOR_PID=$!

echo "System monitoring started (PID: $MONITOR_PID)"

# Start the main training
start_time=$(date +%s)

echo "Launching streaming training with config: $STREAMING_CONFIG"

# Enhanced training with process monitoring
(
    python hybrid/training/train_energy.py \
        --config "$STREAMING_CONFIG" \
        --model_dir "$CHECKPOINT_DIR" \
        --log_dir "$LOG_DIR" \
        --device cuda \
        --streaming_mode \
        --resume_from_checkpoint "$CHECKPOINT_DIR/latest.pt" 2>/dev/null || \
    python hybrid/training/train_energy.py \
        --config "$STREAMING_CONFIG" \
        --model_dir "$CHECKPOINT_DIR" \
        --log_dir "$LOG_DIR" \
        --device cuda \
        --streaming_mode
) &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"

# Monitor training process with enhanced alerting
LAST_CHECKPOINT_TIME=$(date +%s)
CHECKPOINT_TIMEOUT=3600  # 1 hour timeout for checkpoints

while kill -0 $TRAINING_PID 2>/dev/null; do
    sleep 30
    
    # Check if new checkpoints are being created
    if [ -f "$CHECKPOINT_DIR/latest.pt" ]; then
        CHECKPOINT_MTIME=$(stat -c %Y "$CHECKPOINT_DIR/latest.pt" 2>/dev/null || echo 0)
        CURRENT_TIME=$(date +%s)
        
        if [ $CHECKPOINT_MTIME -gt $LAST_CHECKPOINT_TIME ]; then
            LAST_CHECKPOINT_TIME=$CHECKPOINT_MTIME
            echo "✓ Training progress: New checkpoint at $(date)"
        elif [ $((CURRENT_TIME - LAST_CHECKPOINT_TIME)) -gt $CHECKPOINT_TIMEOUT ]; then
            echo "⚠️  WARNING: No new checkpoints in $((CHECKPOINT_TIMEOUT / 60)) minutes"
            echo "   Last checkpoint: $(date -d @$LAST_CHECKPOINT_TIME)"
            # Continue training but log the warning
        fi
    fi
    
    # Check for graceful shutdown request
    if [ "$GRACEFUL_SHUTDOWN" = true ]; then
        echo "Graceful shutdown requested, waiting for training to complete..."
        break
    fi
done

# Wait for training to complete
wait $TRAINING_PID
TRAIN_EXIT=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Training Summary"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Streaming training completed successfully!"
    echo "  Duration: ${duration}s ($(($duration / 3600))h $(($duration % 3600 / 60))m)"
    
    # Check for saved models
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        model_size=$(du -h "$CHECKPOINT_DIR/best_model.pt" | cut -f1)
        echo "  Best model saved: $model_size"
    fi
    
    checkpoint_count=$(find "$CHECKPOINT_DIR" -name "*.pt" | wc -l)
    echo "  Total checkpoints: $checkpoint_count"
    
else
    echo "✗ Training FAILED with exit code: $TRAIN_EXIT"
    echo "  Check error logs in: $LOG_DIR"
fi

# ------------------------------------------------------------------------------
# 7. Training analysis and performance summary
# ------------------------------------------------------------------------------

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "Analyzing training performance..."
    
    ANALYSIS_SCRIPT="$JOB_SCRATCH/analyze_training.py"
    cat > "$ANALYSIS_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path

def analyze_training_performance(log_dir, monitor_log):
    """Analyze training performance metrics"""
    
    results = {
        'training_completed': True,
        'performance_metrics': {},
        'resource_usage': {},
        'recommendations': []
    }
    
    # Analyze system monitoring logs
    if os.path.exists(monitor_log):
        gpu_memory_peak = 0
        cpu_usage_avg = 0
        memory_usage_avg = 0
        sample_count = 0
        
        with open(monitor_log, 'r') as f:
            for line in f:
                try:
                    stats = json.loads(line.strip())
                    if 'gpu_memory_allocated' in stats:
                        gpu_memory_peak = max(gpu_memory_peak, stats['gpu_memory_allocated'])
                    cpu_usage_avg += stats.get('cpu_percent', 0)
                    memory_usage_avg += stats.get('memory_percent', 0)
                    sample_count += 1
                except:
                    continue
        
        if sample_count > 0:
            results['resource_usage'] = {
                'gpu_memory_peak_gb': round(gpu_memory_peak, 2),
                'cpu_usage_avg_percent': round(cpu_usage_avg / sample_count, 2),
                'memory_usage_avg_percent': round(memory_usage_avg / sample_count, 2)
            }
    
    # Performance recommendations
    gpu_peak = results['resource_usage'].get('gpu_memory_peak_gb', 0)
    if gpu_peak > 70:  # A100 80GB usage > 87.5%
        results['recommendations'].append("Consider reducing batch size - GPU memory usage very high")
    elif gpu_peak < 40:  # A100 80GB usage < 50%
        results['recommendations'].append("Could increase batch size - GPU memory underutilized")
    
    return results

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1]
    monitor_log = sys.argv[2]
    output_file = sys.argv[3]
    
    results = analyze_training_performance(log_dir, monitor_log)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Performance Analysis:")
    print(f"  GPU Memory Peak: {results['resource_usage'].get('gpu_memory_peak_gb', 'unknown')}GB")
    print(f"  CPU Usage Avg: {results['resource_usage'].get('cpu_usage_avg_percent', 'unknown')}%")
    print(f"  Memory Usage Avg: {results['resource_usage'].get('memory_usage_avg_percent', 'unknown')}%")
    
    if results['recommendations']:
        print("  Recommendations:")
        for rec in results['recommendations']:
            print(f"    • {rec}")
EOF

    PERFORMANCE_REPORT="$LOG_DIR/performance_analysis.json"
    python "$ANALYSIS_SCRIPT" "$LOG_DIR" "$MONITOR_LOG" "$PERFORMANCE_REPORT"
fi

# ------------------------------------------------------------------------------
# 8. Copy results back and create deployment summary
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/streaming_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_RESULTS_DIR"

echo ""
echo "Copying results to: $FINAL_RESULTS_DIR"

# Copy checkpoints and logs
if [ -d "$CHECKPOINT_DIR" ]; then
    rsync -av "$CHECKPOINT_DIR/" "$FINAL_RESULTS_DIR/checkpoints/"
fi

if [ -d "$LOG_DIR" ]; then
    rsync -av "$LOG_DIR/" "$FINAL_RESULTS_DIR/logs/"
fi

# Copy configuration
cp "$STREAMING_CONFIG" "$FINAL_RESULTS_DIR/config_streaming_production.json"

# Create comprehensive summary
DEPLOYMENT_SUMMARY="$FINAL_RESULTS_DIR/streaming_deployment_summary.json"

# Gather additional metrics
FINAL_CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.pt" 2>/dev/null | wc -l)
TOTAL_LOG_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1 || echo "0")
GPU_UTILIZATION_AVG="unknown"

# Extract GPU utilization from monitoring logs if available
if [ -f "$MONITOR_LOG" ]; then
    GPU_UTILIZATION_AVG=$(python3 -c "
import json
import sys
try:
    total = 0
    count = 0
    with open('$MONITOR_LOG', 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'gpu_utilization' in data and data['gpu_utilization'] != 'unknown':
                    total += data['gpu_utilization']
                    count += 1
            except:
                continue
    if count > 0:
        print(f'{total/count:.1f}%')
    else:
        print('unknown')
except:
    print('unknown')
" 2>/dev/null)
fi

cat > "$DEPLOYMENT_SUMMARY" << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "deployment_type": "streaming_training_a100_production",
    "training_completed": $([ $TRAIN_EXIT -eq 0 ] && echo "true" || echo "false"),
    "training_duration_seconds": $duration,
    "training_duration_human": "$(($duration / 3600))h $(($duration % 3600 / 60))m",
    "training_exit_code": $TRAIN_EXIT,
    "graceful_shutdown": $GRACEFUL_SHUTDOWN,
    "hardware_config": {
        "gpu_type": "A100 80GB",
        "cpu_cores": 16,
        "memory_gb": 250,
        "partition": "gpu_requeue",
        "constraint": "a100",
        "node": "$(hostname)"
    },
    "streaming_config": {
        "cache_dir": "$STREAMING_CACHE",
        "log_dir": "$STREAMING_LOGS",
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 64,
        "workers": 8,
        "cache_memory_mb": 5120,
        "concurrent_downloads": 8,
        "mixed_precision": true,
        "tensor_cores": true
    },
    "results_metrics": {
        "checkpoint_count": $FINAL_CHECKPOINT_COUNT,
        "log_size": "$TOTAL_LOG_SIZE",
        "gpu_utilization_avg": "$GPU_UTILIZATION_AVG"
    },
    "results_location": {
        "checkpoints": "$FINAL_RESULTS_DIR/checkpoints",
        "logs": "$FINAL_RESULTS_DIR/logs",
        "config": "$FINAL_RESULTS_DIR/config_streaming_production.json",
        "performance_analysis": "$FINAL_RESULTS_DIR/logs/performance_analysis.json",
        "deployment_summary": "$DEPLOYMENT_SUMMARY"
    },
    "cluster_environment": {
        "submission_directory": "$SLURM_SUBMIT_DIR",
        "scratch_workspace": "$JOB_SCRATCH",
        "streaming_cache": "$STREAMING_CACHE",
        "completion_time": "$(date --iso-8601=seconds)",
        "slurm_job_id": "$SLURM_JOB_ID"
    },
    "production_features": {
        "signal_handling": true,
        "graceful_shutdown": true,
        "checkpoint_monitoring": true,
        "resource_monitoring": true,
        "error_handling": true,
        "email_notifications": true,
        "requeue_enabled": true
    },
    "next_steps": {
        "evaluation_ready": $([ $TRAIN_EXIT -eq 0 ] && echo "true" || echo "false"),
        "model_location": "$FINAL_RESULTS_DIR/checkpoints/best_model.pt",
        "recommended_actions": [
            "Review performance analysis in logs directory",
            "Run evaluation pipeline if training successful",
            "Archive results to permanent storage",
            "Update training configuration based on performance metrics"
        ]
    }
}
EOF

echo ""
echo "=============================================="
echo "  Streaming Training Deployment Complete"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "🎉 STREAMING TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Production Training Results Summary:"
    echo "  • Duration: $(($duration / 3600))h $(($duration % 3600 / 60))m ($duration seconds)"
    echo "  • Checkpoints Created: $FINAL_CHECKPOINT_COUNT"
    echo "  • Log Data Size: $TOTAL_LOG_SIZE"
    echo "  • Average GPU Utilization: $GPU_UTILIZATION_AVG"
    echo "  • Graceful Shutdown: $GRACEFUL_SHUTDOWN"
    echo ""
    echo "Data Locations:"
    echo "  • Model Checkpoints: $FINAL_RESULTS_DIR/checkpoints/"
    echo "  • Training Logs: $FINAL_RESULTS_DIR/logs/"
    echo "  • Performance Analysis: $FINAL_RESULTS_DIR/logs/performance_analysis.json"
    echo "  • Production Config: $FINAL_RESULTS_DIR/config_streaming_production.json"
    echo ""
    echo "✅ READY FOR EVALUATION PIPELINE"
    
    # Check if best model exists
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        BEST_MODEL_SIZE=$(du -h "$CHECKPOINT_DIR/best_model.pt" | cut -f1)
        echo "  • Best Model: $BEST_MODEL_SIZE ($FINAL_RESULTS_DIR/checkpoints/best_model.pt)"
    fi
    
    FINAL_EXIT=0
else
    echo "❌ STREAMING TRAINING FAILED (Exit Code: $TRAIN_EXIT)"
    echo ""
    echo "Failure Analysis:"
    echo "  • Duration Before Failure: $(($duration / 3600))h $(($duration % 3600 / 60))m"
    echo "  • Partial Checkpoints: $FINAL_CHECKPOINT_COUNT"
    echo "  • Graceful Shutdown: $GRACEFUL_SHUTDOWN"
    echo ""
    echo "Troubleshooting Resources:"
    echo "  • Error Logs: $FINAL_RESULTS_DIR/logs/"
    echo "  • System Monitor: $FINAL_RESULTS_DIR/logs/system_monitor.jsonl"
    echo "  • Configuration: $FINAL_RESULTS_DIR/config_streaming_production.json"
    echo "  • Performance Guide: hybrid/training/a100_performance_tuning_guide.md"
    echo ""
    echo "🔍 REVIEW LOGS FOR DEBUGGING"
    FINAL_EXIT=1
fi

echo ""
echo "=============================================="
echo "Production Deployment Summary:"
echo "  📊 Full Report: $DEPLOYMENT_SUMMARY"
echo "  🔧 SLURM Job ID: $SLURM_JOB_ID"
echo "  🖥️  Compute Node: $(hostname)"
echo "  📁 Results Base: $FINAL_RESULTS_DIR"
echo "  ⏰ Completed: $(date)"
echo "=============================================="

exit $FINAL_EXIT