#!/bin/bash
#SBATCH -J train_hybrid_dev                   # Job name for development
#SBATCH -p gpu_test                               # Use GPU partition
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 8                                 # 8 CPU cores (reduced)
#SBATCH -t 00:30:00                          # 30 minutes for dev testing
#SBATCH --mem=32G                            # 32 GB RAM (reduced)
#SBATCH -o train_hybrid_dev_%j.out           # STDOUT file
#SBATCH -e train_hybrid_dev_%j.err           # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Hybrid ProteinMPNN DEV Training Job Started"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# Set bash strict mode for better error handling
set -euo pipefail

# Function for error handling
handle_error() {
    echo "ERROR: Command failed at line $1"
    echo "Exiting with failure code"
    exit 1
}
trap 'handle_error $LINENO' ERR

# ------------------------------------------------------------------------------
# 1. Configure correct FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/hybrid_proteinmpnn_dev_${SLURM_JOB_ID}"

echo "Lab scratch root: $LAB_SCRATCH_ROOT"
echo "Job scratch dir : $JOB_SCRATCH"

# Create directories with error checking
echo "Creating scratch directories..."
mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

mkdir -p "$JOB_SCRATCH" || {
    echo "ERROR: Cannot create $JOB_SCRATCH"
    exit 1
}

cd "$JOB_SCRATCH" || {
    echo "ERROR: cd to JOB_SCRATCH failed"
    exit 1
}

echo "Now working in scratch: $(pwd)"

# ------------------------------------------------------------------------------
# 2. Clone Git repository (fast fail if repo unavailable)
# ------------------------------------------------------------------------------

REPO_URL="https://github.com/mdkrasnow/energy-based-proteinmpnn.git"
REPO_DIR="$JOB_SCRATCH/energy-based-proteinmpnn"

echo "Cloning repository to get latest codebase..."
echo "Repository URL: $REPO_URL"
echo "Target directory: $REPO_DIR"

# Quick connectivity test
if ! timeout 10 git ls-remote "$REPO_URL" &>/dev/null; then
    echo "ERROR: Cannot connect to repository $REPO_URL within 10 seconds"
    echo "Check network connectivity or repository availability"
    exit 1
fi

# Remove any existing repository directory
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing repository directory..."
    rm -rf "$REPO_DIR"
fi

# Clone with timeout
timeout 60 git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository from $REPO_URL (timeout or failure)"
    exit 1
}

echo "Repository cloned successfully to: $REPO_DIR"

cd "$REPO_DIR" || {
    echo "ERROR: Cannot cd to repository directory"
    exit 1
}

echo "Working from repository directory: $(pwd)"



# ------------------------------------------------------------------------------
# 3. Quick validation of required files
# ------------------------------------------------------------------------------

echo "Validating required files exist..."
REQUIRED_FILES=(
    "hybrid/training/train_energy.py"
    "hybrid/models"
    "proteinmpnn"
)

for req_file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$req_file" ]; then
        echo "ERROR: Required file/directory missing: $req_file"
        echo "Repository structure may be incorrect or incomplete"
        exit 1
    fi
    echo "✓ Found: $req_file"
done

# ------------------------------------------------------------------------------
# 4. Modules & Python environment (with quick fail)
# ------------------------------------------------------------------------------

echo "Loading modules..."
module load python/3.10.9-fasrc01 || {
    echo "ERROR: Failed to load python module"
    exit 1
}

module load cuda/12.2.0-fasrc01

# Explicitly add proteinmpnn to PYTHONPATH to ensure imports work
# Use ${PYTHONPATH:+:${PYTHONPATH}} to safely append only if set (avoids unbound variable error)
export PYTHONPATH="$(pwd)/proteinmpnn:$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
echo "Updated PYTHONPATH: $PYTHONPATH"

# DEBUG: Check python imports (now that python is loaded)
echo "Checking python imports..."
python -c "
import sys
import os
print(f'Python version: {sys.version}')
print(f'CWD: {os.getcwd()}')
print(f'sys.path: {sys.path}')

try:
    import protein_mpnn_utils
    print('SUCCESS: Imported protein_mpnn_utils')
except Exception as e:
    print(f'FAILURE: Could not import protein_mpnn_utils: {e}')

try:
    from hybrid.models import mpnn_encoder
    print('SUCCESS: Imported hybrid.models.mpnn_encoder')
except Exception as e:
    print(f'FAILURE: Could not import hybrid.models.mpnn_encoder: {e}')
"

module load cuda/12.2.0-fasrc01 || {
    echo "ERROR: Failed to load CUDA module"
    exit 1
}

export PATH="$HOME/.local/bin:$PATH"

# Full dependency installation to match production environment
echo "Installing dependencies to ~/.local ..."
timeout 600 pip install --user -q torch torchvision torchaudio \
    numpy pandas matplotlib seaborn \
    tqdm einops accelerate \
    tensorboard scikit-learn \
    biopython biotite mdanalysis \
    ipdb jupyter || {
    echo "ERROR: Failed to install dependencies within 10 minutes"
    exit 1
}

echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Quick GPU check with timeout
echo "Checking GPU availability..."
timeout 30 python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB memory')
        if memory_gb < 4:
            raise RuntimeError(f'GPU {i} has insufficient memory: {memory_gb:.1f}GB < 4GB')
else:
    raise RuntimeError('CUDA not available')
" || {
    echo "ERROR: GPU validation failed"
    exit 1
}

echo "✓ GPU validation passed"

# ------------------------------------------------------------------------------
# 5. Data directories setup for real PDB files
# ------------------------------------------------------------------------------

echo "Setting up data directories..."
mkdir -p "$JOB_SCRATCH/data"
mkdir -p "$JOB_SCRATCH/checkpoints"
mkdir -p "$JOB_SCRATCH/logs"

echo "Copying real PDB files from repository..."
# Copy all available PDB files from the repository
if [ -d "$REPO_DIR/proteinmpnn/inputs" ]; then
    # Copy from all PDB input directories
    for pdb_dir in "$REPO_DIR/proteinmpnn/inputs"/*; do
        if [ -d "$pdb_dir/pdbs" ]; then
            echo "Copying PDB files from $(basename "$pdb_dir")/pdbs/"
            cp "$pdb_dir/pdbs"/*.pdb "$JOB_SCRATCH/data/" 2>/dev/null || true
        fi
    done
    
    # List what we copied
    echo "PDB files available for training:"
    ls -1 "$JOB_SCRATCH/data"/*.pdb 2>/dev/null || {
        echo "ERROR: No PDB files found in proteinmpnn/inputs directories"
        exit 1
    }
else
    echo "ERROR: proteinmpnn/inputs directory not found"
    exit 1
fi

# Quick check for ProteinMPNN weights with fast fail
PROTEINMPNN_WEIGHTS_DIR="$REPO_DIR/proteinmpnn/vanilla_model_weights"
mkdir -p "$PROTEINMPNN_WEIGHTS_DIR"

if [ ! -f "$PROTEINMPNN_WEIGHTS_DIR/v_48_020.pt" ]; then
    echo "Downloading ProteinMPNN weights (with timeout)..."
    cd "$REPO_DIR/proteinmpnn"
    
    # Quick download with short timeout for dev
    timeout 120 wget -q https://files.ipd.uw.edu/pub/training_sets/v_48_020.pt -O vanilla_model_weights/v_48_020.pt || {
        echo "WARNING: Could not download ProteinMPNN weights quickly"
        echo "Creating dummy weights file for dev testing"
        # Create a small dummy file for testing
        touch vanilla_model_weights/v_48_020.pt
        echo "Created dummy weights file (WILL FAIL IN TRAINING - FOR DEV ONLY)"
    }
    
    cd "$REPO_DIR"
else
    echo "ProteinMPNN weights already available"
fi

# ------------------------------------------------------------------------------
# 6. Dev training configuration
# ------------------------------------------------------------------------------

echo "Using dev training configuration..."
TRAINING_CONFIG="$JOB_SCRATCH/dev_training_config.json"

# Copy the dev config from repository and update paths for scratch
cp "$REPO_DIR/hybrid/training/config_dev.json" "$TRAINING_CONFIG"

# Update configuration using python for reliability
python -c "
import json
import os

config_path = '$TRAINING_CONFIG'
scratch_data = '$JOB_SCRATCH/data'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update data directory
    config['data']['data_dir'] = scratch_data
    
    # Ensure backbone extraction is ENABLED for dev run to match production
    config['data']['extract_backbone_features'] = True
    
    # Explicitly disable missing backbone allowance to force strict checking
    if 'model' not in config:
        config['model'] = {}
    config['model']['allow_missing_backbone'] = False

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f'Updated config: data_dir={scratch_data}, backbone_features=True')
except Exception as e:
    print(f'Error updating config: {e}')
    exit(1)
"

echo "Dev training configuration created at: $TRAINING_CONFIG"

# ------------------------------------------------------------------------------
# 7. Quick training run with timeout
# ------------------------------------------------------------------------------

echo "=============================================="
echo "Starting DEV Training (Max 20 minutes)"
echo "=============================================="

CHECKPOINT_DIR="$JOB_SCRATCH/checkpoints"
LOG_DIR="$JOB_SCRATCH/logs"

start_time=$(date +%s)

# Run training with timeout to prevent hanging
timeout 1200 python hybrid/training/train_energy.py \
    --config "$TRAINING_CONFIG" \
    --model_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --device cuda || {
    
    TRAIN_EXIT=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $TRAIN_EXIT -eq 124 ]; then
        echo "ERROR: Training timed out after 20 minutes"
        echo "This suggests training loop issues or very slow convergence"
    else
        echo "ERROR: Training failed with exit code: $TRAIN_EXIT"
        echo "Duration before failure: ${duration}s"
    fi
    
    # Copy logs for debugging
    echo "Copying failure logs..."
    FAILURE_DIR="$SLURM_SUBMIT_DIR/dev_failure_logs_${SLURM_JOB_ID}"
    mkdir -p "$FAILURE_DIR"
    rsync -av "$LOG_DIR/" "$FAILURE_DIR/logs/"
    rsync -av "$JOB_SCRATCH"/*.err "$FAILURE_DIR/" 2>/dev/null || true
    rsync -av "$JOB_SCRATCH"/*.out "$FAILURE_DIR/" 2>/dev/null || true
    /bin/cp "$TRAINING_CONFIG" "$FAILURE_DIR/config.json"
    
    echo "Failure logs saved to: $FAILURE_DIR"
    exit 1
}

TRAIN_EXIT=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# ------------------------------------------------------------------------------
# 8. Quick validation
# ------------------------------------------------------------------------------

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Dev training completed successfully in ${duration}s!"
    
    # Quick model validation
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ] || [ -f "$CHECKPOINT_DIR/final_model.pt" ]; then
        echo "✓ Model checkpoint saved successfully"
        
        # Quick load test
        CHECKPOINT_FILE="$CHECKPOINT_DIR/best_model.pt"
        if [ ! -f "$CHECKPOINT_FILE" ]; then
            CHECKPOINT_FILE="$CHECKPOINT_DIR/final_model.pt"
        fi
        
        timeout 30 python -c "
import torch
checkpoint = torch.load('$CHECKPOINT_FILE', map_location='cpu', weights_only=True)
print('✓ Checkpoint loads successfully')
print(f'Epoch: {checkpoint.get(\"epoch\", \"unknown\")}')
print(f'Model params: {sum(v.numel() for v in checkpoint[\"model_state_dict\"].values()):,}')
        " || {
            echo "WARNING: Checkpoint validation failed"
        }
    else
        echo "WARNING: No checkpoint files found"
    fi
    
    echo ""
    echo "✅ DEV TRAINING SUCCESSFUL!"
    echo "Ready to run full training with: sbatch train_hybrid_proteinmpnn.sh"
    
else
    echo "❌ DEV TRAINING FAILED"
    echo "Fix issues before running full training"
fi

# ------------------------------------------------------------------------------
# 9. Copy dev results
# ------------------------------------------------------------------------------

DEV_RESULTS_DIR="$SLURM_SUBMIT_DIR/dev_training_results_${SLURM_JOB_ID}"
mkdir -p "$DEV_RESULTS_DIR"

echo "Copying dev results to: $DEV_RESULTS_DIR"

# Copy results
if [ -d "$CHECKPOINT_DIR" ]; then
    rsync -av "$CHECKPOINT_DIR/" "$DEV_RESULTS_DIR/checkpoints/"
fi

if [ -d "$LOG_DIR" ]; then
    rsync -av "$LOG_DIR/" "$DEV_RESULTS_DIR/logs/"
fi

/bin/cp "$TRAINING_CONFIG" "$DEV_RESULTS_DIR/dev_config.json"

# Create summary
cat > "$DEV_RESULTS_DIR/dev_summary.json" << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "training_successful": $([ $TRAIN_EXIT -eq 0 ] && echo "true" || echo "false"),
    "training_duration_seconds": $duration,
    "exit_code": $TRAIN_EXIT,
    "timestamp": "$(date --iso-8601=seconds)",
    "purpose": "development_fast_fail_testing"
}
EOF

echo ""
echo "=============================================="
echo "  DEV TRAINING COMPLETE"
echo "=============================================="
echo "Duration: ${duration}s ($(($duration / 60))m)"
echo "Results: $DEV_RESULTS_DIR"
echo "Status: $([ $TRAIN_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "=============================================="

exit $TRAIN_EXIT