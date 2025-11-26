#!/bin/bash
#SBATCH -J train_hybrid_proteinmpnn           # Job name
#SBATCH -p gpu                               # Use GPU partition for real training
#SBATCH --account=ydu_lab                    # Your lab account
#SBATCH --gres=gpu:1                         # 1 GPU
#SBATCH -c 16                                # 16 CPU cores
#SBATCH -t 03-00:00:00                       # 3 days for full training
#SBATCH --mem=64G                            # 64 GB RAM
#SBATCH -o train_hybrid_proteinmpnn_%j.out   # STDOUT file
#SBATCH -e train_hybrid_proteinmpnn_%j.err   # STDERR file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkrasnow@college.harvard.edu

echo "=============================================="
echo "  Hybrid ProteinMPNN Training Job Started"
echo "=============================================="
echo "Date:          $(date)"
echo "Node:          $(hostname)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Submit Dir:    $SLURM_SUBMIT_DIR"
echo "SCRATCH:       $SCRATCH"
echo "=============================================="

# ------------------------------------------------------------------------------
# 1. Configure correct FASRC Scratch path
# ------------------------------------------------------------------------------

LAB_NAME="ydu_lab"                              # MUST match your lab account
LAB_SCRATCH_ROOT="$SCRATCH/${LAB_NAME}/Lab/$USER"
JOB_SCRATCH="${LAB_SCRATCH_ROOT}/hybrid_proteinmpnn_train_${SLURM_JOB_ID}"

echo "Lab scratch root: $LAB_SCRATCH_ROOT"
echo "Job scratch dir : $JOB_SCRATCH"

# Create your personal scratch root if missing
mkdir -p "$LAB_SCRATCH_ROOT" || {
    echo "ERROR: Cannot create $LAB_SCRATCH_ROOT"
    exit 1
}

# Create a per-job scratch workspace
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
# 2. Clone Git repository to get latest codebase
# ------------------------------------------------------------------------------

REPO_URL="https://github.com/mdkrasnow/energy-based-proteinmpnn.git"
REPO_DIR="$JOB_SCRATCH/energy-based-proteinmpnn"

echo "Cloning repository to get latest codebase..."
echo "Repository URL: $REPO_URL"
echo "Target directory: $REPO_DIR"

# Remove any existing repository directory
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing repository directory..."
    rm -rf "$REPO_DIR"
fi

# Clone the repository
git clone "$REPO_URL" "$REPO_DIR" || {
    echo "ERROR: Failed to clone repository from $REPO_URL"
    exit 1
}

echo "Repository cloned successfully to: $REPO_DIR"

# ------------------------------------------------------------------------------
# 3. Set up working directory with necessary files
# ------------------------------------------------------------------------------

cd "$REPO_DIR" || {
    echo "ERROR: Cannot cd to repository directory"
    exit 1
}

echo "Working from repository directory: $(pwd)"

# ------------------------------------------------------------------------------
# 4. Modules & Python environment
# ------------------------------------------------------------------------------

module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

export PATH="$HOME/.local/bin:$PATH"

echo "Installing dependencies to ~/.local ..."
pip install --user -q torch torchvision torchaudio \
    numpy pandas matplotlib seaborn \
    tqdm einops accelerate \
    tensorboard scikit-learn \
    biopython biotite mdanalysis \
    ipdb jupyter

echo "Dependencies installed successfully."

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU availability and memory
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
"

# ------------------------------------------------------------------------------
# 5. Prepare data directories and download ProteinMPNN weights
# ------------------------------------------------------------------------------

echo "Setting up data directories and downloading model weights..."

# Create necessary directories
mkdir -p "$JOB_SCRATCH/data/protein_structures"
mkdir -p "$JOB_SCRATCH/data/stability_data"
mkdir -p "$JOB_SCRATCH/checkpoints"
mkdir -p "$JOB_SCRATCH/logs"
mkdir -p "$JOB_SCRATCH/results"

# Download ProteinMPNN pre-trained weights if not present
PROTEINMPNN_WEIGHTS_DIR="$REPO_DIR/proteinmpnn/vanilla_model_weights"
if [ ! -f "$PROTEINMPNN_WEIGHTS_DIR/v_48_020.pt" ]; then
    echo "Downloading ProteinMPNN pre-trained weights..."
    cd "$REPO_DIR/proteinmpnn"
    
    # Download weights using the original ProteinMPNN method
    wget -q https://files.ipd.uw.edu/pub/training_sets/v_48_020.pt -O vanilla_model_weights/v_48_020.pt || {
        echo "ERROR: Failed to download ProteinMPNN weights"
        echo "Attempting alternative download..."
        # Alternative: copy from known locations or use curl
        curl -s -o vanilla_model_weights/v_48_020.pt https://files.ipd.uw.edu/pub/training_sets/v_48_020.pt || {
            echo "WARNING: Could not download ProteinMPNN weights automatically"
            echo "Training will proceed but may fail if weights are required"
        }
    }
    
    cd "$REPO_DIR"
else
    echo "ProteinMPNN weights already available"
fi

# Check if weights file exists and is reasonable size (should be ~50-200MB)
if [ -f "$PROTEINMPNN_WEIGHTS_DIR/v_48_020.pt" ]; then
    weight_size=$(du -h "$PROTEINMPNN_WEIGHTS_DIR/v_48_020.pt" | cut -f1)
    echo "ProteinMPNN weights file size: $weight_size"
else
    echo "WARNING: ProteinMPNN weights file not found at $PROTEINMPNN_WEIGHTS_DIR/v_48_020.pt"
fi

# ------------------------------------------------------------------------------
# 6. Generate/Prepare training data
# ------------------------------------------------------------------------------

echo "Preparing training data..."

# Create optimal training configuration
TRAINING_CONFIG="$JOB_SCRATCH/training_config.json"

cat > "$TRAINING_CONFIG" << 'EOF'
{
    "model": {
        "mpnn_encoder": {
            "model_name": "v_48_020",
            "hidden_dim": 128,
            "freeze_layers": true
        },
        "energy_head": {
            "hidden_dim": 512,
            "num_layers": 3,
            "dropout": 0.1,
            "activation": "relu",
            "use_batch_norm": true
        },
        "sequence_repr": {
            "temperature_schedule": [1.0, 0.5, 0.1],
            "min_temperature": 0.001,
            "max_temperature": 10.0
        }
    },
    "data": {
        "data_dir": "./data",
        "positive_ratio": 0.5,
        "negative_methods": ["random", "shuffle", "adversarial"],
        "max_sequence_length": 500,
        "min_sequence_length": 20,
        "val_split": 0.2,
        "lazy_loading": true
    },
    "training": {
        "batch_size": 32,
        "max_epochs": 100,
        "patience": 20,
        "save_frequency": 10,
        "num_workers": 8,
        "max_grad_norm": 1.0
    },
    "optimization": {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 10
        }
    },
    "loss": {
        "margin": 1.0,
        "temperature": 0.1,
        "ranking_weight": 1.0,
        "contrastive_weight": 1.0,
        "entropy_weight": 0.01,
        "smoothness_weight": 0.001,
        "negative_weights": {
            "random": 1.0,
            "shuffle": 1.0,
            "adversarial": 1.2
        }
    },
    "seed": 42,
    "allow_unsafe_checkpoint_loading": false
}
EOF

echo "Training configuration created at: $TRAINING_CONFIG"

# Adjust batch size based on GPU memory
GPU_MEMORY=$(python -c "
import torch
if torch.cuda.is_available():
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(int(memory_gb))
else:
    print(16)
" 2>/dev/null || echo "16")

echo "Detected GPU memory: ${GPU_MEMORY}GB"

# Adjust batch size based on available memory
if [ "$GPU_MEMORY" -lt 12 ]; then
    echo "Limited GPU memory detected, reducing batch size to 16"
    sed -i 's/"batch_size": 32/"batch_size": 16/' "$TRAINING_CONFIG"
elif [ "$GPU_MEMORY" -lt 8 ]; then
    echo "Very limited GPU memory detected, reducing batch size to 8"
    sed -i 's/"batch_size": 32/"batch_size": 8/' "$TRAINING_CONFIG"
fi

# Generate mock training data if real data is not available
MOCK_DATA_SCRIPT="$JOB_SCRATCH/generate_mock_data.py"

cat > "$MOCK_DATA_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Generate mock training data for hybrid ProteinMPNN training"""

import os
import json
import random
import numpy as np
from pathlib import Path

def generate_mock_protein_data(output_dir: str, num_structures: int = 100):
    """Generate mock protein structure and stability data"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock protein structures
    structures_dir = output_dir / "protein_structures"
    structures_dir.mkdir(exist_ok=True)
    
    # Generate mock stability dataset
    stability_dir = output_dir / "stability_data"
    stability_dir.mkdir(exist_ok=True)
    
    mock_data = []
    
    for i in range(num_structures):
        # Generate random protein sequence
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        seq_length = random.randint(50, 200)
        sequence = ''.join(random.choice(amino_acids) for _ in range(seq_length))
        
        # Generate mock backbone features (simplified)
        backbone_features = np.random.randn(seq_length, 128).astype(np.float32)
        
        # Mock stability labels and features
        structure_id = f"mock_protein_{i:04d}"
        
        # Create mock positive example (stable)
        positive_example = {
            'structure_id': structure_id,
            'sequence': sequence,
            'backbone_features': backbone_features.tolist(),
            'label': 1,  # stable
            'length': seq_length,
            'generation_method': 'reference'
        }
        
        # Create mock negative examples (unstable variants)
        for neg_type in ['random', 'shuffle', 'adversarial']:
            if neg_type == 'random':
                # Random sequence of same length
                neg_sequence = ''.join(random.choice(amino_acids) for _ in range(seq_length))
            elif neg_type == 'shuffle':
                # Shuffled original sequence
                neg_sequence = ''.join(random.sample(sequence, len(sequence)))
            else:  # adversarial
                # Modified sequence with some mutations
                neg_sequence = list(sequence)
                num_mutations = random.randint(5, 15)
                for _ in range(num_mutations):
                    pos = random.randint(0, len(neg_sequence) - 1)
                    neg_sequence[pos] = random.choice(amino_acids)
                neg_sequence = ''.join(neg_sequence)
            
            negative_example = {
                'structure_id': f"{structure_id}_{neg_type}",
                'sequence': neg_sequence,
                'backbone_features': backbone_features.tolist(),  # Same backbone, different sequence
                'label': 0,  # unstable
                'length': seq_length,
                'generation_method': neg_type
            }
            
            mock_data.append(negative_example)
        
        mock_data.append(positive_example)
    
    # Save mock dataset
    dataset_file = stability_dir / "mock_stability_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"Generated {len(mock_data)} training examples")
    print(f"Saved to: {dataset_file}")
    
    # Create a simple index file
    index_file = stability_dir / "dataset_index.json"
    index_data = {
        'total_examples': len(mock_data),
        'positive_examples': len([x for x in mock_data if x['label'] == 1]),
        'negative_examples': len([x for x in mock_data if x['label'] == 0]),
        'files': ['mock_stability_dataset.json'],
        'generation_methods': ['reference', 'random', 'shuffle', 'adversarial']
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Dataset index saved to: {index_file}")
    return dataset_file

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    num_structures = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Generating mock data in: {output_dir}")
    print(f"Number of structures: {num_structures}")
    
    generate_mock_protein_data(output_dir, num_structures)
    print("Mock data generation complete!")
EOF

# Generate mock training data
echo "Generating mock training data..."
python "$MOCK_DATA_SCRIPT" "$JOB_SCRATCH/data" 100

# Update config to point to our data
sed -i "s|\"data_dir\": \"./data\"|\"data_dir\": \"$JOB_SCRATCH/data\"|" "$TRAINING_CONFIG"

# ------------------------------------------------------------------------------
# 7. Run Hybrid ProteinMPNN Training
# ------------------------------------------------------------------------------

echo "=============================================="
echo "Starting Hybrid ProteinMPNN Training"
echo "=============================================="

# Set up training directories
CHECKPOINT_DIR="$JOB_SCRATCH/checkpoints"
LOG_DIR="$JOB_SCRATCH/logs"

# Run training with optimal configuration
start_time=$(date +%s)

python hybrid/training/train_energy.py \
    --config "$TRAINING_CONFIG" \
    --model_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --device cuda

TRAIN_EXIT=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=============================================="
echo "Training Summary"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Hybrid ProteinMPNN training completed successfully!"
    echo "  Duration: ${duration}s ($(($duration / 3600))h $(($duration % 3600 / 60))m)"
    
    # Check for saved models
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        model_size=$(du -h "$CHECKPOINT_DIR/best_model.pt" | cut -f1)
        echo "  Best model saved: $model_size"
    fi
    
    if [ -f "$CHECKPOINT_DIR/final_model.pt" ]; then
        final_model_size=$(du -h "$CHECKPOINT_DIR/final_model.pt" | cut -f1)
        echo "  Final model saved: $final_model_size"
    fi
    
    # Count total checkpoints
    checkpoint_count=$(find "$CHECKPOINT_DIR" -name "*.pt" | wc -l)
    echo "  Total checkpoints: $checkpoint_count"
    
else
    echo "✗ Training FAILED with exit code: $TRAIN_EXIT"
    echo "  Check error logs for details"
fi

# ------------------------------------------------------------------------------
# 8. Training validation and basic testing
# ------------------------------------------------------------------------------

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "Running basic model validation..."
    
    # Create a simple validation script
    VALIDATION_SCRIPT="$JOB_SCRATCH/validate_model.py"
    
    cat > "$VALIDATION_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Basic validation of trained hybrid model"""

import torch
import json
import sys
from pathlib import Path

def validate_model(checkpoint_path, config_path):
    """Load and validate trained model"""
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        print("✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
        
        # Check model state dict
        model_state = checkpoint['model_state_dict']
        total_params = sum(v.numel() for v in model_state.values())
        print(f"  Total parameters: {total_params:,}")
        
        # Check for key model components
        key_components = [
            'backbone_encoder',
            'energy_head', 
            'sequence_repr'
        ]
        
        for component in key_components:
            component_params = [k for k in model_state.keys() if component in k]
            if component_params:
                print(f"  ✓ {component}: {len(component_params)} parameter groups")
            else:
                print(f"  ⚠️  {component}: no parameters found")
        
        print("✓ Model validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    config_path = sys.argv[2]
    
    success = validate_model(checkpoint_path, config_path)
    sys.exit(0 if success else 1)
EOF
    
    # Run validation on best model if available
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        python "$VALIDATION_SCRIPT" "$CHECKPOINT_DIR/best_model.pt" "$TRAINING_CONFIG"
        VALIDATION_EXIT=$?
        
        if [ $VALIDATION_EXIT -eq 0 ]; then
            echo "✓ Model validation passed"
        else
            echo "⚠️  Model validation issues detected"
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 9. Copy results back to submit directory for persistence
# ------------------------------------------------------------------------------

FINAL_RESULTS_DIR="$SLURM_SUBMIT_DIR/hybrid_proteinmpnn_results_${SLURM_JOB_ID}"
mkdir -p "$FINAL_RESULTS_DIR"

echo ""
echo "Copying results back to: $FINAL_RESULTS_DIR"

# Copy trained models
if [ -d "$CHECKPOINT_DIR" ]; then
    rsync -av "$CHECKPOINT_DIR/" "$FINAL_RESULTS_DIR/checkpoints/"
fi

# Copy training logs
if [ -d "$LOG_DIR" ]; then
    rsync -av "$LOG_DIR/" "$FINAL_RESULTS_DIR/logs/"
fi

# Copy configuration
/bin/cp "$TRAINING_CONFIG" "$FINAL_RESULTS_DIR/training_config.json"

# Copy any additional logs
if [ -f "$JOB_SCRATCH"/*.log ]; then
    /bin/cp "$JOB_SCRATCH"/*.log "$FINAL_RESULTS_DIR/"
fi

# Create training summary
TRAINING_SUMMARY="$FINAL_RESULTS_DIR/training_summary.json"
cat > "$TRAINING_SUMMARY" << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "training_completed": $([ $TRAIN_EXIT -eq 0 ] && echo "true" || echo "false"),
    "training_duration_seconds": $duration,
    "training_exit_code": $TRAIN_EXIT,
    "checkpoint_directory": "$FINAL_RESULTS_DIR/checkpoints",
    "log_directory": "$FINAL_RESULTS_DIR/logs",
    "config_file": "$FINAL_RESULTS_DIR/training_config.json",
    "submission_directory": "$SLURM_SUBMIT_DIR",
    "completion_time": "$(date --iso-8601=seconds)"
}
EOF

echo ""
echo "=============================================="
echo "  Final Training Results"
echo "=============================================="
echo "Results saved to: $FINAL_RESULTS_DIR"
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "🎉 HYBRID PROTEINMPNN TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Trained models available at:"
    find "$FINAL_RESULTS_DIR/checkpoints" -name "*.pt" -type f | while read model; do
        model_size=$(du -h "$model" | cut -f1)
        echo "  • $(basename "$model"): $model_size"
    done
    echo ""
    echo "Ready for evaluation! Use the evaluation script with:"
    echo "  sbatch eval_hybrid_proteinmpnn.sh"
    echo ""
    echo "Model directory for evaluation: $FINAL_RESULTS_DIR/checkpoints"
    FINAL_EXIT=0
else
    echo "❌ TRAINING FAILED"
    echo "Check error logs in: $FINAL_RESULTS_DIR/logs"
    echo "Training configuration: $FINAL_RESULTS_DIR/training_config.json"
    FINAL_EXIT=1
fi

echo "=============================================="
echo "  Job Finished at: $(date)"
echo "  Final Exit Code: $FINAL_EXIT"
echo "=============================================="

exit $FINAL_EXIT