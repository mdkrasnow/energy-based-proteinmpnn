#!/bin/bash
#
# Real Evaluation Data Generator - NO FALLBACK (SLURM-Compatible)
#
# This script generates evaluation data ONLY from trained models.
# It FAILS HARD if models are not available - no synthetic fallback.
#
# Usage from SLURM scripts:
#   source scripts/generate_eval_data_real_only.sh
#   generate_real_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"
#

generate_real_eval_data() {
    local MODEL_DIR="$1"
    local OUTPUT_DIR="$2"
    shift 2
    local PDB_FILES=("$@")

    echo "=========================================="
    echo "Real Evaluation Data Generation"
    echo "=========================================="
    echo "Model dir: $MODEL_DIR"
    echo "Output dir: $OUTPUT_DIR"
    echo "PDB files: ${#PDB_FILES[@]} structures"
    echo ""
    echo "⚠️  NO SYNTHETIC FALLBACK - Will fail if models unavailable"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Create temporary PDB list file
    local PDB_LIST_FILE="$OUTPUT_DIR/pdb_list.json"

    echo "Creating PDB list..."
    cat > "$PDB_LIST_FILE" << EOF
[
$(for i in "${!PDB_FILES[@]}"; do
    pdb_file="${PDB_FILES[$i]}"
    filename=$(basename "$pdb_file" .pdb)
    dirname=$(basename "$(dirname "$pdb_file")" 2>/dev/null || echo "unknown")

    # Determine difficulty
    case "$dirname" in
        "PDB_monomers"|*"monomer"*) difficulty="easy" ;;
        "PDB_complexes"|*"complex"*) difficulty="medium" ;;
        "PDB_homooligomers"|*"homooligomer"*) difficulty="hard" ;;
        *) difficulty="medium" ;;
    esac

    echo "  {"
    echo "    \"id\": \"$filename\","
    echo "    \"pdb_path\": \"$pdb_file\","
    echo "    \"difficulty\": \"$difficulty\","
    echo "    \"category\": \"$dirname\""
    if [ $i -lt $((${#PDB_FILES[@]} - 1)) ]; then
        echo "  },"
    else
        echo "  }"
    fi
done)
]
EOF

    # Also save as benchmark_problems.json (required by evaluation script)
    /bin/cp "$PDB_LIST_FILE" "$OUTPUT_DIR/benchmark_problems.json"

    # Validate prerequisites - FAIL if not met
    echo "Validating prerequisites..."
    echo ""

    # Check 1: Model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ FATAL ERROR: Model directory does not exist: $MODEL_DIR"
        echo ""
        echo "Cannot proceed without trained models."
        echo "Please ensure training completed successfully first."
        echo ""
        return 1
    fi

    # Check 2: Model files exist
    local MODEL_COUNT=$(find "$MODEL_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        echo "❌ FATAL ERROR: No trained model files (.pt) found in: $MODEL_DIR"
        echo ""
        echo "Expected to find files like:"
        echo "  - best_model.pt"
        echo "  - final_model.pt"
        echo "  - checkpoint_*.pt"
        echo ""
        echo "Cannot proceed without trained models."
        echo "Please complete model training first."
        echo ""
        return 1
    fi

    echo "✓ Found $MODEL_COUNT trained model file(s)"

    # Check 3: Python script exists
    # Use BASH_SOURCE for correct path when sourced, fallback to relative search
    if [ -n "${BASH_SOURCE[0]}" ]; then
        local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        local GEN_SCRIPT="$SCRIPT_DIR/generate_real_eval_data.py"
    else
        local GEN_SCRIPT="$(dirname "$0")/generate_real_eval_data.py"
    fi

    # Also check repository location if not found
    if [ ! -f "$GEN_SCRIPT" ] && [ -n "$REPO_DIR" ]; then
        GEN_SCRIPT="$REPO_DIR/scripts/generate_real_eval_data.py"
    fi

    # Also check home directory scripts folder
    if [ ! -f "$GEN_SCRIPT" ]; then
        GEN_SCRIPT="$HOME/scripts/generate_real_eval_data.py"
    fi

    if [ ! -f "$GEN_SCRIPT" ]; then
        echo "❌ FATAL ERROR: Real data generation script not found"
        echo ""
        echo "Searched locations:"
        echo "  - ${SCRIPT_DIR:-<not set>}/generate_real_eval_data.py"
        echo "  - $REPO_DIR/scripts/generate_real_eval_data.py"
        echo "  - $HOME/scripts/generate_real_eval_data.py"
        echo ""
        echo "Please ensure scripts/generate_real_eval_data.py exists in repository"
        echo ""
        return 1
    fi

    echo "✓ Real data generation script available: $GEN_SCRIPT"

    # Check 4: PyTorch available
    if ! timeout 10 python -c "import torch; import numpy" 2>/dev/null; then
        echo "❌ FATAL ERROR: PyTorch dependencies not available"
        echo ""
        echo "Required Python packages:"
        echo "  - torch"
        echo "  - numpy"
        echo ""
        echo "Please load correct Python environment:"
        echo "  module load python/3.10.9-fasrc01"
        echo "  pip install --user torch numpy"
        echo ""
        return 1
    fi

    echo "✓ PyTorch dependencies available"
    echo ""

    # All checks passed - generate REAL data
    echo "=========================================="
    echo "GENERATING REAL EVALUATION DATA"
    echo "=========================================="
    echo "Running TRAINED MODEL on test structures"
    echo "This will generate REAL optimization data"
    echo ""
    echo "⏱️  Expected time: ~1-2 minutes per structure"
    echo "📊 Total structures: ${#PDB_FILES[@]}"
    echo ""

    # Run the Python script to generate real data
    python "$GEN_SCRIPT" \
        --model-dir "$MODEL_DIR" \
        --pdb-files "$PDB_LIST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        --num-samples 1 \
        --max-structures ${MAX_EVAL_STRUCTURES:-100} \
        --verbose

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ REAL EVALUATION DATA GENERATED"
        echo "=========================================="
        echo "Source: Actual trained model optimization"
        echo ""
        echo "Files created:"
        echo "  ✓ $OUTPUT_DIR/optimization_results.json"
        echo "  ✓ $OUTPUT_DIR/landscape_data.json"
        echo "  ✓ $OUTPUT_DIR/benchmark_problems.json"
        echo ""
        echo "Data type: REAL (from trained model)"
        echo "Ready for comprehensive evaluation."
        echo "=========================================="
        echo ""
        return 0
    else
        echo ""
        echo "=========================================="
        echo "❌ REAL DATA GENERATION FAILED"
        echo "=========================================="
        echo "Exit code: $EXIT_CODE"
        echo ""
        echo "Possible causes:"
        echo "  1. Model checkpoint corrupted or incompatible"
        echo "  2. PDB files cannot be parsed"
        echo "  3. Model architecture mismatch"
        echo "  4. CUDA/GPU errors"
        echo "  5. Out of memory"
        echo ""
        echo "Check the error messages above for details."
        echo ""
        echo "Cannot proceed without real evaluation data."
        echo "=========================================="
        echo ""
        return 1
    fi
}

# If script is executed directly (not sourced), run with args
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    if [ $# -lt 2 ]; then
        echo "Usage: $0 MODEL_DIR OUTPUT_DIR [PDB_FILES...]"
        echo ""
        echo "Example:"
        echo "  $0 ./trained_models ./evaluation_data *.pdb"
        echo ""
        echo "Note: This script requires trained models and will fail if unavailable."
        echo "      No synthetic data fallback is provided."
        exit 1
    fi

    generate_real_eval_data "$@"
fi
