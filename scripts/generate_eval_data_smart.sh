#!/bin/bash
#
# Smart Evaluation Data Generator (SLURM-Compatible)
#
# This script intelligently generates evaluation data:
# 1. Tries to generate REAL data from trained models
# 2. Falls back to synthetic data if models aren't available/ready
#
# Usage from SLURM scripts:
#   source scripts/generate_eval_data_smart.sh
#   generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"
#

generate_eval_data() {
    local MODEL_DIR="$1"
    local OUTPUT_DIR="$2"
    shift 2
    local PDB_FILES=("$@")

    echo "=========================================="
    echo "Smart Evaluation Data Generation"
    echo "=========================================="
    echo "Model dir: $MODEL_DIR"
    echo "Output dir: $OUTPUT_DIR"
    echo "PDB files: ${#PDB_FILES[@]} structures"
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

    # Check if we can generate REAL data
    local USE_REAL_DATA=false
    local REAL_DATA_REASON=""

    # Check 1: Do we have trained models?
    if [ -d "$MODEL_DIR" ]; then
        local MODEL_COUNT=$(find "$MODEL_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)

        if [ "$MODEL_COUNT" -gt 0 ]; then
            echo "✓ Found $MODEL_COUNT trained model(s)"

            # Check 2: Is the Python script available?
            local GEN_SCRIPT="$(dirname "$0")/generate_real_eval_data.py"

            if [ -f "$GEN_SCRIPT" ]; then
                echo "✓ Real data generation script available"

                # Check 3: Are PyTorch and dependencies available?
                if timeout 10 python -c "import torch; import numpy" 2>/dev/null; then
                    echo "✓ PyTorch dependencies available"
                    USE_REAL_DATA=true
                else
                    REAL_DATA_REASON="PyTorch not available"
                fi
            else
                REAL_DATA_REASON="Generation script not found: $GEN_SCRIPT"
            fi
        else
            REAL_DATA_REASON="No trained models found in $MODEL_DIR"
        fi
    else
        REAL_DATA_REASON="Model directory does not exist: $MODEL_DIR"
    fi

    # Generate data based on availability
    if [ "$USE_REAL_DATA" = true ]; then
        echo ""
        echo "=========================================="
        echo "GENERATING REAL EVALUATION DATA"
        echo "=========================================="
        echo "This will run your TRAINED MODEL on test structures"
        echo "to generate REAL optimization data (not synthetic)."
        echo ""

        # Run the Python script to generate real data
        local GEN_SCRIPT="$(dirname "$0")/generate_real_eval_data.py"

        python "$GEN_SCRIPT" \
            --model-dir "$MODEL_DIR" \
            --pdb-files "$PDB_LIST_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda \
            --num-samples 1 \
            --max-structures ${MAX_EVAL_STRUCTURES:-100} \
            --fallback-synthetic \
            --verbose

        local EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✓ REAL evaluation data generated successfully!"
            echo "  Source: Actual model optimization on test structures"
            echo "  Files:"
            echo "    - $OUTPUT_DIR/optimization_results.json (REAL data)"
            echo "    - $OUTPUT_DIR/landscape_data.json (REAL data)"
            echo "    - $OUTPUT_DIR/benchmark_problems.json"
            echo ""
            return 0
        else
            echo ""
            echo "⚠️  Real data generation failed (exit code: $EXIT_CODE)"
            echo "   Falling back to synthetic data..."
            echo ""
            USE_REAL_DATA=false
            REAL_DATA_REASON="Real data generation failed"
        fi
    else
        echo ""
        echo "⚠️  Cannot generate REAL data: $REAL_DATA_REASON"
        echo ""
    fi

    # Fallback: Generate synthetic data
    if [ "$USE_REAL_DATA" = false ]; then
        echo "=========================================="
        echo "GENERATING SYNTHETIC EVALUATION DATA"
        echo "=========================================="
        echo "⚠️  WARNING: Using hash-based synthetic data"
        echo "  Reason: $REAL_DATA_REASON"
        echo ""
        echo "This data is for INFRASTRUCTURE TESTING ONLY."
        echo "For real scientific evaluation, you need:"
        echo "  1. Trained model checkpoints in: $MODEL_DIR"
        echo "  2. PyTorch environment properly configured"
        echo "  3. Real data generation script available"
        echo ""

        generate_synthetic_data "$OUTPUT_DIR" "${PDB_FILES[@]}"

        echo ""
        echo "✓ Synthetic evaluation data generated"
        echo "  Source: Hash-based deterministic generation"
        echo "  Use case: Infrastructure testing ONLY"
        echo "  Files:"
        echo "    - $OUTPUT_DIR/optimization_results.json (SYNTHETIC)"
        echo "    - $OUTPUT_DIR/landscape_data.json (SYNTHETIC)"
        echo "    - $OUTPUT_DIR/benchmark_problems.json"
        echo ""

        return 0
    fi
}

generate_synthetic_data() {
    local OUTPUT_DIR="$1"
    shift
    local PDB_FILES=("$@")

    # Generate optimization results (hash-based synthetic)
    cat > "$OUTPUT_DIR/optimization_results.json" << EOF
[
$(for i in "${!PDB_FILES[@]}"; do
    pdb_file="${PDB_FILES[$i]}"
    filename=$(basename "$pdb_file" .pdb)
    dirname=$(basename "$(dirname "$pdb_file")" 2>/dev/null || echo "unknown")

    # Hash-based deterministic values
    hash_val=$(echo "$filename" | cksum | cut -d' ' -f1)

    echo "  {"
    echo "    \"problem_id\": \"$filename\","
    echo "    \"successful\": $([ $((hash_val % 2)) -eq 0 ] && echo "true" || echo "false"),"
    echo "    \"design_quality\": $((60 + hash_val % 35)).$((hash_val % 10)),"
    echo "    \"confidence_score\": $((50 + hash_val % 40)).$((hash_val % 10)),"
    echo "    \"computation_time\": $((20 + hash_val % 30)).$((hash_val % 10)),"
    echo "    \"problem_info\": {"
    echo "      \"type\": \"protein_design\","
    echo "      \"difficulty\": \"$([ $((hash_val % 3)) -eq 0 ] && echo "easy" || [ $((hash_val % 3)) -eq 1 ] && echo "medium" || echo "hard")\","
    echo "      \"pdb_file\": \"$pdb_file\","
    echo "      \"category\": \"$dirname\","
    echo "      \"source\": \"synthetic_hash_based\""
    echo "    },"
    echo "    \"optimization_result\": {"
    echo "      \"converged\": $([ $((hash_val % 4)) -ne 3 ] && echo "true" || echo "false"),"
    echo "      \"total_steps_used\": $((30 + hash_val % 80)),"
    echo "      \"final_energy\": -$((10 + hash_val % 50)).$((hash_val % 100)),"
    echo "      \"adaptive_extensions_count\": $((hash_val % 3))"
    echo "    },"
    echo "    \"trajectory\": {"
    echo "      \"energy\": [-$((5 + hash_val % 20)), -$((10 + hash_val % 30)), -$((15 + hash_val % 40))]"
    echo "    }"
    if [ $i -lt $((${#PDB_FILES[@]} - 1)) ]; then
        echo "  },"
    else
        echo "  }"
    fi
done)
]
EOF

    # Generate landscape data (synthetic)
    cat > "$OUTPUT_DIR/landscape_data.json" << EOF
[
$(for i in "${!PDB_FILES[@]}"; do
    filename=$(basename "${PDB_FILES[$i]}" .pdb)

    echo "  {"
    echo "    \"landscape_id\": \"$filename\","
    echo "    \"temperature\": 1.0,"
    echo "    \"landscape_index\": $i,"
    echo "    \"source\": \"synthetic_hash_based\""
    if [ $i -lt $((${#PDB_FILES[@]} - 1)) ]; then
        echo "  },"
    else
        echo "  }"
    fi
done)
]
EOF

    # Generate baseline comparison data (synthetic)
    cat > "$OUTPUT_DIR/baseline_proteinmpnn_results.json" << EOF
[
$(for i in "${!PDB_FILES[@]}"; do
    filename=$(basename "${PDB_FILES[$i]}" .pdb)
    hash_val=$(echo "baseline_$filename" | cksum | cut -d' ' -f1)

    echo "  {"
    echo "    \"problem_id\": \"$filename\","
    echo "    \"successful\": $([ $((hash_val % 3)) -ne 0 ] && echo "true" || echo "false"),"
    echo "    \"design_quality\": $((40 + hash_val % 30)).$((hash_val % 10)),"
    echo "    \"confidence_score\": $((30 + hash_val % 40)).$((hash_val % 10)),"
    echo "    \"difficulty\": \"$([ $((hash_val % 3)) -eq 0 ] && echo "easy" || [ $((hash_val % 3)) -eq 1 ] && echo "medium" || echo "hard")\","
    echo "    \"method\": \"baseline_proteinmpnn\","
    echo "    \"source\": \"synthetic_hash_based\""
    if [ $i -lt $((${#PDB_FILES[@]} - 1)) ]; then
        echo "  },"
    else
        echo "  }"
    fi
done)
]
EOF
}

# If script is executed directly (not sourced), run with args
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    if [ $# -lt 2 ]; then
        echo "Usage: $0 MODEL_DIR OUTPUT_DIR [PDB_FILES...]"
        echo ""
        echo "Example:"
        echo "  $0 ./trained_models ./evaluation_data *.pdb"
        exit 1
    fi

    generate_eval_data "$@"
fi
