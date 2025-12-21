# Integration Guide: Real Evaluation Data in SLURM Scripts

## What This Does

Modifies your existing `eval_hybrid_proteinmpnn.sh` and `eval_hybrid_proteinmpnn_dev.sh` scripts to:
1. **Try to generate REAL data** from your trained models (actual optimization)
2. **Automatically fallback** to synthetic data if models aren't available yet
3. **Clearly label** which type of data is being used
4. **No manual intervention** needed - works intelligently

---

## Quick Integration (2 Minutes)

### Step 1: Make Scripts Executable

```bash
chmod +x scripts/generate_real_eval_data.py
chmod +x scripts/generate_eval_data_smart.sh
```

### Step 2: Modify `eval_hybrid_proteinmpnn_dev.sh`

Find this section (around line 269-353):
```bash
# Create evaluation data from real PDB structures (dev version)
echo "Creating evaluation data from real PDB structures..."
```

**Replace the entire data generation section (lines 269-353)** with:

```bash
# ------------------------------------------------------------------------------
# 6. Smart Evaluation Data Generation (Real or Synthetic)
# ------------------------------------------------------------------------------

echo "Generating evaluation data (real if models available, synthetic otherwise)..."

# Source the smart data generation script
source "$REPO_DIR/scripts/generate_eval_data_smart.sh"

# Generate evaluation data intelligently
generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo "ERROR: Failed to generate evaluation data"
    exit 1
fi

echo "✓ Evaluation data ready"
```

### Step 3: Modify `eval_hybrid_proteinmpnn.sh`

Find this section (around line 262-440):
```bash
# Generate evaluation data from real PDB structures
echo "Generating evaluation data from real PDB structures..."
```

**Replace the entire data generation section (lines 262-440)** with:

```bash
# ------------------------------------------------------------------------------
# 6. Smart Evaluation Data Generation (Real or Synthetic)
# ------------------------------------------------------------------------------

echo "Generating evaluation data (real if models available, synthetic otherwise)..."

# Export max structures for production run
export MAX_EVAL_STRUCTURES=500

# Source the smart data generation script
source "$REPO_DIR/scripts/generate_eval_data_smart.sh"

# Generate evaluation data intelligently
generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo "ERROR: Failed to generate evaluation data"
    exit 1
fi

echo "✓ Evaluation data ready"

# Generate baseline comparison data
echo "Generating baseline comparison data..."
generate_synthetic_data "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"
echo "✓ Baseline data created"
```

---

## What Changes

### Before (Current Behavior):
```
[2025-12-16 21:10:43] Creating evaluation data from real PDB structures...
[2025-12-16 21:10:43] Using 47 real PDB structures for evaluation data generation
[Generates hash-based synthetic data]
✓ Evaluation data generated from 47 real PDB structures
```
- **Always uses synthetic data** (hash-based fake results)
- **Misleading**: Says "real PDB structures" but generates fake optimization results
- **Not useful**: Can't detect actual model problems

### After (New Behavior):

#### Scenario 1: Trained Models Available
```
[2025-12-16 21:10:43] Smart Evaluation Data Generation
[2025-12-16 21:10:43] ✓ Found 3 trained model(s)
[2025-12-16 21:10:43] ✓ Real data generation script available
[2025-12-16 21:10:43] ✓ PyTorch dependencies available
[2025-12-16 21:10:43]
[2025-12-16 21:10:43] GENERATING REAL EVALUATION DATA
[2025-12-16 21:10:43] This will run your TRAINED MODEL on test structures
[2025-12-16 21:10:43] to generate REAL optimization data (not synthetic).
[2025-12-16 21:10:43]
[2025-12-16 21:10:43] Running real optimization on structures...
[2025-12-16 21:10:45] [1/47] 1A2Y (easy)...
[2025-12-16 21:10:47]   Structure: 89 residues
[2025-12-16 21:10:52]     ✓ Sample 1: converged=true, steps=142, energy=-8.234
[2025-12-16 21:10:52] [2/47] 1HMP (medium)...
...
[2025-12-16 21:25:30] ✓ REAL evaluation data generated successfully!
[2025-12-16 21:25:30]   Source: Actual model optimization on test structures
[2025-12-16 21:25:30]   Success rate: 87.2%
```
- **Runs your actual trained model**
- **Generates real optimization trajectories**
- **Shows actual performance metrics**

#### Scenario 2: No Trained Models (Training Not Done Yet)
```
[2025-12-16 21:10:43] Smart Evaluation Data Generation
[2025-12-16 21:10:43] ⚠️  Cannot generate REAL data: No trained models found
[2025-12-16 21:10:43]
[2025-12-16 21:10:43] GENERATING SYNTHETIC EVALUATION DATA
[2025-12-16 21:10:43] ⚠️  WARNING: Using hash-based synthetic data
[2025-12-16 21:10:43]   Reason: No trained models found in ./trained_models
[2025-12-16 21:10:43]
[2025-12-16 21:10:43] This data is for INFRASTRUCTURE TESTING ONLY.
[2025-12-16 21:10:43] For real scientific evaluation, you need:
[2025-12-16 21:10:43]   1. Trained model checkpoints in: ./trained_models
[2025-12-16 21:10:43]   2. PyTorch environment properly configured
[2025-12-16 21:10:43]
[2025-12-16 21:10:44] ✓ Synthetic evaluation data generated
[2025-12-16 21:10:44]   Source: Hash-based deterministic generation
[2025-12-16 21:10:44]   Use case: Infrastructure testing ONLY
```
- **Falls back to synthetic data**
- **Clearly warns** that it's not real
- **Explains** what's needed for real evaluation

---

## Complete Modified Sections

### For `eval_hybrid_proteinmpnn_dev.sh`

Replace lines 269-353 with:

```bash
# ------------------------------------------------------------------------------
# 6. Smart Evaluation Data Generation (Real or Synthetic)
# ------------------------------------------------------------------------------

echo "Generating evaluation data (real if models available, synthetic otherwise)..."

# Limit dev evaluation to 10 structures
export MAX_EVAL_STRUCTURES=10

# Source the smart data generation script
source "$REPO_DIR/scripts/generate_eval_data_smart.sh"

# Generate evaluation data intelligently
# This will:
# - Try to use trained models to generate REAL data
# - Fall back to synthetic data if models unavailable
# - Clearly label which type was used
generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo "ERROR: Failed to generate evaluation data"
    exit 1
fi

echo "✓ Evaluation data ready for comprehensive evaluation"
echo ""
```

### For `eval_hybrid_proteinmpnn.sh`

Replace lines 262-440 with:

```bash
# ------------------------------------------------------------------------------
# 6. Smart Evaluation Data Generation (Real or Synthetic)
# ------------------------------------------------------------------------------

echo "Generating evaluation data (real if models available, synthetic otherwise)..."

# Production evaluation settings
export MAX_EVAL_STRUCTURES=500  # Process up to 500 structures

# Source the smart data generation script
source "$REPO_DIR/scripts/generate_eval_data_smart.sh"

# Generate evaluation data intelligently
# Priority: Real data from trained model > Synthetic data for infrastructure testing
generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo "ERROR: Failed to generate evaluation data"
    exit 1
fi

echo "✓ Evaluation data ready"
echo ""

# Validate generated JSON files
echo "Validating generated JSON files..."
json_valid=true

for json_file in benchmark_problems.json optimization_results.json landscape_data.json; do
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c "import json; json.load(open('$JOB_SCRATCH/evaluation_data/$json_file'))" 2>/dev/null; then
            echo "  ✓ $json_file: Valid JSON"
        else
            echo "  ✗ $json_file: Invalid JSON"
            json_valid=false
        fi
    fi
done

if [ "$json_valid" = false ]; then
    echo "ERROR: Some generated JSON files are invalid"
    exit 1
fi

echo "✓ All evaluation data files validated successfully"
echo ""
```

---

## Testing the Integration

### Test 1: Infrastructure Testing (No Models)
```bash
# Remove trained models to test fallback
rm -rf trained_models/

# Run dev evaluation
sbatch eval_hybrid_proteinmpnn_dev.sh
```

**Expected:**
- Should generate synthetic data with clear warnings
- Evaluation infrastructure should work
- Should see "INFRASTRUCTURE TESTING ONLY" message

### Test 2: Real Evaluation (With Models)
```bash
# First train a model or have trained models available
ls trained_models/*.pt

# Run dev evaluation
sbatch eval_hybrid_proteinmpnn_dev.sh
```

**Expected:**
- Should detect trained models
- Should run actual optimization
- Should generate REAL evaluation data
- Should take longer (actual computation)

---

## How to Verify Real vs Synthetic Data

Check the generated JSON files:

### Real Data Indicators:
```json
{
  "problem_info": {
    "source": "real_optimization"  // <-- Real data marker
  },
  "optimization_result": {
    "converged": true,
    "total_steps_used": 142,  // <-- Actual optimization steps
    "final_energy": -8.234567  // <-- Precise real values
  },
  "trajectory": {
    "energy": [-2.1, -3.8, -5.2, ...]  // <-- Full trajectory
  }
}
```

### Synthetic Data Indicators:
```json
{
  "problem_info": {
    "source": "synthetic_hash_based"  // <-- Synthetic marker
  },
  "optimization_result": {
    "converged": true,
    "total_steps_used": 87,  // <-- Round numbers
    "final_energy": -45.3  // <-- Simple values
  },
  "trajectory": {
    "energy": [-5, -10, -15]  // <-- Minimal trajectory
  }
}
```

---

## Benefits of This Integration

### 1. Zero Manual Intervention
- Scripts automatically detect what's available
- No need to remember to switch methods
- Works in both dev and production

### 2. Clear Communication
- Always tells you which data type is being used
- Warns when using synthetic data
- Explains what's needed for real data

### 3. Scientific Integrity
- Real data when possible
- Clear labeling prevents misinterpretation
- Synthetic data only for infrastructure testing

### 4. Backwards Compatible
- Still works if Python modules aren't available
- Still works if models haven't been trained yet
- Graceful degradation, not hard failures

### 5. Production Ready
- Handles errors gracefully
- Provides actionable error messages
- Validates data before proceeding

---

## Troubleshooting

### "Real data generation failed"
**Check:**
1. Are models actually trained or just initialized?
   ```bash
   ls -lh trained_models/*.pt
   # Should see models >10MB
   ```

2. Is PyTorch available?
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. Are model imports working?
   ```bash
   python -c "from models.energy_head import EnergyHead; print('OK')"
   ```

### "Optimization always fails"
**Possible causes:**
- Model architecture mismatch (checkpoint vs code)
- Model not actually trained (just initialized weights)
- PDB parsing failures

**Debug:**
```bash
# Run data generation manually
python scripts/generate_real_eval_data.py \
    --model-dir trained_models \
    --pdb-files evaluation_data/pdb_list.json \
    --output-dir test_output \
    --device cpu \
    --verbose
```

### "Falls back to synthetic when models exist"
**Check Python dependencies:**
```bash
# From SLURM job, check environment
module load python/3.10.9-fasrc01
python -c "import sys; print(sys.path)"
python -c "import torch, numpy; print('OK')"
```

---

## Summary

With these changes:
- ✅ **Automatic**: Detects trained models and generates real data when available
- ✅ **Safe**: Falls back to synthetic data for infrastructure testing
- ✅ **Clear**: Always tells you which type of data is being used
- ✅ **Scientific**: Real evaluation data for actual performance analysis
- ✅ **Compatible**: Works with existing SLURM workflow

**Next Steps:**
1. Apply the modifications above to your scripts
2. Test with `sbatch eval_hybrid_proteinmpnn_dev.sh`
3. Verify data type in output logs
4. Check JSON files for "source" markers

Your evaluation pipeline will now generate **real scientific data** when models are trained, and clearly indicate when it's using synthetic data for testing!
