# Fix: Dev Evaluation Errors

## Problems Fixed

The dev evaluation script (`eval_hybrid_proteinmpnn_dev.sh`) had multiple critical errors preventing it from running:

### Error 1: Invalid Config Parameter ❌
```
ComprehensiveEvaluationConfig.__init__() got an unexpected keyword argument 'fast_dev_run'
```

**Root cause**: The config JSON contained `"fast_dev_run": true` which is NOT a valid parameter for `ComprehensiveEvaluationConfig`.

**Impact**: Config loading failed, causing the script to use default configuration which had no data file paths set.

### Error 2: No Evaluation Data Files ❌
```
No optimization_data file provided
No landscape_data file provided
No benchmark_data file provided
FATAL ERROR: No evaluation data files provided or loaded successfully
```

**Root causes**:
1. Config failed to load (see Error 1), so file paths were lost
2. Used relative paths (`./evaluation_data/...`) instead of absolute paths
3. Generated **FAKE hash-based data** instead of real model optimization results

### Error 3: Using Synthetic Data ❌

The original script generated fake optimization results using hash functions:
```bash
hash_val=$(echo "$filename" | cksum | cut -d' ' -f1)
echo "\"design_quality\": $((60 + hash_val % 35))"  # FAKE!
```

This completely defeats the purpose of evaluation - you were evaluating random numbers, not your trained model!

## Solutions Applied

### Fix 1: Remove Invalid Config Parameter ✅

**Before:**
```json
{
    ...
    "fast_dev_run": true,  // ❌ Invalid parameter
    ...
}
```

**After:**
```json
{
    ...
    // Removed fast_dev_run - not a valid parameter
    ...
}
```

### Fix 2: Use Absolute Paths ✅

**Before:**
```json
{
    "output_directory": "./evaluation_results",
    "optimization_data_file": "./evaluation_data/optimization_results.json",
    "benchmark_data_file": "./evaluation_data/baseline_proteinmpnn_results.json"
}
```

**After:**
```json
{
    "output_directory": "$EVAL_RESULTS_DIR",
    "optimization_data_file": "$EVAL_DATA_DIR/optimization_results.json",
    "benchmark_data_file": "$EVAL_DATA_DIR/benchmark_problems.json"
}
```

Where:
- `$EVAL_RESULTS_DIR = "$JOB_SCRATCH/evaluation_results"`
- `$EVAL_DATA_DIR = "$JOB_SCRATCH/evaluation_data"`

These are absolute paths that the script can always find.

### Fix 3: Use REAL Evaluation Data ✅

**Completely replaced** the fake hash-based data generation (85 lines of synthetic data code) with:

```bash
# Use real-only data generation (NO fallback to synthetic)
source "$REPO_DIR/scripts/generate_eval_data_real_only.sh"

# Set maximum structures for dev evaluation
export MAX_EVAL_STRUCTURES=5

# Generate REAL evaluation data - FAILS if models unavailable
generate_real_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

if [ $? -ne 0 ]; then
    echo "❌ DEV EVALUATION FAILED - Need trained models"
    exit 1
fi
```

This now:
- ✅ Loads your **trained model** from dev training
- ✅ Runs **actual optimization** on test structures
- ✅ Generates **real trajectories and results**
- ✅ **Fails explicitly** if models aren't available (no fake data fallback)

## How It Works Now

### Workflow

1. **Dev training runs** → Saves models to `dev_training_results_*/checkpoints/`
2. **Dev eval starts** → Finds the most recent dev training checkpoint
3. **Real data generation**:
   - Loads trained `best_model.pt` or `final_model.pt`
   - Runs model on 5 test PDB structures
   - Generates real optimization trajectories
   - Creates `optimization_results.json` with actual results
4. **Evaluation runs** → Analyzes the REAL performance data

### Model Discovery

The script automatically finds trained models:

```bash
# Check default location
TRAINED_MODEL_DIR="$SLURM_SUBMIT_DIR/dev_training_results_latest/checkpoints"

# If not found, search for most recent dev training
LATEST_DEV=$(ls -td "$SLURM_SUBMIT_DIR"/dev_training_results_* 2>/dev/null | head -1)
if [ -n "$LATEST_DEV" ]; then
    TRAINED_MODEL_DIR="$LATEST_DEV/checkpoints"
fi
```

### Expected Output

```
==============================================
Generating REAL Evaluation Data
==============================================
Found 5 PDB structures for evaluation
Using trained models from: /path/to/dev_training_results_52311022/checkpoints

Real Evaluation Data Generation
Model dir: /path/to/dev_training_results_52311022/checkpoints
Output dir: /n/netscratch/.../evaluation_data
PDB files: 5 structures

⚠️  NO SYNTHETIC FALLBACK - Will fail if models unavailable

Validating prerequisites...
✓ Found 2 trained model file(s)
✓ Real data generation script available
✓ PyTorch dependencies available

==========================================
GENERATING REAL EVALUATION DATA
==========================================
Running TRAINED MODEL on test structures
This will generate REAL optimization data

Running real optimization on structures...
[1/5] 1A2Y (easy)...
  Structure: 89 residues
    ✓ Sample 1: converged=true, steps=142, energy=-8.234

...

==========================================
✅ REAL EVALUATION DATA GENERATED
==========================================
Source: Actual trained model optimization
Data type: REAL (from trained model)
Ready for comprehensive evaluation.
==========================================

✓ Real evaluation data ready

==============================================
Starting DEV Evaluation (Max 15 minutes)
==============================================
Running comprehensive evaluation (dev mode)...
```

## File Changes

### Modified Files

1. **`eval_hybrid_proteinmpnn_dev.sh`** (lines 228-336)
   - Removed `"fast_dev_run": true` from config
   - Changed relative paths to absolute paths
   - Replaced 85 lines of fake data generation with real data generation call
   - Added model discovery logic
   - Added clear error messages if models not found

## Benefits

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Config loads?** | No ❌ (invalid param) | Yes ✅ |
| **Data file paths?** | Relative, not found ❌ | Absolute, always found ✅ |
| **Evaluation data?** | Fake hash-based ❌ | Real model optimization ✅ |
| **Scientific value?** | Zero (random numbers) ❌ | High (actual performance) ✅ |
| **Fail if no models?** | No (generates fake data) ❌ | Yes (explicit error) ✅ |

## Testing

After running dev training successfully, run dev eval:

```bash
sbatch eval_hybrid_proteinmpnn_dev.sh
```

### Success Case
```
✅ REAL EVALUATION DATA GENERATED
Source: Actual trained model optimization
...
[2025-12-21 XX:XX:XX] Starting comprehensive performance evaluation suite
[2025-12-21 XX:XX:XX] Loading evaluation data...
[2025-12-21 XX:XX:XX] Loaded 5 optimization results
[2025-12-21 XX:XX:XX] Running performance analysis...
```

### Failure Case (No Models)
```
❌ FATAL ERROR: No trained model files (.pt) found in: /path/to/checkpoints
Cannot proceed without trained models.
Please complete model training first.

❌ DEV EVALUATION FAILED
Cannot generate real evaluation data.
This dev evaluation requires:
  1. Trained model checkpoints from dev training
  2. Models must be trained (not just initialized)
  3. PyTorch environment properly configured
```

## Summary

✅ **Removed invalid config parameter** (`fast_dev_run`)
✅ **Fixed file paths** (relative → absolute)
✅ **Replaced fake data with real optimization** results
✅ **Added model discovery** from dev training outputs
✅ **Clear error messages** when prerequisites not met
✅ **Scientific integrity** guaranteed (no fake data possible)

The dev evaluation now uses the same rigorous real-data-only approach as the production evaluation pipeline.
