# Integration Guide: Real Data Only (No Synthetic Fallback)

## Philosophy: Fail Fast, No Fake Data

This integration **removes all synthetic data generation**. The evaluation will:
- ✅ **Only run with REAL data** from trained models
- ❌ **Fail explicitly** if models aren't available
- ❌ **Never generate synthetic/fake data**
- ✅ **Clear error messages** explaining what's needed

This ensures **scientific integrity** - you can never accidentally use fake data.

---

## Quick Integration (2 Minutes)

### Step 1: Make Script Executable

```bash
chmod +x scripts/generate_eval_data_real_only.sh
chmod +x scripts/generate_real_eval_data.py
```

### Step 2: Modify `eval_hybrid_proteinmpnn_dev.sh`

Find line 269 (data generation section).

**Replace lines 269-353** with:

```bash
# ------------------------------------------------------------------------------
# 6. Real Evaluation Data Generation (NO SYNTHETIC FALLBACK)
# ------------------------------------------------------------------------------

echo "Generating REAL evaluation data from trained models..."
echo "⚠️  Will FAIL if trained models are not available"
echo ""

# Limit dev evaluation to 10 structures
export MAX_EVAL_STRUCTURES=10

# Source the real-only data generation script
source "$REPO_DIR/scripts/generate_eval_data_real_only.sh"

# Generate REAL evaluation data - FAILS if models unavailable
generate_real_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "❌ EVALUATION FAILED"
    echo "=========================================="
    echo "Cannot generate real evaluation data."
    echo ""
    echo "This evaluation script requires:"
    echo "  1. Trained model checkpoints in: $TRAINED_MODEL_DIR"
    echo "  2. Models must be fully trained (not just initialized)"
    echo "  3. PyTorch environment properly configured"
    echo ""
    echo "Please complete model training first, then retry evaluation."
    echo "=========================================="
    exit 1
fi

echo "✓ Real evaluation data ready"
echo ""
```

### Step 3: Modify `eval_hybrid_proteinmpnn.sh`

Find line 262 (data generation section).

**Replace lines 262-440** with:

```bash
# ------------------------------------------------------------------------------
# 6. Real Evaluation Data Generation (NO SYNTHETIC FALLBACK)
# ------------------------------------------------------------------------------

echo "Generating REAL evaluation data from trained models..."
echo "⚠️  Will FAIL if trained models are not available"
echo ""

# Production evaluation settings
export MAX_EVAL_STRUCTURES=500

# Source the real-only data generation script
source "$REPO_DIR/scripts/generate_eval_data_real_only.sh"

# Generate REAL evaluation data - FAILS if models unavailable
generate_real_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"

DATA_GEN_EXIT=$?
if [ $DATA_GEN_EXIT -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "❌ EVALUATION FAILED"
    echo "=========================================="
    echo "Cannot generate real evaluation data."
    echo ""
    echo "This evaluation script requires:"
    echo "  1. Trained model checkpoints"
    echo "  2. Models must be fully trained (not just initialized)"
    echo "  3. Valid PDB structures for testing"
    echo "  4. PyTorch environment properly configured"
    echo ""
    echo "Common solutions:"
    echo "  - Wait for training to complete"
    echo "  - Check that training succeeded"
    echo "  - Verify model checkpoints exist and are valid"
    echo ""
    echo "=========================================="
    exit 1
fi

echo "✓ Real evaluation data ready"
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

echo "✓ All evaluation data files validated"
echo ""
```

---

## What Happens Now

### Scenario 1: Trained Models Available ✅

```
$ sbatch eval_hybrid_proteinmpnn.sh

[Job Output:]
Generating REAL evaluation data from trained models...
⚠️  Will FAIL if trained models are not available

Validating prerequisites...

✓ Found 3 trained model file(s)
✓ Real data generation script available
✓ PyTorch dependencies available

==========================================
GENERATING REAL EVALUATION DATA
==========================================
Running TRAINED MODEL on test structures
This will generate REAL optimization data

⏱️  Expected time: ~1-2 minutes per structure
📊 Total structures: 47

Running real optimization on structures...
[1/47] 1A2Y (easy)...
  Structure: 89 residues
    ✓ Sample 1: converged=true, steps=142, energy=-8.234

...

==========================================
✅ REAL EVALUATION DATA GENERATED
==========================================
Source: Actual trained model optimization

Files created:
  ✓ optimization_results.json
  ✓ landscape_data.json
  ✓ benchmark_problems.json

Data type: REAL (from trained model)
Ready for comprehensive evaluation.
==========================================
```

**Result:** Evaluation proceeds with real data ✅

---

### Scenario 2: No Trained Models ❌

```
$ sbatch eval_hybrid_proteinmpnn_dev.sh

[Job Output:]
Generating REAL evaluation data from trained models...
⚠️  Will FAIL if trained models are not available

Validating prerequisites...

❌ FATAL ERROR: No trained model files (.pt) found in: ./trained_models

Expected to find files like:
  - best_model.pt
  - final_model.pt
  - checkpoint_*.pt

Cannot proceed without trained models.
Please complete model training first.

==========================================
❌ EVALUATION FAILED
==========================================
Cannot generate real evaluation data.

This evaluation script requires:
  1. Trained model checkpoints in: ./trained_models
  2. Models must be fully trained (not just initialized)
  3. PyTorch environment properly configured

Please complete model training first, then retry evaluation.
==========================================

[Job exits with error code 1]
```

**Result:** Job fails immediately with clear error message ❌

---

### Scenario 3: Model Loading Fails ❌

```
$ sbatch eval_hybrid_proteinmpnn.sh

[Job Output:]
Generating REAL evaluation data from trained models...

✓ Found 3 trained model file(s)
✓ Real data generation script available
✓ PyTorch dependencies available

Using model: best_model.pt
Loading trained model from: ./trained_models/best_model.pt
✗ Failed to load model: KeyError: 'energy_head.0.weight'

==========================================
❌ FATAL ERROR: Failed to load trained model
==========================================

Model file: ./trained_models/best_model.pt

Possible causes:
  1. Model checkpoint is corrupted
  2. Architecture mismatch (checkpoint vs code)
  3. Model file is not a valid PyTorch checkpoint
  4. Model was not fully trained (only initialized)

Cannot proceed without a valid trained model.
This script does NOT use synthetic data fallback.

==========================================

[Job exits with error code 1]
```

**Result:** Job fails with diagnostic information ❌

---

## Error Handling

The system fails at **multiple checkpoints** to ensure data quality:

### Checkpoint 1: Model Directory Exists
```bash
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ FATAL ERROR: Model directory does not exist"
    exit 1
fi
```

### Checkpoint 2: Model Files Present
```bash
MODEL_COUNT=$(find "$MODEL_DIR" -name "*.pt" -type f | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ FATAL ERROR: No trained model files found"
    exit 1
fi
```

### Checkpoint 3: Python Dependencies
```bash
if ! python -c "import torch; import numpy" 2>/dev/null; then
    echo "❌ FATAL ERROR: PyTorch dependencies not available"
    exit 1
fi
```

### Checkpoint 4: Model Loads Successfully
```python
energy_model = load_trained_model(checkpoint_path, device)
if energy_model is None:
    print("❌ FATAL ERROR: Failed to load trained model")
    sys.exit(1)
```

### Checkpoint 5: Optimization Succeeds
```python
result = run_optimization(...)
if not result['successful']:
    # Still records failure in real data
    # Doesn't stop evaluation
```

---

## Comparison: Before vs After

### Before (With Synthetic Fallback)

```
Trained models? → NO
    ↓
Generate synthetic data ⚠️
    ↓
Run evaluation with fake data ❌
    ↓
Get meaningless results ❌
```

**Problem:** Could accidentally publish fake data

---

### After (Real Only, No Fallback)

```
Trained models? → NO
    ↓
FAIL IMMEDIATELY ✅
    ↓
Clear error message ✅
    ↓
User fixes issue ✅
    ↓
Retry with real models ✅
```

**Benefit:** Impossible to use fake data

---

## Benefits

### 1. Scientific Integrity ✅
- **Impossible** to accidentally use synthetic data
- Every evaluation uses real model performance
- Publishable results guaranteed

### 2. Fail Fast ✅
- Immediate failure if prerequisites not met
- Don't waste 1 hour on fake evaluation
- Clear actionable error messages

### 3. Clarity ✅
- No confusion about data source
- No need to check "source" markers
- If job succeeds, data is real

### 4. Development Workflow ✅
```
1. Develop training code
2. Run training
3. Training completes ← checkpoint
4. Run evaluation ← only works after step 3
5. Analyze real results
```

Forces correct workflow: **train first, then evaluate**

---

## Testing the Integration

### Test 1: Without Models (Should Fail)

```bash
# Ensure no models present
rm -rf trained_models/
mkdir -p trained_models/

# Run evaluation - should fail immediately
sbatch eval_hybrid_proteinmpnn_dev.sh

# Expected output:
# ❌ FATAL ERROR: No trained model files (.pt) found
# [Job exits with code 1]
```

### Test 2: With Models (Should Succeed)

```bash
# Ensure models are present
ls -lh trained_models/*.pt
# Should show: best_model.pt, final_model.pt, etc.

# Run evaluation - should generate real data
sbatch eval_hybrid_proteinmpnn_dev.sh

# Expected output:
# ✅ REAL EVALUATION DATA GENERATED
# [Job proceeds to evaluation]
```

### Test 3: Verify Data is Real

```bash
# Check source marker in generated data
jq '.[0].problem_info.source' evaluation_data/optimization_results.json
# Should output: "real_optimization"

# Check trajectory length
jq '.[0].trajectory.energy | length' evaluation_data/optimization_results.json
# Should output: 100-200 (real trajectory length)
```

---

## Troubleshooting

### Error: "No trained model files found"

**Cause:** Training hasn't completed or failed

**Solution:**
```bash
# Check training status
squeue -u $USER | grep train

# Check training results
ls -lh ~/hybrid_proteinmpnn_results_*/checkpoints/

# Wait for training to complete, then retry evaluation
```

---

### Error: "Failed to load trained model"

**Cause:** Model architecture mismatch or corruption

**Solution:**
```bash
# Verify checkpoint is valid
python -c "import torch; print(torch.load('trained_models/best_model.pt', weights_only=True).keys())"

# Check model size (should be >10MB for trained model)
ls -lh trained_models/*.pt

# If models are tiny (<1MB), they're probably just initialized
# Need to complete training first
```

---

### Error: "PyTorch dependencies not available"

**Cause:** Python environment not loaded

**Solution:**
```bash
# Load correct modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01

# Install dependencies
pip install --user torch numpy

# Verify
python -c "import torch; print(torch.__version__)"
```

---

## Summary

With this integration:

✅ **Never use fake data** - Impossible to accidentally use synthetic data
✅ **Fail fast** - Immediate clear errors if prerequisites not met
✅ **Scientific integrity** - Every result is from real model evaluation
✅ **Clear workflow** - Train first, then evaluate
✅ **Production ready** - No confusion about data source

**Integration time:** 2 minutes
**Scientific value:** Guaranteed real data
**Risk of fake data:** ZERO

The evaluation pipeline now **requires trained models** and **fails explicitly** if they're not available. No synthetic fallback, no fake data, no confusion.

**Apply the modifications above to your shell scripts to enable this behavior.**
