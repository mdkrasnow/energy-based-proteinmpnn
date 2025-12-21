# Evaluation Quickstart: Generate Real Data from Your Trained Model

## The Problem You Had

Your evaluation failed with:
```
FATAL ERROR: No evaluation data files provided or loaded successfully.
```

**Why?** The evaluation script needs data files, but the shell scripts only generate **fake/mock data** for testing infrastructure, not real model evaluation.

## The Solution (3 Steps)

### Step 1: Prepare Test Set (5 minutes)

```bash
# Create benchmark_problems.json from real PDB structures
python scripts/prepare_test_set.py \
    --pdb-dir proteinmpnn/inputs \
    --output evaluation_data/benchmark_problems.json \
    --max-per-category 10
```

**What this does:** Creates a list of real protein structures to test your model on.
**Is this "synthetic"?** No - these are real PDB structures. This is standard practice.

---

### Step 2: Generate REAL Evaluation Data (30-60 minutes)

```bash
# Run your TRAINED model on test structures to generate REAL data
python scripts/generate_evaluation_data.py \
    --checkpoint checkpoints/best_model.pt \
    --test-set evaluation_data/benchmark_problems.json \
    --output-dir evaluation_data \
    --device cuda \
    --num-samples 1
```

**What this does:**
1. Loads YOUR trained model from checkpoint
2. Runs ACTUAL optimization on each test structure
3. Saves REAL optimization trajectories and results
4. This is NOT fake data - it's your model's actual performance!

**Time:** ~1-2 minutes per structure on GPU, so 30-60 minutes for 30 structures.

**Output files:**
- `evaluation_data/optimization_results.json` - Real optimization data
- `evaluation_data/landscape_data.json` - Real landscape analysis

---

### Step 3: Run Evaluation Analysis (5 minutes)

```bash
# Analyze your model's real performance
python hybrid/evaluation/run_comprehensive_evaluation.py \
    --optimization-data evaluation_data/optimization_results.json \
    --landscape-data evaluation_data/landscape_data.json \
    --benchmark-data evaluation_data/benchmark_problems.json \
    --output-dir evaluation_results \
    --verbose
```

**What this does:** Analyzes the REAL data from Step 2 to give you:
- Convergence statistics
- Success rates by difficulty
- Landscape quality metrics
- Performance recommendations

---

## Complete Workflow (One Command)

```bash
#!/bin/bash
# run_complete_evaluation.sh

# Step 1: Prepare test set
python scripts/prepare_test_set.py \
    --pdb-dir proteinmpnn/inputs \
    --output evaluation_data/benchmark_problems.json \
    --max-per-category 10

# Step 2: Generate REAL evaluation data (this takes time!)
python scripts/generate_evaluation_data.py \
    --checkpoint checkpoints/best_model.pt \
    --test-set evaluation_data/benchmark_problems.json \
    --output-dir evaluation_data \
    --device cuda

# Step 3: Run comprehensive evaluation
python hybrid/evaluation/run_comprehensive_evaluation.py \
    --optimization-data evaluation_data/optimization_results.json \
    --landscape-data evaluation_data/landscape_data.json \
    --benchmark-data evaluation_data/benchmark_problems.json \
    --output-dir evaluation_results \
    --verbose

echo "Evaluation complete! Results in: evaluation_results/"
```

---

## Understanding "Synthetic" vs "Real" Data

### ✅ ACCEPTABLE: Using Real PDB Structures as Test Cases
```json
{
  "id": "5L33",
  "pdb_path": "/path/to/5L33.pdb",  // <-- Real experimental structure
  "difficulty": "medium",
  "type": "real_structure"
}
```
**Why acceptable:** You're testing on real structural challenges. Standard practice.

### ✅ REQUIRED: Running Your Model to Get Results
```json
{
  "converged": true,             // <-- Actual result from your model
  "final_energy": -8.234,        // <-- Real energy your model computed
  "total_steps_used": 142,       // <-- Actual optimization steps taken
  "trajectory": {
    "energy": [-2.1, -4.3, -6.2, -7.8, -8.234]  // <-- Real optimization path
  }
}
```
**Why required:** This is your model's actual performance - the whole point of evaluation!

### ❌ NOT ACCEPTABLE: Making Up Results (What Shell Scripts Do)
```json
{
  "converged": true,             // <-- Random boolean (hash-based)
  "final_energy": -5.432,        // <-- Made up number (hash-based)
  "total_steps_used": 87,        // <-- Fake value (hash-based)
  "trajectory": {"energy": []}   // <-- No real trajectory
}
```
**Why not acceptable:** Doesn't reflect your model's actual performance. Useless for science.

---

## Key Differences

| Aspect | Shell Script Approach | Proper Approach (Our Scripts) |
|--------|----------------------|-------------------------------|
| **Test structures** | Real PDB files ✓ | Real PDB files ✓ |
| **Optimization** | Hash-based fake data ✗ | Actually run trained model ✓ |
| **Results** | Made up numbers ✗ | Real model performance ✓ |
| **Use case** | Infrastructure testing only | Scientific evaluation ✓ |
| **Time** | Instant (just generates JSON) | 30-60 min (runs model) |
| **Validity** | Not valid for science | Valid for publication |

---

## FAQ

### Q: Why does Step 2 take so long?
**A:** Because it's actually running your trained model on each structure:
- Loads model from checkpoint
- Encodes backbone structure
- Runs 50-200 optimization steps per structure
- Records full trajectory
- This is REAL computation, not fake data generation

### Q: Can I speed it up?
**A:** Yes:
```bash
# Use fewer test structures
python scripts/prepare_test_set.py --max-per-category 5

# Use GPU instead of CPU
python scripts/generate_evaluation_data.py --device cuda

# Reduce samples per structure
python scripts/generate_evaluation_data.py --num-samples 1
```

### Q: What if some optimizations fail?
**A:** That's expected! The script records failures:
```json
{
  "successful": false,
  "optimization_result": {
    "converged": false,
    "optimization_failed": true,
    "failure_reason": "Energy exploded to NaN"
  }
}
```
This tells you your model has issues - that's valuable information!

### Q: Do I need the shell script synthetic data at all?
**A:** Only for quick infrastructure testing:
```bash
# Quick test that evaluation script runs (uses fake data)
bash eval_hybrid_proteinmpnn_dev.sh

# Real evaluation (uses data from your model)
bash run_complete_evaluation.sh  # Uses our scripts
```

### Q: Where's my trained model checkpoint?
**A:** Look in:
- `checkpoints/` directory
- Training script output directory
- Wherever you saved models during training

Common names:
- `best_model.pt`
- `checkpoint_epoch_100.pt`
- `model_final.pt`

---

## Troubleshooting

### Error: "Encoder checkpoint not found"
```bash
# Find your actual checkpoint location
find . -name "*.pt" -o -name "*.pth"

# Use the correct path
python scripts/generate_evaluation_data.py \
    --checkpoint path/to/your/actual/checkpoint.pt
```

### Error: "No CA atoms found in PDB"
Some PDB files might be malformed. Use validation:
```bash
python scripts/prepare_test_set.py --validate
```

### Error: "CUDA out of memory"
```bash
# Use CPU instead
python scripts/generate_evaluation_data.py --device cpu

# Or process fewer structures
python scripts/prepare_test_set.py --max-per-category 5
```

### Optimization always fails
Check your model training:
- Is the checkpoint actually trained or just initialized?
- Did training complete successfully?
- Check training logs for issues

---

## Summary

**The shell scripts generate fake data for testing.**
**Our scripts generate real data by running your trained model.**

For actual scientific evaluation:
1. ✓ Use real PDB structures as test cases (acceptable)
2. ✓ Run your trained model on them (required)
3. ✓ Record actual optimization results (required)
4. ✗ Don't use hash-based fake data (not valid)

**Run the 3-step workflow above to get real evaluation data!**
