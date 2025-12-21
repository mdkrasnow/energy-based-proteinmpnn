# Real vs Synthetic Evaluation Data: Quick Reference

## The Problem You Had

Your shell scripts always generated **synthetic (fake) data**, even though they said "real PDB structures":

```bash
# Lines 291-352 in eval_hybrid_proteinmpnn_dev.sh
hash_val=$(echo "$filename" | cksum | cut -d' ' -f1)
echo "\"final_energy\": -$((10 + hash_val % 50))"  # <-- FAKE!
```

This data is **useless for scientific evaluation**.

---

## The Solution

We created a **smart data generation system** that:

```
┌─────────────────────────────────────────────────────────┐
│  SMART EVALUATION DATA GENERATOR                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Check: Trained models available?                    │
│     ├─ YES → Run REAL optimization → REAL data ✓       │
│     └─ NO  → Generate synthetic → SYNTHETIC data ⚠️      │
│                                                          │
│  2. Label data source clearly                           │
│  3. Warn user if using synthetic                        │
│  4. Proceed with evaluation                             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## What You Get

### Scenario 1: After Training (Real Data)

```
$ sbatch eval_hybrid_proteinmpnn.sh

[Job Output:]
✓ Found 3 trained model(s)
✓ Real data generation script available
✓ PyTorch dependencies available

========================================
GENERATING REAL EVALUATION DATA
========================================
This will run your TRAINED MODEL on test structures

Running real optimization on structures...
[1/47] 1A2Y (easy)...
  Structure: 89 residues
    ✓ Sample 1: converged=true, steps=142, energy=-8.234

...

✓ REAL evaluation data generated successfully!
  Total runs: 47
  Successful: 41 (87.2%)
  Failed: 6 (12.8%)
  Source: Actual model optimization

Files:
  - optimization_results.json (REAL DATA ✓)
  - landscape_data.json (REAL DATA ✓)
  - benchmark_problems.json
```

**What this means:**
- Your model actually ran on test structures
- Numbers reflect real performance
- Can publish these results
- Can identify actual failure modes
- Can optimize hyperparameters

---

### Scenario 2: Before Training (Synthetic Data)

```
$ sbatch eval_hybrid_proteinmpnn_dev.sh

[Job Output:]
⚠️  Cannot generate REAL data: No trained models found

========================================
GENERATING SYNTHETIC EVALUATION DATA
========================================
⚠️  WARNING: Using hash-based synthetic data
  Reason: No trained models found in ./trained_models

This data is for INFRASTRUCTURE TESTING ONLY.
For real scientific evaluation, you need:
  1. Trained model checkpoints
  2. PyTorch environment configured

✓ Synthetic evaluation data generated
  Source: Hash-based deterministic generation
  Use case: Infrastructure testing ONLY

Files:
  - optimization_results.json (SYNTHETIC ⚠️)
  - landscape_data.json (SYNTHETIC ⚠️)
  - benchmark_problems.json
```

**What this means:**
- Made up numbers for testing
- Evaluation infrastructure works
- NOT for scientific conclusions
- Need to train models first

---

## Data Format Comparison

### Real Data (After Integration):
```json
{
  "problem_id": "1A2Y_sample_0",
  "successful": true,
  "problem_info": {
    "difficulty": "medium",
    "pdb_file": "/path/to/1A2Y.pdb",
    "source": "real_optimization"  ⭐ REAL
  },
  "optimization_result": {
    "converged": true,
    "total_steps_used": 142,  ⭐ Actual steps
    "final_energy": -8.234567,  ⭐ Real energy
    "optimization_failed": false,
    "failure_reason": null
  },
  "trajectory": {
    "energy": [  ⭐ Full real trajectory
      -2.134, -3.821, -4.556, -5.892, -6.723,
      -7.234, -7.689, -7.923, -8.089, -8.234
    ],
    "gradient_norms": [0.45, 0.32, 0.18, ...],
    "landscape_indices": [0, 0, 0, 1, 1, ...]
  }
}
```

### Synthetic Data (Current Behavior):
```json
{
  "problem_id": "1A2Y",
  "successful": true,
  "problem_info": {
    "difficulty": "medium",
    "pdb_file": "/path/to/1A2Y.pdb",
    "source": "synthetic_hash_based"  ⭐ FAKE
  },
  "optimization_result": {
    "converged": true,
    "total_steps_used": 87,  ⭐ Random number
    "final_energy": -45.3,  ⭐ Hash-based
    "adaptive_extensions_count": 2  ⭐ Random
  },
  "trajectory": {
    "energy": [-5, -10, -15]  ⭐ Minimal fake data
  }
}
```

---

## Quick Integration

### 1. Make scripts executable:
```bash
chmod +x scripts/generate_real_eval_data.py
chmod +x scripts/generate_eval_data_smart.sh
```

### 2. Modify `eval_hybrid_proteinmpnn_dev.sh` line 269:

**Replace:**
```bash
# Create evaluation data from real PDB structures (dev version)
echo "Creating evaluation data from real PDB structures..."
[... 80 lines of hash-based generation ...]
```

**With:**
```bash
# Smart evaluation data generation (real or synthetic)
source "$REPO_DIR/scripts/generate_eval_data_smart.sh"
export MAX_EVAL_STRUCTURES=10
generate_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"
```

### 3. Modify `eval_hybrid_proteinmpnn.sh` line 262:

**Same replacement**, but with:
```bash
export MAX_EVAL_STRUCTURES=500  # Production limit
```

### 4. Test it:
```bash
sbatch eval_hybrid_proteinmpnn_dev.sh
```

---

## Decision Tree

```
                    ┌─────────────────┐
                    │  Evaluation Job │
                    │     Starts      │
                    └────────┬────────┘
                             │
                    ┌────────▼─────────┐
                    │ Trained models   │
                    │   available?     │
                    └────┬────────┬────┘
                         │        │
                    YES  │        │  NO
                         │        │
         ┌───────────────▼──┐  ┌──▼──────────────────┐
         │  REAL DATA MODE  │  │ SYNTHETIC DATA MODE │
         │                  │  │                     │
         │ 1. Load model    │  │ 1. Warn user       │
         │ 2. Run optimizer │  │ 2. Generate hashes │
         │ 3. Save results  │  │ 3. Label as fake   │
         │                  │  │                     │
         │ Time: 30-60 min  │  │ Time: 1 minute     │
         │ Quality: REAL ✓  │  │ Quality: FAKE ⚠️    │
         └───────────────┬──┘  └──┬──────────────────┘
                         │        │
                    ┌────▼────────▼────┐
                    │  Continue with   │
                    │   Evaluation     │
                    │   (knows type)   │
                    └──────────────────┘
```

---

## Key Differences

| Aspect | Current (Synthetic Only) | New (Smart) |
|--------|-------------------------|-------------|
| **Data Source** | Hash-based generation | Trained model inference |
| **Computation Time** | <1 minute | 30-60 minutes |
| **Energy Values** | Random integers | Actual model output |
| **Trajectories** | 3 fake points | Full optimization path |
| **Convergence** | Random true/false | Actual convergence |
| **Failure Detection** | Can't detect | Detects real failures |
| **Scientific Value** | None (infrastructure test) | Publication-ready |
| **Requires** | Nothing | Trained models |
| **Fallback** | N/A | Auto-fallback to synthetic |
| **User Notification** | Misleading | Clear warnings |

---

## When to Use Each

### Use Real Data (Automatic when available):
- ✅ After model training completes
- ✅ For actual performance evaluation
- ✅ For scientific publications
- ✅ For hyperparameter tuning
- ✅ For identifying model issues
- ✅ For comparing to baselines

### Use Synthetic Data (Automatic fallback):
- ⚠️ Infrastructure testing only
- ⚠️ Before models are trained
- ⚠️ Testing evaluation scripts
- ⚠️ CI/CD pipeline validation
- ⚠️ Quick smoke tests
- ⚠️ **NEVER for science**

---

## Verification Checklist

After integration, verify:

1. **Check job output logs:**
   ```bash
   tail -100 eval_hybrid_proteinmpnn_*.out
   ```
   Look for: "GENERATING REAL EVALUATION DATA" or "SYNTHETIC"

2. **Check JSON source markers:**
   ```bash
   jq '.[0].problem_info.source' evaluation_data/optimization_results.json
   ```
   Should show: `"real_optimization"` or `"synthetic_hash_based"`

3. **Check trajectory detail:**
   ```bash
   jq '.[0].trajectory.energy | length' evaluation_data/optimization_results.json
   ```
   Real data: 100+ points, Synthetic: 3 points

4. **Check energy precision:**
   ```bash
   jq '.[0].optimization_result.final_energy' evaluation_data/optimization_results.json
   ```
   Real: `-8.234567` (high precision), Synthetic: `-45.3` (rounded)

---

## Summary

**Before Integration:**
- Always generates synthetic data ❌
- Misleading ("real PDB structures") ❌
- Can't detect actual model issues ❌
- Useless for science ❌

**After Integration:**
- Generates real data when possible ✅
- Clear labeling (real vs synthetic) ✅
- Detects actual model performance ✅
- Scientific publication-ready ✅
- Automatic fallback for testing ✅

**Integration time:** 5 minutes
**Scientific value:** Enormous

See `INTEGRATE_REAL_EVAL_DATA.md` for detailed integration steps.
