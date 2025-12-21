# Proper Evaluation Workflow: From Training to Real Evaluation Data

## Critical Distinction: What "Synthetic" Really Means

Let me clarify what's **acceptable** vs **NOT acceptable** for scientific evaluation:

### ✅ ACCEPTABLE: Synthetic Benchmark Problems
**Definition:** Using existing PDB structures or computationally generated protein backbones as test cases.

**Why it's acceptable:**
- These are **real protein structures** (experimental or validated)
- You're testing whether your model can design sequences for these **real structural challenges**
- Standard practice in protein design research
- Examples: CASP targets, PDB structures, designed backbones

**This is what the shell scripts do:** They use real PDB files from `proteinmpnn/inputs/` as test structures.

### ❌ NOT ACCEPTABLE: Fake Optimization Results
**Definition:** Making up numbers for optimization trajectories, energies, and convergence behavior.

**Why it's NOT acceptable:**
- Doesn't reflect actual model performance
- Can't detect bugs, convergence issues, or failure modes
- Invalidates scientific conclusions
- This is what the shell scripts do with hash-based generation

**The shell script "synthetic data" is ONLY for infrastructure testing, NOT real evaluation.**

---

## The Proper Training → Evaluation Pipeline

Here's how you should actually evaluate your trained models:

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (You've done this)                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Train energy model on protein structures                │
│ 2. Save checkpoint: best_model.pt                          │
│ 3. Model learns E(backbone, sequence) → stability score    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: INFERENCE (Generate REAL evaluation data)         │
├─────────────────────────────────────────────────────────────┤
│ 1. Load trained model checkpoint                           │
│ 2. Select test protein backbones (real PDB structures)     │
│ 3. Run IRED optimization on each backbone                  │
│ 4. Save REAL optimization trajectories & results           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: EVALUATION (Analyze real performance)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Load real optimization data from Phase 2                │
│ 2. Run comprehensive evaluation analysis                   │
│ 3. Generate performance metrics, plots, insights           │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step: Generate REAL Evaluation Data

### Step 1: Prepare Test Backbones (Synthetic benchmarks are OK)

```python
"""
prepare_test_set.py - Create a test set of protein backbones
This is the ONLY part where "synthetic" is acceptable - the test structures.
"""

import json
from pathlib import Path
from glob import glob

# Use real PDB structures as test cases
pdb_dir = Path("proteinmpnn/inputs")
test_pdbs = []

# Categorize by difficulty based on structure type
difficulty_mapping = {
    "PDB_monomers": "easy",      # Single chain proteins
    "PDB_complexes": "medium",   # Multi-chain complexes
    "PDB_homooligomers": "hard"  # Symmetric assemblies
}

for category, difficulty in difficulty_mapping.items():
    category_dir = pdb_dir / category
    if category_dir.exists():
        for pdb_file in category_dir.glob("*.pdb"):
            test_pdbs.append({
                "id": pdb_file.stem,
                "pdb_path": str(pdb_file),
                "difficulty": difficulty,
                "category": category,
                "type": "real_structure"
            })

# Save test set definition
with open("evaluation_data/benchmark_problems.json", "w") as f:
    json.dump(test_pdbs, f, indent=2)

print(f"Created test set with {len(test_pdbs)} structures")
# Example output:
# Created test set with 47 structures
#   - easy: 18 monomers
#   - medium: 20 complexes
#   - hard: 9 homooligomers
```

**This is acceptable:** You're defining real structural challenges to test your model against.

---

### Step 2: Run REAL Optimization (Generate REAL data)

This is the critical step. You must **actually run your trained model** on test structures.

```python
"""
generate_evaluation_data.py - Run trained model on test set to generate REAL evaluation data
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

# Import your trained components
from models.mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from inference.ired_optimizer import IREDSequenceOptimizer, OptimizationConfig
from inference.design_pipeline import ProteinDesignPipeline, PipelineConfig

def generate_real_evaluation_data(
    checkpoint_path: str,
    test_set_path: str,
    output_dir: str,
    num_samples_per_structure: int = 1
):
    """
    Generate REAL evaluation data by actually running trained model on test structures.

    This is what you MUST do for proper evaluation. No fake data, no hash-based generation.
    We're running the actual optimization algorithm with your trained model.
    """

    # Load test set
    with open(test_set_path) as f:
        test_structures = json.load(f)

    print(f"Running evaluation on {len(test_structures)} test structures...")
    print(f"This will generate REAL optimization data from your trained model.\n")

    # Initialize design pipeline with your trained model
    config = PipelineConfig(
        encoder_checkpoint=checkpoint_path,  # Your trained model
        energy_model_checkpoint=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_proteinmpnn_init=True,
        save_trajectories=True,  # CRITICAL: Save full trajectories
        num_designs_per_target=num_samples_per_structure,
        random_seed=42  # For reproducibility
    )

    pipeline = ProteinDesignPipeline(config)

    # Storage for REAL evaluation data
    optimization_results = []
    landscape_data = []

    # Run REAL optimization on each test structure
    for structure in tqdm(test_structures, desc="Running optimization"):
        pdb_path = structure['pdb_path']
        problem_id = structure['id']

        print(f"\nOptimizing {problem_id} ({structure['difficulty']})...")

        try:
            # THIS IS THE KEY: Actually run your trained model
            result = pipeline.design_sequence(
                backbone_path=pdb_path,
                num_designs=num_samples_per_structure
            )

            # Extract REAL optimization data from actual run
            for opt_idx, opt_result in enumerate(result.optimization_results):

                # This is REAL data from your model's actual optimization
                optimization_results.append({
                    "problem_id": f"{problem_id}_{opt_idx}",
                    "successful": result.success,
                    "design_quality": float(result.validation_metrics.get('quality_score', 0.0)) if result.validation_metrics else 0.0,
                    "confidence_score": float(result.confidence_scores[opt_idx]) if result.confidence_scores is not None else 0.0,
                    "computation_time": result.total_time,

                    # Problem metadata
                    "problem_info": {
                        "type": structure['category'],
                        "difficulty": structure['difficulty'],
                        "pdb_file": pdb_path,
                        "structure_id": problem_id
                    },

                    # REAL optimization result from your model
                    "optimization_result": {
                        "converged": opt_result.converged,
                        "total_steps_used": opt_result.total_steps,
                        "final_energy": float(opt_result.final_energy),
                        "adaptive_extensions_count": opt_result.landscapes_used - 3,  # Assuming 3 base landscapes
                        "optimization_failed": opt_result.optimization_failed,
                        "failure_reason": opt_result.failure_reason
                    },

                    # REAL trajectory from actual optimization
                    "trajectory": {
                        "energy": [float(step['energy']) for step in opt_result.trajectory],
                        "gradient_norms": [float(step.get('grad_norm', 0.0)) for step in opt_result.trajectory],
                        "landscape_indices": [int(step['landscape_idx']) for step in opt_result.trajectory],
                        "step_types": [step.get('step_type', 'regular') for step in opt_result.trajectory]
                    }
                })

                # Extract landscape quality data from actual optimization
                for landscape_idx in range(opt_result.landscapes_used):
                    landscape_steps = [s for s in opt_result.trajectory if s['landscape_idx'] == landscape_idx]

                    if landscape_steps:
                        landscape_data.append({
                            "landscape_id": f"{problem_id}_landscape_{landscape_idx}",
                            "temperature": float(landscape_steps[0].get('temperature', 1.0)),
                            "landscape_index": landscape_idx,
                            "problem_id": problem_id,

                            # REAL landscape characteristics from optimization
                            "smoothness_score": calculate_smoothness(landscape_steps),
                            "gradient_coherence": calculate_gradient_coherence(landscape_steps),
                            "energy_improvement": float(landscape_steps[-1]['energy'] - landscape_steps[0]['energy'])
                        })

            print(f"  ✓ Success: {result.success}")
            print(f"  ✓ Final energy: {result.energies[0]:.3f}")
            print(f"  ✓ Convergence: {result.optimization_results[0].converged}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            # Record failure in evaluation data
            optimization_results.append({
                "problem_id": problem_id,
                "successful": False,
                "problem_info": {
                    "type": structure['category'],
                    "difficulty": structure['difficulty'],
                    "pdb_file": pdb_path
                },
                "optimization_result": {
                    "converged": False,
                    "total_steps_used": 0,
                    "final_energy": float('inf'),
                    "optimization_failed": True,
                    "failure_reason": str(e)
                },
                "trajectory": {"energy": []}
            })

    # Save REAL evaluation data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "optimization_results.json", "w") as f:
        json.dump(optimization_results, f, indent=2)

    with open(output_path / "landscape_data.json", "w") as f:
        json.dump(landscape_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"REAL evaluation data generated:")
    print(f"  - {len(optimization_results)} optimization runs")
    print(f"  - {len(landscape_data)} landscape analyses")
    print(f"  - Success rate: {sum(r['successful'] for r in optimization_results) / len(optimization_results):.1%}")
    print(f"  - Average convergence rate: {sum(r['optimization_result']['converged'] for r in optimization_results) / len(optimization_results):.1%}")
    print(f"{'='*60}\n")

    return optimization_results, landscape_data


def calculate_smoothness(trajectory_steps):
    """Calculate energy landscape smoothness from real trajectory"""
    energies = [step['energy'] for step in trajectory_steps]
    if len(energies) < 2:
        return 0.0

    # Smoothness = inverse of energy variance
    energy_diffs = [abs(energies[i+1] - energies[i]) for i in range(len(energies)-1)]
    return 1.0 / (1.0 + torch.tensor(energy_diffs).std().item())


def calculate_gradient_coherence(trajectory_steps):
    """Calculate gradient coherence from real trajectory"""
    gradients = [step.get('grad_norm', 0.0) for step in trajectory_steps]
    if len(gradients) < 2:
        return 0.0

    # Coherence = consistency of gradient directions (simplified)
    # In real implementation, would use actual gradient vectors
    grad_changes = [abs(gradients[i+1] - gradients[i]) for i in range(len(gradients)-1)]
    return 1.0 / (1.0 + torch.tensor(grad_changes).mean().item())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate REAL evaluation data from trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test-set", default="evaluation_data/benchmark_problems.json")
    parser.add_argument("--output-dir", default="evaluation_data")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per structure")

    args = parser.parse_args()

    generate_real_evaluation_data(
        checkpoint_path=args.checkpoint,
        test_set_path=args.test_set,
        output_dir=args.output_dir,
        num_samples_per_structure=args.num_samples
    )
```

**Usage:**
```bash
# Generate REAL evaluation data from your trained model
python generate_evaluation_data.py \
    --checkpoint checkpoints/best_model.pt \
    --test-set evaluation_data/benchmark_problems.json \
    --output-dir evaluation_data \
    --num-samples 3

# This will:
# 1. Load your TRAINED model from checkpoint
# 2. Run ACTUAL optimization on each test structure
# 3. Save REAL optimization trajectories and results
# 4. Generate REAL landscape quality data from actual runs
```

---

### Step 3: Run Evaluation Analysis (Same as before)

Now that you have **real data** from actual model runs, run the evaluation:

```bash
python hybrid/evaluation/run_comprehensive_evaluation.py \
    --optimization-data evaluation_data/optimization_results.json \
    --landscape-data evaluation_data/landscape_data.json \
    --benchmark-data evaluation_data/benchmark_problems.json \
    --output-dir evaluation_results \
    --verbose
```

---

## Why This Matters: Real vs Fake Data

### With Fake Data (Shell Script Approach):
```json
{
  "converged": true,  // <-- Randomly generated, not real
  "final_energy": -5.432,  // <-- Hash-based, not from your model
  "total_steps_used": 87   // <-- Made up, not actual optimization
}
```
**Problems:**
- Doesn't test if your model actually works
- Can't detect convergence failures
- Can't identify which problem types are hard
- Can't optimize hyperparameters
- Useless for scientific publication

### With Real Data (Proper Approach):
```json
{
  "converged": true,  // <-- Actually converged in optimization
  "final_energy": -8.234,  // <-- Real energy from your trained model
  "total_steps_used": 142,  // <-- Actual steps your optimizer took
  "trajectory": {  // <-- Real optimization trajectory
    "energy": [-2.1, -4.3, -6.2, -7.8, -8.234],
    "gradient_norms": [0.45, 0.32, 0.18, 0.09, 0.02]
  }
}
```
**Benefits:**
- Tests actual model performance
- Reveals convergence issues
- Identifies problematic structures
- Enables hyperparameter tuning
- Publishable scientific results

---

## Complete Evaluation Workflow Script

Here's a complete script you can run:

```bash
#!/bin/bash
# complete_evaluation_workflow.sh
# Proper evaluation workflow from trained model to results

set -e

echo "=========================================="
echo "PROPER EVALUATION WORKFLOW"
echo "=========================================="

# Step 1: Prepare test set (synthetic benchmarks OK)
echo "Step 1: Preparing test set from real PDB structures..."
python prepare_test_set.py \
    --pdb-dir proteinmpnn/inputs \
    --output evaluation_data/benchmark_problems.json

# Step 2: Generate REAL evaluation data
echo "Step 2: Running trained model on test set (this takes time)..."
python generate_evaluation_data.py \
    --checkpoint checkpoints/best_model.pt \
    --test-set evaluation_data/benchmark_problems.json \
    --output-dir evaluation_data \
    --num-samples 3

# Step 3: Run comprehensive evaluation analysis
echo "Step 3: Analyzing real performance data..."
python hybrid/evaluation/run_comprehensive_evaluation.py \
    --optimization-data evaluation_data/optimization_results.json \
    --landscape-data evaluation_data/landscape_data.json \
    --benchmark-data evaluation_data/benchmark_problems.json \
    --output-dir evaluation_results \
    --verbose

echo "=========================================="
echo "EVALUATION COMPLETE"
echo "Results: evaluation_results/"
echo "=========================================="
```

---

## FAQ

### Q: Can I use the shell script synthetic data for testing?
**A:** Yes, but ONLY for:
- Testing the evaluation script infrastructure
- Debugging data format issues
- Verifying the evaluation pipeline runs

**Never** use it for:
- Scientific conclusions
- Model comparison
- Publication results
- Performance analysis

### Q: How long does real data generation take?
**A:** Depends on:
- Number of test structures: 10-100 typical
- Model complexity: Energy head evaluation speed
- Optimization steps: 50-200 per structure typically
- Hardware: GPU vs CPU

Expect: **30 minutes to 2 hours** for a thorough evaluation on 50 structures with GPU.

### Q: What if my model fails on some structures?
**A:** That's exactly what evaluation should reveal! Record the failures:
```json
{
  "successful": false,
  "optimization_result": {
    "converged": false,
    "optimization_failed": true,
    "failure_reason": "Energy exploded to NaN at step 23",
    "total_steps_used": 23
  }
}
```
This tells you your model has problems - fix them!

### Q: Do I need real experimental data?
**A:** No, you're confusing two things:
- **Test structures**: Real PDB files are fine (not "fake")
- **Optimization results**: Must come from **running your model** (not made up)

Using PDB structures as test cases is standard. Making up optimization results is not.

---

## Summary

### The Three-Level Hierarchy:

1. **Benchmark Problems** (Synthetic OK)
   - Real PDB structures as test cases
   - Defines what structural challenges to solve
   - Standard practice in protein design

2. **Optimization Execution** (Must be Real)
   - Run your actual trained model
   - Perform actual IRED optimization
   - Record what really happens

3. **Evaluation Analysis** (Analyzes Real Results)
   - Aggregate real performance metrics
   - Identify real failure modes
   - Draw valid scientific conclusions

### Key Principle:

**Test on synthetic benchmarks, but measure real performance.**

The shell script "synthetic data" conflates levels 1 and 2 - it generates fake results instead of running real optimization. That's acceptable for infrastructure testing but NOT for evaluating model performance.

For proper evaluation: **Run your trained model on test structures and record what actually happens.**
