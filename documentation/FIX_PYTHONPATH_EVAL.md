# Fix: PYTHONPATH Missing in Dev Evaluation

## Problem

After fixing the config and data generation issues, dev evaluation failed with:
```
WARNING: Could not import model components: No module named 'models'
❌ FATAL ERROR: Model components not available
```

## Root Cause

The `eval_hybrid_proteinmpnn_dev.sh` script was **missing PYTHONPATH setup** that's required for Python to find the project modules:
- `models` (energy_head, sequence_repr, mpnn_encoder)
- `inference` (ired_optimizer)
- `data` (vocab, stability_dataset)
- `proteinmpnn` utilities

The **training script** has this setup, but it was missing from the **evaluation script**.

## Solution

Added PYTHONPATH setup to `eval_hybrid_proteinmpnn_dev.sh` after module loading:

```bash
# Set up PYTHONPATH to include project modules
export PYTHONPATH="$(pwd)/proteinmpnn:$(pwd):$(pwd)/hybrid${PYTHONPATH:+:${PYTHONPATH}}"
echo "Updated PYTHONPATH: $PYTHONPATH"
```

This adds three directories to PYTHONPATH:
1. `$(pwd)/proteinmpnn` - ProteinMPNN utilities
2. `$(pwd)` - Project root (for `hybrid.*` imports)
3. `$(pwd)/hybrid` - Hybrid module directory

## Files Modified

- **`eval_hybrid_proteinmpnn_dev.sh`** (lines 205-207)

## Expected Behavior

Now the evaluation data generation script can successfully import:

```python
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from inference.ired_optimizer import IREDSequenceOptimizer
from data.vocab import AMINO_ACID_ALPHABET
```

## All Fixes Applied (Summary)

For the dev evaluation to work, we needed **4 fixes**:

1. ✅ **Remove invalid config param** (`fast_dev_run`)
2. ✅ **Use absolute paths** in config (not relative)
3. ✅ **Replace fake data with real optimization** results
4. ✅ **Add PYTHONPATH setup** for module imports

All fixes are now complete!
