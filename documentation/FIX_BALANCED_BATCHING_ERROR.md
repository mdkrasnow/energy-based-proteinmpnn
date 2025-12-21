# Fix: Dataset Too Small for Balanced Batching

## Problem

Training failed with the error:
```
Training failed: Dataset too small for balanced batching.
Need at least 4 positive and 4 negative samples
```

### Root Cause

The development configuration had conflicting parameters that resulted in insufficient training data:

1. **Limited PDB files**: `max_files_debug: 4` - Only 4 protein structures loaded
2. **High validation split**: `val_split: 0.3` - 30% of data reserved for validation
3. **Batch size**: `batch_size: 4` - Requires 2 positive + 2 negative samples per batch

**The Math:**
- 4 PDB files generate ~8 samples total (4 positive + 4 negative)
- With 30% validation split: ~2-3 samples go to validation
- Training set left with only ~5 positive + 2 negative samples
- BalancedBatchSampler with batch_size=4 needs at least 2 of each class
- **Problem**: Only 2 negative samples total → Cannot create even 1 balanced batch

### Why Balanced Batching is Important

The `BalancedBatchSampler` ensures each batch contains equal numbers of positive (stable) and negative (unstable) sequence examples. This is critical for contrastive learning because:

1. **Prevents class imbalance**: Without balanced batches, the model might see mostly positive or mostly negative examples in a batch
2. **Stabilizes gradients**: Equal representation ensures consistent gradient signals
3. **Improves convergence**: Contrastive loss works best when comparing positive and negative pairs

## Solution

Modified `hybrid/training/config_dev.json` with three changes:

### Change 1: Reduce Batch Size (4 → 2)
```json
"training": {
    "batch_size": 2,  // Changed from 4
    // ...
}
```

**Impact**: Now requires only 1 positive + 1 negative sample per batch instead of 2 + 2

### Change 2: Increase Dataset Size (4 → 10 files)
```json
"data": {
    "max_files_debug": 10,  // Changed from 4
    // ...
}
```

**Impact**: Generates ~20 samples total (10 positive + 10 negative)

### Change 3: Reduce Validation Split (30% → 20%)
```json
"data": {
    "val_split": 0.2,  // Changed from 0.3
    // ...
}
```

**Impact**: Keeps 80% of data for training instead of 70%

## Expected Outcome

With these changes:
- **Total samples**: ~20 (10 positive + 10 negative)
- **Validation set**: 20% = 4 samples (2 positive + 2 negative)
- **Training set**: 80% = 16 samples (8 positive + 8 negative)
- **Batches possible**: 8 positive ÷ 1 per batch = 8 balanced batches ✅

The BalancedBatchSampler can now create 8 batches of size 2 (1 positive + 1 negative each).

## Alternative Solutions

If you still encounter this error, consider these alternatives:

### Option 1: Further Reduce Batch Size
```json
"batch_size": 1  // Extreme case - not recommended (poor gradient estimates)
```

### Option 2: Disable Balanced Batching (For Very Small Datasets)

Edit `hybrid/training/train_energy.py` around line 683:

```python
# Instead of BalancedBatchSampler, use regular random sampling
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Regular random shuffling
    num_workers=num_workers,
    collate_fn=collate_fn,
    drop_last=True
)
```

**Warning**: This removes the guarantee of balanced batches and may hurt training quality.

### Option 3: Use More Data Files

For development testing, increase `max_files_debug`:
```json
"max_files_debug": 20  // More data = more stable training
```

### Option 4: Reduce Validation Split to Minimum

Keep almost all data for training:
```json
"val_split": 0.1  // Only 10% for validation
```

## Verification

After applying the fix, you should see this output during training:

```
Analyzing dataset for balanced sampling...
BalancedBatchSampler found: 8 positive, 8 negative samples
Creating 8 balanced batches of size 2
```

If you still see an error, check:
1. How many PDB files are actually in `proteinmpnn/inputs/PDB_monomers/pdbs/`
2. Whether any PDB files are being skipped due to parsing errors
3. The actual training/validation split counts in the logs

## Best Practices for Dev vs Production

### Development Configuration
- **Purpose**: Fast iteration, quick failure detection
- **Dataset size**: Small (10-20 structures)
- **Batch size**: Small (2-4)
- **Validation split**: Small (10-20%)
- **Epochs**: Few (3-5)

### Production Configuration
- **Purpose**: Full model training for publication
- **Dataset size**: Large (500+ structures)
- **Batch size**: Larger (16-32)
- **Validation split**: Standard (20-30%)
- **Epochs**: Many (50-100+)

## Summary

✅ **Fixed** `config_dev.json` to work with small dev dataset
✅ **Batch size**: 4 → 2 (requires fewer samples per batch)
✅ **Dataset size**: 4 → 10 files (more training data)
✅ **Validation split**: 30% → 20% (keeps more data for training)
✅ **Expected result**: 8 balanced batches of size 2

The dev training should now proceed without the balanced batching error.
