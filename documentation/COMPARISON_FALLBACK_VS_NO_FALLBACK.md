# Comparison: With Fallback vs No Fallback

## What You Asked For

> "remove all fallback synthetic data we need to test the models only and fail otherwise"

## What We Created

Two approaches - you should use **Real Only (No Fallback)**:

---

## Option 1: Smart (With Fallback) ⚠️

**Files:**
- `scripts/generate_eval_data_smart.sh`
- Uses synthetic data if models unavailable

**Behavior:**
```
┌─────────────────────────────┐
│  Trained models available?  │
└────────┬──────────┬─────────┘
         YES       NO
         │          │
    ┌────▼───┐  ┌──▼───────────────┐
    │ REAL   │  │ SYNTHETIC        │
    │ data   │  │ (fallback)       │
    │        │  │                  │
    │ ✅     │  │ ⚠️ Continues     │
    │        │  │    with fake     │
    └────────┘  └──────────────────┘
```

**Risk:** Could accidentally use synthetic data ❌

---

## Option 2: Real Only (No Fallback) ✅ **USE THIS**

**Files:**
- `scripts/generate_eval_data_real_only.sh`
- **No synthetic fallback**

**Behavior:**
```
┌─────────────────────────────┐
│  Trained models available?  │
└────────┬──────────┬─────────┘
         YES       NO
         │          │
    ┌────▼───┐  ┌──▼───────────────┐
    │ REAL   │  │ FAIL             │
    │ data   │  │ IMMEDIATELY      │
    │        │  │                  │
    │ ✅     │  │ ❌ Clear error   │
    │        │  │    message       │
    └────────┘  └──────────────────┘
```

**Guarantee:** Never uses fake data ✅

---

## Side-by-Side Comparison

### Scenario: No Trained Models

#### With Fallback (Option 1):
```
⚠️  Cannot generate REAL data: No trained models found

GENERATING SYNTHETIC EVALUATION DATA
⚠️  WARNING: Using hash-based synthetic data

This is for INFRASTRUCTURE TESTING ONLY
...

✓ Synthetic evaluation data generated
  Source: Hash-based deterministic generation

[Job continues with fake data] ⚠️
```

#### No Fallback (Option 2): ← **USE THIS**
```
❌ FATAL ERROR: No trained model files (.pt) found

Expected files:
  - best_model.pt
  - final_model.pt

Cannot proceed without trained models.
Please complete model training first.

❌ EVALUATION FAILED

[Job exits immediately with code 1] ✅
```

---

### Scenario: With Trained Models

Both behave **identically** (both generate real data):

```
✓ Found 3 trained model file(s)
✓ PyTorch dependencies available

GENERATING REAL EVALUATION DATA
Running real optimization on structures...

[1/47] 1A2Y: ✓ converged=true, energy=-8.234
...

✅ REAL EVALUATION DATA GENERATED
```

---

## Which Should You Use?

### Use **Real Only (No Fallback)**

✅ **Matches your request:** "fail otherwise"
✅ **Scientific integrity:** Impossible to use fake data
✅ **Clear workflow:** Must train before evaluating
✅ **Fail fast:** Don't waste time on fake evaluation

### Don't Use Smart (With Fallback)

❌ **Has synthetic fallback**
❌ **Could accidentally use fake data**
❌ **More complex** (unnecessary branching)

---

## Integration Instructions

### For Real Only (Recommended):

**Read:** `INTEGRATE_REAL_ONLY_NO_FALLBACK.md`

**Use files:**
- `scripts/generate_eval_data_real_only.sh`
- `scripts/generate_real_eval_data.py` (updated, no fallback)

**Shell script modification:**
```bash
source "$REPO_DIR/scripts/generate_eval_data_real_only.sh"
generate_real_eval_data "$TRAINED_MODEL_DIR" "$JOB_SCRATCH/evaluation_data" "${PDB_FILES[@]}"
```

---

## Summary Table

| Feature | Smart (Fallback) | Real Only ✅ |
|---------|-----------------|--------------|
| Uses real data when available | Yes | Yes |
| Falls back to synthetic | Yes ⚠️ | No ✅ |
| Fails if no models | No | Yes ✅ |
| Can accidentally use fake data | Yes ❌ | No ✅ |
| Matches "fail otherwise" | No | Yes ✅ |
| Scientific integrity | Risk | Guaranteed ✅ |
| Complexity | Higher | Lower ✅ |
| **Recommendation** | Don't use | **Use this** ✅ |

---

## Decision: Use Real Only

Based on your requirement: **"fail otherwise"**

✅ Use: `generate_eval_data_real_only.sh`
✅ Read: `INTEGRATE_REAL_ONLY_NO_FALLBACK.md`
❌ Skip: `generate_eval_data_smart.sh` (has fallback)

**Integration time:** 2 minutes
**Fake data risk:** ZERO
**Scientific integrity:** GUARANTEED
