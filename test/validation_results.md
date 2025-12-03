# ProteinMPNN Encoder Validation Results

**Date:** November 26, 2025  
**Stage:** Stage 1 (24-Hour Critical Path Validation)  
**Status:** CRITICAL ISSUES IDENTIFIED - PROCEED TO STAGE 2**

## Executive Summary

Stage 1 validation has revealed **critical issues** that prevent the hybrid ProteinMPNN encoder from being used in production. The encoder implementation appears fundamentally incorrect, with outputs differing significantly from the reference ProteinMPNN implementation.

**Recommendation:** Proceed immediately to Stage 2 to implement targeted fixes for the identified critical issues.

## Detailed Test Results

### ✅ PASSED TESTS

#### 1. Checkpoint Integrity (3/3 models)
- **vanilla/v_48_020**: ✅ PASSED - Model loads correctly, all components present
- **ca_model/v_48_020**: ✅ PASSED - Model loads correctly, all components present  
- **soluble/v_48_020**: ✅ PASSED - Model loads correctly, all components present
- **Verification**: All models have expected encoder layers (3), embedding dim (128), proper freeze status

#### 2. Vocabulary Consistency (2/3 models)  
- **vanilla/v_48_020**: ✅ PASSED - Vocabulary size 21, consistent encoder/decoder
- **ca_model/v_48_020**: ❌ FAILED - Coordinate format incompatibility  
- **soluble/v_48_020**: ✅ PASSED - Vocabulary size 21, consistent encoder/decoder

#### 3. Numerical Stability (2/3 models)
- **vanilla/v_48_020**: ✅ PASSED - No NaN/Inf values, reasonable ranges [-100, 100]
- **ca_model/v_48_020**: ❌ FAILED - Coordinate format incompatibility
- **soluble/v_48_020**: ✅ PASSED - No NaN/Inf values, reasonable ranges [-100, 100]

### ❌ FAILED TESTS

#### 1. Output Equivalence (0/3 models) - **CRITICAL ISSUE**
- **vanilla/v_48_020**: ❌ **CRITICAL FAILURE** 
  - **Max relative difference**: 1.45 (target: < 1e-5)
  - **Issue**: Hybrid encoder outputs differ by >100% from reference ProteinMPNN
  - **Impact**: Encoder is fundamentally incorrect - cannot be used for research
  
- **ca_model/v_48_020**: ❌ FAILED - Coordinate format error (secondary issue)
- **soluble/v_48_020**: ❌ FAILED - Output comparison skipped due to vanilla failure

**Root Cause Analysis**: The wrapper encoder implementation has a fundamental flaw in either:
1. Model parameter initialization or loading
2. Forward pass logic or tensor handling
3. Component extraction from the full ProteinMPNN model

#### 2. CA Model Compatibility (0/2 tests) - **ARCHITECTURAL ISSUE**
- **Error**: `RuntimeError: The size of tensor a (106) must match the size of tensor b (3) at non-singleton dimension 3`
- **Root Cause**: CA-only models expect different coordinate input format (CA atoms only) but wrapper provides full backbone coordinates (N, CA, C, O)
- **Impact**: CA model variants unusable until coordinate handling is fixed

#### 3. Gradient Flow (0/3 models) - **TRAINING ISSUE**
- **Error**: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
- **Root Cause**: Freezing mechanism completely detaches computation graph, preventing any gradient computation
- **Impact**: Fine-tuning capabilities non-functional

## Critical Issues Summary

| Issue | Severity | Models Affected | Impact |
|-------|----------|----------------|--------|
| **Output Equivalence Failure** | **CRITICAL** | All vanilla/soluble | Research results would be invalid |
| **CA Model Incompatibility** | **HIGH** | CA models only | 33% of model types unusable |
| **Gradient Flow Failure** | **MEDIUM** | All models | Fine-tuning impossible |

## Stage 2 Requirements

Based on Stage 1 findings, Stage 2 **MUST** address:

### 1. Output Equivalence Fix (CRITICAL PRIORITY)
- **Investigate**: Model parameter loading, forward pass implementation
- **Root Cause**: Identify why hybrid encoder differs from reference by 145%
- **Validate**: Achieve <1e-5 relative difference on all test structures
- **Timeline**: 2-3 days maximum

### 2. CA Model Compatibility Fix (HIGH PRIORITY)  
- **Implement**: Conditional coordinate handling for CA-only vs full backbone
- **Modify**: `convert_parsed_pdb_to_batch` to handle CA-only format
- **Update**: Encoder wrapper to detect and handle CA models correctly
- **Timeline**: 1 day

### 3. Gradient Flow Fix (MEDIUM PRIORITY)
- **Redesign**: Freezing mechanism to preserve computation graph
- **Implement**: Selective parameter freezing instead of full detachment
- **Test**: Both frozen and unfrozen modes for gradient flow
- **Timeline**: 0.5-1 day

## Reference Data Status

✅ **Reference generation successful** for validation framework:
- Generated: 2/3 model types (vanilla, soluble)  
- Failed: CA models due to coordinate format issue
- Files: `ref_vanilla_v_48_020_5L33_seed42.npz`, `ref_soluble_v_48_020_5L33_seed42.npz`
- Validation: All reference files verified with correct shapes [1, 106, 128]

## Next Steps

1. **Immediate**: Investigate output equivalence failure in vanilla model
2. **Day 1-2**: Implement fixes for output equivalence (critical path)
3. **Day 2-3**: Fix CA model compatibility and gradient flow
4. **Day 3**: Re-run Stage 1 validation to verify all fixes
5. **Day 4**: If validation passes, proceed to production use

## Technical Details

### Test Environment
- **Structures tested**: 5L33.pdb (106 residues)
- **Models tested**: vanilla, ca_model, soluble (v_48_020)
- **Tolerance**: 1e-5 relative difference for output equivalence
- **Framework**: pytest with custom validation fixtures

### Error Details
```
Output mismatch for 5L33.pdb: max_relative_diff=1.45e+00 > 1e-05
RuntimeError: The size of tensor a (106) must match the size of tensor b (3) at non-singleton dimension 3
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

---

**Validation Performed By**: Claude Code Stage 1 Validation Suite  
**Report Generated**: 2025-11-26