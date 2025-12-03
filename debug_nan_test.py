#!/usr/bin/env python3
"""Test script to trigger NaN/Inf error and see debugging output."""

import torch
import torch.nn.functional as F
from hybrid.models.sequence_repr import ContinuousSequenceRepr
from hybrid.models.energy_head import EnergyHead

def test_nan_inf_trigger():
    print("Testing NaN/Inf trigger in sequence representation...")
    
    # Create sequence repr with problematic temperature schedule
    sequence_repr = ContinuousSequenceRepr(
        vocab_size=20,
        temperature_schedule=[1.0, 1e-6, 1e-8],  # Very small temperatures
        min_temperature=1e-8,
        max_temperature=10.0
    )
    
    # Create problematic logits
    batch_size, seq_len = 2, 50
    
    # Test 1: Large logits with tiny temperature
    print("\n=== TEST 1: Large logits with tiny temperature ===")
    large_logits = torch.randn(batch_size, seq_len, 20) * 20.0  # Very large range
    print(f"Input logits: min={large_logits.min().item():.4f}, max={large_logits.max().item():.4f}")
    
    try:
        result = sequence_repr(large_logits, landscape_idx=2, training=True)
        print(f"Result: min={result.min().item():.6f}, max={result.max().item():.6f}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 2: NaN input
    print("\n=== TEST 2: NaN input logits ===")
    nan_logits = torch.randn(batch_size, seq_len, 20)
    nan_logits[0, 0, 0] = float('nan')
    
    try:
        result = sequence_repr(nan_logits, landscape_idx=0, training=False)
        print(f"Result: min={result.min().item():.6f}, max={result.max().item():.6f}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # Test 3: Extreme temperature with division
    print("\n=== TEST 3: Normal logits with extreme temperature division ===")
    normal_logits = torch.randn(batch_size, seq_len, 20)
    
    # Manually simulate the problematic division
    temperature = torch.tensor(1e-8)
    scaled_logits = normal_logits / temperature
    print(f"Scaled logits: min={scaled_logits.min().item():.6f}, max={scaled_logits.max().item():.6f}")
    
    try:
        result_softmax = F.softmax(scaled_logits, dim=-1)
        print(f"Softmax result: min={result_softmax.min().item():.6f}, max={result_softmax.max().item():.6f}")
        print(f"Softmax NaN: {torch.isnan(result_softmax).any().item()}, Inf: {torch.isinf(result_softmax).any().item()}")
    except Exception as e:
        print(f"Exception: {e}")

    # Test 4: Test energy head with problematic sequence_probs
    print("\n=== TEST 4: Energy head with problematic sequence_probs ===")
    backbone_features = torch.randn(batch_size, seq_len, 128)
    
    # Create sequence_probs with NaN
    sequence_probs = torch.randn(batch_size, seq_len, 20)
    sequence_probs = F.softmax(sequence_probs, dim=-1)
    sequence_probs[0, 0, 0] = float('nan')  # Inject NaN
    
    energy_head = EnergyHead(backbone_dim=128, seq_dim=20, hidden_dim=256)
    
    try:
        result = energy_head(backbone_features, sequence_probs)
        print(f"Energy result: {result}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_nan_inf_trigger()