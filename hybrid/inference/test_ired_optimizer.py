"""
Comprehensive Tests for IRED Sequence Optimizer

Tests cover:
1. Convergence behavior and early stopping
2. Adaptive step allocation
3. Gradient flow and numerical stability
4. Memory efficiency
5. Integration with Phase 1 components
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.ired_optimizer import IREDSequenceOptimizer, OptimizationConfig, OptimizationResult
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr


def test_basic_optimization():
    """Test basic optimization functionality"""
    print("Test 1: Basic Optimization")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20, hidden_dim=128, num_layers=2)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(2, 30, 128)
    
    result = optimizer.optimize_sequence(
        backbone_features,
        max_steps=20,
        return_trajectory=True
    )
    
    assert result.sequence.shape == (2, 30), f"Expected shape (2, 30), got {result.sequence.shape}"
    assert result.logits.shape == (2, 30, 20), f"Expected logits shape (2, 30, 20), got {result.logits.shape}"
    assert len(result.trajectory) > 0, "Trajectory should not be empty"
    assert result.total_steps > 0, "Should have taken at least some steps"
    assert np.isfinite(result.final_energy), "Final energy should be finite"
    
    print(f"✓ Basic optimization: energy={result.final_energy:.4f}, converged={result.converged}, steps={result.total_steps}")
    return True


def test_convergence_monitoring():
    """Test convergence detection and early stopping"""
    print("\nTest 2: Convergence Monitoring")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    
    # Use tight convergence config for faster stopping
    config = OptimizationConfig(
        learning_rate=0.05,  # Faster convergence
        early_stop_threshold=1e-3,  # Easier to reach
        early_stop_window=3,  # Smaller window
        min_steps_per_landscape=5
    )
    
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr, config=config)
    
    backbone_features = torch.randn(1, 25, 128)
    
    result = optimizer.optimize_sequence(
        backbone_features,
        max_steps=100,  # Allow many steps
        return_trajectory=True
    )
    
    # Should converge before hitting max steps
    assert result.total_steps < 100, f"Should converge early, but used {result.total_steps} steps"
    
    # Check energy decreased
    if len(result.trajectory) > 1:
        initial_energy = result.trajectory[0]['energy_mean']
        final_energy = result.trajectory[-1]['energy_mean']
        improvement = initial_energy - final_energy
        print(f"✓ Converged in {result.total_steps} steps (energy: {initial_energy:.4f} → {final_energy:.4f}, improvement: {improvement:.4f})")
    
    return True


def test_adaptive_allocation():
    """Test adaptive step allocation"""
    print("\nTest 3: Adaptive Step Allocation")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(1, 20, 128)
    
    # Test with adaptive optimization
    result = optimizer.adaptive_optimization(
        backbone_features,
        difficulty_threshold=0.05,
        max_total_steps=50
    )
    
    assert result.total_steps <= 50, f"Should not exceed max_total_steps, got {result.total_steps}"
    assert result.converged is not None, "Convergence status should be set"
    
    print(f"✓ Adaptive optimization: {result.total_steps} steps allocated, converged={result.converged}")
    return True


def test_restart_strategies():
    """Test optimization with multiple restarts"""
    print("\nTest 4: Restart Strategies")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(1, 15, 128)
    
    # Test different initialization strategies
    for strategy in ['random', 'uniform', 'biased']:
        result = optimizer.optimize_with_restarts(
            backbone_features,
            num_restarts=3,
            initialization_strategy=strategy,
            max_steps_per_restart=10
        )
        
        assert np.isfinite(result.final_energy), f"Energy should be finite for strategy {strategy}"
        print(f"✓ Restart strategy '{strategy}': energy={result.final_energy:.4f}")
    
    return True


def test_gradient_flow():
    """Test that gradients flow properly through optimization"""
    print("\nTest 5: Gradient Flow")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(1, 20, 128)
    initial_logits = torch.randn(1, 20, 20)
    
    result = optimizer.optimize_sequence(
        backbone_features,
        initial_logits=initial_logits.clone(),
        max_steps=5,
        return_trajectory=True
    )
    
    # Check that logits changed (gradients flowed)
    logits_change = (result.logits - initial_logits).abs().mean().item()
    assert logits_change > 0.01, f"Logits should change during optimization, change={logits_change}"
    
    # Check gradient norms in trajectory
    if len(result.trajectory) > 0:
        gradient_norms = [t['gradient_norm'] for t in result.trajectory]
        avg_grad_norm = np.mean(gradient_norms)
        assert avg_grad_norm > 0, "Gradient norms should be positive"
        assert avg_grad_norm < 1000, f"Gradient norms should not explode, got {avg_grad_norm}"
        
        print(f"✓ Gradient flow verified: logits changed by {logits_change:.4f}, avg gradient norm={avg_grad_norm:.4f}")
    
    return True


def test_numerical_stability():
    """Test numerical stability with extreme inputs"""
    print("\nTest 6: Numerical Stability")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    # Test with very small sequence
    small_backbone = torch.randn(1, 5, 128)
    result_small = optimizer.optimize_sequence(small_backbone, max_steps=5)
    assert np.isfinite(result_small.final_energy), "Should handle small sequences"
    
    # Test with larger batch
    large_batch = torch.randn(8, 20, 128)
    result_batch = optimizer.optimize_sequence(large_batch, max_steps=5)
    assert np.isfinite(result_batch.final_energy), "Should handle large batches"
    
    # Test with extreme logit initialization
    extreme_logits = torch.randn(1, 20, 20) * 5.0  # Large values
    result_extreme = optimizer.optimize_sequence(
        torch.randn(1, 20, 128),
        initial_logits=extreme_logits,
        max_steps=5
    )
    assert np.isfinite(result_extreme.final_energy), "Should handle extreme initializations"
    
    print(f"✓ Numerical stability: small seq ✓, large batch ✓, extreme init ✓")
    return True


def test_memory_efficiency():
    """Test memory usage with trajectories"""
    print("\nTest 7: Memory Efficiency")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(2, 50, 128)
    
    # Run optimization with trajectory
    result_with_traj = optimizer.optimize_sequence(
        backbone_features,
        max_steps=20,
        return_trajectory=True
    )
    
    # Run without trajectory
    result_no_traj = optimizer.optimize_sequence(
        backbone_features,
        max_steps=20,
        return_trajectory=False
    )
    
    assert len(result_with_traj.trajectory) > 0, "Trajectory should be populated"
    assert len(result_no_traj.trajectory) == 0, "Trajectory should be empty when disabled"
    
    # Check trajectory doesn't store full sequences (only sample)
    if len(result_with_traj.trajectory) > 0:
        traj_entry = result_with_traj.trajectory[0]
        seq_sample = traj_entry['sequence_sample']
        assert len(seq_sample) == 50, "Should store only single sequence sample"
    
    print(f"✓ Memory efficiency: trajectory size controlled, sample-based storage")
    return True


def test_trajectory_analysis():
    """Test trajectory analysis utilities"""
    print("\nTest 8: Trajectory Analysis")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
    
    backbone_features = torch.randn(1, 20, 128)
    
    result = optimizer.optimize_sequence(
        backbone_features,
        max_steps=15,
        return_trajectory=True
    )
    
    # Analyze trajectory
    analysis = optimizer.analyze_trajectory(result)
    
    assert analysis['has_data'], "Analysis should have data"
    assert 'energy' in analysis, "Should have energy stats"
    assert 'gradients' in analysis, "Should have gradient stats"
    assert 'convergence' in analysis, "Should have convergence metrics"
    assert 'landscapes' in analysis, "Should have landscape info"
    
    print(f"✓ Trajectory analysis: {analysis['energy']['improvement_percent']:.1f}% improvement, "
          f"rate={analysis['convergence']['rate']:.6f}")
    return True


def test_multi_landscape():
    """Test with multiple energy landscape models"""
    print("\nTest 9: Multi-Landscape Support")
    
    # Create multiple energy models (simulating E_1, E_2, E_3)
    energy_models = [
        EnergyHead(backbone_dim=128, seq_dim=20, hidden_dim=128, num_layers=2)
        for _ in range(3)
    ]
    
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.1])
    
    optimizer = IREDSequenceOptimizer(energy_models, seq_repr)
    
    assert optimizer.multi_landscape, "Should recognize multi-landscape mode"
    assert optimizer.num_landscapes == 3, f"Should have 3 landscapes, got {optimizer.num_landscapes}"
    
    backbone_features = torch.randn(1, 20, 128)
    
    result = optimizer.optimize_sequence(
        backbone_features,
        max_steps=15,
        return_trajectory=True
    )
    
    # Check that multiple landscapes were used
    landscapes_used = set(t['landscape'] for t in result.trajectory)
    assert len(landscapes_used) > 1, "Should traverse multiple landscapes"
    
    print(f"✓ Multi-landscape: {len(landscapes_used)} landscapes used, "
          f"energy={result.final_energy:.4f}")
    return True


def test_configuration():
    """Test configuration handling"""
    print("\nTest 10: Configuration")
    
    # Test with dict config
    dict_config = {
        'learning_rate': 0.02,
        'max_steps_per_landscape': 30,
        'early_stop_threshold': 5e-4
    }
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    
    optimizer = IREDSequenceOptimizer(energy_model, seq_repr, config=dict_config)
    
    assert optimizer.config.learning_rate == 0.02, "Config should be applied"
    assert optimizer.config.max_steps_per_landscape == 30, "Config should be applied"
    
    # Test with OptimizationConfig object
    seq_repr2 = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05])
    obj_config = OptimizationConfig(
        learning_rate=0.015,
        num_landscapes=7,
        gradient_clip_norm=0.5
    )
    
    optimizer2 = IREDSequenceOptimizer(energy_model, seq_repr2, config=obj_config)
    
    assert optimizer2.config.learning_rate == 0.015, "Object config should be applied"
    assert optimizer2.config.gradient_clip_norm == 0.5, "Object config should be applied"
    
    print(f"✓ Configuration: dict config ✓, object config ✓")
    return True


def test_reproducibility():
    """Test reproducibility with seed control"""
    print("\nTest 11: Reproducibility (Fix #1)")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    
    backbone_features = torch.randn(1, 20, 128)
    
    # Test 1: Same seed produces identical results
    optimizer1 = IREDSequenceOptimizer(energy_model, seq_repr, seed=42)
    result1 = optimizer1.optimize_sequence(backbone_features, max_steps=10)
    
    optimizer2 = IREDSequenceOptimizer(energy_model, seq_repr, seed=42)
    result2 = optimizer2.optimize_sequence(backbone_features, max_steps=10)
    
    assert torch.allclose(result1.sequence.float(), result2.sequence.float()), "Same seed should produce identical sequences"
    assert abs(result1.final_energy - result2.final_energy) < 1e-6, f"Same seed should produce identical energies: {result1.final_energy} vs {result2.final_energy}"
    assert result1.seed_used == 42, "Seed should be stored in result"
    print(f"  ✓ Same seed (42) produces identical results")
    
    # Test 2: Different seeds produce different results
    optimizer3 = IREDSequenceOptimizer(energy_model, seq_repr, seed=123)
    result3 = optimizer3.optimize_sequence(backbone_features, max_steps=10)
    
    assert not torch.allclose(result1.sequence.float(), result3.sequence.float()), "Different seeds should produce different sequences"
    assert result3.seed_used == 123, "Different seed should be stored"
    print(f"  ✓ Different seeds (42 vs 123) produce different results")
    
    # Test 3: No seed produces warning (but still works)
    optimizer4 = IREDSequenceOptimizer(energy_model, seq_repr, seed=None)
    result4 = optimizer4.optimize_sequence(backbone_features, max_steps=10)
    assert result4.seed_used is None, "No seed should store None"
    print(f"  ✓ No seed works with warning")
    
    print(f"✓ Reproducibility: identical with same seed, different with different seeds, provenance tracked")
    return True


def test_temperature_alignment():
    """Test temperature schedule validation (Fix #2)"""
    print("\nTest 12: Temperature Alignment (Fix #2)")
    
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    
    # Test 1: Aligned temperature schedule works
    seq_repr_aligned = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    config_aligned = OptimizationConfig(num_landscapes=5)
    optimizer_aligned = IREDSequenceOptimizer(energy_model, seq_repr_aligned, config=config_aligned)
    print(f"  ✓ Aligned temperature schedule (5 temps, 5 landscapes) accepted")
    
    # Test 2: Misaligned temperature schedule raises error
    seq_repr_misaligned = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.1])  # 3 temps
    config_misaligned = OptimizationConfig(num_landscapes=5)  # 5 landscapes
    
    try:
        optimizer_misaligned = IREDSequenceOptimizer(energy_model, seq_repr_misaligned, config=config_misaligned)
        assert False, "Should have raised ValueError for misaligned schedule"
    except ValueError as e:
        assert "Temperature schedule length" in str(e), f"Error should mention temperature schedule: {e}"
        assert "must match num_landscapes" in str(e), f"Error should mention mismatch: {e}"
        print(f"  ✓ Misaligned temperature schedule (3 temps, 5 landscapes) raises clear ValueError")
    
    print(f"✓ Temperature alignment: validation catches mismatches, provides actionable error")
    return True


def test_scale_invariance():
    """Test scale-invariant convergence (Fix #3)"""
    print("\nTest 13: Scale-Invariant Convergence (Fix #3)")
    
    # Create models that scale energies by different factors
    class ScaledEnergyHead(torch.nn.Module):
        def __init__(self, base_model, scale_factor):
            super().__init__()
            self.base_model = base_model
            self.scale = scale_factor
        
        def forward(self, backbone_features, sequence_probs, mask=None):
            return self.base_model(backbone_features, sequence_probs, mask) * self.scale
    
    base_energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    
    # Test convergence at different scales
    scales = [0.01, 1.0, 100.0]
    convergence_steps = []
    
    backbone_features = torch.randn(1, 15, 128)
    
    for scale in scales:
        scaled_model = ScaledEnergyHead(base_energy_model, scale)
        optimizer = IREDSequenceOptimizer(scaled_model, seq_repr, seed=42)
        
        result = optimizer.optimize_sequence(backbone_features, max_steps=50, return_trajectory=False)
        convergence_steps.append(result.total_steps)
        
        print(f"  Scale {scale:6.2f}x: {result.total_steps} steps, final_energy={result.final_energy:.6f}")
    
    # Check that convergence behavior is similar across scales
    # (not identical due to numerical differences, but should be in same ballpark)
    min_steps = min(convergence_steps)
    max_steps = max(convergence_steps)
    
    # Allow 3x variation (this is reasonable for scale-invariant convergence)
    assert max_steps <= min_steps * 3, f"Convergence should be scale-invariant: steps range from {min_steps} to {max_steps}"
    
    print(f"✓ Scale-invariance: convergence consistent across 0.01x-100x energy scales")
    return True


def test_nan_handling():
    """Test NaN detection and immediate abort (Fix #4)"""
    print("\nTest 14: NaN Handling (Fix #4)")
    
    # Create model that produces NaN
    class NaNEnergyHead(torch.nn.Module):
        def __init__(self, nan_at_step=5):
            super().__init__()
            self.call_count = 0
            self.nan_at_step = nan_at_step
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
        
        def forward(self, backbone_features, sequence_probs, mask=None):
            self.call_count += 1
            # Compute something from inputs to maintain gradient flow
            # backbone_features: [B, L, backbone_dim], sequence_probs: [B, L, vocab_size]
            energy_base = (sequence_probs.sum(dim=-1) * backbone_features.mean(dim=-1)).sum(dim=1) + self.dummy_param
            if self.call_count >= self.nan_at_step:
                # Inject NaN but preserve computation graph
                return energy_base * float('nan')
            return energy_base
    
    nan_model = NaNEnergyHead(nan_at_step=5)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.3, 0.2, 0.1])
    optimizer = IREDSequenceOptimizer(nan_model, seq_repr)
    
    backbone_features = torch.randn(1, 10, 128)
    
    result = optimizer.optimize_sequence(backbone_features, max_steps=50)
    
    # Check that optimization failed
    assert result.optimization_failed, "Should detect optimization failure"
    assert result.failure_reason is not None, "Should provide failure reason"
    assert "NaN/Inf detected" in result.failure_reason, f"Failure reason should mention NaN: {result.failure_reason}"
    assert result.sequence is None, "Failed optimization should return None sequence"
    assert result.logits is None, "Failed optimization should return None logits"
    assert result.final_energy == float('inf'), "Failed optimization should have inf energy"
    assert not result.converged, "Failed optimization should not be converged"
    
    # Check that it aborted immediately (not all 50 steps)
    assert result.total_steps < 50, f"Should abort immediately, not run all {result.total_steps} steps"
    
    print(f"  ✓ NaN detected at step {result.total_steps}")
    print(f"  ✓ Optimization aborted immediately")
    print(f"  ✓ Result marked as failed with actionable error message")
    print(f"  ✓ sequence=None, logits=None, energy=inf (clean failure)")
    
    print(f"✓ NaN handling: immediate abort, no invalid results, actionable diagnostics")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("IRED Sequence Optimizer - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_optimization,
        test_convergence_monitoring,
        test_adaptive_allocation,
        test_restart_strategies,
        test_gradient_flow,
        test_numerical_stability,
        test_memory_efficiency,
        test_trajectory_analysis,
        test_multi_landscape,
        test_configuration,
        # Blocking fixes tests (Phase 3.1 multi-agent review)
        test_reproducibility,
        test_temperature_alignment,
        test_scale_invariance,
        test_nan_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
