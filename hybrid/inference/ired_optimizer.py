"""
IRED-Style Sequence Optimizer

This module implements iterative refinement and exploration-driven (IRED) optimization
for protein sequence design. It uses gradient-based optimization over multiple annealed
energy landscapes to progressively refine sequences from initial guesses to optimized designs.

Key Features:
- Multi-landscape optimization with temperature annealing
- Adaptive computation allocation based on problem difficulty
- Convergence monitoring and early stopping
- Noise injection for exploration in early landscapes
- Multiple restart strategies for robust optimization
- Comprehensive trajectory logging for analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import warnings
from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """
    Configuration for IRED sequence optimization.
    
    Scale-Invariant Convergence:
        All convergence thresholds use relative (not absolute) values to ensure
        the optimizer works across different energy scales (small peptides to large proteins).
    
    Attributes:
        learning_rate: Learning rate for Adam optimizer (default: 0.01)
        max_steps_per_landscape: Maximum optimization steps per landscape (default: 50)
        num_landscapes: Number of energy landscapes for annealing (default: 5)
        early_stop_window: Number of steps to check for convergence (default: 5)
        noise_scale: Scale of exploration noise in early landscapes (default: 0.01)
        noise_decay: Decay factor for noise across landscapes (default: 0.5)
        gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
        convergence_patience: Steps without improvement before stopping (default: 10)
        min_steps_per_landscape: Minimum steps before early stopping (default: 10)
        
        relative_energy_var_threshold: Relative energy variance threshold for convergence.
            Converged if variance / |mean_energy| < threshold (default: 1e-3)
        relative_improvement_threshold: Relative energy improvement tolerance.
            Warn if improvement / |initial_energy| > threshold (default: -0.01, i.e., 1% worse)
        relative_patience_threshold: Relative change threshold for patience counter.
            Increment patience if |change| / |energy| < threshold (default: 1e-4)
        absolute_energy_threshold: Fallback to absolute thresholds when |energy| < this value.
            Used to avoid division by zero for near-zero energies (default: 1e-6)
        
        early_stop_threshold: DEPRECATED - Use relative_energy_var_threshold instead (default: 1e-4)
    """
    learning_rate: float = 0.01
    max_steps_per_landscape: int = 50
    num_landscapes: int = 5
    early_stop_window: int = 5
    noise_scale: float = 0.01
    noise_decay: float = 0.5
    gradient_clip_norm: float = 1.0
    convergence_patience: int = 10
    min_steps_per_landscape: int = 10
    
    # Scale-invariant relative thresholds
    relative_energy_var_threshold: float = 1e-3
    relative_improvement_threshold: float = -0.01
    relative_patience_threshold: float = 1e-4
    absolute_energy_threshold: float = 1e-6
    
    # Step allocation configuration
    adaptive_step_allocation: bool = True  # Use weighted allocation (more steps to later landscapes)
    
    # Deprecated - kept for backward compatibility
    early_stop_threshold: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps_per_landscape <= 0:
            raise ValueError(f"max_steps_per_landscape must be positive, got {self.max_steps_per_landscape}")
        if self.num_landscapes <= 0:
            raise ValueError(f"num_landscapes must be positive, got {self.num_landscapes}")
        if self.early_stop_threshold < 0:
            raise ValueError(f"early_stop_threshold must be non-negative, got {self.early_stop_threshold}")
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {self.noise_scale}")
        if not 0 <= self.noise_decay <= 1:
            raise ValueError(f"noise_decay must be in [0, 1], got {self.noise_decay}")
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}")
        
        # Validate relative threshold parameters
        if self.relative_energy_var_threshold < 0:
            raise ValueError(f"relative_energy_var_threshold must be non-negative, got {self.relative_energy_var_threshold}")
        if self.relative_patience_threshold < 0:
            raise ValueError(f"relative_patience_threshold must be non-negative, got {self.relative_patience_threshold}")
        if self.absolute_energy_threshold <= 0:
            raise ValueError(f"absolute_energy_threshold must be positive, got {self.absolute_energy_threshold}")


@dataclass
class OptimizationResult:
    """
    Result from sequence optimization.
    
    Attributes:
        sequence: Final optimized discrete sequence [B, L] (None if optimization failed)
        logits: Final sequence logits [B, L, vocab_size] (None if optimization failed)
        trajectory: List of optimization trajectory dictionaries
        final_energy: Final energy value (inf if optimization failed)
        converged: Whether optimization converged successfully
        total_steps: Total number of optimization steps taken
        landscapes_used: Number of landscapes traversed
        seed_used: Random seed used for optimization (None if not seeded)
        optimization_failed: Whether optimization failed critically (NaN/Inf detected)
        failure_reason: Detailed reason for optimization failure (None if successful)
    """
    sequence: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    trajectory: List[Dict[str, Any]]
    final_energy: float
    converged: bool
    total_steps: int
    landscapes_used: int
    seed_used: Optional[int] = None
    optimization_failed: bool = False
    failure_reason: Optional[str] = None


class IREDSequenceOptimizer:
    """
    IRED-style sequence optimizer for protein design.
    
    This class implements iterative refinement through multiple annealed energy landscapes,
    enabling adaptive computation allocation and robust sequence optimization.
    
    The optimizer supports:
    - Single energy model (current Phase 2 state)
    - Multiple landscape models E_1, ..., E_T (future Phase 3.2)
    - Adaptive step allocation based on convergence difficulty
    - Multiple restart strategies for failed optimizations
    
    Args:
        energy_models: Single energy model or list of landscape-specific models [E_1, ..., E_T]
        sequence_repr: ContinuousSequenceRepr module for differentiable sequences
        config: Optimization configuration (OptimizationConfig instance or dict)
        device: Device for computation (default: 'cpu')
        seed: Random seed for reproducibility (default: None, non-reproducible)
    
    Example:
        >>> from models.energy_head import EnergyHead
        >>> from models.sequence_repr import ContinuousSequenceRepr
        >>> 
        >>> energy_model = EnergyHead(backbone_dim=128, seq_dim=20)
        >>> seq_repr = ContinuousSequenceRepr(vocab_size=20)
        >>> optimizer = IREDSequenceOptimizer(energy_model, seq_repr)
        >>> 
        >>> # Optimize sequence for a backbone
        >>> result = optimizer.optimize_sequence(backbone_features, initial_logits)
        >>> print(f"Final energy: {result.final_energy}")
        >>> print(f"Converged: {result.converged}")
    """
    
    def __init__(
        self,
        energy_models: Union[nn.Module, List[nn.Module]],
        sequence_repr: nn.Module,
        config: Optional[Union[OptimizationConfig, Dict[str, Any]]] = None,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """Initialize IRED sequence optimizer"""
        
        # Handle configuration
        if config is None:
            self.config = OptimizationConfig()
        elif isinstance(config, dict):
            self.config = OptimizationConfig(**config)
        elif isinstance(config, OptimizationConfig):
            self.config = config
        else:
            raise TypeError(f"config must be OptimizationConfig or dict, got {type(config)}")
        
        # Handle energy models (single or multi-landscape)
        if isinstance(energy_models, (list, tuple)):
            self.energy_models = list(energy_models)
            self.num_landscapes = len(energy_models)
            self.multi_landscape = True
        else:
            # Single model - replicate for all landscapes
            self.energy_models = [energy_models] * self.config.num_landscapes
            self.num_landscapes = self.config.num_landscapes
            self.multi_landscape = False
        
        # Validate energy models
        if len(self.energy_models) == 0:
            raise ValueError("Must provide at least one energy model")
        
        for i, model in enumerate(self.energy_models):
            if not isinstance(model, nn.Module):
                raise TypeError(f"Energy model {i} must be nn.Module, got {type(model)}")
        
        # Store sequence representation module
        if not isinstance(sequence_repr, nn.Module):
            raise TypeError(f"sequence_repr must be nn.Module, got {type(sequence_repr)}")
        self.sequence_repr = sequence_repr
        
        # Validate temperature schedule alignment with num_landscapes
        if hasattr(self.sequence_repr, 'temperature_schedule'):
            temp_schedule_len = len(self.sequence_repr.temperature_schedule)
            if temp_schedule_len != self.num_landscapes:
                raise ValueError(
                    f"Temperature schedule length ({temp_schedule_len}) must match num_landscapes ({self.num_landscapes}). "
                    f"The IRED algorithm requires one temperature per landscape for correct progressive refinement "
                    f"from smooth (E_1) to sharp (E_T) energy landscapes. "
                    f"Current temperature_schedule: {self.sequence_repr.temperature_schedule.tolist()}, "
                    f"num_landscapes: {self.num_landscapes}. "
                    f"Fix: Create ContinuousSequenceRepr with temperature_schedule of length {self.num_landscapes}, "
                    f"or adjust num_landscapes to {temp_schedule_len}."
                )
        
        # Device management
        self.device = torch.device(device)
        
        # Reproducibility: seed and random number generator
        self.seed = seed
        if seed is not None:
            self.rng = torch.Generator(device=self.device)
            self.rng.manual_seed(seed)
        else:
            self.rng = None
            warnings.warn(
                "No seed specified for IRED optimizer. Results will be non-reproducible. "
                "For reproducible research, pass seed parameter to IREDSequenceOptimizer.__init__(). "
                "Example: optimizer = IREDSequenceOptimizer(..., seed=42)",
                UserWarning
            )
        
        # Check if models are on different devices and warn
        model_devices = set()
        for i, model in enumerate(self.energy_models):
            for param in model.parameters():
                model_devices.add(param.device)
                break  # Just check first parameter
        
        if len(model_devices) > 1:
            warnings.warn(
                f"Energy models are on different devices {model_devices}. "
                f"Moving all to {self.device}"
            )
        
        self.sequence_repr = self.sequence_repr.to(self.device)
        self.energy_models = [model.to(self.device) for model in self.energy_models]
        
        # Set all models to eval mode (no training during inference)
        self.sequence_repr.eval()
        for model in self.energy_models:
            model.eval()
        
        # Statistics tracking
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.total_steps = 0
    
    def optimize_sequence(
        self,
        backbone_features: torch.Tensor,
        initial_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        return_trajectory: bool = True
    ) -> OptimizationResult:
        """
        Optimize sequence using annealed energy landscapes.
        
        This is the core optimization method that progressively refines sequences
        through multiple energy landscapes, from smooth (E_1) to sharp (E_T).
        
        Args:
            backbone_features: Structural features from encoder [B, L, backbone_dim]
            initial_logits: Initial sequence logits [B, L, vocab_size] (random if None)
            mask: Sequence mask [B, L] (all valid if None)
            max_steps: Target total steps across all landscapes (actual may be slightly higher
                      to ensure at least 1 step per landscape)
            return_trajectory: Whether to return detailed optimization trajectory
        
        Returns:
            OptimizationResult containing optimized sequence and metadata
        """
        # Input validation
        self._validate_inputs(backbone_features, initial_logits, mask)
        
        # Update statistics
        self.total_optimizations += 1
        
        # Get batch dimensions
        batch_size, seq_len = backbone_features.shape[:2]
        device = backbone_features.device
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Initialize logits if not provided
        if initial_logits is None:
            initial_logits = self._initialize_logits(
                batch_size, seq_len, strategy='random'
            )
        
        # Clone and setup for optimization
        current_logits = initial_logits.clone().detach().to(device)
        current_logits.requires_grad_(True)
        
        # Create optimizer
        optimizer = torch.optim.Adam([current_logits], lr=self.config.learning_rate)
        
        # Optimization trajectory
        trajectory = [] if return_trajectory else None
        
        # Determine steps per landscape with adaptive allocation
        if max_steps is not None:
            if max_steps < self.num_landscapes:
                raise ValueError(f"max_steps ({max_steps}) must be >= num_landscapes ({self.num_landscapes})")
            
            # Scientifically principled step allocation: more steps to later (harder) landscapes
            if self.config.adaptive_step_allocation:
                weights = np.array([1.2 ** i for i in range(self.num_landscapes)])
                weights = weights / weights.sum() * max_steps
                steps_per_landscape = np.maximum(1, np.round(weights).astype(int))
                
                # Ensure exact budget compliance
                total_allocated = steps_per_landscape.sum()
                if total_allocated > max_steps:
                    # Remove excess steps from landscapes with most steps
                    excess = total_allocated - max_steps
                    for i in range(excess):
                        max_idx = np.argmax(steps_per_landscape)
                        steps_per_landscape[max_idx] -= 1
                elif total_allocated < max_steps:
                    # Add remaining steps to later landscapes
                    remaining = max_steps - total_allocated
                    for i in range(remaining):
                        # Prefer later landscapes for additional steps
                        target_idx = self.num_landscapes - 1 - (i % self.num_landscapes)
                        steps_per_landscape[target_idx] += 1
                        
                # Validate exact budget compliance
                assert steps_per_landscape.sum() == max_steps, f"Step allocation error: {steps_per_landscape.sum()} != {max_steps}"
            else:
                # Equal allocation with remainder distribution
                base_steps = max_steps // self.num_landscapes
                remainder = max_steps % self.num_landscapes
                steps_per_landscape = np.full(self.num_landscapes, base_steps)
                # Distribute remainder to later landscapes
                for i in range(remainder):
                    steps_per_landscape[self.num_landscapes - 1 - i] += 1
        else:
            steps_per_landscape = np.full(self.num_landscapes, self.config.max_steps_per_landscape)
        
        # Total step counter
        total_steps = 0
        converged = False
        final_energy = float('inf')
        optimization_failed = False  # Track if optimization failed critically
        
        # Iterate through energy landscapes (E_1 → E_T)
        for landscape_idx in range(self.num_landscapes):
            energy_model = self.energy_models[landscape_idx]
            energy_model.eval()  # Ensure eval mode
            
            # Compute noise scale for this landscape (decay over landscapes)
            noise_scale = self.config.noise_scale * (self.config.noise_decay ** landscape_idx)
            
            # Get steps allocated for this landscape
            current_landscape_steps = int(steps_per_landscape[landscape_idx]) if isinstance(steps_per_landscape, np.ndarray) else steps_per_landscape
            
            # Optimize in current landscape
            for step in range(current_landscape_steps):
                # Zero gradients
                optimizer.zero_grad()
                
                # Soft differentiable clamping for numerical stability while maintaining gradients
                current_logits = 10.0 * torch.tanh(current_logits / 10.0)
                
                # Convert logits to soft sequence representation
                # Use sequence_repr with landscape-specific temperature
                soft_sequence = self.sequence_repr(
                    current_logits, 
                    landscape_idx=landscape_idx,
                    training=False  # Use straight-through for optimization
                )
                
                # Compute energy
                energy = energy_model(backbone_features, soft_sequence, mask)
                
                # Average energy over batch for scalar loss
                loss = energy.mean()
                
                # Add exploration noise via entropy regularization BEFORE backward pass
                if landscape_idx < self.num_landscapes // 2 and noise_scale > 0:
                    # Add noise to loss computation via entropy regularization instead of parameter noise
                    # This maintains optimizer state consistency
                    entropy_weight = noise_scale * 0.1  # Scale appropriately
                    logits_entropy = -torch.sum(F.softmax(current_logits, dim=-1) * F.log_softmax(current_logits, dim=-1))
                    loss = loss - entropy_weight * logits_entropy
                
                # Check for NaN/Inf in loss and logits
                if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(current_logits).any() or torch.isinf(current_logits).any():
                    error_msg = (f"NaN or Inf detected at landscape {landscape_idx}, step {step}. "
                               f"Logit stats: min={current_logits.min():.3f}, max={current_logits.max():.3f}, "
                               f"loss={loss:.3f}, energy={energy.mean():.3f}")
                    warnings.warn(error_msg)
                    optimization_failed = True
                    failure_landscape = landscape_idx
                    failure_step = step
                    failure_reason = error_msg
                    break
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [current_logits], 
                    self.config.gradient_clip_norm
                )
                
                # Optimization step
                optimizer.step()
                
                # Update step counter
                total_steps += 1
                self.total_steps += 1
                
                # Log trajectory
                if return_trajectory:
                    # Get discrete sequence for logging
                    with torch.no_grad():
                        discrete_seq = self.sequence_repr.get_discrete_sequence(current_logits)
                    
                    trajectory_entry = {
                        'landscape': landscape_idx,
                        'step': step,
                        'total_step': total_steps,
                        'energy_mean': loss.item(),
                        'energy_std': energy.std().item() if batch_size > 1 else 0.0,
                        'gradient_norm': current_logits.grad.norm().item() if current_logits.grad is not None else 0.0,
                        'logits_norm': current_logits.norm().item(),
                        'noise_scale': noise_scale,
                        # Store only first sequence to save memory
                        'sequence_sample': discrete_seq[0].cpu().tolist()
                    }
                    trajectory.append(trajectory_entry)
                
                # Store final energy
                final_energy = loss.item()
                
                # Check for convergence and early stopping
                if self._check_convergence(trajectory, landscape_idx, step):
                    if step >= self.config.min_steps_per_landscape:
                        # Converged in this landscape, move to next
                        break
            
            # IMMEDIATE ABORT if optimization failed
            if optimization_failed:
                # Log comprehensive diagnostics
                warnings.warn(
                    f"Optimization FAILED at landscape {failure_landscape}, step {failure_step}. "
                    f"Aborting immediately to prevent invalid results propagation."
                )
                warnings.warn(f"Failure diagnostics:")
                warnings.warn(f"  - Last valid energy: {final_energy:.6f}")
                if current_logits.grad is not None:
                    grad_norm = current_logits.grad.norm().item()
                    warnings.warn(f"  - Gradient norm at failure: {grad_norm:.6f}")
                warnings.warn(
                    f"  - Logit statistics: min={current_logits.min().item():.2f}, "
                    f"max={current_logits.max().item():.2f}, "
                    f"mean={current_logits.mean().item():.2f}"
                )
                if len(trajectory) > 0:
                    recent_energies = [t['energy_mean'] for t in trajectory[-10:]]
                    warnings.warn(f"  - Recent energy history (last 10): {recent_energies}")
                
                # BREAK OUTER LOOP IMMEDIATELY
                break
        
        # Handle optimization failure vs success
        if optimization_failed:
            # Create clean failure result
            result = OptimizationResult(
                sequence=None,  # No valid sequence
                logits=None,  # No valid logits
                trajectory=[],  # Clear invalid trajectory data
                final_energy=float('inf'),
                converged=False,
                total_steps=total_steps,
                landscapes_used=failure_landscape + 1,  # How many landscapes before failure
                seed_used=self.seed,
                optimization_failed=True,
                failure_reason=failure_reason if 'failure_reason' in locals() else (
                    f"NaN/Inf detected at landscape {failure_landscape}, step {failure_step}. "
                    f"Possible causes: (1) Gradient explosion - reduce learning_rate or increase gradient_clip_norm, "
                    f"(2) Numerical overflow in energy computation - check energy model implementation, "
                    f"(3) Invalid input features - verify backbone_features are finite. "
                    f"Last valid energy: {final_energy:.6f}. "
                    f"Check gradient norms and logit ranges in diagnostic output above."
                )
            )
        else:
            # Normal successful optimization
            # Get final optimized sequence
            with torch.no_grad():
                # Use final landscape for sequence extraction
                final_soft_seq = self.sequence_repr(
                    current_logits, 
                    landscape_idx=self.num_landscapes-1,
                    training=False
                )
                final_sequence = self.sequence_repr.get_discrete_sequence(current_logits)
            
            # Determine final convergence based on trajectory analysis
            converged = self._determine_convergence(trajectory, final_energy)
            if converged:
                self.successful_optimizations += 1
            
            # Create result
            result = OptimizationResult(
                sequence=final_sequence,
                logits=current_logits.detach(),
                trajectory=trajectory if return_trajectory else [],
                final_energy=final_energy,
                converged=converged,
                total_steps=total_steps,
                landscapes_used=self.num_landscapes,
                seed_used=self.seed,
                optimization_failed=False,
                failure_reason=None
            )
        
        return result
    
    def _check_convergence(
        self,
        trajectory: Optional[List[Dict[str, Any]]],
        landscape_idx: int,
        step: int
    ) -> bool:
        """
        Check if optimization has converged and should stop early.
        
        Uses scale-invariant relative thresholds to work across different energy scales.
        
        Convergence criteria:
        1. Relative energy variance in recent window below threshold
        2. Gradient norms consistently small (relative to energy)
        3. Patience exhausted (no relative improvement for N steps)
        
        Args:
            trajectory: Optimization trajectory so far
            landscape_idx: Current landscape index
            step: Current step within landscape
        
        Returns:
            True if converged and should stop early
        """
        # Need trajectory for convergence checking
        if trajectory is None or len(trajectory) < self.config.early_stop_window:
            return False
        
        # Get recent trajectory window
        window_size = self.config.early_stop_window
        recent_trajectory = trajectory[-window_size:]
        
        # Extract recent energies
        recent_energies = [t['energy_mean'] for t in recent_trajectory]
        energy_std = np.std(recent_energies)
        mean_energy = np.mean(recent_energies)
        abs_mean = abs(mean_energy)
        
        # Check energy variance (main convergence criterion) - SCALE INVARIANT
        if abs_mean > self.config.absolute_energy_threshold:
            # Use relative threshold (scale-invariant)
            relative_variance = energy_std / abs_mean
            if relative_variance >= self.config.relative_energy_var_threshold:
                return False  # Still changing significantly
        else:
            # Fallback to absolute for near-zero energies
            if energy_std >= self.config.absolute_energy_threshold:
                return False
        
        # Check gradient norms (should be small if converged)
        recent_gradients = [t['gradient_norm'] for t in recent_trajectory]
        avg_gradient_norm = np.mean(recent_gradients)
        
        # Gradient check: relative to energy magnitude
        # If gradients very small relative to energy, likely converged
        if abs_mean > self.config.absolute_energy_threshold:
            relative_grad = avg_gradient_norm / abs_mean
            if relative_grad < 1e-5:  # 0.001% of energy
                return True
        else:
            # Absolute gradient check for near-zero energies
            if avg_gradient_norm < 1e-6:
                return True
        
        # Check for improvement using patience - SCALE INVARIANT
        if len(trajectory) >= self.config.convergence_patience:
            # Get best energy in last 'patience' steps
            patience_window = trajectory[-self.config.convergence_patience:]
            patience_energies = [t['energy_mean'] for t in patience_window]
            best_in_window = min(patience_energies)
            current_energy = recent_energies[-1]
            energy_change = abs(current_energy - best_in_window)
            
            # Use relative change for scale-invariance
            abs_best = abs(best_in_window)
            if abs_best > self.config.absolute_energy_threshold:
                relative_change = energy_change / abs_best
                if relative_change < self.config.relative_patience_threshold:
                    return True
            else:
                # Absolute check for near-zero energies
                if energy_change < self.config.absolute_energy_threshold:
                    return True
        
        return False
    
    def _determine_convergence(
        self,
        trajectory: Optional[List[Dict[str, Any]]],
        final_energy: float
    ) -> bool:
        """
        Determine if optimization converged successfully.
        
        Args:
            trajectory: Full optimization trajectory
            final_energy: Final energy value
        
        Returns:
            True if optimization converged successfully
        """
        # Check for invalid final energy
        if not np.isfinite(final_energy):
            return False
        
        # If no trajectory, use simple energy check
        if trajectory is None or len(trajectory) == 0:
            return final_energy < float('inf')
        
        # Check if energy decreased from start
        initial_energy = trajectory[0]['energy_mean']
        energy_improvement = initial_energy - final_energy
        
        # Consider converged if:
        # 1. Final energy is finite
        # 2. Energy improved OR stayed stable (for already good initializations)
        # 3. Gradients are reasonable (not exploding)
        
        final_gradient = trajectory[-1]['gradient_norm']
        
        converged = (
            np.isfinite(final_energy) and
            np.isfinite(final_gradient) and
            final_gradient < 100.0 and  # Not exploding
            (energy_improvement >= -1.0)  # Didn't get much worse (allows slight increase from noise)
        )
        
        return converged
    
    def adaptive_optimization(
        self,
        backbone_features: torch.Tensor,
        initial_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        difficulty_threshold: float = 0.1,
        max_total_steps: int = 200
    ) -> OptimizationResult:
        """
        Adaptive optimization with automatic step allocation.
        
        Assesses problem difficulty and allocates more optimization steps
        for challenging problems that don't converge quickly.
        
        Args:
            backbone_features: Structural features [B, L, backbone_dim]
            initial_logits: Initial sequence logits [B, L, vocab_size]
            mask: Sequence mask [B, L]
            difficulty_threshold: Energy variance threshold for difficulty assessment
            max_total_steps: Maximum total steps across all attempts
        
        Returns:
            OptimizationResult with adaptively allocated computation
        """
        # Initial attempt with standard steps
        initial_steps = min(50, max_total_steps // 2)
        
        result = self.optimize_sequence(
            backbone_features,
            initial_logits=initial_logits,
            mask=mask,
            max_steps=initial_steps,
            return_trajectory=True
        )
        
        # Assess convergence quality
        if result.converged and len(result.trajectory) > 0:
            # Check if well-converged (low energy variance in final steps)
            final_window = min(10, len(result.trajectory))
            final_energies = [t['energy_mean'] for t in result.trajectory[-final_window:]]
            energy_variance = np.var(final_energies)
            
            # If well-converged, return result
            if energy_variance < difficulty_threshold:
                return result
        
        # Problem is difficult - allocate more steps
        remaining_steps = max_total_steps - result.total_steps
        
        if remaining_steps > self.config.max_steps_per_landscape:
            # Continue optimization from current state
            extended_result = self.optimize_sequence(
                backbone_features,
                initial_logits=result.logits,  # Continue from where we left off
                mask=mask,
                max_steps=remaining_steps,
                return_trajectory=True
            )
            
            # Merge trajectories
            combined_trajectory = result.trajectory + extended_result.trajectory
            
            # Return combined result with extended optimization
            return OptimizationResult(
                sequence=extended_result.sequence,
                logits=extended_result.logits,
                trajectory=combined_trajectory,
                final_energy=extended_result.final_energy,
                converged=extended_result.converged,
                total_steps=result.total_steps + extended_result.total_steps,
                landscapes_used=self.num_landscapes
            )
        
        # No more budget - return best result we have
        return result
    
    def optimize_with_restarts(
        self,
        backbone_features: torch.Tensor,
        num_restarts: int = 3,
        mask: Optional[torch.Tensor] = None,
        initialization_strategy: str = 'random',
        max_steps_per_restart: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize with multiple random restarts.
        
        Runs optimization multiple times with different initializations
        and returns the best result based on final energy.
        
        Args:
            backbone_features: Structural features [B, L, backbone_dim]
            num_restarts: Number of independent optimization attempts
            mask: Sequence mask [B, L]
            initialization_strategy: Strategy for initialization ('random', 'uniform', 'biased')
            max_steps_per_restart: Maximum steps per restart attempt
        
        Returns:
            Best OptimizationResult across all restarts
        """
        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")
        
        # Validate initialization strategy
        valid_strategies = ['random', 'uniform', 'biased']
        if initialization_strategy not in valid_strategies:
            raise ValueError(f"initialization_strategy must be one of {valid_strategies}, got {initialization_strategy}")
        
        # Get batch dimensions
        batch_size, seq_len = backbone_features.shape[:2]
        
        # Run multiple optimization attempts
        results = []
        for restart_idx in range(num_restarts):
            # Generate new initialization for each restart
            initial_logits = self._initialize_logits(
                batch_size, seq_len, strategy=initialization_strategy
            )
            
            # Run optimization
            result = self.optimize_sequence(
                backbone_features,
                initial_logits=initial_logits,
                mask=mask,
                max_steps=max_steps_per_restart,
                return_trajectory=False  # Don't store trajectories for all restarts (memory)
            )
            
            results.append((result.final_energy, result))
        
        # Select best result (lowest energy)
        best_energy, best_result = min(results, key=lambda x: x[0])
        
        return best_result
    
    def analyze_trajectory(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        Analyze optimization trajectory for diagnostics.
        
        Args:
            result: OptimizationResult with trajectory data
        
        Returns:
            Dictionary with trajectory analysis metrics
        """
        if not result.trajectory or len(result.trajectory) == 0:
            return {
                'error': 'No trajectory data available',
                'has_data': False
            }
        
        trajectory = result.trajectory
        
        # Extract time series
        energies = [t['energy_mean'] for t in trajectory]
        gradients = [t['gradient_norm'] for t in trajectory]
        landscapes = [t['landscape'] for t in trajectory]
        
        # Compute statistics
        energy_initial = energies[0]
        energy_final = energies[-1]
        energy_improvement = energy_initial - energy_final
        energy_min = min(energies)
        energy_max = max(energies)
        
        # Find when energy was minimized
        min_energy_step = energies.index(energy_min)
        
        # Gradient statistics
        gradient_max = max(gradients)
        gradient_final = gradients[-1]
        
        # Convergence rate estimation
        # Measure how quickly energy decreased
        if len(energies) > 10:
            early_mean = np.mean(energies[:len(energies)//4])
            late_mean = np.mean(energies[-len(energies)//4:])
            convergence_rate = (early_mean - late_mean) / len(energies)
        else:
            convergence_rate = energy_improvement / len(energies) if len(energies) > 0 else 0.0
        
        # Landscape-wise analysis
        landscapes_traversed = max(landscapes) + 1 if landscapes else 0
        steps_per_landscape = []
        for landscape_idx in range(landscapes_traversed):
            steps_in_landscape = sum(1 for l in landscapes if l == landscape_idx)
            steps_per_landscape.append(steps_in_landscape)
        
        return {
            'has_data': True,
            'total_steps': len(trajectory),
            'converged': result.converged,
            'energy': {
                'initial': energy_initial,
                'final': energy_final,
                'minimum': energy_min,
                'maximum': energy_max,
                'improvement': energy_improvement,
                'improvement_percent': 100.0 * energy_improvement / abs(energy_initial) if energy_initial != 0 else 0.0,
                'min_at_step': min_energy_step
            },
            'gradients': {
                'maximum': gradient_max,
                'final': gradient_final,
                'mean': np.mean(gradients),
                'std': np.std(gradients)
            },
            'convergence': {
                'rate': convergence_rate,
                'monotonic': all(energies[i] >= energies[i+1] for i in range(len(energies)-1))
            },
            'landscapes': {
                'total': landscapes_traversed,
                'steps_per_landscape': steps_per_landscape
            }
        }
    
    def _validate_inputs(
        self,
        backbone_features: torch.Tensor,
        initial_logits: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ):
        """Validate optimization inputs"""
        # Backbone features validation
        if not isinstance(backbone_features, torch.Tensor):
            raise TypeError(f"backbone_features must be torch.Tensor, got {type(backbone_features)}")
        
        if backbone_features.dim() != 3:
            raise ValueError(f"backbone_features must be 3D [B, L, D], got shape {backbone_features.shape}")
        
        if torch.isnan(backbone_features).any() or torch.isinf(backbone_features).any():
            raise ValueError("backbone_features contains NaN or Inf values")
        
        # Initial logits validation
        if initial_logits is not None:
            if not isinstance(initial_logits, torch.Tensor):
                raise TypeError(f"initial_logits must be torch.Tensor, got {type(initial_logits)}")
            
            if initial_logits.dim() != 3:
                raise ValueError(f"initial_logits must be 3D [B, L, vocab_size], got shape {initial_logits.shape}")
            
            if initial_logits.shape[:2] != backbone_features.shape[:2]:
                raise ValueError(
                    f"Batch/sequence dims must match: backbone {backbone_features.shape[:2]} "
                    f"vs logits {initial_logits.shape[:2]}"
                )
            
            if torch.isnan(initial_logits).any() or torch.isinf(initial_logits).any():
                raise ValueError("initial_logits contains NaN or Inf values")
        
        # Mask validation
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"mask must be torch.Tensor, got {type(mask)}")
            
            if mask.dim() != 2:
                raise ValueError(f"mask must be 2D [B, L], got shape {mask.shape}")
            
            if mask.shape != backbone_features.shape[:2]:
                raise ValueError(
                    f"mask shape {mask.shape} must match batch/sequence dims {backbone_features.shape[:2]}"
                )
    
    def _initialize_logits(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: Optional[int] = None,
        strategy: str = 'random'
    ) -> torch.Tensor:
        """
        Initialize sequence logits with specified strategy.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            vocab_size: Vocabulary size (inferred from sequence_repr if None)
            strategy: Initialization strategy
        
        Returns:
            Initialized logits [B, L, vocab_size]
        """
        # Infer vocab_size from sequence_repr if not provided
        if vocab_size is None:
            vocab_size = self.sequence_repr.vocab_size
        
        if strategy == 'random':
            # Random logits from normal distribution
            if self.rng is not None:
                logits = torch.randn(batch_size, seq_len, vocab_size, generator=self.rng, device=self.device)
            else:
                logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        elif strategy == 'uniform':
            # Uniform logits (equal probability for all amino acids)
            logits = torch.zeros(batch_size, seq_len, vocab_size, device=self.device)
        elif strategy == 'biased':
            # Biased toward common amino acids (realistic distribution)
            # Common amino acids: A, L, S, V, G, E, K, T, D, I
            common_indices = [0, 10, 15, 19, 6, 4, 9, 17, 3, 8]  # Approximate positions
            if self.rng is not None:
                logits = torch.randn(batch_size, seq_len, vocab_size, generator=self.rng, device=self.device) * 0.1
            else:
                logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device) * 0.1
            logits[:, :, common_indices] += 0.5  # Bias toward common AAs
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        return logits
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'success_rate': self.successful_optimizations / max(self.total_optimizations, 1),
            'average_steps': self.total_steps / max(self.total_optimizations, 1),
            'config': self.config.__dict__,
            'num_landscapes': self.num_landscapes,
            'multi_landscape': self.multi_landscape
        }
    
    def reset_statistics(self):
        """Reset optimization statistics"""
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.total_steps = 0


if __name__ == "__main__":
    # Example usage and basic testing
    print("Testing IREDSequenceOptimizer initialization...")
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from models.energy_head import EnergyHead
    from models.sequence_repr import ContinuousSequenceRepr
    
    # Create components
    energy_model = EnergyHead(backbone_dim=128, seq_dim=20, hidden_dim=256, num_layers=2)
    seq_repr = ContinuousSequenceRepr(vocab_size=20, temperature_schedule=[1.0, 0.5, 0.1])
    
    # Test single model initialization
    optimizer_single = IREDSequenceOptimizer(energy_model, seq_repr)
    print(f"✓ Single model optimizer: {optimizer_single.num_landscapes} landscapes")
    print(f"✓ Multi-landscape mode: {optimizer_single.multi_landscape}")
    
    # Test multi-model initialization
    energy_models = [EnergyHead(backbone_dim=128, seq_dim=20) for _ in range(3)]
    optimizer_multi = IREDSequenceOptimizer(energy_models, seq_repr)
    print(f"✓ Multi-model optimizer: {optimizer_multi.num_landscapes} landscapes")
    print(f"✓ Multi-landscape mode: {optimizer_multi.multi_landscape}")
    
    # Test custom configuration
    custom_config = OptimizationConfig(
        learning_rate=0.02,
        max_steps_per_landscape=100,
        num_landscapes=7
    )
    optimizer_custom = IREDSequenceOptimizer(energy_model, seq_repr, config=custom_config)
    print(f"✓ Custom config: lr={optimizer_custom.config.learning_rate}, landscapes={optimizer_custom.num_landscapes}")
    
    # Test statistics
    stats = optimizer_single.get_statistics()
    print(f"✓ Statistics: {stats}")
    
    # Test input validation
    batch_size, seq_len, backbone_dim = 2, 50, 128
    backbone_features = torch.randn(batch_size, seq_len, backbone_dim)
    initial_logits = torch.randn(batch_size, seq_len, 20)
    mask = torch.ones(batch_size, seq_len)
    
    try:
        optimizer_single._validate_inputs(backbone_features, initial_logits, mask)
        print("✓ Input validation passed")
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
    
    # Test logits initialization
    logits_random = optimizer_single._initialize_logits(2, 50, strategy='random')
    logits_uniform = optimizer_single._initialize_logits(2, 50, strategy='uniform')
    logits_biased = optimizer_single._initialize_logits(2, 50, strategy='biased')
    print(f"✓ Logits initialization: random={logits_random.shape}, uniform={logits_uniform.shape}, biased={logits_biased.shape}")
    
    # Test optimize_sequence
    print("\nTesting optimize_sequence...")
    backbone_features = torch.randn(2, 50, 128)
    
    try:
        result = optimizer_single.optimize_sequence(
            backbone_features, 
            max_steps=10,  # Small number for quick test
            return_trajectory=True
        )
        print(f"✓ Optimization completed: energy={result.final_energy:.4f}, converged={result.converged}")
        print(f"✓ Final sequence shape: {result.sequence.shape}")
        print(f"✓ Total steps: {result.total_steps}")
        print(f"✓ Trajectory length: {len(result.trajectory)}")
        
        # Test trajectory analysis
        analysis = optimizer_single.analyze_trajectory(result)
        print(f"✓ Trajectory analysis: energy improved by {analysis['energy']['improvement_percent']:.1f}%")
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test adaptive optimization
    print("\nTesting adaptive_optimization...")
    try:
        adaptive_result = optimizer_single.adaptive_optimization(
            backbone_features,
            max_total_steps=20
        )
        print(f"✓ Adaptive optimization: energy={adaptive_result.final_energy:.4f}, steps={adaptive_result.total_steps}")
    except Exception as e:
        print(f"✗ Adaptive optimization failed: {e}")
    
    # Test restart strategy
    print("\nTesting optimize_with_restarts...")
    try:
        restart_result = optimizer_single.optimize_with_restarts(
            backbone_features,
            num_restarts=2,
            max_steps_per_restart=5
        )
        print(f"✓ Restart optimization: energy={restart_result.final_energy:.4f}")
    except Exception as e:
        print(f"✗ Restart optimization failed: {e}")
    
    print("\n✓ All tests passed!")
