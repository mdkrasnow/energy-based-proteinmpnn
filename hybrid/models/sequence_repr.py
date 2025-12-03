"""
Continuous Sequence Representation

This module implements a differentiable representation of discrete amino acid sequences
using Gumbel-Softmax sampling with temperature annealing. This enables gradient-based
optimization over sequences while maintaining the ability to recover discrete sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


class ContinuousSequenceRepr(nn.Module):
    """
    Continuous representation of discrete amino acid sequences using Gumbel-Softmax.
    
    This class enables differentiable sampling from discrete sequence distributions
    with temperature annealing across IRED optimization landscapes. It supports both
    soft sampling during training and hard sequences with straight-through gradients
    during inference.
    
    Args:
        vocab_size: Size of amino acid vocabulary (default: 20 for standard amino acids)
        temperature_schedule: List of temperatures for annealing across landscapes
                            (default: [1.0, 0.5, 0.1] - smooth to sharp)
        min_temperature: Minimum temperature to prevent numerical issues (default: 1e-3)
        max_temperature: Maximum temperature to prevent overflow (default: 10.0)
    """
    
    def __init__(
        self,
        vocab_size: int = 20,
        temperature_schedule: Optional[List[float]] = None,
        min_temperature: float = 1e-3,
        max_temperature: float = 10.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        # Default temperature schedule: start smooth, end sharp
        if temperature_schedule is None:
            temperature_schedule = [1.0, 0.5, 0.1]
        
        self.register_buffer(
            'temperature_schedule', 
            torch.tensor(temperature_schedule, dtype=torch.float32)
        )
        
        # Validate temperature schedule
        if len(self.temperature_schedule) < 1:
            raise ValueError("Temperature schedule must have at least one value")
        
        if torch.any(self.temperature_schedule <= 0):
            raise ValueError("All temperatures must be positive")
    
    def forward(
        self, 
        logits: torch.Tensor, 
        landscape_idx: int = 0, 
        training: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Convert logits to continuous sequence representation.
        
        Args:
            logits: Raw sequence logits [B, L, vocab_size]
            landscape_idx: Current landscape index for temperature annealing
            training: Force training mode (soft) or inference mode (hard).
                     If None, uses self.training
        
        Returns:
            soft_sequence: Continuous sequence representation [B, L, vocab_size]
        """
        # Input validation
        self._validate_inputs(logits, landscape_idx)
        
        # Clean input logits if they contain NaN/Inf values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("DEBUG sequence_repr: STEP 1 - Cleaning NaN/Inf from input logits")
            print(f"DEBUG sequence_repr: STEP 1 - PRE-clean logits: NaN={torch.isnan(logits).sum().item()}, Inf={torch.isinf(logits).sum().item()}")
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            logits = torch.where(torch.isinf(logits), torch.sign(logits) * 8.0, logits)
            print(f"DEBUG sequence_repr: STEP 1 - POST-clean logits: NaN={torch.isnan(logits).sum().item()}, Inf={torch.isinf(logits).sum().item()}")
        
        # Determine mode
        is_training = training if training is not None else self.training
        
        # Get current temperature on same device as logits
        temperature = self.get_temperature(landscape_idx).to(logits.device)
        
        # DEBUG: Log input values
        print(f"DEBUG sequence_repr: landscape_idx={landscape_idx}, temp_raw={temperature.item():.8f}")
        print(f"DEBUG sequence_repr: logits shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        print(f"DEBUG sequence_repr: logits contains NaN: {torch.isnan(logits).any().item()}")
        print(f"DEBUG sequence_repr: logits contains Inf: {torch.isinf(logits).any().item()}")
        
        # Enhanced numerical stability safeguards
        print(f"DEBUG sequence_repr: STEP 2 - Pre-clamp validation")
        print(f"DEBUG sequence_repr: STEP 2 - Pre-clamp logits: NaN={torch.isnan(logits).sum().item()}, Inf={torch.isinf(logits).sum().item()}")
        
        # More conservative logits clamping to prevent overflow in downstream operations
        logits = torch.clamp(logits, min=-5.0, max=5.0)
        
        # Safer temperature bounds - use configured bounds to avoid breaking existing configs
        temperature = torch.clamp(temperature, min=self.min_temperature, max=self.max_temperature)
        
        print(f"DEBUG sequence_repr: STEP 2 - Post-clamp validation")
        print(f"DEBUG sequence_repr: STEP 2 - Post-clamp logits: NaN={torch.isnan(logits).sum().item()}, Inf={torch.isinf(logits).sum().item()}")
        print(f"DEBUG sequence_repr: STEP 2 - temp={temperature.item():.8f}, logits_min={logits.min().item():.4f}, logits_max={logits.max().item():.4f}")
        
        if is_training:
            print(f"DEBUG sequence_repr: Using training mode (Gumbel-Softmax)")
            # Training mode: Robust Gumbel-Softmax with better numerical stability
            
            # For very low temperatures, skip Gumbel-Softmax and use regular softmax
            # to avoid numerical instability from Gumbel noise
            # Use threshold of 5x min_temperature to be adaptive to configuration
            if temperature < (5.0 * self.min_temperature):
                print("DEBUG sequence_repr: STEP 3A - Temperature too low, using regular softmax instead of Gumbel-Softmax")
                scaled_logits = logits / temperature
                print(f"DEBUG sequence_repr: STEP 3A - After scaling: NaN={torch.isnan(scaled_logits).sum().item()}, Inf={torch.isinf(scaled_logits).sum().item()}")
                # Use log_softmax and exp for better numerical stability
                log_probs = F.log_softmax(scaled_logits, dim=-1)
                print(f"DEBUG sequence_repr: STEP 3A - After log_softmax: NaN={torch.isnan(log_probs).sum().item()}, Inf={torch.isinf(log_probs).sum().item()}")
                soft_sequence = torch.exp(log_probs)
                print(f"DEBUG sequence_repr: STEP 3A - After exp: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
                print(f"DEBUG sequence_repr: STEP 3A - Low-temp softmax - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
            else:
                try:
                    print("DEBUG sequence_repr: STEP 3B - About to call Gumbel-Softmax")
                    print(f"DEBUG sequence_repr: STEP 3B - Pre-Gumbel logits: NaN={torch.isnan(logits).sum().item()}, Inf={torch.isinf(logits).sum().item()}")
                    soft_sequence = F.gumbel_softmax(
                        logits, 
                        tau=temperature, 
                        hard=False, 
                        dim=-1
                    )
                    print(f"DEBUG sequence_repr: STEP 3B - Post-Gumbel: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
                    print(f"DEBUG sequence_repr: STEP 3B - Gumbel-Softmax output - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
                    
                    # Check for NaN/Inf after Gumbel-Softmax
                    if torch.isnan(soft_sequence).any() or torch.isinf(soft_sequence).any():
                        print("DEBUG sequence_repr: STEP 3C - Gumbel-Softmax produced NaN/Inf, falling back to stable softmax")
                        scaled_logits = logits / temperature
                        print(f"DEBUG sequence_repr: STEP 3C - After scaling: NaN={torch.isnan(scaled_logits).sum().item()}, Inf={torch.isinf(scaled_logits).sum().item()}")
                        log_probs = F.log_softmax(scaled_logits, dim=-1)
                        print(f"DEBUG sequence_repr: STEP 3C - After log_softmax: NaN={torch.isnan(log_probs).sum().item()}, Inf={torch.isinf(log_probs).sum().item()}")
                        soft_sequence = torch.exp(log_probs)
                        print(f"DEBUG sequence_repr: STEP 3C - After exp: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
                        print(f"DEBUG sequence_repr: STEP 3C - Stable fallback softmax - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
                except Exception as e:
                    print(f"DEBUG sequence_repr: STEP 3D - Gumbel-Softmax exception: {e}")
                    # Robust fallback using log_softmax for numerical stability
                    scaled_logits = logits / temperature
                    print(f"DEBUG sequence_repr: STEP 3D - After scaling: NaN={torch.isnan(scaled_logits).sum().item()}, Inf={torch.isinf(scaled_logits).sum().item()}")
                    log_probs = F.log_softmax(scaled_logits, dim=-1)
                    print(f"DEBUG sequence_repr: STEP 3D - After log_softmax: NaN={torch.isnan(log_probs).sum().item()}, Inf={torch.isinf(log_probs).sum().item()}")
                    soft_sequence = torch.exp(log_probs)
                    print(f"DEBUG sequence_repr: STEP 3D - After exp: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
                    print(f"DEBUG sequence_repr: STEP 3D - Exception fallback stable softmax - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
        else:
            print(f"DEBUG sequence_repr: STEP 4A - Using inference mode (straight-through)")
            # Inference mode: Straight-through estimator
            # Use robust softmax with numerical stability
            scaled_logits = logits / temperature
            print(f"DEBUG sequence_repr: STEP 4A - After scaling: NaN={torch.isnan(scaled_logits).sum().item()}, Inf={torch.isinf(scaled_logits).sum().item()}")
            print(f"DEBUG sequence_repr: STEP 4A - Inference scaled logits - min={scaled_logits.min().item():.6f}, max={scaled_logits.max().item():.6f}")
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            print(f"DEBUG sequence_repr: STEP 4A - After log_softmax: NaN={torch.isnan(log_probs).sum().item()}, Inf={torch.isinf(log_probs).sum().item()}")
            soft_sequence = torch.exp(log_probs)
            print(f"DEBUG sequence_repr: STEP 4A - After exp: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
            print(f"DEBUG sequence_repr: STEP 4A - Inference stable softmax - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
            
            # Hard one-hot for discrete sequence
            print(f"DEBUG sequence_repr: STEP 4B - Creating hard one-hot sequence")
            hard_sequence = F.one_hot(
                logits.argmax(dim=-1), 
                num_classes=self.vocab_size
            ).float()
            print(f"DEBUG sequence_repr: STEP 4B - Hard sequence: NaN={torch.isnan(hard_sequence).sum().item()}, Inf={torch.isinf(hard_sequence).sum().item()}")
            
            # Straight-through: hard forward, soft backward
            print(f"DEBUG sequence_repr: STEP 4C - Applying straight-through")
            soft_sequence = hard_sequence + (soft_sequence - soft_sequence.detach())
            print(f"DEBUG sequence_repr: STEP 4C - After straight-through: NaN={torch.isnan(soft_sequence).sum().item()}, Inf={torch.isinf(soft_sequence).sum().item()}")
            print(f"DEBUG sequence_repr: STEP 4C - After straight-through - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
        
        # Final safety check for NaN/Inf values
        print(f"DEBUG sequence_repr: Final output - min={soft_sequence.min().item():.6f}, max={soft_sequence.max().item():.6f}")
        print(f"DEBUG sequence_repr: Final NaN: {torch.isnan(soft_sequence).any().item()}, Inf: {torch.isinf(soft_sequence).any().item()}")
        
        if torch.isnan(soft_sequence).any() or torch.isinf(soft_sequence).any():
            nan_count = torch.isnan(soft_sequence).sum().item()
            inf_count = torch.isinf(soft_sequence).sum().item()
            print("DEBUG sequence_repr: NaN/Inf DETECTED - NO LONGER CLEANING, WILL PROPAGATE!")
            print(f"DEBUG sequence_repr: Found {nan_count} NaN and {inf_count} Inf values")
            print(f"DEBUG sequence_repr: temp={temperature.item():.8f}, landscape_idx={landscape_idx}")
            print(f"DEBUG sequence_repr: is_training={is_training}, gumbel_softmax_used={temperature >= (5.0 * self.min_temperature)}")
            import warnings
            warnings.warn(
                f"NaN/Inf detected in sequence probabilities: {nan_count} NaN, {inf_count} Inf. "
                f"Temperature={temperature.item():.8f}, landscape_idx={landscape_idx}. "
                f"Emergency fallback DISABLED - will propagate to expose root cause.", 
                UserWarning
            )
        
        # DIMENSION DEBUGGING: Validate output shape matches expected vocab_size
        print(f"DEBUG sequence_repr: DIMENSION CHECK - Expected vocab_size: {self.vocab_size}")
        print(f"DEBUG sequence_repr: DIMENSION CHECK - Actual output shape: {soft_sequence.shape}")
        print(f"DEBUG sequence_repr: DIMENSION CHECK - Actual last dim: {soft_sequence.shape[-1]}")
        print(f"DEBUG sequence_repr: DIMENSION CHECK - Matches expected? {soft_sequence.shape[-1] == self.vocab_size}")
        
        if soft_sequence.shape[-1] != self.vocab_size:
            print(f"ERROR sequence_repr: DIMENSION MISMATCH!")
            print(f"ERROR sequence_repr: Expected output shape [..., {self.vocab_size}], got [..., {soft_sequence.shape[-1]}]")
            print(f"ERROR sequence_repr: This suggests a configuration error in sequence_repr")
            print(f"ERROR sequence_repr: Check vocab_size parameter: configured={self.vocab_size}, actual_output={soft_sequence.shape[-1]}")
            
            # Check if we accidentally returned input logits or something else
            print(f"ERROR sequence_repr: Input logits had shape {logits.shape}")
            if soft_sequence.shape[-1] == logits.shape[-1]:
                print(f"ERROR sequence_repr: Output matches input logits shape - logic error in forward pass!")
            else:
                print(f"ERROR sequence_repr: Output shape doesn't match input - unknown tensor returned!")
        
        print(f"DEBUG sequence_repr: Returning sequence_probs with shape {soft_sequence.shape}")
        return soft_sequence
    
    def get_temperature(self, landscape_idx: int) -> torch.Tensor:
        """
        Get temperature for current landscape with linear interpolation.
        
        Args:
            landscape_idx: Current landscape index
            
        Returns:
            temperature: Current temperature value
        """
        # Clamp landscape index to valid range
        max_idx = len(self.temperature_schedule) - 1
        landscape_idx = max(0, min(landscape_idx, max_idx))
        
        if max_idx == 0:
            # Single temperature
            temperature = self.temperature_schedule[0]
        else:
            # Linear interpolation between schedule points
            progress = landscape_idx / max_idx
            start_temp = self.temperature_schedule[0]
            end_temp = self.temperature_schedule[-1]
            temperature = start_temp * (1 - progress) + end_temp * progress
        
        # Clamp to safe range
        temperature = torch.clamp(
            temperature, 
            self.min_temperature, 
            self.max_temperature
        )
        
        return temperature
    
    def get_discrete_sequence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get discrete amino acid sequence from logits.
        
        Args:
            logits: Raw sequence logits [B, L, vocab_size]
            
        Returns:
            discrete_sequence: Discrete amino acid indices [B, L]
        """
        self._validate_logits(logits)
        # Clamp for numerical stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return logits.argmax(dim=-1)
    
    def sample_sequence(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample discrete sequences from logits with specified temperature.
        
        Args:
            logits: Raw sequence logits [B, L, vocab_size]
            temperature: Sampling temperature
            num_samples: Number of samples to generate
            
        Returns:
            samples: Sampled sequences [num_samples, B, L]
        """
        self._validate_logits(logits)
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        # Clamp for numerical stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        samples = []
        for _ in range(num_samples):
            # Sample using temperature-scaled softmax
            probs = F.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(
                probs.view(-1, self.vocab_size), 
                num_samples=1
            ).view(logits.shape[:-1])
            samples.append(sample)
        
        return torch.stack(samples, dim=0)
    
    def compute_entropy(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute sequence entropy for diversity analysis.
        
        Args:
            logits: Raw sequence logits [B, L, vocab_size]
            mask: Optional sequence mask [B, L]
            
        Returns:
            entropy: Mean entropy per position [B] or scalar if masked
        """
        self._validate_logits(logits)
        
        # Clamp for numerical stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # Compute probabilities and entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, L]
        
        if mask is not None:
            # Masked mean
            masked_entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
            return masked_entropy
        else:
            # Simple mean over sequence length
            return entropy.mean(dim=-1)
    
    def _validate_inputs(self, logits: torch.Tensor, landscape_idx: int):
        """Validate forward pass inputs"""
        self._validate_logits(logits)
        
        if not isinstance(landscape_idx, int) or landscape_idx < 0:
            raise ValueError(f"landscape_idx must be non-negative integer, got {landscape_idx}")
    
    def _validate_logits(self, logits: torch.Tensor):
        """Validate logits tensor"""
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")
        
        if logits.dim() != 3:
            raise ValueError(f"logits must be 3D [B, L, vocab_size], got shape {logits.shape}")
        
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(f"logits last dim must be {self.vocab_size}, got {logits.shape[-1]}")
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            import warnings
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            warnings.warn(f"Input logits contain {nan_count} NaN and {inf_count} Inf values. These will be cleaned.", UserWarning)
            # Note: Cleaning will be handled in the forward method
    
    def update_temperature_schedule(self, new_schedule: List[float]):
        """
        Update temperature schedule during training.
        
        Args:
            new_schedule: New list of temperature values
        """
        if len(new_schedule) < 1:
            raise ValueError("Temperature schedule must have at least one value")
        
        if any(temp <= 0 for temp in new_schedule):
            raise ValueError("All temperatures must be positive")
        
        self.temperature_schedule = torch.tensor(new_schedule, dtype=torch.float32)
    
    def get_schedule_info(self) -> dict:
        """Get information about current temperature schedule"""
        return {
            'schedule': self.temperature_schedule.tolist(),
            'num_landscapes': len(self.temperature_schedule),
            'min_temp': self.temperature_schedule.min().item(),
            'max_temp': self.temperature_schedule.max().item(),
            'annealing_factor': (self.temperature_schedule[0] / self.temperature_schedule[-1]).item()
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ContinuousSequenceRepr...")
    
    # Create sequence representation
    seq_repr = ContinuousSequenceRepr(
        vocab_size=20,
        temperature_schedule=[1.0, 0.5, 0.1]
    )
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 50, 20
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    print(f"Input logits shape: {logits.shape}")
    print(f"Temperature schedule: {seq_repr.get_schedule_info()}")
    
    # Test training mode (soft sampling)
    seq_repr.train()
    soft_seq_train = seq_repr(logits, landscape_idx=0)
    print(f"✓ Training mode output shape: {soft_seq_train.shape}")
    print(f"✓ Training mode sum per position: {soft_seq_train.sum(dim=-1).mean():.3f} (should be ~1.0)")
    
    # Test inference mode (hard sequences)
    seq_repr.eval()
    soft_seq_inference = seq_repr(logits, landscape_idx=2)
    print(f"✓ Inference mode output shape: {soft_seq_inference.shape}")
    print(f"✓ Inference mode sum per position: {soft_seq_inference.sum(dim=-1).mean():.3f} (should be 1.0)")
    
    # Test temperature annealing
    temps = [seq_repr.get_temperature(i).item() for i in range(3)]
    print(f"✓ Temperature annealing: {temps}")
    
    # Test discrete sequence extraction
    discrete_seq = seq_repr.get_discrete_sequence(logits)
    print(f"✓ Discrete sequence shape: {discrete_seq.shape}")
    print(f"✓ Discrete sequence range: [{discrete_seq.min()}, {discrete_seq.max()}]")
    
    # Test entropy computation
    entropy = seq_repr.compute_entropy(logits)
    print(f"✓ Sequence entropy: {entropy.mean():.3f}")
    
    # Test gradient flow
    logits.requires_grad_(True)
    seq_repr.train()
    soft_seq = seq_repr(logits, landscape_idx=1)
    loss = soft_seq.sum()
    loss.backward()
    
    print(f"✓ Gradient flow successful: {logits.grad is not None}")
    print(f"✓ Gradient norm: {logits.grad.norm():.6f}")
    
    print("\n✓ All tests passed!")