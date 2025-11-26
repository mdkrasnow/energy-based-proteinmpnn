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
        
        # Determine mode
        is_training = training if training is not None else self.training
        
        # Get current temperature on same device as logits
        temperature = self.get_temperature(landscape_idx).to(logits.device)
        
        # Numerical stability: clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        if is_training:
            # Training mode: Gumbel-Softmax for differentiable sampling
            soft_sequence = F.gumbel_softmax(
                logits, 
                tau=temperature, 
                hard=False, 
                dim=-1
            )
        else:
            # Inference mode: Straight-through estimator
            # Soft probabilities for gradients
            soft_sequence = F.softmax(logits / temperature, dim=-1)
            
            # Hard one-hot for discrete sequence
            hard_sequence = F.one_hot(
                logits.argmax(dim=-1), 
                num_classes=self.vocab_size
            ).float()
            
            # Straight-through: hard forward, soft backward
            soft_sequence = hard_sequence + (soft_sequence - soft_sequence.detach())
        
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
            raise ValueError("logits contains NaN or Inf values")
    
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