"""
Loss Functions for Energy-Based Protein Design Training

This module implements loss functions for training protein stability energy models through
contrastive learning. The main ContrastiveLoss class combines multiple loss components:
- Margin-based ranking loss for energy ordering
- Temperature-scaled contrastive loss for smooth learning
- Regularization terms for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum


class NegativeType(Enum):
    """Enumeration of negative sequence types for loss weighting"""
    RANDOM = "random"
    MUTATED = "mutated"
    FAILED_DESIGN = "failed_design"
    HARD_NEGATIVE = "hard_negative"


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training protein stability energy models.
    
    This loss combines multiple components to train an energy model that assigns
    lower energy to stable (positive) sequences and higher energy to unstable 
    (negative) sequences:
    
    1. Margin-based ranking loss: Enforces E_pos + margin < E_neg
    2. Temperature-scaled contrastive loss: Softmax-style learning with temperature
    3. Entropy regularization: Encourages exploration during optimization
    4. Smoothness regularization: Promotes smooth energy landscapes
    5. Weighted loss components: Different weights for different negative types
    
    Args:
        margin: Margin for ranking loss (default: 1.0)
        temperature: Temperature for contrastive loss (default: 0.1)
        ranking_weight: Weight for margin-based ranking loss (default: 1.0)
        contrastive_weight: Weight for temperature-scaled contrastive loss (default: 1.0)
        entropy_weight: Weight for entropy regularization (default: 0.01)
        smoothness_weight: Weight for smoothness regularization (default: 0.001)
        negative_weights: Weights for different negative types (default: uniform)
        reduction: Loss reduction method ('mean', 'sum', 'none') (default: 'mean')
        eps: Small value for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
        ranking_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        entropy_weight: float = 0.01,
        smoothness_weight: float = 0.001,
        negative_weights: Optional[Dict[str, float]] = None,
        reduction: str = 'mean',
        eps: float = 1e-8,
        min_temperature: float = 1e-3,
        max_temperature: float = 10.0
    ):
        super().__init__()
        
        # Store hyperparameters with numerical stability bounds
        self.margin = margin
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        # Clamp temperature and warn if changed
        clamped_temperature = max(min_temperature, min(temperature, max_temperature))
        if clamped_temperature != temperature:
            warnings.warn(f"Temperature {temperature} clamped to {clamped_temperature} for numerical stability")
        self.temperature = clamped_temperature
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        self.entropy_weight = entropy_weight
        self.smoothness_weight = smoothness_weight
        self.reduction = reduction
        self.eps = eps
        
        # Validate inputs (check original temperature before clamping)
        self._validate_config(temperature)
        
        # Default negative weights (uniform if not specified)
        if negative_weights is None:
            negative_weights = {
                NegativeType.RANDOM.value: 1.0,
                NegativeType.MUTATED.value: 1.0,
                NegativeType.FAILED_DESIGN.value: 1.0,
                NegativeType.HARD_NEGATIVE.value: 2.0  # Higher weight for hard negatives
            }
        self.negative_weights = negative_weights
        
        # Loss components for individual tracking
        self.last_losses = {}
    
    def forward(
        self,
        pos_energies: torch.Tensor,
        neg_energies: torch.Tensor,
        pos_sequence_probs: Optional[torch.Tensor] = None,
        neg_sequence_probs: Optional[torch.Tensor] = None,
        negative_types: Optional[List[str]] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss for energy predictions.
        
        Args:
            pos_energies: Energy predictions for positive (stable) sequences [B_pos]
            neg_energies: Energy predictions for negative (unstable) sequences [B_neg]
            pos_sequence_probs: Sequence probabilities for positives [B_pos, L, vocab_size]
            neg_sequence_probs: Sequence probabilities for negatives [B_neg, L, vocab_size]
            negative_types: Type of each negative sample for weighted loss
            pos_mask: Mask for positive sequences [B_pos, L]
            neg_mask: Mask for negative sequences [B_neg, L]
        
        Returns:
            loss: Combined contrastive loss [scalar if reduction != 'none']
        """
        # Input validation
        self._validate_inputs(pos_energies, neg_energies, pos_sequence_probs, 
                            neg_sequence_probs, negative_types)
        
        # Compute core loss components
        ranking_loss = self._compute_ranking_loss(pos_energies, neg_energies, negative_types)
        contrastive_loss = self._compute_contrastive_loss(pos_energies, neg_energies, negative_types)
        
        # Compute regularization terms if sequence probabilities provided
        entropy_loss = torch.tensor(0.0, device=pos_energies.device)
        smoothness_loss = torch.tensor(0.0, device=pos_energies.device)
        
        if pos_sequence_probs is not None:
            entropy_loss = self._compute_entropy_regularization(pos_sequence_probs, pos_mask)
        
        if pos_sequence_probs is not None and neg_sequence_probs is not None:
            smoothness_loss = self._compute_smoothness_regularization(
                pos_energies, neg_energies, pos_sequence_probs, neg_sequence_probs
            )
        
        # Combine loss components
        total_loss = (
            self.ranking_weight * ranking_loss +
            self.contrastive_weight * contrastive_loss +
            self.entropy_weight * entropy_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        # Store individual losses for monitoring
        self.last_losses = {
            'ranking_loss': ranking_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss
    
    def _compute_ranking_loss(
        self, 
        pos_energies: torch.Tensor, 
        neg_energies: torch.Tensor,
        negative_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss: max(0, E_pos - E_neg + margin)
        
        Enforces that positive energies are lower than negative energies by at least margin.
        """
        # Create all pairwise combinations
        pos_expanded = pos_energies.unsqueeze(1)  # [B_pos, 1]
        neg_expanded = neg_energies.unsqueeze(0)  # [1, B_neg]
        
        # Compute margin loss: pos - neg + margin
        margin_losses = torch.clamp(pos_expanded - neg_expanded + self.margin, min=0.0)
        
        # Apply negative type weights if provided
        if negative_types is not None:
            weights = torch.ones_like(neg_energies)
            for i, neg_type in enumerate(negative_types):
                mapped_type = self._map_generation_method_to_negative_type(neg_type)
                if mapped_type in self.negative_weights:
                    weights[i] = self.negative_weights[mapped_type]
            
            # Broadcast weights to match margin_losses shape
            weights = weights.unsqueeze(0)  # [1, B_neg]
            margin_losses = margin_losses * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return margin_losses.mean()
        elif self.reduction == 'sum':
            return margin_losses.sum()
        else:
            return margin_losses
    
    def _compute_contrastive_loss(
        self, 
        pos_energies: torch.Tensor, 
        neg_energies: torch.Tensor,
        negative_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute temperature-scaled contrastive loss.
        
        Uses softmax with temperature to create smooth ranking objectives.
        """
        # Combine all energies
        all_energies = torch.cat([pos_energies, neg_energies], dim=0)
        
        # Create labels: 0 for stable (positive), 1 for unstable (negative)
        pos_labels = torch.zeros(len(pos_energies), device=pos_energies.device)
        neg_labels = torch.ones(len(neg_energies), device=neg_energies.device)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Clamp energies for numerical stability before temperature scaling
        clamped_energies = torch.clamp(all_energies, min=-10.0, max=10.0)
        
        # Apply temperature scaling (temperature already validated in __init__)
        scaled_energies = clamped_energies / self.temperature
        
        # Logits for binary classification: [P(stable), P(unstable)]
        # Low energy = stable, so use -energy for stable class logit
        stable_logits = -scaled_energies
        unstable_logits = scaled_energies
        logits = torch.stack([stable_logits, unstable_logits], dim=1)
        
        # Apply negative type weights if provided
        weights = torch.ones_like(all_labels)
        if negative_types is not None:
            for i, neg_type in enumerate(negative_types):
                neg_idx = len(pos_energies) + i
                mapped_type = self._map_generation_method_to_negative_type(neg_type)
                if mapped_type in self.negative_weights:
                    weights[neg_idx] = self.negative_weights[mapped_type]
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(logits, all_labels.long(), weight=None, reduction='none')
        weighted_loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
    
    def _compute_entropy_regularization(
        self, 
        sequence_probs: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute entropy regularization to encourage exploration.
        
        Higher entropy = more uniform sequence distributions = better exploration.
        """
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(sequence_probs + self.eps)
        entropy = -torch.sum(sequence_probs * log_probs, dim=-1)  # [B, L]
        
        # Apply mask if provided
        if mask is not None:
            entropy = entropy * mask
            # Compute masked mean
            total_positions = mask.sum()
            if total_positions > 0:
                entropy = entropy.sum() / total_positions
            else:
                entropy = entropy.mean()  # Fallback
        else:
            entropy = entropy.mean()
        
        # We want to maximize entropy, so minimize negative entropy
        return -entropy
    
    def _compute_smoothness_regularization(
        self,
        pos_energies: torch.Tensor,
        neg_energies: torch.Tensor,
        pos_sequence_probs: torch.Tensor,
        neg_sequence_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness regularization to promote smooth energy landscapes.
        
        Penalizes large energy differences between similar sequences.
        """
        # Sample pairs of sequences to compare
        num_pairs = min(len(pos_energies), len(neg_energies), 32)  # Limit for efficiency
        
        if num_pairs < 2:
            return torch.tensor(0.0, device=pos_energies.device)
        
        # Randomly sample pairs
        pos_indices = torch.randperm(len(pos_energies))[:num_pairs]
        neg_indices = torch.randperm(len(neg_energies))[:num_pairs]
        
        pos_sample_energies = pos_energies[pos_indices]
        neg_sample_energies = neg_energies[neg_indices]
        pos_sample_probs = pos_sequence_probs[pos_indices]
        neg_sample_probs = neg_sequence_probs[neg_indices]
        
        # Compute sequence similarity (cosine similarity of probability vectors)
        # Flatten sequence probabilities
        pos_flat = pos_sample_probs.view(num_pairs, -1)
        neg_flat = neg_sample_probs.view(num_pairs, -1)
        
        # Compute pairwise cosine similarities
        pos_norm = F.normalize(pos_flat, p=2, dim=1)
        neg_norm = F.normalize(neg_flat, p=2, dim=1)
        
        # Compute cross-similarities between positive and negative samples
        similarities = torch.mm(pos_norm, neg_norm.t())  # [num_pairs, num_pairs]
        
        # Compute energy differences
        energy_diffs = torch.abs(
            pos_sample_energies.unsqueeze(1) - neg_sample_energies.unsqueeze(0)
        )  # [num_pairs, num_pairs]
        
        # Smoothness loss: penalize large energy differences for similar sequences
        # Use exponential weighting: higher similarity = higher penalty for energy difference
        similarity_weights = torch.exp(similarities - 1.0)  # Peaked at similarity=1
        smoothness_loss = (similarity_weights * energy_diffs).mean()
        
        return smoothness_loss
    
    def _validate_config(self, original_temperature: float = None):
        """Validate loss configuration parameters"""
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        
        # Validate original temperature if provided, otherwise use current
        temp_to_check = original_temperature if original_temperature is not None else self.temperature
        if temp_to_check <= 0:
            raise ValueError(f"temperature must be positive, got {temp_to_check}")
        
        if self.min_temperature <= 0 or self.min_temperature >= self.max_temperature:
            raise ValueError(f"min_temperature must be positive and < max_temperature, got {self.min_temperature}")
        if self.max_temperature <= 0:
            raise ValueError(f"max_temperature must be positive, got {self.max_temperature}")
        if self.ranking_weight < 0:
            raise ValueError(f"ranking_weight must be non-negative, got {self.ranking_weight}")
        if self.contrastive_weight < 0:
            raise ValueError(f"contrastive_weight must be non-negative, got {self.contrastive_weight}")
        if self.entropy_weight < 0:
            raise ValueError(f"entropy_weight must be non-negative, got {self.entropy_weight}")
        if self.smoothness_weight < 0:
            raise ValueError(f"smoothness_weight must be non-negative, got {self.smoothness_weight}")
        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {self.reduction}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
    
    def _map_generation_method_to_negative_type(self, generation_method: str) -> str:
        """
        Map generation methods used in datasets to standard negative types.
        
        Args:
            generation_method: Generation method from dataset (e.g., 'random_problematic')
        
        Returns:
            Standard negative type compatible with NegativeType enum
        """
        if generation_method is None:
            return NegativeType.RANDOM.value  # Safe fallback for None inputs
        
        # Case-insensitive lookup
        normalized_method = generation_method.lower()
        
        method_mapping = {
            # Random-based generation methods
            'random': NegativeType.RANDOM.value,
            'random_problematic': NegativeType.RANDOM.value,
            
            # Mutation-based generation methods
            'mutated': NegativeType.MUTATED.value,
            'mutations': NegativeType.MUTATED.value,
            'structure_aware_mutations': NegativeType.MUTATED.value,
            
            # Failed design methods
            'failed_design': NegativeType.FAILED_DESIGN.value,
            'failed_designs': NegativeType.FAILED_DESIGN.value,
            
            # Hard negative methods
            'hard_negative': NegativeType.HARD_NEGATIVE.value,
            'hard_negatives': NegativeType.HARD_NEGATIVE.value,
        }
        
        return method_mapping.get(normalized_method, generation_method)

    def _validate_inputs(
        self,
        pos_energies: torch.Tensor,
        neg_energies: torch.Tensor,
        pos_sequence_probs: Optional[torch.Tensor],
        neg_sequence_probs: Optional[torch.Tensor],
        negative_types: Optional[List[str]]
    ):
        """Validate forward pass inputs"""
        # Type checking
        if not isinstance(pos_energies, torch.Tensor):
            raise TypeError(f"pos_energies must be torch.Tensor, got {type(pos_energies)}")
        if not isinstance(neg_energies, torch.Tensor):
            raise TypeError(f"neg_energies must be torch.Tensor, got {type(neg_energies)}")
        
        # Shape checking
        if pos_energies.dim() != 1:
            raise ValueError(f"pos_energies must be 1D, got shape {pos_energies.shape}")
        if neg_energies.dim() != 1:
            raise ValueError(f"neg_energies must be 1D, got shape {neg_energies.shape}")
        
        # Size checking
        if len(pos_energies) == 0:
            raise ValueError("pos_energies cannot be empty")
        if len(neg_energies) == 0:
            raise ValueError("neg_energies cannot be empty")
        
        # Device compatibility
        if pos_energies.device != neg_energies.device:
            raise ValueError(f"pos_energies and neg_energies must be on same device")
        
        # Value checking
        if torch.isnan(pos_energies).any() or torch.isinf(pos_energies).any():
            raise ValueError("pos_energies contains NaN or Inf values")
        if torch.isnan(neg_energies).any() or torch.isinf(neg_energies).any():
            raise ValueError("neg_energies contains NaN or Inf values")
        
        # Optional tensor validation
        if pos_sequence_probs is not None:
            if not isinstance(pos_sequence_probs, torch.Tensor):
                raise TypeError("pos_sequence_probs must be torch.Tensor")
            if pos_sequence_probs.dim() != 3:
                raise ValueError(f"pos_sequence_probs must be 3D, got shape {pos_sequence_probs.shape}")
            if pos_sequence_probs.shape[0] != len(pos_energies):
                raise ValueError("pos_sequence_probs batch size must match pos_energies")
        
        if neg_sequence_probs is not None:
            if not isinstance(neg_sequence_probs, torch.Tensor):
                raise TypeError("neg_sequence_probs must be torch.Tensor")
            if neg_sequence_probs.dim() != 3:
                raise ValueError(f"neg_sequence_probs must be 3D, got shape {neg_sequence_probs.shape}")
            if neg_sequence_probs.shape[0] != len(neg_energies):
                raise ValueError("neg_sequence_probs batch size must match neg_energies")
        
        # Negative types validation
        if negative_types is not None:
            if not isinstance(negative_types, list):
                raise TypeError("negative_types must be list")
            if len(negative_types) != len(neg_energies):
                raise ValueError("negative_types length must match neg_energies")
            
            # Check for valid negative types using mapping
            valid_types = set(neg_type.value for neg_type in NegativeType)
            for neg_type in negative_types:
                mapped_type = self._map_generation_method_to_negative_type(neg_type)
                if mapped_type not in valid_types and mapped_type not in self.negative_weights:
                    warnings.warn(f"Unknown negative type: {neg_type}")
    
    def get_last_losses(self) -> Dict[str, float]:
        """Get individual loss components from the last forward pass"""
        return self.last_losses.copy()
    
    def get_config(self) -> Dict[str, Union[float, str, Dict]]:
        """Get loss configuration"""
        return {
            'margin': self.margin,
            'temperature': self.temperature,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'ranking_weight': self.ranking_weight,
            'contrastive_weight': self.contrastive_weight,
            'entropy_weight': self.entropy_weight,
            'smoothness_weight': self.smoothness_weight,
            'negative_weights': self.negative_weights,
            'reduction': self.reduction,
            'eps': self.eps
        }


class EnergyRankingLoss(nn.Module):
    """
    Simple ranking loss for energy predictions.
    
    This is a lightweight alternative to ContrastiveLoss for scenarios where
    only basic ranking is needed without additional regularization terms.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self, 
        pos_energies: torch.Tensor, 
        neg_energies: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss: max(0, E_pos - E_neg + margin)
        
        Args:
            pos_energies: Energies for positive samples [B_pos]
            neg_energies: Energies for negative samples [B_neg]
        
        Returns:
            loss: Ranking loss
        """
        # Create all pairwise combinations
        pos_expanded = pos_energies.unsqueeze(1)  # [B_pos, 1]
        neg_expanded = neg_energies.unsqueeze(0)  # [1, B_neg]
        
        # Compute margin loss
        losses = torch.clamp(pos_expanded - neg_expanded + self.margin, min=0.0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


if __name__ == "__main__":
    # Test the loss functions
    print("Testing ContrastiveLoss...")
    
    # Create sample data
    batch_size = 8
    seq_len = 50
    vocab_size = 20
    
    # Simulate energies (positives should be lower than negatives)
    pos_energies = torch.randn(batch_size) - 1.0  # Lower energies
    neg_energies = torch.randn(batch_size) + 1.0  # Higher energies
    
    # Simulate sequence probabilities
    pos_probs = F.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    neg_probs = F.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    
    # Create masks for variable lengths
    pos_mask = torch.ones(batch_size, seq_len)
    neg_mask = torch.ones(batch_size, seq_len)
    
    # Simulate variable lengths
    for i in range(batch_size):
        length = torch.randint(20, seq_len, (1,)).item()
        pos_mask[i, length:] = 0
        neg_mask[i, length:] = 0
    
    # Create negative types
    negative_types = [
        NegativeType.RANDOM.value,
        NegativeType.MUTATED.value,
        NegativeType.FAILED_DESIGN.value,
        NegativeType.HARD_NEGATIVE.value,
        NegativeType.RANDOM.value,
        NegativeType.MUTATED.value,
        NegativeType.FAILED_DESIGN.value,
        NegativeType.HARD_NEGATIVE.value
    ]
    
    print(f"Input shapes:")
    print(f"  Positive energies: {pos_energies.shape}")
    print(f"  Negative energies: {neg_energies.shape}")
    print(f"  Positive probs: {pos_probs.shape}")
    print(f"  Negative probs: {neg_probs.shape}")
    print(f"  Energy ranges: pos [{pos_energies.min():.3f}, {pos_energies.max():.3f}], "
          f"neg [{neg_energies.min():.3f}, {neg_energies.max():.3f}]")
    
    # Test ContrastiveLoss
    loss_fn = ContrastiveLoss(
        margin=1.0,
        temperature=0.1,
        ranking_weight=1.0,
        contrastive_weight=1.0,
        entropy_weight=0.01,
        smoothness_weight=0.001
    )
    
    # Forward pass
    loss = loss_fn(
        pos_energies=pos_energies,
        neg_energies=neg_energies,
        pos_sequence_probs=pos_probs,
        neg_sequence_probs=neg_probs,
        negative_types=negative_types,
        pos_mask=pos_mask,
        neg_mask=neg_mask
    )
    
    print(f"✓ Total loss: {loss.item():.6f}")
    
    # Check individual loss components
    individual_losses = loss_fn.get_last_losses()
    print(f"✓ Loss components:")
    for component, value in individual_losses.items():
        print(f"    {component}: {value:.6f}")
    
    # Test gradient flow
    pos_energies.requires_grad_(True)
    neg_energies.requires_grad_(True)
    
    loss = loss_fn(pos_energies, neg_energies, pos_probs, neg_probs, negative_types)
    loss.backward()
    
    print(f"✓ Gradient flow:")
    print(f"    pos_energies grad norm: {pos_energies.grad.norm():.6f}")
    print(f"    neg_energies grad norm: {neg_energies.grad.norm():.6f}")
    
    # Test EnergyRankingLoss
    print(f"\nTesting EnergyRankingLoss...")
    ranking_loss_fn = EnergyRankingLoss(margin=1.0)
    ranking_loss = ranking_loss_fn(pos_energies.detach(), neg_energies.detach())
    print(f"✓ Ranking loss: {ranking_loss.item():.6f}")
    
    # Test configuration
    print(f"\nLoss configuration:")
    config = loss_fn.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n✓ All loss function tests passed!")