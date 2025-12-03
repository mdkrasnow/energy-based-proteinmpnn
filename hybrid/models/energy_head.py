"""
Energy Head Module

This module implements a neural network that predicts protein stability energy from
backbone structural features and sequence representations. The model fuses per-residue
features and uses global pooling to produce scalar energy predictions for optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class EnergyHead(nn.Module):
    """
    Energy prediction head for protein stability modeling.
    
    Takes backbone structural features and sequence probability distributions as input,
    fuses them through per-residue processing layers, and outputs scalar energy values
    representing protein stability.
    
    Architecture:
    1. Feature fusion: Concatenate backbone features + sequence probabilities
    2. Per-residue processing: Multiple layers with residual connections
    3. Masked global pooling: Handle variable sequence lengths
    4. Energy prediction: Final scalar output
    
    Args:
        backbone_dim: Dimension of backbone features from encoder (default: 128)
        seq_dim: Dimension of sequence representation (default: 20 for amino acids)
        hidden_dim: Hidden dimension for processing layers (default: 512)
        num_layers: Number of per-residue processing layers (default: 3)
        dropout: Dropout probability for regularization (default: 0.1)
        activation: Activation function ('relu', 'gelu', 'swish') (default: 'relu')
        use_batch_norm: Whether to use batch normalization (default: True)
        energy_scale: Scaling factor for energy output (default: 1.0)
    """
    
    def __init__(
        self,
        backbone_dim: int = 128,
        seq_dim: int = 20,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        energy_scale: float = 1.0
    ):
        super().__init__()
        
        # Store configuration
        self.backbone_dim = backbone_dim
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.energy_scale = energy_scale
        
        # Validate inputs
        self._validate_config()
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Feature fusion layer
        input_dim = backbone_dim + seq_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Per-residue processing layers with residual connections
        self.residue_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ResidualBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=self.activation,
                use_batch_norm=use_batch_norm
            )
            self.residue_layers.append(layer)
        
        # Global pooling projection
        self.pooling_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Energy prediction head
        self.energy_head = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(
        self, 
        backbone_features: torch.Tensor, 
        sequence_probs: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict energy from features.
        
        Args:
            backbone_features: Structural features from encoder [B, L, backbone_dim]
            sequence_probs: Sequence probability distribution [B, L, seq_dim]
            mask: Sequence mask for variable lengths [B, L] (1=valid, 0=padded)
        
        Returns:
            energy: Predicted energy values [B]
        """
        # Input validation
        self._validate_inputs(backbone_features, sequence_probs, mask)
        
        batch_size, seq_len = backbone_features.shape[:2]
        device = backbone_features.device
        
        # Create default mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Feature fusion: concatenate backbone features and sequence probabilities
        # [B, L, backbone_dim] + [B, L, seq_dim] → [B, L, backbone_dim + seq_dim]
        fused_features = torch.cat([backbone_features, sequence_probs], dim=-1)
        
        # Initial feature projection
        # [B, L, backbone_dim + seq_dim] → [B, L, hidden_dim]
        x = self.feature_fusion(fused_features)
        
        # Per-residue processing with residual connections
        for layer in self.residue_layers:
            x = layer(x, mask)  # [B, L, hidden_dim] → [B, L, hidden_dim]
        
        # Masked global pooling
        # Apply pooling projection first
        x = self.pooling_proj(x)  # [B, L, hidden_dim] → [B, L, hidden_dim // 2]
        
        # Compute masked mean pooling with safe division
        masked_x = x * mask.unsqueeze(-1)  # Apply mask
        mask_sum = mask.sum(dim=1, keepdim=True)  # [B, 1]
        
        # Prevent division by zero for empty sequences
        mask_sum = torch.clamp(mask_sum, min=1.0)  # At least 1 to avoid division by zero
        pooled_features = masked_x.sum(dim=1) / mask_sum  # [B, hidden_dim // 2]
        
        # Energy prediction
        energy = self.energy_head(pooled_features).squeeze(-1)  # [B, 1] → [B]
        
        # Scale energy if specified
        if self.energy_scale != 1.0:
            energy = energy * self.energy_scale
        
        return energy
    
    def compute_per_residue_features(
        self, 
        backbone_features: torch.Tensor, 
        sequence_probs: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get per-residue features before global pooling (for analysis/visualization).
        
        Args:
            backbone_features: Structural features [B, L, backbone_dim]
            sequence_probs: Sequence probabilities [B, L, seq_dim]
            mask: Sequence mask [B, L]
            
        Returns:
            per_residue_features: Per-residue processed features [B, L, hidden_dim]
        """
        self._validate_inputs(backbone_features, sequence_probs, mask)
        
        # Feature fusion
        fused_features = torch.cat([backbone_features, sequence_probs], dim=-1)
        x = self.feature_fusion(fused_features)
        
        # Per-residue processing
        for layer in self.residue_layers:
            x = layer(x, mask)
        
        return x
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.backbone_dim <= 0:
            raise ValueError(f"backbone_dim must be positive, got {self.backbone_dim}")
        if self.seq_dim <= 0:
            raise ValueError(f"seq_dim must be positive, got {self.seq_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.energy_scale <= 0:
            raise ValueError(f"energy_scale must be positive, got {self.energy_scale}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        
        return activations[activation.lower()]
    
    def _validate_inputs(
        self, 
        backbone_features: torch.Tensor, 
        sequence_probs: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ):
        """Validate forward pass inputs"""
        # Type checking
        if not isinstance(backbone_features, torch.Tensor):
            raise TypeError(f"backbone_features must be torch.Tensor, got {type(backbone_features)}")
        if not isinstance(sequence_probs, torch.Tensor):
            raise TypeError(f"sequence_probs must be torch.Tensor, got {type(sequence_probs)}")
        
        # Shape checking
        if backbone_features.dim() != 3:
            raise ValueError(f"backbone_features must be 3D [B, L, D], got shape {backbone_features.shape}")
        if sequence_probs.dim() != 3:
            raise ValueError(f"sequence_probs must be 3D [B, L, D], got shape {sequence_probs.shape}")
        
        # Dimension compatibility
        if backbone_features.shape[:2] != sequence_probs.shape[:2]:
            raise ValueError(f"Batch/sequence dimensions must match: backbone {backbone_features.shape[:2]} vs sequence {sequence_probs.shape[:2]}")
        
        if backbone_features.shape[2] != self.backbone_dim:
            raise ValueError(f"backbone_features last dim must be {self.backbone_dim}, got {backbone_features.shape[2]}")
        
        if sequence_probs.shape[2] != self.seq_dim:
            raise ValueError(f"sequence_probs last dim must be {self.seq_dim}, got {sequence_probs.shape[2]}")
        
        # Mask validation
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"mask must be torch.Tensor, got {type(mask)}")
            if mask.dim() != 2:
                raise ValueError(f"mask must be 2D [B, L], got shape {mask.shape}")
            if mask.shape != backbone_features.shape[:2]:
                raise ValueError(f"mask shape {mask.shape} must match batch/sequence dims {backbone_features.shape[:2]}")
        
        # Value checking with detailed debugging
        print(f"DEBUG energy_head: backbone_features shape={backbone_features.shape}, min={backbone_features.min().item():.6f}, max={backbone_features.max().item():.6f}")
        print(f"DEBUG energy_head: backbone_features NaN: {torch.isnan(backbone_features).any().item()}, Inf: {torch.isinf(backbone_features).any().item()}")
        
        print(f"DEBUG energy_head: sequence_probs shape={sequence_probs.shape}, min={sequence_probs.min().item():.6f}, max={sequence_probs.max().item():.6f}")
        print(f"DEBUG energy_head: sequence_probs NaN: {torch.isnan(sequence_probs).any().item()}, Inf: {torch.isinf(sequence_probs).any().item()}")
        print(f"DEBUG energy_head: Expected seq_dim={self.seq_dim}, actual last_dim={sequence_probs.shape[-1]}")
        print(f"DEBUG energy_head: Dimension check: {sequence_probs.shape[-1]} != {self.seq_dim} = {sequence_probs.shape[-1] != self.seq_dim}")
        
        if torch.isnan(backbone_features).any() or torch.isinf(backbone_features).any():
            print("DEBUG energy_head: BACKBONE FEATURES CONTAIN NaN/Inf!")
            raise ValueError("backbone_features contains NaN or Inf values")
        # Validate sequence_probs dimensions (should be correct now)
        if sequence_probs.shape[-1] != self.seq_dim:
            print(f"ERROR energy_head: DIMENSION MISMATCH! sequence_probs has shape {sequence_probs.shape}, expected last dim {self.seq_dim}")
            if sequence_probs.shape[-1] == backbone_features.shape[-1]:
                raise ValueError(f"sequence_probs appears to be backbone_features! Shape: {sequence_probs.shape}. This indicates a model configuration error.")
            else:
                raise ValueError(f"sequence_probs has wrong dimension {sequence_probs.shape[-1]}, expected {self.seq_dim}. Check sequence_repr configuration.")

        # Validate sequence_probs for NaN/Inf values - NO CLEANING, WILL CRASH TO EXPOSE ROOT CAUSE
        if torch.isnan(sequence_probs).any() or torch.isinf(sequence_probs).any():
            nan_count = torch.isnan(sequence_probs).sum().item()
            inf_count = torch.isinf(sequence_probs).sum().item()
            print(f"ERROR energy_head: SEQUENCE PROBS CONTAIN {nan_count} NaN and {inf_count} Inf values!")
            print(f"ERROR energy_head: This should help identify where NaNs originate - no cleaning performed!")
            print(f"ERROR energy_head: Check sequence_repr implementation and input sequence_logits")
            raise ValueError(f"sequence_probs contains {nan_count} NaN and {inf_count} Inf values. "
                           f"This indicates a numerical stability issue. Cleaning DISABLED to expose root cause. "
                           f"Check sequence_repr implementation, temperature settings, and input sequence_logits.")
        
        # Check for potential memory issues with very long sequences
        seq_len = backbone_features.shape[1]
        if seq_len > 2000:
            import warnings
            warnings.warn(f"Very long sequence (length {seq_len}). Consider using gradient checkpointing for memory efficiency.")
        
        # Warn about empty sequences
        if mask is not None and (mask.sum(dim=1) == 0).any():
            import warnings
            warnings.warn("Some sequences are completely masked (empty). Energy predictions may be unreliable.")
    
    def _initialize_weights(self):
        """Initialize model weights with appropriate schemes"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # Standard batch norm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'backbone_dim': self.backbone_dim,
            'seq_dim': self.seq_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'energy_scale': self.energy_scale,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ResidualBlock(nn.Module):
    """
    Residual processing block for per-residue features.
    
    Applies linear transformation with residual connection, optional batch normalization,
    and dropout for stable training.
    """
    
    def __init__(
        self, 
        hidden_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        
        # Main transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Optional batch norm after residual
        self.final_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input features [B, L, hidden_dim]
            mask: Optional mask [B, L]
            
        Returns:
            output: Transformed features [B, L, hidden_dim]
        """
        residual = x
        
        # For batch norm, we need to reshape to [B*L, hidden_dim]
        if self.use_batch_norm:
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.view(-1, hidden_dim)
            
            # Apply transformation
            out_flat = self.transform(x_flat)
            out = out_flat.view(batch_size, seq_len, hidden_dim)
            
            # Residual connection
            out = out + residual
            
            # Final batch norm
            out_flat = out.view(-1, hidden_dim)
            out_flat = self.final_norm(out_flat)
            out = out_flat.view(batch_size, seq_len, hidden_dim)
        else:
            # Standard path without batch norm
            out = self.transform(x) + residual
        
        # Apply mask if provided
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        
        return out


if __name__ == "__main__":
    # Example usage and testing
    print("Testing EnergyHead...")
    
    # Create energy head
    energy_head = EnergyHead(
        backbone_dim=128,
        seq_dim=20,
        hidden_dim=512,
        num_layers=3
    )
    
    print(f"Model configuration: {energy_head.get_config()}")
    
    # Test data
    batch_size, seq_len = 4, 75
    backbone_features = torch.randn(batch_size, seq_len, 128)
    sequence_probs = F.softmax(torch.randn(batch_size, seq_len, 20), dim=-1)
    mask = torch.ones(batch_size, seq_len)
    
    # Mask some positions for testing variable lengths
    mask[0, 50:] = 0  # First sequence has length 50
    mask[1, 60:] = 0  # Second sequence has length 60
    
    print(f"Input shapes:")
    print(f"  Backbone features: {backbone_features.shape}")
    print(f"  Sequence probs: {sequence_probs.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Effective lengths: {mask.sum(dim=1).int().tolist()}")
    
    # Forward pass
    with torch.no_grad():
        energy = energy_head(backbone_features, sequence_probs, mask)
        print(f"✓ Energy output shape: {energy.shape}")
        print(f"✓ Energy values: {energy.tolist()}")
    
    # Test gradient flow
    backbone_features.requires_grad_(True)
    sequence_probs.requires_grad_(True)
    
    energy = energy_head(backbone_features, sequence_probs, mask)
    loss = energy.sum()
    loss.backward()
    
    print(f"✓ Backbone gradient flow: {backbone_features.grad is not None}")
    print(f"✓ Sequence gradient flow: {sequence_probs.grad is not None}")
    print(f"✓ Backbone gradient norm: {backbone_features.grad.norm():.6f}")
    print(f"✓ Sequence gradient norm: {sequence_probs.grad.norm():.6f}")
    
    # Test per-residue features
    with torch.no_grad():
        per_res_features = energy_head.compute_per_residue_features(
            backbone_features, sequence_probs, mask
        )
        print(f"✓ Per-residue features shape: {per_res_features.shape}")
    
    # Test different sequence lengths
    short_backbone = torch.randn(2, 20, 128)
    short_sequence = F.softmax(torch.randn(2, 20, 20), dim=-1)
    short_mask = torch.ones(2, 20)
    
    with torch.no_grad():
        short_energy = energy_head(short_backbone, short_sequence, short_mask)
        print(f"✓ Short sequence energy: {short_energy.shape}")
    
    print("\n✓ All tests passed!")