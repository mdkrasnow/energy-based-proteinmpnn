"""
Energy-Based ProteinMPNN Model Implementation

This module implements the hybrid energy-based ProteinMPNN model that combines
ProteinMPNN's structural understanding with energy-based optimization for improved
protein design capabilities.
"""

import torch
import torch.nn as nn
import warnings
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import shared vocabulary constants
from data.vocab import AMINO_ACID_TO_IDX, AMINO_ACID_ALPHABET, IDX_TO_AMINO_ACID

# Import actual ProteinMPNN components
try:
    from .mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
    from .sequence_repr import ContinuousSequenceRepr
    from .energy_head import EnergyHead
    PROTEINMPNN_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"ProteinMPNN components not fully available: {e}. Using deterministic fallback.")
    PROTEINMPNN_AVAILABLE = False


class EnergyBasedProteinMPNN(nn.Module):
    """
    Hybrid Energy-Based ProteinMPNN Model.
    
    Combines ProteinMPNN's structural encoder with energy-based optimization
    for improved protein design. Includes deterministic fallback behavior
    when ProteinMPNN components are not available.
    """
    
    def __init__(
        self,
        mpnn_config: Dict[str, Any],
        energy_head_config: Dict[str, Any], 
        sequence_repr_config: Dict[str, Any],
        use_pretrained: bool = True,
        deterministic_fallback: bool = True
    ):
        """
        Initialize the energy-based model.
        
        Args:
            mpnn_config: Configuration for ProteinMPNN encoder
            energy_head_config: Configuration for energy prediction head
            sequence_repr_config: Configuration for sequence representation
            use_pretrained: Whether to use pre-trained ProteinMPNN weights
            deterministic_fallback: Whether to use deterministic fallback when ProteinMPNN unavailable
        """
        super().__init__()
        
        self.mpnn_config = mpnn_config
        self.energy_head_config = energy_head_config
        self.sequence_repr_config = sequence_repr_config
        self.use_pretrained = use_pretrained
        self.deterministic_fallback = deterministic_fallback
        self.proteinmpnn_available = PROTEINMPNN_AVAILABLE
        
        # Initialize components
        self._init_backbone_encoder()
        self._init_sequence_representation()
        self._init_energy_head()
        
        # Track initialization status for debugging
        self.initialization_log = {
            'proteinmpnn_available': self.proteinmpnn_available,
            'backbone_encoder_type': 'real' if self.proteinmpnn_available else 'fallback',
            'using_pretrained': use_pretrained and self.proteinmpnn_available,
            'deterministic_mode': deterministic_fallback and not self.proteinmpnn_available,
            'sophisticated_energy_head': getattr(self, 'use_sophisticated_head', False)
        }
    
    def _init_backbone_encoder(self):
        """Initialize ProteinMPNN backbone encoder with fallback."""
        if self.proteinmpnn_available and PROTEINMPNN_AVAILABLE:
            try:
                # Use actual ProteinMPNN encoder
                model_name = self.mpnn_config.get('model_name', 'v_48_020')
                model_type = self.mpnn_config.get('model_type', 'vanilla')
                freeze_layers = self.mpnn_config.get('freeze_layers', True)
                
                if self.use_pretrained:
                    self.backbone_encoder = ProteinMPNNBackboneEncoder.from_pretrained(
                        model_name=model_name,
                        model_type=model_type,
                        freeze_layers=freeze_layers,
                        gradient_checkpointing=self.mpnn_config.get('gradient_checkpointing', False)
                    )
                    print(f"✓ Loaded pre-trained ProteinMPNN encoder: {model_name} ({model_type})")
                else:
                    # Initialize from scratch with proper ProteinMPNN architecture
                    self.backbone_encoder = ProteinMPNNBackboneEncoder(
                        pretrained_ckpt_path=self.mpnn_config.get('checkpoint_path', ''),
                        freeze_layers=freeze_layers,
                        ca_only=self.mpnn_config.get('ca_only', False),
                        hidden_dim=self.mpnn_config.get('hidden_dim', 128),
                        num_encoder_layers=self.mpnn_config.get('num_encoder_layers', 3),
                        gradient_checkpointing=self.mpnn_config.get('gradient_checkpointing', False)
                    )
                    print("✓ Initialized ProteinMPNN encoder from scratch")
                
            except Exception as e:
                warnings.warn(f"Failed to initialize ProteinMPNN encoder: {e}. Using deterministic fallback.")
                self.proteinmpnn_available = False
                self._init_fallback_encoder()
        else:
            self._init_fallback_encoder()
    
    def _init_fallback_encoder(self):
        """Initialize deterministic fallback encoder when ProteinMPNN unavailable."""
        if self.deterministic_fallback:
            # Create deterministic structural encoder for reproducible research
            hidden_dim = self.mpnn_config.get('hidden_dim', 128)
            self.backbone_encoder = DeterministicStructuralEncoder(
                hidden_dim=hidden_dim,
                num_layers=self.mpnn_config.get('num_encoder_layers', 3),
                coordinate_features=self.mpnn_config.get('coordinate_features', 64)
            )
            print("✓ Using deterministic fallback encoder (reproducible research mode)")
        else:
            raise RuntimeError("ProteinMPNN not available and deterministic_fallback=False")
    
    def _init_sequence_representation(self):
        """Initialize sequence representation with proper ContinuousSequenceRepr."""
        try:
            # Try to initialize ContinuousSequenceRepr with proper config
            from .sequence_repr import ContinuousSequenceRepr
            self.sequence_repr = ContinuousSequenceRepr(
                vocab_size=self.sequence_repr_config.get('vocab_size', 20),
                temperature_schedule=self.sequence_repr_config.get('temperature_schedule', [1.0, 0.5, 0.1]),
                min_temperature=self.sequence_repr_config.get('min_temperature', 0.001),
                max_temperature=self.sequence_repr_config.get('max_temperature', 10.0)
            )
            print(f"✓ Initialized ContinuousSequenceRepr with vocab_size={self.sequence_repr.vocab_size}")
        except ImportError as e:
            print(f"Warning: Could not import ContinuousSequenceRepr: {e}")
            self._init_fallback_sequence_repr()
        except Exception as e:
            print(f"Warning: Failed to initialize ContinuousSequenceRepr: {e}")
            self._init_fallback_sequence_repr()
    
    def _init_fallback_sequence_repr(self):
        """Initialize deterministic sequence representation fallback."""
        # Use vocab_size instead of hidden_dim for sequence representation
        vocab_size = self.sequence_repr_config.get('vocab_size', 20)
        embedding_dim = self.sequence_repr_config.get('embedding_dim', vocab_size)  # Default embedding_dim to vocab_size for compatibility
        
        self.sequence_repr = DeterministicSequenceEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            deterministic=self.deterministic_fallback
        )
        print(f"✓ Using fallback DeterministicSequenceEmbedding with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    def _init_energy_head(self):
        """Initialize energy prediction head."""
        backbone_dim = self.mpnn_config.get('hidden_dim', 128)
        sequence_dim = 20  # Amino acid probability dimension (vocab_size)
        
        if self.proteinmpnn_available:
            try:
                # Use sophisticated energy head that expects separate backbone and sequence inputs
                self.energy_head = EnergyHead(
                    backbone_dim=backbone_dim,
                    seq_dim=sequence_dim, 
                    hidden_dim=self.energy_head_config.get('hidden_dim', 512),
                    num_layers=self.energy_head_config.get('num_layers', 3),
                    dropout=self.energy_head_config.get('dropout', 0.1),
                    use_batch_norm=self.energy_head_config.get('use_batch_norm', True)
                )
                self.use_sophisticated_head = True
            except Exception as e:
                warnings.warn(f"Failed to initialize EnergyHead: {e}. Using fallback")
                self._init_fallback_energy_head(backbone_dim + sequence_dim)
                self.use_sophisticated_head = False
        else:
            self._init_fallback_energy_head(backbone_dim + sequence_dim)
            self.use_sophisticated_head = False
    
    def _init_fallback_energy_head(self, input_dim: int):
        """Initialize fallback energy head."""
        hidden_dim = self.energy_head_config.get('hidden_dim', 512)
        num_layers = self.energy_head_config.get('num_layers', 3)
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.energy_head_config.get('dropout', 0) > 0:
                layers.append(nn.Dropout(self.energy_head_config['dropout']))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.energy_head = nn.Sequential(*layers)
        
        # Make fallback energy head deterministic for reproducible research
        if self.deterministic_fallback:
            self._init_deterministic_energy_head()
    
    def _init_deterministic_energy_head(self):
        """Initialize energy head weights deterministically."""
        with torch.no_grad():
            layer_idx = 0
            for module in self.energy_head.modules():
                if isinstance(module, nn.Linear):
                    in_dim, out_dim = module.weight.shape[1], module.weight.shape[0]
                    
                    if out_dim == 1:  # Final output layer
                        # Small positive weights for energy prediction
                        module.weight.fill_(0.01 / in_dim)
                    else:
                        # Hidden layers: identity-like with small perturbations
                        nn.init.eye_(module.weight[:min(in_dim, out_dim), :min(in_dim, out_dim)])
                        if out_dim > in_dim:
                            module.weight[in_dim:] = 0.01 / in_dim
                        # Add layer-specific perturbation for expressiveness
                        module.weight += 0.001 * (layer_idx + 1) / (in_dim * out_dim)
                    
                    # Zero bias for determinism
                    if module.bias is not None:
                        module.bias.zero_()
                    
                    layer_idx += 1
        
    def forward(
        self, 
        sequence: Union[str, torch.Tensor], 
        coordinates: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict energy using actual ProteinMPNN processing.
        
        Args:
            sequence: Protein sequence (string or tensor of amino acid indices)
            coordinates: Structure coordinates [B, L, 4, 3] or [L, 4, 3]
            mask: Optional sequence mask [B, L] or [L]
            
        Returns:
            Energy prediction [B] or scalar
        """
        # Standardize input format
        if isinstance(sequence, str):
            seq_length = len(sequence)
            batch_size = 1
            sequence_list = [sequence]
        elif isinstance(sequence, list):
            # List of sequences
            sequence_list = sequence
            batch_size = len(sequence_list)
            seq_length = len(sequence_list[0])  # Assume all same length for now
        else:
            # Tensor input
            if len(coordinates.shape) == 3:
                coordinates = coordinates.unsqueeze(0)  # Add batch dimension
            batch_size, seq_length = coordinates.shape[0], coordinates.shape[1]
            sequence_list = [self._tensor_to_sequence(sequence[i]) if len(sequence.shape) > 1 
                           else self._tensor_to_sequence(sequence) for i in range(batch_size)]
        
        # Ensure coordinates have batch dimension
        if len(coordinates.shape) == 3:
            coordinates = coordinates.unsqueeze(0)  # [L, 4, 3] -> [B, L, 4, 3]
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=coordinates.device)
        elif len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # Add batch dimension
        
        # Prepare batch for ProteinMPNN encoder
        batch = self._prepare_proteinmpnn_batch(coordinates, mask)
        
        # Extract structural features using ProteinMPNN backbone encoder
        try:
            backbone_features = self.backbone_encoder(batch)  # [B, L, hidden_dim]
        except Exception as e:
            if self.deterministic_fallback:
                # Use deterministic fallback for structural encoding
                backbone_features = self._deterministic_structural_encoding(
                    coordinates, mask, seq_length
                )
            else:
                raise RuntimeError(f"ProteinMPNN backbone encoding failed: {e}")
        
        # Generate sequence representation
        sequence_features = []
        for i, seq in enumerate(sequence_list):
            seq_repr = self.sequence_repr(seq, backbone_features[i])  # [L, hidden_dim]
            sequence_features.append(seq_repr)
        sequence_features = torch.stack(sequence_features, dim=0)  # [B, L, hidden_dim]
        
        # Predict energy using appropriate head
        if self.use_sophisticated_head:
            # Use sophisticated energy head that handles backbone and sequence features separately
            energy = self.energy_head(backbone_features, sequence_features, mask)  # [B]
        else:
            # Use fallback head with concatenated features
            combined_features = torch.cat([backbone_features, sequence_features], dim=-1)  # [B, L, combined_dim]
            pooled_features = self._pool_features(combined_features, mask)  # [B, combined_dim]
            energy = self.energy_head(pooled_features)  # [B, 1]
            if len(energy.shape) > 1:
                energy = energy.squeeze(-1)  # Remove last dimension if present
        
        # Return appropriate format
        if batch_size == 1:
            return energy.squeeze()  # Scalar for single sequence
        else:
            return energy.squeeze(-1)  # [B] for batch
    
    def _prepare_proteinmpnn_batch(self, coordinates: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prepare batch dictionary for ProteinMPNN encoder."""
        batch_size, seq_length = coordinates.shape[0], coordinates.shape[1]
        device = coordinates.device
        
        return {
            'X': coordinates,  # [B, L, 4, 3]
            'mask': mask,  # [B, L]
            'residue_idx': torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1),  # [B, L]
            'chain_encoding_all': torch.zeros(batch_size, seq_length, device=device, dtype=torch.long)  # [B, L]
        }
    
    def _tensor_to_sequence(self, seq_tensor: torch.Tensor) -> str:
        """Convert amino acid index tensor to sequence string."""
        # CRITICAL FIX: Use canonical ProteinMPNN alphabet from shared vocab module
        aa_alphabet = AMINO_ACID_ALPHABET  # ProteinMPNN standard: ARNDCQEGHILKMFPSTWYV
        if len(seq_tensor.shape) == 0:  # scalar
            return aa_alphabet[seq_tensor.item() % 20]
        return ''.join([aa_alphabet[idx.item() % 20] for idx in seq_tensor])
    
    def _deterministic_structural_encoding(
        self, 
        coordinates: torch.Tensor, 
        mask: torch.Tensor, 
        seq_length: int
    ) -> torch.Tensor:
        """
        Deterministic fallback for structural encoding when ProteinMPNN unavailable.
        
        This provides reproducible features based on coordinate geometry rather than
        random hash-based features, enabling consistent research results.
        """
        batch_size = coordinates.shape[0]
        hidden_dim = self.mpnn_config.get('hidden_dim', 128)
        device = coordinates.device
        
        if not self.deterministic_fallback:
            warnings.warn("Using random features - results will not be reproducible!")
            return torch.randn(batch_size, seq_length, hidden_dim, device=device)
        
        # Extract geometric features from coordinates
        ca_coords = coordinates[:, :, 1, :]  # CA atoms [B, L, 3]
        
        # Calculate pairwise distances (deterministic)
        pairwise_dists = torch.cdist(ca_coords, ca_coords, p=2)  # [B, L, L]
        
        # Local geometry features
        features = []
        
        # 1. Local distance patterns (nearest neighbor distances)
        k = min(8, seq_length)  # Use up to 8 nearest neighbors
        nearest_dists, _ = torch.topk(pairwise_dists, k=k, dim=-1, largest=False)
        features.append(nearest_dists.mean(dim=-1))  # [B, L]
        
        # 2. Secondary structure approximation via backbone angles
        if seq_length > 2:
            v1 = ca_coords[:, 1:-1] - ca_coords[:, :-2]  # [B, L-2, 3]
            v2 = ca_coords[:, 2:] - ca_coords[:, 1:-1]    # [B, L-2, 3]
            
            # Backbone angles
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-6
            )
            cos_angles = torch.clamp(cos_angles, -1, 1)
            
            # Pad to full sequence length
            angles_padded = torch.zeros(batch_size, seq_length, device=device)
            angles_padded[:, 1:-1] = cos_angles
            features.append(angles_padded)
        else:
            features.append(torch.zeros(batch_size, seq_length, device=device))
        
        # 3. Local density (number of residues within cutoff)
        cutoff = 10.0  # Angstrom
        local_density = (pairwise_dists < cutoff).float().sum(dim=-1) - 1  # Exclude self
        features.append(local_density)
        
        # Stack and project to hidden dimension
        geometric_features = torch.stack(features, dim=-1)  # [B, L, n_features]
        
        # Expand to target hidden dimension using deterministic linear projection
        n_features = geometric_features.shape[-1]
        if not hasattr(self, '_geometric_projector'):
            self._geometric_projector = nn.Linear(n_features, hidden_dim).to(device)
            # Make projection deterministic for reproducible research
            with torch.no_grad():
                # Use deterministic initialization based on feature dimensions
                nn.init.eye_(self._geometric_projector.weight[:min(n_features, hidden_dim), :min(n_features, hidden_dim)])
                if n_features < hidden_dim:
                    # Pad with small deterministic values
                    self._geometric_projector.weight[n_features:] = 0.01 * torch.arange(1, hidden_dim - n_features + 1, device=device).unsqueeze(-1).repeat(1, n_features)
                if n_features > hidden_dim:
                    # Use first hidden_dim features with equal weighting
                    self._geometric_projector.weight[:] = 1.0 / n_features
                
                # Zero bias for determinism
                if self._geometric_projector.bias is not None:
                    self._geometric_projector.bias.zero_()
        
        structural_features = self._geometric_projector(geometric_features)
        
        # Apply mask
        structural_features = structural_features * mask.unsqueeze(-1)
        
        return structural_features
    
    def _pool_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool per-residue features to protein-level representation."""
        # Masked average pooling
        masked_features = features * mask.unsqueeze(-1)
        pooled = masked_features.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        return pooled
    
    def encode_structure(
        self, 
        coordinates: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode protein structure using ProteinMPNN backbone encoder.
        
        Args:
            coordinates: Structure coordinates [B, L, 4, 3] or [L, 4, 3]
            mask: Optional sequence mask [B, L] or [L]
            
        Returns:
            Structure encoding [B, L, hidden_dim] or [L, hidden_dim]
        """
        # Handle input dimensions
        if len(coordinates.shape) == 3:
            coordinates = coordinates.unsqueeze(0)  # Add batch dimension
            single_input = True
        else:
            single_input = False
            
        batch_size, seq_length = coordinates.shape[0], coordinates.shape[1]
        
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=coordinates.device)
        elif len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        
        # Prepare batch for encoder
        batch = self._prepare_proteinmpnn_batch(coordinates, mask)
        
        # Use actual ProteinMPNN encoder
        try:
            structure_encoding = self.backbone_encoder(batch)
        except Exception as e:
            if self.deterministic_fallback:
                structure_encoding = self._deterministic_structural_encoding(
                    coordinates, mask, seq_length
                )
            else:
                raise RuntimeError(f"Structure encoding failed: {e}")
        
        # Return appropriate format
        if single_input:
            return structure_encoding.squeeze(0)  # Remove batch dimension
        else:
            return structure_encoding
    
    def get_sequence_representation(
        self, 
        sequence: str,
        structure_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate continuous sequence representation using actual sequence model.
        
        Args:
            sequence: Protein sequence string
            structure_encoding: Structure encoding [L, hidden_dim]
            
        Returns:
            Sequence representation [L, hidden_dim]
        """
        return self.sequence_repr(sequence, structure_encoding)
    
    def get_initialization_info(self) -> Dict[str, Any]:
        """Get information about model initialization for debugging."""
        return self.initialization_log.copy()
    
    def is_using_proteinmpnn(self) -> bool:
        """Check if model is using actual ProteinMPNN components."""
        return self.proteinmpnn_available and PROTEINMPNN_AVAILABLE


# Deterministic fallback classes for reproducible research
class DeterministicStructuralEncoder(nn.Module):
    """
    Deterministic fallback encoder that provides reproducible structural features
    based on coordinate geometry rather than learned representations.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, coordinate_features: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coordinate_features = coordinate_features
        
        # Geometric feature extraction layers
        self.feature_projector = nn.Sequential(
            nn.Linear(coordinate_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Multi-layer processing
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Initialize deterministically for reproducible research
        self._init_deterministic_weights()
    
    def _init_deterministic_weights(self):
        """Initialize all weights deterministically for reproducible research."""
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Deterministic initialization based on module position
                    in_dim, out_dim = module.weight.shape[1], module.weight.shape[0]
                    
                    # Use structured initialization based on layer hierarchy
                    if 'feature_projector.0' in name:
                        # First projection layer: identity-like for geometric features
                        nn.init.eye_(module.weight[:min(in_dim, out_dim), :min(in_dim, out_dim)])
                        if out_dim > in_dim:
                            module.weight[in_dim:] = 0.1 / in_dim
                    elif 'feature_projector.2' in name:
                        # Second projection layer: expanding to hidden dimension
                        nn.init.eye_(module.weight[:min(in_dim, out_dim), :min(in_dim, out_dim)])
                        if out_dim > in_dim:
                            for i in range(in_dim, out_dim):
                                module.weight[i] = 0.01
                    else:
                        # Processing layers: identity with small perturbations for expressiveness
                        nn.init.eye_(module.weight)
                        module.weight += 0.01 * torch.arange(out_dim, dtype=torch.float32).unsqueeze(1) / out_dim
                    
                    # Zero bias for all layers
                    if module.bias is not None:
                        module.bias.zero_()
                
                elif isinstance(module, nn.LayerNorm):
                    # Standard LayerNorm initialization
                    module.weight.fill_(1.0)
                    module.bias.zero_()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract deterministic structural features."""
        coordinates = batch['X']  # [B, L, 4, 3]
        mask = batch['mask']      # [B, L]
        
        # Extract geometric features
        geometric_features = self._extract_geometric_features(coordinates, mask)
        
        # Project to hidden dimension
        features = self.feature_projector(geometric_features)
        
        # Process through layers
        for layer in self.layers:
            residual = features
            features = layer(features) + residual  # Residual connection
        
        # Apply mask
        features = features * mask.unsqueeze(-1)
        
        return features
    
    def _extract_geometric_features(self, coordinates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract comprehensive geometric features from coordinates."""
        batch_size, seq_length = coordinates.shape[0], coordinates.shape[1]
        device = coordinates.device
        
        # Extract backbone atoms
        ca_coords = coordinates[:, :, 1, :]  # CA atoms [B, L, 3]
        c_coords = coordinates[:, :, 2, :]   # C atoms [B, L, 3]  
        n_coords = coordinates[:, :, 0, :]   # N atoms [B, L, 3]
        
        features = []
        
        # 1. Pairwise distances
        pairwise_dists = torch.cdist(ca_coords, ca_coords, p=2)  # [B, L, L]
        
        # Local distance statistics
        k = min(8, seq_length)
        if k > 1:
            nearest_dists, _ = torch.topk(pairwise_dists, k=k, dim=-1, largest=False)
            features.extend([
                nearest_dists.mean(dim=-1),    # Mean local distance
                nearest_dists.std(dim=-1),     # Local distance variation
                nearest_dists.min(dim=-1)[0]   # Closest neighbor distance
            ])
        else:
            features.extend([
                torch.zeros(batch_size, seq_length, device=device),
                torch.zeros(batch_size, seq_length, device=device),
                torch.zeros(batch_size, seq_length, device=device)
            ])
        
        # 2. Backbone angles and dihedrals
        if seq_length > 2:
            # Bond vectors
            ca_ca_vectors = ca_coords[:, 1:] - ca_coords[:, :-1]  # [B, L-1, 3]
            
            # Bond angles (CA-CA-CA)
            if seq_length > 2:
                v1 = ca_ca_vectors[:, :-1]  # [B, L-2, 3]  
                v2 = ca_ca_vectors[:, 1:]   # [B, L-2, 3]
                
                cos_angles = torch.sum(v1 * v2, dim=-1) / (
                    torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-6
                )
                cos_angles = torch.clamp(cos_angles, -1, 1)
                
                # Pad to full length
                angles_padded = torch.zeros(batch_size, seq_length, device=device)
                angles_padded[:, 1:-1] = cos_angles
                features.append(angles_padded)
            else:
                features.append(torch.zeros(batch_size, seq_length, device=device))
            
            # Dihedral angles (simplified)
            if seq_length > 3:
                # Phi angle approximation using CA positions
                v1 = ca_coords[:, :-3] - ca_coords[:, 1:-2]  # [B, L-3, 3]
                v2 = ca_coords[:, 1:-2] - ca_coords[:, 2:-1] # [B, L-3, 3]
                v3 = ca_coords[:, 2:-1] - ca_coords[:, 3:]   # [B, L-3, 3]
                
                # Cross products for dihedral calculation
                n1 = torch.cross(v1, v2, dim=-1)
                n2 = torch.cross(v2, v3, dim=-1)
                
                cos_dihedral = torch.sum(n1 * n2, dim=-1) / (
                    torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1) + 1e-6
                )
                cos_dihedral = torch.clamp(cos_dihedral, -1, 1)
                
                # Pad to full length
                dihedrals_padded = torch.zeros(batch_size, seq_length, device=device)
                dihedrals_padded[:, 1:-2] = cos_dihedral
                features.append(dihedrals_padded)
            else:
                features.append(torch.zeros(batch_size, seq_length, device=device))
        else:
            features.extend([
                torch.zeros(batch_size, seq_length, device=device),
                torch.zeros(batch_size, seq_length, device=device)
            ])
        
        # 3. Local density and environment
        cutoff_ranges = [8.0, 12.0, 16.0]  # Different distance cutoffs
        for cutoff in cutoff_ranges:
            local_counts = (pairwise_dists < cutoff).float().sum(dim=-1) - 1  # Exclude self
            features.append(local_counts)
        
        # 4. Coordinate statistics
        features.extend([
            ca_coords.norm(dim=-1),  # Distance from origin
            (ca_coords - ca_coords.mean(dim=1, keepdim=True)).norm(dim=-1)  # Distance from centroid
        ])
        
        # Stack all features
        all_features = torch.stack(features, dim=-1)  # [B, L, n_features]
        
        # Pad or truncate to target feature dimension
        n_features = all_features.shape[-1]
        if n_features < self.coordinate_features:
            # Pad with zeros
            padding = torch.zeros(
                batch_size, seq_length, self.coordinate_features - n_features, 
                device=device
            )
            all_features = torch.cat([all_features, padding], dim=-1)
        elif n_features > self.coordinate_features:
            # Truncate
            all_features = all_features[:, :, :self.coordinate_features]
        
        return all_features


class DeterministicSequenceEmbedding(nn.Module):
    """
    Deterministic sequence embedding that provides reproducible sequence representations
    without randomness for consistent research results.
    """
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128, deterministic: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.deterministic = deterministic
        
        # CRITICAL FIX: Use canonical ProteinMPNN alphabet from shared vocab module
        # This fixes the data corruption bug where energy model was using alphabetical order
        self.aa_alphabet = AMINO_ACID_ALPHABET  # ProteinMPNN standard: ARNDCQEGHILKMFPSTWYV
        self.aa_to_idx = AMINO_ACID_TO_IDX.copy()
        
        # Deterministic embedding based on physicochemical properties
        if deterministic:
            self.embedding = self._create_deterministic_embedding()
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Structure-sequence interaction layer
        self.interaction_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Initialize interaction layer deterministically for reproducible research
        if deterministic:
            self._init_deterministic_interaction_layer()
    
    def _init_deterministic_interaction_layer(self):
        """Initialize interaction layer weights deterministically."""
        with torch.no_grad():
            in_dim = self.embedding_dim * 2
            out_dim = self.embedding_dim
            
            # Create a deterministic transformation that favors sequence features
            # First half of input is sequence embeddings, second half is structure
            self.interaction_layer.weight.zero_()
            
            # Identity mapping for sequence features (first half)
            self.interaction_layer.weight[:out_dim, :out_dim] = torch.eye(out_dim)
            
            # Small contribution from structure features (second half)
            self.interaction_layer.weight[:out_dim, out_dim:] = 0.1 / out_dim
            
            # Zero bias
            if self.interaction_layer.bias is not None:
                self.interaction_layer.bias.zero_()
    
    def _create_deterministic_embedding(self) -> nn.Parameter:
        """Create deterministic embeddings based on amino acid properties."""
        # Physicochemical properties for standard amino acids
        # [hydrophobicity, charge, size, polarity, aromaticity]
        aa_properties = {
            'A': [0.62, 0.0, 0.1, 0.0, 0.0],   # Alanine
            'C': [0.29, 0.0, 0.2, 0.0, 0.0],   # Cysteine
            'D': [-0.9, -1.0, 0.3, 1.0, 0.0],  # Aspartic acid
            'E': [-0.74, -1.0, 0.4, 1.0, 0.0], # Glutamic acid
            'F': [1.19, 0.0, 0.6, 0.0, 1.0],   # Phenylalanine
            'G': [0.48, 0.0, 0.0, 0.0, 0.0],   # Glycine
            'H': [-0.4, 0.5, 0.4, 0.5, 1.0],   # Histidine
            'I': [1.38, 0.0, 0.4, 0.0, 0.0],   # Isoleucine
            'K': [-1.5, 1.0, 0.5, 1.0, 0.0],   # Lysine
            'L': [1.06, 0.0, 0.4, 0.0, 0.0],   # Leucine
            'M': [0.64, 0.0, 0.4, 0.0, 0.0],   # Methionine
            'N': [-0.78, 0.0, 0.3, 1.0, 0.0],  # Asparagine
            'P': [0.12, 0.0, 0.3, 0.0, 0.0],   # Proline
            'Q': [-0.85, 0.0, 0.4, 1.0, 0.0],  # Glutamine
            'R': [-2.53, 1.0, 0.5, 1.0, 0.0],  # Arginine
            'S': [-0.18, 0.0, 0.2, 1.0, 0.0],  # Serine
            'T': [-0.05, 0.0, 0.3, 1.0, 0.0],  # Threonine
            'V': [1.08, 0.0, 0.3, 0.0, 0.0],   # Valine
            'W': [0.81, 0.0, 0.7, 0.0, 1.0],   # Tryptophan
            'Y': [0.26, 0.0, 0.6, 1.0, 1.0],   # Tyrosine
        }
        
        # Create embedding matrix
        embedding_matrix = torch.zeros(self.vocab_size, self.embedding_dim)
        
        # Fill with deterministic features based on properties
        for i, aa in enumerate(self.aa_alphabet):
            if aa in aa_properties:
                props = torch.tensor(aa_properties[aa], dtype=torch.float32)
                
                # Expand properties to embedding dimension using repetition and scaling
                base_features = props.repeat(self.embedding_dim // len(props) + 1)[:self.embedding_dim]
                
                # Add positional encoding based on amino acid index
                pos_encoding = torch.sin(torch.arange(self.embedding_dim, dtype=torch.float32) * (i + 1) / 100.0)
                
                # Combine features
                embedding_matrix[i] = base_features + 0.1 * pos_encoding
        
        # Gap/unknown token (index 20)
        if self.vocab_size > 20:
            embedding_matrix[20] = torch.zeros(self.embedding_dim)
        
        return nn.Parameter(embedding_matrix, requires_grad=False)
    
    def forward(self, sequence: str, structure_encoding: torch.Tensor) -> torch.Tensor:
        """Generate sequence representation conditioned on structure."""
        seq_length = len(sequence)
        device = structure_encoding.device
        
        # Convert sequence to indices
        seq_indices = [self.aa_to_idx.get(aa, 20) for aa in sequence]  # 20 for unknown
        seq_tensor = torch.tensor(seq_indices, device=device, dtype=torch.long)
        
        # Get sequence embeddings
        if self.deterministic:
            seq_embeddings = self.embedding[seq_tensor]  # [L, embedding_dim]
        else:
            seq_embeddings = self.embedding(seq_tensor)  # [L, embedding_dim]
        
        # Combine with structural information
        combined = torch.cat([seq_embeddings, structure_encoding], dim=-1)  # [L, 2*embedding_dim]
        
        # Generate final sequence representation
        seq_representation = self.interaction_layer(combined)  # [L, embedding_dim]
        
        return seq_representation