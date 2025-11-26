"""
ProteinMPNN Backbone Encoder

This module extracts and wraps the encoder components from ProteinMPNN for use in the hybrid
energy-based design system. It provides a frozen/fine-tunable backbone encoder that produces
per-residue structural embeddings.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add proteinmpnn to path with robust absolute path resolution
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Go up from models/ to hybrid/ to project root/
proteinmpnn_path = project_root / "proteinmpnn"

# Ensure proteinmpnn directory exists before adding to path
if proteinmpnn_path.exists():
    sys.path.insert(0, str(proteinmpnn_path))
else:
    # Fallback: try to find proteinmpnn relative to current working directory
    fallback_path = Path.cwd() / "proteinmpnn"
    if fallback_path.exists():
        sys.path.insert(0, str(fallback_path))
    else:
        raise ImportError(f"Cannot locate proteinmpnn directory. Checked: {proteinmpnn_path}, {fallback_path}")

try:
    # Primary import attempt from protein_mpnn_utils
    from protein_mpnn_utils import ProteinMPNN, ProteinFeatures, CA_ProteinFeatures, EncLayer
    from protein_mpnn_utils import gather_nodes
except ImportError as primary_error:
    try:
        # Fallback 1: Try package-style import
        from proteinmpnn.protein_mpnn_utils import ProteinMPNN, ProteinFeatures, CA_ProteinFeatures, EncLayer
        from proteinmpnn.protein_mpnn_utils import gather_nodes
    except ImportError as fallback_error:
        try:
            # Fallback 2: Try adding project root to path and importing again
            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))
            from proteinmpnn.protein_mpnn_utils import ProteinMPNN, ProteinFeatures, CA_ProteinFeatures, EncLayer
            from proteinmpnn.protein_mpnn_utils import gather_nodes
        except ImportError as final_error:
            # All import strategies failed - provide comprehensive error message
            error_details = [
                f"Primary import error: {primary_error}",
                f"Package fallback error: {fallback_error}", 
                f"Final fallback error: {final_error}",
                f"Checked paths: {[str(p) for p in sys.path if 'proteinmpnn' in str(p)]}",
                f"Working directory: {Path.cwd()}",
                f"Script location: {Path(__file__).resolve()}"
            ]
            raise ImportError(
                f"Could not import ProteinMPNN utilities after trying multiple strategies.\n" +
                "\n".join(error_details) +
                f"\n\nPlease ensure:\n"
                f"1. The proteinmpnn directory exists in the project root\n"
                f"2. protein_mpnn_utils.py is present in proteinmpnn/\n"
                f"3. All required dependencies are installed\n"
                f"4. Python path includes the proteinmpnn directory"
            )


class ProteinMPNNBackboneEncoder(nn.Module):
    """
    ProteinMPNN Backbone Encoder wrapper that extracts structural features from protein backbones.
    
    This class wraps the encoder components of a pre-trained ProteinMPNN model, providing
    per-residue embeddings for downstream energy modeling with optional gradient checkpointing
    for memory optimization during large protein processing.
    
    Args:
        pretrained_ckpt_path: Path to pre-trained ProteinMPNN checkpoint
        freeze_layers: If True, freeze all encoder parameters (default: True)
        ca_only: If True, use CA-only model (default: False)
        hidden_dim: Hidden dimension size (default: 128)
        num_encoder_layers: Number of encoder layers (default: 3)
        node_features: Node feature dimension (default: 128)
        edge_features: Edge feature dimension (default: 128)
        k_neighbors: Number of neighbors for graph construction (default: 64)
        gradient_checkpointing: If True, use gradient checkpointing to save memory (default: False)
    """
    
    def __init__(
        self,
        pretrained_ckpt_path: str,
        freeze_layers: bool = True,
        ca_only: bool = False,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        node_features: int = 128,
        edge_features: int = 128,
        k_neighbors: int = 64,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.freeze_layers = freeze_layers
        self.ca_only = ca_only
        self.gradient_checkpointing = gradient_checkpointing
        
        # Load pre-trained ProteinMPNN model
        self._load_pretrained_model(pretrained_ckpt_path)
        
        # Extract components needed for encoding
        self._extract_encoder_components()
        
        # Freeze parameters if requested
        if freeze_layers:
            self._freeze_parameters()
    
    def _load_pretrained_model(self, checkpoint_path: str):
        """Load pre-trained ProteinMPNN model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create ProteinMPNN model with standard parameters
            # These are the standard ProteinMPNN parameters
            self.full_model = ProteinMPNN(
                num_letters=21,
                node_features=128,  
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=21,
                k_neighbors=64,
                augment_eps=0.05,
                dropout=0.1,
                ca_only=self.ca_only
            )
            
            # Load state dict
            self.full_model.load_state_dict(checkpoint, strict=False)
            self.full_model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ProteinMPNN checkpoint: {e}")
    
    def _extract_encoder_components(self):
        """Extract the encoder components from the full ProteinMPNN model"""
        # Extract graph builder (features)
        self.graph_builder = self.full_model.features
        
        # Extract encoder layers
        self.encoder_layers = self.full_model.encoder_layers
        
        # Extract edge embedding layer
        self.W_e = self.full_model.W_e
        
        # Store for later use
        self.node_features = self.full_model.node_features
        self.edge_features = self.full_model.edge_features
    
    def _freeze_parameters(self):
        """Freeze all encoder parameters while preserving computation graph"""
        # Set model to eval mode for frozen behavior
        self.full_model.eval()
        
        # Store original requires_grad states for potential unfreezing
        if not hasattr(self, '_original_requires_grad'):
            self._original_requires_grad = {}
            for name, param in self.full_model.named_parameters():
                self._original_requires_grad[name] = param.requires_grad
        
        # Disable gradients for frozen behavior
        for param in self.full_model.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_indices: Optional[list] = None):
        """
        Unfreeze specific encoder layers for fine-tuning
        
        Args:
            layer_indices: List of layer indices to unfreeze. If None, unfreeze all layers.
        """
        # Set to training mode for unfrozen behavior
        self.full_model.train()
        
        if layer_indices is None:
            # Unfreeze all layers - restore original requires_grad states
            if hasattr(self, '_original_requires_grad'):
                for name, param in self.full_model.named_parameters():
                    if name in self._original_requires_grad:
                        param.requires_grad = self._original_requires_grad[name]
            else:
                # Fallback: enable all parameters
                for param in self.full_model.parameters():
                    param.requires_grad = True
        else:
            # Unfreeze specific encoder layers
            for idx in layer_indices:
                if 0 <= idx < len(self.encoder_layers):
                    for param in self.encoder_layers[idx].parameters():
                        param.requires_grad = True
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the encoder to extract backbone features
        
        Args:
            batch: Dictionary containing:
                - X: Backbone coordinates [B, L, 4, 3] (N, CA, C, O)
                - mask: Sequence mask [B, L] 
                - residue_idx: Residue indices [B, L]
                - chain_encoding_all: Chain encoding [B, L]
        
        Returns:
            node_features: Per-residue embeddings [B, L, hidden_dim]
        """
        device = batch['X'].device
        
        # Extract inputs
        X = batch['X']  # [B, L, 4, 3] for full backbone or [B, L, 3] for CA-only
        mask = batch['mask']  # [B, L]
        residue_idx = batch['residue_idx']  # [B, L]
        chain_encoding_all = batch['chain_encoding_all']  # [B, L]
        
        # Handle coordinate format based on model type
        if self.ca_only:
            # CA-only models expect [B, L, 3] coordinates
            if len(X.shape) != 3 or X.shape[-1] != 3:
                raise ValueError(f"CA-only model expects coordinates shape [B, L, 3], got {X.shape}")
        else:
            # Full backbone models expect [B, L, 4, 3] coordinates  
            if len(X.shape) != 4 or X.shape[-2:] != (4, 3):
                raise ValueError(f"Full backbone model expects coordinates shape [B, L, 4, 3], got {X.shape}")
        
        # Build graph representation
        E, E_idx = self.graph_builder(X, mask, residue_idx, chain_encoding_all)
        
        # Initialize node and edge features
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)
        
        # Create attention mask for encoder (unmasked self-attention)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # Run through encoder layers with optional gradient checkpointing
        if self.gradient_checkpointing and self.training and self._should_use_checkpointing():
            # Use gradient checkpointing to save memory during training
            for layer in self.encoder_layers:
                h_V, h_E = self._checkpoint_encoder_layer(layer, h_V, h_E, E_idx, mask, mask_attend)
        else:
            # Standard forward pass (frozen layers, inference, or checkpointing disabled)
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        return h_V
    
    def _should_use_checkpointing(self) -> bool:
        """
        Determine whether gradient checkpointing should be used.
        
        Checkpointing is beneficial when:
        1. At least some layers have gradients enabled (not completely frozen)
        2. Model is in training mode
        3. Gradient checkpointing is explicitly enabled
        
        Returns:
            True if checkpointing should be used
        """
        if self.freeze_layers:
            # If all layers are frozen, no point in checkpointing
            return False
        
        # Check if any encoder layer parameters require gradients
        any_gradients_enabled = any(
            param.requires_grad 
            for layer in self.encoder_layers 
            for param in layer.parameters()
        )
        
        return any_gradients_enabled
    
    def _checkpoint_encoder_layer(self, layer, h_V, h_E, E_idx, mask, mask_attend):
        """
        Apply gradient checkpointing to an encoder layer with proper error handling.
        
        This method wraps the encoder layer forward pass with gradient checkpointing
        to trade computation for memory usage during large protein training.
        
        Args:
            layer: ProteinMPNN encoder layer
            h_V: Node features [B, L, hidden_dim]
            h_E: Edge features [B, L, K, edge_dim]
            E_idx: Edge indices [B, L, K]
            mask: Sequence mask [B, L]
            mask_attend: Attention mask [B, L, K]
        
        Returns:
            Tuple of updated (h_V, h_E)
        """
        # Ensure all tensors are on the same device
        device = h_V.device
        E_idx = E_idx.to(device)
        mask = mask.to(device)
        mask_attend = mask_attend.to(device)
        
        # Create a wrapper function that explicitly passes all arguments
        # This avoids variable capture issues and ensures proper device handling
        def layer_forward_wrapper(h_V_input, h_E_input, E_idx_input, mask_input, mask_attend_input):
            return layer(h_V_input, h_E_input, E_idx_input, mask_input, mask_attend_input)
        
        try:
            # Use gradient checkpointing with explicit argument passing
            return checkpoint.checkpoint(
                layer_forward_wrapper, 
                h_V, h_E, E_idx, mask, mask_attend,
                use_reentrant=False
            )
        except RuntimeError as e:
            # Fallback to standard forward pass if checkpointing fails
            import warnings
            warnings.warn(f"Gradient checkpointing failed, falling back to standard forward: {e}")
            return layer(h_V, h_E, E_idx, mask, mask_attend)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing (use standard forward pass)."""
        self.gradient_checkpointing = False
    
    def get_embedding_dim(self) -> int:
        """Return the dimension of the output embeddings"""
        return self.hidden_dim
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: str = "v_48_020", 
        model_type: str = "vanilla",
        freeze_layers: bool = True,
        **kwargs
    ):
        """
        Load a pre-trained ProteinMPNN encoder from standard model weights
        
        Args:
            model_name: Model version (e.g., "v_48_020")
            model_type: Model type ("vanilla", "ca_model", "soluble")
            freeze_layers: Whether to freeze the loaded parameters
            **kwargs: Additional arguments for the encoder
        """
        # Construct path to model weights
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'proteinmpnn')
        
        model_dirs = {
            "vanilla": "vanilla_model_weights",
            "ca_model": "ca_model_weights", 
            "soluble": "soluble_model_weights"
        }
        
        if model_type not in model_dirs:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_dirs.keys())}")
        
        checkpoint_path = os.path.join(base_path, model_dirs[model_type], f"{model_name}.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Set ca_only flag for CA models
        ca_only = (model_type == "ca_model")
        
        return cls(
            pretrained_ckpt_path=checkpoint_path,
            freeze_layers=freeze_layers,
            ca_only=ca_only,
            **kwargs
        )


def load_pretrained_encoder(
    model_name: str = "v_48_020",
    model_type: str = "vanilla", 
    freeze_layers: bool = True,
    device: str = "cpu"
) -> ProteinMPNNBackboneEncoder:
    """
    Convenience function to load a pre-trained ProteinMPNN encoder
    
    Args:
        model_name: Model version to load
        model_type: Type of model (vanilla, ca_model, soluble)
        freeze_layers: Whether to freeze parameters
        device: Device to load model on
        
    Returns:
        Loaded ProteinMPNN encoder
    """
    encoder = ProteinMPNNBackboneEncoder.from_pretrained(
        model_name=model_name,
        model_type=model_type,
        freeze_layers=freeze_layers
    )
    
    return encoder.to(device)


if __name__ == "__main__":
    # Example usage
    print("Loading ProteinMPNN encoder...")
    
    try:
        # Load encoder with default settings
        encoder = load_pretrained_encoder()
        print(f"✓ Encoder loaded successfully")
        print(f"✓ Output embedding dimension: {encoder.get_embedding_dim()}")
        print(f"✓ Frozen layers: {encoder.freeze_layers}")
        
        # Test with dummy data
        batch_size, seq_len = 2, 100
        dummy_batch = {
            'X': torch.randn(batch_size, seq_len, 4, 3),
            'mask': torch.ones(batch_size, seq_len),
            'residue_idx': torch.arange(seq_len).repeat(batch_size, 1),
            'chain_encoding_all': torch.zeros(batch_size, seq_len)
        }
        
        with torch.no_grad():
            features = encoder(dummy_batch)
            print(f"✓ Forward pass successful: {features.shape}")
        
        # Test gradient checkpointing
        print(f"✓ Gradient checkpointing enabled: {encoder.gradient_checkpointing}")
        encoder.enable_gradient_checkpointing()
        print(f"✓ Gradient checkpointing after enable: {encoder.gradient_checkpointing}")
        encoder.disable_gradient_checkpointing()
        print(f"✓ Gradient checkpointing after disable: {encoder.gradient_checkpointing}")
        
        # Test with gradient checkpointing (requires unfrozen model)
        encoder_checkpointed = load_pretrained_encoder(freeze_layers=False)
        encoder_checkpointed.enable_gradient_checkpointing()
        encoder_checkpointed.train()  # Set to training mode
        
        dummy_batch_grad = {k: v.clone().detach() for k, v in dummy_batch.items()}
        features_checkpointed = encoder_checkpointed(dummy_batch_grad)
        print(f"✓ Gradient checkpointing forward pass: {features_checkpointed.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")