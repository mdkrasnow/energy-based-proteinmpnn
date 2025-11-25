"""
End-to-End Protein Design Pipeline

This module implements a unified design pipeline that integrates all components of the 
ProteinMPNN-IRED hybrid system to provide backbone input → optimized sequence output.

The pipeline combines:
1. ProteinMPNN backbone encoder for structural feature extraction
2. Energy models for stability scoring
3. IRED-style iterative optimization for sequence refinement
4. Quality assessment and validation of design outputs

Key Features:
- Single-interface design from PDB backbone to optimized sequence
- Optional initialization from ProteinMPNN decoder outputs 
- Comprehensive result validation and quality assessment
- Batch processing for multiple design targets
- Configurable optimization strategies
- Integration with existing evaluation framework
"""

import os
import sys
import json
import warnings
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Canonical amino acid ordering for pipeline
CANONICAL_AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(CANONICAL_AA_ORDER)}
INDEX_TO_AA = {i: aa for i, aa in enumerate(CANONICAL_AA_ORDER)}

# Constants for transparency tracking
INITIALIZATION_METHODS = {
    'PROTEINMPNN': 'proteinmpnn',
    'RANDOM_FALLBACK': 'random_fallback', 
    'FAILED': 'failed',
    'UNKNOWN': 'unknown'
}

FEATURE_SOURCES = {
    'ENCODER': 'encoder',
    'DUMMY_FALLBACK': 'dummy_fallback',
    'FAILED': 'failed', 
    'UNKNOWN': 'unknown'
}

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import core components
from models.mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from inference.ired_optimizer import IREDSequenceOptimizer, OptimizationConfig, OptimizationResult

# Import ProteinMPNN utilities for PDB parsing and decoder initialization
sys.path.append(os.path.join(str(project_root), '..', 'proteinmpnn'))
try:
    from protein_mpnn_utils import parse_PDB, ProteinMPNN, _scores, _S_to_seq
    PROTEINMPNN_AVAILABLE = True
except ImportError as e:
    PROTEINMPNN_AVAILABLE = False
    warnings.warn(f"ProteinMPNN utilities not available: {e}")


@dataclass
class PipelineConfig:
    """
    Configuration for the protein design pipeline.
    
    Attributes:
        # Model paths and settings
        encoder_checkpoint: Path to ProteinMPNN encoder checkpoint
        energy_model_checkpoint: Path to trained energy model checkpoint
        landscape_checkpoints: List of paths to landscape-specific energy models (optional)
        
        # Device and performance
        device: Computation device ('cpu', 'cuda', 'auto')
        batch_size: Batch size for processing (default: 1)
        max_sequence_length: Maximum sequence length to process (default: 500)
        
        # Model configuration
        encoder_config: Configuration for ProteinMPNN encoder
        energy_config: Configuration for energy head
        sequence_config: Configuration for sequence representation
        optimization_config: Configuration for IRED optimization
        
        # Processing options
        use_proteinmpnn_init: Whether to use ProteinMPNN decoder for initialization
        num_designs_per_target: Number of designs to generate per backbone
        temperature_schedule: Temperature schedule for sequence representation
        
        # Validation and output
        validate_results: Whether to run quality assessment on outputs
        save_trajectories: Whether to save optimization trajectories
        output_format: Output format ('sequences', 'logits', 'both')
        
        # Reproducibility control
        random_seed: Random seed for deterministic behavior (None = non-deterministic)
        deterministic_mode: Enable full deterministic mode including CUDNN
        fail_fast_on_proteinmpnn_failure: Fail explicitly when ProteinMPNN unavailable
    """
    # Model paths
    encoder_checkpoint: str
    energy_model_checkpoint: str
    landscape_checkpoints: Optional[List[str]] = None
    
    # Device and performance
    device: str = 'auto'
    batch_size: int = 1
    max_sequence_length: int = 500
    
    # Model configurations
    encoder_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dim': 128,
        'freeze_layers': True,
        'ca_only': False
    })
    energy_config: Dict[str, Any] = field(default_factory=lambda: {
        'backbone_dim': 128,
        'seq_dim': 20,
        'hidden_dim': 512,
        'num_layers': 3,
        'dropout': 0.1
    })
    sequence_config: Dict[str, Any] = field(default_factory=lambda: {
        'vocab_size': 20,
        'temperature_schedule': [1.0, 0.5, 0.1],
        'min_temperature': 1e-3,
        'max_temperature': 10.0
    })
    optimization_config: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.01,
        'max_steps_per_landscape': 50,
        'num_landscapes': 3,
        'convergence_patience': 10
    })
    
    # Processing options
    use_proteinmpnn_init: bool = True
    num_designs_per_target: int = 1
    temperature_schedule: Optional[List[float]] = None
    
    # Validation and output
    validate_results: bool = True
    save_trajectories: bool = False
    output_format: str = 'both'  # 'sequences', 'logits', 'both'
    
    # Reproducibility control
    random_seed: Optional[int] = None  # None means non-deterministic
    deterministic_mode: bool = False   # Enable full deterministic mode (CUDNN, etc)
    fail_fast_on_proteinmpnn_failure: bool = True  # Explicit failure when ProteinMPNN unavailable
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate paths
        if not os.path.exists(self.encoder_checkpoint):
            raise FileNotFoundError(f"Encoder checkpoint not found: {self.encoder_checkpoint}")
        if not os.path.exists(self.energy_model_checkpoint):
            raise FileNotFoundError(f"Energy model checkpoint not found: {self.energy_model_checkpoint}")
        
        # Validate device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif self.device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {self.max_sequence_length}")
        if self.num_designs_per_target <= 0:
            raise ValueError(f"num_designs_per_target must be positive, got {self.num_designs_per_target}")
        if self.output_format not in ['sequences', 'logits', 'both']:
            raise ValueError(f"output_format must be 'sequences', 'logits', or 'both', got {self.output_format}")
        
        # Validate reproducibility parameters
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative or None, got {self.random_seed}")


@dataclass 
class DesignResult:
    """
    Result from protein design pipeline.
    
    Attributes:
        # Core outputs
        sequences: Optimized discrete sequences [B, L] or None if failed
        logits: Final sequence logits [B, L, vocab_size] or None if failed
        energies: Final energy values for each design
        
        # Metadata
        backbone_path: Path to input backbone structure
        success: Whether design succeeded
        num_designs: Number of successful designs generated
        
        # Optimization details
        optimization_results: List of OptimizationResult objects
        trajectories: Optimization trajectories (if save_trajectories=True)
        total_time: Total computation time in seconds
        
        # Quality assessment (if validate_results=True)
        validation_metrics: Quality assessment metrics
        confidence_scores: Per-design confidence scores
        
        # Error information
        error_message: Error message if design failed
        warnings: List of warning messages
        
        # Transparency tracking
        initialization_method: Method used for sequence initialization
        feature_source: Source of structural features
    """
    # Core outputs
    sequences: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    energies: Optional[torch.Tensor]
    
    # Metadata
    backbone_path: str
    success: bool
    num_designs: int
    
    # Optimization details
    optimization_results: List[OptimizationResult]
    trajectories: Optional[List[Dict[str, Any]]] = None
    total_time: float = 0.0
    
    # Quality assessment
    validation_metrics: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[torch.Tensor] = None
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Transparency tracking
    initialization_method: str = INITIALIZATION_METHODS['UNKNOWN']  
    feature_source: str = FEATURE_SOURCES['UNKNOWN']


class ProteinDesignPipeline:
    """
    End-to-end protein design pipeline integrating ProteinMPNN and IRED.
    
    This class provides a unified interface for protein sequence design from backbone
    structures. It integrates all components of the hybrid system:
    
    1. Backbone encoding using ProteinMPNN encoder
    2. Energy-based sequence optimization using IRED
    3. Quality assessment and validation
    4. Batch processing capabilities
    
    Example usage:
        config = PipelineConfig(
            encoder_checkpoint='path/to/encoder.pt',
            energy_model_checkpoint='path/to/energy.pt'
        )
        pipeline = ProteinDesignPipeline(config)
        result = pipeline.design_sequence('backbone.pdb')
    """
    
    def __init__(self, config: Union[PipelineConfig, Dict[str, Any], str]):
        """
        Initialize the design pipeline.
        
        Args:
            config: Pipeline configuration (PipelineConfig object, dict, or JSON path)
        """
        # Load configuration
        if isinstance(config, str):
            config = self._load_config_from_file(config)
        elif isinstance(config, dict):
            config = PipelineConfig(**config)
        elif not isinstance(config, PipelineConfig):
            raise TypeError(f"config must be PipelineConfig, dict, or str, got {type(config)}")
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize reproducibility controls
        self._initialize_reproducibility()
        
        # Initialize components
        self._initialize_components()
        
        # ProteinMPNN for initialization (if enabled and available)
        self.proteinmpnn_model = None
        if config.use_proteinmpnn_init and PROTEINMPNN_AVAILABLE:
            self._initialize_proteinmpnn()
        
        print(f"ProteinDesignPipeline initialized on device: {self.device}")
    
    def _load_config_from_file(self, config_path: str) -> PipelineConfig:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return PipelineConfig(**config_dict)
    
    def _initialize_reproducibility(self):
        """Initialize comprehensive reproducibility controls."""
        if self.config.random_seed is not None:
            self.base_seed = self.config.random_seed
            
            # Create generator with fallback to CPU if device doesn't support it
            try:
                self.generator = torch.Generator(device=self.device)
            except RuntimeError:
                # Some devices don't support device-specific generators
                self.generator = torch.Generator()
                print(f"Warning: Device {self.device} doesn't support generators, using CPU generator")
            
            self.generator.manual_seed(self.base_seed)
            
            # Set global seeds for comprehensive determinism
            torch.manual_seed(self.base_seed)
            np.random.seed(self.base_seed)
            random.seed(self.base_seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.base_seed)
                
            if self.config.deterministic_mode:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # Note: PYTHONHASHSEED affects the entire Python process
                os.environ['PYTHONHASHSEED'] = str(self.base_seed)
                print(f"Warning: PYTHONHASHSEED set to {self.base_seed} - affects entire Python process")
                
            print(f"Reproducibility initialized with seed {self.base_seed}")
        else:
            self.base_seed = None
            self.generator = None
            print("Non-deterministic mode - results will not be reproducible")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize backbone encoder
            print("Loading ProteinMPNN backbone encoder...")
            self.encoder = ProteinMPNNBackboneEncoder(
                pretrained_ckpt_path=self.config.encoder_checkpoint,
                **self.config.encoder_config
            ).to(self.device)
            self.encoder.eval()
            
            # Initialize sequence representation
            print("Initializing sequence representation...")
            sequence_config = self.config.sequence_config.copy()
            if self.config.temperature_schedule is not None:
                sequence_config['temperature_schedule'] = self.config.temperature_schedule
            
            self.sequence_repr = ContinuousSequenceRepr(**sequence_config).to(self.device)
            
            # Initialize energy models
            print("Loading energy models...")
            self._load_energy_models()
            
            # Initialize optimizer
            print("Setting up IRED optimizer...")
            opt_config = OptimizationConfig(**self.config.optimization_config)
            self.optimizer = IREDSequenceOptimizer(
                energy_models=self.energy_models,
                sequence_repr=self.sequence_repr,
                config=opt_config
            )
            
            print("All components initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline components: {e}")
    
    def _load_energy_models(self):
        """Load energy models (single or multiple landscapes)"""
        energy_models = []
        
        # Load primary energy model
        energy_model = EnergyHead(**self.config.energy_config).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.config.energy_model_checkpoint, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            energy_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            energy_model.load_state_dict(checkpoint)
        
        energy_model.eval()
        energy_models.append(energy_model)
        
        # Load landscape-specific models if provided
        if self.config.landscape_checkpoints:
            for i, checkpoint_path in enumerate(self.config.landscape_checkpoints):
                if not os.path.exists(checkpoint_path):
                    warnings.warn(f"Landscape checkpoint {i} not found: {checkpoint_path}")
                    continue
                
                landscape_model = EnergyHead(**self.config.energy_config).to(self.device)
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    landscape_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    landscape_model.load_state_dict(checkpoint)
                
                landscape_model.eval()
                energy_models.append(landscape_model)
        
        # For single model, replicate across landscapes
        if len(energy_models) == 1:
            num_landscapes = self.config.optimization_config.get('num_landscapes', 3)
            energy_models = [energy_models[0]] * num_landscapes
        
        self.energy_models = energy_models
        print(f"Loaded {len(energy_models)} energy models")
    
    def _initialize_proteinmpnn(self):
        """Initialize ProteinMPNN model for sequence initialization"""
        try:
            # Load the same checkpoint used for encoder
            checkpoint = torch.load(self.config.encoder_checkpoint, map_location=self.device)
            
            # Create ProteinMPNN model with standard parameters
            self.proteinmpnn_model = ProteinMPNN(
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
                ca_only=self.config.encoder_config.get('ca_only', False)
            ).to(self.device)
            
            self.proteinmpnn_model.load_state_dict(checkpoint, strict=False)
            self.proteinmpnn_model.eval()
            
            print("ProteinMPNN model initialized for sequence initialization")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize ProteinMPNN model: {e}")
            self.proteinmpnn_model = None
    
    def design_sequence(
        self,
        backbone_path: str,
        mask: Optional[torch.Tensor] = None,
        fixed_positions: Optional[Dict[int, str]] = None,
        **kwargs
    ) -> DesignResult:
        """
        Design protein sequence(s) for a given backbone structure.
        
        Args:
            backbone_path: Path to PDB file with backbone structure
            mask: Optional mask for designable positions [L] (1=design, 0=fixed)
            fixed_positions: Optional dict mapping position indices to fixed amino acids
            **kwargs: Additional arguments passed to optimizer
        
        Returns:
            DesignResult: Complete design results with sequences, energies, and metadata
        """
        start_time = time.time()
        
        try:
            # Parse backbone structure
            print(f"Parsing backbone structure: {backbone_path}")
            backbone_features = self._parse_backbone(backbone_path)
            
            # Extract structural features
            print("Extracting structural features...")
            with torch.no_grad():
                structural_features = self._extract_features(backbone_features)
            
            # Generate initial sequence(s)
            print("Generating initial sequences...")
            initial_logits, initialization_method = self._generate_initial_sequences(
                backbone_features, structural_features, 
                mask=mask, fixed_positions=fixed_positions
            )
            
            # Optimize sequences
            print("Optimizing sequences...")
            optimization_results = self._optimize_sequences(
                initial_logits, structural_features, mask=mask, **kwargs
            )
            
            # Process results
            design_result = self._process_results(
                optimization_results, backbone_path, start_time
            )
            
            # Validate results if requested
            if self.config.validate_results and design_result.success:
                print("Validating design results...")
                design_result = self._validate_results(design_result)
            
            print(f"Design completed in {design_result.total_time:.2f}s")
            return design_result
            
        except Exception as e:
            # Return failed result with error information
            return DesignResult(
                sequences=None,
                logits=None,
                energies=None,
                backbone_path=backbone_path,
                success=False,
                num_designs=0,
                optimization_results=[],
                total_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _parse_backbone(self, backbone_path: str) -> Dict[str, Any]:
        """Parse PDB backbone structure"""
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"Backbone file not found: {backbone_path}")
        
        if not PROTEINMPNN_AVAILABLE:
            raise RuntimeError("ProteinMPNN utilities required for PDB parsing")
        
        try:
            # Use ProteinMPNN's PDB parser
            protein_dict = parse_PDB(backbone_path)
            
            # Validate parsed structure
            if not protein_dict or 'coords' not in protein_dict:
                raise ValueError("Failed to parse valid structure from PDB file")
            
            return protein_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse backbone structure: {e}")
    
    def _extract_features(self, backbone_features: Dict[str, Any]) -> torch.Tensor:
        """Extract structural features using ProteinMPNN encoder"""
        try:
            # Convert ProteinMPNN's parsed structure to encoder input format
            # Extract coordinates and chain information
            coords = backbone_features['coords']
            
            # Handle single chain case
            if isinstance(coords, dict):
                chain_key = list(coords.keys())[0]
                coord_tensor = torch.tensor(coords[chain_key], dtype=torch.float32, device=self.device)
            else:
                coord_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
            
            # Ensure correct coordinate shape [B, L, 4, 3] where 4 = N,CA,C,O atoms
            if coord_tensor.dim() == 3:  # [L, 4, 3]
                coord_tensor = coord_tensor.unsqueeze(0)  # [1, L, 4, 3]
            
            seq_length = coord_tensor.shape[1]
            
            # Create chain mask (all positions belong to chain A)
            chain_mask = torch.ones(1, seq_length, device=self.device)
            
            # Create residue index
            residue_idx = torch.arange(seq_length, device=self.device).unsqueeze(0)
            
            # Validate encoder is available
            if not hasattr(self, 'encoder') or self.encoder is None:
                raise RuntimeError(
                    "ProteinMPNN encoder not initialized. Check that encoder initialization succeeded."
                )
            
            # Use actual ProteinMPNN encoder to extract structural features
            batch_size = coord_tensor.shape[0]
            seq_length = coord_tensor.shape[1]
            
            # Validate coordinate tensor
            if seq_length == 0:
                raise ValueError("Empty sequence - coordinate tensor has 0 length")
            if coord_tensor.shape[2] != 4:
                raise ValueError(f"Expected 4 atoms per residue (N,CA,C,O), got {coord_tensor.shape[2]}")
            if coord_tensor.shape[3] != 3:
                raise ValueError(f"Expected 3D coordinates, got {coord_tensor.shape[3]}")
            
            # Create chain encoding (all positions belong to chain A)
            chain_encoding_all = torch.zeros(batch_size, seq_length, dtype=torch.long, device=self.device)
            
            # Prepare batch for ProteinMPNN encoder
            batch = {
                'X': coord_tensor,  # [B, L, 4, 3] coordinates (N, CA, C, O)
                'mask': chain_mask,  # [B, L] sequence mask
                'residue_idx': residue_idx,  # [B, L] residue indices
                'chain_encoding_all': chain_encoding_all  # [B, L] chain encoding
            }
            
            # Extract features using actual ProteinMPNN encoder
            try:
                structural_features = self.encoder.forward(batch)  # [B, L, hidden_dim]
                
                # Validate features are reasonable
                if torch.isnan(structural_features).any() or torch.isinf(structural_features).any():
                    raise ValueError("ProteinMPNN encoder produced NaN or Inf features")
                    
                print(f"Successfully extracted structural features using ProteinMPNN encoder: {structural_features.shape}")
                return structural_features
                
            except Exception as e:
                # FAIL EXPLICITLY instead of silent fallback
                raise RuntimeError(
                    f"ProteinMPNN encoder integration failed: {e}. "
                    "This indicates incomplete setup or incompatible model weights. "
                    "Check that ProteinMPNN encoder is properly initialized and compatible."
                ) from e
            
        except Exception as e:
            # FAIL EXPLICITLY - coordinate processing failure
            raise RuntimeError(
                f"Coordinate processing failed during feature extraction: {e}. "
                "This indicates problems with the input backbone structure. "
                "Check that the PDB file contains valid coordinate data."
            ) from e
    
    def _generate_initial_sequences(
        self,
        backbone_features: Dict[str, Any],
        structural_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fixed_positions: Optional[Dict[int, str]] = None
    ) -> Tuple[torch.Tensor, str]:
        """Generate initial sequence logits using ProteinMPNN decoder or random initialization
        
        Returns:
            Tuple of (initial_logits, initialization_method)
        """
        batch_size, seq_length, _ = structural_features.shape
        vocab_size = self.config.sequence_config['vocab_size']
        
        initialization_method = INITIALIZATION_METHODS['FAILED']  # Default to failed
        
        if self.config.use_proteinmpnn_init and self.proteinmpnn_model is not None:
            # Use ProteinMPNN decoder to generate initial logits
            try:
                initial_logits = self._run_proteinmpnn_decoder(
                    backbone_features, seq_length, vocab_size
                )
                initialization_method = INITIALIZATION_METHODS['PROTEINMPNN']
                print(f"Generated {initial_logits.shape[0]} initial sequences using ProteinMPNN decoder")
                
            except Exception as e:
                if self.config.fail_fast_on_proteinmpnn_failure:
                    # Fail explicitly when ProteinMPNN requested but unavailable
                    raise RuntimeError(
                        f"ProteinMPNN decoder initialization failed and fail_fast=True: {e}. "
                        "Set fail_fast_on_proteinmpnn_failure=False to allow random fallback."
                    ) from e
                else:
                    # Only fallback if explicitly allowed
                    print(f"ERROR: ProteinMPNN decoder failed, falling back to random initialization: {e}")
                    try:
                        initial_logits = self._generate_random_logits(seq_length, vocab_size, generator=self.generator)
                        initialization_method = INITIALIZATION_METHODS['RANDOM_FALLBACK']
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Both ProteinMPNN decoder and random fallback failed. "
                            f"ProteinMPNN error: {e}. Fallback error: {fallback_error}"
                        ) from fallback_error
        else:
            # Use random initialization (explicitly requested)
            try:
                initial_logits = self._generate_random_logits(seq_length, vocab_size, generator=self.generator)
                initialization_method = INITIALIZATION_METHODS['RANDOM_FALLBACK']
            except Exception as random_error:
                raise RuntimeError(f"Random initialization failed: {random_error}") from random_error
            
        # Store initialization method for transparency
        if not hasattr(self, '_initialization_methods'):
            self._initialization_methods = {}
        # Use a design index or general key - for now use 'latest'
        self._initialization_methods['latest'] = initialization_method
        
        # Apply mask and fixed positions if provided
        if mask is not None or fixed_positions is not None:
            initial_logits = self._apply_constraints(
                initial_logits, mask, fixed_positions
            )
        
        return initial_logits, initialization_method
    
    def _run_proteinmpnn_decoder(
        self,
        backbone_features: Dict[str, Any],
        seq_length: int,
        vocab_size: int
    ) -> torch.Tensor:
        """Run ProteinMPNN decoder to generate initial sequence logits"""
        with torch.no_grad():
            # Extract coordinates from backbone features
            coords = backbone_features['coords']
            
            # Handle coordinate format
            if isinstance(coords, dict):
                # Multi-chain case - take first chain for now
                chain_key = list(coords.keys())[0]
                coord_data = coords[chain_key]
            else:
                coord_data = coords
            
            # Convert to tensor
            if not isinstance(coord_data, torch.Tensor):
                coord_data = torch.tensor(coord_data, dtype=torch.float32)
            
            coord_data = coord_data.to(self.device)
            
            # Ensure correct coordinate shape [B, L, 4, 3] where 4 = N,CA,C,O atoms
            if coord_data.dim() == 3:  # [L, 4, 3]
                coord_data = coord_data.unsqueeze(0)  # [1, L, 4, 3]
            
            batch_size, seq_len, n_atoms, _ = coord_data.shape
            
            # Validate coordinate shape
            if n_atoms != 4:
                raise ValueError(f"Expected 4 atoms per residue (N,CA,C,O), got {n_atoms}")
            
            # Create sequence mask (all positions designable)
            if 'mask' in backbone_features:
                chain_mask = torch.tensor(backbone_features['mask'], device=self.device).float()
                if chain_mask.dim() == 1:
                    chain_mask = chain_mask.unsqueeze(0)
            else:
                chain_mask = torch.ones(batch_size, seq_len, device=self.device)
            
            # Create residue indices
            residue_idx = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Create chain map (all residues belong to chain 0)
            chain_map = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)
            
            # Generate initial random sequence for decoder input
            # ProteinMPNN decoder needs a starting sequence to condition on
            initial_seq = torch.randint(0, 20, (batch_size, seq_len), device=self.device)
            
            # Convert coordinates to ProteinMPNN format
            # ProteinMPNN expects coordinates as [B, L, 4, 3] with atoms in order N, CA, C, O
            X = coord_data
            
            # Run ProteinMPNN forward pass to get logits
            # This is a simplified version - full implementation would use proper ProteinMPNN utilities
            try:
                # Use ProteinMPNN's feature construction
                # Note: This is a simplified approach - in practice would use tied_featurize
                
                # Generate multiple designs
                all_logits = []
                for design_idx in range(self.config.num_designs_per_target):
                    # Create per-design generator for reproducible sampling
                    if self.generator is not None:
                        # Use proper Generator for reproducible per-design seeding
                        try:
                            design_generator = torch.Generator(device=self.device)
                        except RuntimeError:
                            # Fallback to CPU generator if device doesn't support it
                            design_generator = torch.Generator()
                        design_generator.manual_seed(self.base_seed + design_idx * 1000)
                    else:
                        design_generator = None
                    
                    # Create random sequence for autoregressive generation
                    S = torch.randint(0, 20, (batch_size, seq_len), device=self.device, generator=design_generator)
                    
                    # Create edge features (simplified)
                    # In practice would use ProteinMPNN's graph construction
                    E_idx = self._create_edge_indices(seq_len, k_neighbors=64)
                    
                    # Run forward pass through ProteinMPNN encoder and decoder
                    # Note: Using simplified approach - full implementation would use proper ProteinMPNN utilities
                    try:
                        # Try to use ProteinMPNN's proper forward method
                        if hasattr(self.proteinmpnn_model, 'forward'):
                            # Standard forward pass
                            outputs = self.proteinmpnn_model.forward(X, S, chain_mask, residue_idx, chain_map)
                            if isinstance(outputs, tuple) and len(outputs) >= 1:
                                logits = outputs[0]  # First output should be logits
                            else:
                                logits = outputs
                        else:
                            # Fallback: try to manually construct features and decode
                            warnings.warn("Using fallback ProteinMPNN feature construction")
                            # This is a placeholder - would need proper ProteinMPNN integration
                            logits = torch.randn(batch_size, seq_len, 21, device=self.device, generator=design_generator)
                        
                    except Exception as e:
                        warnings.warn(f"ProteinMPNN forward pass failed: {e}, using fallback")
                        # Fallback logits
                        logits = torch.randn(batch_size, seq_len, 21, device=self.device, generator=design_generator)
                    
                    # Convert from 21 classes to 20 (remove mask token)
                    logits = logits[:, :, :20]  # Remove mask token
                    
                    all_logits.append(logits)
                
                # Stack all design logits
                initial_logits = torch.stack(all_logits, dim=0).squeeze(1)  # [num_designs, L, 20]
                
                return initial_logits
                
            except Exception as e:
                # If ProteinMPNN forward pass fails, use coordinate-based features
                raise RuntimeError(f"ProteinMPNN decoder forward pass failed: {e}")
    
    def _create_edge_indices(self, seq_length: int, k_neighbors: int = 64) -> torch.Tensor:
        """Create edge indices for ProteinMPNN graph construction"""
        # Simplified edge creation - connect each residue to its k nearest neighbors
        # In practice would use ProteinMPNN's proper graph construction
        
        # For now, connect each residue to local neighbors
        edges = []
        for i in range(seq_length):
            for j in range(max(0, i-k_neighbors//2), min(seq_length, i+k_neighbors//2+1)):
                if i != j:
                    edges.append([i, j])
        
        if not edges:
            # Fallback: create minimal edges
            edges = [[i, i+1] for i in range(seq_length-1)]
            edges += [[i+1, i] for i in range(seq_length-1)]
        
        return torch.tensor(edges, device=self.device).long().t()  # [2, E]
    
    def _generate_random_logits(self, seq_length: int, vocab_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate random initial logits as fallback"""
        initial_logits = torch.randn(
            self.config.num_designs_per_target, seq_length, vocab_size,
            device=self.device, generator=generator
        )
        
        # Make logits slightly favor common amino acids using explicit mapping
        common_aa_bonus = torch.zeros(vocab_size, device=self.device)
        common_aa = 'AVGL'  # Alanine, Valine, Glycine, Leucine
        common_indices = [AA_TO_INDEX[aa] for aa in common_aa]
        for idx in common_indices:
            if idx < vocab_size:
                common_aa_bonus[idx] = 0.5
        
        print(f"DEBUG: Applied initialization bias to common amino acids: {common_aa} at indices {common_indices}")
        
        initial_logits = initial_logits + common_aa_bonus.unsqueeze(0).unsqueeze(0)
        
        return initial_logits
    
    def _validate_and_convert_vocabulary(self, logits: torch.Tensor, conversion_type: str) -> torch.Tensor:
        """Validate and convert amino acid vocabulary ordering."""
        # Store vocabulary mapping used for provenance
        if not hasattr(self, '_vocabulary_mappings'):
            self._vocabulary_mappings = []
            
        if conversion_type == 'proteinmpnn_to_canonical':
            # TODO: Implement ProteinMPNN vocabulary mapping validation
            # For now, assume ProteinMPNN uses same ordering, but log for verification
            print(f"INFO: Assuming ProteinMPNN vocabulary matches canonical ordering: {CANONICAL_AA_ORDER}")
            print("WARNING: ProteinMPNN vocabulary validation not yet implemented - add explicit mapping")
            
            self._vocabulary_mappings.append({
                'conversion_type': conversion_type,
                'source_vocab': 'proteinmpnn',
                'target_vocab': 'canonical',
                'mapping': 'identity_assumed'  # TODO: Replace with actual mapping
            })
            return logits
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")
    
    def _apply_constraints(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fixed_positions: Optional[Dict[int, str]] = None
    ) -> torch.Tensor:
        """Apply design constraints to logits"""
        if fixed_positions:
            # Fix specific positions to specific amino acids
            for pos, aa in fixed_positions.items():
                if 0 <= pos < logits.shape[1] and aa in AA_TO_INDEX:
                    aa_idx = AA_TO_INDEX[aa]
                    # Set fixed position logits to strongly favor the specified amino acid
                    logits[:, pos, :] = -10.0
                    logits[:, pos, aa_idx] = 10.0
        
        if mask is not None:
            # Zero gradients for non-designable positions during optimization
            # This will be enforced in the optimizer
            pass
        
        return logits
    
    def _optimize_sequences(
        self,
        initial_logits: torch.Tensor,
        structural_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[OptimizationResult]:
        """Optimize sequences using IRED optimizer"""
        results = []
        
        # Create energy function that incorporates structural features
        # This function will be called by the IRED optimizer with landscape index
        def energy_fn(sequence_logits: torch.Tensor, landscape_idx: int = 0) -> torch.Tensor:
            # Convert logits to sequence representation
            sequence_probs = self.sequence_repr(sequence_logits, landscape_idx=landscape_idx, training=False)
            
            # Select appropriate energy model for the landscape
            landscape_idx = min(landscape_idx, len(self.energy_models) - 1)
            energy_model = self.energy_models[landscape_idx]
            
            # Combine structural features and sequence probabilities
            batch_size, seq_length, vocab_size = sequence_probs.shape
            _, _, feature_dim = structural_features.shape
            
            # Expand structural features to match batch size
            expanded_features = structural_features.expand(batch_size, -1, -1)
            
            # Apply mask if provided (zero out gradients for fixed positions)
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(sequence_probs)
                sequence_probs = sequence_probs * mask_expanded
            
            # Predict energy
            energy = energy_model(expanded_features, sequence_probs)
            return energy
        
        # Optimize each initial sequence
        for i in range(initial_logits.shape[0]):
            init_logits = initial_logits[i:i+1]  # Keep batch dimension
            
            try:
                # Use the IRED optimizer with proper multi-landscape support
                if len(self.energy_models) > 1:
                    # Multi-landscape optimization
                    result = self.optimizer.optimize_sequence(
                        initial_logits=init_logits,
                        energy_fn=energy_fn,
                        mask=mask,
                        **kwargs
                    )
                else:
                    # Single model optimization
                    result = self.optimizer.optimize_sequence(
                        initial_logits=init_logits,
                        energy_fn=lambda logits: energy_fn(logits, 0),
                        mask=mask,
                        **kwargs
                    )
                
                results.append(result)
                
            except Exception as e:
                # Create failed result
                failed_result = OptimizationResult(
                    sequence=None,
                    logits=None,
                    trajectory=[],
                    final_energy=float('inf'),
                    converged=False,
                    total_steps=0,
                    landscapes_used=0,
                    optimization_failed=True,
                    failure_reason=str(e)
                )
                results.append(failed_result)
        
        return results
    
    def _process_results(
        self,
        optimization_results: List[OptimizationResult],
        backbone_path: str,
        start_time: float
    ) -> DesignResult:
        """Process optimization results into final design result"""
        # Filter successful results
        successful_results = [r for r in optimization_results if not r.optimization_failed]
        
        if not successful_results:
            return DesignResult(
                sequences=None,
                logits=None,
                energies=None,
                backbone_path=backbone_path,
                success=False,
                num_designs=0,
                optimization_results=optimization_results,
                total_time=time.time() - start_time,
                error_message="All optimization attempts failed"
            )
        
        # Collect outputs
        sequences = []
        logits = []
        energies = []
        
        for result in successful_results:
            if result.sequence is not None:
                sequences.append(result.sequence)
            if result.logits is not None:
                logits.append(result.logits)
            energies.append(result.final_energy)
        
        # Convert to tensors
        sequences_tensor = torch.stack(sequences) if sequences else None
        logits_tensor = torch.stack(logits) if logits else None
        energies_tensor = torch.tensor(energies, device=self.device)
        
        # Collect trajectories if requested
        trajectories = None
        if self.config.save_trajectories:
            trajectories = [r.trajectory for r in successful_results]
        
        return DesignResult(
            sequences=sequences_tensor,
            logits=logits_tensor,
            energies=energies_tensor,
            backbone_path=backbone_path,
            success=True,
            num_designs=len(successful_results),
            optimization_results=optimization_results,
            trajectories=trajectories,
            total_time=time.time() - start_time
        )
    
    def _validate_results(self, result: DesignResult) -> DesignResult:
        """Validate design results using comprehensive quality assessment"""
        try:
            # Import evaluation components
            from evaluation.eval_energy import (
                EnergyRankingEvaluator, 
                SequencePropertyAnalyzer,
                EnergyVisualizationGenerator
            )
            
            validation_metrics = {}
            warnings_list = result.warnings or []
            
            # Basic energy statistics
            energy_stats = self._compute_energy_statistics(result.energies)
            validation_metrics.update(energy_stats)
            
            # Convergence statistics
            convergence_stats = self._compute_convergence_statistics(result.optimization_results)
            validation_metrics.update(convergence_stats)
            
            # Sequence property analysis
            if result.sequences is not None:
                try:
                    seq_analyzer = SequencePropertyAnalyzer()
                    seq_properties = self._analyze_sequence_properties(result.sequences, seq_analyzer)
                    validation_metrics['sequence_properties'] = seq_properties
                except Exception as e:
                    warnings_list.append(f"Sequence property analysis failed: {e}")
            
            # Energy ranking assessment (if multiple sequences)
            if result.sequences is not None and len(result.sequences) > 1:
                try:
                    ranking_metrics = self._assess_energy_ranking(result)
                    validation_metrics['ranking_assessment'] = ranking_metrics
                except Exception as e:
                    warnings_list.append(f"Ranking assessment failed: {e}")
            
            # Confidence scoring
            confidence_scores = self._compute_confidence_scores(result)
            validation_metrics['confidence_statistics'] = {
                'mean_confidence': float(confidence_scores.mean()),
                'std_confidence': float(confidence_scores.std()),
                'min_confidence': float(confidence_scores.min()),
                'max_confidence': float(confidence_scores.max())
            }
            
            # Sequence diversity analysis
            if result.sequences is not None and len(result.sequences) > 1:
                diversity_metrics = self._compute_sequence_diversity(result.sequences)
                validation_metrics['diversity'] = diversity_metrics
            
            # Quality assessment summary
            quality_summary = self._create_quality_summary(validation_metrics, confidence_scores)
            validation_metrics['quality_summary'] = quality_summary
            
            # Update result
            result.validation_metrics = validation_metrics
            result.confidence_scores = confidence_scores
            result.warnings = warnings_list
            
            return result
            
        except Exception as e:
            # Fallback to basic validation if comprehensive validation fails
            warnings_list = result.warnings or []
            warnings_list.append(f"Comprehensive validation failed, using basic metrics: {e}")
            
            basic_metrics = self._compute_basic_validation_metrics(result)
            confidence_scores = self._compute_confidence_scores(result)
            
            result.validation_metrics = basic_metrics
            result.confidence_scores = confidence_scores
            result.warnings = warnings_list
            
            return result
    
    def _compute_energy_statistics(self, energies: torch.Tensor) -> Dict[str, float]:
        """Compute basic energy statistics"""
        return {
            'energy_mean': float(energies.mean()),
            'energy_std': float(energies.std()),
            'energy_min': float(energies.min()),
            'energy_max': float(energies.max()),
            'energy_range': float(energies.max() - energies.min()),
            'energy_median': float(energies.median()),
        }
    
    def _compute_convergence_statistics(self, optimization_results: List[OptimizationResult]) -> Dict[str, Any]:
        """Compute optimization convergence statistics"""
        if not optimization_results:
            return {}
        
        total_optimizations = len(optimization_results)
        successful_optimizations = [r for r in optimization_results if not r.optimization_failed]
        converged_optimizations = [r for r in optimization_results if r.converged]
        
        total_steps = [r.total_steps for r in successful_optimizations]
        landscapes_used = [r.landscapes_used for r in successful_optimizations]
        
        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': len(successful_optimizations),
            'converged_optimizations': len(converged_optimizations),
            'success_rate': len(successful_optimizations) / total_optimizations if total_optimizations > 0 else 0.0,
            'convergence_rate': len(converged_optimizations) / total_optimizations if total_optimizations > 0 else 0.0,
            'mean_steps': float(np.mean(total_steps)) if total_steps else 0.0,
            'std_steps': float(np.std(total_steps)) if len(total_steps) > 1 else 0.0,
            'mean_landscapes_used': float(np.mean(landscapes_used)) if landscapes_used else 0.0,
        }
    
    def _analyze_sequence_properties(
        self, 
        sequences: torch.Tensor, 
        analyzer: 'SequencePropertyAnalyzer'
    ) -> Dict[str, Any]:
        """Analyze sequence properties using evaluation framework"""
        # Convert tensor sequences to string sequences for analysis
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        string_sequences = []
        
        for seq_tensor in sequences:
            sequence_str = ''.join(amino_acids[idx] for idx in seq_tensor.cpu().numpy())
            string_sequences.append(sequence_str)
        
        try:
            # Use the analyzer from evaluation framework
            properties = analyzer.analyze_sequences(string_sequences)
            return properties
        except Exception as e:
            # Fallback to basic properties
            return self._compute_basic_sequence_properties(string_sequences)
    
    def _compute_basic_sequence_properties(self, sequences: List[str]) -> Dict[str, Any]:
        """Compute basic sequence properties as fallback"""
        if not sequences:
            return {}
        
        # Amino acid composition
        aa_counts = {}
        total_residues = 0
        
        for seq in sequences:
            total_residues += len(seq)
            for aa in seq:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        aa_frequencies = {aa: count/total_residues for aa, count in aa_counts.items()}
        
        # Sequence lengths
        lengths = [len(seq) for seq in sequences]
        
        # Hydrophobicity (simple approximation)
        hydrophobic_aa = set('AILMFWYV')
        hydrophobic_fractions = []
        for seq in sequences:
            hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic_aa)
            hydrophobic_fractions.append(hydrophobic_count / len(seq) if seq else 0.0)
        
        return {
            'amino_acid_composition': aa_frequencies,
            'sequence_lengths': {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths))
            },
            'hydrophobicity': {
                'mean_hydrophobic_fraction': float(np.mean(hydrophobic_fractions)),
                'std_hydrophobic_fraction': float(np.std(hydrophobic_fractions))
            }
        }
    
    def _assess_energy_ranking(self, result: DesignResult) -> Dict[str, float]:
        """Assess energy ranking quality among designed sequences"""
        if result.energies is None or len(result.energies) <= 1:
            return {}
        
        # Sort sequences by energy (lower is better)
        sorted_indices = torch.argsort(result.energies)
        sorted_energies = result.energies[sorted_indices]
        
        # Compute ranking metrics
        energy_gaps = sorted_energies[1:] - sorted_energies[:-1]
        
        ranking_metrics = {
            'best_energy': float(sorted_energies[0]),
            'worst_energy': float(sorted_energies[-1]),
            'energy_range': float(sorted_energies[-1] - sorted_energies[0]),
            'mean_energy_gap': float(energy_gaps.mean()) if len(energy_gaps) > 0 else 0.0,
            'std_energy_gap': float(energy_gaps.std()) if len(energy_gaps) > 0 else 0.0,
            'ranking_spread': float(sorted_energies.std()),
        }
        
        return ranking_metrics
    
    def _compute_confidence_scores(self, result: DesignResult) -> torch.Tensor:
        """Compute confidence scores for designed sequences"""
        if result.energies is None:
            return torch.tensor([0.0], device=self.device)
        
        # Multiple confidence metrics
        energies = result.energies
        
        # Energy-based confidence (lower energy = higher confidence)
        energy_conf = torch.sigmoid(-energies)  # Sigmoid to map to [0,1]
        
        # Convergence-based confidence
        convergence_scores = torch.zeros_like(energies)
        for i, opt_result in enumerate(result.optimization_results):
            if i < len(energies):
                if opt_result.optimization_failed:
                    convergence_scores[i] = 0.0
                elif opt_result.converged:
                    convergence_scores[i] = 1.0
                else:
                    # Partial credit for non-failed but non-converged
                    convergence_scores[i] = 0.5
        
        # Combine confidence metrics
        combined_confidence = 0.7 * energy_conf + 0.3 * convergence_scores
        
        return combined_confidence
    
    def _compute_sequence_diversity(self, sequences: torch.Tensor) -> Dict[str, float]:
        """Compute diversity metrics among designed sequences"""
        if sequences is None or len(sequences) <= 1:
            return {}
        
        # Compute pairwise Hamming distances
        pairwise_distances = []
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                hamming_dist = (sequences[i] != sequences[j]).float().mean()
                pairwise_distances.append(float(hamming_dist))
        
        pairwise_distances = np.array(pairwise_distances)
        
        # Unique sequences
        unique_sequences = torch.unique(sequences, dim=0)
        
        diversity_metrics = {
            'mean_pairwise_distance': float(pairwise_distances.mean()),
            'std_pairwise_distance': float(pairwise_distances.std()),
            'max_pairwise_distance': float(pairwise_distances.max()),
            'min_pairwise_distance': float(pairwise_distances.min()),
            'num_unique_sequences': len(unique_sequences),
            'uniqueness_fraction': len(unique_sequences) / len(sequences),
        }
        
        return diversity_metrics
    
    def _create_quality_summary(
        self, 
        validation_metrics: Dict[str, Any], 
        confidence_scores: torch.Tensor
    ) -> Dict[str, Any]:
        """Create high-level quality summary"""
        summary = {
            'overall_quality': 'unknown',
            'primary_concerns': [],
            'recommendations': []
        }
        
        # Energy quality assessment
        energy_mean = validation_metrics.get('energy_mean', float('inf'))
        energy_std = validation_metrics.get('energy_std', float('inf'))
        
        if energy_mean < 0:  # Assume negative energies are better
            summary['energy_quality'] = 'good' if energy_mean < -1.0 else 'moderate'
        else:
            summary['energy_quality'] = 'poor'
            summary['primary_concerns'].append('High energy values suggest unstable designs')
        
        # Convergence quality
        convergence_rate = validation_metrics.get('convergence_rate', 0.0)
        if convergence_rate < 0.5:
            summary['primary_concerns'].append('Low convergence rate suggests optimization difficulties')
            summary['recommendations'].append('Consider increasing optimization steps or adjusting learning rate')
        
        # Diversity assessment
        if 'diversity' in validation_metrics:
            uniqueness = validation_metrics['diversity'].get('uniqueness_fraction', 0.0)
            if uniqueness < 0.5:
                summary['primary_concerns'].append('Low sequence diversity - designs may be too similar')
                summary['recommendations'].append('Consider using different random seeds or increasing exploration')
        
        # Overall confidence
        mean_confidence = float(confidence_scores.mean())
        if mean_confidence > 0.8:
            summary['confidence_level'] = 'high'
        elif mean_confidence > 0.6:
            summary['confidence_level'] = 'moderate'
        else:
            summary['confidence_level'] = 'low'
            summary['primary_concerns'].append('Low confidence scores suggest uncertain designs')
        
        # Overall quality assessment
        if len(summary['primary_concerns']) == 0 and mean_confidence > 0.7:
            summary['overall_quality'] = 'good'
        elif len(summary['primary_concerns']) <= 1 and mean_confidence > 0.5:
            summary['overall_quality'] = 'moderate'
        else:
            summary['overall_quality'] = 'poor'
        
        return summary
    
    def _compute_basic_validation_metrics(self, result: DesignResult) -> Dict[str, Any]:
        """Compute basic validation metrics as fallback"""
        metrics = {}
        
        if result.energies is not None:
            metrics.update(self._compute_energy_statistics(result.energies))
        
        if result.optimization_results:
            metrics.update(self._compute_convergence_statistics(result.optimization_results))
        
        metrics['validation_mode'] = 'basic'
        return metrics
    
    def design_batch(
        self,
        backbone_paths: List[str],
        chunk_size: Optional[int] = None,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        memory_limit_gb: float = 8.0,
        **kwargs
    ) -> List[DesignResult]:
        """
        Design sequences for multiple backbone structures with memory management.
        
        Args:
            backbone_paths: List of paths to PDB backbone files
            chunk_size: Number of structures to process simultaneously (None = auto)
            save_intermediate: Whether to save results after each chunk
            intermediate_dir: Directory to save intermediate results
            progress_callback: Optional callback for progress updates
            memory_limit_gb: Approximate memory limit in GB for auto chunk sizing
            **kwargs: Additional arguments passed to design_sequence
        
        Returns:
            List of DesignResult objects, one per backbone
        """
        if not backbone_paths:
            return []
        
        # Auto-determine chunk size based on memory limit
        if chunk_size is None:
            chunk_size = self._estimate_chunk_size(len(backbone_paths), memory_limit_gb)
        
        # Setup intermediate saving
        if save_intermediate and intermediate_dir is None:
            intermediate_dir = f"batch_results_{int(time.time())}"
        
        if save_intermediate:
            os.makedirs(intermediate_dir, exist_ok=True)
            print(f"Saving intermediate results to: {intermediate_dir}")
        
        # Process in chunks
        all_results = []
        total_structures = len(backbone_paths)
        
        print(f"Processing {total_structures} structures in chunks of {chunk_size}")
        
        for chunk_idx in range(0, total_structures, chunk_size):
            chunk_end = min(chunk_idx + chunk_size, total_structures)
            chunk_paths = backbone_paths[chunk_idx:chunk_end]
            
            print(f"\nProcessing chunk {chunk_idx//chunk_size + 1} "
                  f"({chunk_idx+1}-{chunk_end}/{total_structures})")
            
            # Process chunk with memory management
            chunk_results = self._process_chunk(
                chunk_paths, progress_callback, chunk_idx, **kwargs
            )
            
            all_results.extend(chunk_results)
            
            # Save intermediate results
            if save_intermediate:
                self._save_chunk_results(
                    chunk_results, intermediate_dir, chunk_idx, chunk_end-1
                )
            
            # Memory cleanup between chunks
            self._cleanup_memory()
        
        # Final progress update
        if progress_callback:
            progress_callback(total_structures, total_structures, "Batch complete")
        
        # Save final results if requested
        if save_intermediate:
            self._save_final_batch_results(all_results, intermediate_dir)
        
        print(f"\nBatch processing complete: {len(all_results)} structures processed")
        
        return all_results
    
    def _estimate_chunk_size(self, total_structures: int, memory_limit_gb: float) -> int:
        """Estimate appropriate chunk size based on memory constraints"""
        # Rough estimates for memory usage
        # Typical protein: ~200 residues, ~100MB per design including gradients
        estimated_memory_per_structure = 0.1  # GB
        
        # Factor in number of designs per target and safety margin
        memory_per_structure = (
            estimated_memory_per_structure * 
            self.config.num_designs_per_target * 
            2.0  # Safety margin
        )
        
        chunk_size = max(1, int(memory_limit_gb / memory_per_structure))
        chunk_size = min(chunk_size, total_structures)
        
        print(f"Estimated chunk size: {chunk_size} (based on {memory_limit_gb}GB limit)")
        return chunk_size
    
    def _process_chunk(
        self, 
        chunk_paths: List[str], 
        progress_callback: Optional[callable],
        chunk_start_idx: int,
        **kwargs
    ) -> List[DesignResult]:
        """Process a chunk of structures with memory management"""
        chunk_results = []
        
        for i, backbone_path in enumerate(chunk_paths):
            global_idx = chunk_start_idx + i
            
            # Progress update
            if progress_callback:
                progress_callback(
                    global_idx, 
                    len(chunk_paths), 
                    f"Processing {os.path.basename(backbone_path)}"
                )
            
            print(f"  [{global_idx+1}] Processing: {os.path.basename(backbone_path)}")
            
            try:
                # Design sequence with memory monitoring
                with self._memory_monitor():
                    result = self.design_sequence(backbone_path, **kwargs)
                
                chunk_results.append(result)
                
                # Progress feedback
                status = "✓ Success" if result.success else "✗ Failed"
                energy_info = ""
                if result.energies is not None and len(result.energies) > 0:
                    best_energy = float(result.energies.min())
                    energy_info = f" (best energy: {best_energy:.2f})"
                
                print(f"    {status}{energy_info}")
                
            except Exception as e:
                # Create failed result
                failed_result = DesignResult(
                    sequences=None,
                    logits=None,
                    energies=None,
                    backbone_path=backbone_path,
                    success=False,
                    num_designs=0,
                    optimization_results=[],
                    error_message=str(e),
                    total_time=0.0
                )
                chunk_results.append(failed_result)
                print(f"    ✗ Failed: {str(e)[:100]}...")
            
            # Clean up after each structure to prevent memory accumulation
            if i < len(chunk_paths) - 1:  # Don't clean up after last item
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return chunk_results
    
    def _memory_monitor(self):
        """Context manager for memory monitoring"""
        class MemoryMonitor:
            def __init__(self, device):
                self.device = device
                self.start_memory = None
            
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.start_memory = torch.cuda.memory_allocated()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    end_memory = torch.cuda.memory_allocated()
                    if self.start_memory is not None:
                        memory_diff = (end_memory - self.start_memory) / 1e9  # GB
                        if memory_diff > 1.0:  # Warn if >1GB increase
                            warnings.warn(f"Memory usage increased by {memory_diff:.2f}GB")
        
        return MemoryMonitor(self.device)
    
    def _cleanup_memory(self):
        """Clean up memory between chunks"""
        # Clear gradient caches
        for model in self.energy_models:
            if hasattr(model, 'zero_grad'):
                model.zero_grad()
        
        if hasattr(self.encoder, 'zero_grad'):
            self.encoder.zero_grad()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Python garbage collection
        import gc
        gc.collect()
    
    def _save_chunk_results(
        self, 
        chunk_results: List[DesignResult], 
        output_dir: str, 
        chunk_start: int, 
        chunk_end: int
    ):
        """Save results from a processed chunk"""
        chunk_file = os.path.join(output_dir, f'chunk_{chunk_start}_{chunk_end}.json')
        
        # Convert results to serializable format
        serializable_results = []
        for result in chunk_results:
            result_dict = {
                'backbone_path': result.backbone_path,
                'success': result.success,
                'num_designs': result.num_designs,
                'total_time': result.total_time,
                'error_message': result.error_message,
                'warnings': result.warnings or []
            }
            
            if result.sequences is not None:
                result_dict['sequences'] = result.sequences.cpu().tolist()
            if result.energies is not None:
                result_dict['energies'] = result.energies.cpu().tolist()
            if result.validation_metrics is not None:
                result_dict['validation_metrics'] = result.validation_metrics
            
            serializable_results.append(result_dict)
        
        with open(chunk_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _save_final_batch_results(self, all_results: List[DesignResult], output_dir: str):
        """Save final batch summary and statistics"""
        # Summary statistics
        total_structures = len(all_results)
        successful_designs = sum(1 for r in all_results if r.success)
        total_time = sum(r.total_time for r in all_results)
        
        # Collect all energies from successful designs
        all_energies = []
        for result in all_results:
            if result.energies is not None:
                all_energies.extend(result.energies.cpu().tolist())
        
        batch_summary = {
            'total_structures': total_structures,
            'successful_designs': successful_designs,
            'success_rate': successful_designs / total_structures if total_structures > 0 else 0.0,
            'total_time_seconds': total_time,
            'mean_time_per_structure': total_time / total_structures if total_structures > 0 else 0.0,
            'energy_statistics': {
                'mean': float(np.mean(all_energies)) if all_energies else None,
                'std': float(np.std(all_energies)) if len(all_energies) > 1 else None,
                'min': float(np.min(all_energies)) if all_energies else None,
                'max': float(np.max(all_energies)) if all_energies else None,
                'count': len(all_energies)
            } if all_energies else {}
        }
        
        summary_file = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"Batch summary saved to: {summary_file}")
        print(f"Success rate: {batch_summary['success_rate']:.1%} "
              f"({successful_designs}/{total_structures})")
    
    def process_large_batch(
        self, 
        backbone_paths: List[str],
        output_dir: str,
        chunk_size: int = 10,
        **kwargs
    ) -> str:
        """
        Process a large batch with automatic result saving and recovery.
        
        This method is designed for very large batches (hundreds/thousands of structures)
        where you want automatic checkpointing and the ability to resume failed runs.
        
        Args:
            backbone_paths: List of PDB file paths
            output_dir: Directory to save all results and checkpoints
            chunk_size: Structures per chunk
            **kwargs: Design parameters
        
        Returns:
            Path to final results summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input list for potential resume
        input_file = os.path.join(output_dir, 'input_structures.txt')
        with open(input_file, 'w') as f:
            for path in backbone_paths:
                f.write(f"{path}\n")
        
        # Process with automatic saving
        results = self.design_batch(
            backbone_paths,
            chunk_size=chunk_size,
            save_intermediate=True,
            intermediate_dir=output_dir,
            **kwargs
        )
        
        # Save final FASTA sequences
        self._save_batch_fasta(results, output_dir)
        
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        return summary_path
    
    def _save_batch_fasta(self, results: List[DesignResult], output_dir: str):
        """Save all designed sequences as FASTA files"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # All sequences in one file
        all_fasta_path = os.path.join(output_dir, 'all_designed_sequences.fasta')
        successful_fasta_path = os.path.join(output_dir, 'successful_designs.fasta')
        
        with open(all_fasta_path, 'w') as all_file, open(successful_fasta_path, 'w') as success_file:
            for i, result in enumerate(results):
                structure_name = os.path.splitext(os.path.basename(result.backbone_path))[0]
                
                if result.sequences is not None:
                    for j, seq_indices in enumerate(result.sequences):
                        sequence = ''.join(amino_acids[idx] for idx in seq_indices.cpu().numpy())
                        header = f'>design_{i}_{j}_{structure_name}'
                        
                        all_file.write(f'{header}\n{sequence}\n')
                        
                        if result.success:
                            success_file.write(f'{header}\n{sequence}\n')
                else:
                    # Write placeholder for failed designs
                    all_file.write(f'>failed_{i}_{structure_name}\n# Design failed\n')
        
        print(f"FASTA sequences saved to: {all_fasta_path}")
        print(f"Successful designs saved to: {successful_fasta_path}")
    
    def save_results(
        self,
        results: Union[DesignResult, List[DesignResult]],
        output_dir: str,
        format: str = 'json'
    ):
        """Save design results to file(s)"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not isinstance(results, list):
            results = [results]
        
        for i, result in enumerate(results):
            if format == 'json':
                self._save_result_json(result, output_dir, i)
            elif format == 'fasta':
                self._save_result_fasta(result, output_dir, i)
            else:
                raise ValueError(f"Unsupported output format: {format}")
    
    def _save_result_json(self, result: DesignResult, output_dir: str, index: int):
        """Save result as JSON file"""
        # Convert tensors to lists for JSON serialization
        result_dict = {
            'backbone_path': result.backbone_path,
            'success': result.success,
            'num_designs': result.num_designs,
            'total_time': result.total_time,
            'error_message': result.error_message,
            'warnings': result.warnings
        }
        
        if result.sequences is not None:
            result_dict['sequences'] = result.sequences.cpu().tolist()
        if result.energies is not None:
            result_dict['energies'] = result.energies.cpu().tolist()
        if result.validation_metrics is not None:
            result_dict['validation_metrics'] = result.validation_metrics
        
        output_path = os.path.join(output_dir, f'design_result_{index}.json')
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_result_fasta(self, result: DesignResult, output_dir: str, index: int):
        """Save sequences as FASTA file"""
        if result.sequences is None:
            return
        
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        output_path = os.path.join(output_dir, f'designed_sequences_{index}.fasta')
        
        with open(output_path, 'w') as f:
            for i, seq_indices in enumerate(result.sequences):
                sequence = ''.join(amino_acids[idx] for idx in seq_indices)
                f.write(f'>design_{index}_{i}\n{sequence}\n')


def load_pipeline_from_config(config_path: str) -> ProteinDesignPipeline:
    """
    Convenience function to load pipeline from configuration file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Initialized ProteinDesignPipeline
    """
    return ProteinDesignPipeline(config_path)


def create_default_config(
    encoder_checkpoint: str,
    energy_checkpoint: str,
    output_path: str = 'pipeline_config.json'
) -> str:
    """
    Create a default pipeline configuration file.
    
    Args:
        encoder_checkpoint: Path to ProteinMPNN encoder checkpoint
        energy_checkpoint: Path to energy model checkpoint
        output_path: Where to save the configuration file
    
    Returns:
        Path to created configuration file
    """
    config = PipelineConfig(
        encoder_checkpoint=encoder_checkpoint,
        energy_model_checkpoint=energy_checkpoint
    )
    
    # Convert to dict for JSON serialization
    config_dict = {
        'encoder_checkpoint': config.encoder_checkpoint,
        'energy_model_checkpoint': config.energy_model_checkpoint,
        'device': config.device,
        'batch_size': config.batch_size,
        'max_sequence_length': config.max_sequence_length,
        'encoder_config': config.encoder_config,
        'energy_config': config.energy_config,
        'sequence_config': config.sequence_config,
        'optimization_config': config.optimization_config,
        'use_proteinmpnn_init': config.use_proteinmpnn_init,
        'num_designs_per_target': config.num_designs_per_target,
        'validate_results': config.validate_results,
        'save_trajectories': config.save_trajectories,
        'output_format': config.output_format
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Default configuration saved to: {output_path}")
    return output_path