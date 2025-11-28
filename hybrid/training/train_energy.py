#!/usr/bin/env python3
"""
Energy Model Training Script

This script implements the training pipeline for the energy-based protein design system.
It trains the energy model to distinguish between stable (positive) and unstable (negative)
sequence/structure pairs using contrastive learning.

Key Components:
1. Model initialization: ProteinMPNN encoder + Energy head
2. Dataset loading: StabilityDataset with positive/negative pairs
3. Training loop: Contrastive loss optimization with validation monitoring
4. Checkpointing: Save/load model state for resuming training
5. Logging: Comprehensive metrics and visualizations
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Sampler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Try to import optional dependencies
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install tensorboard for logging: pip install tensorboard")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Install for plotting: pip install matplotlib seaborn")

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import project modules
from models.mpnn_encoder import ProteinMPNNBackboneEncoder
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from data.stability_dataset import StabilityDataset
from training.losses import ContrastiveLoss, NegativeType


class BalancedBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch contains both positive and negative samples.
    
    This sampler creates batches by interleaving positive and negative samples to ensure
    balanced composition for contrastive learning.
    """
    
    def __init__(self, dataset, batch_size, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Identify positive and negative sample indices
        self.positive_indices = []
        self.negative_indices = []
        
        print("Analyzing dataset for balanced sampling...")
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                label = sample.get('label')
                if label == 1:
                    self.positive_indices.append(idx)
                elif label == 0:
                    self.negative_indices.append(idx)
            except Exception as e:
                print(f"Warning: Error accessing sample {idx}: {e}")
                continue
        
        print(f"BalancedBatchSampler found: {len(self.positive_indices)} positive, {len(self.negative_indices)} negative samples")
        
        if len(self.positive_indices) == 0:
            raise ValueError("No positive samples found in dataset")
        if len(self.negative_indices) == 0:
            raise ValueError("No negative samples found in dataset")
        
        # Calculate number of batches
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        samples_per_batch_half = self.batch_size // 2
        max_batches = min_samples // samples_per_batch_half
        
        if max_batches == 0:
            raise ValueError(f"Dataset too small for balanced batching. Need at least {samples_per_batch_half} positive and {samples_per_batch_half} negative samples")
        
        self.num_batches = max_batches
        print(f"Creating {self.num_batches} balanced batches of size {self.batch_size}")
    
    def __iter__(self):
        # Shuffle both positive and negative indices
        positive_shuffled = self.positive_indices.copy()
        negative_shuffled = self.negative_indices.copy()
        
        # Use deterministic shuffling for reproducibility (epoch-based seeding)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        
        positive_perm = torch.randperm(len(positive_shuffled), generator=generator)
        negative_perm = torch.randperm(len(negative_shuffled), generator=generator)
        
        positive_shuffled = [positive_shuffled[i] for i in positive_perm]
        negative_shuffled = [negative_shuffled[i] for i in negative_perm]
        
        # Create balanced batches
        samples_per_half = self.batch_size // 2
        
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * samples_per_half
            
            # Get positive samples
            pos_start = start_idx % len(positive_shuffled)
            pos_samples = []
            for i in range(samples_per_half):
                pos_samples.append(positive_shuffled[(pos_start + i) % len(positive_shuffled)])
            
            # Get negative samples
            neg_start = start_idx % len(negative_shuffled)
            neg_samples = []
            for i in range(samples_per_half):
                neg_samples.append(negative_shuffled[(neg_start + i) % len(negative_shuffled)])
            
            # Combine and shuffle within batch
            batch_indices = pos_samples + neg_samples
            batch_perm = torch.randperm(len(batch_indices), generator=generator)
            batch_indices = [batch_indices[i] for i in batch_perm]
            
            # Handle odd batch sizes
            if self.batch_size % 2 == 1:
                # Add one more sample (alternating positive/negative)
                if batch_idx % 2 == 0 and positive_shuffled:
                    extra_idx = positive_shuffled[(start_idx + samples_per_half) % len(positive_shuffled)]
                elif negative_shuffled:
                    extra_idx = negative_shuffled[(start_idx + samples_per_half) % len(negative_shuffled)]
                else:
                    extra_idx = positive_shuffled[(start_idx + samples_per_half) % len(positive_shuffled)]
                batch_indices.append(extra_idx)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class EnergyModelTrainer:
    """
    Complete training pipeline for energy-based protein design.
    
    This class orchestrates the entire training process including model initialization,
    data loading, optimization, validation monitoring, checkpointing, and logging.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config: Training configuration dictionary
            model_dir: Directory for saving model checkpoints
            log_dir: Directory for logging outputs
            device: Training device ('cuda', 'cpu', 'mps', or None for auto)
        """
        self.config = config
        
        # Set up directories
        self.model_dir = Path(model_dir) if model_dir else Path("checkpoints")
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.get('seed', 42))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'energy_stats': []
        }
        
        print("EnergyModelTrainer initialized successfully")
    
    def setup(self):
        """Set up all training components: model, data, optimizer, loss function"""
        print("Setting up training components...")
        
        # Initialize model
        self._setup_model()
        
        # Setup data loaders
        self._setup_data()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup logging
        self._setup_logging()
        
        print("All components set up successfully")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training'].get('patience', 20)
        
        start_time = time.time()
        
        # Validate dataset composition before starting training
        self._validate_dataset_composition()
        
        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                train_metrics = self._train_epoch()
                
                # Validate
                val_metrics = self._validate_epoch()
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics)
                
                # Check early stopping
                improved = val_metrics['loss'] < self.best_val_loss
                if improved:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pt', is_best=True)
                else:
                    self.patience_counter += 1
                
                # Save regular checkpoint
                if (epoch + 1) % self.config['training'].get('save_frequency', 10) == 0:
                    self._save_checkpoint(f'epoch_{epoch+1}.pt')
                
                # Print progress
                print(f"Epoch {epoch+1}/{max_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.6f} - "
                      f"Val Loss: {val_metrics['loss']:.6f} - "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        finally:
            total_time = time.time() - start_time
            print(f"Training completed in {total_time/3600:.2f} hours")
            
            # Save final checkpoint
            self._save_checkpoint('final_model.pt')
            
            # Generate training summary
            self._generate_training_summary()
            
            if self.writer:
                self.writer.close()
    
    def _setup_model(self):
        """Initialize the complete energy prediction model"""
        model_config = self.config['model']
        
        print("Initializing model components...")
        
        # Initialize ProteinMPNN encoder
        mpnn_config = model_config['mpnn_encoder']
        
        # Create encoder (will handle loading pre-trained weights internally)
        encoder = ProteinMPNNBackboneEncoder.from_pretrained(
            model_name=mpnn_config.get('model_name', 'v_48_020'),
            freeze_layers=mpnn_config.get('freeze_layers', True)
        ).to(self.device)
        
        # Initialize energy head
        energy_config = model_config['energy_head']
        energy_head = EnergyHead(
            backbone_dim=mpnn_config.get('hidden_dim', 128),
            seq_dim=20,  # Standard amino acids
            hidden_dim=energy_config.get('hidden_dim', 512),
            num_layers=energy_config.get('num_layers', 3),
            dropout=energy_config.get('dropout', 0.1),
            activation=energy_config.get('activation', 'relu'),
            use_batch_norm=energy_config.get('use_batch_norm', True)
        )
        
        # Initialize sequence representation
        sequence_config = model_config.get('sequence_repr', {})
        sequence_repr = ContinuousSequenceRepr(
            vocab_size=20,
            temperature_schedule=sequence_config.get('temperature_schedule', [1.0, 0.5, 0.1]),
            min_temperature=sequence_config.get('min_temperature', 1e-3),
            max_temperature=sequence_config.get('max_temperature', 10.0)
        )
        
        # Create combined model
        self.model = EnergyPredictionModel(encoder, energy_head, sequence_repr)
        self.model = self.model.to(self.device)
        
        # Load checkpoint if specified
        checkpoint_path = self.config.get('resume_from_checkpoint')
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _setup_data(self):
        """Set up data loaders for training and validation"""
        data_config = self.config['data']
        
        print("Setting up data loaders...")
        
        # Create dataset
        dataset = StabilityDataset(
            data_dir=data_config['data_dir'],
            positive_ratio=data_config.get('positive_ratio', 0.5),
            negative_methods=data_config.get('negative_methods'),
            max_sequence_length=data_config.get('max_sequence_length', 500),
            min_sequence_length=data_config.get('min_sequence_length', 20),
            max_files=data_config.get('max_files_debug'),
            lazy_loading=data_config.get('lazy_loading', True),
            seed=self.config.get('seed', 42)
        )
        
        # Validate dataset has samples (force loading to check actual sample count)
        try:
            # Force loading if lazy loading is enabled to get accurate sample count
            if hasattr(dataset, '_lazy_loaded') and not dataset._lazy_loaded:
                dataset._load_dataset()
                dataset._lazy_loaded = True
            
            # Check actual sample count
            actual_sample_count = len(dataset.samples) if hasattr(dataset, 'samples') else len(dataset)
            if actual_sample_count == 0:
                raise ValueError(
                    "Dataset is empty! This is likely due to missing dependencies.\n"
                    "Please install BioPython for PDB parsing:\n"
                    "  pip install biopython\n"
                    "Or check that your data directory contains valid PDB files."
                )
        except Exception as e:
            if "missing dependencies" in str(e) or "Dataset is empty" in str(e):
                raise  # Re-raise our own error
            else:
                # Wrap other errors with context
                raise ValueError(
                    f"Failed to validate dataset: {e}\n"
                    "This may be due to missing dependencies. Please install BioPython:\n"
                    "  pip install biopython"
                ) from e
        
        # Split into train/validation
        val_split = data_config.get('val_split', 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        
        # Amino acid encoding utilities
        # Standard amino acid alphabet (same as in StabilityDataset)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        def encode_sequence(sequence):
            """Convert amino acid sequence string to tensor of indices"""
            if isinstance(sequence, str):
                if not sequence:
                    return torch.tensor([], dtype=torch.long)
                
                # Convert string sequence to indices
                indices = []
                for aa in sequence:
                    aa_upper = aa.upper()
                    if aa_upper in aa_to_idx:
                        indices.append(aa_to_idx[aa_upper])
                    # Skip unknown amino acids (don't raise error, just filter)
                
                return torch.tensor(indices, dtype=torch.long)
            elif isinstance(sequence, torch.Tensor):
                # Already encoded, return as-is
                return sequence
            else:
                # Try to convert directly
                return torch.tensor(sequence, dtype=torch.long)
        
        # Custom collate function to handle protein data
        def collate_fn(batch):
            """Custom collate function for protein dataset with batch balance validation"""
            # Batch is a list of individual samples from StabilityDataset
            collated = {}
            
            # Extract labels for balance checking
            labels = [item.get('label') for item in batch]
            pos_count = sum(1 for label in labels if label == 1)
            neg_count = sum(1 for label in labels if label == 0)
            
            # Warn if batch is imbalanced (but don't fail - BalancedBatchSampler should handle this)
            if pos_count == 0 or neg_count == 0:
                print(f"Warning: Imbalanced batch detected - {pos_count} positive, {neg_count} negative samples")
            
            # Stack/pad tensors appropriately with strict error handling
            if 'backbone_features' in batch[0]:
                backbone_features = []
                for item in batch:
                    feat = item['backbone_features']
                    if not isinstance(feat, torch.Tensor):
                        feat = torch.tensor(feat)
                    backbone_features.append(feat)
                collated['backbone_features'] = torch.stack(backbone_features)
            
            if 'sequence' in batch[0]:
                sequences = []
                for item in batch:
                    seq = item['sequence']
                    seq = encode_sequence(seq)
                    sequences.append(seq)
                
                # Pad sequences to same length for stacking
                if sequences:
                    max_len = max(seq.size(0) for seq in sequences)
                    padded_sequences = []
                    for seq in sequences:
                        if seq.size(0) < max_len:
                            # Pad with zeros (could also use a special padding token)
                            padding = torch.zeros(max_len - seq.size(0), dtype=torch.long)
                            seq = torch.cat([seq, padding])
                        padded_sequences.append(seq)
                    collated['sequence'] = torch.stack(padded_sequences)
                else:
                    collated['sequence'] = torch.empty((0,), dtype=torch.long)
            
            if 'mask' in batch[0]:
                masks = []
                for item in batch:
                    mask = item['mask']
                    if not isinstance(mask, torch.Tensor):
                        mask = torch.tensor(mask)
                    masks.append(mask)
                collated['mask'] = torch.stack(masks)
            
            # Handle scalars and labels - fail-fast on errors
            collated['label'] = torch.tensor([item['label'] for item in batch])
            collated['length'] = torch.tensor([item['length'] for item in batch])
            
            # Handle string/categorical data - use .get() for optional fields
            collated['generation_method'] = [item.get('generation_method', None) for item in batch]
            if 'structure_id' in batch[0]:
                collated['structure_id'] = [item['structure_id'] for item in batch]
            
            return collated
        
        # Use BalancedBatchSampler for training to ensure balanced positive/negative batches
        train_sampler = BalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            seed=self.config.get('seed', 42)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )
        print("Using BalancedBatchSampler for training data")
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )
        
        print(f"Data loaded - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    def _setup_optimization(self):
        """Set up optimizer and learning rate scheduler"""
        opt_config = self.config['optimization']
        
        # Create optimizer
        optimizer_type = opt_config.get('optimizer', 'adamw')
        lr = opt_config.get('learning_rate', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Create scheduler
        scheduler_config = opt_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        print(f"Optimization setup - Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    
    def _setup_loss_function(self):
        """Set up the contrastive loss function"""
        loss_config = self.config['loss']
        
        self.loss_fn = ContrastiveLoss(
            margin=loss_config.get('margin', 1.0),
            temperature=loss_config.get('temperature', 0.1),
            ranking_weight=loss_config.get('ranking_weight', 1.0),
            contrastive_weight=loss_config.get('contrastive_weight', 1.0),
            entropy_weight=loss_config.get('entropy_weight', 0.01),
            smoothness_weight=loss_config.get('smoothness_weight', 0.001),
            negative_weights=loss_config.get('negative_weights'),
            reduction='mean'
        )
        
        print(f"Loss function configured: {self.loss_fn.get_config()}")
    
    def _setup_logging(self):
        """Set up tensorboard logging"""
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"energy_training_{timestamp}"
            self.writer = SummaryWriter(self.log_dir / log_name)
            print(f"TensorBoard logging to: {self.log_dir / log_name}")
        else:
            self.writer = None
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch and return metrics"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        loss_components = {'ranking': 0.0, 'contrastive': 0.0, 'entropy': 0.0, 'smoothness': 0.0}
        energy_stats = {'pos_mean': 0.0, 'neg_mean': 0.0, 'pos_std': 0.0, 'neg_std': 0.0}
        
        # Early warning counters
        total_batches = 0
        skipped_batches = 0
        warning_threshold = 0.8  # Warn if >80% batches are skipped
        warning_issued = False
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            total_batches += 1
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self._forward_pass(batch)
                
                # Skip batch if empty (no positive or negative samples)
                if outputs.get('skip_batch', False):
                    skipped_batches += 1
                    
                    # Issue early warning if too many batches are being skipped
                    if not warning_issued and total_batches >= min(10, len(self.train_loader)):  # Adapt to dataset size
                        skip_rate = skipped_batches / total_batches
                        if skip_rate > warning_threshold:
                            warnings.warn(
                                f"High batch skip rate detected: {skip_rate:.1%} of batches skipped "
                                f"({skipped_batches}/{total_batches}). This may lead to training failure. "
                                "Check dataset composition - each batch needs both positive and negative samples."
                            )
                            warning_issued = True
                    continue
                
                # Compute loss
                loss = self.loss_fn(
                    pos_energies=outputs['pos_energies'],
                    neg_energies=outputs['neg_energies'],
                    pos_sequence_probs=outputs.get('pos_sequence_probs'),
                    neg_sequence_probs=outputs.get('neg_sequence_probs'),
                    negative_types=outputs.get('negative_types')
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Accumulate statistics
                batch_size = len(outputs['pos_energies'])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accumulate loss components
                individual_losses = self.loss_fn.get_last_losses()
                for component, value in individual_losses.items():
                    if component != 'total_loss' and component in loss_components:
                        loss_components[component] += value * batch_size
                
                # Energy statistics
                with torch.no_grad():
                    pos_energies = outputs['pos_energies']
                    neg_energies = outputs['neg_energies']
                    
                    energy_stats['pos_mean'] += pos_energies.mean().item() * batch_size
                    energy_stats['neg_mean'] += neg_energies.mean().item() * batch_size
                    energy_stats['pos_std'] += pos_energies.std().item() * batch_size
                    energy_stats['neg_std'] += neg_energies.std().item() * batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Pos_E': f"{pos_energies.mean().item():.3f}",
                    'Neg_E': f"{neg_energies.mean().item():.3f}"
                })
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    warnings.warn(f"NaN/Inf loss detected at batch {batch_idx}")
                    skipped_batches += 1
                    
                    # Check for early warning
                    if not warning_issued and total_batches >= min(10, len(self.train_loader)):
                        skip_rate = skipped_batches / total_batches
                        if skip_rate > warning_threshold:
                            warnings.warn(
                                f"High batch skip rate detected: {skip_rate:.1%} of batches skipped "
                                f"({skipped_batches}/{total_batches}). This may lead to training failure. "
                                "Check for NaN/Inf losses and dataset composition issues."
                            )
                            warning_issued = True
                    continue
            
            except Exception as e:
                warnings.warn(f"Error in training batch {batch_idx}: {e}")
                skipped_batches += 1
                
                # Check for early warning
                if not warning_issued and total_batches >= min(10, len(self.train_loader)):
                    skip_rate = skipped_batches / total_batches
                    if skip_rate > warning_threshold:
                        warnings.warn(
                            f"High batch skip rate detected: {skip_rate:.1%} of batches skipped "
                            f"({skipped_batches}/{total_batches}). This may lead to training failure. "
                            "Check for batch processing errors and dataset integrity."
                        )
                        warning_issued = True
                continue
        
        # Compute epoch averages
        if total_samples == 0:
            raise RuntimeError(
                "No training samples were processed in this epoch. This could be due to:\n"
                "- All batches being skipped (lack of both positive and negative samples)\n"
                "- All batches having NaN/Inf losses\n"
                "- Dataset loading errors\n"
                "- Missing dependencies (e.g., ProteinMPNN utilities)\n"
                "Check dataset integrity and batch composition."
            )
        
        avg_loss = total_loss / total_samples
        for component in loss_components:
            loss_components[component] /= total_samples
        for stat in energy_stats:
            energy_stats[stat] /= total_samples
        
        return {
            'loss': avg_loss,
            'loss_components': loss_components,
            'energy_stats': energy_stats
        }
    
    def _validate_dataset_composition(self) -> None:
        """Validate that the dataset contains properly balanced batches before training"""
        print("Validating dataset composition...")
        
        # First, check the dataset composition before batching
        print("\n=== Dataset Analysis ===")
        train_dataset = self.train_loader.dataset
        total_samples = len(train_dataset)
        print(f"Total dataset samples: {total_samples}")
        
        # Count positive/negative samples in the raw dataset
        positive_count = 0
        negative_count = 0
        sample_errors = 0
        
        print("Sampling dataset to check positive/negative balance...")
        sample_size = min(100, total_samples)  # Sample up to 100 items
        sample_indices = list(range(0, total_samples, max(1, total_samples // sample_size)))
        
        for idx in sample_indices:
            try:
                sample = train_dataset[idx]
                label = sample.get('label')
                if label == 1:
                    positive_count += 1
                elif label == 0:
                    negative_count += 1
                else:
                    sample_errors += 1
                    print(f"  Warning: Sample {idx} has invalid label: {label}")
            except Exception as e:
                sample_errors += 1
                print(f"  Error accessing sample {idx}: {e}")
        
        print(f"Raw dataset composition (sampled {len(sample_indices)} items):")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")
        print(f"  Sample errors: {sample_errors}")
        print(f"  Positive ratio: {positive_count / (positive_count + negative_count):.3f}")
        
        if positive_count == 0:
            raise RuntimeError("Dataset contains NO positive samples")
        if negative_count == 0:
            raise RuntimeError("Dataset contains NO negative samples")
        
        print("\n=== Batch Analysis ===")
        # Sample first few batches to check for positive/negative balance
        sample_limit = min(10, len(self.train_loader))  # Check more batches
        total_batches_checked = 0
        problematic_batches = 0
        
        # Create iterator to avoid affecting the main training loop
        temp_loader = iter(self.train_loader)
        
        for i in range(sample_limit):
            try:
                batch = next(temp_loader)
                total_batches_checked += 1
                
                # Debug: Check what we actually got
                print(f"  Debug - Batch {i+1} type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"    Keys: {list(batch.keys())}")
                    if 'label' in batch:
                        print(f"    Labels type: {type(batch['label'])}, value: {batch['label']}")
                
                # Check batch composition (no need to move to device for validation)
                if not isinstance(batch, dict):
                    print(f"    Error: Expected dict batch, got {type(batch)}")
                    problematic_batches += 1
                    continue
                    
                labels = batch.get('label')
                if labels is not None:
                    # Handle different label formats
                    if isinstance(labels, (list, tuple)):
                        labels = torch.tensor(labels)
                    elif not isinstance(labels, torch.Tensor):
                        try:
                            labels = torch.tensor(labels)
                        except Exception as label_error:
                            print(f"    Error converting labels to tensor: {label_error}")
                            print(f"    Labels content: {labels}")
                            problematic_batches += 1
                            continue
                    
                    pos_count = (labels == 1).sum().item()
                    neg_count = (labels == 0).sum().item()
                    
                    print(f"  Batch {i+1}: {pos_count} positive, {neg_count} negative samples")
                    
                    if pos_count == 0 or neg_count == 0:
                        problematic_batches += 1
                        print(f"    ⚠️  IMBALANCED: Only {'positive' if pos_count > 0 else 'negative'} samples!")
                else:
                    print(f"  Warning: Batch {i+1} has no label field")
                    problematic_batches += 1
                    
            except StopIteration:
                print(f"  Dataset exhausted after {total_batches_checked} batches")
                break
            except Exception as e:
                print(f"  Error checking batch {i+1}: {e}")
                total_batches_checked += 1
                problematic_batches += 1
        
        # Analysis and warnings
        if total_batches_checked == 0:
            raise RuntimeError("Dataset appears to be empty - no batches could be loaded")
        
        problem_rate = problematic_batches / total_batches_checked
        print(f"Dataset validation complete: {problematic_batches}/{total_batches_checked} "
              f"batches ({problem_rate:.1%}) are problematic")
        
        # Check if we're using BalancedBatchSampler (which should fix the issue)
        using_balanced_sampler = hasattr(self.train_loader, 'batch_sampler') and \
                                isinstance(getattr(self.train_loader, 'batch_sampler', None), BalancedBatchSampler)
        
        if using_balanced_sampler:
            print("✅ Using BalancedBatchSampler - batch composition should be balanced")
            if problem_rate > 0:
                print(f"⚠️  Note: {problem_rate:.1%} validation errors detected, but BalancedBatchSampler should handle training correctly")
        elif problem_rate > 0.5:  # More than 50% problematic
            raise RuntimeError(
                f"Dataset validation failed: {problem_rate:.1%} of sampled batches contain only "
                "positive OR negative samples, but training requires both types in each batch. "
                "Check dataset creation and batch composition."
            )
        elif problem_rate > 0:  # Some problematic batches
            warnings.warn(
                f"Dataset contains {problem_rate:.1%} problematic batches that will be skipped "
                "during training. Consider rebalancing the dataset for better performance."
            )
        else:
            print("✓ Dataset validation passed - batches contain proper positive/negative balance")
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch and return metrics"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        energy_stats = {'pos_mean': 0.0, 'neg_mean': 0.0, 'ranking_accuracy': 0.0}
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                try:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    outputs = self._forward_pass(batch)
                    
                    # Skip batch if empty (no positive or negative samples)
                    if outputs.get('skip_batch', False):
                        continue
                    
                    # Compute loss
                    loss = self.loss_fn(
                        pos_energies=outputs['pos_energies'],
                        neg_energies=outputs['neg_energies'],
                        pos_sequence_probs=outputs.get('pos_sequence_probs'),
                        neg_sequence_probs=outputs.get('neg_sequence_probs'),
                        negative_types=outputs.get('negative_types')
                    )
                    
                    # Accumulate statistics
                    batch_size = len(outputs['pos_energies'])
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    # Energy statistics
                    pos_energies = outputs['pos_energies']
                    neg_energies = outputs['neg_energies']
                    
                    energy_stats['pos_mean'] += pos_energies.mean().item() * batch_size
                    energy_stats['neg_mean'] += neg_energies.mean().item() * batch_size
                    
                    # Ranking accuracy (percentage of pos < neg comparisons)
                    pos_expanded = pos_energies.unsqueeze(1)
                    neg_expanded = neg_energies.unsqueeze(0)
                    correct_rankings = (pos_expanded < neg_expanded).float().mean()
                    energy_stats['ranking_accuracy'] += correct_rankings.item() * batch_size
                
                except Exception as e:
                    warnings.warn(f"Error in validation batch: {e}")
                    continue
        
        # Compute averages
        avg_loss = total_loss / total_samples
        for stat in energy_stats:
            energy_stats[stat] /= total_samples
        
        return {
            'loss': avg_loss,
            'energy_stats': energy_stats
        }
    
    def _forward_pass(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Perform forward pass through the model"""
        # Split batch by labels
        labels = batch['label']  # [B] - 1 for positive, 0 for negative
        
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            warnings.warn("Batch contains only positive or only negative samples, skipping")
            # Return empty result to signal skip
            return {
                'pos_energies': torch.tensor([], device=self.device),
                'neg_energies': torch.tensor([], device=self.device),
                'negative_types': [],
                'skip_batch': True
            }
        
        # Process positive samples
        if pos_mask.sum() > 0:
            pos_backbone = batch['backbone_features'][pos_mask]
            pos_sequence = batch['sequence'][pos_mask]
            pos_energies = self.model(
                backbone_features=pos_backbone,
                sequence=pos_sequence,
                mask=batch.get('mask')[pos_mask] if batch.get('mask') is not None else None
            )
        else:
            pos_energies = torch.tensor([], device=self.device)
        
        # Process negative samples
        if neg_mask.sum() > 0:
            neg_backbone = batch['backbone_features'][neg_mask]
            neg_sequence = batch['sequence'][neg_mask]
            neg_energies = self.model(
                backbone_features=neg_backbone,
                sequence=neg_sequence,
                mask=batch.get('mask')[neg_mask] if batch.get('mask') is not None else None
            )
            
            # Extract negative types for negative samples (filter None values)
            negative_types = [
                batch['generation_method'][i] 
                for i in range(len(labels)) 
                if neg_mask[i] and batch['generation_method'][i] is not None
            ]
        else:
            neg_energies = torch.tensor([], device=self.device)
            negative_types = []
        
        return {
            'pos_energies': pos_energies,
            'neg_energies': neg_energies,
            'negative_types': negative_types
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to the training device"""
        def move_dict_to_device(data_dict):
            moved_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    moved_dict[key] = value.to(self.device)
                elif isinstance(value, dict):
                    moved_dict[key] = move_dict_to_device(value)
                else:
                    moved_dict[key] = value
            return moved_dict
        
        return move_dict_to_device(batch)
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for the current epoch"""
        epoch = self.current_epoch
        
        # Store in history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.training_history['energy_stats'].append({
            'train': train_metrics['energy_stats'],
            'val': val_metrics['energy_stats']
        })
        
        # TensorBoard logging
        if self.writer:
            # Loss curves
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Loss components (training only)
            if 'loss_components' in train_metrics:
                for component, value in train_metrics['loss_components'].items():
                    self.writer.add_scalar(f'Loss_Components/{component}', value, epoch)
            
            # Energy statistics
            train_energy = train_metrics['energy_stats']
            val_energy = val_metrics['energy_stats']
            
            self.writer.add_scalar('Energy/Train_Pos_Mean', train_energy['pos_mean'], epoch)
            self.writer.add_scalar('Energy/Train_Neg_Mean', train_energy['neg_mean'], epoch)
            self.writer.add_scalar('Energy/Val_Pos_Mean', val_energy['pos_mean'], epoch)
            self.writer.add_scalar('Energy/Val_Neg_Mean', val_energy['neg_mean'], epoch)
            self.writer.add_scalar('Energy/Val_Ranking_Accuracy', val_energy['ranking_accuracy'], epoch)
            
            # Energy distributions histogram
            if epoch % 10 == 0:  # Less frequent for performance
                self.writer.add_histogram('Energy_Distribution/Train_Positive', 
                                        torch.tensor([train_energy['pos_mean']]), epoch)
                self.writer.add_histogram('Energy_Distribution/Train_Negative', 
                                        torch.tensor([train_energy['neg_mean']]), epoch)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history,
            'random_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state()
            }
        }
        
        filepath = self.model_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"New best model saved: {filepath}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint for resuming training"""
        # PyTorch 2.6 secure loading: use weights_only=True to prevent arbitrary code execution
        # Only set weights_only=False for trusted legacy checkpoints
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint with weights_only=True: {e}. "
                          f"Checkpoint may contain non-tensor data. Only load from trusted sources.")
            # Require explicit confirmation for unsafe loading
            if not self.config.get('allow_unsafe_checkpoint_loading', False):
                raise ValueError(f"Checkpoint requires unsafe loading (weights_only=False). "
                                f"Set allow_unsafe_checkpoint_loading=True in config only if you trust this file: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if self.model is None:
            raise RuntimeError("Model must be initialized before loading checkpoint")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif 'optimizer_state_dict' in checkpoint:
            warnings.warn("Checkpoint contains optimizer state but no optimizer is initialized")
        
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Restore random states for reproducibility
        if 'random_state' in checkpoint:
            try:
                random.setstate(checkpoint['random_state']['python'])
                np.random.set_state(checkpoint['random_state']['numpy'])
                torch.set_rng_state(checkpoint['random_state']['torch'])
            except Exception as e:
                warnings.warn(f"Could not restore random states: {e}")
        
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.current_epoch}")
    
    def _generate_training_summary(self):
        """Generate and save training summary"""
        summary = {
            'training_config': self.config,
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
                'total_epochs': self.current_epoch
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_history': self.training_history
        }
        
        # Save as JSON
        summary_path = self.log_dir / f"training_summary_epoch_{self.current_epoch}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Training summary saved to: {summary_path}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Make operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EnergyPredictionModel(nn.Module):
    """
    Complete energy prediction model combining ProteinMPNN encoder and energy head.
    
    This model integrates all components for end-to-end energy prediction from
    protein backbone structures and sequences.
    """
    
    def __init__(
        self,
        backbone_encoder: ProteinMPNNBackboneEncoder,
        energy_head: EnergyHead,
        sequence_repr: ContinuousSequenceRepr
    ):
        super().__init__()
        
        self.backbone_encoder = backbone_encoder
        self.energy_head = energy_head
        self.sequence_repr = sequence_repr
    
    def forward(
        self,
        backbone_features: torch.Tensor,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        landscape_idx: int = 0
    ) -> torch.Tensor:
        """
        Predict energy for a batch of structures and sequences.
        
        Args:
            backbone_features: Pre-computed backbone features [B, L, D] or raw coordinates
            sequence: Amino acid sequences as indices [B, L] or logits [B, L, 20]
            mask: Sequence mask [B, L]
            landscape_idx: Current optimization landscape index
        
        Returns:
            energies: Predicted energies [B]
        """
        # Encode backbone if needed (handle both pre-encoded and raw coordinates)
        if backbone_features.dim() == 3 and backbone_features.shape[-1] == 128:
            # Already encoded backbone features
            encoded_backbone = backbone_features
        else:
            # Raw coordinates - need to encode
            # This would require implementation of coordinate processing in ProteinMPNN encoder
            # For now, assume pre-encoded features are provided
            encoded_backbone = backbone_features
        
        # Process sequence representation
        if sequence.dim() == 2:
            # Convert indices to one-hot then to proper logits
            sequence_onehot = F.one_hot(sequence.long(), num_classes=20).float()
            # Convert to logits: use large positive value (10.0) for selected position, 0 for others
            # This creates valid logits that produce reasonable probabilities when passed through softmax
            sequence_logits = sequence_onehot * 10.0
        else:
            # Already logits
            sequence_logits = sequence
        
        # Get continuous sequence representation
        sequence_probs = self.sequence_repr(sequence_logits, landscape_idx, training=self.training)
        
        # Predict energy
        energies = self.energy_head(encoded_backbone, sequence_probs, mask)
        
        return energies


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def main():
    """Main training script entry point"""
    parser = argparse.ArgumentParser(description="Train energy model for protein design")
    
    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file (JSON)')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                       help='Directory for saving model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logging outputs')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='Training device (auto-detect if not specified)')
    
    # Override config options
    parser.add_argument('--batch_size', type=int, 
                       help='Override batch size from config')
    parser.add_argument('--learning_rate', type=float,
                       help='Override learning rate from config')
    parser.add_argument('--max_epochs', type=int,
                       help='Override max epochs from config')
    parser.add_argument('--resume_from', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['optimization']['learning_rate'] = args.learning_rate
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    if args.resume_from:
        config['resume_from_checkpoint'] = args.resume_from
    
    # Print configuration
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    print("-" * 50)
    
    # Initialize trainer
    trainer = EnergyModelTrainer(
        config=config,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Setup and train
    try:
        trainer.setup()
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()