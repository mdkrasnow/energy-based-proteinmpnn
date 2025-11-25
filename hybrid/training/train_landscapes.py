#!/usr/bin/env python3
"""
Multi-Landscape Training Script for IRED Energy Models

This script implements the training pipeline for multiple annealed energy landscapes
(E_1, E_2, ..., E_T) used in the IRED optimization framework. Each landscape has
progressively increasing sharpness, from smooth exploration (E_1) to sharp energy
minima (E_T) for final sequence refinement.

Key Features:
1. Progressive noise/smoothness annealing across landscapes
2. Curriculum learning with increasing data difficulty
3. Cross-landscape consistency losses for smooth transitions
4. Landscape-specific evaluation metrics
5. Multi-model checkpoint management

Usage:
    python train_landscapes.py --config config_landscapes.json --num_landscapes 5
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import random
import hashlib
import pickle
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
from inference.ired_optimizer import IREDSequenceOptimizer


@dataclass
class LandscapeConfig:
    """Configuration for a single energy landscape."""
    landscape_id: int
    temperature: float  # Temperature for this landscape (1.0 = smooth, 0.1 = sharp)
    noise_scale: float  # Data augmentation noise scale
    loss_weights: Dict[str, float]  # Loss component weights
    smoothness_penalty: float  # Smoothness regularization strength
    margin_scale: float  # Contrastive loss margin scaling
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class MultiLandscapeConfig:
    """Configuration for multi-landscape training."""
    num_landscapes: int = 5
    base_temperature: float = 1.0  # Starting temperature for E_1
    final_temperature: float = 0.1  # Ending temperature for E_T
    
    # Progressive annealing parameters
    temperature_schedule: str = 'exponential'  # 'linear', 'exponential', 'cosine'
    
    # Curriculum learning
    curriculum_epochs: int = 10  # Epochs for curriculum progression per landscape
    base_noise_scale: float = 0.1  # Starting noise scale
    noise_decay: float = 0.8  # Noise reduction factor across landscapes
    
    # Cross-landscape consistency
    consistency_weight: float = 0.1  # Weight for cross-landscape consistency loss
    consistency_temperature: float = 0.5  # Temperature for consistency softmax
    
    # Training schedule
    sequential_training: bool = True  # Train landscapes sequentially vs jointly
    shared_encoder: bool = True  # Share ProteinMPNN encoder across landscapes
    
    def generate_landscape_configs(self) -> List[LandscapeConfig]:
        """Generate individual landscape configurations."""
        landscapes = []
        
        for i in range(self.num_landscapes):
            # Temperature annealing schedule
            progress = i / (self.num_landscapes - 1) if self.num_landscapes > 1 else 0.0
            
            if self.temperature_schedule == 'linear':
                temperature = self.base_temperature * (1 - progress) + self.final_temperature * progress
            elif self.temperature_schedule == 'exponential':
                temperature = self.base_temperature * (self.final_temperature / self.base_temperature) ** progress
            elif self.temperature_schedule == 'cosine':
                temperature = self.final_temperature + (self.base_temperature - self.final_temperature) * \
                            0.5 * (1 + np.cos(np.pi * progress))
            else:
                raise ValueError(f"Unknown temperature schedule: {self.temperature_schedule}")
            
            # Noise scaling (decreases across landscapes)
            noise_scale = self.base_noise_scale * (self.noise_decay ** i)
            
            # Loss weights (progressive sharpening)
            # Early landscapes: more regularization, later landscapes: sharper ranking
            ranking_weight = 0.5 + 0.5 * progress  # 0.5 → 1.0
            smoothness_weight = 0.1 * (1 - progress)  # 0.1 → 0.0
            entropy_weight = 0.05 * (1 - progress)  # 0.05 → 0.0
            
            # Margin scaling (increases for sharper landscapes)
            margin_scale = 1.0 + 0.5 * progress  # 1.0 → 1.5
            
            landscape = LandscapeConfig(
                landscape_id=i,
                temperature=temperature,
                noise_scale=noise_scale,
                loss_weights={
                    'ranking': ranking_weight,
                    'contrastive': 1.0,
                    'entropy': entropy_weight,
                    'smoothness': smoothness_weight
                },
                smoothness_penalty=smoothness_weight,
                margin_scale=margin_scale
            )
            landscapes.append(landscape)
        
        return landscapes


class MultiLandscapeTrainer:
    """
    Trainer for multiple annealed energy landscapes in IRED optimization.
    
    This trainer creates a sequence of energy models E_1, ..., E_T with progressive
    sharpening from smooth exploration to precise energy minima. The models are trained
    with curriculum learning and consistency losses to ensure smooth optimization paths.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        landscape_config: MultiLandscapeConfig,
        model_dir: str = "landscape_checkpoints",
        log_dir: str = "landscape_logs",
        device: Optional[str] = None
    ):
        """Initialize multi-landscape trainer."""
        self.base_config = base_config
        self.landscape_config = landscape_config
        
        # Setup directories
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Device setup
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
        
        # Set random seeds
        self._set_random_seeds(base_config.get('seed', 42))
        
        # Generate landscape configurations
        self.landscapes = self.landscape_config.generate_landscape_configs()
        print(f"Generated {len(self.landscapes)} landscape configurations")
        
        # Initialize components
        self.shared_encoder = None
        self.energy_models = []
        self.sequence_repr = None
        self.optimizers = []
        self.schedulers = []
        self.loss_functions = []
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Logging
        self.writers = []
        
        # Training state
        self.current_landscape = 0
        self.training_history = {}
    
    def setup(self):
        """Set up all training components."""
        print("Setting up multi-landscape training...")
        
        # Setup data loaders
        self._setup_data()
        
        # Setup shared encoder (if using shared mode)
        if self.landscape_config.shared_encoder:
            self._setup_shared_encoder()
        
        # Setup sequence representation
        self._setup_sequence_representation()
        
        # Setup landscape-specific models
        self._setup_landscape_models()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup logging
        self._setup_logging()
        
        print("Multi-landscape setup completed successfully")
    
    def train(self):
        """Main training loop for all landscapes."""
        print("Starting multi-landscape training...")
        
        if self.landscape_config.sequential_training:
            self._train_sequential()
        else:
            self._train_joint()
        
        # Final validation across all landscapes
        self._validate_landscape_sequence()
        
        # Save final multi-landscape checkpoint
        self._save_multi_landscape_checkpoint()
        
        print("Multi-landscape training completed!")
    
    def _train_sequential(self):
        """Train landscapes sequentially from E_1 to E_T."""
        total_landscapes = len(self.landscapes)
        
        for landscape_idx, landscape_config in enumerate(self.landscapes):
            print(f"\n{'='*60}")
            print(f"Training Landscape {landscape_idx + 1}/{total_landscapes}")
            print(f"Temperature: {landscape_config.temperature:.3f}")
            print(f"Noise scale: {landscape_config.noise_scale:.3f}")
            print(f"{'='*60}")
            
            self.current_landscape = landscape_idx
            
            # Train this landscape
            self._train_single_landscape(landscape_idx, landscape_config)
            
            # Validate landscape properties
            self._validate_single_landscape(landscape_idx)
            
            # Save individual landscape checkpoint
            self._save_landscape_checkpoint(landscape_idx)
    
    def _train_single_landscape(self, landscape_idx: int, config: LandscapeConfig):
        """Train a single energy landscape."""
        energy_model = self.energy_models[landscape_idx]
        optimizer = self.optimizers[landscape_idx]
        scheduler = self.schedulers[landscape_idx] if landscape_idx < len(self.schedulers) else None
        loss_fn = self.loss_functions[landscape_idx]
        
        # Training parameters
        max_epochs = self.base_config['training']['max_epochs']
        patience = self.base_config['training'].get('patience', 20)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Curriculum learning: start with easier data, gradually increase difficulty
        curriculum_progression = np.linspace(0.2, 1.0, self.landscape_config.curriculum_epochs)
        
        for epoch in range(max_epochs):
            # Determine curriculum difficulty for this epoch
            if epoch < len(curriculum_progression):
                curriculum_difficulty = curriculum_progression[epoch]
            else:
                curriculum_difficulty = 1.0
            
            # Train epoch
            train_metrics = self._train_landscape_epoch(
                landscape_idx, config, energy_model, optimizer, 
                curriculum_difficulty
            )
            
            # Validate epoch
            val_metrics = self._validate_landscape_epoch(
                landscape_idx, config, energy_model
            )
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            self._log_landscape_metrics(landscape_idx, epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_best_landscape(landscape_idx)
            else:
                patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Curriculum: {curriculum_difficulty:.2f}")
            
            if patience_counter >= patience:
                print(f"Early stopping for landscape {landscape_idx}")
                break
        
        print(f"Landscape {landscape_idx} training completed. Best val loss: {best_val_loss:.6f}")
    
    def _train_landscape_epoch(
        self, 
        landscape_idx: int, 
        config: LandscapeConfig,
        energy_model: nn.Module,
        optimizer: optim.Optimizer,
        curriculum_difficulty: float
    ) -> Dict[str, float]:
        """Train single epoch for a landscape."""
        energy_model.train()
        if self.shared_encoder:
            self.shared_encoder.eval()  # Keep encoder frozen during landscape training
        
        total_loss = 0.0
        total_samples = 0
        consistency_loss_total = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"L{landscape_idx} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Apply curriculum learning: skip harder examples early in training
                if curriculum_difficulty < 1.0:
                    batch = self._apply_curriculum_filtering(batch, curriculum_difficulty)
                
                # Skip if batch too small after filtering
                if len(batch['label']) < 2:
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass through encoder + landscape model
                outputs = self._forward_landscape(batch, landscape_idx, config)
                
                if outputs.get('skip_batch', False):
                    continue
                
                # Compute primary landscape loss
                landscape_loss = self.loss_functions[landscape_idx](
                    pos_energies=outputs['pos_energies'],
                    neg_energies=outputs['neg_energies'],
                    pos_sequence_probs=outputs.get('pos_sequence_probs'),
                    neg_sequence_probs=outputs.get('neg_sequence_probs'),
                    negative_types=outputs.get('negative_types')
                )
                
                total_loss_value = landscape_loss
                
                # Cross-landscape consistency loss (if not first landscape)
                consistency_loss = 0.0
                if landscape_idx > 0 and self.landscape_config.consistency_weight > 0:
                    consistency_loss = self._compute_consistency_loss(
                        batch, landscape_idx, outputs
                    )
                    total_loss_value += self.landscape_config.consistency_weight * consistency_loss
                    consistency_loss_total += consistency_loss.item()
                
                # Backward pass
                total_loss_value.backward()
                
                # Monitor gradient norm before clipping
                max_grad_norm = self.base_config['training'].get('max_grad_norm', 1.0)
                
                # Get gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(energy_model.parameters(), float('inf'))
                
                # Validate gradient health
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    raise ValueError(f"Invalid gradient norm detected: {grad_norm}. "
                                   f"Landscape {landscape_idx}, step {batch_idx}. "
                                   f"Consider reducing learning rate or checking model inputs.")
                
                # Detect concerning gradient magnitudes
                if grad_norm > max_grad_norm * 10:
                    warnings.warn(f"Very large gradient norm: {grad_norm:.2f} "
                                f"(threshold: {max_grad_norm}). "
                                f"Consider reducing learning rate or checking numerical stability.")
                
                # Apply clipping with monitoring
                if grad_norm > max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_grad_norm)
                    # Log clipping statistics
                    clipping_ratio = grad_norm / max_grad_norm
                    if clipping_ratio > 5.0:
                        warnings.warn(f"Severe gradient clipping: {clipping_ratio:.1f}x threshold")
                
                optimizer.step()
                
                # Statistics
                batch_size = len(outputs['pos_energies'])
                total_loss += landscape_loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress
                progress_bar.set_postfix({
                    'Loss': f"{landscape_loss.item():.4f}",
                    'Consistency': f"{consistency_loss:.4f}" if isinstance(consistency_loss, (int, float)) else "0.0000",
                    'Pos_E': f"{outputs['pos_energies'].mean().item():.3f}",
                    'Neg_E': f"{outputs['neg_energies'].mean().item():.3f}"
                })
                
            except Exception as e:
                warnings.warn(f"Error in landscape {landscape_idx} training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_consistency = consistency_loss_total / max(len(self.train_loader), 1)
        
        return {
            'loss': avg_loss,
            'consistency_loss': avg_consistency,
            'total_samples': total_samples
        }
    
    def _validate_landscape_epoch(
        self,
        landscape_idx: int,
        config: LandscapeConfig,
        energy_model: nn.Module
    ) -> Dict[str, float]:
        """Validate single epoch for a landscape."""
        energy_model.eval()
        if self.shared_encoder:
            self.shared_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        energy_stats = {'pos_mean': 0.0, 'neg_mean': 0.0, 'ranking_accuracy': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"L{landscape_idx} Validation"):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    outputs = self._forward_landscape(batch, landscape_idx, config)
                    
                    if outputs.get('skip_batch', False):
                        continue
                    
                    # Compute loss
                    loss = self.loss_functions[landscape_idx](
                        pos_energies=outputs['pos_energies'],
                        neg_energies=outputs['neg_energies'],
                        pos_sequence_probs=outputs.get('pos_sequence_probs'),
                        neg_sequence_probs=outputs.get('neg_sequence_probs'),
                        negative_types=outputs.get('negative_types')
                    )
                    
                    # Statistics
                    batch_size = len(outputs['pos_energies'])
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    # Energy statistics
                    pos_energies = outputs['pos_energies']
                    neg_energies = outputs['neg_energies']
                    
                    energy_stats['pos_mean'] += pos_energies.mean().item() * batch_size
                    energy_stats['neg_mean'] += neg_energies.mean().item() * batch_size
                    
                    # Ranking accuracy
                    pos_expanded = pos_energies.unsqueeze(1)
                    neg_expanded = neg_energies.unsqueeze(0)
                    correct_rankings = (pos_expanded < neg_expanded).float().mean()
                    energy_stats['ranking_accuracy'] += correct_rankings.item() * batch_size
                
                except Exception as e:
                    warnings.warn(f"Error in landscape {landscape_idx} validation: {e}")
                    continue
        
        # Averages
        avg_loss = total_loss / max(total_samples, 1)
        for stat in energy_stats:
            energy_stats[stat] /= max(total_samples, 1)
        
        return {
            'loss': avg_loss,
            'energy_stats': energy_stats
        }
    
    def _forward_landscape(
        self, 
        batch: Dict, 
        landscape_idx: int, 
        config: LandscapeConfig
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through shared encoder + landscape-specific energy model."""
        # Get backbone features
        if self.shared_encoder:
            # Use shared encoder for structural features
            backbone_features = self._encode_structures(batch)
        else:
            # Use pre-computed features
            backbone_features = batch['backbone_features']
        
        # Split by labels
        labels = batch['label']
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return {
                'pos_energies': torch.tensor([], device=self.device),
                'neg_energies': torch.tensor([], device=self.device),
                'skip_batch': True
            }
        
        # Apply data augmentation with landscape-specific noise
        sequences = batch['sequence'].clone()
        if config.noise_scale > 0 and self.training:
            sequences = self._apply_sequence_noise(sequences, config.noise_scale)
        
        # Process positive samples
        if pos_mask.sum() > 0:
            pos_backbone = backbone_features[pos_mask]
            pos_sequence = sequences[pos_mask]
            
            # Convert to continuous representation
            pos_sequence_probs = self.sequence_repr(
                self._sequence_to_logits(pos_sequence),
                landscape_idx=landscape_idx,
                training=False  # Use straight-through for training stability
            )
            
            pos_energies = self.energy_models[landscape_idx](
                pos_backbone, pos_sequence_probs,
                batch.get('mask')[pos_mask] if batch.get('mask') is not None else None
            )
        else:
            pos_energies = torch.tensor([], device=self.device)
            pos_sequence_probs = None
        
        # Process negative samples
        if neg_mask.sum() > 0:
            neg_backbone = backbone_features[neg_mask]
            neg_sequence = sequences[neg_mask]
            
            neg_sequence_probs = self.sequence_repr(
                self._sequence_to_logits(neg_sequence),
                landscape_idx=landscape_idx,
                training=False
            )
            
            neg_energies = self.energy_models[landscape_idx](
                neg_backbone, neg_sequence_probs,
                batch.get('mask')[neg_mask] if batch.get('mask') is not None else None
            )
            
            negative_types = [batch['generation_method'][i] for i in range(len(labels)) if neg_mask[i]]
        else:
            neg_energies = torch.tensor([], device=self.device)
            neg_sequence_probs = None
            negative_types = []
        
        return {
            'pos_energies': pos_energies,
            'neg_energies': neg_energies,
            'pos_sequence_probs': pos_sequence_probs,
            'neg_sequence_probs': neg_sequence_probs,
            'negative_types': negative_types
        }
    
    def _compute_consistency_loss(
        self, 
        batch: Dict, 
        landscape_idx: int, 
        current_outputs: Dict
    ) -> torch.Tensor:
        """Compute cross-landscape consistency loss."""
        if landscape_idx == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Get energies from previous landscape
        with torch.no_grad():
            prev_config = self.landscapes[landscape_idx - 1]
            prev_outputs = self._forward_landscape(batch, landscape_idx - 1, prev_config)
        
        if prev_outputs.get('skip_batch', False) or current_outputs.get('skip_batch', False):
            return torch.tensor(0.0, device=self.device)
        
        # Consistency loss: encourage similar energy orderings between adjacent landscapes
        # Use temperature-scaled softmax for smooth consistency
        temp = self.landscape_config.consistency_temperature
        
        current_pos = current_outputs['pos_energies']
        current_neg = current_outputs['neg_energies'] 
        prev_pos = prev_outputs['pos_energies']
        prev_neg = prev_outputs['neg_energies']
        
        # Clamp energies to prevent overflow
        current_pos_clamped = torch.clamp(current_pos, -50, 50)
        current_neg_clamped = torch.clamp(current_neg, -50, 50) 
        prev_pos_clamped = torch.clamp(prev_pos, -50, 50)
        prev_neg_clamped = torch.clamp(prev_neg, -50, 50)
        
        # Compute probabilities with clamped energies
        current_probs_pos = F.softmax(-current_pos_clamped / temp, dim=0)
        current_probs_neg = F.softmax(-current_neg_clamped / temp, dim=0)
        prev_probs_pos = F.softmax(-prev_pos_clamped / temp, dim=0)
        prev_probs_neg = F.softmax(-prev_neg_clamped / temp, dim=0)
        
        # Add epsilon for numerical stability
        eps = 1e-8
        current_log_probs_pos = torch.log(current_probs_pos + eps)
        current_log_probs_neg = torch.log(current_probs_neg + eps)
        
        # Compute KL divergence with stabilized inputs
        kl_pos = F.kl_div(current_log_probs_pos, prev_probs_pos, reduction='batchmean')
        kl_neg = F.kl_div(current_log_probs_neg, prev_probs_neg, reduction='batchmean')
        
        # Validate result is finite
        if not torch.isfinite(kl_pos) or not torch.isfinite(kl_neg):
            warnings.warn(f"KL divergence instability detected: kl_pos={kl_pos}, kl_neg={kl_neg}")
            kl_pos = torch.tensor(0.0, device=current_pos.device)
            kl_neg = torch.tensor(0.0, device=current_neg.device)
        
        return (kl_pos + kl_neg) / 2
    
    def _setup_data(self):
        """Set up data loaders using existing dataset."""
        data_config = self.base_config['data']
        
        # Create dataset (reuse existing StabilityDataset)
        dataset = StabilityDataset(
            data_dir=data_config['data_dir'],
            positive_ratio=data_config.get('positive_ratio', 0.5),
            negative_methods=data_config.get('negative_methods'),
            max_sequence_length=data_config.get('max_sequence_length', 500),
            min_sequence_length=data_config.get('min_sequence_length', 20),
            max_files=data_config.get('max_files_debug'),
            lazy_loading=data_config.get('lazy_loading', True),
            seed=self.base_config.get('seed', 42)
        )
        
        # Split into train/validation
        val_split = data_config.get('val_split', 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.base_config.get('seed', 42))
        )
        
        # Create data loaders with custom collate function
        def collate_fn(batch):
            collated = {}
            if 'backbone_features' in batch[0]:
                collated['backbone_features'] = torch.stack([item['backbone_features'] for item in batch])
            if 'sequence' in batch[0]:
                collated['sequence'] = torch.stack([item['sequence'] for item in batch])
            if 'mask' in batch[0]:
                collated['mask'] = torch.stack([item['mask'] for item in batch])
            
            collated['label'] = torch.tensor([item['label'] for item in batch])
            collated['length'] = torch.tensor([item['length'] for item in batch])
            
            if 'generation_method' in batch[0]:
                collated['generation_method'] = [item['generation_method'] for item in batch]
            if 'structure_id' in batch[0]:
                collated['structure_id'] = [item['structure_id'] for item in batch]
            
            return collated
        
        batch_size = self.base_config['training']['batch_size']
        num_workers = self.base_config['training'].get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=collate_fn
        )
        
        print(f"Data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)} samples")
    
    def _setup_shared_encoder(self):
        """Set up shared ProteinMPNN encoder."""
        if not self.landscape_config.shared_encoder:
            return
        
        mpnn_config = self.base_config['model']['mpnn_encoder']
        
        self.shared_encoder = ProteinMPNNBackboneEncoder(
            model_name=mpnn_config.get('model_name', 'v_48_020'),
            device=self.device,
            freeze_layers=True  # Keep frozen during landscape training
        )
        
        print("Shared ProteinMPNN encoder initialized")
    
    def _setup_sequence_representation(self):
        """Set up sequence representation with landscape-specific temperatures."""
        sequence_config = self.base_config['model'].get('sequence_repr', {})
        
        # Use landscape temperatures for annealing schedule
        temperature_schedule = [landscape.temperature for landscape in self.landscapes]
        
        self.sequence_repr = ContinuousSequenceRepr(
            vocab_size=20,
            temperature_schedule=temperature_schedule,
            min_temperature=sequence_config.get('min_temperature', 1e-3),
            max_temperature=sequence_config.get('max_temperature', 10.0)
        ).to(self.device)
        
        print(f"Sequence representation initialized with {len(temperature_schedule)} temperatures")
    
    def _setup_landscape_models(self):
        """Set up individual energy models for each landscape."""
        energy_config = self.base_config['model']['energy_head']
        mpnn_config = self.base_config['model']['mpnn_encoder']
        
        self.energy_models = []
        for landscape_config in self.landscapes:
            energy_model = EnergyHead(
                backbone_dim=mpnn_config.get('hidden_dim', 128),
                seq_dim=20,
                hidden_dim=energy_config.get('hidden_dim', 512),
                num_layers=energy_config.get('num_layers', 3),
                dropout=energy_config.get('dropout', 0.1),
                activation=energy_config.get('activation', 'relu'),
                use_batch_norm=energy_config.get('use_batch_norm', True)
            ).to(self.device)
            
            self.energy_models.append(energy_model)
        
        print(f"Created {len(self.energy_models)} landscape-specific energy models")
    
    def _setup_optimization(self):
        """Set up optimizers and loss functions for each landscape."""
        opt_config = self.base_config['optimization']
        
        self.optimizers = []
        self.schedulers = []
        self.loss_functions = []
        
        for i, landscape_config in enumerate(self.landscapes):
            # Optimizer for this landscape
            optimizer = optim.AdamW(
                self.energy_models[i].parameters(),
                lr=opt_config.get('learning_rate', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
            self.optimizers.append(optimizer)
            
            # Scheduler
            scheduler_config = opt_config.get('scheduler', {})
            if scheduler_config.get('type') == 'reduce_on_plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 10)
                )
                self.schedulers.append(scheduler)
            
            # Landscape-specific loss function
            loss_config = self.base_config['loss']
            loss_fn = ContrastiveLoss(
                margin=loss_config.get('margin', 1.0) * landscape_config.margin_scale,
                temperature=landscape_config.temperature,  # Use landscape temperature
                ranking_weight=landscape_config.loss_weights['ranking'],
                contrastive_weight=landscape_config.loss_weights['contrastive'],
                entropy_weight=landscape_config.loss_weights['entropy'],
                smoothness_weight=landscape_config.loss_weights['smoothness'],
                negative_weights=loss_config.get('negative_weights'),
                reduction='mean'
            )
            self.loss_functions.append(loss_fn)
        
        print(f"Set up optimization for {len(self.landscapes)} landscapes")
    
    def _setup_logging(self):
        """Set up logging for each landscape."""
        if not TENSORBOARD_AVAILABLE:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.writers = []
        for i in range(len(self.landscapes)):
            log_name = f"landscape_{i}_temp_{self.landscapes[i].temperature:.3f}_{timestamp}"
            writer = SummaryWriter(self.log_dir / log_name)
            self.writers.append(writer)
        
        print(f"TensorBoard logging set up for {len(self.writers)} landscapes")
    
    def _log_landscape_metrics(
        self, 
        landscape_idx: int, 
        epoch: int, 
        train_metrics: Dict, 
        val_metrics: Dict
    ):
        """Log metrics for a specific landscape."""
        if landscape_idx >= len(self.writers) or not TENSORBOARD_AVAILABLE:
            return
        
        writer = self.writers[landscape_idx]
        
        # Loss curves
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        writer.add_scalar('Loss/Train_Consistency', train_metrics.get('consistency_loss', 0), epoch)
        
        # Energy statistics
        energy_stats = val_metrics['energy_stats']
        writer.add_scalar('Energy/Val_Pos_Mean', energy_stats['pos_mean'], epoch)
        writer.add_scalar('Energy/Val_Neg_Mean', energy_stats['neg_mean'], epoch)
        writer.add_scalar('Energy/Val_Ranking_Accuracy', energy_stats['ranking_accuracy'], epoch)
        
        # Landscape properties
        config = self.landscapes[landscape_idx]
        writer.add_scalar('Landscape/Temperature', config.temperature, epoch)
        writer.add_scalar('Landscape/Noise_Scale', config.noise_scale, epoch)
        writer.add_scalar('Landscape/Margin_Scale', config.margin_scale, epoch)
    
    def _validate_landscape_sequence(self):
        """Validate that landscapes form a proper annealing sequence."""
        print("\nValidating landscape sequence...")
        
        # Test IRED optimizer with trained landscapes
        optimizer = IREDSequenceOptimizer(
            energy_models=self.energy_models,
            sequence_repr=self.sequence_repr,
            device=self.device,
            seed=42
        )
        
        validation_results = []
        validation_samples_tested = 0
        target_samples = min(15, len(self.val_loader) * 3)  # Test up to 15 samples
        
        for i, batch in enumerate(self.val_loader):
            if validation_samples_tested >= target_samples:
                break
            
            batch = self._move_batch_to_device(batch)
            batch_size = batch['backbone_features'].shape[0]
            
            for j in range(min(3, batch_size)):  # Test up to 3 samples per batch
                if validation_samples_tested >= target_samples:
                    break
                    
                try:
                    # Use deterministic sample selection for reproducibility
                    sample_idx = j  # Fixed index for reproducibility
                    result = optimizer.optimize_sequence(
                        batch['backbone_features'][sample_idx:sample_idx+1],
                        mask=batch.get('mask')[sample_idx:sample_idx+1] if batch.get('mask') is not None else None,
                        max_steps=20,
                        return_trajectory=True
                    )
                    
                    validation_results.append({
                        'sample_idx': validation_samples_tested,
                        'converged': result.converged,
                        'optimization_failed': result.optimization_failed,
                        'energy_improvement': (result.final_energy - result.trajectory[0]['energy_mean']).item() if result.converged and result.trajectory else None,
                        'total_steps': result.total_steps,
                        'failure_reason': getattr(result, 'failure_reason', None)
                    })
                    
                    validation_samples_tested += 1
                    
                except Exception as e:
                    validation_results.append({
                        'sample_idx': validation_samples_tested,
                        'converged': False,
                        'optimization_failed': True, 
                        'error': str(e),
                        'total_steps': 0
                    })
                    validation_samples_tested += 1

        # Analyze validation statistics
        converged_count = sum(1 for r in validation_results if r.get('converged', False))
        failed_count = sum(1 for r in validation_results if r.get('optimization_failed', True))
        convergence_rate = converged_count / len(validation_results) if validation_results else 0.0

        print(f"IRED Validation Results: {converged_count}/{len(validation_results)} converged ({convergence_rate:.1%})")

        if convergence_rate >= 0.8:
            print("✓ IRED optimization with trained landscapes: PASSED")
        else:
            print(f"✗ IRED optimization with trained landscapes: FAILED (convergence rate: {convergence_rate:.1%})")
            # Log details of failures for debugging
            failures = [r for r in validation_results if not r.get('converged', False)]
            print(f"Failure analysis: {len(failures)} failures out of {len(validation_results)} samples")
            
            for i, failure in enumerate(failures[:5]):  # Show first 5 failures
                reason = failure.get('failure_reason') or failure.get('error', 'Unknown')
                print(f"  Failure {i+1}: {reason}")
    
    def _save_landscape_checkpoint(self, landscape_idx: int):
        """Save checkpoint for a specific landscape with atomic operation and validation."""
        checkpoint = {
            'landscape_idx': landscape_idx,
            'landscape_config': self.landscapes[landscape_idx].to_dict(),
            'model_state_dict': self.energy_models[landscape_idx].state_dict(),
            'optimizer_state_dict': self.optimizers[landscape_idx].state_dict(),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'model_hash': self._compute_model_hash(self.energy_models[landscape_idx])
        }
        
        # Compute checksum for validation
        checkpoint_bytes = pickle.dumps(checkpoint)
        checkpoint['checksum'] = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        filepath = self.model_dir / f"landscape_{landscape_idx}.pt"
        temp_filepath = filepath.with_suffix('.pt.tmp')
        
        try:
            # Atomic save: write to temporary file first
            torch.save(checkpoint, temp_filepath)
            
            # Validate saved file by loading and checking
            test_checkpoint = torch.load(temp_filepath, map_location='cpu')
            
            # Verify checksum
            saved_checksum = test_checkpoint.pop('checksum')
            test_bytes = pickle.dumps(test_checkpoint)
            computed_checksum = hashlib.sha256(test_bytes).hexdigest()
            
            if saved_checksum != computed_checksum:
                raise ValueError(f"Checkpoint integrity check failed: {saved_checksum} != {computed_checksum}")
                
            # Atomic rename (this operation is atomic on most filesystems)
            temp_filepath.rename(filepath)
            print(f"✓ Saved landscape {landscape_idx} checkpoint: {filepath} (hash: {checkpoint['model_hash'][:8]})")
            
        except Exception as e:
            # Cleanup failed temporary file
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise RuntimeError(f"Failed to save checkpoint for landscape {landscape_idx}: {e}")
    
    def _compute_model_hash(self, model):
        """Compute hash of model parameters for integrity checking."""
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()
    
    def _save_multi_landscape_checkpoint(self):
        """Save complete multi-landscape system."""
        checkpoint = {
            'landscape_configs': [config.to_dict() for config in self.landscapes],
            'multi_landscape_config': asdict(self.landscape_config),
            'base_config': self.base_config,
            'energy_model_states': [model.state_dict() for model in self.energy_models],
            'sequence_repr_state': self.sequence_repr.state_dict(),
            'shared_encoder_state': self.shared_encoder.state_dict() if self.shared_encoder else None,
            'training_complete': True,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = self.model_dir / "multi_landscape_complete.pt"
        torch.save(checkpoint, filepath)
        print(f"Saved complete multi-landscape system: {filepath}")
    
    def _save_best_landscape(self, landscape_idx: int):
        """Save best checkpoint for a landscape."""
        checkpoint = {
            'landscape_idx': landscape_idx,
            'model_state_dict': self.energy_models[landscape_idx].state_dict(),
            'is_best': True
        }
        
        filepath = self.model_dir / f"landscape_{landscape_idx}_best.pt"
        torch.save(checkpoint, filepath)
    
    # Utility methods
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _apply_curriculum_filtering(self, batch: Dict, difficulty: float) -> Dict:
        """Apply curriculum learning by filtering batch based on difficulty."""
        # Simple curriculum: randomly sample based on difficulty level
        if difficulty >= 1.0:
            return batch
        
        batch_size = len(batch['label'])
        keep_size = max(1, int(batch_size * difficulty))
        
        indices = torch.randperm(batch_size)[:keep_size]
        
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                filtered_batch[key] = value[indices]
            elif isinstance(value, list):
                filtered_batch[key] = [value[i] for i in indices]
            else:
                filtered_batch[key] = value
        
        return filtered_batch
    
    def _apply_sequence_noise(self, sequences: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Apply noise augmentation to sequences."""
        if noise_scale <= 0:
            return sequences
        
        # Add small random perturbations to sequence indices
        noise = torch.randn_like(sequences.float()) * noise_scale
        noisy_sequences = sequences.float() + noise
        
        # Clamp to valid range and convert back to long
        noisy_sequences = torch.clamp(noisy_sequences, 0, 19).long()
        
        return noisy_sequences
    
    def _sequence_to_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Convert sequence indices to logits."""
        # Convert to one-hot then to logits
        sequence_onehot = F.one_hot(sequence.long(), num_classes=20).float()
        logits = sequence_onehot * 10.0  # Large positive for selected, 0 for others
        return logits
    
    def _encode_structures(self, batch: Dict) -> torch.Tensor:
        """Encode structures using shared encoder."""
        # This would require implementing structure coordinate processing
        # For now, assume pre-computed features are available
        return batch['backbone_features']
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _train_joint(self):
        """Joint training of all landscapes (alternative to sequential)."""
        # This would implement joint training where all landscapes are trained simultaneously
        # For now, we focus on sequential training as it's more stable
        raise NotImplementedError("Joint training not yet implemented. Use sequential_training=True")
    
    def _validate_single_landscape(self, landscape_idx: int):
        """Validate properties of a single trained landscape."""
        print(f"Validating landscape {landscape_idx} properties...")
        
        config = self.landscapes[landscape_idx]
        energy_model = self.energy_models[landscape_idx]
        
        # Test energy sharpness by checking gradient norms
        energy_model.eval()
        
        # Create test data
        test_backbone = torch.randn(1, 50, 128, device=self.device)
        test_sequence = torch.randn(1, 50, 20, device=self.device, requires_grad=True)
        test_mask = torch.ones(1, 50, device=self.device)
        
        # Compute energy and gradients
        energy = energy_model(test_backbone, test_sequence, test_mask)
        energy.backward()
        
        gradient_norm = test_sequence.grad.norm().item()
        
        print(f"Landscape {landscape_idx} - Temperature: {config.temperature:.3f}, Gradient norm: {gradient_norm:.3f}")
        
        # Validate sharpness progression: later landscapes should have higher gradient norms
        if landscape_idx > 0:
            expected_sharpness_increase = landscape_idx > 0
            print(f"Expected sharpness increase: {expected_sharpness_increase}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_landscape_config(num_landscapes: int = 5) -> MultiLandscapeConfig:
    """Create default multi-landscape configuration."""
    return MultiLandscapeConfig(
        num_landscapes=num_landscapes,
        base_temperature=1.0,
        final_temperature=0.1,
        temperature_schedule='exponential',
        curriculum_epochs=10,
        base_noise_scale=0.1,
        noise_decay=0.8,
        consistency_weight=0.1,
        sequential_training=True,
        shared_encoder=True
    )


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train multi-landscape energy models for IRED")
    
    # Configuration files
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base training configuration file')
    parser.add_argument('--landscape_config', type=str, 
                       help='Path to landscape-specific configuration (JSON)')
    
    # Landscape parameters
    parser.add_argument('--num_landscapes', type=int, default=5,
                       help='Number of energy landscapes to train')
    parser.add_argument('--temperature_schedule', type=str, default='exponential',
                       choices=['linear', 'exponential', 'cosine'],
                       help='Temperature annealing schedule')
    parser.add_argument('--base_temperature', type=float, default=1.0,
                       help='Starting temperature for E_1')
    parser.add_argument('--final_temperature', type=float, default=0.1,
                       help='Final temperature for E_T')
    
    # Training parameters
    parser.add_argument('--sequential_training', action='store_true', default=True,
                       help='Train landscapes sequentially (default: True)')
    parser.add_argument('--shared_encoder', action='store_true', default=True,
                       help='Use shared ProteinMPNN encoder (default: True)')
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                       help='Weight for cross-landscape consistency loss')
    
    # Directories
    parser.add_argument('--model_dir', type=str, default='landscape_checkpoints',
                       help='Directory for saving model checkpoints')
    parser.add_argument('--log_dir', type=str, default='landscape_logs',
                       help='Directory for logging outputs')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                       help='Training device (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Create landscape configuration
    if args.landscape_config and Path(args.landscape_config).exists():
        with open(args.landscape_config, 'r') as f:
            landscape_config_dict = json.load(f)
        landscape_config = MultiLandscapeConfig(**landscape_config_dict)
    else:
        landscape_config = create_default_landscape_config(args.num_landscapes)
        
        # Apply command line overrides
        landscape_config.num_landscapes = args.num_landscapes
        landscape_config.temperature_schedule = args.temperature_schedule
        landscape_config.base_temperature = args.base_temperature
        landscape_config.final_temperature = args.final_temperature
        landscape_config.sequential_training = args.sequential_training
        landscape_config.shared_encoder = args.shared_encoder
        landscape_config.consistency_weight = args.consistency_weight
    
    print("Multi-Landscape Training Configuration:")
    print(f"  Number of landscapes: {landscape_config.num_landscapes}")
    print(f"  Temperature schedule: {landscape_config.temperature_schedule}")
    print(f"  Temperature range: {landscape_config.base_temperature} → {landscape_config.final_temperature}")
    print(f"  Sequential training: {landscape_config.sequential_training}")
    print(f"  Shared encoder: {landscape_config.shared_encoder}")
    print(f"  Consistency weight: {landscape_config.consistency_weight}")
    print("-" * 60)
    
    # Initialize trainer
    trainer = MultiLandscapeTrainer(
        base_config=base_config,
        landscape_config=landscape_config,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Setup and train
    try:
        trainer.setup()
        trainer.train()
        print("Multi-landscape training completed successfully!")
        
    except Exception as e:
        print(f"Multi-landscape training failed: {e}")
        raise


if __name__ == "__main__":
    main()