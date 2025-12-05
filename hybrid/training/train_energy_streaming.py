"""
Training script for energy-based ProteinMPNN with streaming data infrastructure.

This script implements streaming training for protein structure energy prediction using:
- StreamingProteinDataset for infinite data iteration
- PDBCache for efficient structure management  
- Dynamic negative sampling strategies
- A100-optimized performance configurations
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available, logging to files only")

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hybrid.data import StreamingProteinDataset, PDBCache, PDBManager
from hybrid.data.vocab import AMINO_ACID_TO_IDX, AMINO_ACID_ALPHABET
from hybrid.models import EnergyBasedProteinMPNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streaming_training.log')
    ]
)
logger = logging.getLogger(__name__)


class StreamingTrainer:
    """
    Trainer class for streaming energy-based ProteinMPNN training.
    
    Handles the complete training pipeline with streaming data infrastructure,
    A100 optimizations, and comprehensive monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize streaming trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.setup_streaming_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.setup_monitoring()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_validation_loss = float('inf')
        
        logger.info(f"StreamingTrainer initialized on device: {self.device}")
        
    def setup_directories(self):
        """Setup training directories."""
        self.cache_dir = Path(os.path.expandvars(self.config['streaming']['cache_dir']))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
    def setup_streaming_dataset(self):
        """Initialize streaming dataset and data infrastructure."""
        logger.info("Setting up streaming dataset...")
        
        # Initialize PDB cache
        cache_config = self.config['cache_config']['pdb_cache']
        self.cache = PDBCache(
            cache_dir=self.cache_dir / "pdb_cache",
            max_memory_mb=cache_config['max_memory_mb'],
            max_disk_gb=cache_config['max_disk_gb'],
            max_concurrent_downloads=cache_config['a100_optimization']['concurrent_downloads']
        )
        
        # Initialize PDB manager
        self.pdb_manager = PDBManager(
            data_sources=self.config['data_sources'],
            metadata_db_path=Path(os.path.expandvars(
                self.config['cache_config']['metadata_cache']['db_path']
            )),
            quality_filters=self.config.get('quality_filters', {})
        )
        
        # Create streaming dataset
        streaming_config = self.config['streaming']
        data_config = self.config['data']
        
        self.dataset = StreamingProteinDataset(
            data_sources=self.config['data_sources'],
            cache_dir=self.cache_dir,
            batch_size=self.config['training']['batch_size'],
            prefetch_factor=streaming_config['prefetch_factor'],
            num_workers=streaming_config['num_workers'],
            negative_sampling_ratio=data_config['negative_sampling_ratio'],
            max_sequence_length=data_config['max_sequence_length'],
            min_sequence_length=data_config['min_sequence_length'],
            augmentation_config=data_config.get('augmentation', {}),
            seed=self.config.get('seed', 42),
            enable_timing=True
        )
        
        # Apply A100 optimizations if enabled
        if streaming_config['a100_optimizations']['enabled']:
            logger.info("Applying A100 streaming optimizations...")
            optimization_results = self.dataset.optimize_for_a100_streaming()
            logger.info(f"A100 optimizations applied: {optimization_results}")
            
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=None,  # Streaming dataset handles batching
            num_workers=0,    # Streaming dataset has its own workers
            pin_memory=True,
            persistent_workers=False
        )
        
        logger.info("Streaming dataset setup complete")
        
    def setup_model(self):
        """Initialize model."""
        logger.info("Setting up model...")
        
        # Initialize energy-based ProteinMPNN model
        model_config = self.config['model']
        self.model = EnergyBasedProteinMPNN(
            mpnn_config=model_config['mpnn_encoder'],
            energy_head_config=model_config['energy_head'],
            sequence_repr_config=model_config['sequence_repr']
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Enable mixed precision if configured
        self.scaler = None
        if self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
            
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        opt_config = self.config['optimization']
        
        # Initialize optimizer
        if opt_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
            
        # Initialize scheduler
        scheduler_config = opt_config['scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience']
            )
        else:
            self.scheduler = None
            
        logger.info(f"Optimizer and scheduler setup complete")
        
    def setup_monitoring(self):
        """Setup monitoring and logging."""
        # TensorBoard
        if self.config['monitoring']['tensorboard']['enabled']:
            log_dir = Path(os.path.expandvars(
                self.config['monitoring']['tensorboard']['log_dir']
            ))
            log_dir.mkdir(parents=True, exist_ok=True)
            if TENSORBOARD_AVAILABLE:
                self.writer = SummaryWriter(log_dir=log_dir)
            else:
                self.writer = None
            logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
            
        # Monitoring flags
        self.monitor_cache = self.config['monitoring']['metrics']['track_cache_stats']
        self.monitor_memory = self.config['monitoring']['metrics']['track_memory_usage']
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Get iterator from streaming dataset
        data_iter = iter(self.dataloader)
        
        try:
            for batch_idx in range(self.config['training']['max_steps']):
                try:
                    # Get next sample from streaming dataset
                    sample = next(data_iter)
                    
                    if sample is None:
                        continue
                        
                    # Move to device
                    sample = self.move_to_device(sample)
                    
                    # Forward pass with mixed precision if enabled
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            loss = self.compute_loss(sample)
                        
                        # Backward pass
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if self.config['training'].get('max_grad_norm'):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['max_grad_norm']
                            )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss = self.compute_loss(sample)
                        loss.backward()
                        
                        # Gradient clipping
                        if self.config['training'].get('max_grad_norm'):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['max_grad_norm']
                            )
                        
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Update tracking
                    epoch_loss += loss.item()
                    num_batches += 1
                    self.step += 1
                    
                    # Logging
                    if self.step % self.config['training']['log_frequency'] == 0:
                        self.log_training_step(loss.item(), sample)
                    
                    # Monitoring
                    if self.step % self.config['training']['eval_frequency'] == 0:
                        self.log_monitoring_stats()
                    
                    # Check stopping criteria
                    if self.step >= self.config['training']['max_steps']:
                        break
                        
                except StopIteration:
                    logger.info("Data iterator exhausted, continuing...")
                    data_iter = iter(self.dataloader)
                    continue
                except Exception as e:
                    logger.warning(f"Error in training step: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(num_batches, 1)
        
        logger.info(f"Epoch {self.epoch} completed: avg_loss={avg_loss:.6f}, "
                   f"steps={num_batches}, time={epoch_time:.2f}s")
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'epoch_time': epoch_time
        }
        
    def compute_loss(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss for a sample."""
        # Extract inputs
        sequence = sample['sequence']
        coordinates = sample['coordinates']
        mask = sample['mask']
        label = sample['label']
        
        # Convert sequence string to tensor if needed (fix for string logits error)
        if isinstance(sequence, str):
            # CRITICAL FIX: Use canonical ProteinMPNN alphabet from shared vocab module
            # This fixes the data corruption bug where streaming data uses ProteinMPNN order
            # but training code was using alphabetical order (ACDEFGHIKLMNPQRSTVWY)
            aa_to_idx = AMINO_ACID_TO_IDX  # ProteinMPNN standard: ARNDCQEGHILKMFPSTWYV
            
            if not sequence:
                sequence = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                # Convert string sequence to indices with logged placeholder mapping
                indices = []
                for i, aa in enumerate(sequence):
                    aa_upper = aa.upper()
                    if aa_upper in aa_to_idx:
                        indices.append(aa_to_idx[aa_upper])
                    else:
                        # CRITICAL FIX: Map unknown amino acids to Alanine placeholder
                        # This preserves sequence length and maintains sequence-coordinate alignment
                        logger.warning(
                            f"Unknown amino acid '{aa}' at position {i} in sequence, "
                            f"mapping to Alanine (A) placeholder"
                        )
                        indices.append(aa_to_idx['A'])  # Alanine = index 0 in ProteinMPNN order
                sequence = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        # CRITICAL VALIDATION: Ensure sequence-coordinate alignment after conversion
        # This catches data corruption early with clear error messages
        if isinstance(sequence, torch.Tensor) and len(sequence.shape) > 0:
            seq_length = sequence.shape[0]
            
            # Validate against coordinates dimensions
            if coordinates is not None and isinstance(coordinates, torch.Tensor):
                if len(coordinates.shape) >= 3:  # Expected: [L, atoms, 3] or [B, L, atoms, 3]
                    coord_length = coordinates.shape[-3]  # Length dimension
                elif len(coordinates.shape) == 2:  # Simplified: [L, 3]
                    coord_length = coordinates.shape[0]  # Length dimension
                else:
                    coord_length = None  # Skip validation for unexpected shapes
                
                if coord_length is not None:
                    if seq_length != coord_length:
                        raise ValueError(
                            f"Sequence-coordinate dimension mismatch: "
                            f"sequence length {seq_length} != coordinates length {coord_length}. "
                            f"This indicates data corruption in sequence conversion."
                        )
            
            # Validate against mask dimensions  
            if mask is not None and isinstance(mask, torch.Tensor):
                if len(mask.shape) >= 1:  # Expected: [L] or [B, L]
                    mask_length = mask.shape[-1]  # Length dimension
                    if seq_length != mask_length:
                        raise ValueError(
                            f"Sequence-mask dimension mismatch: "
                            f"sequence length {seq_length} != mask length {mask_length}. "
                            f"This indicates data corruption in sequence conversion."
                        )
        
        # Forward pass through model
        energy_prediction = self.model(sequence, coordinates, mask)
        
        # Compute energy-based loss
        loss_config = self.config['loss']
        
        if label == 1:  # Positive sample (stable)
            # Minimize energy for stable structures
            loss = energy_prediction.mean()
        else:  # Negative sample (unstable)
            # Maximize energy for unstable structures
            margin = loss_config['margin']
            loss = torch.relu(margin - energy_prediction).mean()
            
        return loss
        
    def move_to_device(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Move sample tensors to training device."""
        if torch.cuda.is_available():
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.cuda(non_blocking=True)
        return sample
        
    def log_training_step(self, loss: float, sample: Dict[str, Any]):
        """Log training step information."""
        logger.info(f"Step {self.step}: loss={loss:.6f}, "
                   f"sample_type={sample.get('source_type', 'unknown')}, "
                   f"sequence_length={sample.get('length', 0)}")
        
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.step)
            self.writer.add_scalar('train/sequence_length', 
                                  sample.get('length', 0), self.step)
            
    def log_monitoring_stats(self):
        """Log monitoring and performance statistics."""
        if self.monitor_cache:
            # Cache statistics
            cache_stats = self.cache.get_performance_metrics()
            logger.info(f"Cache hit rate: {cache_stats['cache_performance']['hit_rate']:.2%}")
            
            if self.writer:
                self.writer.add_scalar('cache/hit_rate', 
                                      cache_stats['cache_performance']['hit_rate'], 
                                      self.step)
                self.writer.add_scalar('cache/memory_usage_mb',
                                      cache_stats['resource_utilization']['memory_cache']['used_mb'],
                                      self.step)
                                      
        if self.monitor_memory and torch.cuda.is_available():
            # GPU memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            logger.info(f"GPU memory: {memory_allocated:.2f}GB allocated, "
                       f"{memory_reserved:.2f}GB reserved")
                       
            if self.writer:
                self.writer.add_scalar('gpu/memory_allocated_gb', memory_allocated, self.step)
                self.writer.add_scalar('gpu/memory_reserved_gb', memory_reserved, self.step)
        
        # Dataset performance
        if hasattr(self.dataset, 'get_timing_stats'):
            timing_stats = self.dataset.get_timing_stats()
            if timing_stats.get('throughput'):
                throughput = timing_stats['throughput']
                logger.info(f"Dataset throughput: {throughput['samples_per_second']:.2f} samples/sec")
                
                if self.writer:
                    self.writer.add_scalar('dataset/samples_per_second',
                                          throughput['samples_per_second'], self.step)
                                          
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_validation_loss': self.best_validation_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_validation_loss = checkpoint['best_validation_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
    def train(self):
        """Main training loop."""
        logger.info("Starting streaming training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config['training']['max_epochs']):
                self.epoch = epoch
                
                logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['max_epochs']}")
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step(epoch_metrics['loss'])
                
                # Save checkpoint
                if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                    self.save_checkpoint()
                    
                # Check stopping criteria
                if self.step >= self.config['training']['max_steps']:
                    logger.info(f"Reached maximum steps ({self.config['training']['max_steps']})")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Final checkpoint
            self.save_checkpoint()
            
            # Close monitoring
            if self.writer:
                self.writer.close()
                
            training_time = time.time() - start_time
            logger.info(f"Training completed. Total time: {training_time:.2f}s")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train energy-based ProteinMPNN with streaming")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, 
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = StreamingTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()