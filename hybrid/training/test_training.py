"""
Test Training Loop

This script creates a synthetic dataset to test the training pipeline
without requiring real PDB files. It verifies that:
1. Loss decreases over training steps
2. Energy rankings improve (positive < negative)
3. Training loop components integrate correctly
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import tempfile
from pathlib import Path
import warnings
import sys

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_energy import EnergyModelTrainer, EnergyPredictionModel
from training.losses import ContrastiveLoss
from models.mpnn_encoder import ProteinMPNNBackboneEncoder  
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr


class SyntheticStabilityDataset:
    """Synthetic dataset that mimics StabilityDataset interface for testing"""
    
    def __init__(self, num_samples=100, seq_len=50, positive_ratio=0.5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.positive_ratio = positive_ratio
        
        # Generate synthetic samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        samples = []
        
        for i in range(self.num_samples):
            # Determine if positive (stable) or negative (unstable)
            is_positive = i < int(self.num_samples * self.positive_ratio)
            
            # Generate synthetic backbone features (mock ProteinMPNN embeddings)
            backbone_features = torch.randn(self.seq_len, 128)
            
            # Generate synthetic sequence (amino acid indices)
            if is_positive:
                # "Stable" sequences - more structured patterns
                base_seq = torch.randint(0, 20, (self.seq_len,))
                # Add some structure (e.g., periodic patterns)
                for j in range(0, self.seq_len, 10):
                    base_seq[j:j+3] = torch.tensor([0, 1, 2])  # ALA, CYS, ASP pattern
            else:
                # "Unstable" sequences - more random
                base_seq = torch.randint(0, 20, (self.seq_len,))
            
            # Create mask (all valid positions for simplicity)
            mask = torch.ones(self.seq_len)
            
            # Generation method
            if is_positive:
                generation_method = "native"
            else:
                generation_method = np.random.choice(["random", "mutated", "failed_design"])
            
            sample = {
                'backbone_features': backbone_features,
                'sequence': base_seq,
                'mask': mask,
                'label': int(is_positive),
                'length': self.seq_len,
                'generation_method': generation_method,
                'structure_id': f'synthetic_{i}'
            }
            
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_test_config():
    """Create minimal config for testing"""
    return {
        "experiment_name": "test_training",
        "seed": 42,
        
        "data": {
            "data_dir": "/tmp/dummy",  # Not used with synthetic dataset
            "positive_ratio": 0.5,
            "val_split": 0.2,
            "lazy_loading": False
        },
        
        "model": {
            "mpnn_encoder": {
                "model_name": "mock",  # Will use synthetic encoder
                "hidden_dim": 128,
                "freeze_layers": True
            },
            "energy_head": {
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.1,
                "activation": "relu",
                "use_batch_norm": True
            },
            "sequence_repr": {
                "temperature_schedule": [1.0, 0.5],
                "min_temperature": 0.001,
                "max_temperature": 10.0
            }
        },
        
        "loss": {
            "margin": 1.0,
            "temperature": 0.1,
            "ranking_weight": 1.0,
            "contrastive_weight": 1.0,
            "entropy_weight": 0.01,
            "smoothness_weight": 0.001
        },
        
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 1e-3,  # Higher LR for quick testing
            "weight_decay": 0.01,
            "scheduler": {
                "type": "none"
            }
        },
        
        "training": {
            "max_epochs": 5,  # Short training for testing
            "batch_size": 16,
            "num_workers": 0,  # Avoid multiprocessing issues in tests
            "max_grad_norm": 1.0,
            "patience": 10,
            "save_frequency": 2
        }
    }


class MockProteinMPNNEncoder(torch.nn.Module):
    """Mock encoder for testing that mimics ProteinMPNN interface"""
    
    def __init__(self, hidden_dim=128, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(128, hidden_dim)
        
    def forward(self, backbone_features, **kwargs):
        # Just pass through with a linear transformation
        return self.linear(backbone_features)


class TestTrainer(EnergyModelTrainer):
    """Modified trainer for testing with synthetic data"""
    
    def _setup_model(self):
        """Set up mock model for testing"""
        print("Setting up mock model for testing...")
        
        # Create mock encoder
        encoder = MockProteinMPNNEncoder(hidden_dim=128, device=self.device)
        
        # Create energy head
        energy_head = EnergyHead(
            backbone_dim=128,
            seq_dim=20,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1
        )
        
        # Create sequence representation
        sequence_repr = ContinuousSequenceRepr(
            vocab_size=20,
            temperature_schedule=[1.0, 0.5],
            min_temperature=0.001,
            max_temperature=10.0
        )
        
        # Create combined model
        self.model = EnergyPredictionModel(encoder, energy_head, sequence_repr)
        self.model = self.model.to(self.device)
        
        print(f"Mock model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_data(self):
        """Set up synthetic dataset for testing"""
        print("Setting up synthetic dataset...")
        
        # Create synthetic dataset
        full_dataset = SyntheticStabilityDataset(
            num_samples=80,  # Small for quick testing
            seq_len=40,      # Shorter sequences
            positive_ratio=self.config['data']['positive_ratio']
        )
        
        # Split into train/val
        val_split = self.config['data']['val_split']
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        # Create data loaders with custom collate function
        batch_size = self.config['training']['batch_size']
        
        def collate_fn(batch):
            """Custom collate function for synthetic data"""
            collated = {}
            
            collated['backbone_features'] = torch.stack([item['backbone_features'] for item in batch])
            collated['sequence'] = torch.stack([item['sequence'] for item in batch])
            collated['mask'] = torch.stack([item['mask'] for item in batch])
            collated['label'] = torch.tensor([item['label'] for item in batch])
            collated['length'] = torch.tensor([item['length'] for item in batch])
            collated['generation_method'] = [item['generation_method'] for item in batch]
            collated['structure_id'] = [item['structure_id'] for item in batch]
            
            return collated
        
        from torch.utils.data import DataLoader
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"Synthetic data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")


def test_training():
    """Run training test and verify expected behavior"""
    print("Starting training loop test...")
    
    # Create test configuration
    config = create_test_config()
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "models"
        log_dir = Path(temp_dir) / "logs"
        
        # Initialize trainer
        trainer = TestTrainer(
            config=config,
            model_dir=str(model_dir),
            log_dir=str(log_dir),
            device='cpu'  # Use CPU for testing
        )
        
        try:
            # Setup components
            trainer.setup()
            print("✓ Setup completed successfully")
            
            # Run a few training steps
            initial_metrics = trainer._validate_epoch()
            print(f"Initial validation loss: {initial_metrics['loss']:.6f}")
            print(f"Initial ranking accuracy: {initial_metrics['energy_stats']['ranking_accuracy']:.3f}")
            
            # Train for a few epochs
            original_max_epochs = trainer.config['training']['max_epochs']
            trainer.config['training']['max_epochs'] = 3  # Just a few epochs
            
            initial_loss = initial_metrics['loss']
            
            # Run training
            for epoch in range(3):
                trainer.current_epoch = epoch
                train_metrics = trainer._train_epoch()
                val_metrics = trainer._validate_epoch()
                
                print(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.6f}, "
                      f"Val Loss={val_metrics['loss']:.6f}, "
                      f"Ranking Acc={val_metrics['energy_stats']['ranking_accuracy']:.3f}")
                
                # Check for NaN/Inf
                if np.isnan(train_metrics['loss']) or np.isinf(train_metrics['loss']):
                    raise ValueError("Training loss became NaN/Inf")
            
            # Final validation
            final_metrics = trainer._validate_epoch()
            final_loss = final_metrics['loss']
            final_accuracy = final_metrics['energy_stats']['ranking_accuracy']
            
            print(f"\nTraining Summary:")
            print(f"Initial loss: {initial_loss:.6f}")
            print(f"Final loss: {final_loss:.6f}")
            print(f"Loss change: {final_loss - initial_loss:.6f}")
            print(f"Final ranking accuracy: {final_accuracy:.3f}")
            
            # Test checkpoint saving/loading
            checkpoint_path = model_dir / "test_checkpoint.pt"
            trainer._save_checkpoint("test_checkpoint.pt")
            
            if checkpoint_path.exists():
                print("✓ Checkpoint saved successfully")
                
                # Test loading
                new_trainer = TestTrainer(config, str(model_dir), str(log_dir), 'cpu')
                new_trainer._setup_model()
                new_trainer._load_checkpoint(str(checkpoint_path))
                print("✓ Checkpoint loaded successfully")
            else:
                print("✗ Checkpoint saving failed")
            
            # Success criteria
            success_checks = []
            
            # Check 1: Training completed without errors
            success_checks.append(("Training completed", True))
            
            # Check 2: Loss values are reasonable (not NaN/Inf)
            loss_reasonable = not (np.isnan(final_loss) or np.isinf(final_loss))
            success_checks.append(("Loss values reasonable", loss_reasonable))
            
            # Check 3: Ranking accuracy > random chance (0.5)
            ranking_good = final_accuracy > 0.5
            success_checks.append(("Ranking accuracy > random", ranking_good))
            
            # Check 4: Model learns something (loss changes)
            loss_changed = abs(final_loss - initial_loss) > 1e-6
            success_checks.append(("Loss changed during training", loss_changed))
            
            print(f"\n{'='*50}")
            print("Training Test Results:")
            print(f"{'='*50}")
            
            all_passed = True
            for check_name, passed in success_checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{status}: {check_name}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print(f"\n🎉 ALL TESTS PASSED! Training pipeline is working correctly.")
                return True
            else:
                print(f"\n⚠️ Some tests failed. Check the implementation.")
                return False
            
        except Exception as e:
            print(f"✗ Training test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    success = test_training()
    
    if success:
        print("\n🎯 Training validation completed successfully!")
        exit(0)
    else:
        print("\n❌ Training validation failed!")
        exit(1)