#!/usr/bin/env python3
"""
Test Script for Multi-Landscape Training

This script tests the multi-landscape training implementation with synthetic data
to validate the core functionality before running on real protein data.

Key Tests:
1. Configuration generation and validation
2. Model initialization for multiple landscapes  
3. Training loop functionality
4. Cross-landscape consistency
5. IRED optimizer integration
6. Memory and computational requirements
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from training.train_landscapes import MultiLandscapeTrainer, MultiLandscapeConfig, LandscapeConfig
from inference.ired_optimizer import IREDSequenceOptimizer


class SyntheticProteinDataset(Dataset):
    """Synthetic dataset for testing multi-landscape training."""
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 50, backbone_dim: int = 128):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.backbone_dim = backbone_dim
        
        # Generate synthetic data
        self.data = []
        for i in range(num_samples):
            # Random backbone features
            backbone_features = torch.randn(seq_length, backbone_dim)
            
            # Random sequence
            sequence = torch.randint(0, 20, (seq_length,))
            
            # Random mask (simulate variable lengths)
            actual_length = torch.randint(20, seq_length + 1, (1,)).item()
            mask = torch.zeros(seq_length)
            mask[:actual_length] = 1
            
            # Random label (positive/negative)
            label = torch.randint(0, 2, (1,)).item()
            
            # Generation method for negatives
            if label == 0:
                methods = ['random', 'mutated', 'failed_design']
                generation_method = np.random.choice(methods)
            else:
                generation_method = 'native'
            
            sample = {
                'backbone_features': backbone_features,
                'sequence': sequence,
                'mask': mask,
                'label': label,
                'length': actual_length,
                'generation_method': generation_method,
                'structure_id': f'synthetic_{i}'
            }
            self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_test_config() -> Dict[str, Any]:
    """Create test configuration for multi-landscape training."""
    return {
        'model': {
            'mpnn_encoder': {
                'model_name': 'v_48_020',
                'hidden_dim': 128,
                'freeze_layers': True
            },
            'energy_head': {
                'hidden_dim': 256,  # Smaller for testing
                'num_layers': 2,
                'dropout': 0.1,
                'activation': 'relu',
                'use_batch_norm': True
            },
            'sequence_repr': {
                'min_temperature': 1e-3,
                'max_temperature': 10.0
            }
        },
        'data': {
            'data_dir': 'synthetic',  # Will use synthetic dataset
            'positive_ratio': 0.5,
            'negative_methods': ['random', 'mutated', 'failed_design'],
            'max_sequence_length': 100,
            'min_sequence_length': 20,
            'val_split': 0.2,
            'lazy_loading': False
        },
        'training': {
            'batch_size': 4,  # Small for testing
            'max_epochs': 3,  # Short for testing
            'num_workers': 0,  # No multiprocessing for testing
            'patience': 5,
            'save_frequency': 1,
            'max_grad_norm': 1.0
        },
        'optimization': {
            'optimizer': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'scheduler': {
                'type': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 2
            }
        },
        'loss': {
            'margin': 1.0,
            'temperature': 0.1,
            'ranking_weight': 1.0,
            'contrastive_weight': 1.0,
            'entropy_weight': 0.01,
            'smoothness_weight': 0.001
        },
        'seed': 42
    }


def test_landscape_config_generation():
    """Test landscape configuration generation."""
    print("Testing landscape configuration generation...")
    
    # Test different schedules
    schedules = ['linear', 'exponential', 'cosine']
    
    for schedule in schedules:
        config = MultiLandscapeConfig(
            num_landscapes=5,
            temperature_schedule=schedule,
            base_temperature=1.0,
            final_temperature=0.1
        )
        
        landscapes = config.generate_landscape_configs()
        
        assert len(landscapes) == 5, f"Expected 5 landscapes, got {len(landscapes)}"
        
        # Check temperature progression
        temperatures = [l.temperature for l in landscapes]
        assert temperatures[0] == 1.0, f"First temperature should be 1.0, got {temperatures[0]}"
        assert abs(temperatures[-1] - 0.1) < 0.01, f"Last temperature should be ~0.1, got {temperatures[-1]}"
        
        # Check monotonic decrease
        assert all(temperatures[i] >= temperatures[i+1] for i in range(len(temperatures)-1)), \
            f"Temperatures should decrease monotonically: {temperatures}"
        
        print(f"✓ {schedule} schedule: {[f'{t:.3f}' for t in temperatures]}")
    
    print("✓ Landscape configuration generation tests passed")


def test_multi_landscape_trainer_init():
    """Test MultiLandscapeTrainer initialization."""
    print("Testing MultiLandscapeTrainer initialization...")
    
    base_config = create_test_config()
    landscape_config = MultiLandscapeConfig(num_landscapes=3, sequential_training=True)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_dir = temp_path / "models"
        log_dir = temp_path / "logs"
        
        trainer = MultiLandscapeTrainer(
            base_config=base_config,
            landscape_config=landscape_config,
            model_dir=str(model_dir),
            log_dir=str(log_dir),
            device='cpu'
        )
        
        # Check initialization
        assert len(trainer.landscapes) == 3, f"Expected 3 landscapes, got {len(trainer.landscapes)}"
        assert trainer.device.type == 'cpu'
        assert trainer.model_dir.exists()
        assert trainer.log_dir.exists()
        
        print("✓ MultiLandscapeTrainer initialization passed")


def test_synthetic_data_setup():
    """Test synthetic data setup."""
    print("Testing synthetic data setup...")
    
    # Create synthetic dataset
    dataset = SyntheticProteinDataset(num_samples=100, seq_length=50)
    
    # Test dataset properties
    assert len(dataset) == 100
    
    sample = dataset[0]
    assert 'backbone_features' in sample
    assert 'sequence' in sample
    assert 'label' in sample
    assert sample['backbone_features'].shape == (50, 128)
    assert sample['sequence'].shape == (50,)
    
    # Test data loader
    def collate_fn(batch):
        collated = {}
        collated['backbone_features'] = torch.stack([item['backbone_features'] for item in batch])
        collated['sequence'] = torch.stack([item['sequence'] for item in batch])
        collated['mask'] = torch.stack([item['mask'] for item in batch])
        collated['label'] = torch.tensor([item['label'] for item in batch])
        collated['length'] = torch.tensor([item['length'] for item in batch])
        collated['generation_method'] = [item['generation_method'] for item in batch]
        collated['structure_id'] = [item['structure_id'] for item in batch]
        return collated
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    assert batch['backbone_features'].shape == (4, 50, 128)
    assert batch['sequence'].shape == (4, 50)
    assert len(batch['label']) == 4
    
    print("✓ Synthetic data setup tests passed")


class MockTrainer(MultiLandscapeTrainer):
    """Mock trainer that uses synthetic data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = True  # Set training mode for testing
    
    def _setup_data(self):
        """Override to use synthetic dataset."""
        # Create synthetic datasets
        train_dataset = SyntheticProteinDataset(num_samples=80, seq_length=50)
        val_dataset = SyntheticProteinDataset(num_samples=20, seq_length=50)
        
        # Custom collate function
        def collate_fn(batch):
            collated = {}
            collated['backbone_features'] = torch.stack([item['backbone_features'] for item in batch])
            collated['sequence'] = torch.stack([item['sequence'] for item in batch])
            collated['mask'] = torch.stack([item['mask'] for item in batch])
            collated['label'] = torch.tensor([item['label'] for item in batch])
            collated['length'] = torch.tensor([item['length'] for item in batch])
            collated['generation_method'] = [item['generation_method'] for item in batch]
            collated['structure_id'] = [item['structure_id'] for item in batch]
            return collated
        
        batch_size = self.base_config['training']['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"Synthetic data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)} samples")
    
    def _setup_shared_encoder(self):
        """Override to skip encoder setup for testing."""
        if not self.landscape_config.shared_encoder:
            return
        self.shared_encoder = None  # Skip encoder for testing
        print("Skipped shared encoder setup (testing mode)")
    
    def _encode_structures(self, batch: Dict) -> torch.Tensor:
        """Override to return pre-computed features."""
        return batch['backbone_features']


def test_training_loop():
    """Test the multi-landscape training loop."""
    print("Testing multi-landscape training loop...")
    
    base_config = create_test_config()
    landscape_config = MultiLandscapeConfig(
        num_landscapes=3,
        sequential_training=True,
        shared_encoder=False  # Skip encoder for testing
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        model_dir = temp_path / "models"
        log_dir = temp_path / "logs"
        
        # Create mock trainer
        trainer = MockTrainer(
            base_config=base_config,
            landscape_config=landscape_config,
            model_dir=str(model_dir),
            log_dir=str(log_dir),
            device='cpu'
        )
        
        try:
            # Setup trainer
            trainer.setup()
            
            # Check that components are initialized
            assert len(trainer.energy_models) == 3
            assert len(trainer.optimizers) == 3
            assert len(trainer.loss_functions) == 3
            assert trainer.sequence_repr is not None
            
            # Check temperature schedule
            temperatures = [trainer.sequence_repr.temperature_schedule[i].item() 
                          for i in range(len(trainer.energy_models))]
            print(f"Temperature schedule: {[f'{t:.3f}' for t in temperatures]}")
            
            # Test single landscape training (short version)
            print("Running short training test...")
            
            # Train first landscape briefly
            energy_model = trainer.energy_models[0]
            optimizer = trainer.optimizers[0]
            config = trainer.landscapes[0]
            
            # Set training mode
            energy_model.train()
            trainer.training = True  # Add missing training attribute
            total_loss = 0.0
            num_batches = 0
            
            for batch in trainer.train_loader:
                if num_batches >= 2:  # Only test 2 batches
                    break
                
                batch = trainer._move_batch_to_device(batch)
                
                optimizer.zero_grad()
                outputs = trainer._forward_landscape(batch, 0, config)
                
                if outputs.get('skip_batch', False):
                    continue
                
                loss = trainer.loss_functions[0](
                    pos_energies=outputs['pos_energies'],
                    neg_energies=outputs['neg_energies'],
                    negative_types=outputs.get('negative_types', [])
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                print(f"Batch {num_batches}: Loss = {loss.item():.4f}")
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Average training loss: {avg_loss:.4f}")
                assert avg_loss > 0, "Loss should be positive"
                assert not np.isnan(avg_loss), "Loss should not be NaN"
            
            print("✓ Training loop test passed")
            
        except Exception as e:
            print(f"✗ Training loop test failed: {e}")
            raise


def test_ired_integration():
    """Test integration with IRED optimizer."""
    print("Testing IRED optimizer integration...")
    
    # Create simple energy models
    energy_models = []
    temperatures = [1.0, 0.5, 0.1]
    
    for temp in temperatures:
        model = EnergyHead(backbone_dim=128, seq_dim=20, hidden_dim=256, num_layers=2)
        energy_models.append(model)
    
    # Create sequence representation
    sequence_repr = ContinuousSequenceRepr(
        vocab_size=20,
        temperature_schedule=temperatures
    )
    
    # Create IRED optimizer
    optimizer = IREDSequenceOptimizer(
        energy_models=energy_models,
        sequence_repr=sequence_repr,
        device='cpu',
        seed=42
    )
    
    # Test optimization
    batch_size, seq_len = 2, 30
    backbone_features = torch.randn(batch_size, seq_len, 128)
    mask = torch.ones(batch_size, seq_len)
    
    # Run optimization
    result = optimizer.optimize_sequence(
        backbone_features,
        mask=mask,
        max_steps=10,  # Short test
        return_trajectory=True
    )
    
    # Check result
    assert result.sequence is not None, "Optimization should return a sequence"
    assert result.sequence.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {result.sequence.shape}"
    assert len(result.trajectory) > 0, "Should have trajectory data"
    assert result.total_steps > 0, "Should have taken optimization steps"
    assert result.landscapes_used == len(temperatures), f"Should use all {len(temperatures)} landscapes"
    
    print(f"✓ IRED optimization: {result.total_steps} steps, energy = {result.final_energy:.4f}")
    print("✓ IRED integration test passed")


def test_consistency_loss():
    """Test cross-landscape consistency loss computation."""
    print("Testing cross-landscape consistency loss...")
    
    base_config = create_test_config()
    landscape_config = MultiLandscapeConfig(
        num_landscapes=3,
        consistency_weight=0.1
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        trainer = MockTrainer(
            base_config=base_config,
            landscape_config=landscape_config,
            model_dir=str(temp_path / "models"),
            log_dir=str(temp_path / "logs"),
            device='cpu'
        )
        
        trainer.setup()
        
        # Create test batch
        batch_size = 4
        seq_len = 30
        
        batch = {
            'backbone_features': torch.randn(batch_size, seq_len, 128),
            'sequence': torch.randint(0, 20, (batch_size, seq_len)),
            'mask': torch.ones(batch_size, seq_len),
            'label': torch.tensor([1, 0, 1, 0]),  # Mix of positive/negative
            'generation_method': ['native', 'random', 'native', 'mutated']
        }
        
        batch = trainer._move_batch_to_device(batch)
        
        # Test consistency loss computation
        config1 = trainer.landscapes[1]
        outputs1 = trainer._forward_landscape(batch, 1, config1)
        
        if not outputs1.get('skip_batch', False):
            consistency_loss = trainer._compute_consistency_loss(batch, 1, outputs1)
            
            assert isinstance(consistency_loss, torch.Tensor), "Consistency loss should be tensor"
            assert consistency_loss.numel() == 1, "Should be scalar"
            assert consistency_loss.item() >= 0, "Consistency loss should be non-negative"
            
            print(f"✓ Consistency loss: {consistency_loss.item():.6f}")
        
        print("✓ Consistency loss test passed")


def test_memory_requirements():
    """Test memory requirements for multi-landscape training."""
    print("Testing memory requirements...")
    
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create larger test setup
    base_config = create_test_config()
    base_config['training']['batch_size'] = 8
    
    landscape_config = MultiLandscapeConfig(
        num_landscapes=5,
        sequential_training=True,
        shared_encoder=False
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        trainer = MockTrainer(
            base_config=base_config,
            landscape_config=landscape_config,
            model_dir=str(temp_path / "models"),
            log_dir=str(temp_path / "logs"),
            device='cpu'
        )
        
        trainer.setup()
        
        after_setup_memory = process.memory_info().rss / 1024 / 1024
        
        # Test forward pass memory
        batch = next(iter(trainer.train_loader))
        batch = trainer._move_batch_to_device(batch)
        
        # Forward pass on all landscapes
        for i, config in enumerate(trainer.landscapes):
            outputs = trainer._forward_landscape(batch, i, config)
            if not outputs.get('skip_batch', False):
                loss = trainer.loss_functions[i](
                    pos_energies=outputs['pos_energies'],
                    neg_energies=outputs['neg_energies'],
                    negative_types=outputs.get('negative_types', [])
                )
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        # Clean up
        del trainer
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  After setup: {after_setup_memory:.1f} MB (+{after_setup_memory - initial_memory:.1f} MB)")
        print(f"  Peak: {peak_memory:.1f} MB (+{peak_memory - initial_memory:.1f} MB)")
        print(f"  Final: {final_memory:.1f} MB")
        
        # Check memory is reasonable (less than 500 MB increase for CPU testing)
        memory_increase = peak_memory - initial_memory
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.1f} MB"
        
        print("✓ Memory requirements test passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Multi-Landscape Training Test Suite")
    print("=" * 60)
    
    test_functions = [
        test_landscape_config_generation,
        test_multi_landscape_trainer_init,
        test_synthetic_data_setup,
        test_training_loop,
        test_ired_integration,
        test_consistency_loss,
        test_memory_requirements
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False
    else:
        print("✅ All tests passed! Multi-landscape training is ready.")
        return True


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 Multi-landscape training implementation is validated!")
        print("\nNext steps:")
        print("1. Prepare real protein structure data")
        print("2. Run: python train_landscapes.py --config config_landscapes_template.json")
        print("3. Monitor landscape convergence and energy progression")
        print("4. Test trained models with IRED optimizer")
    else:
        exit(1)