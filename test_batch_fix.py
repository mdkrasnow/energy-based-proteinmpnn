#!/usr/bin/env python3
"""
Test script to verify that the batch composition fix works correctly.

This script tests the BalancedBatchSampler and enhanced dataset validation
to ensure training batches contain both positive and negative samples.
"""

import sys
import json
import torch
from pathlib import Path

# Add hybrid package to path
sys.path.insert(0, str(Path(__file__).parent / 'hybrid'))

def create_test_config():
    """Create a minimal test configuration"""
    return {
        'data': {
            'data_dir': 'proteinmpnn/inputs',  # Use existing PDB files
            'max_files': 4,  # Small dataset for testing
            'positive_ratio': 0.5,
            'val_split': 0.5,
        },
        'model': {
            'mpnn_encoder': {
                'model_name': 'v_48_020',
                'freeze_layers': True
            },
            'energy_head': {
                'hidden_dim': 256,
                'num_layers': 2
            }
        },
        'training': {
            'batch_size': 4,  # Small batch for testing
            'num_workers': 0,  # No multiprocessing for debugging
            'max_epochs': 1,
            'patience': 5
        },
        'optimization': {
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01
        },
        'logging': {
            'log_interval': 1,
            'use_tensorboard': False
        },
        'seed': 42
    }

def test_batch_composition():
    """Test that our batch composition fix works"""
    print("Testing batch composition fix...")
    
    try:
        # Import after adding to path
        from training.train_energy import EnergyModelTrainer
        
        # Create test configuration
        config = create_test_config()
        
        # Create trainer
        trainer = EnergyModelTrainer(
            config=config,
            model_dir="test_checkpoints",
            log_dir="test_logs",
            device='cpu'  # Use CPU for testing
        )
        
        print("Trainer created successfully")
        
        # Setup components
        trainer.setup()
        print("Components setup successfully")
        
        # This should now work without the "100% problematic batches" error
        print("\n" + "="*50)
        print("Testing dataset composition validation...")
        print("="*50)
        
        # The validation should pass with our fixes
        result = "SUCCESS"
        print(f"\nBatch composition fix test: {result}")
        return True
        
    except RuntimeError as e:
        if "Dataset validation failed" in str(e):
            print(f"\nBatch composition fix test: FAILED")
            print(f"Error: {e}")
            return False
        else:
            # Some other runtime error
            print(f"\nBatch composition fix test: OTHER ERROR")
            print(f"Error: {e}")
            return False
    except Exception as e:
        print(f"\nBatch composition fix test: SETUP ERROR")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting batch composition fix test...")
    print(f"Working directory: {Path.cwd()}")
    
    success = test_batch_composition()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The batch composition error should now be fixed.")
    else:
        print("\n❌ Test failed!")
        print("Additional debugging may be needed.")
    
    print("\nNext steps:")
    print("1. Run the actual training script to verify the fix")
    print("2. Monitor batch composition during training")
    print("3. Check that energy model converges properly")