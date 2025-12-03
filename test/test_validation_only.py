#!/usr/bin/env python3
"""
Test specifically the dataset validation method that was failing.
"""

import sys
import json
import torch
from pathlib import Path

# Add hybrid package to path
sys.path.insert(0, str(Path(__file__).parent / 'hybrid'))

def test_dataset_validation():
    """Test specifically the _validate_dataset_composition method"""
    print("Testing dataset validation method...")
    
    try:
        from training.train_energy import EnergyModelTrainer
        
        config = {
            'data': {
                'data_dir': 'proteinmpnn/inputs',
                'max_files': 4,
                'positive_ratio': 0.5,
                'val_split': 0.5,
            },
            'model': {
                'mpnn_encoder': {'model_name': 'v_48_020', 'freeze_layers': True},
                'energy_head': {'hidden_dim': 256, 'num_layers': 2}
            },
            'training': {
                'batch_size': 4,
                'num_workers': 0,
                'max_epochs': 1
            },
            'optimization': {
                'optimizer': 'adamw',
                'learning_rate': 1e-4
            },
            'logging': {'log_interval': 1},
            'seed': 42
        }
        
        trainer = EnergyModelTrainer(config=config, device='cpu')
        
        # Setup model and data only
        trainer._setup_model()
        trainer._setup_data()
        
        print("\n" + "="*60)
        print("Running dataset composition validation...")
        print("="*60)
        
        # This is the method that was failing before our fix
        trainer._validate_dataset_composition()
        
        print("\n✅ Dataset validation PASSED!")
        print("The batch composition error has been fixed!")
        return True
        
    except RuntimeError as e:
        if "Dataset validation failed" in str(e) and "100.0%" in str(e):
            print(f"\n❌ Dataset validation FAILED with the original error:")
            print(f"Error: {e}")
            return False
        else:
            print(f"\n❓ Different error occurred: {e}")
            return False
    except Exception as e:
        print(f"\n❓ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_validation()
    
    if success:
        print("\n🎉 BATCH COMPOSITION FIX IS WORKING!")
        print("You can now run the training script successfully.")
    else:
        print("\n⚠️  Additional work may be needed.")