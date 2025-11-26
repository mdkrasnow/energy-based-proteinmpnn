"""
Generate Reference Outputs for ProteinMPNN Encoder Validation

This script generates reference encoder outputs from standalone ProteinMPNN models
for use in validating the hybrid encoder integration. The outputs serve as ground
truth for comparison testing.

Generates references for:
- Multiple model types (vanilla, ca_model, soluble)
- Multiple test structures (small, medium, large)
- Multiple random seeds for reproducibility testing

Output format: NPZ files with encoder features and metadata
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
project_root = Path(__file__).parent.parent
proteinmpnn_path = project_root / "proteinmpnn"
sys.path.append(str(proteinmpnn_path))
sys.path.append(str(project_root / "hybrid"))

try:
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, gather_nodes
    # Import validation helpers directly since hybrid isn't a package
    sys.path.append(str(project_root / "hybrid"))
    from utils.validation_helpers import setup_deterministic_environment, convert_parsed_pdb_to_batch
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Configuration
MODEL_CONFIGS = {
    "vanilla": {"ca_only": False, "model_dir": "vanilla_model_weights"},
    "ca_model": {"ca_only": True, "model_dir": "ca_model_weights"},
    "soluble": {"ca_only": False, "model_dir": "soluble_model_weights"}
}

MODEL_VERSIONS = ["v_48_020"]  # Start with one version for testing

TEST_STRUCTURES = {
    "5L33.pdb": {"category": "small", "dir": "PDB_monomers"},
    # "3HTN.pdb": {"category": "medium", "dir": "PDB_complexes"},  
    # "4GYT.pdb": {"category": "large", "dir": "PDB_homooligomers"}
}

RANDOM_SEEDS = [42]  # Start with one seed for testing


class ReferenceGenerator:
    """Generate reference encoder outputs from standalone ProteinMPNN"""
    
    def __init__(self, proteinmpnn_dir: Path, output_dir: Path):
        self.proteinmpnn_dir = Path(proteinmpnn_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify proteinmpnn directory structure
        if not self.proteinmpnn_dir.exists():
            raise FileNotFoundError(f"ProteinMPNN directory not found: {proteinmpnn_dir}")
    
    def load_proteinmpnn_model(self, model_type: str, model_version: str) -> ProteinMPNN:
        """Load standalone ProteinMPNN model"""
        config = MODEL_CONFIGS[model_type]
        model_dir = self.proteinmpnn_dir / config["model_dir"]
        checkpoint_path = model_dir / f"{model_version}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading {model_type} model {model_version} from {checkpoint_path}")
        
        # Create model with standard ProteinMPNN parameters
        model = ProteinMPNN(
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
            ca_only=config["ca_only"]
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        return model
    
    def load_test_structure(self, structure_name: str) -> Tuple[Dict, str]:
        """Load and parse test structure"""
        structure_info = TEST_STRUCTURES[structure_name]
        structure_dir = self.proteinmpnn_dir / "inputs" / structure_info["dir"] / "pdbs"
        structure_path = structure_dir / structure_name
        
        if not structure_path.exists():
            raise FileNotFoundError(f"Test structure not found: {structure_path}")
        
        logger.info(f"Loading structure {structure_name} from {structure_path}")
        
        # Parse using ProteinMPNN's parser
        parsed_pdb = parse_PDB(str(structure_path))
        
        return parsed_pdb, str(structure_path)
    
    def generate_reference_features(
        self, 
        model: ProteinMPNN, 
        parsed_pdb: Dict,
        chain_id: Optional[str] = None,
        seed: int = 42
    ) -> Dict[str, torch.Tensor]:
        """Generate reference encoder features from standalone ProteinMPNN"""
        
        # Setup deterministic environment
        setup_deterministic_environment(seed)
        
        # Convert to batch format
        batch = convert_parsed_pdb_to_batch(parsed_pdb, chain_id)
        
        with torch.no_grad():
            # Extract batch components
            X = batch['X']
            mask = batch['mask'] 
            residue_idx = batch['residue_idx']
            chain_encoding_all = batch['chain_encoding_all']
            
            # Build graph representation using ProteinMPNN features
            E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
            
            # Initialize node and edge features
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
            h_E = model.W_e(E)
            
            # Create attention mask for encoder
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            
            # Run through encoder layers
            encoder_outputs = []
            for layer_idx, layer in enumerate(model.encoder_layers):
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                encoder_outputs.append(h_V.clone())
            
            # Collect outputs
            reference_data = {
                "final_features": h_V,
                "layer_outputs": encoder_outputs,
                "edge_features": h_E,
                "edge_indices": E_idx,
                "attention_mask": mask_attend,
                "sequence_mask": mask,
                "input_coordinates": X
            }
            
            return reference_data
    
    def generate_all_references(self, overwrite: bool = False) -> None:
        """Generate reference outputs for all combinations"""
        
        total_combinations = len(MODEL_CONFIGS) * len(MODEL_VERSIONS) * len(TEST_STRUCTURES) * len(RANDOM_SEEDS)
        logger.info(f"Generating {total_combinations} reference outputs...")
        
        completed = 0
        errors = []
        
        for model_type in MODEL_CONFIGS.keys():
            for model_version in MODEL_VERSIONS:
                try:
                    # Load model once per type/version
                    model = self.load_proteinmpnn_model(model_type, model_version)
                    
                    for structure_name, structure_info in TEST_STRUCTURES.items():
                        try:
                            # Load structure once per structure
                            parsed_pdb, structure_path = self.load_test_structure(structure_name)
                            
                            # Get first chain from parsed PDB format
                            structure_data = parsed_pdb[0]  # First structure
                            coord_keys = [k for k in structure_data.keys() if k.startswith('coords_chain_')]
                            if not coord_keys:
                                raise ValueError(f"No coordinate data found in {structure_name}")
                            chain_id = coord_keys[0].replace('coords_chain_', '')
                            
                            for seed in RANDOM_SEEDS:
                                # Generate output filename
                                output_filename = f"ref_{model_type}_{model_version}_{structure_name.split('.')[0]}_seed{seed}.npz"
                                output_path = self.output_dir / output_filename
                                
                                if output_path.exists() and not overwrite:
                                    logger.info(f"Skipping existing reference: {output_filename}")
                                    completed += 1
                                    continue
                                
                                try:
                                    # Generate reference features
                                    reference_data = self.generate_reference_features(
                                        model, parsed_pdb, chain_id, seed
                                    )
                                    
                                    # Convert tensors to numpy for saving
                                    save_data = {}
                                    for key, value in reference_data.items():
                                        if isinstance(value, torch.Tensor):
                                            save_data[key] = value.numpy()
                                        elif isinstance(value, list):
                                            # Handle list of tensors (layer outputs)
                                            save_data[key] = np.array([v.numpy() for v in value])
                                        else:
                                            save_data[key] = value
                                    
                                    # Add metadata
                                    save_data.update({
                                        "model_type": model_type,
                                        "model_version": model_version,
                                        "structure_name": structure_name,
                                        "structure_path": structure_path,
                                        "chain_id": chain_id,
                                        "random_seed": seed,
                                        "num_encoder_layers": len(reference_data["layer_outputs"]),
                                        "feature_dim": reference_data["final_features"].shape[-1],
                                        "sequence_length": reference_data["final_features"].shape[1]
                                    })
                                    
                                    # Save reference data
                                    np.savez_compressed(output_path, **save_data)
                                    
                                    logger.info(f"Generated reference: {output_filename}")
                                    completed += 1
                                    
                                except Exception as e:
                                    error_msg = f"Failed to generate {output_filename}: {e}"
                                    logger.error(error_msg)
                                    errors.append(error_msg)
                        
                        except Exception as e:
                            error_msg = f"Failed to load structure {structure_name}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            continue
                
                except Exception as e:
                    error_msg = f"Failed to load model {model_type}/{model_version}: {e}"
                    logger.error(error_msg) 
                    errors.append(error_msg)
                    continue
        
        # Summary
        logger.info(f"\nReference generation complete:")
        logger.info(f"  Successfully generated: {completed}/{total_combinations}")
        if errors:
            logger.info(f"  Errors encountered: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                logger.info(f"    - {error}")
            if len(errors) > 5:
                logger.info(f"    ... and {len(errors) - 5} more errors")
    
    def validate_references(self) -> Dict[str, Any]:
        """Validate generated reference files"""
        logger.info("Validating reference files...")
        
        results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": [],
            "missing_combinations": [],
            "file_statistics": {}
        }
        
        # Check for all expected combinations
        expected_files = []
        for model_type in MODEL_CONFIGS.keys():
            for model_version in MODEL_VERSIONS:
                for structure_name in TEST_STRUCTURES.keys():
                    for seed in RANDOM_SEEDS:
                        filename = f"ref_{model_type}_{model_version}_{structure_name.split('.')[0]}_seed{seed}.npz"
                        expected_files.append(filename)
        
        for filename in expected_files:
            file_path = self.output_dir / filename
            results["total_files"] += 1
            
            if not file_path.exists():
                results["missing_combinations"].append(filename)
                continue
            
            try:
                # Load and validate file
                data = np.load(file_path)
                
                # Check required fields
                required_fields = ["final_features", "layer_outputs", "model_type", "structure_name", "random_seed"]
                missing_fields = [field for field in required_fields if field not in data.files]
                
                if missing_fields:
                    results["invalid_files"].append(f"{filename}: missing fields {missing_fields}")
                    continue
                
                # Check data shapes and types
                features = data["final_features"]
                if len(features.shape) != 3:
                    results["invalid_files"].append(f"{filename}: invalid feature shape {features.shape}")
                    continue
                
                # File is valid
                results["valid_files"] += 1
                
                # Collect statistics
                results["file_statistics"][filename] = {
                    "feature_shape": features.shape,
                    "num_layers": len(data["layer_outputs"]),
                    "model_type": str(data["model_type"]),
                    "structure": str(data["structure_name"]),
                    "seed": int(data["random_seed"])
                }
                
            except Exception as e:
                results["invalid_files"].append(f"{filename}: {e}")
        
        logger.info(f"Validation complete: {results['valid_files']}/{results['total_files']} files valid")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate ProteinMPNN encoder validation references")
    parser.add_argument("--proteinmpnn_dir", type=str, default="proteinmpnn",
                       help="Path to ProteinMPNN directory")
    parser.add_argument("--output_dir", type=str, default="tests/validation_references", 
                       help="Output directory for reference files")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing reference files")
    parser.add_argument("--validate_only", action="store_true", 
                       help="Only validate existing reference files")
    
    args = parser.parse_args()
    
    # Setup paths
    if not os.path.isabs(args.proteinmpnn_dir):
        proteinmpnn_dir = project_root / args.proteinmpnn_dir
    else:
        proteinmpnn_dir = Path(args.proteinmpnn_dir)
    
    if not os.path.isabs(args.output_dir):
        output_dir = project_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    try:
        generator = ReferenceGenerator(proteinmpnn_dir, output_dir)
        
        if args.validate_only:
            # Only validate existing files
            results = generator.validate_references()
            
            # Save validation report
            report_path = output_dir / "validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")
        else:
            # Generate references
            generator.generate_all_references(overwrite=args.overwrite)
            
            # Validate generated references
            results = generator.validate_references()
            
            # Save validation report  
            report_path = output_dir / "validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")
            
            if results["missing_combinations"] or results["invalid_files"]:
                logger.warning("Some reference files are missing or invalid")
                sys.exit(1)
            else:
                logger.info("All reference files generated successfully!")
    
    except Exception as e:
        logger.error(f"Reference generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()