#!/usr/bin/env python3
"""
Example usage of PDBListManager for training data preparation.

This script demonstrates how to use the PDBListManager to fetch
filtered PDB structures for training energy-based models.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path  
sys.path.insert(0, '/Users/mkrasnow/Desktop/energy-based-proteinmpnn')

from hybrid.data.pdb_manager import PDBListManager

def setup_logging():
    """Setup informative logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Demonstrate PDBListManager usage for training."""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🧬 PDBListManager Example for Training Data Preparation")
    logger.info("=" * 60)
    
    # Initialize PDBListManager with production settings
    cache_dir = Path("./pdb_training_cache")
    manager = PDBListManager(
        cache_dir=cache_dir,
        search_timeout=30,
        max_retries=3,
        rate_limit_delay=0.2  # Be respectful to RCSB servers
    )
    
    logger.info(f"📂 Cache directory: {cache_dir.absolute()}")
    
    # Example 1: Get structures for energy model training
    logger.info("\n🎯 Fetching PDB structures for energy model training...")
    
    training_pdbs = manager.get_filtered_pdb_list(
        max_resolution=3.5,      # ≤ 3.5Å as per requirements
        min_length=20,           # Minimum 20 residues
        max_length=500,          # Maximum 500 residues
        experimental_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        target_count=5000,       # Target 5000+ structures
        use_cache=True,          # Use cache to speed up subsequent calls
        cache_max_age_hours=24   # Cache expires after 24 hours
    )
    
    logger.info(f"✅ Retrieved {len(training_pdbs)} PDB structures for training")
    logger.info(f"📋 Sample PDB IDs: {training_pdbs[:20]}")
    
    # Example 2: Get biological query sets for different scenarios
    logger.info("\n🔬 Getting biological query sets...")
    
    query_sets = manager.get_biological_query_sets()
    
    for set_name, pdb_list in query_sets.items():
        logger.info(f"  📊 {set_name}: {len(pdb_list)} structures")
        logger.info(f"      Sample: {pdb_list[:5]}")
    
    # Example 3: Get high-quality structures for validation
    logger.info("\n🏆 Fetching high-quality structures for validation...")
    
    validation_pdbs = manager.get_filtered_pdb_list(
        max_resolution=2.0,      # Higher resolution for validation
        min_length=50,
        max_length=300,
        target_count=1000,
        use_cache=True
    )
    
    logger.info(f"✅ Retrieved {len(validation_pdbs)} high-quality structures")
    
    # Example 4: Cache statistics and monitoring
    logger.info("\n📈 Cache Statistics:")
    
    stats = manager.get_statistics()
    for key, value in stats.items():
        logger.info(f"  📊 {key}: {value}")
    
    # Example 5: Different protein size categories
    logger.info("\n📏 Protein size categories:")
    
    size_categories = {
        "small": {"max_length": 100, "target_count": 500},
        "medium": {"min_length": 100, "max_length": 300, "target_count": 500},
        "large": {"min_length": 300, "max_length": 500, "target_count": 500}
    }
    
    size_results = {}
    for category, params in size_categories.items():
        pdbs = manager.get_filtered_pdb_list(
            max_resolution=3.5,
            **params,
            use_cache=True
        )
        size_results[category] = pdbs
        logger.info(f"  📏 {category} proteins: {len(pdbs)} structures")
    
    # Example 6: Preparing for training pipeline integration
    logger.info("\n🚀 Training Pipeline Integration Example:")
    
    # This is how you'd use it in your training script
    def prepare_training_data():
        """Example function for training data preparation."""
        
        # Get diverse set of training structures
        pdb_manager = PDBListManager(cache_dir="./training_cache")
        
        # Primary training set
        training_set = pdb_manager.get_filtered_pdb_list(
            max_resolution=3.5,
            min_length=20,
            max_length=500,
            target_count=8000
        )
        
        # Validation set (higher quality)
        validation_set = pdb_manager.get_filtered_pdb_list(
            max_resolution=2.5,
            min_length=30,
            max_length=400,
            target_count=1000
        )
        
        # Test set (highest quality)
        test_set = pdb_manager.get_filtered_pdb_list(
            max_resolution=2.0,
            min_length=40,
            max_length=300,
            target_count=1000
        )
        
        return {
            'train': training_set,
            'validation': validation_set, 
            'test': test_set
        }
    
    # Demonstrate the preparation
    datasets = prepare_training_data()
    
    for split_name, pdb_list in datasets.items():
        logger.info(f"  📚 {split_name}: {len(pdb_list)} structures")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 PDBListManager Example Complete!")
    logger.info(f"📊 Total unique structures accessed: {len(set(training_pdbs))}")
    logger.info(f"💾 Cache location: {cache_dir.absolute()}")
    logger.info("✅ Ready for integration with training pipeline!")
    
    return training_pdbs

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)