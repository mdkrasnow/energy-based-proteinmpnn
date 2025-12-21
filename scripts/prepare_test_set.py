#!/usr/bin/env python3
"""
Prepare Test Set for Evaluation

Creates benchmark_problems.json from existing PDB structures.
This defines the test set of protein structures to evaluate your model on.

Note: Using real PDB structures as test cases is perfectly acceptable.
What matters is that you run REAL optimization on them (not fake results).

Usage:
    python scripts/prepare_test_set.py \\
        --pdb-dir proteinmpnn/inputs \\
        --output evaluation_data/benchmark_problems.json \\
        --max-per-category 20
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def discover_pdb_files(pdb_dir: Path, max_per_category: int = None) -> List[Dict]:
    """
    Discover PDB files and categorize by difficulty.

    Returns list of benchmark problems with structure metadata.
    """

    # Difficulty mapping based on structure type
    difficulty_mapping = {
        "PDB_monomers": "easy",          # Single chain proteins
        "PDB_complexes": "medium",       # Multi-chain complexes
        "PDB_homooligomers": "hard",     # Symmetric assemblies
    }

    test_structures = []
    category_counts = {cat: 0 for cat in difficulty_mapping.keys()}

    for category, difficulty in difficulty_mapping.items():
        category_dir = pdb_dir / category

        if not category_dir.exists():
            print(f"Warning: Category directory not found: {category_dir}")
            continue

        pdb_files = list(category_dir.glob("*.pdb"))
        print(f"Found {len(pdb_files)} PDB files in {category}")

        # Limit per category if specified
        if max_per_category:
            pdb_files = pdb_files[:max_per_category]
            print(f"  Using first {len(pdb_files)} files")

        for pdb_file in pdb_files:
            test_structures.append({
                "id": pdb_file.stem,
                "pdb_path": str(pdb_file.absolute()),
                "difficulty": difficulty,
                "category": category,
                "type": "real_structure",
                "source": "ProteinMPNN dataset"
            })
            category_counts[category] += 1

    # Print summary
    print(f"\nTest set summary:")
    print(f"  Total structures: {len(test_structures)}")
    for category, count in category_counts.items():
        difficulty = difficulty_mapping[category]
        print(f"  - {difficulty:8s} ({category}): {count}")

    return test_structures


def validate_pdb_file(pdb_path: Path) -> bool:
    """Basic validation that PDB file contains CA atoms"""
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test set from PDB structures"
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("proteinmpnn/inputs"),
        help="Directory containing PDB structures"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_data/benchmark_problems.json"),
        help="Output path for benchmark_problems.json"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Maximum structures per difficulty category"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate PDB files (slower but safer)"
    )

    args = parser.parse_args()

    print("="*60)
    print("PREPARING TEST SET")
    print("="*60)
    print(f"PDB directory: {args.pdb_dir}")
    print(f"Output: {args.output}")
    print(f"Max per category: {args.max_per_category or 'unlimited'}")
    print("="*60)
    print()

    # Discover PDB files
    test_structures = discover_pdb_files(args.pdb_dir, args.max_per_category)

    if not test_structures:
        print("\nError: No PDB files found!")
        print(f"Check that {args.pdb_dir} exists and contains PDB files.")
        return 1

    # Validate PDB files if requested
    if args.validate:
        print("\nValidating PDB files...")
        valid_structures = []
        for structure in test_structures:
            if validate_pdb_file(Path(structure['pdb_path'])):
                valid_structures.append(structure)
            else:
                print(f"  Skipping invalid: {structure['id']}")

        test_structures = valid_structures
        print(f"  {len(test_structures)} valid structures")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save test set
    with open(args.output, 'w') as f:
        json.dump(test_structures, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TEST SET CREATED: {args.output}")
    print(f"  Total structures: {len(test_structures)}")
    print(f"{'='*60}")
    print("\nNext step: Generate real evaluation data:")
    print(f"  python scripts/generate_evaluation_data.py \\")
    print(f"    --checkpoint checkpoints/best_model.pt \\")
    print(f"    --test-set {args.output} \\")
    print(f"    --output-dir evaluation_data")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
