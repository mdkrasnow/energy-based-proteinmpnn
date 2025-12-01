"""
Stability Dataset for Energy-Based Protein Design Training

This module implements a PyTorch dataset for training protein stability energy models through
contrastive learning. The dataset provides positive pairs (native/stable sequences) and 
negative pairs (random/mutated/failed sequences) for energy ranking optimization.
"""

import os
import sys
import json
import random
import warnings
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# BioPython imports with error handling
try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.SeqUtils import seq1
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available. PDB parsing will be limited.")

# Type annotations fallback for when BioPython is not available
if not BIOPYTHON_AVAILABLE:
    from typing import Any
    Chain = Any  # Fallback type annotation when BioPython imports fail

# Safely add proteinmpnn to path for utilities
def _add_proteinmpnn_to_path():
    """Safely add ProteinMPNN module to Python path with validation"""
    current_dir = Path(__file__).parent
    # Use fixed relative path - no user input
    proteinmpnn_dir = current_dir.parent.parent / 'proteinmpnn'
    
    try:
        # Validate the path exists, is a directory, and is readable
        if (proteinmpnn_dir.exists() and 
            proteinmpnn_dir.is_dir() and 
            os.access(proteinmpnn_dir, os.R_OK)):
            # Resolve to absolute path and validate it's safe
            resolved_path = proteinmpnn_dir.resolve()
            # Add to path only if not already there, using append to maintain precedence
            resolved_str = str(resolved_path)
            if resolved_str not in sys.path:
                sys.path.append(resolved_str)
            return True
    except OSError:
        # Only catch file system errors, let other exceptions bubble up
        pass
    return False

_proteinmpnn_path_added = _add_proteinmpnn_to_path()

try:
    from protein_mpnn_utils import parse_PDB_biounits, _S_to_seq
    PROTEINMPNN_AVAILABLE = True
except ImportError as e:
    # Retry with explicit path addition if not found
    import sys
    import os
    
    # Store the original error for debugging
    _original_import_error = str(e)
    
    try:
        # Try to find proteinmpnn in common locations
        possible_paths = [
            os.path.join(os.getcwd(), 'proteinmpnn'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'proteinmpnn')
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)  # Use insert to prioritize
                
        from protein_mpnn_utils import parse_PDB_biounits, _S_to_seq
        PROTEINMPNN_AVAILABLE = True
    except ImportError as e2:
        PROTEINMPNN_AVAILABLE = False
        # Print detailed error for debugging
        import sys
        print(f"DEBUG: Failed to import protein_mpnn_utils", file=sys.stderr)
        print(f"  Original error: {_original_import_error}", file=sys.stderr)
        print(f"  Retry error: {str(e2)}", file=sys.stderr)
        print(f"  sys.path: {sys.path[:5]}", file=sys.stderr)
        print(f"  cwd: {os.getcwd()}", file=sys.stderr)

# Import MPNN encoder for backbone features extraction
try:
    from ..models.mpnn_encoder import ProteinMPNNBackboneEncoder
    MPNN_ENCODER_AVAILABLE = True
except (ImportError, ValueError):
    try:
        # Fallback: try absolute import
        from hybrid.models.mpnn_encoder import ProteinMPNNBackboneEncoder
        MPNN_ENCODER_AVAILABLE = True
    except ImportError:
        MPNN_ENCODER_AVAILABLE = False


class StabilityDataset(Dataset):
    """
    PyTorch Dataset for protein stability training with positive/negative sequence pairs.
    
    This dataset loads protein structures and generates training pairs for contrastive energy
    learning. Positive examples come from native protein sequences, while negative examples
    are generated through various methods (random sequences, destabilizing mutations, etc.).
    
    Args:
        data_dir: Directory containing protein structure files (PDB format)
        positive_ratio: Ratio of positive to total samples (default: 0.5)
        negative_methods: Methods for generating negative sequences (default: all methods)
        max_sequence_length: Maximum sequence length to include (default: 500)
        min_sequence_length: Minimum sequence length to include (default: 20)
        structure_extensions: File extensions to consider as structures (default: ['.pdb'])
        cache_dir: Directory for caching processed data (default: None)
        seed: Random seed for reproducibility (default: 42)
        transform: Optional transform function for data augmentation
        target_transform: Optional transform for labels
        lazy_loading: Load structures on-demand vs pre-load (default: True)
        include_coordinates: Include 3D coordinates in output (default: True)
        max_files: Maximum number of structure files to load (default: None = all)
        extract_backbone_features: Extract backbone features using MPNN encoder (default: True)
        mpnn_model_path: Path to MPNN model weights (default: None = use default)
        max_coord_seq_mismatch_ratio: Maximum ratio of coordinate/sequence length mismatch 
            before rejecting sample (returns None). Ratio calculated as abs(coord_len - seq_len) / max(coord_len, seq_len). 
            Values between 0.0-1.0. (default: 0.5)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        positive_ratio: float = 0.5,
        negative_methods: Optional[List[str]] = None,
        max_sequence_length: int = 500,
        min_sequence_length: int = 20,
        structure_extensions: List[str] = ['.pdb'],
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lazy_loading: bool = True,
        include_coordinates: bool = True,
        max_files: Optional[int] = None,
        extract_backbone_features: bool = True,
        mpnn_model_path: Optional[str] = None,
        max_coord_seq_mismatch_ratio: float = 0.5
    ):
        # Store configuration
        self.data_dir = Path(data_dir)
        self.positive_ratio = positive_ratio
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.structure_extensions = structure_extensions
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.lazy_loading = lazy_loading
        self.include_coordinates = include_coordinates
        self.max_files = max_files
        self.extract_backbone_features = extract_backbone_features
        self.mpnn_model_path = mpnn_model_path
        
        # Validate and store coordinate/sequence mismatch parameters
        if not 0.0 <= max_coord_seq_mismatch_ratio <= 1.0:
            raise ValueError(f"max_coord_seq_mismatch_ratio must be a float between 0.0 and 1.0, got {max_coord_seq_mismatch_ratio} ({type(max_coord_seq_mismatch_ratio)})")
        self.max_coord_seq_mismatch_ratio = max_coord_seq_mismatch_ratio
        
        # Initialize backbone encoder if requested and available
        self.backbone_encoder = None
        if self.extract_backbone_features:
            if not MPNN_ENCODER_AVAILABLE:
                warnings.warn(
                    "Backbone features extraction requested but MPNN encoder not available. "
                    "Install ProteinMPNN dependencies and ensure the mpnn_encoder module is accessible."
                )
            elif not PROTEINMPNN_AVAILABLE:
                warnings.warn(
                    "Backbone features extraction requested but ProteinMPNN utilities not available. "
                    "Check that proteinmpnn directory exists and protein_mpnn_utils.py is accessible."
                )
        
        if self.extract_backbone_features and MPNN_ENCODER_AVAILABLE and PROTEINMPNN_AVAILABLE:
            model_path = mpnn_model_path
            
            # Fallback to default model path if none provided
            if not model_path:
                # Try to find default model in proteinmpnn directory
                default_paths = [
                    str(Path(__file__).parent.parent.parent / "proteinmpnn" / "vanilla_model_weights" / "v_48_020.pt"),
                    str(Path(__file__).parent.parent.parent / "proteinmpnn" / "v_48_020.pt")
                ]
                for path in default_paths:
                    if Path(path).exists():
                        model_path = path
                        break
            
            if model_path and Path(model_path).exists():
                try:
                    self.backbone_encoder = ProteinMPNNBackboneEncoder(
                        pretrained_ckpt_path=model_path,
                        freeze_layers=True
                    )
                    self.backbone_encoder.eval()
                except Exception as e:
                    warnings.warn(f"Failed to initialize backbone encoder: {e}")
            else:
                if self.extract_backbone_features:
                    warnings.warn("Backbone features extraction requested but no valid model path found")
        
        # Default negative generation methods
        if negative_methods is None:
            negative_methods = ['random', 'mutations', 'failed_designs']
        self.negative_methods = negative_methods
        
        # Validate configuration
        self._validate_config()
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Initialize data structures
        self.structure_files = []
        self.positive_samples = []
        self.negative_samples = []
        self.samples = []  # Combined list of (sample_data, label) tuples
        
        # Amino acid properties for sequence generation
        self._init_amino_acid_properties()
        
        # Load and process data
        self._discover_structure_files()
        if not self.lazy_loading:
            self._load_dataset()
        else:
            # For lazy loading, just initialize empty lists
            self.positive_samples = []
            self.negative_samples = []
            self.samples = []
            self._lazy_loaded = False
        
        # Initialize thread locks for safe multi-worker DataLoader usage
        self._lazy_loading_lock = threading.Lock()
        
        print(f"StabilityDataset initialized:")
        print(f"  {len(self.structure_files)} structure files found")
        print(f"  {len(self.positive_samples)} positive samples")
        print(f"  {len(self.negative_samples)} negative samples") 
        print(f"  {len(self.samples)} total samples")
        print(f"  Positive ratio: {self.positive_ratio:.3f}")
    
    def _validate_config(self):
        """Validate dataset configuration parameters"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        if not 0 < self.positive_ratio < 1:
            raise ValueError(f"positive_ratio must be in (0, 1), got {self.positive_ratio}")
        
        if self.min_sequence_length <= 0:
            raise ValueError(f"min_sequence_length must be positive, got {self.min_sequence_length}")
        
        if self.max_sequence_length <= self.min_sequence_length:
            raise ValueError(f"max_sequence_length ({self.max_sequence_length}) must be > min_sequence_length ({self.min_sequence_length})")
        
        if not self.structure_extensions:
            raise ValueError("structure_extensions cannot be empty")
        
        if self.cache_dir and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not PROTEINMPNN_AVAILABLE and 'failed_designs' in self.negative_methods:
            warnings.warn("ProteinMPNN not available, removing 'failed_designs' from negative methods")
            self.negative_methods = [m for m in self.negative_methods if m != 'failed_designs']
    
    def _init_amino_acid_properties(self):
        """Initialize amino acid properties for sequence generation"""
        # Standard amino acid alphabet (20 canonical)
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {i: aa for i, aa in enumerate(self.amino_acids)}
        
        # Natural amino acid frequencies (from Swiss-Prot)
        self.natural_frequencies = {
            'A': 0.0825, 'C': 0.0137, 'D': 0.0545, 'E': 0.0674, 'F': 0.0386,
            'G': 0.0708, 'H': 0.0227, 'I': 0.0593, 'K': 0.0584, 'L': 0.0966,
            'M': 0.0242, 'N': 0.0406, 'P': 0.0470, 'Q': 0.0393, 'R': 0.0553,
            'S': 0.0656, 'T': 0.0534, 'V': 0.0686, 'W': 0.0108, 'Y': 0.0292
        }
        
        # Physicochemical properties for mutation generation
        self.hydrophobic = set("AILVMFPWY")
        self.polar = set("NQST")
        self.charged_positive = set("KRH")
        self.charged_negative = set("DE")
        self.aromatic = set("FWY")
        self.small = set("AGSV")
        
        # Conservative mutation groups (Blosum62 inspired)
        self.conservative_groups = [
            set("LVIM"),      # Hydrophobic aliphatic
            set("FWY"),       # Aromatic
            set("ST"),        # Small polar
            set("KR"),        # Positive
            set("DE"),        # Negative
            set("QN"),        # Amide
        ]
    
    def _discover_structure_files(self):
        """Find all structure files in the data directory"""
        self.structure_files = []
        
        # Validate and resolve data directory path to prevent traversal attacks
        try:
            allowed_dir = self.data_dir.resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid data directory path: {self.data_dir}") from e
        
        for ext in self.structure_extensions:
            pattern = f"*{ext}"
            files = list(self.data_dir.glob(pattern))
            if self.data_dir.is_dir():
                # Also search in subdirectories
                files.extend(list(self.data_dir.rglob(pattern)))
            
            # Validate each discovered file is within allowed directory
            for file_path in files:
                try:
                    resolved_file = file_path.resolve()
                    # Ensure the resolved file path is within the allowed directory
                    # Use Path.is_relative_to() for secure path checking (Python 3.9+)
                    try:
                        if not resolved_file.is_relative_to(allowed_dir):
                            warnings.warn(f"Rejecting file outside data directory: {file_path}")
                            continue
                    except AttributeError:
                        # Fallback for Python < 3.9 - use path comparison with separator
                        allowed_str = str(allowed_dir)
                        if not allowed_str.endswith(os.sep):
                            allowed_str += os.sep
                        if not str(resolved_file).startswith(allowed_str):
                            warnings.warn(f"Rejecting file outside data directory: {file_path}")
                            continue
                    self.structure_files.append(file_path)
                except (OSError, ValueError):
                    warnings.warn(f"Skipping invalid file path: {file_path}")
                    continue
        
        # Remove duplicates and sort
        self.structure_files = sorted(list(set(self.structure_files)))
        
        # Limit number of files if specified
        if self.max_files:
            self.structure_files = self.structure_files[:self.max_files]
        
        if not self.structure_files:
            raise ValueError(f"No structure files found in {self.data_dir} with extensions {self.structure_extensions}")
        
        print(f"Found {len(self.structure_files)} structure files")
    
    def _load_dataset(self):
        """Load and process the complete dataset"""
        print("Loading positive samples from structures...")
        self._load_positive_samples()
        
        print("Generating negative samples...")
        self._generate_negative_samples()
        
        print("Combining samples and balancing dataset...")
        self._combine_and_balance_samples()
        
        print("Dataset loading complete")
    
    def _load_positive_samples(self):
        """Load positive samples (native sequences) from structure files"""
        self.positive_samples = []
        
        for struct_file in tqdm(self.structure_files, desc="Processing structures"):
            try:
                sample = self._process_structure_file(struct_file)
                if sample:
                    self.positive_samples.append(sample)
            except FileNotFoundError:
                warnings.warn(f"Structure file not found: {struct_file}")
                continue
            except PermissionError:
                warnings.warn(f"Permission denied accessing: {struct_file}")
                continue
            except Exception as e:
                warnings.warn(f"Unexpected error processing {struct_file}: {type(e).__name__}: {e}")
                continue
    
    def _process_structure_file(self, struct_file: Path) -> Optional[Dict[str, Any]]:
        """Process a single structure file into a training sample"""
        try:
            if PROTEINMPNN_AVAILABLE:
                # Use ProteinMPNN's parsing function
                # Request all 4 backbone atoms: N, CA, C, O
                xyz, seq = parse_PDB_biounits(str(struct_file), atoms=['N', 'CA', 'C', 'O'])
                
                # Check if parsing failed
                if isinstance(xyz, str) and xyz == 'no_chain':
                    return None
                
                # xyz and seq are returned as numpy arrays
                # seq is a list/array of sequences (one per chain)
                if len(seq) == 0 or not seq[0]:
                    return None
                
                # Use first chain
                coordinates = xyz  # [L, 4, 3] format (N, CA, C, O atoms)
                sequence = seq[0]  # First sequence string
                first_chain = 'A'  # Default chain ID
                
            elif BIOPYTHON_AVAILABLE:
                # Fallback to BioPython parsing
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('protein', struct_file)
                
                # Extract coordinates and sequence from first chain
                model = structure[0]
                chains = list(model.get_chains())
                if not chains:
                    warnings.warn(f"No chains found in {struct_file}")
                    return None
                
                chain = chains[0]
                coordinates, sequence = self._extract_chain_data(chain)
                first_chain = chain.id
            else:
                warnings.warn(f"No PDB parsing libraries available for {struct_file}")
                return None
            
            # Validate sequence
            if not sequence or not isinstance(sequence, str):
                warnings.warn(f"Empty or invalid sequence in {struct_file}")
                return None
            
            # Clean sequence and validate amino acids
            sequence = self._clean_and_validate_sequence(sequence)
            if not sequence:
                warnings.warn(f"No valid amino acids found in {struct_file}")
                return None
            
            # Validate sequence length
            seq_len = len(sequence)
            if not (self.min_sequence_length <= seq_len <= self.max_sequence_length):
                return None
            
            # Validate structure quality if coordinates are included
            if self.include_coordinates and coordinates is not None:
                if not self._validate_structure_quality(coordinates, sequence):
                    warnings.warn(f"Structure quality validation failed for {struct_file} (seq_len: {len(sequence)}, coord_shape: {coordinates.shape})")
                    return None
            
            # Create sample data structure
            sample = {
                'sequence': sequence,
                'coordinates': coordinates if self.include_coordinates else None,
                'structure_file': str(struct_file),
                'chain_id': first_chain,
                'length': seq_len,
                'label': 1  # Positive sample
            }
            
            return sample
            
        except Exception as e:
            warnings.warn(f"Error processing {struct_file}: {e}")
            return None
    
    def _extract_chain_data(self, chain: Chain) -> Tuple[np.ndarray, str]:
        """
        Extract coordinates and sequence from a BioPython chain.
        
        Only processes standard amino acid residues, filtering out:
        - Heteroatoms (waters, ligands, ions) 
        - Non-standard residues
        - Residues with missing backbone atoms
        
        Args:
            chain: BioPython Chain object
            
        Returns:
            Tuple of (coordinates array [L, 4, 3], sequence string)
        """
        coords = []
        sequence = []
        
        for residue in chain:
            # Skip heteroatoms and non-standard residues
            # In BioPython PDB format, residue.id[0] is the "hetero flag":
            # ' ' = standard amino acid, 'H' = heteroatom, 'W' = water
            if residue.id[0] != ' ':
                continue
            
            # Get backbone atoms (N, CA, C, O)
            try:
                n = residue['N'].coord
                ca = residue['CA'].coord
                c = residue['C'].coord
                o = residue['O'].coord
                
                coords.append([n, ca, c, o])
                
                # Convert 3-letter to 1-letter amino acid code
                try:
                    aa_code = seq1(residue.resname)
                    if aa_code in self.amino_acids:  # Only standard amino acids
                        sequence.append(aa_code)
                    else:
                        # Skip non-standard amino acids
                        coords.pop()  # Remove the coordinates we just added
                        continue
                except Exception:
                    # Skip unknown residues
                    coords.pop()
                    continue
                
            except KeyError:
                # Skip residues with missing backbone atoms
                continue
        
        if not coords:
            coordinates = np.array([], dtype=np.float32).reshape(0, 4, 3)
        else:
            coordinates = np.array(coords, dtype=np.float32)  # [L, 4, 3]
        
        sequence = ''.join(sequence)
        
        return coordinates, sequence
    
    def _generate_negative_samples(self):
        """Generate negative samples using various methods"""
        self.negative_samples = []
        
        # Determine how many negative samples to generate
        n_positive = len(self.positive_samples)
        n_negative = int(n_positive * (1 - self.positive_ratio) / self.positive_ratio)
        
        # Distribute negative samples across methods
        n_per_method = max(1, n_negative // len(self.negative_methods))
        
        for method in self.negative_methods:
            if method == 'random':
                samples = self._generate_random_sequences(n_per_method)
            elif method == 'mutations':
                samples = self._generate_destabilizing_mutations(n_per_method)
            elif method == 'failed_designs':
                samples = self._generate_failed_designs(n_per_method)
            else:
                warnings.warn(f"Unknown negative generation method: {method}")
                continue
            
            self.negative_samples.extend(samples)
        
        # Trim to exact count if needed
        if len(self.negative_samples) > n_negative:
            self.negative_samples = self.negative_samples[:n_negative]
    
    def _generate_random_sequences(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate random sequences with realistic but problematic amino acid patterns"""
        samples = []
        
        for _ in range(n_samples):
            # Sample sequence length from positive examples
            if self.positive_samples:
                ref_sample = self.rng.choice(self.positive_samples)
                target_length = len(ref_sample['sequence'])
                # Add some variation
                length = self.rng.randint(
                    max(self.min_sequence_length, target_length - 10),
                    min(self.max_sequence_length, target_length + 10)
                )
                ref_coords = ref_sample.get('coordinates')
            else:
                length = self.rng.randint(self.min_sequence_length, self.max_sequence_length)
                ref_coords = None
            
            # Generate problematic but realistic sequence
            sequence = self._generate_problematic_sequence(length)
            
            sample = {
                'sequence': sequence,
                'coordinates': ref_coords,  # Use template structure if available
                'structure_file': ref_sample.get('structure_file') if self.positive_samples else None,
                'chain_id': ref_sample.get('chain_id') if self.positive_samples else None,
                'length': length,
                'label': 0,  # Negative sample
                'generation_method': 'random_problematic',
                'problem_types': self._identify_sequence_problems(sequence)
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_problematic_sequence(self, length: int) -> str:
        """Generate a sequence with realistic but stability-compromising patterns"""
        sequence = []
        
        # Choose a problematic pattern strategy
        strategies = [
            'hydrophobic_clusters',
            'charge_clusters', 
            'proline_excess',
            'aromatic_stacking',
            'glycine_excess',
            'mixed_patterns'
        ]
        
        strategy = self.rng.choice(strategies)
        
        if strategy == 'hydrophobic_clusters':
            # Create large hydrophobic patches (aggregation-prone)
            for i in range(length):
                if self.rng.random() < 0.7:  # 70% hydrophobic
                    sequence.append(self.rng.choice(['A', 'I', 'L', 'V', 'F', 'W', 'Y']))
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        elif strategy == 'charge_clusters':
            # Alternate between highly positive and negative regions
            cluster_size = max(3, length // 8)
            current_charge = 'positive'
            for i in range(length):
                if i % cluster_size == 0:
                    current_charge = 'positive' if current_charge == 'negative' else 'negative'
                
                if current_charge == 'positive' and self.rng.random() < 0.8:
                    sequence.append(self.rng.choice(['K', 'R', 'H']))
                elif current_charge == 'negative' and self.rng.random() < 0.8:
                    sequence.append(self.rng.choice(['D', 'E']))
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        elif strategy == 'proline_excess':
            # High proline content (structure disruption)
            for i in range(length):
                if self.rng.random() < 0.25:  # 25% proline
                    sequence.append('P')
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        elif strategy == 'aromatic_stacking':
            # Excessive aromatic residues (aggregation/misfolding)
            for i in range(length):
                if self.rng.random() < 0.4:  # 40% aromatic
                    sequence.append(self.rng.choice(['F', 'W', 'Y']))
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        elif strategy == 'glycine_excess':
            # Too much flexibility (no stable structure)
            for i in range(length):
                if self.rng.random() < 0.35:  # 35% glycine
                    sequence.append('G')
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        else:  # mixed_patterns
            # Combination of multiple problems
            for i in range(length):
                if i % 4 == 0:  # Hydrophobic patches
                    sequence.append(self.rng.choice(['I', 'L', 'V', 'F']))
                elif i % 4 == 1:  # Charged residues
                    sequence.append(self.rng.choice(['K', 'R', 'D', 'E']))
                elif i % 4 == 2:  # Structure disruptors
                    sequence.append(self.rng.choice(['P', 'G']))
                else:  # Natural distribution
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        return ''.join(sequence)
    
    def _identify_sequence_problems(self, sequence: str) -> List[str]:
        """Identify potential stability problems in a sequence"""
        problems = []
        
        # Calculate composition
        composition = {aa: sequence.count(aa) / len(sequence) for aa in set(sequence)}
        
        # Check for composition problems
        if composition.get('P', 0) > 0.15:  # > 15% proline
            problems.append('excessive_proline')
        
        if composition.get('G', 0) > 0.20:  # > 20% glycine
            problems.append('excessive_glycine')
        
        # Check hydrophobic content
        hydrophobic_count = sum(sequence.count(aa) for aa in self.hydrophobic)
        if hydrophobic_count / len(sequence) > 0.6:  # > 60% hydrophobic
            problems.append('excessive_hydrophobic')
        
        # Check charge clustering
        positive_patches = []
        negative_patches = []
        current_pos_patch = 0
        current_neg_patch = 0
        
        for aa in sequence:
            if aa in self.charged_positive:
                current_pos_patch += 1
                current_neg_patch = 0
            elif aa in self.charged_negative:
                current_neg_patch += 1
                current_pos_patch = 0
            else:
                if current_pos_patch >= 3:
                    positive_patches.append(current_pos_patch)
                if current_neg_patch >= 3:
                    negative_patches.append(current_neg_patch)
                current_pos_patch = 0
                current_neg_patch = 0
        
        if positive_patches or negative_patches:
            problems.append('charge_clustering')
        
        # Check aromatic clustering
        aromatic_count = sum(sequence.count(aa) for aa in self.aromatic)
        if aromatic_count / len(sequence) > 0.25:  # > 25% aromatic
            problems.append('excessive_aromatic')
        
        return problems
    
    def _generate_destabilizing_mutations(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate sequences with structurally-informed destabilizing mutations"""
        samples = []
        
        if not self.positive_samples:
            return samples
        
        for _ in range(n_samples):
            # Select random positive sample as template
            template = self.rng.choice(self.positive_samples)
            sequence = list(template['sequence'])
            coordinates = template.get('coordinates')
            
            # Determine mutation strategy based on structure availability
            if coordinates is not None and coordinates.shape[0] > 0:
                mutations = self._generate_structure_aware_mutations(sequence, coordinates)
            else:
                mutations = self._generate_sequence_based_mutations(sequence)
            
            # Apply mutations
            mutation_records = []
            for pos, new_aa, reason in mutations:
                original_aa = sequence[pos]
                sequence[pos] = new_aa
                mutation_records.append({
                    'position': pos,
                    'original': original_aa,
                    'mutant': new_aa,
                    'reason': reason
                })
            
            mutated_sequence = ''.join(sequence)
            
            sample = {
                'sequence': mutated_sequence,
                'coordinates': template['coordinates'],
                'structure_file': template['structure_file'],
                'chain_id': template['chain_id'],
                'length': len(mutated_sequence),
                'label': 0,  # Negative sample
                'generation_method': 'structure_aware_mutations',
                'template_sample': template,
                'mutations': mutation_records,
                'n_mutations': len(mutation_records)
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_structure_aware_mutations(self, sequence: List[str], 
                                          coordinates: np.ndarray) -> List[Tuple[int, str, str]]:
        """Generate mutations based on structural context"""
        mutations = []
        seq_len = len(sequence)
        
        # Number of mutations (3-15% of sequence length)
        n_mutations = max(1, int(self.rng.uniform(0.03, 0.15) * seq_len))
        
        # Calculate structural features
        burial_scores = self._estimate_burial(coordinates)
        flexibility_scores = self._estimate_flexibility(coordinates)
        
        # Select positions for mutation based on structural context
        mutation_positions = self._select_mutation_positions(
            sequence, burial_scores, flexibility_scores, n_mutations
        )
        
        for pos in mutation_positions:
            original_aa = sequence[pos]
            burial = burial_scores[pos]
            flexibility = flexibility_scores[pos]
            
            # Choose mutation based on structural context
            if burial > 0.7:  # Buried position
                if original_aa in self.hydrophobic:
                    # Bury a charged residue (very destabilizing)
                    new_aa = self.rng.choice(['K', 'R', 'D', 'E'])
                    reason = 'buried_charge'
                elif original_aa in self.small:
                    # Replace small residue with bulky one (steric clash)
                    new_aa = self.rng.choice(['W', 'Y', 'F', 'R'])
                    reason = 'buried_clash'
                else:
                    # General destabilizing mutation in buried region
                    new_aa = self.rng.choice(['P', 'G'])  # Structure disruptors
                    reason = 'buried_disruptor'
            
            elif burial < 0.3:  # Surface position
                if original_aa in self.charged_positive | self.charged_negative:
                    # Replace charged with hydrophobic (hydrophobic patch)
                    new_aa = self.rng.choice(['I', 'L', 'V', 'F'])
                    reason = 'surface_hydrophobic'
                elif original_aa in self.polar:
                    # Replace polar with aromatic (potential aggregation)
                    new_aa = self.rng.choice(['F', 'W', 'Y'])
                    reason = 'surface_aromatic'
                else:
                    # Introduce charged clusters
                    new_aa = self.rng.choice(['K', 'K', 'R', 'D', 'D', 'E'])  # Bias toward charged
                    reason = 'charge_clustering'
            
            else:  # Intermediate burial
                if flexibility > 0.6:  # Flexible region
                    # Introduce rigidity (proline)
                    new_aa = 'P'
                    reason = 'flexibility_disruption'
                elif flexibility < 0.3:  # Rigid region
                    # Introduce excess flexibility
                    new_aa = 'G'
                    reason = 'rigidity_disruption'
                else:
                    # General destabilizing mutation
                    candidates = self._get_destabilizing_candidates(original_aa)
                    new_aa = self.rng.choice(candidates) if candidates else 'A'
                    reason = 'general_destabilizing'
            
            mutations.append((pos, new_aa, reason))
        
        return mutations
    
    def _generate_sequence_based_mutations(self, sequence: List[str]) -> List[Tuple[int, str, str]]:
        """Generate mutations based on sequence patterns (when structure unavailable)"""
        mutations = []
        seq_len = len(sequence)
        
        # Number of mutations (5-20% of sequence length)
        n_mutations = max(1, int(self.rng.uniform(0.05, 0.2) * seq_len))
        
        # Select random positions
        positions = self.rng.choice(seq_len, size=n_mutations, replace=False)
        
        for pos in positions:
            original_aa = sequence[pos]
            
            # Choose destabilizing mutation based on amino acid properties
            candidates = self._get_destabilizing_candidates(original_aa)
            new_aa = self.rng.choice(candidates) if candidates else 'A'
            
            mutations.append((pos, new_aa, 'sequence_based'))
        
        return mutations
    
    def _estimate_burial(self, coordinates: np.ndarray) -> np.ndarray:
        """Estimate burial score for each residue (0=surface, 1=buried)"""
        if coordinates.shape[0] == 0:
            return np.array([])
        
        ca_coords = coordinates[:, 1, :]  # CA atoms
        burial_scores = np.zeros(len(ca_coords))
        
        # Simple burial estimation based on number of neighbors
        for i, coord_i in enumerate(ca_coords):
            distances = np.linalg.norm(ca_coords - coord_i, axis=1)
            # Count neighbors within 10 Å
            neighbors = np.sum((distances > 0) & (distances < 10.0))
            # Normalize to [0,1] range (assume max 20 neighbors = fully buried)
            burial_scores[i] = min(neighbors / 20.0, 1.0)
        
        return burial_scores
    
    def _estimate_flexibility(self, coordinates: np.ndarray) -> np.ndarray:
        """Estimate flexibility score for each residue (0=rigid, 1=flexible)"""
        if coordinates.shape[0] <= 2:
            return np.ones(coordinates.shape[0])  # All flexible if too short
        
        ca_coords = coordinates[:, 1, :]  # CA atoms
        flexibility_scores = np.ones(len(ca_coords))
        
        # Estimate flexibility based on local geometry
        for i in range(1, len(ca_coords) - 1):
            # Calculate angles with neighbors
            v1 = ca_coords[i] - ca_coords[i-1]
            v2 = ca_coords[i+1] - ca_coords[i]
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Higher angles indicate more flexibility
            flexibility_scores[i] = abs(angle - np.pi/2) / (np.pi/2)  # Normalize
        
        return flexibility_scores
    
    def _select_mutation_positions(self, sequence: List[str], burial: np.ndarray, 
                                 flexibility: np.ndarray, n_mutations: int) -> List[int]:
        """Select positions for mutation based on structural features"""
        seq_len = len(sequence)
        
        # Create selection weights based on structural context
        weights = np.ones(seq_len)
        
        for i in range(seq_len):
            # Prefer positions that would be maximally destabilizing
            if i < len(burial):
                if burial[i] > 0.7:  # Buried
                    weights[i] *= 2.0  # Mutations here are very destabilizing
                elif burial[i] < 0.3:  # Surface  
                    weights[i] *= 1.5  # Moderately important
            
            if i < len(flexibility):
                # Slight preference for positions with intermediate flexibility
                flex_factor = 1.0 + 0.5 * abs(flexibility[i] - 0.5)
                weights[i] *= flex_factor
        
        # Select positions based on weights
        probabilities = weights / weights.sum()
        positions = self.rng.choice(
            seq_len, size=min(n_mutations, seq_len), 
            replace=False, p=probabilities
        )
        
        return positions.tolist()
    
    def _get_destabilizing_candidates(self, original_aa: str) -> List[str]:
        """Get destabilizing mutation candidates for a given amino acid"""
        destabilizing = []
        
        if original_aa in self.hydrophobic:
            # Hydrophobic -> charged/polar (polarity clash)
            destabilizing.extend(['K', 'R', 'D', 'E', 'N', 'Q'])
        elif original_aa in self.charged_positive:
            # Positive -> negative (charge flip)
            destabilizing.extend(['D', 'E'])
            # Positive -> hydrophobic (charge loss)
            destabilizing.extend(['I', 'L', 'V', 'F'])
        elif original_aa in self.charged_negative:
            # Negative -> positive (charge flip)
            destabilizing.extend(['K', 'R'])
            # Negative -> hydrophobic (charge loss)
            destabilizing.extend(['I', 'L', 'V', 'F'])
        elif original_aa in self.polar:
            # Polar -> hydrophobic (loss of H-bonding)
            destabilizing.extend(['I', 'L', 'V', 'A'])
        elif original_aa in self.aromatic:
            # Aromatic -> charged (different properties)
            destabilizing.extend(['K', 'R', 'D', 'E'])
        elif original_aa in self.small:
            # Small -> bulky (steric clash)
            destabilizing.extend(['W', 'Y', 'F', 'R', 'K'])
        
        # General destabilizing options
        destabilizing.extend(['P', 'G'])  # Structure disruptors
        
        # Remove original amino acid
        destabilizing = [aa for aa in destabilizing if aa != original_aa]
        
        # Fall back to random if no specific candidates
        if not destabilizing:
            destabilizing = [aa for aa in self.amino_acids if aa != original_aa]
        
        return destabilizing
    
    def _generate_failed_designs(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate sequences that mimic common protein design failures"""
        samples = []
        
        # Define common failure modes in protein design
        failure_modes = [
            'low_confidence_bias',      # Bias toward uncertain amino acids
            'repeat_patterns',          # Repetitive sequences that models get stuck on
            'incompatible_constraints', # Sequences that satisfy some but not all constraints
            'boundary_artifacts',       # Edge effects from model training
            'overfitting_artifacts'     # Sequences that exploit model biases
        ]
        
        for _ in range(n_samples):
            if self.positive_samples:
                ref_sample = self.rng.choice(self.positive_samples)
                length = len(ref_sample['sequence'])
                coordinates = ref_sample['coordinates']
                structure_file = ref_sample['structure_file']
                chain_id = ref_sample['chain_id']
            else:
                length = self.rng.randint(self.min_sequence_length, self.max_sequence_length)
                coordinates = None
                structure_file = None
                chain_id = None
            
            # Choose failure mode
            failure_mode = self.rng.choice(failure_modes)
            sequence = self._generate_failed_sequence(length, failure_mode, ref_sample if self.positive_samples else None)
            
            sample = {
                'sequence': sequence,
                'coordinates': coordinates,
                'structure_file': structure_file,
                'chain_id': chain_id,
                'length': length,
                'label': 0,  # Negative sample
                'generation_method': 'failed_designs',
                'failure_mode': failure_mode,
                'template_sample': ref_sample if self.positive_samples else None
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_failed_sequence(self, length: int, failure_mode: str, 
                                template_sample: Optional[Dict] = None) -> str:
        """Generate a sequence exhibiting a specific failure mode"""
        
        if failure_mode == 'low_confidence_bias':
            # Mimic model uncertainty by biasing toward "safe" amino acids
            # Models often default to common, neutral amino acids when uncertain
            safe_bias = {
                'A': 0.25, 'L': 0.20, 'S': 0.15, 'G': 0.15,  # Very common/safe
                'V': 0.10, 'T': 0.10, 'I': 0.05              # Moderately common
            }
            sequence = []
            for _ in range(length):
                if self.rng.random() < 0.7:  # 70% chance of "safe" AA
                    aa = self.rng.choice(list(safe_bias.keys()), p=list(safe_bias.values()))
                else:
                    aa = self.rng.choice(list(self.natural_frequencies.keys()))
                sequence.append(aa)
            
        elif failure_mode == 'repeat_patterns':
            # Generate sequences with repetitive motifs (model gets stuck)
            motifs = ['ALG', 'VLI', 'KRK', 'DED', 'GPG', 'STS', 'FWY']
            motif = self.rng.choice(motifs)
            sequence = []
            
            i = 0
            while i < length:
                if self.rng.random() < 0.6:  # 60% chance of motif
                    sequence.extend(list(motif))
                    i += len(motif)
                else:
                    sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
                    i += 1
            
            # Trim to exact length
            sequence = sequence[:length]
            
        elif failure_mode == 'incompatible_constraints':
            # Create sequences that partially satisfy conflicting requirements
            # E.g., both hydrophobic (stability) and charged (solubility)
            sequence = []
            for i in range(length):
                if i % 2 == 0:  # Alternating pattern
                    # Hydrophobic for "stability"
                    sequence.append(self.rng.choice(['I', 'L', 'V', 'F']))
                else:
                    # Charged for "solubility"
                    sequence.append(self.rng.choice(['K', 'R', 'D', 'E']))
            
        elif failure_mode == 'boundary_artifacts':
            # Edge effects - biased composition at sequence ends
            sequence = [None] * length
            
            # Biased N-terminus (models often struggle with termini)
            n_term_bias = ['M', 'A', 'S', 'T']  # Common N-terminal residues in datasets
            for i in range(min(3, length)):
                sequence[i] = self.rng.choice(n_term_bias)
            
            # Biased C-terminus
            c_term_bias = ['A', 'G', 'S']  # Common C-terminal residues
            for i in range(max(0, length-3), length):
                if sequence[i] is None:
                    sequence[i] = self.rng.choice(c_term_bias)
            
            # Fill middle with natural distribution
            for i in range(length):
                if sequence[i] is None:
                    sequence[i] = self.rng.choice(list(self.natural_frequencies.keys()))
            
        else:  # overfitting_artifacts
            # Exploit known model biases (e.g., over-representation of certain patterns)
            if template_sample and 'sequence' in template_sample:
                # Start with template and add "model-preferred" mutations
                template_seq = template_sample['sequence']
                sequence = list(template_seq[:length])  # Trim if needed
                
                # Add model-biased mutations (models often favor certain substitutions)
                n_mutations = max(1, length // 10)  # 10% mutations
                positions = self.rng.choice(len(sequence), size=n_mutations, replace=False)
                
                model_preferred = {
                    # Common substitutions models make incorrectly
                    'A': ['S', 'G', 'V'],
                    'V': ['I', 'L', 'A'],
                    'L': ['I', 'V', 'M'],
                    'S': ['T', 'A', 'G'],
                    'T': ['S', 'A', 'V'],
                    'K': ['R', 'Q'],
                    'R': ['K', 'Q'],
                    'D': ['E', 'N'],
                    'E': ['D', 'Q']
                }
                
                for pos in positions:
                    original = sequence[pos]
                    if original in model_preferred:
                        sequence[pos] = self.rng.choice(model_preferred[original])
            else:
                # Generate from scratch with model biases
                sequence = []
                for _ in range(length):
                    # Bias toward amino acids models "prefer"
                    if self.rng.random() < 0.4:
                        # Models often over-predict common hydrophobic residues
                        sequence.append(self.rng.choice(['L', 'A', 'V', 'I']))
                    else:
                        sequence.append(self.rng.choice(list(self.natural_frequencies.keys())))
        
        return ''.join(sequence)
    
    def apply_data_augmentation(self, sample: Dict[str, Any], 
                              augmentation_types: Optional[List[str]] = None,
                              augmentation_strength: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply data augmentation to increase training diversity while preserving biological realism.
        
        Args:
            sample: Original sample dictionary
            augmentation_types: Types of augmentation to apply (None = all available)
            augmentation_strength: Strength of augmentation (0.0 = minimal, 1.0 = maximal)
            
        Returns:
            List of augmented samples (including original)
        """
        if augmentation_types is None:
            augmentation_types = [
                'conservative_mutations',
                'length_variations', 
                'chain_masking',
                'sequence_permutations'
            ]
        
        augmented_samples = [sample.copy()]  # Include original
        
        for aug_type in augmentation_types:
            try:
                if aug_type == 'conservative_mutations':
                    aug_sample = self._apply_conservative_mutations(sample, augmentation_strength)
                elif aug_type == 'length_variations':
                    aug_sample = self._apply_length_variations(sample, augmentation_strength)
                elif aug_type == 'chain_masking':
                    aug_sample = self._apply_chain_masking(sample, augmentation_strength)
                elif aug_type == 'sequence_permutations':
                    aug_sample = self._apply_sequence_permutations(sample, augmentation_strength)
                else:
                    warnings.warn(f"Unknown augmentation type: {aug_type}")
                    continue
                
                if aug_sample:
                    aug_sample['augmentation_type'] = aug_type
                    aug_sample['augmentation_strength'] = augmentation_strength
                    aug_sample['parent_sample'] = sample
                    augmented_samples.append(aug_sample)
                    
            except Exception as e:
                warnings.warn(f"Failed to apply {aug_type} augmentation: {e}")
                continue
        
        return augmented_samples
    
    def _apply_conservative_mutations(self, sample: Dict[str, Any], 
                                    strength: float) -> Optional[Dict[str, Any]]:
        """Apply conservative mutations within physicochemical groups"""
        sequence = sample['sequence']
        if not sequence:
            return None
        
        # Number of mutations based on strength (1-10% of sequence length)
        max_mutations = max(1, int(0.1 * len(sequence) * strength))
        n_mutations = self.rng.randint(1, max_mutations + 1)
        
        # Select positions for mutation
        positions = self.rng.choice(len(sequence), size=min(n_mutations, len(sequence)), replace=False)
        
        new_sequence = list(sequence)
        mutations = []
        
        for pos in positions:
            original_aa = sequence[pos]
            conservative_options = self._get_conservative_mutations(original_aa)
            
            if conservative_options:
                new_aa = self.rng.choice(conservative_options)
                new_sequence[pos] = new_aa
                mutations.append({
                    'position': pos,
                    'original': original_aa,
                    'mutant': new_aa,
                    'type': 'conservative'
                })
        
        # Create augmented sample
        aug_sample = sample.copy()
        aug_sample['sequence'] = ''.join(new_sequence)
        aug_sample['length'] = len(new_sequence)
        aug_sample['mutations'] = mutations
        
        return aug_sample
    
    def _get_conservative_mutations(self, amino_acid: str) -> List[str]:
        """Get conservative mutation options for an amino acid"""
        # Conservative substitution groups (maintain similar properties)
        conservative_map = {
            # Hydrophobic aliphatic
            'A': ['V', 'I', 'L'], 'V': ['A', 'I', 'L'], 'I': ['V', 'L', 'M'], 'L': ['I', 'V', 'M'],
            'M': ['I', 'L'],
            
            # Aromatic
            'F': ['Y', 'W'], 'Y': ['F', 'W'], 'W': ['F', 'Y'],
            
            # Polar uncharged
            'S': ['T'], 'T': ['S'], 
            'N': ['Q'], 'Q': ['N'],
            'C': [],  # Cysteine is unique (disulfide bonds)
            
            # Positively charged
            'K': ['R'], 'R': ['K', 'H'], 'H': ['R'],
            
            # Negatively charged  
            'D': ['E'], 'E': ['D'],
            
            # Special cases
            'G': [],  # Glycine is unique (smallest, most flexible)
            'P': []   # Proline is unique (rigid)
        }
        
        return conservative_map.get(amino_acid, [])
    
    def _apply_length_variations(self, sample: Dict[str, Any], 
                               strength: float) -> Optional[Dict[str, Any]]:
        """Apply length variations (insertions/deletions in flexible regions)"""
        sequence = sample['sequence']
        coordinates = sample.get('coordinates')
        
        if not sequence or len(sequence) < 10:  # Too short for length variations
            return None
        
        # Determine flexible regions for safe insertions/deletions
        if coordinates is not None and coordinates.shape[0] > 0:
            flexibility = self._estimate_flexibility(coordinates)
            # Find regions with high flexibility (potential loops)
            flexible_positions = np.where(flexibility > 0.6)[0].tolist()
        else:
            # Fallback: assume termini and middle regions are more flexible
            flexible_positions = (list(range(min(3, len(sequence)))) + 
                                list(range(len(sequence)//3, 2*len(sequence)//3)) + 
                                list(range(max(0, len(sequence)-3), len(sequence))))
        
        if not flexible_positions:
            return None
        
        # Decide on insertion or deletion based on strength
        if self.rng.random() < 0.5:  # Insertion
            return self._apply_insertion(sample, flexible_positions, strength)
        else:  # Deletion
            return self._apply_deletion(sample, flexible_positions, strength)
    
    def _apply_insertion(self, sample: Dict[str, Any], flexible_positions: List[int], 
                       strength: float) -> Optional[Dict[str, Any]]:
        """Apply sequence insertion at flexible position"""
        sequence = sample['sequence']
        
        # Choose insertion position
        insert_pos = self.rng.choice(flexible_positions)
        
        # Insertion length (1-3 amino acids based on strength)
        max_insert_len = max(1, int(3 * strength))
        insert_len = self.rng.randint(1, max_insert_len + 1)
        
        # Generate insertion sequence (prefer flexible/loop amino acids)
        loop_amino_acids = ['G', 'S', 'T', 'A', 'N', 'D']  # Common in loops
        insert_sequence = ''.join(self.rng.choice(loop_amino_acids) for _ in range(insert_len))
        
        # Create new sequence
        new_sequence = sequence[:insert_pos] + insert_sequence + sequence[insert_pos:]
        
        # Create augmented sample
        aug_sample = sample.copy()
        aug_sample['sequence'] = new_sequence
        aug_sample['length'] = len(new_sequence)
        aug_sample['coordinates'] = None  # Coordinates invalid after insertion
        aug_sample['modification'] = {
            'type': 'insertion',
            'position': insert_pos,
            'inserted_sequence': insert_sequence,
            'length_change': insert_len
        }
        
        return aug_sample
    
    def _apply_deletion(self, sample: Dict[str, Any], flexible_positions: List[int], 
                      strength: float) -> Optional[Dict[str, Any]]:
        """Apply sequence deletion from flexible region"""
        sequence = sample['sequence']
        
        # Choose deletion start position
        delete_start = self.rng.choice(flexible_positions)
        
        # Deletion length (1-2 amino acids based on strength)
        max_delete_len = max(1, int(2 * strength))
        delete_len = self.rng.randint(1, min(max_delete_len + 1, len(sequence) - delete_start))
        
        # Ensure we don't delete too much
        if delete_start + delete_len >= len(sequence) or delete_len >= len(sequence) * 0.1:
            return None
        
        # Create new sequence
        new_sequence = sequence[:delete_start] + sequence[delete_start + delete_len:]
        
        # Create augmented sample
        aug_sample = sample.copy()
        aug_sample['sequence'] = new_sequence
        aug_sample['length'] = len(new_sequence)
        aug_sample['coordinates'] = None  # Coordinates invalid after deletion
        aug_sample['modification'] = {
            'type': 'deletion',
            'start_position': delete_start,
            'deleted_sequence': sequence[delete_start:delete_start + delete_len],
            'length_change': -delete_len
        }
        
        return aug_sample
    
    def _apply_chain_masking(self, sample: Dict[str, Any], 
                           strength: float) -> Optional[Dict[str, Any]]:
        """Apply chain masking for training robustness"""
        sequence = sample['sequence']
        
        if len(sequence) < 10:  # Too short for meaningful masking
            return None
        
        # Mask length based on strength (5-30% of sequence)
        mask_fraction = 0.05 + (0.25 * strength)  # 5-30%
        mask_length = max(1, int(len(sequence) * mask_fraction))
        
        # Choose masking strategy
        strategies = ['terminal_mask', 'internal_mask', 'random_mask']
        strategy = self.rng.choice(strategies)
        
        if strategy == 'terminal_mask':
            # Mask N or C terminus
            if self.rng.random() < 0.5:  # N-terminal
                masked_sequence = 'X' * mask_length + sequence[mask_length:]
                mask_positions = list(range(mask_length))
            else:  # C-terminal
                masked_sequence = sequence[:-mask_length] + 'X' * mask_length
                mask_positions = list(range(len(sequence) - mask_length, len(sequence)))
        
        elif strategy == 'internal_mask':
            # Mask contiguous internal region
            start_pos = self.rng.randint(0, len(sequence) - mask_length)
            masked_sequence = (sequence[:start_pos] + 
                             'X' * mask_length + 
                             sequence[start_pos + mask_length:])
            mask_positions = list(range(start_pos, start_pos + mask_length))
        
        else:  # random_mask
            # Mask random positions
            mask_positions = sorted(self.rng.choice(len(sequence), size=mask_length, replace=False))
            masked_sequence = list(sequence)
            for pos in mask_positions:
                masked_sequence[pos] = 'X'
            masked_sequence = ''.join(masked_sequence)
        
        # Create augmented sample
        aug_sample = sample.copy()
        aug_sample['sequence'] = masked_sequence
        aug_sample['original_sequence'] = sequence  # Keep for reference
        aug_sample['mask_info'] = {
            'strategy': strategy,
            'positions': mask_positions,
            'mask_fraction': mask_fraction
        }
        
        return aug_sample
    
    def _apply_sequence_permutations(self, sample: Dict[str, Any], 
                                   strength: float) -> Optional[Dict[str, Any]]:
        """Apply controlled sequence permutations in non-essential regions"""
        sequence = sample['sequence']
        coordinates = sample.get('coordinates')
        
        if len(sequence) < 15:  # Too short for meaningful permutation
            return None
        
        # Identify permutable regions (assume loop regions are more permutable)
        if coordinates is not None and coordinates.shape[0] > 0:
            flexibility = self._estimate_flexibility(coordinates)
            permutable_regions = self._identify_permutable_regions(sequence, flexibility)
        else:
            # Fallback: identify potential loop regions by sequence patterns
            permutable_regions = self._identify_loop_regions_by_sequence(sequence)
        
        if not permutable_regions:
            return None
        
        # Choose region to permute based on strength
        region = self.rng.choice(permutable_regions)
        region_start, region_end = region
        
        # Permutation strength determines how much to shuffle
        if strength < 0.3:
            # Minimal: small local swaps
            aug_sample = self._apply_local_swaps(sample, region_start, region_end)
        elif strength < 0.7:
            # Moderate: segment reversals
            aug_sample = self._apply_segment_reversal(sample, region_start, region_end)
        else:
            # Strong: full region shuffle
            aug_sample = self._apply_region_shuffle(sample, region_start, region_end)
        
        return aug_sample
    
    def _identify_permutable_regions(self, sequence: str, 
                                   flexibility: np.ndarray) -> List[Tuple[int, int]]:
        """Identify regions suitable for permutation based on flexibility"""
        if len(flexibility) != len(sequence):
            return []
        
        regions = []
        in_flexible_region = False
        region_start = 0
        
        for i, flex in enumerate(flexibility):
            if flex > 0.7:  # High flexibility threshold
                if not in_flexible_region:
                    region_start = i
                    in_flexible_region = True
            else:
                if in_flexible_region and i - region_start >= 5:  # Minimum region size
                    regions.append((region_start, i))
                in_flexible_region = False
        
        # Check final region
        if in_flexible_region and len(sequence) - region_start >= 5:
            regions.append((region_start, len(sequence)))
        
        return regions
    
    def _identify_loop_regions_by_sequence(self, sequence: str) -> List[Tuple[int, int]]:
        """Identify potential loop regions using sequence patterns"""
        # Simple heuristic: regions rich in loop-favoring amino acids
        loop_amino_acids = set(['G', 'S', 'T', 'N', 'D', 'P'])
        
        regions = []
        window_size = 10
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            loop_fraction = sum(1 for aa in window if aa in loop_amino_acids) / len(window)
            
            if loop_fraction > 0.4:  # > 40% loop amino acids
                regions.append((i, i + window_size))
        
        # Merge overlapping regions
        if regions:
            merged = [regions[0]]
            for start, end in regions[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            regions = merged
        
        return regions
    
    def _apply_local_swaps(self, sample: Dict[str, Any], 
                         region_start: int, region_end: int) -> Dict[str, Any]:
        """Apply small local amino acid swaps"""
        sequence = list(sample['sequence'])
        
        # Number of swaps (1-3 based on region size)
        region_size = region_end - region_start
        n_swaps = min(3, max(1, region_size // 5))
        
        swaps = []
        for _ in range(n_swaps):
            if region_end - region_start >= 2:
                pos1 = self.rng.randint(region_start, region_end - 1)
                pos2 = self.rng.randint(region_start, region_end - 1)
                
                if pos1 != pos2:
                    sequence[pos1], sequence[pos2] = sequence[pos2], sequence[pos1]
                    swaps.append((pos1, pos2))
        
        aug_sample = sample.copy()
        aug_sample['sequence'] = ''.join(sequence)
        aug_sample['coordinates'] = None  # Coordinates invalid after permutation
        aug_sample['permutation_info'] = {
            'type': 'local_swaps',
            'region': (region_start, region_end),
            'swaps': swaps
        }
        
        return aug_sample
    
    def _apply_segment_reversal(self, sample: Dict[str, Any], 
                              region_start: int, region_end: int) -> Dict[str, Any]:
        """Apply segment reversal within region"""
        sequence = list(sample['sequence'])
        
        # Choose segment to reverse (50-80% of region)
        region_size = region_end - region_start
        segment_size = max(3, int(region_size * self.rng.uniform(0.5, 0.8)))
        
        if segment_size < region_size:
            segment_start = region_start + self.rng.randint(0, region_size - segment_size)
            segment_end = segment_start + segment_size
        else:
            segment_start, segment_end = region_start, region_end
        
        # Reverse the segment
        segment = sequence[segment_start:segment_end]
        segment.reverse()
        sequence[segment_start:segment_end] = segment
        
        aug_sample = sample.copy()
        aug_sample['sequence'] = ''.join(sequence)
        aug_sample['coordinates'] = None  # Coordinates invalid after permutation
        aug_sample['permutation_info'] = {
            'type': 'segment_reversal',
            'region': (region_start, region_end),
            'reversed_segment': (segment_start, segment_end)
        }
        
        return aug_sample
    
    def _apply_region_shuffle(self, sample: Dict[str, Any], 
                            region_start: int, region_end: int) -> Dict[str, Any]:
        """Apply full shuffle within region"""
        sequence = list(sample['sequence'])
        
        # Extract region and shuffle
        region_sequence = sequence[region_start:region_end]
        self.rng.shuffle(region_sequence)
        sequence[region_start:region_end] = region_sequence
        
        aug_sample = sample.copy()
        aug_sample['sequence'] = ''.join(sequence)
        aug_sample['coordinates'] = None  # Coordinates invalid after permutation
        aug_sample['permutation_info'] = {
            'type': 'region_shuffle',
            'region': (region_start, region_end)
        }
        
        return aug_sample
    
    def load_native_sequences(self, pdb_files: List[Union[str, Path]], 
                            chain_ids: Optional[List[str]] = None,
                            validate_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Load native sequences from specific PDB files with enhanced validation.
        
        This is a utility function for loading positive samples from a curated
        set of PDB files, with additional validation options.
        
        Args:
            pdb_files: List of PDB file paths
            chain_ids: Optional list of specific chain IDs to extract
            validate_structure: Whether to perform structural validation
            
        Returns:
            List of sample dictionaries with sequences and metadata
        """
        native_samples = []
        
        for i, pdb_file in enumerate(tqdm(pdb_files, desc="Loading native sequences")):
            pdb_path = Path(pdb_file)
            
            if not pdb_path.exists():
                warnings.warn(f"PDB file not found: {pdb_path}")
                continue
            
            try:
                # Extract chain ID if provided
                target_chain = chain_ids[i] if chain_ids and i < len(chain_ids) else None
                
                samples = self._parse_pdb_structure(pdb_path, target_chain, validate_structure)
                native_samples.extend(samples)
                
            except Exception as e:
                warnings.warn(f"Failed to process {pdb_path}: {type(e).__name__}: {e}")
                continue
        
        return native_samples
    
    def _parse_pdb_structure(self, pdb_path: Path, 
                           target_chain: Optional[str] = None,
                           validate_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Parse a PDB structure and extract sequence/coordinate data.
        
        Args:
            pdb_path: Path to PDB file
            target_chain: Specific chain ID to extract (None = all chains)  
            validate_structure: Whether to perform structural validation
            
        Returns:
            List of sample dictionaries (one per chain)
        """
        samples = []
        
        try:
            if PROTEINMPNN_AVAILABLE:
                # Use ProteinMPNN's parsing - request all 4 backbone atoms
                xyz, seq = parse_PDB_biounits(str(pdb_path), atoms=['N', 'CA', 'C', 'O'])
                
                # Check if parsing failed
                if isinstance(xyz, str) and xyz == 'no_chain':
                    return samples
                
                # For single-chain PDBs, xyz and seq are arrays, not dicts
                # Convert to consistent format
                if not isinstance(seq, dict):
                    # Single chain case - seq is array of sequences
                    if len(seq) > 0:
                        sequence = seq[0]
                        sample = self._create_sample_dict(
                            sequence=sequence,
                            coordinates=xyz,
                            pdb_path=pdb_path,
                            chain_id='A',
                            validate_structure=validate_structure
                        )
                        if sample:
                            samples.append(sample)
                else:
                    # Multi-chain case (though parse_PDB_biounits doesn't return dicts)
                    for chain_id, coordinates in xyz.items():
                        if target_chain and chain_id != target_chain:
                            continue
                        
                        sequence = seq.get(chain_id, "")
                        if not sequence:
                            continue
                        
                        sample = self._create_sample_dict(
                            sequence=sequence,
                            coordinates=coordinates,
                            pdb_path=pdb_path,
                            chain_id=chain_id,
                            validate_structure=validate_structure
                        )
                        if sample:
                            samples.append(sample)
            
            elif BIOPYTHON_AVAILABLE:
                # Use BioPython parsing
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('protein', pdb_path)
                
                # Handle multiple models (take first)
                model = structure[0]
                
                for chain in model.get_chains():
                    if target_chain and chain.id != target_chain:
                        continue
                    
                    coordinates, sequence = self._extract_chain_data(chain)
                    
                    if not sequence:
                        continue
                    
                    sample = self._create_sample_dict(
                        sequence, coordinates, pdb_path, chain.id, validate_structure
                    )
                    
                    if sample:
                        samples.append(sample)
            else:
                warnings.warn(f"No PDB parsing libraries available for {pdb_path}")
        
        except Exception as e:
            warnings.warn(f"Error parsing PDB structure {pdb_path}: {e}")
        
        return samples
    
    def _create_sample_dict(self, sequence: str, coordinates: np.ndarray,
                          pdb_path: Path, chain_id: str,
                          validate_structure: bool = True) -> Optional[Dict[str, Any]]:
        """
        Create a standardized sample dictionary from sequence and structure data.
        
        Args:
            sequence: Amino acid sequence string
            coordinates: Coordinate array [L, 4, 3] 
            pdb_path: Path to source PDB file
            chain_id: Chain identifier
            validate_structure: Whether to validate structure quality
            
        Returns:
            Sample dictionary or None if validation fails
        """
        # Clean and validate sequence
        clean_sequence = self._clean_and_validate_sequence(sequence)
        if not clean_sequence:
            return None
        
        # Check sequence length constraints
        seq_len = len(clean_sequence)
        if not (self.min_sequence_length <= seq_len <= self.max_sequence_length):
            return None
        
        # Validate structure if requested
        if validate_structure and coordinates is not None:
            if not self._validate_structure_quality(coordinates, clean_sequence):
                warnings.warn(f"Structure quality validation failed for {pdb_path}:{chain_id}")
                return None
        
        # Create sample dictionary
        sample = {
            'sequence': clean_sequence,
            'coordinates': coordinates if self.include_coordinates else None,
            'structure_file': str(pdb_path),
            'chain_id': chain_id,
            'length': seq_len,
            'label': 1,  # Positive sample
            'source_type': 'native',
            'resolution': self._extract_resolution(pdb_path),
            'organism': self._extract_organism(pdb_path)
        }
        
        return sample
    
    def _validate_structure_quality(self, coordinates: np.ndarray, sequence: str) -> bool:
        """
        Validate basic structure quality metrics.
        
        Args:
            coordinates: Coordinate array [L, 4, 3]
            sequence: Amino acid sequence
            
        Returns:
            True if structure passes quality checks
        """
        if coordinates.shape[0] == 0:
            return False
        
        # Check coordinate/sequence length consistency  
        if coordinates.shape[0] != len(sequence):
            warnings.warn(f"Coordinate/sequence length mismatch: {coordinates.shape[0]} vs {len(sequence)}")
            return False
        
        # Check for reasonable coordinate values (not all zeros, not extreme)
        if np.all(coordinates == 0):
            return False
        
        # Check coordinate bounds (typical protein coordinates)
        coord_range = np.ptp(coordinates.reshape(-1, 3), axis=0)
        if np.any(coord_range > 1000):  # > 1000 Å is suspicious
            return False
        
        # Check for excessive gaps in backbone (broken chains)
        if coordinates.shape[0] > 1:
            ca_coords = coordinates[:, 1, :]  # CA atoms
            distances = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
            
            # Typical CA-CA distance is ~3.8 Å, > 10 Å suggests missing residues
            if np.any(distances > 10.0):
                gap_count = np.sum(distances > 10.0)
                if gap_count > len(sequence) * 0.1:  # > 10% gaps
                    return False
        
        return True
    
    def _extract_resolution(self, pdb_path: Path) -> Optional[float]:
        """Extract resolution from PDB file header (if available)"""
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('REMARK   2 RESOLUTION.'):
                        # Extract resolution value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'ANGSTROMS.' and i > 0:
                                try:
                                    return float(parts[i-1])
                                except ValueError:
                                    continue
                        break
                    elif line.startswith('ATOM') or line.startswith('HETATM'):
                        # Stop searching once we hit coordinate data
                        break
        except Exception:
            pass
        
        return None
    
    def _extract_organism(self, pdb_path: Path) -> Optional[str]:
        """Extract organism information from PDB file header"""
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('COMPND') and 'ORGANISM' in line:
                        # Extract organism name
                        organism_start = line.find('ORGANISM:') + 9
                        if organism_start > 8:
                            organism = line[organism_start:].strip().rstrip(';')
                            return organism
                    elif line.startswith('ATOM') or line.startswith('HETATM'):
                        break
        except Exception:
            pass
        
        return None
    
    def _clean_and_validate_sequence(self, sequence: str) -> str:
        """Clean sequence and validate amino acids"""
        if not sequence:
            return ""
        
        # Convert to uppercase and filter to only standard amino acids
        cleaned_sequence = ''.join(aa.upper() for aa in sequence if aa.upper() in self.amino_acids)
        
        return cleaned_sequence
    
    def _combine_and_balance_samples(self):
        """Combine positive and negative samples and balance the dataset"""
        self.samples = []
        
        # Add all samples
        self.samples.extend(self.positive_samples)
        self.samples.extend(self.negative_samples)
        
        # Shuffle the combined dataset
        indices = list(range(len(self.samples)))
        self.rng.shuffle(indices)
        self.samples = [self.samples[i] for i in indices]
        
        # Verify balance
        n_positive = sum(1 for s in self.samples if s['label'] == 1)
        n_negative = len(self.samples) - n_positive
        actual_positive_ratio = n_positive / len(self.samples) if self.samples else 0
        
        print(f"Dataset balance: {n_positive} positive, {n_negative} negative")
        print(f"Actual positive ratio: {actual_positive_ratio:.3f} (target: {self.positive_ratio:.3f})")
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset"""
        # Lazy loading check
        if self.lazy_loading and not self._lazy_loaded:
            # For lazy loading, we need to at least count the structure files
            # and estimate total samples
            n_structures = len(self.structure_files)
            estimated_negatives = int(n_structures * (1 - self.positive_ratio) / self.positive_ratio)
            return n_structures + estimated_negatives
        
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset"""
        # Thread-safe lazy loading check
        if self.lazy_loading and not self._lazy_loaded:
            with self._lazy_loading_lock:
                # Double-check locking pattern to avoid redundant loading
                if not self._lazy_loaded:
                    self._load_dataset()
                    self._lazy_loaded = True
        
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx].copy()  # Copy to avoid modifying original
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform and 'label' in sample:
            sample['label'] = self.target_transform(sample['label'])
        
        # Extract backbone features if available and requested
        print(f"\n=== DEBUG: __getitem__ for sample {idx} ===")
        print(f"  extract_backbone_features: {self.extract_backbone_features}")
        print(f"  backbone_encoder is not None: {self.backbone_encoder is not None}")
        print(f"  'coordinates' in sample: {'coordinates' in sample}")
        if 'coordinates' in sample:
            print(f"  sample['coordinates'] is not None: {sample['coordinates'] is not None}")
            if sample['coordinates'] is not None:
                print(f"  coordinates shape: {sample['coordinates'].shape if hasattr(sample['coordinates'], 'shape') else 'N/A'}")
        
        # Always ensure backbone_features key exists with valid tensor
        if (self.extract_backbone_features and 
            self.backbone_encoder is not None and 
            'coordinates' in sample and 
            sample['coordinates'] is not None):
            print(f"  Attempting to extract backbone features...")
            try:
                backbone_features = self._extract_backbone_features(sample)
                if backbone_features is not None:
                    print(f"  Successfully extracted backbone features, shape: {backbone_features.shape}")
                    sample['backbone_features'] = backbone_features
                    sample['backbone_features_valid'] = True
                else:
                    print(f"  WARNING: _extract_backbone_features returned None, creating placeholder")
                    sample['backbone_features'] = self._create_placeholder_backbone_features(sample['sequence'])
                    sample['backbone_features_valid'] = False
                    warnings.warn("Using placeholder backbone features due to coordinate issues - model performance may be degraded")
            except Exception as e:
                print(f"  ERROR: Failed to extract backbone features: {type(e).__name__}: {e}")
                warnings.warn(f"Failed to extract backbone features: {e}")
                sample['backbone_features'] = self._create_placeholder_backbone_features(sample['sequence'])
                sample['backbone_features_valid'] = False
        else:
            print(f"  Skipping backbone feature extraction, creating placeholder")
            sample['backbone_features'] = self._create_placeholder_backbone_features(sample['sequence'])
            sample['backbone_features_valid'] = False
        
        print(f"  Final sample keys: {list(sample.keys())}")
        print(f"  Has backbone_features: {'backbone_features' in sample}")
        print(f"=== END DEBUG __getitem__ ===")
        
        return sample
    
    def _create_placeholder_backbone_features(self, sequence: List[str]) -> torch.Tensor:
        """Create placeholder backbone features when extraction fails"""
        seq_len = len(sequence)
        backbone_dim = 128  # Standard backbone feature dimension
        
        # Create zero-filled placeholder with correct shape [seq_len, backbone_dim]
        placeholder = torch.zeros(seq_len, backbone_dim, dtype=torch.float32)
        
        # Add small amount of noise to avoid all-zero artifacts that might confuse training
        placeholder += torch.randn_like(placeholder) * 0.01
        
        return placeholder
    
    def _extract_backbone_features(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract backbone features from coordinates using MPNN encoder"""
        if self.backbone_encoder is None or sample.get('coordinates') is None:
            return None
        
        coordinates = sample['coordinates']  # Expected: [L, 4, 3]
        sequence = sample['sequence']
        seq_len = len(sequence)
        
        # Validate inputs
        if seq_len == 0:
            return None
        
        try:
            # DEBUG: Log input information
            print(f"\n=== DEBUG: _extract_backbone_features ===")
            print(f"  Sequence length: {seq_len}")
            print(f"  Coordinates type: {type(coordinates)}")
            
            # Convert numpy to torch tensor if needed
            if isinstance(coordinates, np.ndarray):
                print(f"  Converting numpy array to tensor, shape: {coordinates.shape}, dtype: {coordinates.dtype}")
                coords_tensor = torch.from_numpy(coordinates).float()
            else:
                print(f"  Coordinates already tensor, shape: {coordinates.shape}, dtype: {coordinates.dtype}")
                coords_tensor = coordinates
            
            # Validate and fix coordinate shape
            print(f"  Coords tensor shape after conversion: {coords_tensor.shape}")
            if len(coords_tensor.shape) != 3:
                warnings.warn(f"Invalid coordinate tensor shape: expected 3D tensor [L, 4, 3], got {coords_tensor.shape}")
                print(f"  ERROR: Invalid coordinate tensor dimensionality - returning None")
                return None
            
            if coords_tensor.shape[1] != 4 or coords_tensor.shape[2] != 3:
                warnings.warn(f"Invalid coordinate tensor format: expected [L, 4, 3], got {coords_tensor.shape}")
                print(f"  ERROR: Invalid coordinate format - returning None")
                return None
                
            coord_len = coords_tensor.shape[0]
            if coord_len != seq_len:
                # =============================================================================
                # COORDINATE/SEQUENCE LENGTH MISMATCH HANDLING
                # =============================================================================
                # PDB files sometimes have mismatches between sequence length and structural 
                # coordinates due to:
                # 1. Missing residues in crystal structures (unresolved loops, termini)
                # 2. Extra coordinates for ligands, waters, or modified residues 
                # 3. Sequence annotation inconsistencies
                # 4. Different chain definitions between sequence and structure
                #
                # Our strategy balances structural information preservation with model compatibility:
                # - Small mismatches (<configurable threshold): Fix via padding/truncation  
                # - Large mismatches (>threshold): Reject entirely (likely annotation errors)
                #
                # This preprocessing ensures the MPNN encoder receives valid inputs while maintaining
                # reasonable structural integrity for small discrepancies commonly found in PDB files.
                # ============================================================================= 
                
                # Extract sample identification for enhanced logging
                structure_file = sample.get('structure_file', 'unknown')
                chain_id = sample.get('chain_id', 'unknown')
                sample_id = f"{structure_file}:{chain_id}" if chain_id != 'unknown' else str(structure_file)
                
                # =============================================================================
                # VALIDATION: Coordinate order assumptions for end-based padding/truncation
                # =============================================================================
                # Our padding/truncation strategy assumes:
                # 1. Coordinate array order corresponds to sequence order (residue 0 -> N-terminus)
                # 2. Mismatches are primarily due to missing/extra residues at termini
                # 3. Core structured regions have coordinate-sequence alignment
                #
                # IMPORTANT: If coordinates are not in sequence order, our end-based approach
                # could disrupt structured regions. However, this is the standard assumption
                # for PDB files processed through structure parsers, and misalignment in the 
                # core would typically result in larger mismatch ratios that get rejected.
                #
                # NOTE: Future improvements could add sequence-structure alignment validation,
                # but this would require significant computational overhead for each sample.
                # =============================================================================
                
                # Calculate mismatch severity using relative ratio (not absolute difference)
                # This accounts for protein size: 10 missing residues is more concerning in a 50-residue 
                # protein (20% mismatch) than in a 500-residue protein (2% mismatch)
                mismatch_ratio = abs(coord_len - seq_len) / max(coord_len, seq_len) if max(coord_len, seq_len) > 0 else 1.0
                
                # Reject samples with excessive mismatches (configurable threshold, default 50%)
                # Large discrepancies typically indicate systematic annotation errors or non-standard
                # PDB entries that would produce poor quality training data
                if mismatch_ratio > self.max_coord_seq_mismatch_ratio:
                    warnings.warn(f"[{sample_id}] Large coordinate/sequence length mismatch ({mismatch_ratio:.1%}): coord_len={coord_len}, seq_len={seq_len}. Rejecting sample.")
                    print(f"  ERROR: Excessive length mismatch - returning None")
                    return None
                
                warnings.warn(f"[{sample_id}] Coordinate/sequence length mismatch: coord_len={coord_len}, seq_len={seq_len}. Attempting to fix...")
                print(f"  WARNING: Length mismatch (seq_len={seq_len}, coord_len={coord_len}) - attempting fix")
                
                if coord_len > seq_len:
                    # TRUNCATION STRATEGY: More coordinates than sequence residues
                    # This commonly occurs when PDB contains extra atoms (ligands, waters, ions) or
                    # when sequence annotation doesn't include all resolved residues in the structure.
                    # 
                    # We truncate from the END rather than beginning/middle because:
                    # 1. Extra coordinates are typically at C-terminus or non-protein molecules
                    # 2. N-terminal regions often contain key folding nucleation sites
                    # 3. MPNN models expect sequential coordinate order to match sequence order
                    #
                    # PERFORMANCE IMPACT: Some structural context is lost, but this is preferable to
                    # coordinate/sequence misalignment which would confuse the encoder entirely.
                    #
                    # VALIDATION: This operation truncates from the END by taking coords_tensor[:seq_len]
                    # which preserves the first seq_len coordinates (assumed to be N-terminal region)
                    coords_tensor = coords_tensor[:seq_len]  # END-BASED TRUNCATION
                    print(f"  Fixed by truncating coordinates to {seq_len} residues")
                    warnings.warn(f"[{sample_id}] Truncated {coord_len - seq_len} coordinates - structural information lost")
                elif coord_len < seq_len:
                    # PADDING STRATEGY: Fewer coordinates than sequence residues  
                    # This occurs when crystal structures have missing residues (unresolved loops, 
                    # flexible regions, or terminal residues) but the full sequence is annotated.
                    #
                    # DESIGN CHOICES:
                    # 1. REPEAT LAST COORDINATE (chosen) vs alternatives:
                    #    - Zero coordinates: Would create artificial (0,0,0) positions that don't exist 
                    #      in nature and could mislead the model about protein geometry
                    #    - Random coordinates: Would introduce noise and inconsistency 
                    #    - Interpolation: Too complex and assumes structural knowledge we don't have
                    #
                    # 2. PAD AT END (chosen) vs beginning: 
                    #    - Missing residues are more commonly at flexible termini than in structured core
                    #    - Preserves the structural integrity of the resolved portion
                    #    - MPNN models can partially handle duplicate coordinates better than misaligned ones
                    #
                    # PERFORMANCE IMPACT: Padding degrades feature quality for missing regions but
                    # allows the model to still extract useful information from the resolved structure.
                    # The warning alerts users that some features may be compromised.
                    padding_size = seq_len - coord_len
                    if coord_len > 0:
                        # Repeat last known coordinate (typically C-terminal residue) 
                        last_coord = coords_tensor[-1:].expand(padding_size, -1, -1)
                        # VALIDATION: This operation pads at the END using torch.cat([existing, padding], dim=0)
                        # which appends padding after all existing coordinates (assumed C-terminal extension)
                        coords_tensor = torch.cat([coords_tensor, last_coord], dim=0)  # END-BASED PADDING
                        print(f"  Fixed by padding {padding_size} coordinates with last known position")
                    else:
                        # Pathological case: No coordinates available at all
                        # This indicates severely corrupted or empty structure data
                        print(f"  ERROR: No coordinates available for padding - returning None")
                        return None
                    warnings.warn(f"[{sample_id}] Padded {padding_size} coordinates - features may be degraded")
                    
                print(f"  Final coordinate shape: {coords_tensor.shape}")
            
            # Add batch dimension: [1, L, 4, 3]
            print(f"  Adding batch dimension...")
            coords_batch = coords_tensor.unsqueeze(0)
            print(f"  Coords batch shape: {coords_batch.shape}")
            
            # Create required inputs for MPNN encoder
            print(f"  Creating MPNN encoder inputs...")
            batch = {
                'X': coords_batch,
                'mask': torch.ones(1, seq_len, dtype=torch.float32),
                'residue_idx': torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
                'chain_encoding_all': torch.zeros(1, seq_len, dtype=torch.long)
            }
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  X shape: {batch['X'].shape}, dtype: {batch['X'].dtype}")
            print(f"  mask shape: {batch['mask'].shape}, dtype: {batch['mask'].dtype}")
            print(f"  residue_idx shape: {batch['residue_idx'].shape}, dtype: {batch['residue_idx'].dtype}")
            print(f"  chain_encoding_all shape: {batch['chain_encoding_all'].shape}, dtype: {batch['chain_encoding_all'].dtype}")
            
            # Extract backbone features
            print(f"  Calling backbone_encoder...")
            with torch.no_grad():
                try:
                    backbone_features = self.backbone_encoder(batch)  # [1, L, hidden_dim]
                    print(f"  Backbone features extracted successfully, shape: {backbone_features.shape}")
                except Exception as encoder_error:
                    print(f"  ERROR in backbone_encoder: {type(encoder_error).__name__}: {encoder_error}")
                    import traceback
                    print(f"  Traceback:\n{traceback.format_exc()}")
                    raise
                
            # Remove batch dimension and return
            result = backbone_features.squeeze(0)  # [L, hidden_dim]
            print(f"  Final result shape: {result.shape}")
            print(f"=== END DEBUG ===")
            return result
            
        except Exception as e:
            print(f"\n=== ERROR in _extract_backbone_features ===")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            import traceback
            print(f"  Full traceback:\n{traceback.format_exc()}")
            print(f"=== END ERROR ===")
            warnings.warn(f"Error extracting backbone features: {e}")
            return None
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get information about the dataset composition"""
        if not self.samples:
            return {}
        
        labels = [s['label'] for s in self.samples]
        lengths = [s['length'] for s in self.samples]
        
        info = {
            'total_samples': len(self.samples),
            'positive_samples': sum(labels),
            'negative_samples': len(labels) - sum(labels),
            'positive_ratio': sum(labels) / len(labels),
            'sequence_lengths': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths)
            },
            'negative_methods': {}
        }
        
        # Count negative generation methods
        for sample in self.samples:
            if sample['label'] == 0 and 'generation_method' in sample:
                method = sample['generation_method']
                info['negative_methods'][method] = info['negative_methods'].get(method, 0) + 1
        
        return info
    
    def save_cache(self, cache_path: Optional[Union[str, Path]] = None):
        """Save processed dataset to cache"""
        if cache_path is None:
            if self.cache_dir is None:
                raise ValueError("No cache directory specified")
            cache_path = self.cache_dir / "stability_dataset_cache.json"
        
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare cache data (exclude large arrays)
        cache_data = {
            'config': {
                'data_dir': str(self.data_dir),
                'positive_ratio': self.positive_ratio,
                'negative_methods': self.negative_methods,
                'max_sequence_length': self.max_sequence_length,
                'min_sequence_length': self.min_sequence_length,
                'seed': self.seed
            },
            'samples': []
        }
        
        # Save sample metadata (excluding coordinates for space)
        for sample in self.samples:
            cache_sample = sample.copy()
            if 'coordinates' in cache_sample:
                cache_sample['coordinates'] = None  # Don't cache large arrays
            cache_data['samples'].append(cache_sample)
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Dataset cached to {cache_path}")
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, cluster_threshold: float = 0.3,
                     stratify_by: Optional[List[str]] = None,
                     min_cluster_size: int = 1, random_seed: int = 42) -> Dict[str, List[int]]:
        """
        Create train/validation/test splits with sequence clustering and stratification.
        
        Args:
            train_ratio: Fraction of data for training (default: 0.7)
            val_ratio: Fraction of data for validation (default: 0.15)
            test_ratio: Fraction of data for testing (default: 0.15)
            cluster_threshold: Sequence identity threshold for clustering (default: 0.3)
            stratify_by: Properties to stratify by (default: ['length', 'label'])
            min_cluster_size: Minimum cluster size (default: 1)
            random_seed: Random seed for reproducibility (default: 42)
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing sample indices
        """
        if not self.samples:
            raise ValueError("No samples loaded. Cannot create splits.")
        
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Default stratification
        if stratify_by is None:
            stratify_by = ['length', 'label']
        
        # Set random seed for reproducibility
        split_rng = np.random.RandomState(random_seed)
        
        print(f"Creating splits with clustering (threshold={cluster_threshold}) and stratification by {stratify_by}...")
        
        # Step 1: Cluster sequences by identity to prevent data leakage
        sequence_clusters = self._cluster_sequences_by_identity(cluster_threshold, min_cluster_size)
        print(f"Found {len(sequence_clusters)} sequence clusters")
        
        # Step 2: Create stratification groups
        strata = self._create_stratification_groups(stratify_by)
        print(f"Created {len(strata)} stratification groups")
        
        # Step 3: Assign clusters to splits while maintaining stratification
        cluster_splits = self._assign_clusters_to_splits(
            sequence_clusters, strata, train_ratio, val_ratio, test_ratio, split_rng
        )
        
        # Step 4: Convert cluster assignments to sample indices
        splits = {'train': [], 'val': [], 'test': []}
        for cluster_id, split_name in cluster_splits.items():
            cluster_indices = sequence_clusters[cluster_id]
            splits[split_name].extend(cluster_indices)
        
        # Step 5: Validate splits
        self._validate_splits(splits, stratify_by)
        
        return splits
    
    def _cluster_sequences_by_identity(self, threshold: float, 
                                     min_cluster_size: int) -> Dict[int, List[int]]:
        """
        Cluster sequences by sequence identity to prevent data leakage.
        
        Uses a greedy clustering approach based on sequence identity.
        """
        sequences = [sample['sequence'] for sample in self.samples]
        n_sequences = len(sequences)
        
        # Compute pairwise sequence identities
        identity_matrix = self._compute_sequence_identities(sequences)
        
        # Greedy clustering
        clusters = {}
        cluster_id = 0
        assigned = [False] * n_sequences
        
        for i in range(n_sequences):
            if assigned[i]:
                continue
            
            # Start new cluster
            cluster_members = [i]
            assigned[i] = True
            
            # Find all sequences similar to this one
            for j in range(i + 1, n_sequences):
                if assigned[j]:
                    continue
                
                # Check if j is similar to any member of current cluster
                max_identity = max(identity_matrix[j][k] for k in cluster_members)
                if max_identity >= threshold:
                    cluster_members.append(j)
                    assigned[j] = True
            
            # Only keep clusters that meet minimum size
            if len(cluster_members) >= min_cluster_size:
                clusters[cluster_id] = cluster_members
                cluster_id += 1
        
        # Handle singletons (unassigned sequences)
        for i in range(n_sequences):
            if not assigned[i]:
                clusters[cluster_id] = [i]
                cluster_id += 1
        
        return clusters
    
    def _compute_sequence_identities(self, sequences: List[str]) -> np.ndarray:
        """
        Compute pairwise sequence identity matrix.
        
        Uses simple character-by-character comparison for speed.
        For more sophisticated alignment-based similarity, consider using BioPython.
        """
        n_seq = len(sequences)
        identity_matrix = np.zeros((n_seq, n_seq))
        
        for i in range(n_seq):
            for j in range(i, n_seq):
                if i == j:
                    identity_matrix[i][j] = 1.0
                else:
                    identity = self._compute_sequence_identity(sequences[i], sequences[j])
                    identity_matrix[i][j] = identity
                    identity_matrix[j][i] = identity
        
        return identity_matrix
    
    def _compute_sequence_identity(self, seq1: str, seq2: str) -> float:
        """
        Compute sequence identity between two sequences.
        
        Uses simple character matching. For sequences of different lengths,
        computes identity over the shorter sequence.
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Align sequences (simple: compare from start)
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        
        # Identity = matches / length of shorter sequence
        return matches / min_len
    
    def _create_stratification_groups(self, stratify_by: List[str]) -> Dict[str, List[int]]:
        """
        Create stratification groups based on specified properties.
        
        Groups samples with similar properties together for balanced splitting.
        """
        strata = {}
        
        for idx, sample in enumerate(self.samples):
            # Create stratum key based on properties
            key_parts = []
            
            for prop in stratify_by:
                if prop == 'length':
                    # Bin by length (groups of 50)
                    length_bin = (sample['length'] // 50) * 50
                    key_parts.append(f'len_{length_bin}')
                
                elif prop == 'label':
                    key_parts.append(f'label_{sample["label"]}')
                
                elif prop == 'generation_method':
                    method = sample.get('generation_method', 'native')
                    key_parts.append(f'method_{method}')
                
                elif prop == 'source_type':
                    source = sample.get('source_type', 'unknown')
                    key_parts.append(f'source_{source}')
                
                elif prop == 'composition':
                    # Bin by hydrophobic content
                    sequence = sample['sequence']
                    if sequence:
                        hydrophobic_fraction = sum(1 for aa in sequence if aa in self.hydrophobic) / len(sequence)
                        hydrophobic_bin = int(hydrophobic_fraction * 10) / 10  # 0.0, 0.1, 0.2, etc.
                        key_parts.append(f'hydro_{hydrophobic_bin}')
                    else:
                        key_parts.append('hydro_unknown')
                
                else:
                    # Generic property
                    value = sample.get(prop, 'unknown')
                    key_parts.append(f'{prop}_{value}')
            
            stratum_key = '_'.join(key_parts)
            
            if stratum_key not in strata:
                strata[stratum_key] = []
            strata[stratum_key].append(idx)
        
        return strata
    
    def _assign_clusters_to_splits(self, clusters: Dict[int, List[int]], 
                                 strata: Dict[str, List[int]], 
                                 train_ratio: float, val_ratio: float, test_ratio: float,
                                 rng: np.random.RandomState) -> Dict[int, str]:
        """
        Assign sequence clusters to train/val/test splits while maintaining stratification.
        """
        cluster_assignments = {}
        
        # For each stratum, distribute its clusters across splits
        for stratum_key, sample_indices in strata.items():
            # Find clusters that contain samples from this stratum
            stratum_clusters = {}
            for cluster_id, cluster_indices in clusters.items():
                intersection = set(cluster_indices) & set(sample_indices)
                if intersection:
                    stratum_clusters[cluster_id] = len(intersection)
            
            if not stratum_clusters:
                continue
            
            # Sort clusters by size (largest first for more balanced splits)
            sorted_clusters = sorted(stratum_clusters.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            # Distribute clusters to splits in round-robin fashion
            split_names = ['train', 'val', 'test']
            split_ratios = [train_ratio, val_ratio, test_ratio]
            split_counts = {name: 0 for name in split_names}
            target_counts = {name: int(len(sorted_clusters) * ratio) 
                           for name, ratio in zip(split_names, split_ratios)}
            
            # Assign clusters
            for cluster_id, cluster_size in sorted_clusters:
                if cluster_id in cluster_assignments:
                    continue  # Already assigned
                
                # Find split with most room relative to target
                best_split = None
                best_deficit = -1
                
                for split_name in split_names:
                    target = target_counts[split_name]
                    current = split_counts[split_name]
                    deficit = target - current
                    
                    if deficit > best_deficit:
                        best_deficit = deficit
                        best_split = split_name
                
                # Assign to best split
                if best_split:
                    cluster_assignments[cluster_id] = best_split
                    split_counts[best_split] += 1
                else:
                    # Fallback: assign to train
                    cluster_assignments[cluster_id] = 'train'
        
        # Handle any unassigned clusters
        for cluster_id in clusters:
            if cluster_id not in cluster_assignments:
                # Assign randomly based on ratios
                rand_val = rng.random()
                if rand_val < train_ratio:
                    cluster_assignments[cluster_id] = 'train'
                elif rand_val < train_ratio + val_ratio:
                    cluster_assignments[cluster_id] = 'val'
                else:
                    cluster_assignments[cluster_id] = 'test'
        
        return cluster_assignments
    
    def _validate_splits(self, splits: Dict[str, List[int]], stratify_by: List[str]):
        """Validate that splits are reasonable and properly balanced"""
        total_samples = len(self.samples)
        
        print("\\nSplit validation:")
        for split_name, indices in splits.items():
            n_samples = len(indices)
            ratio = n_samples / total_samples if total_samples > 0 else 0
            
            # Count positive/negative samples
            n_positive = sum(1 for idx in indices if self.samples[idx]['label'] == 1)
            n_negative = n_samples - n_positive
            pos_ratio = n_positive / n_samples if n_samples > 0 else 0
            
            print(f"  {split_name}: {n_samples} samples ({ratio:.1%}), "
                  f"{n_positive} pos ({pos_ratio:.1%}), {n_negative} neg")
        
        # Check for overlap between splits
        all_indices = set()
        overlaps_found = False
        for split_name, indices in splits.items():
            split_set = set(indices)
            overlap = all_indices & split_set
            if overlap:
                warnings.warn(f"Overlap found in {split_name}: {len(overlap)} samples")
                overlaps_found = True
            all_indices.update(split_set)
        
        if not overlaps_found:
            print("  ✓ No overlap between splits")
        
        # Check coverage
        coverage = len(all_indices) / total_samples if total_samples > 0 else 0
        print(f"  Coverage: {len(all_indices)}/{total_samples} samples ({coverage:.1%})")
        
        if coverage < 0.95:
            warnings.warn(f"Low coverage: only {coverage:.1%} of samples assigned to splits")
    
    def get_split_datasets(self, splits: Dict[str, List[int]]) -> Dict[str, 'StabilityDataset']:
        """
        Create separate dataset instances for each split.
        
        Args:
            splits: Dictionary from create_splits() with sample indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' dataset instances
        """
        split_datasets = {}
        
        for split_name, sample_indices in splits.items():
            # Create new dataset instance with subset of samples
            split_dataset = StabilityDataset.__new__(StabilityDataset)
            
            # Copy configuration from parent
            for attr in ['data_dir', 'positive_ratio', 'negative_methods', 'max_sequence_length',
                        'min_sequence_length', 'structure_extensions', 'cache_dir', 'seed',
                        'transform', 'target_transform', 'lazy_loading', 'include_coordinates']:
                setattr(split_dataset, attr, getattr(self, attr))
            
            # Copy amino acid properties and other setup
            split_dataset.amino_acids = self.amino_acids
            split_dataset.aa_to_idx = self.aa_to_idx
            split_dataset.idx_to_aa = self.idx_to_aa
            split_dataset.natural_frequencies = self.natural_frequencies
            split_dataset.hydrophobic = self.hydrophobic
            split_dataset.polar = self.polar
            split_dataset.charged_positive = self.charged_positive
            split_dataset.charged_negative = self.charged_negative
            split_dataset.aromatic = self.aromatic
            split_dataset.small = self.small
            split_dataset.conservative_groups = self.conservative_groups
            split_dataset.rng = np.random.RandomState(self.seed + hash(split_name) % 1000)
            split_dataset._lazy_loaded = True  # Split datasets are already loaded
            
            # Set subset of samples
            split_dataset.samples = [self.samples[i] for i in sample_indices]
            split_dataset.positive_samples = [s for s in split_dataset.samples if s['label'] == 1]
            split_dataset.negative_samples = [s for s in split_dataset.samples if s['label'] == 0]
            split_dataset.structure_files = []  # Not needed for split datasets
            
            split_datasets[split_name] = split_dataset
        
        return split_datasets
    
    def analyze_splits(self, splits: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Analyze split quality and properties.
        
        Args:
            splits: Dictionary from create_splits()
            
        Returns:
            Analysis results with statistics and quality metrics
        """
        analysis = {
            'split_sizes': {},
            'label_distributions': {},
            'length_distributions': {},
            'sequence_identity_stats': {},
            'quality_metrics': {}
        }
        
        # Basic statistics
        for split_name, indices in splits.items():
            n_samples = len(indices)
            split_samples = [self.samples[i] for i in indices]
            
            # Size and label distribution
            n_positive = sum(1 for s in split_samples if s['label'] == 1)
            n_negative = n_samples - n_positive
            
            analysis['split_sizes'][split_name] = n_samples
            analysis['label_distributions'][split_name] = {
                'positive': n_positive,
                'negative': n_negative,
                'positive_ratio': n_positive / n_samples if n_samples > 0 else 0
            }
            
            # Length distribution
            lengths = [s['length'] for s in split_samples]
            analysis['length_distributions'][split_name] = {
                'mean': np.mean(lengths) if lengths else 0,
                'std': np.std(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0
            }
        
        # Cross-split sequence identity analysis
        for split1_name, indices1 in splits.items():
            for split2_name, indices2 in splits.items():
                if split1_name >= split2_name:  # Avoid duplicate comparisons
                    continue
                
                sequences1 = [self.samples[i]['sequence'] for i in indices1[:100]]  # Sample for speed
                sequences2 = [self.samples[i]['sequence'] for i in indices2[:100]]
                
                max_identity = 0.0
                identity_count = 0
                
                for seq1 in sequences1:
                    for seq2 in sequences2:
                        identity = self._compute_sequence_identity(seq1, seq2)
                        max_identity = max(max_identity, identity)
                        if identity > 0.3:  # High identity threshold
                            identity_count += 1
                
                pair_key = f'{split1_name}_vs_{split2_name}'
                analysis['sequence_identity_stats'][pair_key] = {
                    'max_identity': max_identity,
                    'high_identity_pairs': identity_count
                }
        
        # Quality metrics
        total_samples = len(self.samples)
        coverage = sum(len(indices) for indices in splits.values()) / total_samples
        
        # Check label balance across splits
        label_variance = np.var([
            analysis['label_distributions'][split_name]['positive_ratio'] 
            for split_name in splits.keys()
        ])
        
        analysis['quality_metrics'] = {
            'coverage': coverage,
            'label_balance_variance': label_variance,
            'total_splits': len(splits),
            'smallest_split_size': min(len(indices) for indices in splits.values()),
            'largest_split_size': max(len(indices) for indices in splits.values())
        }
        
        return analysis
    
    def create_hard_negative_miner(self, mining_strategy: str = 'energy_based',
                                 difficulty_schedule: Optional[Dict] = None,
                                 cache_size: int = 1000,
                                 refresh_rate: float = 0.1) -> 'HardNegativeMiner':
        """
        Create a hard negative mining utility for dynamic negative sample selection.
        
        Args:
            mining_strategy: Strategy for mining ('energy_based', 'gradient_based', 'hybrid')
            difficulty_schedule: Schedule for mining difficulty over training
            cache_size: Number of negative candidates to maintain in cache
            refresh_rate: Fraction of cache to refresh each epoch
            
        Returns:
            HardNegativeMiner instance configured for this dataset
        """
        miner = HardNegativeMiner(
            dataset=self,
            strategy=mining_strategy,
            difficulty_schedule=difficulty_schedule,
            cache_size=cache_size,
            refresh_rate=refresh_rate
        )
        
        return miner


class HardNegativeMiner:
    """
    Hard negative mining utility for dynamic selection of challenging negative samples.
    
    This class implements online hard negative mining to improve training efficiency
    by focusing on the most challenging negative examples that help the model learn
    better decision boundaries.
    
    Args:
        dataset: StabilityDataset instance to mine from
        strategy: Mining strategy ('energy_based', 'gradient_based', 'hybrid')
        difficulty_schedule: Dictionary defining difficulty progression over training
        cache_size: Maximum number of negative candidates to cache
        refresh_rate: Fraction of cache to refresh each epoch (0.0-1.0)
        temperature: Temperature for softmax sampling of hard negatives
        easy_negative_ratio: Ratio of easy negatives to include for training stability
    """
    
    def __init__(self, dataset: StabilityDataset, strategy: str = 'energy_based',
                 difficulty_schedule: Optional[Dict] = None,
                 cache_size: int = 1000, refresh_rate: float = 0.1,
                 temperature: float = 2.0, easy_negative_ratio: float = 0.2):
        
        self.dataset = dataset
        self.strategy = strategy
        self.cache_size = cache_size
        self.refresh_rate = refresh_rate
        self.temperature = temperature
        self.easy_negative_ratio = easy_negative_ratio
        
        # Default difficulty schedule
        if difficulty_schedule is None:
            difficulty_schedule = {
                'start_epoch': 0,
                'end_epoch': 100,
                'start_difficulty': 0.3,  # Start with easier negatives
                'end_difficulty': 0.8,   # Progress to harder negatives
                'schedule_type': 'linear'  # 'linear', 'exponential', or 'step'
            }
        
        self.difficulty_schedule = difficulty_schedule
        
        # Mining state
        self.current_epoch = 0
        self.negative_cache = []
        self.candidate_pool = []
        self.mining_history = []
        self.model_ref = None  # Will be set during training
        
        # Statistics
        self.stats = {
            'samples_mined': 0,
            'cache_refreshes': 0,
            'avg_difficulty': 0.0,
            'mining_efficiency': 0.0
        }
        
        # Thread locks for cache operations
        self._cache_lock = threading.RLock()  # RLock for nested cache operations
        self._stats_lock = threading.Lock()   # Separate lock for statistics
        
        # Initialize negative candidate pool
        self._initialize_candidate_pool()
    
    def _initialize_candidate_pool(self):
        """Initialize the pool of negative candidates for mining"""
        print("Initializing hard negative mining candidate pool...")
        
        # Generate diverse negative candidates using existing methods
        pool_size = min(self.cache_size * 3, 5000)  # 3x cache size, max 5000
        
        # Use different negative generation methods
        methods = ['random', 'mutations', 'failed_designs']
        samples_per_method = pool_size // len(methods)
        
        for method in methods:
            if method == 'random':
                candidates = self.dataset._generate_random_sequences(samples_per_method)
            elif method == 'mutations':
                candidates = self.dataset._generate_destabilizing_mutations(samples_per_method)
            elif method == 'failed_designs':
                candidates = self.dataset._generate_failed_designs(samples_per_method)
            
            # Add mining metadata
            for candidate in candidates:
                candidate['mining_metadata'] = {
                    'generation_epoch': 0,
                    'difficulty_score': 0.0,
                    'selection_count': 0,
                    'last_selected_epoch': -1
                }
            
            self.candidate_pool.extend(candidates)
        
        # Initialize cache with random selection
        self.negative_cache = self.candidate_pool[:self.cache_size].copy()
        
        print(f"Initialized mining pool with {len(self.candidate_pool)} candidates")
        print(f"Cache size: {len(self.negative_cache)} samples")
    
    def update_epoch(self, epoch: int, model=None):
        """
        Update mining state for new epoch.
        
        Args:
            epoch: Current training epoch
            model: Current model instance for difficulty assessment
        """
        self.current_epoch = epoch
        self.model_ref = model
        
        # Check if cache refresh is needed
        if epoch > 0 and (epoch % max(1, int(1.0 / self.refresh_rate))) == 0:
            self._refresh_cache()
    
    def mine_hard_negatives(self, batch_size: int, positive_samples: List[Dict],
                          model=None, device='cpu') -> List[Dict]:
        """
        Mine hard negative samples for the current batch.
        
        Args:
            batch_size: Number of negative samples to mine
            positive_samples: Positive samples in current batch for context
            model: Current model instance for difficulty assessment
            device: Device for model inference
            
        Returns:
            List of hard negative samples
        """
        if model is not None:
            self.model_ref = model
        
        current_difficulty = self._get_current_difficulty()
        
        # Determine how many hard vs easy negatives to include
        n_hard = int(batch_size * (1 - self.easy_negative_ratio))
        n_easy = batch_size - n_hard
        
        hard_negatives = []
        easy_negatives = []
        
        if self.strategy == 'energy_based':
            hard_negatives = self._mine_energy_based(n_hard, current_difficulty, positive_samples)
            easy_negatives = self._mine_easy_negatives(n_easy)
        
        elif self.strategy == 'gradient_based':
            hard_negatives = self._mine_gradient_based(n_hard, current_difficulty, positive_samples, device)
            easy_negatives = self._mine_easy_negatives(n_easy)
        
        elif self.strategy == 'hybrid':
            n_energy = n_hard // 2
            n_gradient = n_hard - n_energy
            
            energy_negatives = self._mine_energy_based(n_energy, current_difficulty, positive_samples)
            gradient_negatives = self._mine_gradient_based(n_gradient, current_difficulty, positive_samples, device)
            
            hard_negatives = energy_negatives + gradient_negatives
            easy_negatives = self._mine_easy_negatives(n_easy)
        
        else:
            raise ValueError(f"Unknown mining strategy: {self.strategy}")
        
        # Combine and shuffle
        mined_negatives = hard_negatives + easy_negatives
        random.shuffle(mined_negatives)
        
        # Update statistics
        self._update_mining_stats(hard_negatives, easy_negatives)
        
        return mined_negatives[:batch_size]
    
    def _get_current_difficulty(self) -> float:
        """Get current difficulty level based on schedule"""
        schedule = self.difficulty_schedule
        start_epoch = schedule['start_epoch']
        end_epoch = schedule['end_epoch']
        start_diff = schedule['start_difficulty']
        end_diff = schedule['end_difficulty']
        schedule_type = schedule['schedule_type']
        
        if self.current_epoch <= start_epoch:
            return start_diff
        elif self.current_epoch >= end_epoch:
            return end_diff
        
        # Interpolate based on schedule type
        progress = (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
        
        if schedule_type == 'linear':
            difficulty = start_diff + progress * (end_diff - start_diff)
        elif schedule_type == 'exponential':
            # Exponential growth in difficulty
            difficulty = start_diff * ((end_diff / start_diff) ** progress)
        elif schedule_type == 'step':
            # Step increases at quartiles
            if progress < 0.25:
                difficulty = start_diff
            elif progress < 0.5:
                difficulty = start_diff + 0.25 * (end_diff - start_diff)
            elif progress < 0.75:
                difficulty = start_diff + 0.5 * (end_diff - start_diff)
            else:
                difficulty = start_diff + 0.75 * (end_diff - start_diff)
        else:
            difficulty = start_diff + progress * (end_diff - start_diff)  # Default to linear
        
        return np.clip(difficulty, 0.0, 1.0)
    
    def _mine_energy_based(self, n_samples: int, difficulty: float, 
                         positive_samples: List[Dict]) -> List[Dict]:
        """Mine hard negatives based on energy predictions"""
        if not self.model_ref or not self.negative_cache:
            return self._mine_random_from_cache(n_samples)
        
        try:
            # Get energy scores for cached negatives
            with torch.no_grad():
                energies = self._compute_energies_for_candidates(self.negative_cache)
            
            # For hard negative mining, we want negatives with energies close to positives
            # (i.e., negatives that the model incorrectly thinks might be stable)
            target_energy = self._estimate_positive_energy_range(positive_samples)
            
            # Compute difficulty scores based on energy proximity to positives
            difficulty_scores = []
            for energy in energies:
                # Closer to positive range = harder negative
                score = 1.0 - abs(energy - target_energy) / (target_energy + 1e-6)
                difficulty_scores.append(max(0.0, score))
            
            # Select based on difficulty and current difficulty level
            selected_indices = self._select_by_difficulty(difficulty_scores, n_samples, difficulty)
            
            return [self.negative_cache[i] for i in selected_indices]
        
        except Exception as e:
            warnings.warn(f"Energy-based mining failed: {e}, falling back to random")
            return self._mine_random_from_cache(n_samples)
    
    def _mine_gradient_based(self, n_samples: int, difficulty: float,
                           positive_samples: List[Dict], device: str) -> List[Dict]:
        """Mine hard negatives based on gradient information"""
        if not self.model_ref or not self.negative_cache:
            return self._mine_random_from_cache(n_samples)
        
        # Gradient-based mining is more computationally expensive
        # For now, implement a simplified version based on loss gradients
        try:
            # Compute approximate gradient norms for cached negatives
            gradient_norms = self._compute_gradient_norms(self.negative_cache, device)
            
            # Higher gradient norm = harder example (model is more confused)
            difficulty_scores = gradient_norms.numpy() if hasattr(gradient_norms, 'numpy') else gradient_norms
            
            # Normalize scores
            if len(difficulty_scores) > 1:
                difficulty_scores = (difficulty_scores - difficulty_scores.min()) / (difficulty_scores.max() - difficulty_scores.min() + 1e-6)
            
            selected_indices = self._select_by_difficulty(difficulty_scores, n_samples, difficulty)
            
            return [self.negative_cache[i] for i in selected_indices]
        
        except Exception as e:
            warnings.warn(f"Gradient-based mining failed: {e}, falling back to random")
            return self._mine_random_from_cache(n_samples)
    
    def _mine_easy_negatives(self, n_samples: int) -> List[Dict]:
        """Mine easy negatives for training stability"""
        if not self.negative_cache:
            return []
        
        # Select negatives with low difficulty scores (easy examples)
        easy_candidates = [
            sample for sample in self.negative_cache
            if sample.get('mining_metadata', {}).get('difficulty_score', 0.5) < 0.3
        ]
        
        if len(easy_candidates) < n_samples:
            easy_candidates.extend(self.negative_cache[:n_samples - len(easy_candidates)])
        
        return random.sample(easy_candidates, min(n_samples, len(easy_candidates)))
    
    def _mine_random_from_cache(self, n_samples: int) -> List[Dict]:
        """Fallback: random mining from cache"""
        if not self.negative_cache:
            return []
        
        return random.sample(self.negative_cache, min(n_samples, len(self.negative_cache)))
    
    def _compute_energies_for_candidates(self, candidates: List[Dict]) -> np.ndarray:
        """Compute energy predictions for candidate samples"""
        # This is a simplified implementation
        # In practice, you would need to integrate with your specific model architecture
        
        # Placeholder: random energies for demonstration
        # Replace with actual model inference
        energies = np.random.randn(len(candidates))
        return energies
    
    def _estimate_positive_energy_range(self, positive_samples: List[Dict]) -> float:
        """Estimate energy range of positive samples"""
        if not positive_samples:
            return 0.0
        
        # Placeholder: return average energy
        # Replace with actual model inference on positive samples
        return -1.0  # Assume negative energies for stable sequences
    
    def _compute_gradient_norms(self, candidates: List[Dict], device: str) -> np.ndarray:
        """Compute gradient norms for candidates (simplified)"""
        # Placeholder implementation
        # In practice, you would compute actual gradients w.r.t. model parameters
        gradient_norms = np.random.rand(len(candidates))
        return gradient_norms
    
    def _select_by_difficulty(self, difficulty_scores: np.ndarray, n_samples: int, 
                            target_difficulty: float) -> List[int]:
        """Select samples based on difficulty scores and target difficulty"""
        if len(difficulty_scores) == 0:
            return []
        
        difficulty_scores = np.array(difficulty_scores)
        
        # Weight selection probability based on target difficulty
        # Higher target_difficulty = prefer harder examples
        weights = np.exp(target_difficulty * difficulty_scores / self.temperature)
        weights = weights / (weights.sum() + 1e-8)
        
        # Sample without replacement
        selected_indices = np.random.choice(
            len(difficulty_scores), 
            size=min(n_samples, len(difficulty_scores)), 
            replace=False, 
            p=weights
        ).tolist()
        
        return selected_indices
    
    def _refresh_cache(self):
        """Refresh the negative cache with new candidates"""
        with self._cache_lock:
            refresh_count = max(1, int(self.cache_size * self.refresh_rate))
            
            # Remove old samples (least recently used)
            cache_with_metadata = [(i, sample) for i, sample in enumerate(self.negative_cache)]
            cache_with_metadata.sort(
                key=lambda x: x[1].get('mining_metadata', {}).get('last_selected_epoch', -1)
            )
            
            # Keep most recently used samples
            keep_indices = [i for i, _ in cache_with_metadata[refresh_count:]]
            self.negative_cache = [self.negative_cache[i] for i in keep_indices]
            
            # OPTIMIZED: Use set-based membership testing to avoid O(N²) complexity
            # Create set of sequences currently in cache for O(1) lookups
            cache_sequences = {self._get_sample_key(sample) for sample in self.negative_cache}
            
            # Filter candidate pool using set difference - O(N) instead of O(N²)
            available_candidates = [
                sample for sample in self.candidate_pool
                if self._get_sample_key(sample) not in cache_sequences
            ]
            
            if available_candidates:
                new_samples = random.sample(
                    available_candidates, 
                    min(refresh_count, len(available_candidates))
                )
                self.negative_cache.extend(new_samples)
        
        # Update stats with separate lock to avoid long-held locks
        with self._stats_lock:
            self.stats['cache_refreshes'] += 1
        print(f"Cache refreshed: {refresh_count} samples replaced")
    
    def _get_sample_key(self, sample: Dict[str, Any]) -> str:
        """Generate unique key for sample to enable fast set-based operations"""
        # Use sequence + chain_id as unique identifier
        sequence = sample.get('sequence', '')
        chain_id = sample.get('chain_id', '')
        file_path = str(sample.get('file_path', ''))
        
        # Create unique key combining critical identifiers
        return f"{sequence}:{chain_id}:{file_path}"
    
    def _update_mining_stats(self, hard_negatives: List[Dict], easy_negatives: List[Dict]):
        """Update mining statistics"""
        with self._stats_lock:
            total_mined = len(hard_negatives) + len(easy_negatives)
            self.stats['samples_mined'] += total_mined
        
        # Update difficulty scores for selected samples
        for sample in hard_negatives:
            metadata = sample.get('mining_metadata', {})
            metadata['selection_count'] = metadata.get('selection_count', 0) + 1
            metadata['last_selected_epoch'] = self.current_epoch
            metadata['difficulty_score'] = self._get_current_difficulty()
        
        for sample in easy_negatives:
            metadata = sample.get('mining_metadata', {})
            metadata['selection_count'] = metadata.get('selection_count', 0) + 1
            metadata['last_selected_epoch'] = self.current_epoch
            metadata['difficulty_score'] = 0.2  # Easy samples
        
        # Compute average difficulty
        if total_mined > 0:
            avg_difficulty = (len(hard_negatives) * self._get_current_difficulty() + 
                            len(easy_negatives) * 0.2) / total_mined
            self.stats['avg_difficulty'] = avg_difficulty
    
    def get_mining_stats(self) -> Dict[str, Any]:
        """Get current mining statistics"""
        return {
            'current_epoch': self.current_epoch,
            'current_difficulty': self._get_current_difficulty(),
            'cache_size': len(self.negative_cache),
            'candidate_pool_size': len(self.candidate_pool),
            'stats': self.stats.copy(),
            'strategy': self.strategy
        }
    
    def save_mining_state(self, filepath: Union[str, Path]):
        """Save mining state for resuming training"""
        state = {
            'current_epoch': self.current_epoch,
            'negative_cache': self.negative_cache,
            'candidate_pool': self.candidate_pool,
            'stats': self.stats,
            'strategy': self.strategy,
            'difficulty_schedule': self.difficulty_schedule
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_mining_state(self, filepath: Union[str, Path]):
        """Load mining state for resuming training"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_epoch = state['current_epoch']
        self.negative_cache = state['negative_cache']
        self.candidate_pool = state['candidate_pool']
        self.stats = state['stats']
        # Strategy and difficulty_schedule loaded during initialization


if __name__ == "__main__":
    # Example usage
    print("Testing StabilityDataset...")
    
    # Create dummy test directory structure
    test_data_dir = Path("test_structures")
    if not test_data_dir.exists():
        print("No test data directory found. Creating dummy dataset for testing...")
        
        # For actual usage, point to real PDB files
        print("To use this dataset, provide a directory containing PDB structure files.")
        print("Example:")
        print("  dataset = StabilityDataset('path/to/pdb/files')")
        
        # Show configuration options
        dataset_config = {
            'data_dir': 'path/to/pdb/files',
            'positive_ratio': 0.6,  # 60% positive samples
            'negative_methods': ['random', 'mutations', 'failed_designs'],
            'max_sequence_length': 300,
            'min_sequence_length': 50,
            'seed': 42
        }
        
        print(f"Example configuration: {dataset_config}")
    else:
        # Test with actual data
        try:
            dataset = StabilityDataset(
                data_dir=test_data_dir,
                max_files=5,  # Limit for testing
                positive_ratio=0.6
            )
            
            print(f"Dataset created successfully!")
            print(f"Sample info: {dataset.get_sample_info()}")
            
            # Test data loading
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"First sample keys: {sample.keys()}")
                print(f"Sequence length: {len(sample['sequence'])}")
                print(f"Label: {sample['label']}")
                
        except Exception as e:
            print(f"Testing failed: {e}")
            print("Make sure test_structures directory contains PDB files for testing.")