# Dynamic PDB Streaming Implementation Plan

## Executive Summary

This implementation plan addresses the critical need for dynamic PDB downloading during training to enable large-scale protein design model training with ~19,000 samples without overwhelming local storage. The current ProteinMPNN training pipeline expects all PDB files to be pre-downloaded, but this becomes infeasible at scale.

**Core Problem**: ProteinMPNN's data loading assumes static file directories, but we need to stream thousands of PDBs dynamically during training while managing storage constraints and avoiding redundant downloads.

**Solution**: Build a streaming data pipeline that downloads PDBs on-demand, maintains an intelligent cache, and integrates seamlessly with existing ProteinMPNN data loaders.

## Problem Analysis

### Root Causes

1. **Static Data Assumption**: `StructureDataset` and `StructureDatasetPDB` in `protein_mpnn_utils.py` expect all data files to be pre-existing in local directories
2. **Storage Limitations**: 19,000+ PDB files would require ~5-10GB of storage, which may exceed available space
3. **Download Inefficiency**: Re-downloading the same PDB files across training runs wastes bandwidth and time  
4. **No Streaming Support**: No built-in mechanism for "fetch PDB from RCSB during training loop"
5. **Batch Size Constraints**: Current implementation requires minimum 4 positive and 4 negative samples per batch

### Why This Occurs

- ProteinMPNN was designed for research settings where datasets are manually curated and pre-processed
- The original training used a fixed snapshot of PDB structures (Aug 2021) rather than dynamic downloading
- No consideration for storage-constrained environments or large-scale streaming scenarios

## Architecture Overview

The streaming system consists of four main components:

1. **PDB Cache Manager**: Intelligent local storage with LRU eviction
2. **Dynamic Dataset**: Custom PyTorch Dataset that fetches PDBs on-demand  
3. **Download Coordinator**: Manages concurrent downloads and prevents duplicates
4. **Storage Monitor**: Tracks disk usage and triggers cleanup

## Detailed Technical Specifications

### Phase 1: PDB Cache Management System

**Component**: `hybrid/data/pdb_cache.py`

```python
import os
import time
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import requests
import warnings

class PDBCacheManager:
    """
    Intelligent cache for PDB files with automatic downloading and LRU eviction
    """
    
    def __init__(self, 
                 cache_dir: str = "./pdb_cache",
                 max_cache_size_gb: float = 2.0,
                 max_concurrent_downloads: int = 5,
                 rcsb_base_url: str = "https://files.rcsb.org/download"):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.rcsb_base_url = rcsb_base_url
        
        # Thread-safe data structures
        self.cache_lock = threading.RLock()
        self.download_lock = threading.RLock()
        
        # LRU tracking: pdb_id -> last_access_time
        self.access_times: OrderedDict = OrderedDict()
        
        # Download management
        self.downloading: Dict[str, threading.Event] = {}
        self.download_semaphore = threading.Semaphore(max_concurrent_downloads)
        
        # Performance tracking
        self.download_stats = {
            'total_downloads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_download_time': 0.0,
            'failed_downloads': 0
        }
        
        # Initialize cache by scanning existing files
        self._scan_existing_cache()
    
    def get_pdb_path(self, pdb_id: str) -> str:
        """
        Get path to PDB file, downloading if necessary
        
        Args:
            pdb_id: 4-character PDB identifier (e.g., "1ABC")
            
        Returns:
            Path to local PDB file
            
        Raises:
            FileNotFoundError: If download fails
        """
        pdb_id = pdb_id.upper()
        file_path = self.cache_dir / f"{pdb_id}.pdb"
        
        with self.cache_lock:
            # Check if file exists locally
            if file_path.exists():
                self._update_access_time(pdb_id)
                self.download_stats['cache_hits'] += 1
                return str(file_path)
        
        # File not in cache, need to download
        self.download_stats['cache_misses'] += 1
        return self._download_pdb(pdb_id)
    
    def _download_pdb(self, pdb_id: str) -> str:
        """Download PDB file with deduplication and concurrent download management"""
        file_path = self.cache_dir / f"{pdb_id}.pdb"
        
        with self.download_lock:
            # Check if another thread is already downloading this PDB
            if pdb_id in self.downloading:
                download_event = self.downloading[pdb_id]
            else:
                download_event = threading.Event()
                self.downloading[pdb_id] = download_event
        
        # Wait for download if another thread is handling it
        if not download_event.is_set():
            if threading.current_thread() != download_event:
                download_event.wait(timeout=60)  # Wait up to 60s
                
                # Check if download succeeded
                if file_path.exists():
                    with self.cache_lock:
                        self._update_access_time(pdb_id)
                    return str(file_path)
                else:
                    raise FileNotFoundError(f"Download of {pdb_id} failed or timed out")
        
        # This thread is responsible for downloading
        try:
            with self.download_semaphore:  # Limit concurrent downloads
                start_time = time.time()
                
                # Make storage space if needed
                self._ensure_cache_space()
                
                # Download the file
                url = f"{self.rcsb_base_url}/{pdb_id}.pdb"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Write to temporary file first, then rename (atomic operation)
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    f.write(response.text)
                
                temp_path.rename(file_path)
                
                # Update stats and cache tracking
                download_time = time.time() - start_time
                
                with self.cache_lock:
                    self._update_access_time(pdb_id)
                    self.download_stats['total_downloads'] += 1
                    self.download_stats['total_download_time'] += download_time
                
                # Signal completion to waiting threads
                download_event.set()
                
                return str(file_path)
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            download_event.set()  # Release waiting threads
            raise FileNotFoundError(f"Failed to download {pdb_id}: {str(e)}")
        
        finally:
            # Cleanup download tracking
            with self.download_lock:
                self.downloading.pop(pdb_id, None)
    
    def _ensure_cache_space(self, target_free_bytes: int = 100_000_000):  # 100MB buffer
        """Evict old files if cache is getting full"""
        current_size = self._get_cache_size()
        
        if current_size + target_free_bytes > self.max_cache_size_bytes:
            bytes_to_free = current_size + target_free_bytes - self.max_cache_size_bytes
            self._evict_lru_files(bytes_to_free)
    
    def _evict_lru_files(self, bytes_to_free: int):
        """Remove least recently used files to free up space"""
        freed_bytes = 0
        
        # Sort by access time (oldest first)
        sorted_files = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for pdb_id, _ in sorted_files:
            if freed_bytes >= bytes_to_free:
                break
                
            file_path = self.cache_dir / f"{pdb_id}.pdb"
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                freed_bytes += file_size
                
                # Remove from tracking
                del self.access_times[pdb_id]
                
                warnings.warn(f"Evicted {pdb_id}.pdb ({file_size} bytes) from cache")
    
    def get_cache_stats(self) -> Dict:
        """Return cache performance statistics"""
        total_requests = self.download_stats['cache_hits'] + self.download_stats['cache_misses']
        hit_rate = self.download_stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        avg_download_time = 0
        if self.download_stats['total_downloads'] > 0:
            avg_download_time = self.download_stats['total_download_time'] / self.download_stats['total_downloads']
        
        return {
            'cache_size_mb': self._get_cache_size() / (1024**2),
            'cache_hit_rate': hit_rate,
            'total_files': len(self.access_times),
            'avg_download_time_sec': avg_download_time,
            **self.download_stats
        }
```

**Key Features**:
- **Thread-safe**: Handles concurrent downloads without race conditions
- **LRU Eviction**: Automatically removes old files when storage limit reached  
- **Deduplication**: Prevents multiple downloads of the same PDB
- **Performance Tracking**: Monitors cache hit rates and download times
- **Atomic Downloads**: Uses temporary files to prevent corrupted downloads

### Phase 2: Streaming Dataset Implementation

**Component**: `hybrid/data/streaming_dataset.py`

```python
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset
from protein_mpnn_utils import parse_PDB, StructureDataset

class StreamingPDBDataset(IterableDataset):
    """
    Streaming dataset that downloads PDBs on-demand during training
    Integrates with existing ProteinMPNN data loading pipeline
    """
    
    def __init__(self,
                 pdb_list: List[str],
                 cache_manager: PDBCacheManager,
                 target_samples: int = 19000,
                 positive_ratio: float = 0.5,
                 negative_methods: List[str] = None,
                 batch_size: int = 4,
                 timing_enabled: bool = True,
                 max_sequence_length: int = 500,
                 min_sequence_length: int = 20):
        
        super().__init__()
        
        self.pdb_list = pdb_list
        self.cache_manager = cache_manager
        self.target_samples = target_samples
        self.positive_ratio = positive_ratio
        self.negative_methods = negative_methods or ['random', 'mutated']
        self.timing_enabled = timing_enabled
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        
        # Sample generation tracking
        self.samples_generated = 0
        self.positive_samples = 0
        self.negative_samples = 0
        
        # Performance timing
        self.timing_stats = {
            'download_time': 0.0,
            'parse_time': 0.0,
            'augmentation_time': 0.0,
            'total_time': 0.0,
            'samples_per_second': 0.0
        }
        
        # Validate batch size requirements
        min_positive = int(batch_size * positive_ratio)
        min_negative = batch_size - min_positive
        
        if min_positive < 1 or min_negative < 1:
            raise ValueError(f"Batch size {batch_size} too small for positive ratio {positive_ratio}")
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate through samples, generating them dynamically"""
        
        start_time = time.time() if self.timing_enabled else 0
        
        while self.samples_generated < self.target_samples:
            
            # Determine if we need positive or negative sample
            current_pos_ratio = self.positive_samples / max(1, self.samples_generated)
            need_positive = current_pos_ratio < self.positive_ratio
            
            if need_positive:
                sample = self._generate_positive_sample()
            else:
                sample = self._generate_negative_sample()
            
            if sample is not None:
                yield sample
                self.samples_generated += 1
                
                # Update timing statistics
                if self.timing_enabled and self.samples_generated % 100 == 0:
                    self._update_timing_stats(start_time)
    
    def _generate_positive_sample(self) -> Optional[Dict]:
        """Generate a positive (stable) training sample"""
        
        # Select random PDB
        pdb_id = random.choice(self.pdb_list)
        
        try:
            # Download PDB if needed (with timing)
            download_start = time.time()
            pdb_path = self.cache_manager.get_pdb_path(pdb_id)
            download_time = time.time() - download_start
            
            # Parse PDB to get structure and sequence (with timing)
            parse_start = time.time()
            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=None)
            parse_time = time.time() - parse_start
            
            if not pdb_dict_list:
                return None
            
            # Use first chain/assembly
            pdb_dict = pdb_dict_list[0]
            
            # Filter by sequence length
            sequence_length = len(pdb_dict['seq'])
            if not (self.min_sequence_length <= sequence_length <= self.max_sequence_length):
                return None
            
            # Create positive sample
            sample = {
                'pdb_id': pdb_id,
                'coordinates': pdb_dict['coords'],
                'sequence': pdb_dict['seq'],
                'mask': pdb_dict['mask'],
                'chain_encoding': pdb_dict.get('chain_encoding', None),
                'residue_idx': pdb_dict.get('residue_idx', None),
                'label': 1,  # Positive label
                'generation_method': 'native',
                'download_time': download_time,
                'parse_time': parse_time,
                'length': sequence_length
            }
            
            self.positive_samples += 1
            
            # Update timing
            if self.timing_enabled:
                self.timing_stats['download_time'] += download_time
                self.timing_stats['parse_time'] += parse_time
            
            return sample
            
        except Exception as e:
            warnings.warn(f"Failed to process {pdb_id}: {str(e)}")
            return None
    
    def _generate_negative_sample(self) -> Optional[Dict]:
        """Generate a negative (unstable) training sample"""
        
        # Select random PDB as backbone
        pdb_id = random.choice(self.pdb_list)
        
        try:
            # Get backbone structure
            pdb_path = self.cache_manager.get_pdb_path(pdb_id)
            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=None)
            
            if not pdb_dict_list:
                return None
                
            pdb_dict = pdb_dict_list[0]
            
            # Filter by length
            if not (self.min_sequence_length <= len(pdb_dict['seq']) <= self.max_sequence_length):
                return None
            
            # Generate negative sequence using specified method
            aug_start = time.time()
            negative_method = random.choice(self.negative_methods)
            
            if negative_method == 'random':
                negative_seq = self._generate_random_sequence(len(pdb_dict['seq']))
            elif negative_method == 'mutated':
                negative_seq = self._mutate_sequence(pdb_dict['seq'], mutation_rate=0.3)
            else:
                warnings.warn(f"Unknown negative method: {negative_method}")
                return None
            
            aug_time = time.time() - aug_start
            
            sample = {
                'pdb_id': pdb_id,
                'coordinates': pdb_dict['coords'],
                'sequence': negative_seq,
                'mask': pdb_dict['mask'],
                'chain_encoding': pdb_dict.get('chain_encoding', None),
                'residue_idx': pdb_dict.get('residue_idx', None),
                'label': 0,  # Negative label
                'generation_method': negative_method,
                'download_time': 0.0,  # Already cached from positive samples
                'parse_time': 0.0,
                'augmentation_time': aug_time,
                'length': len(negative_seq)
            }
            
            self.negative_samples += 1
            
            if self.timing_enabled:
                self.timing_stats['augmentation_time'] += aug_time
            
            return sample
            
        except Exception as e:
            warnings.warn(f"Failed to generate negative for {pdb_id}: {str(e)}")
            return None
    
    def _generate_random_sequence(self, length: int) -> str:
        """Generate random protein sequence with realistic amino acid frequencies"""
        
        # Amino acid frequencies from natural proteins
        aa_frequencies = {
            'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
            'Q': 0.037, 'E': 0.058, 'G': 0.074, 'H': 0.029, 'I': 0.038,
            'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
            'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068
        }
        
        amino_acids = list(aa_frequencies.keys())
        weights = list(aa_frequencies.values())
        
        return ''.join(random.choices(amino_acids, weights=weights, k=length))
    
    def _mutate_sequence(self, sequence: str, mutation_rate: float = 0.3) -> str:
        """Create destabilizing mutations of a sequence"""
        
        amino_acids = 'ARNDCQEGHILKMFPSTWYV'
        mutated_seq = list(sequence)
        
        num_mutations = max(1, int(len(sequence) * mutation_rate))
        positions = random.sample(range(len(sequence)), num_mutations)
        
        for pos in positions:
            original_aa = mutated_seq[pos]
            # Choose different amino acid
            new_aa = random.choice([aa for aa in amino_acids if aa != original_aa])
            mutated_seq[pos] = new_aa
        
        return ''.join(mutated_seq)
    
    def _update_timing_stats(self, start_time: float):
        """Update timing statistics for performance monitoring"""
        
        total_time = time.time() - start_time
        self.timing_stats['total_time'] = total_time
        self.timing_stats['samples_per_second'] = self.samples_generated / total_time
    
    def get_progress(self) -> Dict:
        """Get current dataset generation progress"""
        
        return {
            'samples_generated': self.samples_generated,
            'target_samples': self.target_samples,
            'progress_percent': (self.samples_generated / self.target_samples) * 100,
            'positive_samples': self.positive_samples,
            'negative_samples': self.negative_samples,
            'actual_positive_ratio': self.positive_samples / max(1, self.samples_generated),
            'target_positive_ratio': self.positive_ratio,
            'timing_stats': self.timing_stats,
            'cache_stats': self.cache_manager.get_cache_stats()
        }
```

### Phase 3: PDB List Generation and Management

**Component**: `hybrid/data/pdb_manager.py`

```python
import requests
import json
from typing import List, Dict, Optional, Set
from pathlib import Path

class PDBListManager:
    """
    Manages lists of PDB IDs for training, with filtering and validation
    """
    
    def __init__(self, cache_dir: str = "./pdb_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # PDB search API endpoint
        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
    def get_filtered_pdb_list(self,
                              max_resolution: float = 3.5,
                              max_length: int = 500,
                              min_length: int = 20,
                              experimental_methods: List[str] = None,
                              target_count: int = 20000) -> List[str]:
        """
        Get filtered list of PDB IDs suitable for training
        
        Args:
            max_resolution: Maximum resolution in Angstroms
            max_length: Maximum sequence length  
            min_length: Minimum sequence length
            experimental_methods: List of experimental methods (default: X-ray, cryo-EM)
            target_count: Target number of structures
            
        Returns:
            List of PDB IDs
        """
        
        if experimental_methods is None:
            experimental_methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
        
        # Build search query
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "in",
                            "value": experimental_methods
                        }
                    },
                    {
                        "type": "terminal", 
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": max_resolution
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text", 
                        "parameters": {
                            "attribute": "entity_poly.rcsb_sample_sequence_length",
                            "operator": "range",
                            "value": [min_length, max_length]
                        }
                    }
                ]
            },
            "request_options": {
                "return_all_hits": True
            },
            "return_type": "entry"
        }
        
        # Execute search
        try:
            response = requests.post(self.search_url, json=query)
            response.raise_for_status()
            
            results = response.json()
            pdb_ids = [hit['identifier'] for hit in results.get('result_set', [])]
            
            # Limit to target count
            if len(pdb_ids) > target_count:
                import random
                random.shuffle(pdb_ids)
                pdb_ids = pdb_ids[:target_count]
            
            print(f"Found {len(pdb_ids)} PDB structures matching criteria")
            
            # Cache the list
            cache_file = self.cache_dir / "filtered_pdb_list.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'pdb_ids': pdb_ids,
                    'filters': {
                        'max_resolution': max_resolution,
                        'max_length': max_length,
                        'min_length': min_length,
                        'experimental_methods': experimental_methods
                    },
                    'total_found': len(pdb_ids)
                }, f, indent=2)
            
            return pdb_ids
            
        except Exception as e:
            print(f"Error searching PDB: {e}")
            
            # Fallback to cached list if search fails
            cache_file = self.cache_dir / "filtered_pdb_list.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data['pdb_ids']
            
            # Final fallback to a small manually curated list
            return self._get_fallback_pdb_list()
    
    def _get_fallback_pdb_list(self) -> List[str]:
        """Fallback list of high-quality structures for testing"""
        
        return [
            # High-resolution crystal structures
            "1UBQ", "1VII", "2CRO", "1ROP", "1TEN", "1PIN", "1FSD", "1TIT",
            "1L2Y", "1YRF", "2GB1", "1PGB", "1SHG", "1TGR", "1CLB", "1CTF",
            # Additional structures for diversity
            "3HTN", "4YOW", "4GYT", "6EHB", "5L33", "6MRR", "1HIV", "1RB9"
        ] * 100  # Repeat to reach target sample count through augmentation
```

### Phase 4: Integration with Training Pipeline

**Component**: `hybrid/training/train_energy_streaming.py`

```python
def create_streaming_data_loader(config: Dict) -> torch.utils.data.DataLoader:
    """
    Create data loader that integrates streaming PDB dataset with existing training
    """
    
    # Initialize PDB management
    pdb_manager = PDBListManager(cache_dir=config['data']['cache_dir'])
    pdb_list = pdb_manager.get_filtered_pdb_list(
        target_count=config['data']['target_pdb_count']
    )
    
    # Initialize cache with reasonable defaults
    cache_manager = PDBCacheManager(
        cache_dir=config['data']['cache_dir'],
        max_cache_size_gb=config['data']['max_cache_gb'],
        max_concurrent_downloads=config['data']['max_downloads']
    )
    
    # Create streaming dataset
    dataset = StreamingPDBDataset(
        pdb_list=pdb_list,
        cache_manager=cache_manager,
        target_samples=config['data']['target_samples'],
        positive_ratio=config['data']['positive_ratio'],
        negative_methods=config['data']['negative_methods'],
        batch_size=config['training']['batch_size'],
        timing_enabled=config['monitoring']['timing_enabled']
    )
    
    # Custom collate function for variable-length sequences
    def collate_fn(batch):
        # Convert streaming samples to ProteinMPNN format
        return StructureDataset.collate_sequences(batch)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return loader, cache_manager

# Modified training loop with timing instrumentation
class StreamingEnergyTrainer:
    
    def __init__(self, config: Dict):
        self.config = config
        self.timing_enabled = config['monitoring']['timing_enabled']
        
        # Training timing statistics
        self.epoch_times = []
        self.batch_times = []
        self.sample_timing_log = []
        
    def train_epoch(self, model, loader, optimizer, epoch: int):
        """Modified training loop with comprehensive timing"""
        
        model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            batch_start = time.time()
            
            # Extract timing info from samples (if available)
            if self.timing_enabled and 'download_time' in batch:
                batch_timing = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'download_time': batch['download_time'].mean().item(),
                    'parse_time': batch['parse_time'].mean().item(),
                    'batch_size': len(batch['sequence'])
                }
                self.sample_timing_log.append(batch_timing)
            
            # Standard training step
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = self.compute_loss(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch timing
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time)
            
            # Progress logging
            if batch_idx % 100 == 0:
                avg_batch_time = np.mean(self.batch_times[-100:])
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss={loss.item():.4f}, "
                      f"Batch_time={batch_time:.3f}s, "
                      f"Avg_batch_time={avg_batch_time:.3f}s")
        
        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)
        
        # Save timing statistics
        if self.timing_enabled:
            self.save_timing_stats(epoch)
        
        return total_loss / num_batches
    
    def save_timing_stats(self, epoch: int):
        """Save comprehensive timing statistics for analysis"""
        
        stats = {
            'epoch': epoch,
            'epoch_times': self.epoch_times,
            'batch_times': self.batch_times[-1000:],  # Keep recent batches
            'sample_timing_log': self.sample_timing_log[-1000:],
            'avg_epoch_time': np.mean(self.epoch_times),
            'avg_batch_time': np.mean(self.batch_times[-100:]),
            'samples_per_second': self.config['training']['batch_size'] / np.mean(self.batch_times[-100:]),
            'estimated_total_training_time': len(self.epoch_times) * np.mean(self.epoch_times)
        }
        
        timing_file = Path(self.config['monitoring']['timing_log_dir']) / f"timing_epoch_{epoch}.json"
        with open(timing_file, 'w') as f:
            json.dump(stats, f, indent=2)
```

### Phase 5: Configuration and Monitoring

**Component**: `hybrid/training/config_streaming.json`

```json
{
  "experiment_name": "streaming_energy_model_training",
  "seed": 42,
  
  "data": {
    "cache_dir": "./pdb_streaming_cache",
    "target_samples": 19000,
    "target_pdb_count": 5000,
    "positive_ratio": 0.5,
    "negative_methods": ["random", "mutated"],
    "max_cache_gb": 2.0,
    "max_downloads": 8,
    "max_sequence_length": 500,
    "min_sequence_length": 20
  },
  
  "model": {
    "mpnn_encoder": {
      "model_name": "v_48_020",
      "hidden_dim": 128,
      "freeze_layers": true
    },
    "energy_head": {
      "hidden_dim": 512,
      "num_layers": 3,
      "dropout": 0.1
    }
  },
  
  "training": {
    "max_epochs": 50,
    "batch_size": 8,
    "num_workers": 4,
    "max_grad_norm": 1.0,
    "patience": 15
  },
  
  "monitoring": {
    "timing_enabled": true,
    "timing_log_dir": "./timing_logs",
    "cache_stats_frequency": 1000,
    "progress_log_frequency": 100
  }
}
```

## Implementation Timeline

### Phase 1: Infrastructure Foundation (Week 1)
- **Day 1-2**: Implement `PDBCacheManager` with basic downloading and caching
- **Day 3-4**: Add LRU eviction and thread-safe concurrent downloads  
- **Day 5-7**: Test cache with sample PDB downloads, validate performance

**Success Criteria**: 
- Cache can download and store PDBs without conflicts
- LRU eviction works correctly when storage limits reached
- Thread-safe operation verified under concurrent access

### Phase 2: Streaming Dataset (Week 2)  
- **Day 1-3**: Implement `StreamingPDBDataset` with basic iteration
- **Day 4-5**: Add negative sample generation and timing instrumentation
- **Day 6-7**: Integration testing with existing ProteinMPNN data loaders

**Success Criteria**:
- Dataset generates balanced positive/negative samples  
- Integrates with ProteinMPNN `parse_PDB` and data structures
- Timing data collection works correctly

### Phase 3: PDB Management and Training Integration (Week 3)
- **Day 1-2**: Implement `PDBListManager` with RCSB search API
- **Day 3-4**: Modify training pipeline to use streaming data loader
- **Day 5-7**: End-to-end testing with small-scale training run

**Success Criteria**:
- Can generate PDB lists matching training criteria
- Training loop works with streaming data
- No memory leaks or performance issues

### Phase 4: Optimization and Production (Week 4)
- **Day 1-2**: Performance optimization and memory management
- **Day 3-4**: Comprehensive testing with full 19K sample target
- **Day 5-7**: Documentation and configuration tuning

**Success Criteria**:
- Can handle 19,000 samples with reasonable memory usage
- Training speed comparable to static dataset loading  
- Cache management works effectively over long training runs

## Risk Mitigation

### Technical Risks

**Risk**: Network failures during PDB download  
**Mitigation**: Implement retry logic with exponential backoff, fallback to cached lists
**Fallback**: Pre-download critical PDBs, use smaller reliable dataset

**Risk**: Storage management failures  
**Mitigation**: Robust LRU implementation, storage monitoring, configurable limits
**Fallback**: Fail gracefully to download-only mode without permanent storage

**Risk**: Integration issues with existing ProteinMPNN code  
**Mitigation**: Extensive testing, maintain compatibility with existing interfaces
**Fallback**: Wrapper layer that converts streaming samples to expected format

### Performance Risks

**Risk**: Download speed bottlenecks training  
**Mitigation**: Aggressive caching, concurrent downloads, prefetching
**Fallback**: Increase cache size, reduce target sample count

**Risk**: Memory usage grows uncontrollably  
**Mitigation**: Strict memory monitoring, immediate cleanup after use
**Fallback**: Reduce batch size, implement sample-level caching

## Success Metrics

### Primary Metrics
1. **Training Throughput**: Samples processed per second during training
2. **Cache Efficiency**: Cache hit rate and download frequency
3. **Memory Usage**: Peak and average memory consumption during training
4. **Storage Management**: Effective utilization of cache space

### Secondary Metrics  
1. **Network Efficiency**: Total bandwidth used vs. useful data obtained
2. **Error Rates**: Failed downloads and parsing errors
3. **Training Stability**: Consistent performance across epochs

### Performance Targets
- **Training Speed**: ≥90% of static dataset performance
- **Cache Hit Rate**: ≥70% after initial warmup
- **Memory Usage**: ≤4GB peak usage for 19K samples
- **Storage Efficiency**: ≤2GB cache size for target dataset

## Conclusion

This implementation plan provides a comprehensive solution to the dynamic PDB downloading problem while maintaining compatibility with existing ProteinMPNN infrastructure. The streaming approach enables large-scale training without overwhelming local storage, while intelligent caching ensures efficient resource usage.

Key innovations:
- **Thread-safe concurrent downloading** prevents bottlenecks
- **LRU cache management** maintains bounded storage usage
- **Timing instrumentation** provides performance insights
- **Configurable sample targets** enables flexible dataset sizing
- **Seamless integration** with existing training pipeline

The phased implementation approach reduces risk and allows for incremental validation, ensuring a robust production-ready system for large-scale protein design model training.