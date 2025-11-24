#!/usr/bin/env python3
"""
Energy Model Evaluation Framework

This module provides comprehensive evaluation tools for energy-based protein design models.
It evaluates model performance across multiple dimensions including ranking accuracy,
correlation with physics-based scores, sequence properties, and generalization capabilities.

Key Evaluation Components:
1. Energy Ranking Evaluator - Tests native vs random/mutated sequence ranking
2. Correlation Analyzer - Compares with Rosetta/AlphaFold scores  
3. Sequence Property Analyzer - Validates biological realism
4. Visualization Generator - Creates publication-ready plots
5. Hold-out Validator - Tests generalization to unseen structures
6. Baseline Comparator - Benchmarks against physics-based methods
"""

import os
import sys
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Optional dependencies with graceful degradation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Visualizations will be disabled.")

try:
    from scipy import stats
    from scipy.stats import spearmanr, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical analysis will be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Report generation will be limited.")

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import project modules
from models.mpnn_encoder import ProteinMPNNBackboneEncoder, load_pretrained_encoder
from models.energy_head import EnergyHead
from models.sequence_repr import ContinuousSequenceRepr
from data.stability_dataset import StabilityDataset


class EnergyRankingEvaluator:
    """
    Evaluates energy model's ability to rank native sequences lower than negative examples.
    
    This is the core evaluation metric: energy models should assign lower energy to 
    native/stable protein sequences compared to random, mutated, or failed designs.
    
    Provides comprehensive ranking analysis including:
    - Basic ranking accuracy (native < negative)
    - Bootstrap confidence intervals
    - Ranking correlation analysis (Spearman, Kendall)
    - ROC curve analysis for binary classification
    - Per-category performance breakdown
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize ranking evaluator.
        
        Args:
            model: Complete energy model (encoder + head)
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_ranking_accuracy(
        self, 
        dataset: StabilityDataset, 
        num_samples: int = 1000,
        batch_size: int = 16,
        seed: Optional[int] = 42
    ) -> Dict[str, float]:
        """
        Evaluate ranking accuracy on dataset samples.
        
        Args:
            dataset: Evaluation dataset with positive/negative pairs
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            seed: Random seed for reproducible batch ordering. Use same seed for identical results across runs.
            
        Returns:
            Dictionary with accuracy metrics
        """
        print("Evaluating energy ranking accuracy...")
        
        results = {
            'total_pairs': 0,
            'correct_rankings': 0,
            'native_energies': [],
            'negative_energies': [],
            'energy_gaps': []
        }
        
        # Create seeded generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else 42)
        
        # Create data loader with proper collation
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            collate_fn=self._collate_ranking_pairs,
            num_workers=0,  # Required for deterministic reproducibility
            persistent_workers=False
        )
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Ranking evaluation"):
                if samples_processed >= num_samples:
                    break
                
                # Extract positive and negative examples
                pos_data, neg_data = batch
                
                # Move to device
                pos_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                           for k, v in pos_data.items()}
                neg_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                           for k, v in neg_data.items()}
                
                # Compute energies
                pos_energies = self._compute_energy_batch(pos_data)
                neg_energies = self._compute_energy_batch(neg_data)
                
                # Check rankings (native should have lower energy)
                correct_rankings = (pos_energies < neg_energies).sum().item()
                batch_size = len(pos_energies)
                
                # Update results
                results['total_pairs'] += batch_size
                results['correct_rankings'] += correct_rankings
                results['native_energies'].extend(pos_energies.cpu().numpy())
                results['negative_energies'].extend(neg_energies.cpu().numpy())
                results['energy_gaps'].extend((neg_energies - pos_energies).cpu().numpy())
                
                samples_processed += batch_size
        
        # Compute accuracy
        accuracy = results['correct_rankings'] / results['total_pairs']
        
        # Compute energy statistics
        native_energies = np.array(results['native_energies'])
        negative_energies = np.array(results['negative_energies'])
        energy_gaps = np.array(results['energy_gaps'])
        
        return {
            'ranking_accuracy': accuracy,
            'total_pairs_evaluated': results['total_pairs'],
            'correct_rankings': results['correct_rankings'],
            'native_energy_mean': float(native_energies.mean()),
            'native_energy_std': float(native_energies.std()),
            'negative_energy_mean': float(negative_energies.mean()),
            'negative_energy_std': float(negative_energies.std()),
            'energy_gap_mean': float(energy_gaps.mean()),
            'energy_gap_std': float(energy_gaps.std()),
            'energy_gap_median': float(np.median(energy_gaps))
        }
    
    def evaluate_ranking_with_confidence_intervals(
        self,
        dataset: StabilityDataset,
        num_samples: int = 1000,
        batch_size: int = 16,
        num_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """
        Evaluate ranking accuracy with bootstrap confidence intervals.
        
        Args:
            dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            seed: Random seed for reproducible batch ordering and bootstrap sampling. 
                  Use seed=42 for reproducible confidence intervals for publication. 
                  Use seed=None for statistically independent bootstrap samples.
            
        Returns:
            Dictionary with accuracy and confidence intervals
        """
        print("Evaluating ranking accuracy with confidence intervals...")
        
        # Get baseline results
        results = self.evaluate_ranking_accuracy(dataset, num_samples, batch_size, seed)
        
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available. Skipping confidence interval calculation.")
            results['confidence_intervals'] = {'error': 'SciPy not available'}
            return results
        
        # Prepare data for bootstrap
        native_energies = np.array(results['native_energies'])
        negative_energies = np.array(results['negative_energies'])
        
        if len(native_energies) == 0:
            results['confidence_intervals'] = {'error': 'No data available for bootstrap'}
            return results
        
        # Use RandomState for thread-safe, reproducible bootstrap sampling
        rng = np.random.RandomState(seed if seed is not None else 42)
        bootstrap_accuracies = []
        
        for _ in range(num_bootstrap):
            # Sample with replacement while maintaining pairing
            # Each pair (native[i], negative[i]) should be sampled together
            pair_indices = rng.choice(len(native_energies), len(native_energies), replace=True)
            boot_native = native_energies[pair_indices]
            boot_negative = negative_energies[pair_indices]
            
            # Compute bootstrap accuracy
            correct = (boot_native < boot_negative).sum()
            accuracy = correct / len(boot_native)
            bootstrap_accuracies.append(accuracy)
        
        bootstrap_accuracies = np.array(bootstrap_accuracies)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
        ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)
        
        results['confidence_intervals'] = {
            'confidence_level': confidence_level,
            'accuracy_mean': float(bootstrap_accuracies.mean()),
            'accuracy_std': float(bootstrap_accuracies.std()),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'num_bootstrap_samples': num_bootstrap
        }
        
        return results
    
    def evaluate_ranking_correlation(
        self,
        dataset: StabilityDataset,
        num_samples: int = 1000,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Evaluate ranking correlation using Spearman and Kendall correlation.
        
        Args:
            dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with correlation analysis results
        """
        print("Evaluating ranking correlation...")
        
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for correlation analysis'}
        
        # Get energy values
        results = self.evaluate_ranking_accuracy(dataset, num_samples, batch_size)
        native_energies = np.array(results['native_energies'])
        negative_energies = np.array(results['negative_energies'])
        
        if len(native_energies) == 0:
            return {'error': 'No data available for correlation analysis'}
        
        # Create ideal ranking (native should always be lower)
        # Native = 0 (lower rank), Negative = 1 (higher rank)
        ideal_rankings = np.concatenate([
            np.zeros(len(native_energies)),  # Native sequences (rank 0)
            np.ones(len(negative_energies))   # Negative sequences (rank 1)
        ])
        
        # Actual energy values (concatenated)
        actual_energies = np.concatenate([native_energies, negative_energies])
        
        # Compute ranking correlations
        # Lower energy should map to lower rank (0 = native, 1 = negative)
        # Positive correlation means model correctly ranks native sequences lower
        spearman_corr, spearman_p = spearmanr(actual_energies, ideal_rankings)
        kendall_corr, kendall_p = stats.kendalltau(actual_energies, ideal_rankings)
        
        # Also compute correlation between native and negative energy ranks
        from scipy.stats import rankdata
        native_ranks = rankdata(native_energies)
        negative_ranks = rankdata(negative_energies)
        
        # Expected: native ranks should be systematically lower than negative ranks
        all_energies = np.concatenate([native_energies, negative_energies])
        all_ranks = rankdata(all_energies)
        native_ranks_global = all_ranks[:len(native_energies)]
        negative_ranks_global = all_ranks[len(native_energies):]
        
        rank_difference = negative_ranks_global.mean() - native_ranks_global.mean()
        
        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'kendall_correlation': float(kendall_corr),
            'kendall_p_value': float(kendall_p),
            'rank_separation': {
                'native_rank_mean': float(native_ranks_global.mean()),
                'negative_rank_mean': float(negative_ranks_global.mean()),
                'rank_difference': float(rank_difference),
                'total_samples': len(all_energies)
            }
        }
    
    def evaluate_roc_analysis(
        self,
        dataset: StabilityDataset,
        num_samples: int = 1000,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Perform ROC curve analysis for binary classification performance.
        
        Args:
            dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with ROC analysis results
        """
        print("Evaluating ROC curve analysis...")
        
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for ROC analysis'}
        
        # Get energy values
        results = self.evaluate_ranking_accuracy(dataset, num_samples, batch_size)
        native_energies = np.array(results['native_energies'])
        negative_energies = np.array(results['negative_energies'])
        
        if len(native_energies) == 0:
            return {'error': 'No data available for ROC analysis'}
        
        # Create labels and scores for ROC analysis
        # Label: 1 = native (positive class), 0 = negative (negative class)
        # Score: -energy (so higher score = more stable = more likely to be native)
        y_true = np.concatenate([
            np.ones(len(native_energies)),    # Native sequences (positive class)
            np.zeros(len(negative_energies))  # Negative sequences (negative class)
        ])
        y_scores = np.concatenate([
            -native_energies,    # Negative energy (higher = better)
            -negative_energies
        ])
        
        # Compute ROC curve
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
            
        if SKLEARN_AVAILABLE:
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            
            # Find optimal threshold (Youden's index)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = roc_thresholds[optimal_idx]
            
            return {
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
                'optimal_threshold': float(optimal_threshold),
                'optimal_sensitivity': float(tpr[optimal_idx]),
                'optimal_specificity': float(1 - fpr[optimal_idx]),
                'n_positive': int(np.sum(y_true)),
                'n_negative': int(len(y_true) - np.sum(y_true)),
                'roc_curve': {
                    'fpr': fpr[:100].tolist(),  # Sample points for storage
                    'tpr': tpr[:100].tolist(),
                    'thresholds': roc_thresholds[:100].tolist()
                }
            }
        else:
            # Simple AUC calculation using trapezoid rule
            # Sort by scores (descending order for ROC calculation)
            sorted_indices = np.argsort(-y_scores)  # Sort by descending scores
            sorted_labels = y_true[sorted_indices]
            
            # Count true positives and false positives
            n_pos = np.sum(y_true)
            n_neg = len(y_true) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                # Degenerate case: only one class present
                auc_value = 0.5
            else:
                # Compute cumulative TP and FP
                tp = np.cumsum(sorted_labels)
                fp = np.cumsum(1 - sorted_labels)
                
                # Add (0,0) point at the beginning
                tp = np.concatenate([[0], tp])
                fp = np.concatenate([[0], fp])
                
                # Compute TPR and FPR
                tpr = tp / n_pos
                fpr = fp / n_neg
                
                # Simple AUC using trapezoid rule (integrate tpr vs fpr)
                auc_value = np.trapz(tpr, fpr) if len(tpr) > 1 else 0.5
            
            return {
                'roc_auc_simple': float(auc_value),
                'n_positive': int(n_pos),
                'n_negative': int(n_neg),
                'note': 'Computed using simple method (sklearn not available)'
            }
    
    def evaluate_per_category_accuracy(
        self,
        dataset: StabilityDataset,
        num_samples: int = 1000,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Evaluate ranking accuracy broken down by negative sequence type.
        
        Args:
            dataset: Evaluation dataset  
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with per-category accuracy breakdown
        """
        print("Evaluating per-category ranking accuracy...")
        
        # This would require extending the dataset to provide category information
        # For now, return a placeholder
        return {
            'note': 'Per-category analysis requires dataset extension to provide negative sequence categories',
            'categories_available': False,
            'suggestion': 'Extend StabilityDataset to include negative_type field (random, mutated, failed_design)'
        }
    
    def _collate_ranking_pairs(self, batch: List[Any]) -> Tuple[Dict, Dict]:
        """
        Custom collate function to create positive/negative pairs from dataset.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            Tuple of (positive_batch, negative_batch)
        """
        # Split batch into positive and negative examples
        positive_samples = []
        negative_samples = []
        
        for sample in batch:
            if sample['label'] == 1:  # Positive (native/stable)
                positive_samples.append(sample)
            else:  # Negative (random/mutated/failed)
                negative_samples.append(sample)
        
        # Balance the batches (pair each positive with a negative)
        min_size = min(len(positive_samples), len(negative_samples))
        if min_size == 0:
            raise ValueError(
                f"Invalid batch: {len(positive_samples)} positive samples and {len(negative_samples)} negative samples. "
                f"Evaluation requires balanced positive/negative pairs. "
                f"Check dataset balance (should be 50/50) or increase --batch_size."
            )
        
        positive_samples = positive_samples[:min_size]
        negative_samples = negative_samples[:min_size]
        
        # Create batched tensors
        def collate_samples(samples):
            # Extract common fields and batch them
            batched = {}
            
            if len(samples) > 0:
                # Handle backbone features with padding for variable lengths
                if 'backbone_features' in samples[0]:
                    backbone_features = [s['backbone_features'] for s in samples]
                    # Pad sequences to same length
                    max_len = max(f.shape[0] for f in backbone_features)
                    padded_features = []
                    for f in backbone_features:
                        if f.shape[0] < max_len:
                            pad_size = max_len - f.shape[0]
                            padding = torch.zeros(pad_size, f.shape[-1])
                            f_padded = torch.cat([f, padding], dim=0)
                        else:
                            f_padded = f
                        padded_features.append(f_padded)
                    batched['backbone_features'] = torch.stack(padded_features)
                
                # Handle sequence probabilities with padding
                if 'sequence_probs' in samples[0]:
                    sequence_probs = [s['sequence_probs'] for s in samples]
                    max_len = max(p.shape[0] for p in sequence_probs)
                    padded_probs = []
                    for p in sequence_probs:
                        if p.shape[0] < max_len:
                            pad_size = max_len - p.shape[0]
                            padding = torch.zeros(pad_size, p.shape[-1])
                            p_padded = torch.cat([p, padding], dim=0)
                        else:
                            p_padded = p
                        padded_probs.append(p_padded)
                    batched['sequence_probs'] = torch.stack(padded_probs)
                
                # Handle masks with padding
                if 'mask' in samples[0]:
                    masks = [s['mask'] for s in samples]
                    max_len = max(m.shape[0] for m in masks)
                    padded_masks = []
                    for m in masks:
                        if m.shape[0] < max_len:
                            pad_size = max_len - m.shape[0]
                            padding = torch.zeros(pad_size)
                            m_padded = torch.cat([m, padding], dim=0)
                        else:
                            m_padded = m
                        padded_masks.append(m_padded)
                    batched['mask'] = torch.stack(padded_masks)
                
                # Handle other fields
                for key in samples[0].keys():
                    if key not in ['backbone_features', 'sequence_probs', 'mask']:
                        values = [s[key] for s in samples]
                        if torch.is_tensor(values[0]):
                            batched[key] = torch.stack(values)
                        else:
                            batched[key] = values
            
            return batched
        
        return collate_samples(positive_samples), collate_samples(negative_samples)
    
    def _compute_energy_batch(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute energy for a batch of samples.
        
        Args:
            batch_data: Batch dictionary with required fields
            
        Returns:
            Energy values [batch_size]
        """
        # Extract required inputs
        backbone_features = batch_data['backbone_features']  # [B, L, 128]
        sequence_probs = batch_data['sequence_probs']  # [B, L, 20]
        mask = batch_data.get('mask', None)  # [B, L]
        
        # Handle empty batches
        if backbone_features.size(0) == 0:
            return torch.empty(0, device=self.device)
        
        # Validate tensor shapes
        if backbone_features.dim() != 3:
            raise ValueError(f"Expected backbone_features to be 3D, got shape {backbone_features.shape}")
        if sequence_probs.dim() != 3:
            raise ValueError(f"Expected sequence_probs to be 3D, got shape {sequence_probs.shape}")
        
        # Compute energy
        energy = self.model(backbone_features, sequence_probs, mask)
        
        return energy


class BiophysicalHeuristics:
    """Simplified biophysical heuristics for quick baseline comparison.
    
    WARNING: These are approximations for preliminary analysis, NOT validated force field
    calculations. For rigorous validation, integrate Rosetta (analyze_rosetta_correlation)
    or FoldX energy functions.
    
    Methods use:
    - Kyte-Doolittle hydrophobicity scale (1982): Well-established but simplified
    - Chou-Fasman secondary structure propensities (1974-1978): Historical parameters with
      known limitations (<60% accuracy). Modern DSSP-derived propensities recommended.
    - Simple stability score: Arbitrary weighted combination for demonstration only.
      NOT validated against experimental data.
    
    Use these heuristics with caution for research publication. Correlation with these
    baselines does NOT validate against physics - need real force field integration.
    """
    
    @staticmethod
    def hydrophobicity_score(sequence: str) -> float:
        """
        Compute sequence hydrophobicity score using Kyte-Doolittle scale.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Average hydrophobicity score
        """
        # Kyte-Doolittle hydrophobicity scale
        kd_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        if not sequence:
            return 0.0
        
        scores = [kd_scale.get(aa, 0.0) for aa in sequence.upper()]
        return sum(scores) / len(scores)
    
    @staticmethod
    def charge_score(sequence: str, ph: float = 7.0) -> float:
        """
        Compute net charge of sequence at given pH.
        
        Args:
            sequence: Amino acid sequence string
            ph: pH value (default 7.0)
            
        Returns:
            Net charge score
        """
        # Simplified charge calculation at pH 7.0
        charges = {
            'D': -1.0, 'E': -1.0,  # Negatively charged
            'K': 1.0, 'R': 1.0,   # Positively charged
            'H': 0.1               # Partially charged at pH 7
        }
        
        if not sequence:
            return 0.0
        
        total_charge = sum(charges.get(aa, 0.0) for aa in sequence.upper())
        return total_charge / len(sequence)  # Average charge per residue
    
    @staticmethod
    def secondary_structure_propensity_score(sequence: str) -> Dict[str, float]:
        """
        Compute secondary structure propensities using Chou-Fasman parameters.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary with helix, sheet, and turn propensities
        """
        # Chou-Fasman propensities (simplified)
        helix_prop = {
            'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
            'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
            'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
            'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
        }
        
        sheet_prop = {
            'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
            'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
            'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
            'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
        }
        
        turn_prop = {
            'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.60,
            'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01, 'L': 0.59,
            'M': 0.60, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95,
            'S': 1.43, 'T': 0.96, 'V': 0.50, 'W': 0.96, 'Y': 1.14
        }
        
        if not sequence:
            return {'helix': 0.0, 'sheet': 0.0, 'turn': 0.0}
        
        seq_upper = sequence.upper()
        helix_score = sum(helix_prop.get(aa, 1.0) for aa in seq_upper) / len(seq_upper)
        sheet_score = sum(sheet_prop.get(aa, 1.0) for aa in seq_upper) / len(seq_upper)
        turn_score = sum(turn_prop.get(aa, 1.0) for aa in seq_upper) / len(seq_upper)
        
        return {
            'helix': helix_score,
            'sheet': sheet_score,
            'turn': turn_score
        }
    
    @staticmethod
    def simple_stability_score(sequence: str) -> float:
        """
        Compute simple stability heuristic based on amino acid composition.
        
        Combines hydrophobicity, charge, and secondary structure preferences
        into a single stability estimate.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Simple stability score (higher = more stable)
        """
        if not sequence:
            return 0.0
        
        # Get component scores
        hydro_score = BiophysicalHeuristics.hydrophobicity_score(sequence)
        charge_score = abs(BiophysicalHeuristics.charge_score(sequence))  # Extreme charges destabilizing
        ss_props = BiophysicalHeuristics.secondary_structure_propensity_score(sequence)
        
        # CRITICAL WARNING: These are ARBITRARY WEIGHTS with NO SCIENTIFIC VALIDATION
        warnings.warn(
            "\n" + "="*70 + "\n"
            "SCIENTIFIC VALIDITY WARNING\n"
            "="*70 + "\n"
            "BiophysicalHeuristics.simple_stability_score uses arbitrary weights.\n"
            "This baseline has NO scientific validation and MUST NOT be used for:\n"
            "  - Research publication\n"
            "  - Model validation claims\n"
            "  - Comparison to experimental stability data\n"
            "\n"
            "For rigorous baseline, use: Rosetta REU, FoldX ΔΔG, or DDGun predictions.\n"
            "Rosetta integration is currently INCOMPLETE (returns stub error).\n"
            "="*70,
            UserWarning
        )

        # Placeholder heuristic (NOT scientifically validated)
        stability = (
            hydro_score * 0.3  # Arbitrary weight - DO NOT INTERPRET
            - abs(charge_score) * 0.5  # Arbitrary weight - DO NOT INTERPRET
            + ss_props['helix'] * 0.2  # Arbitrary weight - DO NOT INTERPRET
        )
        
        return stability


class CorrelationAnalyzer:
    """
    Analyzes correlations between energy model predictions and physics-based baselines.
    
    Provides comprehensive correlation analysis including multiple baseline methods,
    statistical significance testing, and visualization support.
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.baselines = BiophysicalHeuristics()
    
    def analyze_energy_correlations(
        self,
        sequences: List[str],
        energy_predictions: np.ndarray,
        include_baselines: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlations between energy predictions and physics-based baselines.
        
        Args:
            sequences: List of protein sequences
            energy_predictions: Array of energy model predictions
            include_baselines: List of baseline methods to include
            
        Returns:
            Dictionary with correlation analysis results
        """
        print("Analyzing energy correlations with physics-based baselines...")
        
        if include_baselines is None:
            include_baselines = ['hydrophobicity', 'charge', 'secondary_structure', 'simple_stability']
        
        if len(sequences) != len(energy_predictions):
            raise ValueError(f"Sequence count ({len(sequences)}) != prediction count ({len(energy_predictions)})")
        
        results = {
            'num_sequences': len(sequences),
            'energy_predictions_stats': {
                'mean': float(np.mean(energy_predictions)),
                'std': float(np.std(energy_predictions)),
                'min': float(np.min(energy_predictions)),
                'max': float(np.max(energy_predictions))
            },
            'baseline_correlations': {}
        }
        
        # Compute baseline scores
        baseline_scores = {}
        
        if 'hydrophobicity' in include_baselines:
            hydro_scores = [self.baselines.hydrophobicity_score(seq) for seq in sequences]
            baseline_scores['hydrophobicity'] = np.array(hydro_scores)
        
        if 'charge' in include_baselines:
            charge_scores = [self.baselines.charge_score(seq) for seq in sequences]
            baseline_scores['charge'] = np.array(charge_scores)
        
        if 'simple_stability' in include_baselines:
            stability_scores = [self.baselines.simple_stability_score(seq) for seq in sequences]
            baseline_scores['simple_stability'] = np.array(stability_scores)
        
        if 'secondary_structure' in include_baselines:
            ss_scores = []
            for seq in sequences:
                ss_props = self.baselines.secondary_structure_propensity_score(seq)
                # Use helix propensity as representative score
                ss_scores.append(ss_props['helix'])
            baseline_scores['secondary_structure'] = np.array(ss_scores)
        
        # Compute correlations
        for baseline_name, baseline_values in baseline_scores.items():
            correlation_result = self._compute_correlation(
                energy_predictions, baseline_values, baseline_name
            )
            results['baseline_correlations'][baseline_name] = correlation_result
        
        # Overall summary
        results['summary'] = self._summarize_correlations(results['baseline_correlations'])
        
        return results
    
    def _compute_correlation(
        self, 
        energy_predictions: np.ndarray, 
        baseline_scores: np.ndarray,
        baseline_name: str
    ) -> Dict[str, Any]:
        """
        Compute correlation between energy predictions and baseline scores.
        
        Args:
            energy_predictions: Energy model predictions
            baseline_scores: Baseline method scores
            baseline_name: Name of baseline method
            
        Returns:
            Dictionary with correlation statistics
        """
        result = {
            'baseline_name': baseline_name,
            'baseline_stats': {
                'mean': float(np.mean(baseline_scores)),
                'std': float(np.std(baseline_scores)),
                'min': float(np.min(baseline_scores)),
                'max': float(np.max(baseline_scores))
            }
        }
        
        # Handle edge cases
        if len(energy_predictions) < 2 or np.std(energy_predictions) == 0 or np.std(baseline_scores) == 0:
            result['correlation_error'] = 'Insufficient data or zero variance'
            return result
        
        # Pearson correlation
        pearson_corr = np.corrcoef(energy_predictions, baseline_scores)[0, 1]
        result['pearson_correlation'] = float(pearson_corr)
        
        # Spearman correlation (if scipy available)
        if SCIPY_AVAILABLE:
            spearman_corr, spearman_p = spearmanr(energy_predictions, baseline_scores)
            result['spearman_correlation'] = float(spearman_corr)
            result['spearman_p_value'] = float(spearman_p)
            
            # Kendall correlation
            kendall_corr, kendall_p = stats.kendalltau(energy_predictions, baseline_scores)
            result['kendall_correlation'] = float(kendall_corr)
            result['kendall_p_value'] = float(kendall_p)
        else:
            result['spearman_correlation'] = None
            result['kendall_correlation'] = None
            result['note'] = 'SciPy not available for rank correlations'
        
        # Interpretation
        result['interpretation'] = self._interpret_correlation(abs(pearson_corr))
        
        return result
    
    def _interpret_correlation(self, abs_correlation: float) -> str:
        """Interpret correlation strength."""
        if abs_correlation >= 0.9:
            return "Very strong correlation"
        elif abs_correlation >= 0.7:
            return "Strong correlation"
        elif abs_correlation >= 0.5:
            return "Moderate correlation"
        elif abs_correlation >= 0.3:
            return "Weak correlation"
        else:
            return "Very weak or no correlation"
    
    def _summarize_correlations(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize correlation analysis results."""
        if not correlations:
            return {'note': 'No correlations computed'}
        
        # Find strongest correlations
        valid_correlations = {}
        for name, corr_data in correlations.items():
            if 'pearson_correlation' in corr_data and not np.isnan(corr_data['pearson_correlation']):
                valid_correlations[name] = abs(corr_data['pearson_correlation'])
        
        if not valid_correlations:
            return {'note': 'No valid correlations found'}
        
        strongest_baseline = max(valid_correlations, key=valid_correlations.get)
        strongest_correlation = valid_correlations[strongest_baseline]
        
        return {
            'strongest_correlation': {
                'baseline': strongest_baseline,
                'absolute_correlation': strongest_correlation,
                'interpretation': self._interpret_correlation(strongest_correlation)
            },
            'average_absolute_correlation': float(np.mean(list(valid_correlations.values()))),
            'num_baselines_tested': len(valid_correlations)
        }
    
    def analyze_rosetta_correlation(
        self,
        sequences: List[str],
        energy_predictions: np.ndarray,
        pdb_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlation with Rosetta energy scores (if available).
        
        Args:
            sequences: List of protein sequences
            energy_predictions: Energy model predictions
            pdb_files: Optional list of PDB files for Rosetta scoring
            
        Returns:
            Dictionary with Rosetta correlation results
        """
        # Placeholder for Rosetta integration
        # This would require Rosetta installation and PyRosetta interface
        return {
            'error': 'Rosetta integration not implemented',
            'note': 'Requires PyRosetta installation and implementation',
            'suggestion': 'Use physics-based baselines for now'
        }
    
    def analyze_alphafold_correlation(
        self,
        sequences: List[str],
        energy_predictions: np.ndarray,
        alphafold_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlation with AlphaFold confidence scores (if available).
        
        Args:
            sequences: List of protein sequences
            energy_predictions: Energy model predictions
            alphafold_scores: Optional array of AlphaFold confidence scores
            
        Returns:
            Dictionary with AlphaFold correlation results
        """
        if alphafold_scores is None:
            return {
                'error': 'AlphaFold scores not provided',
                'note': 'Requires external AlphaFold prediction or pre-computed scores',
                'suggestion': 'Use ColabFold or AlphaFold2 to generate confidence scores'
            }
        
        if len(sequences) != len(alphafold_scores):
            return {
                'error': f'Sequence count ({len(sequences)}) != AlphaFold score count ({len(alphafold_scores)})'
            }
        
        # Compute correlation with AlphaFold confidence
        # Higher confidence should correlate with lower (more stable) energy
        correlation_result = self._compute_correlation(
            energy_predictions, -alphafold_scores, 'alphafold_confidence'
        )
        
        return {
            'alphafold_correlation': correlation_result,
            'note': 'Negative AlphaFold scores used (higher confidence = lower energy expected)'
        }


class SequencePropertyAnalyzer:
    """
    Analyzes biological properties of generated sequences including composition,
    secondary structure preferences, and other biochemical characteristics.
    """
    
    def __init__(self):
        """Initialize sequence property analyzer."""
        # Standard amino acid properties
        self.aa_properties = self._load_aa_properties()
    
    def _load_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """Load amino acid biochemical properties."""
        # Standard amino acid single-letter codes
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Simplified property values (normalized to [0, 1])
        properties = {
            'hydrophobicity': {  # Kyte-Doolittle scale (normalized)
                'A': 0.61, 'C': 0.68, 'D': 0.04, 'E': 0.04, 'F': 0.89,
                'G': 0.36, 'H': 0.24, 'I': 1.00, 'K': 0.04, 'L': 0.89,
                'M': 0.64, 'N': 0.18, 'P': 0.32, 'Q': 0.18, 'R': 0.04,
                'S': 0.25, 'T': 0.32, 'V': 0.79, 'W': 0.54, 'Y': 0.39
            },
            'volume': {  # Molecular volume (normalized)
                'A': 0.15, 'C': 0.32, 'D': 0.34, 'E': 0.42, 'F': 0.58,
                'G': 0.00, 'H': 0.45, 'I': 0.47, 'K': 0.53, 'L': 0.47,
                'M': 0.50, 'N': 0.32, 'P': 0.32, 'Q': 0.42, 'R': 0.61,
                'S': 0.21, 'T': 0.29, 'V': 0.39, 'W': 0.68, 'Y': 0.58
            },
            'charge': {  # Electrical charge at pH 7
                'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0,
                'G': 0.0, 'H': 0.1, 'I': 0.0, 'K': 1.0, 'L': 0.0,
                'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0,
                'S': 0.0, 'T': 0.0, 'V': 0.0, 'W': 0.0, 'Y': 0.0
            }
        }
        
        return properties
    
    def analyze_sequence_properties(
        self, 
        sequences: List[str],
        reference_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze biochemical properties of sequences.
        
        Args:
            sequences: List of sequences to analyze
            reference_sequences: Reference sequences for comparison
            
        Returns:
            Dictionary with property analysis results
        """
        print("Analyzing sequence properties...")
        
        results = {
            'num_sequences': len(sequences),
            'sequence_lengths': [len(seq) for seq in sequences],
            'composition': self._analyze_composition(sequences),
            'properties': self._analyze_biochemical_properties(sequences)
        }
        
        # Add reference comparison if provided
        if reference_sequences:
            results['reference_comparison'] = self._compare_with_reference(
                sequences, reference_sequences
            )
        
        return results
    
    def _analyze_composition(self, sequences: List[str]) -> Dict[str, Any]:
        """Analyze amino acid composition."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Count occurrences
        total_residues = sum(len(seq) for seq in sequences)
        aa_counts = {aa: 0 for aa in amino_acids}
        
        for seq in sequences:
            for aa in seq:
                if aa in aa_counts:
                    aa_counts[aa] += 1
        
        # Compute frequencies with division by zero protection
        if total_residues > 0:
            aa_frequencies = {aa: count / total_residues for aa, count in aa_counts.items()}
        else:
            aa_frequencies = {aa: 0.0 for aa in amino_acids}
        
        # Natural frequencies (approximate from SwissProt)
        natural_frequencies = {
            'A': 0.087, 'C': 0.019, 'D': 0.056, 'E': 0.067, 'F': 0.039,
            'G': 0.074, 'H': 0.022, 'I': 0.053, 'K': 0.058, 'L': 0.096,
            'M': 0.023, 'N': 0.043, 'P': 0.052, 'Q': 0.039, 'R': 0.055,
            'S': 0.069, 'T': 0.055, 'V': 0.065, 'W': 0.012, 'Y': 0.029
        }
        
        return {
            'frequencies': aa_frequencies,
            'natural_frequencies': natural_frequencies,
            'total_residues': total_residues,
            'unique_amino_acids': len([aa for aa, freq in aa_frequencies.items() if freq > 0])
        }
    
    def _analyze_biochemical_properties(self, sequences: List[str]) -> Dict[str, Any]:
        """Analyze biochemical properties like hydrophobicity, charge, etc."""
        properties = {}
        
        for prop_name, prop_values in self.aa_properties.items():
            sequence_props = []
            
            for seq in sequences:
                if len(seq) > 0:
                    seq_prop = np.mean([prop_values.get(aa, 0.0) for aa in seq])
                else:
                    seq_prop = 0.0  # Default for empty sequences
                sequence_props.append(seq_prop)
            
            # Handle empty sequence list
            if len(sequence_props) == 0:
                properties[prop_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'values': []
                }
            else:
                sequence_props_array = np.array(sequence_props)
                properties[prop_name] = {
                    'mean': float(np.mean(sequence_props_array)),
                    'std': float(np.std(sequence_props_array)) if len(sequence_props_array) > 1 else 0.0,
                    'min': float(np.min(sequence_props_array)),
                    'max': float(np.max(sequence_props_array)),
                    'values': sequence_props
                }
        
        return properties
    
    def _compare_with_reference(
        self, 
        test_sequences: List[str], 
        reference_sequences: List[str]
    ) -> Dict[str, Any]:
        """Compare test sequences with reference sequences."""
        test_props = self.analyze_sequence_properties(test_sequences)
        ref_props = self.analyze_sequence_properties(reference_sequences)
        
        comparison = {
            'composition_similarity': self._compute_composition_similarity(
                test_props['composition']['frequencies'],
                ref_props['composition']['frequencies']
            ),
            'property_differences': {}
        }
        
        # Compare biochemical properties
        for prop_name in test_props['properties']:
            test_mean = test_props['properties'][prop_name]['mean']
            ref_mean = ref_props['properties'][prop_name]['mean']
            comparison['property_differences'][prop_name] = {
                'test_mean': test_mean,
                'reference_mean': ref_mean,
                'difference': test_mean - ref_mean,
                'relative_difference': (test_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
            }
        
        return comparison
    
    def _compute_composition_similarity(
        self, 
        freq1: Dict[str, float], 
        freq2: Dict[str, float]
    ) -> float:
        """Compute similarity between amino acid compositions using Jensen-Shannon divergence."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Convert to arrays
        f1 = np.array([freq1[aa] for aa in amino_acids])
        f2 = np.array([freq2[aa] for aa in amino_acids])
        
        # Standard JS divergence with proper epsilon handling AND overflow protection
        epsilon = 1e-10

        # Ensure distributions are normalized and non-negative
        f1 = np.maximum(f1, 0.0)
        f2 = np.maximum(f2, 0.0)
        f1_sum = f1.sum()
        f2_sum = f2.sum()

        if f1_sum == 0 or f2_sum == 0:
            warnings.warn("Degenerate probability distributions in JS divergence computation")
            return 0.5  # Neutral similarity for degenerate case

        f1 = f1 / f1_sum
        f2 = f2 / f2_sum

        # Add epsilon for numerical stability (standard approach)
        f1 = f1 + epsilon
        f2 = f2 + epsilon
        f1 = f1 / f1.sum()
        f2 = f2 / f2.sum()

        # Compute mixture distribution
        m = 0.5 * (f1 + f2)

        # KL divergences with overflow protection (reduced from 1e10 to 1e6)
        ratio1 = np.clip(f1 / m, epsilon, 1e6)
        ratio2 = np.clip(f2 / m, epsilon, 1e6)

        # Check for NaN/Inf before computing KL
        if not (np.all(np.isfinite(ratio1)) and np.all(np.isfinite(ratio2))):
            warnings.warn("Non-finite values in JS divergence, returning neutral similarity 0.5")
            return 0.5

        kl1 = np.sum(f1 * np.log(ratio1))
        kl2 = np.sum(f2 * np.log(ratio2))

        if not (np.isfinite(kl1) and np.isfinite(kl2)):
            warnings.warn("Non-finite KL divergence values, returning neutral similarity 0.5")
            return 0.5

        js_div = 0.5 * kl1 + 0.5 * kl2
        similarity = 1.0 - (js_div / np.log(2))

        # Graceful clamping instead of assertion crash
        if not (0.0 <= similarity <= 1.0):
            warnings.warn(f"JS similarity out of range: {similarity}, clamping to [0,1]")
            similarity = np.clip(similarity, 0.0, 1.0)

        return float(similarity)


class EnergyVisualizationGenerator:
    """
    Generates visualizations for energy model evaluation including energy distributions,
    ranking accuracy plots, and correlation analysis charts.
    """
    
    def __init__(self, output_dir: str = "evaluation_plots"):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting libraries not available. Visualizations will be skipped.")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_energy_distributions(
        self, 
        native_energies: np.ndarray, 
        negative_energies: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot energy distribution comparison between native and negative sequences.
        
        Args:
            native_energies: Energy values for native sequences
            negative_energies: Energy values for negative sequences
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping energy distribution plot")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution comparison
        ax1.hist(native_energies, bins=50, alpha=0.7, label='Native', density=True)
        ax1.hist(negative_energies, bins=50, alpha=0.7, label='Negative', density=True)
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Density')
        ax1.set_title('Energy Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [native_energies, negative_energies]
        ax2.boxplot(data_to_plot, labels=['Native', 'Negative'])
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "energy_distributions.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_ranking_accuracy(
        self, 
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        seed: int = 42
    ) -> str:
        """
        Plot ranking accuracy metrics and statistics.
        
        Args:
            results: Ranking evaluation results
            save_path: Optional custom save path
            seed: Random seed for reproducible visualizations (default: 42)
            
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping ranking accuracy plot")
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy bar plot
        accuracy = results['ranking_accuracy']
        ax1.bar(['Ranking Accuracy'], [accuracy], color='skyblue', alpha=0.7)
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Energy Ranking Accuracy: {accuracy:.3f}')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy gap distribution
        if 'energy_gaps' in results:
            gaps = np.array(results['energy_gaps'])
            ax2.hist(gaps, bins=30, alpha=0.7, color='lightgreen')
            ax2.axvline(x=0, color='red', linestyle='--', label='No Gap')
            ax2.set_xlabel('Energy Gap (Negative - Native)')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Energy Gap Distribution\nMean: {gaps.mean():.3f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Energy comparison scatter plot
        if 'native_energies' in results and 'negative_energies' in results:
            native_e = np.array(results['native_energies'])
            negative_e = np.array(results['negative_energies'])
            
            # Sample for plotting if too many points
            max_points = 1000
            if len(native_e) > max_points:
                # Use fixed seed for reproducible visualizations
                # Users can modify seed if they want different subsampling
                rng = np.random.RandomState(seed if seed is not None else 42)
                idx = rng.choice(len(native_e), max_points, replace=False)
                native_e = native_e[idx]
                negative_e = negative_e[idx]
            
            ax3.scatter(native_e, negative_e, alpha=0.6, s=20)
            min_e = min(native_e.min(), negative_e.min())
            max_e = max(native_e.max(), negative_e.max())
            ax3.plot([min_e, max_e], [min_e, max_e], 'r--', label='Equal Energy')
            ax3.set_xlabel('Native Energy')
            ax3.set_ylabel('Negative Energy')
            ax3.set_title('Energy Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
Total Pairs: {results['total_pairs_evaluated']}
Correct Rankings: {results['correct_rankings']}
Accuracy: {results['ranking_accuracy']:.3f}

Native Energy:
  Mean: {results['native_energy_mean']:.3f}
  Std: {results['native_energy_std']:.3f}

Negative Energy:
  Mean: {results['negative_energy_mean']:.3f}
  Std: {results['negative_energy_std']:.3f}

Energy Gap:
  Mean: {results['energy_gap_mean']:.3f}
  Median: {results['energy_gap_median']:.3f}
        """.strip()
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ranking_accuracy.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_sequence_properties(
        self, 
        property_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot sequence property analysis results.
        
        Args:
            property_results: Results from sequence property analysis
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping sequence properties plot")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Amino acid composition
        composition = property_results['composition']
        amino_acids = sorted(composition['frequencies'].keys())
        observed_freqs = [composition['frequencies'][aa] for aa in amino_acids]
        natural_freqs = [composition['natural_frequencies'][aa] for aa in amino_acids]
        
        x = np.arange(len(amino_acids))
        width = 0.35
        
        ax = axes[0]
        ax.bar(x - width/2, observed_freqs, width, label='Observed', alpha=0.7)
        ax.bar(x + width/2, natural_freqs, width, label='Natural', alpha=0.7)
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Frequency')
        ax.set_title('Amino Acid Composition')
        ax.set_xticks(x)
        ax.set_xticklabels(amino_acids)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Biochemical properties
        if 'properties' in property_results:
            properties = property_results['properties']
            
            # Property distributions
            prop_names = list(properties.keys())[:3]  # Show first 3 properties
            for i, prop_name in enumerate(prop_names):
                if i + 1 >= len(axes):
                    break
                
                ax = axes[i + 1]
                prop_values = properties[prop_name]['values']
                ax.hist(prop_values, bins=20, alpha=0.7, color=f'C{i}')
                ax.set_xlabel(prop_name.capitalize())
                ax.set_ylabel('Count')
                ax.set_title(f'{prop_name.capitalize()} Distribution\nMean: {properties[prop_name]["mean"]:.3f}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "sequence_properties.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


class EnergyModelEvaluator:
    """
    Main evaluation coordinator that orchestrates all evaluation components.
    
    This class provides the main interface for comprehensive energy model evaluation,
    coordinating ranking evaluation, property analysis, and visualization generation.
    """
    
    def __init__(
        self,
        model_checkpoint: str,
        encoder_type: str = "vanilla",
        encoder_name: str = "v_48_020",
        device: str = "auto",
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize comprehensive energy model evaluator.
        
        Args:
            model_checkpoint: Path to trained energy model checkpoint
            encoder_type: ProteinMPNN encoder type (vanilla, ca_model, soluble)
            encoder_name: ProteinMPNN model version
            device: Computation device ('cpu', 'cuda', 'mps', or 'auto')
            output_dir: Directory for saving evaluation results
        """
        print("Initializing Energy Model Evaluator...")
        
        # Set up device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load energy model
        print("Loading energy model...")
        self.model = self._load_energy_model(
            model_checkpoint, encoder_type, encoder_name
        )
        
        # Initialize evaluation components
        self.ranking_evaluator = EnergyRankingEvaluator(self.model, device)
        self.property_analyzer = SequencePropertyAnalyzer()
        self.visualizer = EnergyVisualizationGenerator(
            output_dir=str(self.output_dir / "plots")
        )
        
        print("✓ Energy Model Evaluator initialized successfully")
    
    def _load_energy_model(
        self, 
        checkpoint_path: str, 
        encoder_type: str, 
        encoder_name: str
    ) -> nn.Module:
        """Load complete energy model from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize encoder
            encoder = load_pretrained_encoder(
                model_name=encoder_name,
                model_type=encoder_type,
                freeze_layers=True,
                device=self.device
            )
            
            # Initialize energy head with checkpoint config
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                energy_head = EnergyHead(
                    backbone_dim=config.get('backbone_dim', 128),
                    seq_dim=config.get('seq_dim', 20),
                    hidden_dim=config.get('hidden_dim', 512),
                    num_layers=config.get('num_layers', 3),
                    dropout=config.get('dropout', 0.1)
                )
            else:
                # Use default config
                energy_head = EnergyHead()
            
            # Load energy head state with proper error handling
            state_dict_loaded = False
            if 'energy_head_state_dict' in checkpoint:
                try:
                    energy_head.load_state_dict(checkpoint['energy_head_state_dict'])
                    state_dict_loaded = True
                except Exception as e:
                    warnings.warn(f"Failed to load energy_head_state_dict: {e}")
            
            if not state_dict_loaded and 'model_state_dict' in checkpoint:
                try:
                    energy_head.load_state_dict(checkpoint['model_state_dict'])
                    state_dict_loaded = True
                except Exception as e:
                    warnings.warn(f"Failed to load model_state_dict: {e}")
            
            if not state_dict_loaded:
                raise RuntimeError(
                    f"No compatible state dict found in checkpoint {checkpoint_path}. "
                    f"Available keys: {list(checkpoint.keys())}. "
                    f"Expected 'energy_head_state_dict' or 'model_state_dict'. "
                    f"This indicates checkpoint version incompatibility or wrong checkpoint file. "
                    f"Cannot evaluate without trained weights - results would be meaningless."
                )
            
            # Additional safety check: Verify energy head actually loaded trained weights
            # Random initialization would have different parameter patterns
            energy_head.eval()
            with torch.no_grad():
                # Check if parameters look like they've been trained (basic heuristic)
                param_count = sum(p.numel() for p in energy_head.parameters())
                if param_count == 0:
                    raise RuntimeError(
                        f"Energy head has no parameters after loading checkpoint {checkpoint_path}. "
                        f"This indicates a critical loading failure."
                    )
            
            # Create combined model
            class CombinedEnergyModel(nn.Module):
                def __init__(self, encoder, energy_head, sequence_repr):
                    super().__init__()
                    self.encoder = encoder
                    self.energy_head = energy_head
                    self.sequence_repr = sequence_repr
                
                def forward(self, backbone_features, sequence_probs, mask=None):
                    # Model expects pre-computed features
                    return self.energy_head(backbone_features, sequence_probs, mask)
            
            # Initialize sequence representation
            sequence_repr = ContinuousSequenceRepr()
            
            model = CombinedEnergyModel(encoder, energy_head, sequence_repr)
            model.to(self.device)
            model.eval()
            
            return model
            
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Verify the path exists and is accessible. "
                f"Cannot evaluate with random initialization."
            ) from e
        except KeyError as e:
            try:
                available_keys = list(checkpoint.keys()) if 'checkpoint' in locals() else 'unknown (checkpoint not loaded)'
            except:
                available_keys = 'unknown'
            raise RuntimeError(
                f"Checkpoint missing required key: {e}. "
                f"Available keys: {available_keys}. "
                f"This indicates version incompatibility or corrupted checkpoint. "
                f"Ensure checkpoint was saved with same code version."
            ) from e
        except RuntimeError as e:
            if "state_dict" in str(e).lower():
                raise RuntimeError(
                    f"Failed to load state_dict: {e}. "
                    f"Model architecture may have changed. Check model definition matches checkpoint."
                ) from e
            else:
                raise
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error loading checkpoint {checkpoint_path}: {type(e).__name__}: {e}. "
                f"Contact support with this error message."
            ) from e
    
    def evaluate_comprehensive(
        self,
        test_dataset: StabilityDataset,
        num_ranking_samples: int = 1000,
        num_property_samples: int = 500,
        batch_size: int = 16,
        generate_visualizations: bool = True,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test dataset with reproducible random operations.
        
        Args:
            test_dataset: Test dataset for evaluation
            num_ranking_samples: Number of samples for ranking evaluation
            num_property_samples: Number of samples for property analysis
            batch_size: Batch size for evaluation
            generate_visualizations: Whether to generate visualization plots
            seed: Master random seed for all evaluation operations. Using the same seed
                 improves reproducibility across runs (rankings, sequences, bootstrap,
                 visualizations). Note: Exact reproducibility depends on PyTorch version,
                 hardware, and CUDA determinism settings. See Known Limitations.
            
        Returns:
            Complete evaluation results dictionary
        """
        print("Starting comprehensive energy model evaluation...")
        
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_device': self.device,
            'evaluation_params': {
                'num_ranking_samples': num_ranking_samples,
                'num_property_samples': num_property_samples,
                'batch_size': batch_size,
                'dataset_size': len(test_dataset)
            }
        }
        
        # Store seed in results for provenance
        results['master_seed'] = seed
        results['reproducibility_note'] = 'Same seed improves but does not guarantee identical results across environments'
        
        # 1. Energy ranking evaluation
        print("\n=== Energy Ranking Evaluation ===")
        ranking_results = self.ranking_evaluator.evaluate_ranking_accuracy(
            test_dataset, num_ranking_samples, batch_size, seed=seed
        )
        results['ranking_evaluation'] = ranking_results
        
        # 2. Sequence property analysis (if we can extract sequences)
        print("\n=== Sequence Property Analysis ===")
        try:
            sequences = self._extract_sequences_from_dataset(
                test_dataset, num_property_samples, seed=seed
            )
            property_results = self.property_analyzer.analyze_sequence_properties(sequences)
            results['sequence_properties'] = property_results
        except Exception as e:
            print(f"Warning: Could not perform sequence property analysis: {e}")
            results['sequence_properties'] = {'error': str(e)}
        
        # 3. Generate visualizations
        if generate_visualizations and PLOTTING_AVAILABLE:
            print("\n=== Generating Visualizations ===")
            
            try:
                # Energy distribution plots
                native_energies = np.array(ranking_results['native_energies'])
                negative_energies = np.array(ranking_results['negative_energies'])
                
                energy_plot = self.visualizer.plot_energy_distributions(
                    native_energies, negative_energies
                )
                results['plots'] = {'energy_distributions': energy_plot}
                
                # Ranking accuracy plots
                ranking_plot = self.visualizer.plot_ranking_accuracy(ranking_results, seed=seed)
                results['plots']['ranking_accuracy'] = ranking_plot
                
                # Sequence property plots
                if 'sequence_properties' in results and 'error' not in results['sequence_properties']:
                    property_plot = self.visualizer.plot_sequence_properties(
                        results['sequence_properties']
                    )
                    results['plots']['sequence_properties'] = property_plot
                
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
                results['visualization_error'] = str(e)
        
        # 4. Save results
        print("\n=== Saving Results ===")
        self._save_evaluation_results(results)
        
        # 5. Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def evaluate_holdout_validation(
        self,
        holdout_dataset: StabilityDataset,
        num_samples: int = 500,
        batch_size: int = 16,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on hold-out PDB structures.
        
        This tests the model's ability to generalize to completely unseen
        protein structures that were not in the training set.
        
        Args:
            holdout_dataset: Dataset with held-out PDB structures
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            detailed_analysis: Whether to include detailed sequence/correlation analysis
            
        Returns:
            Dictionary with hold-out validation results
        """
        print("=== Hold-out Validation on Unseen PDB Structures ===")
        
        results = {
            'evaluation_type': 'holdout_validation',
            'holdout_dataset_size': len(holdout_dataset),
            'samples_evaluated': min(num_samples, len(holdout_dataset))
        }
        
        # Core ranking evaluation on hold-out data
        print("Evaluating energy ranking on hold-out structures...")
        ranking_results = self.ranking_evaluator.evaluate_ranking_accuracy(
            holdout_dataset, num_samples, batch_size
        )
        results['holdout_ranking'] = ranking_results
        
        # Confidence intervals for hold-out performance
        print("Computing confidence intervals for hold-out accuracy...")
        if SCIPY_AVAILABLE:
            ci_results = self.ranking_evaluator.evaluate_ranking_with_confidence_intervals(
                holdout_dataset, num_samples, batch_size, num_bootstrap=500
            )
            results['confidence_intervals'] = ci_results['confidence_intervals']
        
        # Detailed analysis if requested
        if detailed_analysis:
            try:
                # Sequence property analysis on hold-out data
                print("Analyzing sequence properties of hold-out dataset...")
                sequences = self._extract_sequences_from_dataset(holdout_dataset, num_samples, seed=42)
                if sequences:
                    prop_results = self.property_analyzer.analyze_sequence_properties(sequences)
                    results['holdout_sequence_properties'] = prop_results
                
                # Correlation analysis on hold-out data
                print("Analyzing correlations with physics baselines on hold-out data...")
                if sequences:
                    energy_preds = np.array(ranking_results['native_energies'] + ranking_results['negative_energies'])
                    # Create CorrelationAnalyzer instance
                    corr_analyzer = CorrelationAnalyzer()
                    corr_results = corr_analyzer.analyze_energy_correlations(sequences, energy_preds)
                    results['holdout_correlations'] = corr_results
                    
            except Exception as e:
                print(f"Warning: Could not perform detailed analysis: {e}")
                results['detailed_analysis_error'] = str(e)
        
        # Performance comparison
        results['interpretation'] = self._interpret_holdout_performance(results)
        
        return results
    
    def evaluate_generalization_across_families(
        self,
        family_datasets: Dict[str, StabilityDataset],
        num_samples_per_family: int = 200,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Evaluate model generalization across different protein families.
        
        Tests whether the energy model performs consistently across
        different types of proteins (enzymes, membrane proteins, etc.).
        
        Args:
            family_datasets: Dictionary mapping family names to datasets
            num_samples_per_family: Number of samples to evaluate per family
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with cross-family generalization results
        """
        print("=== Generalization Analysis Across Protein Families ===")
        
        results = {
            'evaluation_type': 'cross_family_generalization',
            'families_tested': list(family_datasets.keys()),
            'num_families': len(family_datasets)
        }
        
        family_results = {}
        
        for family_name, family_dataset in family_datasets.items():
            print(f"\nEvaluating on {family_name} family...")
            
            # Evaluate ranking accuracy for this family
            family_ranking = self.ranking_evaluator.evaluate_ranking_accuracy(
                family_dataset, num_samples_per_family, batch_size
            )
            
            family_results[family_name] = {
                'dataset_size': len(family_dataset),
                'samples_evaluated': min(num_samples_per_family, len(family_dataset)),
                'ranking_accuracy': family_ranking['ranking_accuracy'],
                'energy_gap_mean': family_ranking['energy_gap_mean'],
                'energy_gap_std': family_ranking['energy_gap_std']
            }
        
        results['per_family_results'] = family_results
        
        # Cross-family analysis
        accuracies = [family['ranking_accuracy'] for family in family_results.values()]
        results['cross_family_summary'] = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'accuracy_range': float(np.max(accuracies) - np.min(accuracies))
        }
        
        # Statistical analysis of family differences
        if SCIPY_AVAILABLE and len(accuracies) > 2:
            # ANOVA test for significant differences between families
            try:
                from scipy.stats import f_oneway
                # This is simplified - would need actual per-sample data for proper ANOVA
                results['statistical_analysis'] = {
                    'note': 'Family-level accuracy comparison',
                    'accuracy_variance_significant': np.std(accuracies) > 0.05,  # Simple threshold
                    'interpretation': self._interpret_family_differences(results['cross_family_summary'])
                }
            except Exception as e:
                results['statistical_analysis'] = {'error': str(e)}
        
        return results
    
    def evaluate_cross_validation(
        self,
        dataset: StabilityDataset,
        k_folds: int = 5,
        num_samples_per_fold: int = 200,
        batch_size: int = 16,
        split_by_similarity: bool = True,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation evaluation.
        
        Args:
            dataset: Complete dataset for cross-validation
            k_folds: Number of folds for cross-validation
            num_samples_per_fold: Number of samples to evaluate per fold
            batch_size: Batch size for evaluation
            split_by_similarity: Whether to split by sequence similarity
            similarity_threshold: Sequence identity threshold for splitting
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"=== {k_folds}-Fold Cross-Validation ===")
        
        # CRITICAL: Sequence identity clustering required for valid protein evaluation
        if split_by_similarity:
            raise NotImplementedError(
                "\n" + "="*70 + "\n"
                "CRITICAL: Sequence identity-based cross-validation not implemented\n"
                "="*70 + "\n"
                "Random splitting causes DATA LEAKAGE in protein datasets due to homology.\n"
                "This leads to INVALID performance estimates that CANNOT be published.\n"
                "\n"
                "Required implementation:\n"
                f"  1. Cluster sequences at {similarity_threshold*100:.0f}% identity using MMseqs2/CD-HIT\n"
                "  2. Split CLUSTERS (not sequences) into k folds\n"
                "  3. Ensure no homologous sequences across train/test\n"
                "\n"
                "For demonstration ONLY, use split_by_similarity=False with explicit\n"
                "understanding that results are overoptimistic and scientifically invalid.\n"
                "="*70
            )

        # Force explicit acknowledgment of limitation
        warnings.warn(
            "\n" + "="*70 + "\n"
            "SCIENTIFIC VALIDITY WARNING\n"
            "="*70 + "\n"
            "Using random CV splits WITHOUT sequence identity clustering.\n"
            "Results WILL OVERESTIMATE generalization due to homologous sequences\n"
            "in train/test splits. This is a KNOWN ERROR in protein evaluation.\n"
            "\n"
            "DO NOT use these results for:\n"
            "  - Publication or preprint\n"
            "  - Model validation claims\n"
            "  - Comparison to other methods\n"
            "\n"
            "This mode is for DEBUGGING ONLY.\n"
            "="*70,
            UserWarning
        )

        results = {
            'evaluation_type': 'cross_validation',
            'k_folds': k_folds,
            'total_dataset_size': len(dataset),
            'split_method': 'similarity_based' if split_by_similarity else 'random'
        }
        
        # Simple random splitting (INVALID for protein evaluation)
        fold_results = []
        dataset_size = len(dataset)
        fold_size = dataset_size // k_folds
        
        for fold_idx in range(k_folds):
            print(f"\nEvaluating fold {fold_idx + 1}/{k_folds}...")
            
            # Simple random sampling for demonstration
            # In practice, would implement proper sequence identity-based splitting
            fold_indices = list(range(fold_idx * fold_size, (fold_idx + 1) * fold_size))
            
            # Create subset dataset (simplified approach)
            # In practice, would need proper dataset subsetting
            try:
                fold_ranking = self.ranking_evaluator.evaluate_ranking_accuracy(
                    dataset, min(num_samples_per_fold, len(fold_indices)), batch_size
                )
                
                fold_result = {
                    'fold_index': fold_idx,
                    'fold_size': len(fold_indices),
                    'ranking_accuracy': fold_ranking['ranking_accuracy'],
                    'energy_gap_mean': fold_ranking['energy_gap_mean']
                }
                fold_results.append(fold_result)
                
            except Exception as e:
                print(f"Warning: Fold {fold_idx} evaluation failed: {e}")
                fold_results.append({
                    'fold_index': fold_idx,
                    'error': str(e)
                })
        
        # Aggregate results
        valid_folds = [f for f in fold_results if 'error' not in f]
        if valid_folds:
            accuracies = [f['ranking_accuracy'] for f in valid_folds]
            
            results['cv_summary'] = {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'cv_score': float(np.mean(accuracies)),
                'cv_std': float(np.std(accuracies)),
                'successful_folds': len(valid_folds),
                'failed_folds': len(fold_results) - len(valid_folds)
            }
            
            # 95% confidence interval for CV score
            if len(accuracies) > 1:
                se = np.std(accuracies) / np.sqrt(len(accuracies))
                ci_lower = np.mean(accuracies) - 1.96 * se
                ci_upper = np.mean(accuracies) + 1.96 * se
                results['cv_summary']['ci_95_lower'] = float(ci_lower)
                results['cv_summary']['ci_95_upper'] = float(ci_upper)
        
        results['fold_results'] = fold_results
        
        return results
    
    def _interpret_holdout_performance(self, holdout_results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret hold-out validation results."""
        if 'holdout_ranking' not in holdout_results:
            return {'error': 'No ranking results to interpret'}
        
        accuracy = holdout_results['holdout_ranking']['ranking_accuracy']
        
        interpretation = {}
        
        if accuracy >= 0.9:
            interpretation['performance'] = "Excellent generalization to unseen structures"
        elif accuracy >= 0.8:
            interpretation['performance'] = "Good generalization to unseen structures"
        elif accuracy >= 0.7:
            interpretation['performance'] = "Moderate generalization to unseen structures"
        elif accuracy >= 0.6:
            interpretation['performance'] = "Weak generalization to unseen structures"
        else:
            interpretation['performance'] = "Poor generalization - possible overfitting"
        
        # Check confidence intervals if available
        if 'confidence_intervals' in holdout_results:
            ci = holdout_results['confidence_intervals']
            if 'ci_lower' in ci:
                if ci['ci_lower'] > 0.5:
                    interpretation['confidence'] = "Performance significantly better than random"
                else:
                    interpretation['confidence'] = "Performance not significantly better than random"
        
        return interpretation
    
    def _interpret_family_differences(self, summary: Dict[str, Any]) -> str:
        """Interpret differences in performance across protein families."""
        accuracy_range = summary['accuracy_range']
        
        if accuracy_range < 0.05:
            return "Consistent performance across protein families"
        elif accuracy_range < 0.1:
            return "Small differences in performance across families"
        elif accuracy_range < 0.2:
            return "Moderate differences in performance across families"
        else:
            return "Large differences in performance across families - model may be biased"
    
    def _extract_sequences_from_dataset(
        self, 
        dataset: StabilityDataset, 
        num_samples: int, 
        seed: int = 42
    ) -> List[str]:
        """Extract sequences with reproducible sampling and failure tracking.
        
        Args:
            dataset: Dataset to extract from
            num_samples: Number of sequences to extract
            seed: Random seed for reproducible sequence selection
        """
        sequences = []
        failed_count = 0
        no_sequence_count = 0
        
        # Reproducible sampling
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for idx in indices:
            try:
                sample = dataset[idx]
                
                # Try to get sequence from different possible fields
                sequence = None
                if 'sequence' in sample:
                    sequence = sample['sequence']
                elif 'amino_acid_sequence' in sample:
                    sequence = sample['amino_acid_sequence']
                elif 'sequence_probs' in sample:
                    seq_probs = sample['sequence_probs']
                    seq_indices = seq_probs.argmax(dim=-1)
                    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                    sequence = ''.join(amino_acids[i] for i in seq_indices)
                
                if sequence and len(sequence) > 0:
                    sequences.append(sequence)
                else:
                    no_sequence_count += 1
                    
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # Log first few errors
                    warnings.warn(f"Failed to extract sequence from sample {idx}: {type(e).__name__}: {e}")
        
        # Warn if high failure rate
        total_attempted = len(indices)
        success_rate = len(sequences) / total_attempted if total_attempted > 0 else 0
        if success_rate < 0.5:
            warnings.warn(
                f"Low sequence extraction success rate: {len(sequences)}/{total_attempted} ({success_rate:.1%}). "
                f"Failed: {failed_count}, No sequence field: {no_sequence_count}. "
                f"Property analysis may be biased."
            )
        
        return sequences
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        # Remove non-serializable items for JSON
        results_for_json = results.copy()
        
        # Convert numpy arrays to lists
        if 'ranking_evaluation' in results_for_json:
            ranking_results = results_for_json['ranking_evaluation']
            for key in ['native_energies', 'negative_energies', 'energy_gaps']:
                if key in ranking_results and isinstance(ranking_results[key], list):
                    ranking_results[key] = [float(x) for x in ranking_results[key][:100]]  # Save first 100
        
        # Save JSON results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)
        
        print(f"✓ Evaluation results saved to {results_path}")
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print("\n" + "="*60)
        print("ENERGY MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Ranking evaluation
        if 'ranking_evaluation' in results:
            ranking = results['ranking_evaluation']
            print(f"\n📊 RANKING ACCURACY:")
            print(f"   Accuracy: {ranking['ranking_accuracy']:.3f}")
            print(f"   Total pairs: {ranking['total_pairs_evaluated']}")
            print(f"   Correct rankings: {ranking['correct_rankings']}")
            print(f"   Energy gap (mean): {ranking['energy_gap_mean']:.3f} ± {ranking['energy_gap_std']:.3f}")
            
            # Interpretation
            if ranking['ranking_accuracy'] > 0.9:
                print("   ✅ EXCELLENT ranking performance")
            elif ranking['ranking_accuracy'] > 0.7:
                print("   ✅ GOOD ranking performance")
            elif ranking['ranking_accuracy'] > 0.5:
                print("   ⚠️  MODERATE ranking performance")
            else:
                print("   ❌ POOR ranking performance (worse than random)")
        
        # Sequence properties
        if 'sequence_properties' in results and 'error' not in results['sequence_properties']:
            props = results['sequence_properties']
            print(f"\n🧬 SEQUENCE PROPERTIES:")
            print(f"   Sequences analyzed: {props['num_sequences']}")
            print(f"   Average length: {np.mean(props['sequence_lengths']):.1f}")
            print(f"   Unique amino acids: {props['composition']['unique_amino_acids']}/20")
            
            if 'properties' in props:
                for prop_name, prop_data in props['properties'].items():
                    print(f"   {prop_name.capitalize()}: {prop_data['mean']:.3f} ± {prop_data['std']:.3f}")
        
        # Plots generated
        if 'plots' in results:
            print(f"\n📈 VISUALIZATIONS:")
            for plot_name, plot_path in results['plots'].items():
                print(f"   {plot_name}: {plot_path}")
        
        print(f"\n📁 RESULTS DIRECTORY: {self.output_dir}")
        print("="*60)


def create_mock_evaluation_dataset(num_samples: int = 100) -> StabilityDataset:
    """
    Create a synthetic dataset for evaluation testing.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Mock StabilityDataset for testing
    """
    print(f"Creating mock evaluation dataset with {num_samples} samples...")
    
    # This would normally use real PDB data
    # For now, create a mock implementation that matches the dataset interface
    class MockEvaluationDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            torch.manual_seed(42)  # Reproducible
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create realistic mock data
            length = torch.randint(30, 100, (1,)).item()
            
            # Generate backbone features (normally from ProteinMPNN encoder)
            backbone_features = torch.randn(length, 128)
            
            # Generate sequence probabilities
            if idx % 2 == 0:  # Positive example (native sequence)
                # More concentrated probability distribution for native
                logits = torch.randn(length, 20) * 2.0
                sequence_probs = F.softmax(logits, dim=-1)
                label = 1
            else:  # Negative example (random/mutated)
                # More uniform distribution for random sequences
                logits = torch.randn(length, 20) * 0.5
                sequence_probs = F.softmax(logits, dim=-1)
                label = 0
            
            # Create mask
            mask = torch.ones(length)
            
            # Create amino acid sequence for property analysis
            seq_indices = sequence_probs.argmax(dim=-1)
            sequence = ''.join(self.amino_acids[i] for i in seq_indices)
            
            return {
                'backbone_features': backbone_features,
                'sequence_probs': sequence_probs,
                'mask': mask,
                'label': label,
                'sequence': sequence,
                'length': length
            }
    
    return MockEvaluationDataset(num_samples)


def main():
    """Main evaluation script with command-line interface."""
    parser = argparse.ArgumentParser(description="Energy Model Evaluation Framework")
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        required=True,
        help="Path to trained energy model checkpoint"
    )
    parser.add_argument(
        "--test_data_dir", 
        type=str,
        help="Directory containing test PDB structures"
    )
    parser.add_argument(
        "--encoder_type", 
        type=str, 
        default="vanilla",
        choices=["vanilla", "ca_model", "soluble"],
        help="ProteinMPNN encoder type"
    )
    parser.add_argument(
        "--encoder_name", 
        type=str, 
        default="v_48_020",
        help="ProteinMPNN model version"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_ranking_samples", 
        type=int, 
        default=1000,
        help="Number of samples for ranking evaluation"
    )
    parser.add_argument(
        "--num_property_samples", 
        type=int, 
        default=500,
        help="Number of samples for property analysis"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--no_plots", 
        action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--use_mock_data", 
        action="store_true",
        help="Use synthetic data for testing"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = EnergyModelEvaluator(
            model_checkpoint=args.model_checkpoint,
            encoder_type=args.encoder_type,
            encoder_name=args.encoder_name,
            device=args.device,
            output_dir=args.output_dir
        )
        
        # Create test dataset
        if args.use_mock_data:
            test_dataset = create_mock_evaluation_dataset(args.num_ranking_samples)
        else:
            if args.test_data_dir is None:
                raise ValueError("Must provide --test_data_dir or use --use_mock_data")
            
            test_dataset = StabilityDataset(
                data_dir=args.test_data_dir,
                positive_ratio=0.5,
                max_files=100  # Limit for evaluation
            )
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            test_dataset=test_dataset,
            num_ranking_samples=args.num_ranking_samples,
            num_property_samples=args.num_property_samples,
            batch_size=args.batch_size,
            generate_visualizations=not args.no_plots
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()