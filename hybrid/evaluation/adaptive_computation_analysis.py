#!/usr/bin/env python3
"""
Adaptive Computation Effectiveness Analysis for IRED Optimization

This module analyzes the effectiveness of adaptive computation allocation in IRED-style
sequence optimization, evaluating how well the system allocates additional computational
resources to harder problems and whether this allocation improves success rates.

Phase 4.3 Optimization Analysis - Adaptive Computation Component:

Key Analysis Areas:
1. Resource Allocation Analysis:
   - How computational steps are allocated across problem difficulties
   - Relationship between problem characteristics and resource allocation
   - Efficiency of adaptive vs fixed allocation strategies

2. Allocation Effectiveness Analysis:
   - Success rate improvements from adaptive allocation
   - Cost-benefit analysis of additional computation
   - Diminishing returns analysis

3. Problem Difficulty Assessment:
   - Automatic difficulty prediction accuracy
   - Correlation between predicted and actual resource needs
   - Refinement recommendations for difficulty assessment

4. Resource Utilization Patterns:
   - Distribution of computational effort across problems
   - Identification of over/under-allocation patterns
   - Optimization opportunities for allocation strategies

5. Comparative Analysis:
   - Adaptive vs fixed budget comparisons
   - Best-case vs worst-case allocation scenarios
   - Performance improvements and computational costs

Features:
- Statistical analysis of adaptive allocation effectiveness
- Visualization of resource allocation patterns
- Performance recommendations for allocation strategies
- Integration with convergence analysis results
"""

import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
from collections import defaultdict

import torch
import torch.nn as nn

# Optional dependencies with graceful degradation
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI threading issues
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Visualizations will be disabled.")

try:
    from scipy import stats
    from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced statistical analysis will be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Tabular analysis will be limited.")

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import project modules
from inference.ired_optimizer import IREDSequenceOptimizer, OptimizationConfig, OptimizationResult


@dataclass
class AdaptiveComputationConfig:
    """
    Configuration for adaptive computation effectiveness analysis.
    
    Analysis Settings:
        analyze_allocation_patterns: Analyze computational resource allocation patterns
        analyze_effectiveness: Study allocation effectiveness vs success rates
        analyze_difficulty_assessment: Evaluate difficulty prediction accuracy
        analyze_resource_utilization: Study resource utilization efficiency
        analyze_comparative_performance: Compare adaptive vs fixed allocation
        
    Statistical Settings:
        min_samples_per_group: Minimum samples required for statistical comparisons
        significance_threshold: P-value threshold for statistical significance tests
        effect_size_threshold: Minimum effect size to consider practically significant
        
    Resource Analysis Settings:
        fixed_budget_baseline: Fixed step budget for comparison analysis
        resource_efficiency_window: Window size for efficiency calculations
        allocation_granularity: Granularity for allocation pattern analysis
        
    Visualization Settings:
        generate_allocation_plots: Generate resource allocation visualizations
        generate_effectiveness_plots: Generate effectiveness analysis plots
        max_problems_plot: Maximum number of problems to plot individually
    """
    # Analysis types
    analyze_allocation_patterns: bool = True
    analyze_effectiveness: bool = True
    analyze_difficulty_assessment: bool = True
    analyze_resource_utilization: bool = True
    analyze_comparative_performance: bool = True
    
    # Statistical settings
    min_samples_per_group: int = 5
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.3
    bootstrap_samples: int = 1000
    
    # Resource analysis settings
    fixed_budget_baseline: int = 200  # Fixed step budget for comparison
    resource_efficiency_window: int = 10
    allocation_granularity: str = 'difficulty_level'  # 'difficulty_level', 'problem_type', 'both'
    
    # Analysis parameters
    success_rate_improvement_threshold: float = 0.1  # 10% improvement threshold
    resource_waste_threshold: float = 0.3  # 30% waste threshold
    
    # Visualization settings
    generate_allocation_plots: bool = True
    generate_effectiveness_plots: bool = True
    max_problems_plot: int = 100
    
    # Output settings
    save_detailed_analysis: bool = True
    include_statistical_tests: bool = True


@dataclass
class AdaptiveAllocationMetrics:
    """
    Metrics for a single adaptive computation allocation.
    
    Attributes:
        problem_id: Unique identifier for this problem
        problem_type: Type of optimization problem
        difficulty: Problem difficulty level
        predicted_difficulty: System-predicted difficulty score
        initial_allocation: Initial computational budget allocated
        total_allocation: Total computational resources used
        adaptive_extensions: Number of times budget was extended
        success: Whether optimization succeeded
        final_energy: Final energy achieved
        energy_improvement: Total energy improvement
        allocation_efficiency: Energy improvement per computational unit
        allocation_decision_quality: Quality score for allocation decisions
        resource_utilization: Fraction of allocated resources actually used effectively
        marginal_returns: Energy improvement in final allocation extension
    """
    problem_id: str
    problem_type: str
    difficulty: str
    predicted_difficulty: Optional[float]
    initial_allocation: int
    total_allocation: int
    adaptive_extensions: int
    success: bool
    final_energy: float
    energy_improvement: float
    allocation_efficiency: float
    allocation_decision_quality: float
    resource_utilization: float
    marginal_returns: Optional[float]


@dataclass
class AdaptiveComputationResult:
    """
    Results from adaptive computation effectiveness analysis.
    
    Attributes:
        timestamp: Analysis timestamp
        config: Configuration used for analysis
        problem_count: Number of problems analyzed
        allocation_pattern_analysis: Analysis of resource allocation patterns
        effectiveness_analysis: Allocation effectiveness vs success rates
        difficulty_assessment_analysis: Difficulty prediction accuracy analysis
        resource_utilization_analysis: Resource utilization efficiency analysis
        comparative_performance_analysis: Adaptive vs fixed allocation comparison
        computational_efficiency_metrics: Overall computational efficiency analysis
        optimization_recommendations: Recommendations for allocation improvement
    """
    timestamp: str
    config: AdaptiveComputationConfig
    problem_count: int
    allocation_pattern_analysis: Dict[str, Any]
    effectiveness_analysis: Dict[str, Any]
    difficulty_assessment_analysis: Dict[str, Any]
    resource_utilization_analysis: Dict[str, Any]
    comparative_performance_analysis: Dict[str, Any]
    computational_efficiency_metrics: Dict[str, Any]
    optimization_recommendations: List[str]


class AdaptiveComputationAnalyzer:
    """
    Analyzer for adaptive computation allocation effectiveness.
    
    This class evaluates how well the IRED optimization system allocates computational
    resources adaptively to problem difficulty, measuring both the allocation patterns
    and their effectiveness in improving success rates.
    
    The analyzer examines:
    - How resources are allocated across different problem types and difficulties
    - Whether adaptive allocation improves success rates vs fixed budgets
    - Efficiency of resource utilization and allocation decisions
    - Quality of difficulty assessment and prediction
    - Optimization opportunities for allocation strategies
    
    Args:
        config: Adaptive computation analysis configuration
        
    Example:
        >>> config = AdaptiveComputationConfig(
        ...     analyze_allocation_patterns=True,
        ...     fixed_budget_baseline=150
        ... )
        >>> analyzer = AdaptiveComputationAnalyzer(config)
        >>> results = analyzer.analyze_adaptive_allocation(optimization_data)
        >>> print(f"Allocation effectiveness: {results.effectiveness_analysis['overall_improvement']:.2%}")
    """
    
    def __init__(self, config: AdaptiveComputationConfig):
        """Initialize adaptive computation analyzer"""
        self.config = config
        self.allocation_metrics = []
        
        # Analysis results storage
        self.allocation_patterns = {}
        self.effectiveness_analysis = {}
        self.difficulty_analysis = {}
        self.utilization_analysis = {}
        self.comparative_analysis = {}
        self.efficiency_metrics = {}
    
    def analyze_adaptive_allocation(
        self, 
        optimization_data: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> AdaptiveComputationResult:
        """
        Analyze adaptive computation allocation effectiveness.
        
        Args:
            optimization_data: List of optimization result dictionaries
            output_dir: Directory to save analysis outputs (optional)
            
        Returns:
            AdaptiveComputationResult with comprehensive analysis
        """
        
        # Validate input
        if not isinstance(optimization_data, list):
            raise TypeError(f"optimization_data must be a list, got {type(optimization_data)}")
        
        if len(optimization_data) == 0:
            print("Warning: No optimization data provided for analysis")
            return self._create_empty_result()
        
        print(f"Analyzing adaptive computation effectiveness for {len(optimization_data)} optimization runs...")
        
        # Extract allocation metrics from optimization data
        self._extract_allocation_metrics(optimization_data)
        
        if len(self.allocation_metrics) == 0:
            print("Warning: No valid allocation metrics extracted")
            return self._create_empty_result()
        
        # Run analysis components based on configuration
        if self.config.analyze_allocation_patterns:
            self._analyze_allocation_patterns()
        
        if self.config.analyze_effectiveness:
            self._analyze_allocation_effectiveness()
        
        if self.config.analyze_difficulty_assessment:
            self._analyze_difficulty_assessment()
        
        if self.config.analyze_resource_utilization:
            self._analyze_resource_utilization()
        
        if self.config.analyze_comparative_performance:
            self._analyze_comparative_performance()
        
        # Calculate overall efficiency metrics
        self._calculate_efficiency_metrics()
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations()
        
        # Create visualizations if requested
        if output_dir and PLOTTING_AVAILABLE:
            self._generate_visualizations(output_dir)
        
        # Compile results
        result = AdaptiveComputationResult(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            problem_count=len(self.allocation_metrics),
            allocation_pattern_analysis=self.allocation_patterns,
            effectiveness_analysis=self.effectiveness_analysis,
            difficulty_assessment_analysis=self.difficulty_analysis,
            resource_utilization_analysis=self.utilization_analysis,
            comparative_performance_analysis=self.comparative_analysis,
            computational_efficiency_metrics=self.efficiency_metrics,
            optimization_recommendations=recommendations
        )
        
        # Save detailed analysis if requested
        if output_dir and self.config.save_detailed_analysis:
            self._save_detailed_analysis(result, output_dir)
        
        print(f"Adaptive computation analysis complete. Analyzed {len(self.allocation_metrics)} problems.")
        
        return result
    
    def _create_empty_result(self) -> AdaptiveComputationResult:
        """Create empty result for cases with no data"""
        return AdaptiveComputationResult(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            problem_count=0,
            allocation_pattern_analysis={},
            effectiveness_analysis={},
            difficulty_assessment_analysis={},
            resource_utilization_analysis={},
            comparative_performance_analysis={},
            computational_efficiency_metrics={},
            optimization_recommendations=["No optimization data available for analysis"]
        )
    
    def _extract_allocation_metrics(self, optimization_data: List[Dict[str, Any]]):
        """Extract adaptive allocation metrics from optimization data"""
        
        self.allocation_metrics = []
        processed_count = 0
        skipped_count = 0
        
        for i, opt_data in enumerate(optimization_data):
            try:
                # Validate optimization data structure
                if not isinstance(opt_data, dict):
                    print(f"Warning: Optimization data {i} is not a dictionary, skipping")
                    skipped_count += 1
                    continue
                
                # Extract basic problem information
                problem_info = opt_data.get('problem_info', {})
                optimization_result = opt_data.get('optimization_result', {})
                trajectory_info = opt_data.get('trajectory', [])
                
                if not isinstance(problem_info, dict):
                    problem_info = {}
                
                problem_id = f"problem_{i:04d}"
                problem_type = problem_info.get('type', 'unknown')
                difficulty = problem_info.get('difficulty', 'unknown')
                
                # Extract adaptive allocation information
                initial_allocation = optimization_result.get('initial_steps_allocated', 0)
                total_allocation = optimization_result.get('total_steps_used', 0)
                adaptive_extensions = optimization_result.get('adaptive_extensions_count', 0)
                
                # If no explicit allocation info, infer from trajectory
                if total_allocation == 0 and isinstance(trajectory_info, list):
                    total_allocation = len(trajectory_info)
                
                # Extract performance metrics
                success = optimization_result.get('converged', False)
                final_energy = optimization_result.get('final_energy', float('inf'))
                initial_energy = optimization_result.get('initial_energy', final_energy)
                energy_improvement = initial_energy - final_energy if np.isfinite(initial_energy) and np.isfinite(final_energy) else 0.0
                
                # Calculate derived metrics
                allocation_efficiency = self._calculate_allocation_efficiency(energy_improvement, total_allocation)
                allocation_decision_quality = self._assess_allocation_decision_quality(opt_data)
                resource_utilization = self._calculate_resource_utilization(trajectory_info, total_allocation)
                marginal_returns = self._calculate_marginal_returns(trajectory_info, adaptive_extensions)
                
                # Extract difficulty prediction if available
                predicted_difficulty = optimization_result.get('predicted_difficulty')
                
                # Create allocation metrics object
                metrics = AdaptiveAllocationMetrics(
                    problem_id=problem_id,
                    problem_type=problem_type,
                    difficulty=difficulty,
                    predicted_difficulty=predicted_difficulty,
                    initial_allocation=initial_allocation,
                    total_allocation=total_allocation,
                    adaptive_extensions=adaptive_extensions,
                    success=success,
                    final_energy=final_energy,
                    energy_improvement=energy_improvement,
                    allocation_efficiency=allocation_efficiency,
                    allocation_decision_quality=allocation_decision_quality,
                    resource_utilization=resource_utilization,
                    marginal_returns=marginal_returns
                )
                
                self.allocation_metrics.append(metrics)
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to process optimization data {i}: {str(e)}")
                skipped_count += 1
                continue
        
        # Log processing statistics
        print(f"Allocation metrics extraction complete: {processed_count} processed, {skipped_count} skipped")
    
    def _calculate_allocation_efficiency(self, energy_improvement: float, total_allocation: int) -> float:
        """Calculate energy improvement per computational unit"""
        if total_allocation > 0 and np.isfinite(energy_improvement):
            return energy_improvement / total_allocation
        return 0.0
    
    def _assess_allocation_decision_quality(self, opt_data: Dict[str, Any]) -> float:
        """Assess the quality of adaptive allocation decisions"""
        
        # This is a simplified assessment - in real implementation would be more sophisticated
        optimization_result = opt_data.get('optimization_result', {})
        
        success = optimization_result.get('converged', False)
        adaptive_extensions = optimization_result.get('adaptive_extensions_count', 0)
        total_steps = optimization_result.get('total_steps_used', 1)
        
        # Good decision quality indicators:
        # - Success with minimal extensions (efficient)
        # - Extensions led to eventual success (effective)
        # - No extensions when problem was easy (conservative)
        
        if success:
            if adaptive_extensions == 0:
                return 0.9  # Efficient success without needing extensions
            elif adaptive_extensions <= 2:
                return 0.8  # Successful with reasonable extensions
            else:
                return 0.6  # Success but required many extensions
        else:
            if adaptive_extensions == 0:
                return 0.4  # Failed without trying extensions (might be under-allocation)
            else:
                return 0.2  # Failed despite extensions (poor problem or strategy)
    
    def _calculate_resource_utilization(self, trajectory_info: List[Dict[str, Any]], total_allocation: int) -> float:
        """Calculate effective utilization of allocated resources"""
        
        if not isinstance(trajectory_info, list) or total_allocation <= 0:
            return 0.0
        
        # Simple utilization: fraction of steps that made progress
        if len(trajectory_info) == 0:
            return 0.0
        
        # Count steps that improved energy
        improving_steps = 0
        for i in range(1, len(trajectory_info)):
            prev_energy = trajectory_info[i-1].get('energy', float('inf'))
            curr_energy = trajectory_info[i].get('energy', float('inf'))
            
            if np.isfinite(prev_energy) and np.isfinite(curr_energy) and curr_energy < prev_energy:
                improving_steps += 1
        
        return improving_steps / len(trajectory_info) if len(trajectory_info) > 0 else 0.0
    
    def _calculate_marginal_returns(self, trajectory_info: List[Dict[str, Any]], adaptive_extensions: int) -> Optional[float]:
        """Calculate energy improvement in final adaptive extension"""
        
        if adaptive_extensions == 0 or not isinstance(trajectory_info, list) or len(trajectory_info) == 0:
            return None
        
        # Approximate the energy improvement in the final extension
        # This is simplified - real implementation would track extension boundaries
        final_quarter = len(trajectory_info) // 4
        if final_quarter < 2:
            return None
        
        final_steps = trajectory_info[-final_quarter:]
        if len(final_steps) < 2:
            return None
        
        initial_energy = final_steps[0].get('energy', float('inf'))
        final_energy = final_steps[-1].get('energy', float('inf'))
        
        if np.isfinite(initial_energy) and np.isfinite(final_energy):
            return initial_energy - final_energy
        
        return None
    
    def _analyze_allocation_patterns(self):
        """Analyze computational resource allocation patterns"""
        
        self.allocation_patterns = {
            'allocation_distribution': {},
            'allocation_by_difficulty': {},
            'allocation_by_type': {},
            'extension_patterns': {},
            'allocation_trends': {}
        }
        
        if not self.allocation_metrics:
            return
        
        # Overall allocation distribution
        total_allocations = [m.total_allocation for m in self.allocation_metrics]
        initial_allocations = [m.initial_allocation for m in self.allocation_metrics]
        extensions = [m.adaptive_extensions for m in self.allocation_metrics]
        
        self.allocation_patterns['allocation_distribution'] = {
            'mean_total_allocation': np.mean(total_allocations),
            'median_total_allocation': np.median(total_allocations),
            'std_total_allocation': np.std(total_allocations),
            'mean_initial_allocation': np.mean(initial_allocations),
            'mean_extensions': np.mean(extensions),
            'extension_frequency': sum(1 for e in extensions if e > 0) / len(extensions)
        }
        
        # Allocation by difficulty
        difficulty_groups = defaultdict(list)
        for metrics in self.allocation_metrics:
            difficulty_groups[metrics.difficulty].append(metrics.total_allocation)
        
        for difficulty, allocations in difficulty_groups.items():
            self.allocation_patterns['allocation_by_difficulty'][difficulty] = {
                'mean_allocation': np.mean(allocations),
                'median_allocation': np.median(allocations),
                'problem_count': len(allocations)
            }
        
        # Allocation by problem type
        type_groups = defaultdict(list)
        for metrics in self.allocation_metrics:
            type_groups[metrics.problem_type].append(metrics.total_allocation)
        
        for problem_type, allocations in type_groups.items():
            self.allocation_patterns['allocation_by_type'][problem_type] = {
                'mean_allocation': np.mean(allocations),
                'median_allocation': np.median(allocations),
                'problem_count': len(allocations)
            }
        
        # Extension patterns
        extension_groups = defaultdict(list)
        for metrics in self.allocation_metrics:
            if metrics.adaptive_extensions > 0:
                key = f"{metrics.difficulty}_{metrics.problem_type}"
                extension_groups[key].append(metrics.adaptive_extensions)
        
        for group_key, ext_counts in extension_groups.items():
            self.allocation_patterns['extension_patterns'][group_key] = {
                'mean_extensions': np.mean(ext_counts),
                'max_extensions': max(ext_counts),
                'problems_with_extensions': len(ext_counts)
            }
    
    def _analyze_allocation_effectiveness(self):
        """Analyze effectiveness of adaptive allocation decisions"""
        
        self.effectiveness_analysis = {
            'success_rate_by_allocation': {},
            'efficiency_analysis': {},
            'extension_effectiveness': {},
            'allocation_success_correlation': {}
        }
        
        if not self.allocation_metrics:
            return
        
        # Success rates by allocation level
        allocation_bins = self._create_allocation_bins()
        for bin_name, (min_alloc, max_alloc) in allocation_bins.items():
            bin_metrics = [m for m in self.allocation_metrics 
                          if min_alloc <= m.total_allocation < max_alloc]
            
            if bin_metrics:
                success_rate = sum(1 for m in bin_metrics if m.success) / len(bin_metrics)
                mean_efficiency = np.mean([m.allocation_efficiency for m in bin_metrics])
                
                self.effectiveness_analysis['success_rate_by_allocation'][bin_name] = {
                    'success_rate': success_rate,
                    'mean_efficiency': mean_efficiency,
                    'problem_count': len(bin_metrics)
                }
        
        # Efficiency analysis
        successful_metrics = [m for m in self.allocation_metrics if m.success]
        failed_metrics = [m for m in self.allocation_metrics if not m.success]
        
        self.effectiveness_analysis['efficiency_analysis'] = {
            'mean_efficiency_successful': np.mean([m.allocation_efficiency for m in successful_metrics]) if successful_metrics else 0.0,
            'mean_efficiency_failed': np.mean([m.allocation_efficiency for m in failed_metrics]) if failed_metrics else 0.0,
            'mean_allocation_successful': np.mean([m.total_allocation for m in successful_metrics]) if successful_metrics else 0.0,
            'mean_allocation_failed': np.mean([m.total_allocation for m in failed_metrics]) if failed_metrics else 0.0
        }
        
        # Extension effectiveness
        extended_metrics = [m for m in self.allocation_metrics if m.adaptive_extensions > 0]
        non_extended_metrics = [m for m in self.allocation_metrics if m.adaptive_extensions == 0]
        
        if extended_metrics and non_extended_metrics:
            extended_success_rate = sum(1 for m in extended_metrics if m.success) / len(extended_metrics)
            non_extended_success_rate = sum(1 for m in non_extended_metrics if m.success) / len(non_extended_metrics)
            
            self.effectiveness_analysis['extension_effectiveness'] = {
                'extended_success_rate': extended_success_rate,
                'non_extended_success_rate': non_extended_success_rate,
                'extension_benefit': extended_success_rate - non_extended_success_rate,
                'problems_with_extensions': len(extended_metrics),
                'problems_without_extensions': len(non_extended_metrics)
            }
        
        # Allocation-success correlation (with sample size check)
        if SCIPY_AVAILABLE and len(self.allocation_metrics) >= self.config.min_samples_per_group:
            allocations = [m.total_allocation for m in self.allocation_metrics]
            successes = [1 if m.success else 0 for m in self.allocation_metrics]
            
            correlation, p_value = spearmanr(allocations, successes)
            
            self.effectiveness_analysis['allocation_success_correlation'] = {
                'correlation': correlation if np.isfinite(correlation) else 0.0,
                'p_value': p_value if np.isfinite(p_value) else 1.0,
                'significant': p_value < self.config.significance_threshold if np.isfinite(p_value) else False,
                'sample_size': len(self.allocation_metrics),
                'sufficient_sample_size': True
            }
        else:
            # Insufficient sample size for meaningful correlation
            self.effectiveness_analysis['allocation_success_correlation'] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'significant': False,
                'sample_size': len(self.allocation_metrics),
                'sufficient_sample_size': False,
                'message': f'Insufficient sample size for correlation analysis (need >= {self.config.min_samples_per_group}, got {len(self.allocation_metrics)})'
            }
    
    def _create_allocation_bins(self) -> Dict[str, Tuple[int, int]]:
        """Create allocation bins for analysis"""
        
        allocations = [m.total_allocation for m in self.allocation_metrics]
        if not allocations:
            return {}
        
        min_alloc = min(allocations)
        max_alloc = max(allocations)
        
        # Create 5 bins
        bin_size = (max_alloc - min_alloc) / 5
        
        bins = {}
        for i in range(5):
            bin_min = min_alloc + i * bin_size
            bin_max = min_alloc + (i + 1) * bin_size
            bins[f"bin_{i+1}"] = (int(bin_min), int(bin_max))
        
        return bins
    
    def _analyze_difficulty_assessment(self):
        """Analyze accuracy of difficulty assessment and prediction"""
        
        self.difficulty_analysis = {
            'prediction_accuracy': {},
            'difficulty_allocation_correlation': {},
            'misallocation_analysis': {}
        }
        
        # Filter metrics with difficulty predictions
        predicted_metrics = [m for m in self.allocation_metrics if m.predicted_difficulty is not None]
        
        if not predicted_metrics:
            self.difficulty_analysis['prediction_accuracy'] = {
                'message': 'No difficulty predictions available for analysis'
            }
            return
        
        # Analyze prediction accuracy
        # Map difficulty levels to numeric scores for correlation analysis
        difficulty_mapping = {'easy': 1, 'medium': 2, 'hard': 3}
        
        actual_difficulties = []
        predicted_difficulties = []
        
        for metrics in predicted_metrics:
            if metrics.difficulty in difficulty_mapping:
                actual_difficulties.append(difficulty_mapping[metrics.difficulty])
                predicted_difficulties.append(metrics.predicted_difficulty)
        
        # Check minimum sample size for meaningful correlation analysis
        if len(actual_difficulties) >= self.config.min_samples_per_group and SCIPY_AVAILABLE:
            correlation, p_value = spearmanr(actual_difficulties, predicted_difficulties)
            
            # Determine prediction quality based on correlation strength and significance
            is_significant = p_value < self.config.significance_threshold if np.isfinite(p_value) else False
            correlation_value = correlation if np.isfinite(correlation) else 0.0
            
            if is_significant and correlation_value > 0.6:
                quality = 'good'
            elif is_significant and correlation_value > 0.3:
                quality = 'moderate'
            else:
                quality = 'poor'
            
            self.difficulty_analysis['prediction_accuracy'] = {
                'prediction_correlation': correlation_value,
                'correlation_p_value': p_value if np.isfinite(p_value) else 1.0,
                'samples_with_predictions': len(predicted_metrics),
                'prediction_quality': quality,
                'statistically_significant': is_significant,
                'sufficient_sample_size': len(actual_difficulties) >= self.config.min_samples_per_group
            }
        else:
            # Insufficient data for statistical analysis
            self.difficulty_analysis['prediction_accuracy'] = {
                'prediction_correlation': 0.0,
                'correlation_p_value': 1.0,
                'samples_with_predictions': len(predicted_metrics),
                'prediction_quality': 'insufficient_data',
                'statistically_significant': False,
                'sufficient_sample_size': False,
                'message': f'Insufficient sample size for correlation analysis (need >= {self.config.min_samples_per_group}, got {len(actual_difficulties)})'
            }
        
        # Analyze difficulty-allocation correlation
        difficulty_allocations = defaultdict(list)
        for metrics in self.allocation_metrics:
            difficulty_allocations[metrics.difficulty].append(metrics.total_allocation)
        
        # Calculate mean allocations by difficulty
        difficulty_means = {}
        for difficulty, allocations in difficulty_allocations.items():
            difficulty_means[difficulty] = np.mean(allocations)
        
        self.difficulty_analysis['difficulty_allocation_correlation'] = difficulty_means
        
        # Misallocation analysis: cases where allocation doesn't match outcome
        # Calculate mean allocation once for efficiency
        if self.allocation_metrics:
            mean_allocation = np.mean([m.total_allocation for m in self.allocation_metrics])
        else:
            mean_allocation = 0.0
        
        misallocations = []
        for metrics in self.allocation_metrics:
            # High allocation but failure
            if metrics.total_allocation > mean_allocation and not metrics.success:
                misallocations.append(('over_allocation', metrics))
            # Low allocation but could have succeeded with more
            elif metrics.total_allocation < mean_allocation and not metrics.success:
                misallocations.append(('under_allocation', metrics))
        
        misallocation_counts = defaultdict(int)
        for misalloc_type, _ in misallocations:
            misallocation_counts[misalloc_type] += 1
        
        self.difficulty_analysis['misallocation_analysis'] = {
            'total_misallocations': len(misallocations),
            'misallocation_types': dict(misallocation_counts),
            'misallocation_rate': len(misallocations) / len(self.allocation_metrics) if self.allocation_metrics else 0.0
        }
    
    def _analyze_resource_utilization(self):
        """Analyze resource utilization efficiency patterns"""
        
        self.utilization_analysis = {
            'utilization_statistics': {},
            'utilization_by_success': {},
            'utilization_efficiency_correlation': {},
            'waste_analysis': {}
        }
        
        if not self.allocation_metrics:
            return
        
        # Overall utilization statistics
        utilizations = [m.resource_utilization for m in self.allocation_metrics]
        self.utilization_analysis['utilization_statistics'] = {
            'mean_utilization': np.mean(utilizations),
            'median_utilization': np.median(utilizations),
            'std_utilization': np.std(utilizations),
            'min_utilization': min(utilizations),
            'max_utilization': max(utilizations)
        }
        
        # Utilization by success
        successful_utilizations = [m.resource_utilization for m in self.allocation_metrics if m.success]
        failed_utilizations = [m.resource_utilization for m in self.allocation_metrics if not m.success]
        
        self.utilization_analysis['utilization_by_success'] = {
            'mean_utilization_successful': np.mean(successful_utilizations) if successful_utilizations else 0.0,
            'mean_utilization_failed': np.mean(failed_utilizations) if failed_utilizations else 0.0,
            'utilization_difference': (np.mean(successful_utilizations) - np.mean(failed_utilizations)) 
                                     if successful_utilizations and failed_utilizations else 0.0
        }
        
        # Utilization-efficiency correlation
        if SCIPY_AVAILABLE and len(self.allocation_metrics) > 1:
            utilizations = [m.resource_utilization for m in self.allocation_metrics]
            efficiencies = [m.allocation_efficiency for m in self.allocation_metrics]
            
            correlation, p_value = spearmanr(utilizations, efficiencies)
            
            self.utilization_analysis['utilization_efficiency_correlation'] = {
                'correlation': correlation if np.isfinite(correlation) else 0.0,
                'p_value': p_value if np.isfinite(p_value) else 1.0,
                'significant': p_value < self.config.significance_threshold if np.isfinite(p_value) else False
            }
        
        # Resource waste analysis
        low_utilization_problems = [m for m in self.allocation_metrics 
                                   if m.resource_utilization < self.config.resource_waste_threshold]
        
        self.utilization_analysis['waste_analysis'] = {
            'low_utilization_count': len(low_utilization_problems),
            'low_utilization_rate': len(low_utilization_problems) / len(self.allocation_metrics),
            'wasted_resources': sum(m.total_allocation * (1 - m.resource_utilization) 
                                   for m in low_utilization_problems),
            'waste_by_difficulty': self._analyze_waste_by_difficulty(low_utilization_problems)
        }
    
    def _analyze_waste_by_difficulty(self, low_utilization_problems: List[AdaptiveAllocationMetrics]) -> Dict[str, int]:
        """Analyze resource waste by problem difficulty"""
        waste_by_difficulty = defaultdict(int)
        for metrics in low_utilization_problems:
            waste_by_difficulty[metrics.difficulty] += 1
        return dict(waste_by_difficulty)
    
    def _analyze_comparative_performance(self):
        """Compare adaptive vs fixed allocation performance"""
        
        self.comparative_analysis = {
            'fixed_budget_comparison': {},
            'adaptive_vs_fixed_success_rates': {},
            'computational_efficiency_comparison': {},
            'cost_benefit_analysis': {}
        }
        
        if not self.allocation_metrics:
            return
        
        # Simulate fixed budget performance
        fixed_budget = self.config.fixed_budget_baseline
        
        # Count how many problems would succeed with fixed budget
        fixed_budget_successes = 0
        adaptive_successes = 0
        
        fixed_budget_total_cost = 0
        adaptive_total_cost = 0
        
        for metrics in self.allocation_metrics:
            # Fixed budget success: problem succeeded and used <= fixed budget
            if metrics.success and metrics.total_allocation <= fixed_budget:
                fixed_budget_successes += 1
            
            # Adaptive success: problem succeeded regardless of allocation
            if metrics.success:
                adaptive_successes += 1
            
            # Calculate costs
            fixed_budget_total_cost += fixed_budget  # Always allocate full budget
            adaptive_total_cost += metrics.total_allocation
        
        total_problems = len(self.allocation_metrics)
        
        # Protect against division by zero
        if total_problems == 0:
            print("Warning: No allocation metrics available for comparative analysis")
            return
        
        self.comparative_analysis['fixed_budget_comparison'] = {
            'fixed_budget_size': fixed_budget,
            'fixed_budget_successes': fixed_budget_successes,
            'fixed_budget_success_rate': fixed_budget_successes / total_problems,
            'adaptive_successes': adaptive_successes,
            'adaptive_success_rate': adaptive_successes / total_problems,
            'success_rate_improvement': (adaptive_successes - fixed_budget_successes) / total_problems
        }
        
        # Computational efficiency comparison
        fixed_budget_efficiency = fixed_budget_successes / fixed_budget_total_cost if fixed_budget_total_cost > 0 else 0.0
        adaptive_efficiency = adaptive_successes / adaptive_total_cost if adaptive_total_cost > 0 else 0.0
        
        self.comparative_analysis['computational_efficiency_comparison'] = {
            'fixed_budget_efficiency': fixed_budget_efficiency,
            'adaptive_efficiency': adaptive_efficiency,
            'efficiency_ratio': adaptive_efficiency / fixed_budget_efficiency if fixed_budget_efficiency > 0 else 0.0,
            'total_cost_fixed': fixed_budget_total_cost,
            'total_cost_adaptive': adaptive_total_cost,
            'cost_difference': adaptive_total_cost - fixed_budget_total_cost
        }
        
        # Cost-benefit analysis
        additional_successes = adaptive_successes - fixed_budget_successes
        additional_cost = adaptive_total_cost - fixed_budget_total_cost
        
        self.comparative_analysis['cost_benefit_analysis'] = {
            'additional_successes': additional_successes,
            'additional_cost': additional_cost,
            'cost_per_additional_success': additional_cost / additional_successes if additional_successes > 0 else float('inf'),
            'break_even_point': additional_cost / additional_successes if additional_successes > 0 else None
        }
    
    def _calculate_efficiency_metrics(self):
        """Calculate overall computational efficiency metrics"""
        
        self.efficiency_metrics = {
            'overall_efficiency': {},
            'efficiency_distribution': {},
            'top_performers': {},
            'efficiency_trends': {}
        }
        
        if not self.allocation_metrics:
            return
        
        # Overall efficiency
        efficiencies = [m.allocation_efficiency for m in self.allocation_metrics]
        utilizations = [m.resource_utilization for m in self.allocation_metrics]
        
        self.efficiency_metrics['overall_efficiency'] = {
            'mean_allocation_efficiency': np.mean(efficiencies),
            'median_allocation_efficiency': np.median(efficiencies),
            'mean_resource_utilization': np.mean(utilizations),
            'combined_efficiency_score': np.mean(efficiencies) * np.mean(utilizations)
        }
        
        # Efficiency distribution
        efficiency_bins = {
            'very_low': [e for e in efficiencies if e < 0.001],
            'low': [e for e in efficiencies if 0.001 <= e < 0.01],
            'medium': [e for e in efficiencies if 0.01 <= e < 0.1],
            'high': [e for e in efficiencies if e >= 0.1]
        }
        
        self.efficiency_metrics['efficiency_distribution'] = {
            level: len(values) for level, values in efficiency_bins.items()
        }
        
        # Top performers
        sorted_metrics = sorted(self.allocation_metrics, 
                               key=lambda m: m.allocation_efficiency, reverse=True)
        top_10_percent = max(1, len(sorted_metrics) // 10)
        
        top_performers = sorted_metrics[:top_10_percent]
        self.efficiency_metrics['top_performers'] = {
            'count': len(top_performers),
            'mean_efficiency': np.mean([m.allocation_efficiency for m in top_performers]),
            'mean_allocation': np.mean([m.total_allocation for m in top_performers]),
            'success_rate': sum(1 for m in top_performers if m.success) / len(top_performers)
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for improving adaptive allocation"""
        
        recommendations = []
        
        if not self.allocation_metrics:
            recommendations.append("No allocation data available for analysis.")
            return recommendations
        
        # Analyze overall effectiveness
        if 'extension_effectiveness' in self.effectiveness_analysis:
            effectiveness = self.effectiveness_analysis['extension_effectiveness']
            extension_benefit = effectiveness.get('extension_benefit', 0.0)
            
            if extension_benefit > self.config.success_rate_improvement_threshold:
                recommendations.append(
                    f"Adaptive extensions are effective (+{extension_benefit:.1%} success rate). "
                    "Consider more aggressive extension policies for difficult problems."
                )
            elif extension_benefit < 0:
                recommendations.append(
                    f"Adaptive extensions reduce success rates ({extension_benefit:.1%}). "
                    "Review extension criteria and consider more conservative allocation."
                )
        
        # Analyze resource waste
        if 'waste_analysis' in self.utilization_analysis:
            waste_rate = self.utilization_analysis['waste_analysis'].get('low_utilization_rate', 0.0)
            
            if waste_rate > 0.3:  # More than 30% waste
                recommendations.append(
                    f"High resource waste detected ({waste_rate:.1%} of problems show low utilization). "
                    "Consider tighter allocation policies or improved early stopping criteria."
                )
        
        # Analyze difficulty prediction
        if 'prediction_accuracy' in self.difficulty_analysis:
            prediction_quality = self.difficulty_analysis['prediction_accuracy'].get('prediction_quality', 'unknown')
            
            if prediction_quality == 'poor':
                recommendations.append(
                    "Difficulty prediction accuracy is low. Improve difficulty assessment algorithms "
                    "to enable better resource allocation decisions."
                )
        
        # Analyze computational efficiency
        if 'computational_efficiency_comparison' in self.comparative_analysis:
            efficiency_ratio = self.comparative_analysis['computational_efficiency_comparison'].get('efficiency_ratio', 0.0)
            
            if efficiency_ratio < 1.0:
                recommendations.append(
                    f"Fixed budget allocation is more efficient (ratio: {efficiency_ratio:.2f}). "
                    "Adaptive allocation needs improvement or problems may not benefit from adaptive strategies."
                )
            elif efficiency_ratio > 1.5:
                recommendations.append(
                    f"Adaptive allocation is highly efficient (ratio: {efficiency_ratio:.2f}). "
                    "Consider expanding adaptive strategies to more problem types."
                )
        
        # Analyze marginal returns
        marginal_returns = [m.marginal_returns for m in self.allocation_metrics 
                           if m.marginal_returns is not None]
        
        if marginal_returns:
            mean_marginal = np.mean(marginal_returns)
            if mean_marginal < 0.001:  # Very low marginal returns
                recommendations.append(
                    f"Low marginal returns from extensions (avg: {mean_marginal:.4f}). "
                    "Consider reducing maximum extension limits or improving stopping criteria."
                )
        
        if not recommendations:
            recommendations.append(
                "Adaptive computation allocation appears to be working effectively. "
                "Continue monitoring performance and consider fine-tuning based on specific problem patterns."
            )
        
        return recommendations
    
    def _generate_visualizations(self, output_dir: str):
        """Generate adaptive computation analysis visualizations"""
        
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping visualizations.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating adaptive computation analysis visualizations...")
        
        try:
            # Create multi-panel summary plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Allocation distribution by difficulty
            if self.config.generate_allocation_plots:
                self._plot_allocation_by_difficulty(axes[0, 0])
            
            # 2. Success rate by allocation level
            self._plot_success_rate_by_allocation(axes[0, 1])
            
            # 3. Resource utilization distribution
            self._plot_resource_utilization_distribution(axes[0, 2])
            
            # 4. Adaptive vs fixed budget comparison
            if self.config.analyze_comparative_performance:
                self._plot_adaptive_vs_fixed_comparison(axes[1, 0])
            
            # 5. Allocation efficiency scatter
            self._plot_allocation_efficiency_scatter(axes[1, 1])
            
            # 6. Extension effectiveness
            self._plot_extension_effectiveness(axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(output_path / "adaptive_computation_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            print(f"Failed to generate visualizations: {str(e)}")
    
    def _plot_allocation_by_difficulty(self, ax):
        """Plot resource allocation by problem difficulty"""
        
        if 'allocation_by_difficulty' not in self.allocation_patterns:
            ax.text(0.5, 0.5, 'No allocation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Allocation by Difficulty')
            return
        
        difficulties = []
        allocations = []
        
        for difficulty, data in self.allocation_patterns['allocation_by_difficulty'].items():
            difficulties.append(difficulty.title())
            allocations.append(data['mean_allocation'])
        
        if difficulties and allocations:
            colors = {'Easy': 'lightgreen', 'Medium': 'orange', 'Hard': 'lightcoral'}
            bar_colors = [colors.get(d, 'gray') for d in difficulties]
            
            bars = ax.bar(difficulties, allocations, color=bar_colors, alpha=0.7)
            ax.set_ylabel('Mean Allocation (steps)')
            ax.set_title('Resource Allocation by Difficulty')
            
            # Add value labels
            for bar, alloc in zip(bars, allocations):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{alloc:.0f}', ha='center', va='bottom')
    
    def _plot_success_rate_by_allocation(self, ax):
        """Plot success rate by allocation level"""
        
        if 'success_rate_by_allocation' not in self.effectiveness_analysis:
            ax.text(0.5, 0.5, 'No effectiveness data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Rate by Allocation')
            return
        
        bins = []
        success_rates = []
        
        for bin_name, data in self.effectiveness_analysis['success_rate_by_allocation'].items():
            bins.append(bin_name.replace('bin_', 'Bin '))
            success_rates.append(data['success_rate'])
        
        if bins and success_rates:
            bars = ax.bar(bins, success_rates, color='skyblue', alpha=0.7)
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rate by Allocation Level')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
    
    def _plot_resource_utilization_distribution(self, ax):
        """Plot distribution of resource utilization"""
        
        utilizations = [m.resource_utilization for m in self.allocation_metrics]
        
        if utilizations:
            ax.hist(utilizations, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Resource Utilization')
            ax.set_ylabel('Frequency')
            ax.set_title('Resource Utilization Distribution')
            
            # Add mean line
            mean_util = np.mean(utilizations)
            ax.axvline(mean_util, color='red', linestyle='--', label=f'Mean: {mean_util:.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Resource Utilization Distribution')
    
    def _plot_adaptive_vs_fixed_comparison(self, ax):
        """Plot adaptive vs fixed budget comparison"""
        
        if 'fixed_budget_comparison' not in self.comparative_analysis:
            ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Adaptive vs Fixed Comparison')
            return
        
        comparison = self.comparative_analysis['fixed_budget_comparison']
        
        methods = ['Fixed Budget', 'Adaptive']
        success_rates = [
            comparison['fixed_budget_success_rate'],
            comparison['adaptive_success_rate']
        ]
        
        colors = ['lightcoral', 'lightblue']
        bars = ax.bar(methods, success_rates, color=colors, alpha=0.7)
        ax.set_ylabel('Success Rate')
        ax.set_title('Adaptive vs Fixed Budget Performance')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
    
    def _plot_allocation_efficiency_scatter(self, ax):
        """Plot allocation efficiency scatter plot"""
        
        allocations = []
        efficiencies = []
        colors = []
        
        for metrics in self.allocation_metrics:
            allocations.append(metrics.total_allocation)
            efficiencies.append(metrics.allocation_efficiency)
            colors.append('green' if metrics.success else 'red')
        
        if allocations and efficiencies:
            ax.scatter(allocations, efficiencies, c=colors, alpha=0.6)
            ax.set_xlabel('Total Allocation (steps)')
            ax.set_ylabel('Allocation Efficiency')
            ax.set_title('Allocation Efficiency vs Resources Used')
            
            # Add trend line if scipy available
            if SCIPY_AVAILABLE and len(allocations) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(allocations, efficiencies)
                x_trend = np.array([min(allocations), max(allocations)])
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, 'b--', alpha=0.8,
                       label=f'Trend (R²={r_value**2:.3f})')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No efficiency data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Allocation Efficiency vs Resources Used')
    
    def _plot_extension_effectiveness(self, ax):
        """Plot effectiveness of adaptive extensions"""
        
        if 'extension_effectiveness' not in self.effectiveness_analysis:
            ax.text(0.5, 0.5, 'No extension data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Extension Effectiveness')
            return
        
        effectiveness = self.effectiveness_analysis['extension_effectiveness']
        
        categories = ['With Extensions', 'Without Extensions']
        success_rates = [
            effectiveness['extended_success_rate'],
            effectiveness['non_extended_success_rate']
        ]
        
        colors = ['lightblue', 'lightgray']
        bars = ax.bar(categories, success_rates, color=colors, alpha=0.7)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate: Extensions vs No Extensions')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
        
        # Add benefit annotation
        benefit = effectiveness.get('extension_benefit', 0.0)
        ax.text(0.5, 0.9, f'Benefit: {benefit:+.2%}', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    def _save_detailed_analysis(self, result: AdaptiveComputationResult, output_dir: str):
        """Save detailed analysis results"""
        
        output_path = Path(output_dir)
        
        # Save main results
        results_file = output_path / "adaptive_computation_results.json"
        with open(results_file, 'w') as f:
            # Convert result to dictionary for JSON serialization
            result_dict = {
                'timestamp': result.timestamp,
                'problem_count': result.problem_count,
                'allocation_pattern_analysis': result.allocation_pattern_analysis,
                'effectiveness_analysis': result.effectiveness_analysis,
                'difficulty_assessment_analysis': result.difficulty_assessment_analysis,
                'resource_utilization_analysis': result.resource_utilization_analysis,
                'comparative_performance_analysis': result.comparative_performance_analysis,
                'computational_efficiency_metrics': result.computational_efficiency_metrics,
                'optimization_recommendations': result.optimization_recommendations
            }
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save individual allocation metrics
        metrics_file = output_path / "allocation_metrics.json"
        with open(metrics_file, 'w') as f:
            metrics_list = []
            for metrics in self.allocation_metrics:
                metrics_dict = {
                    'problem_id': metrics.problem_id,
                    'problem_type': metrics.problem_type,
                    'difficulty': metrics.difficulty,
                    'predicted_difficulty': metrics.predicted_difficulty,
                    'initial_allocation': metrics.initial_allocation,
                    'total_allocation': metrics.total_allocation,
                    'adaptive_extensions': metrics.adaptive_extensions,
                    'success': metrics.success,
                    'final_energy': metrics.final_energy,
                    'energy_improvement': metrics.energy_improvement,
                    'allocation_efficiency': metrics.allocation_efficiency,
                    'allocation_decision_quality': metrics.allocation_decision_quality,
                    'resource_utilization': metrics.resource_utilization,
                    'marginal_returns': metrics.marginal_returns
                }
                metrics_list.append(metrics_dict)
            json.dump(metrics_list, f, indent=2, default=str)
        
        print(f"Detailed adaptive computation analysis saved to {output_path}")


def main():
    """Command-line interface for adaptive computation analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Computation Effectiveness Analysis')
    parser.add_argument('--optimization-file', type=str, required=True,
                       help='JSON file containing optimization result data')
    parser.add_argument('--output-dir', type=str, default='./adaptive_computation_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--fixed-budget', type=int, default=200,
                       help='Fixed budget baseline for comparison')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Load optimization data
    try:
        with open(args.optimization_file, 'r') as f:
            optimization_data = json.load(f)
    except Exception as e:
        print(f"Failed to load optimization data: {str(e)}")
        return
    
    # Create configuration
    config = AdaptiveComputationConfig(
        fixed_budget_baseline=args.fixed_budget,
        generate_allocation_plots=not args.no_plots,
        generate_effectiveness_plots=not args.no_plots
    )
    
    # Run analysis
    analyzer = AdaptiveComputationAnalyzer(config)
    results = analyzer.analyze_adaptive_allocation(optimization_data, args.output_dir)
    
    # Print summary
    print("\nAdaptive Computation Analysis Summary:")
    print(f"Problems analyzed: {results.problem_count}")
    
    if results.effectiveness_analysis.get('extension_effectiveness'):
        ext_benefit = results.effectiveness_analysis['extension_effectiveness'].get('extension_benefit', 0.0)
        print(f"Extension benefit: {ext_benefit:+.1%}")
    
    if results.comparative_analysis.get('computational_efficiency_comparison'):
        eff_ratio = results.comparative_analysis['computational_efficiency_comparison'].get('efficiency_ratio', 0.0)
        print(f"Efficiency vs fixed budget: {eff_ratio:.2f}x")
    
    print(f"\nRecommendations ({len(results.optimization_recommendations)}):")
    for i, rec in enumerate(results.optimization_recommendations, 1):
        print(f"{i}. {rec}")


if __name__ == '__main__':
    main()