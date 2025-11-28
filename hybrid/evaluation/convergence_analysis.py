#!/usr/bin/env python3
"""
Convergence Behavior Analysis for IRED Sequence Optimization

This module analyzes the convergence behavior of IRED-style sequence optimization
across different problem types, providing insights into optimization effectiveness,
step efficiency, and convergence patterns.

Phase 4.3 Optimization Analysis - Convergence Component:

Key Analysis Areas:
1. Convergence Rate Analysis:
   - Time to convergence across problem types
   - Success vs failure patterns
   - Convergence quality assessment

2. Energy Landscape Progression:
   - Energy evolution through optimization landscapes
   - Landscape transition effectiveness
   - Energy variance and stability analysis

3. Step Efficiency Analysis:
   - Steps required for convergence by problem difficulty
   - Adaptive step allocation effectiveness
   - Resource utilization patterns

4. Trajectory Quality Metrics:
   - Monotonicity of energy improvements
   - Gradient coherence and direction consistency
   - Exploration vs exploitation balance

5. Failure Mode Analysis:
   - Convergence timeout patterns
   - Energy explosion detection
   - Oscillation and instability analysis

Features:
- Statistical analysis of optimization trajectories
- Visualization of convergence patterns
- Performance recommendations based on analysis
- Integration with main performance analysis framework
"""

import os
import sys
import json
import random
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
    from scipy.stats import spearmanr, pearsonr, kendalltau
    from scipy.signal import find_peaks
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
class ConvergenceAnalysisConfig:
    """
    Configuration for convergence behavior analysis.
    
    Analysis Settings:
        analyze_convergence_rates: Analyze time to convergence patterns
        analyze_energy_progression: Study energy evolution through landscapes
        analyze_step_efficiency: Analyze step allocation and efficiency
        analyze_trajectory_quality: Study trajectory characteristics
        analyze_failure_modes: Analyze convergence failures
        
    Statistical Settings:
        min_trajectory_length: Minimum trajectory length for analysis inclusion
        energy_convergence_threshold: Energy variance threshold for convergence detection
        gradient_analysis_window: Window size for gradient coherence analysis
        
    Visualization Settings:
        generate_trajectory_plots: Generate individual trajectory visualizations
        generate_summary_plots: Generate aggregate analysis plots
        max_trajectories_plot: Maximum number of trajectories to plot individually
    """
    # Analysis types
    analyze_convergence_rates: bool = True
    analyze_energy_progression: bool = True
    analyze_step_efficiency: bool = True
    analyze_trajectory_quality: bool = True
    analyze_failure_modes: bool = True
    
    # Statistical settings
    min_trajectory_length: int = 5
    energy_convergence_threshold: float = 1e-4
    gradient_analysis_window: int = 5
    step_efficiency_window: int = 10
    
    # Analysis parameters
    convergence_patience: int = 10
    energy_improvement_threshold: float = 1e-3
    oscillation_detection_threshold: float = 0.1
    
    # Visualization settings
    generate_trajectory_plots: bool = True
    generate_summary_plots: bool = True
    max_trajectories_plot: int = 20
    
    # Output settings
    save_detailed_analysis: bool = True
    include_statistical_tests: bool = True


@dataclass
class ConvergenceMetrics:
    """
    Metrics characterizing optimization convergence behavior.
    
    Attributes:
        trajectory_id: Unique identifier for this trajectory
        problem_type: Type of optimization problem
        difficulty: Problem difficulty level
        total_steps: Total optimization steps taken
        converged: Whether optimization converged successfully
        convergence_step: Step at which convergence was detected (None if not converged)
        final_energy: Final energy value achieved
        initial_energy: Initial energy value
        energy_improvement: Total energy improvement (initial - final)
        energy_variance: Energy variance in final convergence window
        step_efficiency: Energy improvement per step
        landscape_progression: Analysis of progression through energy landscapes
        gradient_metrics: Gradient-based convergence quality metrics
        failure_mode: Failure mode if optimization failed (None if successful)
    """
    trajectory_id: str
    problem_type: str
    difficulty: str
    total_steps: int
    converged: bool
    convergence_step: Optional[int]
    final_energy: float
    initial_energy: float
    energy_improvement: float
    energy_variance: float
    step_efficiency: float
    landscape_progression: Dict[str, Any]
    gradient_metrics: Dict[str, Any]
    failure_mode: Optional[str]


@dataclass
class ConvergenceAnalysisResult:
    """
    Results from convergence behavior analysis.
    
    Attributes:
        timestamp: Analysis timestamp
        config: Configuration used for analysis
        trajectory_count: Number of trajectories analyzed
        convergence_statistics: Overall convergence statistics
        convergence_rate_analysis: Analysis of convergence rates by problem type
        energy_progression_analysis: Energy landscape progression analysis
        step_efficiency_analysis: Step allocation and efficiency analysis
        trajectory_quality_analysis: Trajectory quality metrics analysis
        failure_mode_analysis: Analysis of convergence failure patterns
        performance_recommendations: Optimization tuning recommendations
    """
    timestamp: str
    config: ConvergenceAnalysisConfig
    trajectory_count: int
    convergence_statistics: Dict[str, Any]
    convergence_rate_analysis: Dict[str, Any]
    energy_progression_analysis: Dict[str, Any]
    step_efficiency_analysis: Dict[str, Any]
    trajectory_quality_analysis: Dict[str, Any]
    failure_mode_analysis: Dict[str, Any]
    performance_recommendations: List[str]


class ConvergenceAnalyzer:
    """
    Analyzer for IRED optimization convergence behavior.
    
    This class provides comprehensive analysis of optimization trajectories to understand:
    - How quickly and reliably optimization converges
    - Patterns in energy landscape progression
    - Efficiency of step allocation strategies
    - Quality characteristics of successful trajectories
    - Common failure modes and their causes
    
    Args:
        config: Convergence analysis configuration
        
    Example:
        >>> config = ConvergenceAnalysisConfig(
        ...     analyze_convergence_rates=True,
        ...     generate_summary_plots=True
        ... )
        >>> analyzer = ConvergenceAnalyzer(config)
        >>> results = analyzer.analyze_trajectories(trajectory_data)
        >>> print(f"Convergence rate: {results.convergence_statistics['overall_rate']:.2%}")
    """
    
    def __init__(self, config: ConvergenceAnalysisConfig):
        """Initialize convergence analyzer"""
        self.config = config
        self.trajectory_metrics = []
        
        # Analysis results storage
        self.convergence_stats = {}
        self.rate_analysis = {}
        self.progression_analysis = {}
        self.efficiency_analysis = {}
        self.quality_analysis = {}
        self.failure_analysis = {}
    
    def _create_empty_result(self) -> ConvergenceAnalysisResult:
        """Create empty result for cases with no trajectory data"""
        return ConvergenceAnalysisResult(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            trajectory_count=0,
            convergence_statistics={
                'total_trajectories_analyzed': 0,
                'overall_convergence_rate': 0.0,
                'mean_convergence_steps': 0.0,
                'mean_energy_improvement': 0.0,
                'mean_step_efficiency': 0.0
            },
            convergence_rate_analysis={},
            energy_progression_analysis={},
            step_efficiency_analysis={},
            trajectory_quality_analysis={},
            failure_mode_analysis={},
            performance_recommendations=["No trajectory data available for analysis"]
        )
    
    def analyze_trajectories(
        self, 
        trajectory_data: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> ConvergenceAnalysisResult:
        """
        Analyze optimization trajectories for convergence behavior.
        
        Args:
            trajectory_data: List of trajectory dictionaries from optimization runs
            output_dir: Directory to save analysis outputs (optional)
            
        Returns:
            ConvergenceAnalysisResult with comprehensive analysis
        """
        
        # Validate input
        if not isinstance(trajectory_data, list):
            raise TypeError(f"trajectory_data must be a list, got {type(trajectory_data)}")
        
        if len(trajectory_data) == 0:
            print("Warning: No trajectory data provided for analysis")
            # Return empty result
            return self._create_empty_result()
        
        print(f"Analyzing convergence behavior for {len(trajectory_data)} trajectories...")
        
        # Check for memory limits with large datasets
        if len(trajectory_data) > 10000:  # Arbitrary large dataset threshold
            print(f"Warning: Large dataset ({len(trajectory_data)} trajectories) may consume significant memory")
        
        # Extract convergence metrics from trajectories
        self._extract_convergence_metrics(trajectory_data)
        
        # Run analysis components based on configuration
        if self.config.analyze_convergence_rates:
            self._analyze_convergence_rates()
        
        if self.config.analyze_energy_progression:
            self._analyze_energy_progression()
        
        if self.config.analyze_step_efficiency:
            self._analyze_step_efficiency()
        
        if self.config.analyze_trajectory_quality:
            self._analyze_trajectory_quality()
        
        if self.config.analyze_failure_modes:
            self._analyze_failure_modes()
        
        # Generate overall convergence statistics
        self._calculate_convergence_statistics()
        
        # Generate performance recommendations
        recommendations = self._generate_performance_recommendations()
        
        # Create visualizations if requested
        if output_dir and PLOTTING_AVAILABLE:
            self._generate_visualizations(output_dir)
        
        # Compile results
        result = ConvergenceAnalysisResult(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            trajectory_count=len(self.trajectory_metrics),
            convergence_statistics=self.convergence_stats,
            convergence_rate_analysis=self.rate_analysis,
            energy_progression_analysis=self.progression_analysis,
            step_efficiency_analysis=self.efficiency_analysis,
            trajectory_quality_analysis=self.quality_analysis,
            failure_mode_analysis=self.failure_analysis,
            performance_recommendations=recommendations
        )
        
        # Save detailed analysis if requested
        if output_dir and self.config.save_detailed_analysis:
            self._save_detailed_analysis(result, output_dir)
        
        print(f"Convergence analysis complete. Overall convergence rate: {self.convergence_stats.get('overall_convergence_rate', 0.0):.2%}")
        
        return result
    
    def _extract_convergence_metrics(self, trajectory_data: List[Dict[str, Any]]):
        """Extract convergence metrics from optimization trajectories"""
        
        self.trajectory_metrics = []
        
        processed_count = 0
        skipped_count = 0
        
        for i, trajectory_info in enumerate(trajectory_data):
            try:
                # Validate trajectory info structure
                if not isinstance(trajectory_info, dict):
                    print(f"Warning: Trajectory {i} is not a dictionary, skipping")
                    skipped_count += 1
                    continue
                
                # Extract trajectory details with validation
                trajectory = trajectory_info.get('trajectory', [])
                if not isinstance(trajectory, list):
                    print(f"Warning: Trajectory {i} 'trajectory' field is not a list, skipping")
                    skipped_count += 1
                    continue
                
                problem_info = trajectory_info.get('problem_info', {})
                if not isinstance(problem_info, dict):
                    problem_info = {}  # Use default empty dict
                
                optimization_result = trajectory_info.get('optimization_result', {})
                if not isinstance(optimization_result, dict):
                    optimization_result = {}  # Use default empty dict
                
                # Skip trajectories that are too short
                if len(trajectory) < self.config.min_trajectory_length:
                    skipped_count += 1
                    continue
                
                # Extract basic information
                trajectory_id = f"traj_{i:04d}"
                problem_type = problem_info.get('type', 'unknown')
                difficulty = problem_info.get('difficulty', 'unknown')
                total_steps = len(trajectory)
                
                # Extract energy progression
                energies = [step.get('energy', float('inf')) for step in trajectory]
                initial_energy = energies[0] if energies else float('inf')
                final_energy = energies[-1] if energies else float('inf')
                
                # Check for valid energies
                if not all(np.isfinite(e) for e in energies):
                    failure_mode = 'invalid_energy'
                    converged = False
                    convergence_step = None
                else:
                    # Detect convergence
                    converged, convergence_step = self._detect_convergence(energies)
                    failure_mode = optimization_result.get('failure_reason') if not converged else None
                
                # Calculate metrics
                energy_improvement = initial_energy - final_energy if np.isfinite(initial_energy) and np.isfinite(final_energy) else 0.0
                energy_variance = self._calculate_energy_variance(energies)
                step_efficiency = energy_improvement / total_steps if total_steps > 0 else 0.0
                
                # Analyze landscape progression
                landscape_progression = self._analyze_landscape_progression(trajectory)
                
                # Calculate gradient metrics
                gradient_metrics = self._calculate_gradient_metrics(energies)
                
                # Create convergence metrics object
                metrics = ConvergenceMetrics(
                    trajectory_id=trajectory_id,
                    problem_type=problem_type,
                    difficulty=difficulty,
                    total_steps=total_steps,
                    converged=converged,
                    convergence_step=convergence_step,
                    final_energy=final_energy,
                    initial_energy=initial_energy,
                    energy_improvement=energy_improvement,
                    energy_variance=energy_variance,
                    step_efficiency=step_efficiency,
                    landscape_progression=landscape_progression,
                    gradient_metrics=gradient_metrics,
                    failure_mode=failure_mode
                )
                
                self.trajectory_metrics.append(metrics)
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to process trajectory {i}: {str(e)}")
                skipped_count += 1
                continue
        
        # Log processing statistics
        print(f"Trajectory processing complete: {processed_count} processed, {skipped_count} skipped")
    
    def _detect_convergence(self, energies: List[float]) -> Tuple[bool, Optional[int]]:
        """Detect if and when optimization converged"""
        
        if len(energies) < self.config.convergence_patience:
            return False, None
        
        # Look for stable energy region
        for i in range(self.config.convergence_patience, len(energies)):
            window_start = i - self.config.convergence_patience
            window_energies = energies[window_start:i]
            
            # Calculate energy variance in window
            if len(window_energies) > 1:
                energy_var = np.var(window_energies)
                energy_mean = np.mean(window_energies)
                
                # Use relative variance to handle different energy scales
                if abs(energy_mean) > 1e-6:
                    relative_var = energy_var / abs(energy_mean)
                else:
                    relative_var = energy_var
                
                if relative_var < self.config.energy_convergence_threshold:
                    return True, i
        
        return False, None
    
    def _calculate_energy_variance(self, energies: List[float]) -> float:
        """Calculate energy variance in final convergence window"""
        
        if len(energies) < 2:
            return 0.0
        
        # Use final window for variance calculation
        window_size = min(self.config.convergence_patience, len(energies))
        final_energies = energies[-window_size:]
        
        return np.var(final_energies) if len(final_energies) > 1 else 0.0
    
    def _analyze_landscape_progression(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze progression through energy landscapes"""
        
        landscape_analysis = {
            'landscapes_traversed': 0,
            'steps_per_landscape': [],
            'energy_improvement_per_landscape': [],
            'landscape_transition_quality': []
        }
        
        # Group trajectory steps by landscape
        landscape_groups = defaultdict(list)
        for step in trajectory:
            landscape_idx = step.get('landscape', 0)
            landscape_groups[landscape_idx].append(step)
        
        landscape_analysis['landscapes_traversed'] = len(landscape_groups)
        
        # Analyze each landscape
        for landscape_idx in sorted(landscape_groups.keys()):
            steps = landscape_groups[landscape_idx]
            landscape_analysis['steps_per_landscape'].append(len(steps))
            
            # Calculate energy improvement in this landscape
            if len(steps) >= 2:
                initial_energy = steps[0].get('energy', 0.0)
                final_energy = steps[-1].get('energy', 0.0)
                improvement = initial_energy - final_energy
                landscape_analysis['energy_improvement_per_landscape'].append(improvement)
            else:
                landscape_analysis['energy_improvement_per_landscape'].append(0.0)
        
        # Calculate transition quality (energy improvement consistency)
        improvements = landscape_analysis['energy_improvement_per_landscape']
        if len(improvements) > 1:
            # Positive improvements indicate good progression
            positive_improvements = [imp for imp in improvements if imp > 0]
            landscape_analysis['landscape_transition_quality'] = len(positive_improvements) / len(improvements)
        else:
            landscape_analysis['landscape_transition_quality'] = 1.0 if improvements and improvements[0] > 0 else 0.0
        
        return landscape_analysis
    
    def _calculate_gradient_metrics(self, energies: List[float]) -> Dict[str, Any]:
        """Calculate gradient-based convergence quality metrics"""
        
        gradient_metrics = {
            'energy_monotonicity': 0.0,
            'improvement_consistency': 0.0,
            'gradient_coherence': 0.0,
            'oscillation_measure': 0.0
        }
        
        if len(energies) < 2:
            return gradient_metrics
        
        # Calculate energy differences (pseudo-gradients)
        energy_diffs = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
        
        # Energy monotonicity: fraction of steps with energy decrease
        improving_steps = sum(1 for diff in energy_diffs if diff < 0)
        gradient_metrics['energy_monotonicity'] = improving_steps / len(energy_diffs) if energy_diffs else 0.0
        
        # Improvement consistency: variance of energy improvements
        improvements = [-diff for diff in energy_diffs if diff < 0]  # Positive improvements
        if len(improvements) > 1:
            improvement_std = np.std(improvements)
            improvement_mean = np.mean(improvements)
            # Lower coefficient of variation indicates more consistent improvement
            # Guard against division by zero and very small means
            if improvement_mean > 1e-8:  # Use stricter threshold
                coefficient_of_variation = improvement_std / improvement_mean
                gradient_metrics['improvement_consistency'] = max(0.0, 1.0 - coefficient_of_variation)
            else:
                gradient_metrics['improvement_consistency'] = 0.0
        
        # Gradient coherence: smoothness of energy progression  
        if len(energy_diffs) > self.config.gradient_analysis_window:
            windowed_coherence = []
            for i in range(len(energy_diffs) - self.config.gradient_analysis_window + 1):
                window = energy_diffs[i:i + self.config.gradient_analysis_window]
                # Coherence measured as consistency of direction
                negative_count = sum(1 for x in window if x < 0)
                coherence = max(negative_count, len(window) - negative_count) / len(window)
                windowed_coherence.append(coherence)
            gradient_metrics['gradient_coherence'] = np.mean(windowed_coherence)
        
        # Oscillation measure: detect energy oscillations
        if SCIPY_AVAILABLE and len(energies) > 10:
            # Find local maxima as oscillation indicator
            peaks, _ = find_peaks(energies, height=None)
            gradient_metrics['oscillation_measure'] = len(peaks) / len(energies)
        
        return gradient_metrics
    
    def _analyze_convergence_rates(self):
        """Analyze convergence rates by problem type and difficulty"""
        
        self.rate_analysis = {
            'overall_rate': 0.0,
            'by_problem_type': {},
            'by_difficulty': {},
            'by_type_and_difficulty': {},
            'convergence_step_statistics': {}
        }
        
        if not self.trajectory_metrics:
            return
        
        # Overall convergence rate
        converged_count = sum(1 for m in self.trajectory_metrics if m.converged)
        self.rate_analysis['overall_rate'] = converged_count / len(self.trajectory_metrics)
        
        # Group by problem type
        type_groups = defaultdict(list)
        for metrics in self.trajectory_metrics:
            type_groups[metrics.problem_type].append(metrics)
        
        for problem_type, metrics_list in type_groups.items():
            converged_in_type = sum(1 for m in metrics_list if m.converged)
            self.rate_analysis['by_problem_type'][problem_type] = {
                'convergence_rate': converged_in_type / len(metrics_list),
                'total_trajectories': len(metrics_list),
                'converged_trajectories': converged_in_type
            }
        
        # Group by difficulty
        difficulty_groups = defaultdict(list)
        for metrics in self.trajectory_metrics:
            difficulty_groups[metrics.difficulty].append(metrics)
        
        for difficulty, metrics_list in difficulty_groups.items():
            converged_in_difficulty = sum(1 for m in metrics_list if m.converged)
            self.rate_analysis['by_difficulty'][difficulty] = {
                'convergence_rate': converged_in_difficulty / len(metrics_list),
                'total_trajectories': len(metrics_list),
                'converged_trajectories': converged_in_difficulty
            }
        
        # Combined type and difficulty analysis
        for metrics in self.trajectory_metrics:
            key = f"{metrics.problem_type}_{metrics.difficulty}"
            if key not in self.rate_analysis['by_type_and_difficulty']:
                self.rate_analysis['by_type_and_difficulty'][key] = []
            self.rate_analysis['by_type_and_difficulty'][key].append(metrics.converged)
        
        # Calculate combined rates
        for key, converged_list in self.rate_analysis['by_type_and_difficulty'].items():
            rate = sum(converged_list) / len(converged_list)
            self.rate_analysis['by_type_and_difficulty'][key] = {
                'convergence_rate': rate,
                'total_trajectories': len(converged_list),
                'converged_trajectories': sum(converged_list)
            }
        
        # Convergence step statistics
        converged_steps = [m.convergence_step for m in self.trajectory_metrics if m.converged and m.convergence_step is not None]
        if converged_steps:
            self.rate_analysis['convergence_step_statistics'] = {
                'mean_steps_to_convergence': np.mean(converged_steps),
                'median_steps_to_convergence': np.median(converged_steps),
                'std_steps_to_convergence': np.std(converged_steps),
                'min_steps_to_convergence': min(converged_steps),
                'max_steps_to_convergence': max(converged_steps)
            }
    
    def _analyze_energy_progression(self):
        """Analyze energy landscape progression patterns"""
        
        self.progression_analysis = {
            'landscape_utilization': {},
            'energy_improvement_patterns': {},
            'landscape_transition_analysis': {}
        }
        
        if not self.trajectory_metrics:
            return
        
        # Analyze landscape utilization
        landscapes_traversed = [m.landscape_progression['landscapes_traversed'] for m in self.trajectory_metrics]
        self.progression_analysis['landscape_utilization'] = {
            'mean_landscapes_used': np.mean(landscapes_traversed),
            'median_landscapes_used': np.median(landscapes_traversed),
            'landscape_usage_distribution': {
                str(i): landscapes_traversed.count(i) for i in range(1, max(landscapes_traversed) + 1) if landscapes_traversed
            }
        }
        
        # Energy improvement patterns
        energy_improvements = [m.energy_improvement for m in self.trajectory_metrics if np.isfinite(m.energy_improvement)]
        if energy_improvements:
            self.progression_analysis['energy_improvement_patterns'] = {
                'mean_total_improvement': np.mean(energy_improvements),
                'median_total_improvement': np.median(energy_improvements),
                'std_total_improvement': np.std(energy_improvements),
                'improvement_success_correlation': self._calculate_improvement_success_correlation()
            }
        
        # Landscape transition quality analysis
        transition_qualities = [m.landscape_progression['landscape_transition_quality'] for m in self.trajectory_metrics]
        self.progression_analysis['landscape_transition_analysis'] = {
            'mean_transition_quality': np.mean(transition_qualities),
            'median_transition_quality': np.median(transition_qualities),
            'poor_transitions_fraction': sum(1 for q in transition_qualities if q < 0.5) / len(transition_qualities)
        }
    
    def _calculate_improvement_success_correlation(self) -> float:
        """Calculate correlation between energy improvement and convergence success"""
        
        improvements = []
        successes = []
        
        for metrics in self.trajectory_metrics:
            if np.isfinite(metrics.energy_improvement):
                improvements.append(metrics.energy_improvement)
                successes.append(1 if metrics.converged else 0)
        
        if len(improvements) > 1 and SCIPY_AVAILABLE:
            correlation, _ = spearmanr(improvements, successes)
            return correlation if np.isfinite(correlation) else 0.0
        
        return 0.0
    
    def _analyze_step_efficiency(self):
        """Analyze step allocation and efficiency patterns"""
        
        self.efficiency_analysis = {
            'step_efficiency_statistics': {},
            'efficiency_by_problem_type': {},
            'efficiency_vs_convergence': {},
            'resource_utilization_analysis': {}
        }
        
        if not self.trajectory_metrics:
            return
        
        # Overall step efficiency statistics
        step_efficiencies = [m.step_efficiency for m in self.trajectory_metrics if np.isfinite(m.step_efficiency)]
        if step_efficiencies:
            self.efficiency_analysis['step_efficiency_statistics'] = {
                'mean_step_efficiency': np.mean(step_efficiencies),
                'median_step_efficiency': np.median(step_efficiencies),
                'std_step_efficiency': np.std(step_efficiencies),
                'efficiency_range': [min(step_efficiencies), max(step_efficiencies)]
            }
        
        # Efficiency by problem type
        type_groups = defaultdict(list)
        for metrics in self.trajectory_metrics:
            if np.isfinite(metrics.step_efficiency):
                type_groups[metrics.problem_type].append(metrics.step_efficiency)
        
        for problem_type, efficiencies in type_groups.items():
            self.efficiency_analysis['efficiency_by_problem_type'][problem_type] = {
                'mean_efficiency': np.mean(efficiencies),
                'median_efficiency': np.median(efficiencies),
                'efficiency_count': len(efficiencies)
            }
        
        # Efficiency vs convergence correlation
        efficiencies = []
        convergence_success = []
        
        for metrics in self.trajectory_metrics:
            if np.isfinite(metrics.step_efficiency):
                efficiencies.append(metrics.step_efficiency)
                convergence_success.append(1 if metrics.converged else 0)
        
        if len(efficiencies) > 1 and SCIPY_AVAILABLE:
            correlation, p_value = spearmanr(efficiencies, convergence_success)
            self.efficiency_analysis['efficiency_vs_convergence'] = {
                'correlation': correlation if np.isfinite(correlation) else 0.0,
                'p_value': p_value if np.isfinite(p_value) else 1.0,
                'significant': p_value < 0.05 if np.isfinite(p_value) else False
            }
        
        # Resource utilization analysis  
        total_steps = [m.total_steps for m in self.trajectory_metrics]
        converged_steps = [m.total_steps for m in self.trajectory_metrics if m.converged]
        
        self.efficiency_analysis['resource_utilization_analysis'] = {
            'mean_steps_all': np.mean(total_steps),
            'mean_steps_converged': np.mean(converged_steps) if converged_steps else 0.0,
            'step_overhead_unconverged': (np.mean(total_steps) - np.mean(converged_steps)) if converged_steps else 0.0
        }
    
    def _analyze_trajectory_quality(self):
        """Analyze trajectory quality characteristics"""
        
        self.quality_analysis = {
            'monotonicity_analysis': {},
            'gradient_coherence_analysis': {},
            'oscillation_analysis': {},
            'quality_success_correlation': {}
        }
        
        if not self.trajectory_metrics:
            return
        
        # Monotonicity analysis
        monotonicities = [m.gradient_metrics['energy_monotonicity'] for m in self.trajectory_metrics]
        self.quality_analysis['monotonicity_analysis'] = {
            'mean_monotonicity': np.mean(monotonicities),
            'median_monotonicity': np.median(monotonicities),
            'high_monotonicity_fraction': sum(1 for m in monotonicities if m > 0.8) / len(monotonicities)
        }
        
        # Gradient coherence analysis
        coherences = [m.gradient_metrics['gradient_coherence'] for m in self.trajectory_metrics]
        self.quality_analysis['gradient_coherence_analysis'] = {
            'mean_coherence': np.mean(coherences),
            'median_coherence': np.median(coherences),
            'high_coherence_fraction': sum(1 for c in coherences if c > 0.7) / len(coherences)
        }
        
        # Oscillation analysis
        oscillations = [m.gradient_metrics['oscillation_measure'] for m in self.trajectory_metrics]
        self.quality_analysis['oscillation_analysis'] = {
            'mean_oscillation': np.mean(oscillations),
            'median_oscillation': np.median(oscillations),
            'high_oscillation_fraction': sum(1 for o in oscillations if o > 0.2) / len(oscillations)
        }
        
        # Quality vs success correlation
        if SCIPY_AVAILABLE:
            qualities = [m.gradient_metrics['energy_monotonicity'] for m in self.trajectory_metrics]
            successes = [1 if m.converged else 0 for m in self.trajectory_metrics]
            
            if len(qualities) > 1:
                correlation, p_value = spearmanr(qualities, successes)
                self.quality_analysis['quality_success_correlation'] = {
                    'monotonicity_success_correlation': correlation if np.isfinite(correlation) else 0.0,
                    'p_value': p_value if np.isfinite(p_value) else 1.0
                }
    
    def _analyze_failure_modes(self):
        """Analyze convergence failure patterns"""
        
        self.failure_analysis = {
            'failure_rate': 0.0,
            'failure_mode_distribution': {},
            'failure_patterns_by_type': {},
            'failure_step_analysis': {}
        }
        
        if not self.trajectory_metrics:
            return
        
        # Overall failure rate
        failed_count = sum(1 for m in self.trajectory_metrics if not m.converged)
        self.failure_analysis['failure_rate'] = failed_count / len(self.trajectory_metrics)
        
        # Failure mode distribution
        failure_modes = defaultdict(int)
        for metrics in self.trajectory_metrics:
            if not metrics.converged:
                mode = metrics.failure_mode or 'unknown'
                failure_modes[mode] += 1
        
        total_failures = sum(failure_modes.values())
        if total_failures > 0:
            self.failure_analysis['failure_mode_distribution'] = {
                mode: count / total_failures for mode, count in failure_modes.items()
            }
        
        # Failure patterns by problem type
        type_failures = defaultdict(lambda: defaultdict(int))
        for metrics in self.trajectory_metrics:
            if not metrics.converged:
                mode = metrics.failure_mode or 'unknown'
                type_failures[metrics.problem_type][mode] += 1
        
        self.failure_analysis['failure_patterns_by_type'] = dict(type_failures)
        
        # Failure step analysis
        failed_steps = [m.total_steps for m in self.trajectory_metrics if not m.converged]
        if failed_steps:
            self.failure_analysis['failure_step_analysis'] = {
                'mean_steps_before_failure': np.mean(failed_steps),
                'median_steps_before_failure': np.median(failed_steps),
                'early_failure_fraction': sum(1 for s in failed_steps if s < 10) / len(failed_steps)
            }
    
    def _calculate_convergence_statistics(self):
        """Calculate overall convergence statistics"""
        
        self.convergence_stats = {
            'total_trajectories_analyzed': len(self.trajectory_metrics),
            'overall_convergence_rate': 0.0,
            'mean_convergence_steps': 0.0,
            'mean_energy_improvement': 0.0,
            'mean_step_efficiency': 0.0
        }
        
        if not self.trajectory_metrics:
            return
        
        # Overall convergence rate
        converged_count = sum(1 for m in self.trajectory_metrics if m.converged)
        self.convergence_stats['overall_convergence_rate'] = converged_count / len(self.trajectory_metrics)
        
        # Mean convergence steps (for converged trajectories only)
        converged_steps = [m.convergence_step for m in self.trajectory_metrics if m.converged and m.convergence_step is not None]
        self.convergence_stats['mean_convergence_steps'] = np.mean(converged_steps) if converged_steps else 0.0
        
        # Mean energy improvement
        improvements = [m.energy_improvement for m in self.trajectory_metrics if np.isfinite(m.energy_improvement)]
        self.convergence_stats['mean_energy_improvement'] = np.mean(improvements) if improvements else 0.0
        
        # Mean step efficiency
        efficiencies = [m.step_efficiency for m in self.trajectory_metrics if np.isfinite(m.step_efficiency)]
        self.convergence_stats['mean_step_efficiency'] = np.mean(efficiencies) if efficiencies else 0.0
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate optimization tuning recommendations based on analysis"""
        
        recommendations = []
        
        if not self.trajectory_metrics:
            recommendations.append("No trajectory data available for analysis. Ensure optimization trajectories are properly logged.")
            return recommendations
        
        # Convergence rate recommendations
        convergence_rate = self.convergence_stats.get('overall_convergence_rate', 0.0)
        if convergence_rate < 0.5:
            recommendations.append(
                f"Low convergence rate ({convergence_rate:.1%}). Consider increasing max_steps_per_landscape "
                "or adjusting convergence criteria for better success rates."
            )
        
        # Step efficiency recommendations
        if 'step_efficiency_statistics' in self.efficiency_analysis:
            mean_efficiency = self.efficiency_analysis['step_efficiency_statistics'].get('mean_step_efficiency', 0.0)
            if mean_efficiency < 0.01:  # Arbitrary threshold for low efficiency
                recommendations.append(
                    f"Low step efficiency ({mean_efficiency:.4f}). Consider adjusting learning rate "
                    "or implementing adaptive step sizing."
                )
        
        # Monotonicity recommendations
        if 'monotonicity_analysis' in self.quality_analysis:
            mean_monotonicity = self.quality_analysis['monotonicity_analysis'].get('mean_monotonicity', 0.0)
            if mean_monotonicity < 0.6:
                recommendations.append(
                    f"Low energy monotonicity ({mean_monotonicity:.2f}). Consider adding gradient clipping "
                    "or reducing learning rate to improve optimization stability."
                )
        
        # Oscillation recommendations
        if 'oscillation_analysis' in self.quality_analysis:
            high_oscillation_fraction = self.quality_analysis['oscillation_analysis'].get('high_oscillation_fraction', 0.0)
            if high_oscillation_fraction > 0.3:
                recommendations.append(
                    f"High oscillation detected in {high_oscillation_fraction:.1%} of trajectories. "
                    "Consider reducing learning rate or implementing momentum-based optimization."
                )
        
        # Failure mode recommendations
        if 'failure_mode_distribution' in self.failure_analysis:
            failure_modes = self.failure_analysis['failure_mode_distribution']
            
            if failure_modes.get('convergence_timeout', 0.0) > 0.3:
                recommendations.append(
                    f"Frequent convergence timeouts ({failure_modes['convergence_timeout']:.1%}). "
                    "Consider implementing adaptive step allocation or increasing step limits for difficult problems."
                )
            
            if failure_modes.get('energy_explosion', 0.0) > 0.1:
                recommendations.append(
                    f"Energy explosions detected ({failure_modes.get('energy_explosion', 0.0):.1%}). "
                    "Implement stronger gradient clipping and learning rate scheduling."
                )
        
        # Landscape utilization recommendations
        if 'landscape_utilization' in self.progression_analysis:
            mean_landscapes = self.progression_analysis['landscape_utilization'].get('mean_landscapes_used', 0.0)
            if mean_landscapes < 3:
                recommendations.append(
                    f"Low landscape utilization (mean: {mean_landscapes:.1f}). "
                    "Consider adjusting landscape progression criteria or increasing landscape count."
                )
        
        if not recommendations:
            recommendations.append(
                "Convergence analysis shows good optimization performance. Consider fine-tuning "
                "parameters for marginal improvements based on specific problem requirements."
            )
        
        return recommendations
    
    def _generate_visualizations(self, output_dir: str):
        """Generate convergence analysis visualizations"""
        
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping visualizations.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating convergence analysis visualizations...")
        
        try:
            # Create multi-panel summary plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Convergence rates by problem type
            self._plot_convergence_rates_by_type(axes[0, 0])
            
            # 2. Convergence rates by difficulty
            self._plot_convergence_rates_by_difficulty(axes[0, 1])
            
            # 3. Steps to convergence distribution
            self._plot_convergence_steps_distribution(axes[0, 2])
            
            # 4. Energy improvement vs steps
            self._plot_energy_improvement_vs_steps(axes[1, 0])
            
            # 5. Trajectory quality metrics
            self._plot_trajectory_quality_metrics(axes[1, 1])
            
            # 6. Failure mode distribution
            self._plot_failure_mode_distribution(axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(output_path / "convergence_analysis_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate individual trajectory plots if requested
            if self.config.generate_trajectory_plots:
                self._plot_individual_trajectories(output_path)
            
            print(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            print(f"Failed to generate visualizations: {str(e)}")
    
    def _plot_convergence_rates_by_type(self, ax):
        """Plot convergence rates by problem type"""
        
        if 'by_problem_type' not in self.rate_analysis:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Rates by Problem Type')
            return
        
        types = []
        rates = []
        
        for problem_type, data in self.rate_analysis['by_problem_type'].items():
            types.append(problem_type.replace('_', ' ').title())
            rates.append(data['convergence_rate'])
        
        if types and rates:
            bars = ax.bar(types, rates, alpha=0.7, color='skyblue')
            ax.set_ylabel('Convergence Rate')
            ax.set_title('Convergence Rates by Problem Type')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_convergence_rates_by_difficulty(self, ax):
        """Plot convergence rates by problem difficulty"""
        
        if 'by_difficulty' not in self.rate_analysis:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Rates by Difficulty')
            return
        
        difficulties = []
        rates = []
        
        for difficulty, data in self.rate_analysis['by_difficulty'].items():
            difficulties.append(difficulty.title())
            rates.append(data['convergence_rate'])
        
        if difficulties and rates:
            colors = {'Easy': 'lightgreen', 'Medium': 'orange', 'Hard': 'lightcoral'}
            bar_colors = [colors.get(d, 'gray') for d in difficulties]
            
            bars = ax.bar(difficulties, rates, alpha=0.7, color=bar_colors)
            ax.set_ylabel('Convergence Rate')
            ax.set_title('Convergence Rates by Difficulty')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
    
    def _plot_convergence_steps_distribution(self, ax):
        """Plot distribution of steps to convergence"""
        
        converged_steps = [m.convergence_step for m in self.trajectory_metrics 
                          if m.converged and m.convergence_step is not None]
        
        if converged_steps:
            ax.hist(converged_steps, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax.set_xlabel('Steps to Convergence')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Steps to Convergence')
            
            # Add mean line
            mean_steps = np.mean(converged_steps)
            ax.axvline(mean_steps, color='red', linestyle='--', label=f'Mean: {mean_steps:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No converged trajectories', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Steps to Convergence')
    
    def _plot_energy_improvement_vs_steps(self, ax):
        """Plot energy improvement vs total steps"""
        
        steps = []
        improvements = []
        colors = []
        
        for metrics in self.trajectory_metrics:
            if np.isfinite(metrics.energy_improvement):
                steps.append(metrics.total_steps)
                improvements.append(metrics.energy_improvement)
                colors.append('green' if metrics.converged else 'red')
        
        if steps and improvements:
            ax.scatter(steps, improvements, c=colors, alpha=0.6)
            ax.set_xlabel('Total Steps')
            ax.set_ylabel('Energy Improvement')
            ax.set_title('Energy Improvement vs Steps')
            
            # Add trend line if scipy available
            if SCIPY_AVAILABLE and len(steps) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(steps, improvements)
                x_trend = np.array([min(steps), max(steps)])
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, 'b--', alpha=0.8, 
                       label=f'Trend (R²={r_value**2:.3f})')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Energy Improvement vs Steps')
    
    def _plot_trajectory_quality_metrics(self, ax):
        """Plot trajectory quality metrics"""
        
        if not self.trajectory_metrics:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trajectory Quality Metrics')
            return
        
        metrics = ['Monotonicity', 'Coherence', 'Consistency']
        values = []
        
        # Extract quality metrics
        monotonicities = [m.gradient_metrics['energy_monotonicity'] for m in self.trajectory_metrics]
        coherences = [m.gradient_metrics['gradient_coherence'] for m in self.trajectory_metrics]
        consistencies = [m.gradient_metrics['improvement_consistency'] for m in self.trajectory_metrics]
        
        values.append(np.mean(monotonicities))
        values.append(np.mean(coherences))
        values.append(np.mean(consistencies))
        
        bars = ax.bar(metrics, values, alpha=0.7, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('Mean Quality Score')
        ax.set_title('Trajectory Quality Metrics')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_failure_mode_distribution(self, ax):
        """Plot distribution of failure modes"""
        
        if 'failure_mode_distribution' not in self.failure_analysis:
            ax.text(0.5, 0.5, 'No failure data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Failure Mode Distribution')
            return
        
        failure_modes = self.failure_analysis['failure_mode_distribution']
        
        if failure_modes:
            modes = [mode.replace('_', ' ').title() for mode in failure_modes.keys()]
            frequencies = list(failure_modes.values())
            
            ax.pie(frequencies, labels=modes, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Failure Modes')
        else:
            ax.text(0.5, 0.5, 'No failures detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Failure Modes')
    
    def _plot_individual_trajectories(self, output_path: Path):
        """
        Plot individual optimization trajectories.
        
        Creates trajectory summaries and energy progression plots for selected trajectories.
        If step-by-step energy data is not available, plots energy improvements and metrics.
        """
        
        # Select subset of trajectories for plotting
        plot_count = min(self.config.max_trajectories_plot, len(self.trajectory_metrics))
        
        if plot_count == 0:
            return
        
        print(f"Plotting {plot_count} individual trajectories...")
        
        # Create individual trajectory plots
        trajectories_dir = output_path / "individual_trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        # Select diverse set of trajectories for plotting
        selected_trajectories = self._select_representative_trajectories(plot_count)
        
        # Create individual plots for each selected trajectory
        for i, metrics in enumerate(selected_trajectories):
            try:
                self._plot_single_trajectory(metrics, trajectories_dir, i)
            except Exception as e:
                print(f"Failed to plot trajectory {metrics.trajectory_id}: {e}")
                continue
        
        print(f"Individual trajectory plots saved to {trajectories_dir}")
    
    def _select_representative_trajectories(self, count: int) -> List[ConvergenceMetrics]:
        """Select representative trajectories for plotting"""
        if len(self.trajectory_metrics) <= count:
            return self.trajectory_metrics
        
        # Categorize trajectories
        converged = [m for m in self.trajectory_metrics if m.converged]
        failed = [m for m in self.trajectory_metrics if not m.converged]
        
        # Select mix of converged and failed trajectories
        selected = []
        
        # Include some converged trajectories (best and worst performers)
        if converged:
            converged_sorted = sorted(converged, key=lambda x: x.step_efficiency, reverse=True)
            selected.extend(converged_sorted[:min(count//2, len(converged))])
        
        # Include some failed trajectories
        if failed and len(selected) < count:
            remaining = count - len(selected)
            selected.extend(failed[:remaining])
        
        # Fill remaining slots with random selection
        if len(selected) < count:
            remaining_metrics = [m for m in self.trajectory_metrics if m not in selected]
            random.shuffle(remaining_metrics)
            selected.extend(remaining_metrics[:count - len(selected)])
        
        return selected[:count]
    
    def _plot_single_trajectory(self, metrics: ConvergenceMetrics, output_dir: Path, index: int):
        """Plot analysis for a single trajectory"""
        
        # Create figure with subplots for different aspects
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Trajectory {metrics.trajectory_id} - {metrics.problem_type}', fontsize=14)
        
        # Plot 1: Energy progression summary
        self._plot_energy_summary(ax1, metrics)
        
        # Plot 2: Convergence metrics
        self._plot_convergence_metrics(ax2, metrics) 
        
        # Plot 3: Landscape progression
        self._plot_landscape_progression(ax3, metrics)
        
        # Plot 4: Quality metrics
        self._plot_quality_metrics(ax4, metrics)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"trajectory_{index:03d}_{metrics.trajectory_id.replace('/', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_summary(self, ax, metrics: ConvergenceMetrics):
        """Plot energy improvement summary for a trajectory"""
        
        # Create bar plot showing initial, final, and improvement
        categories = ['Initial', 'Final', 'Improvement']
        values = [metrics.initial_energy, metrics.final_energy, metrics.energy_improvement]
        colors = ['lightblue', 'lightgreen', 'orange']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            # Calculate text offset based on value range
            value_range = max(values) - min(values) if values else 1.0
            offset = max(0.01, value_range * 0.02)  # At least 0.01, or 2% of range
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Energy')
        ax.set_title('Energy Summary')
        
        # Add convergence status
        status = "Converged" if metrics.converged else f"Failed ({metrics.failure_mode})"
        ax.text(0.5, 0.95, status, transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='lightgreen' if metrics.converged else 'lightcoral'))
    
    def _plot_convergence_metrics(self, ax, metrics: ConvergenceMetrics):
        """Plot convergence-related metrics for a trajectory"""
        
        metric_names = ['Step Efficiency', 'Energy Variance', 'Convergence Step']
        metric_values = [
            metrics.step_efficiency,
            metrics.energy_variance,
            (metrics.convergence_step / metrics.total_steps) if (metrics.convergence_step and metrics.total_steps > 0) else 0
        ]
        
        # Normalize values for visualization (0-1 scale)
        normalized_values = []
        for i, (name, value) in enumerate(zip(metric_names, metric_values)):
            if i == 2:  # Convergence step ratio is already 0-1
                normalized_values.append(value)
            else:
                # Use simple scaling for other metrics
                normalized_values.append(min(abs(value), 1.0))
        
        bars = ax.barh(metric_names, normalized_values, alpha=0.7,
                      color=['skyblue', 'lightgreen', 'orange'])
        
        # Add value labels
        for bar, orig_value in zip(bars, metric_values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{orig_value:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Normalized Value')
        ax.set_title('Convergence Metrics')
        ax.set_xlim(0, 1.2)
    
    def _plot_landscape_progression(self, ax, metrics: ConvergenceMetrics):
        """Plot landscape progression analysis for a trajectory"""
        
        if not metrics.landscape_progression:
            ax.text(0.5, 0.5, 'No landscape data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Landscape Progression')
            return
        
        # Extract landscape usage data if available
        landscapes_used = metrics.landscape_progression.get('landscapes_used', [])
        steps_per_landscape = metrics.landscape_progression.get('steps_per_landscape', [])
        
        if landscapes_used and steps_per_landscape:
            # Plot steps spent in each landscape
            ax.pie(steps_per_landscape, labels=[f'L{i}' for i in landscapes_used],
                  autopct='%1.1f%%', alpha=0.7)
            ax.set_title('Time in Each Landscape')
        else:
            # Fallback: show total landscapes used
            total_landscapes = metrics.landscape_progression.get('total_landscapes_used', 0)
            ax.bar(['Landscapes Used'], [total_landscapes], alpha=0.7, color='lightblue')
            ax.set_ylabel('Count')
            ax.set_title('Landscape Usage Summary')
    
    def _plot_quality_metrics(self, ax, metrics: ConvergenceMetrics):
        """Plot trajectory quality metrics"""
        
        gradient_metrics = metrics.gradient_metrics
        
        quality_names = ['Monotonicity', 'Coherence', 'Consistency', 'Oscillation']
        quality_values = [
            gradient_metrics.get('energy_monotonicity', 0.0),
            gradient_metrics.get('gradient_coherence', 0.0),
            gradient_metrics.get('improvement_consistency', 0.0),
            1.0 - gradient_metrics.get('oscillation_measure', 0.0)  # Invert oscillation
        ]
        
        # Create radar-style plot (simplified as bar plot)
        bars = ax.bar(quality_names, quality_values, alpha=0.7,
                     color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
        
        # Add value labels
        for bar, value in zip(bars, quality_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Quality Score')
        ax.set_title('Trajectory Quality Metrics')
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis='x', rotation=45)
    
    def _save_detailed_analysis(self, result: ConvergenceAnalysisResult, output_dir: str):
        """Save detailed analysis results"""
        
        output_path = Path(output_dir)
        
        # Save main results
        results_file = output_path / "convergence_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert result to dictionary for JSON serialization
            result_dict = {
                'timestamp': result.timestamp,
                'trajectory_count': result.trajectory_count,
                'convergence_statistics': result.convergence_statistics,
                'convergence_rate_analysis': result.convergence_rate_analysis,
                'energy_progression_analysis': result.energy_progression_analysis,
                'step_efficiency_analysis': result.step_efficiency_analysis,
                'trajectory_quality_analysis': result.trajectory_quality_analysis,
                'failure_mode_analysis': result.failure_mode_analysis,
                'performance_recommendations': result.performance_recommendations
            }
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save individual trajectory metrics if requested
        metrics_file = output_path / "trajectory_metrics.json"
        with open(metrics_file, 'w') as f:
            metrics_list = []
            for metrics in self.trajectory_metrics:
                metrics_dict = {
                    'trajectory_id': metrics.trajectory_id,
                    'problem_type': metrics.problem_type,
                    'difficulty': metrics.difficulty,
                    'total_steps': metrics.total_steps,
                    'converged': metrics.converged,
                    'convergence_step': metrics.convergence_step,
                    'final_energy': metrics.final_energy,
                    'initial_energy': metrics.initial_energy,
                    'energy_improvement': metrics.energy_improvement,
                    'energy_variance': metrics.energy_variance,
                    'step_efficiency': metrics.step_efficiency,
                    'landscape_progression': metrics.landscape_progression,
                    'gradient_metrics': metrics.gradient_metrics,
                    'failure_mode': metrics.failure_mode
                }
                metrics_list.append(metrics_dict)
            json.dump(metrics_list, f, indent=2, default=str)
        
        print(f"Detailed analysis saved to {output_path}")


def main():
    """Command-line interface for convergence analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convergence Behavior Analysis for IRED Optimization')
    parser.add_argument('--trajectory-file', type=str, required=True, 
                       help='JSON file containing trajectory data')
    parser.add_argument('--output-dir', type=str, default='./convergence_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--max-trajectories-plot', type=int, default=20,
                       help='Maximum number of individual trajectories to plot')
    
    args = parser.parse_args()
    
    # Load trajectory data
    try:
        with open(args.trajectory_file, 'r') as f:
            trajectory_data = json.load(f)
    except Exception as e:
        print(f"Failed to load trajectory data: {str(e)}")
        return
    
    # Create configuration
    config = ConvergenceAnalysisConfig(
        generate_trajectory_plots=not args.no_plots,
        generate_summary_plots=not args.no_plots,
        max_trajectories_plot=args.max_trajectories_plot
    )
    
    # Run analysis
    analyzer = ConvergenceAnalyzer(config)
    results = analyzer.analyze_trajectories(trajectory_data, args.output_dir)
    
    # Print summary
    print("\nConvergence Analysis Summary:")
    print(f"Total trajectories analyzed: {results.trajectory_count}")
    print(f"Overall convergence rate: {results.convergence_statistics['overall_convergence_rate']:.2%}")
    print(f"Mean steps to convergence: {results.convergence_statistics['mean_convergence_steps']:.1f}")
    print(f"Mean energy improvement: {results.convergence_statistics['mean_energy_improvement']:.4f}")
    
    print(f"\nRecommendations ({len(results.performance_recommendations)}):")
    for i, rec in enumerate(results.performance_recommendations, 1):
        print(f"{i}. {rec}")


if __name__ == '__main__':
    main()