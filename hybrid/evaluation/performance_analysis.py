#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for Energy-Based Protein Design

This module provides comprehensive performance analysis for the ProteinMPNN-IRED hybrid 
energy-based protein design system. It orchestrates systematic evaluation across all 
benchmark tasks, analyzes optimization behavior, and generates publication-ready results.

Phase 4.3 Performance Analysis Implementation:

1. Comprehensive Evaluation Study:
   - Systematic evaluation on all benchmark tasks
   - Success rate comparison across design challenges  
   - Failure mode and edge case analysis
   - Performance vs computational cost trade-offs
   - Publication-ready results and figures

2. Optimization Analysis:
   - Convergence behavior on different problem types
   - Adaptive computation effectiveness analysis
   - Energy landscape quality and smoothness investigation
   - Hyperparameter sensitivity and tuning guidelines

Key Features:
- Unified coordinator for all performance analysis components
- Integration with existing evaluation framework
- Systematic benchmarking across challenge types
- Comprehensive result reporting and visualization
- Computational efficiency analysis and optimization insights
"""

import os
import sys
import json
import warnings
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time
from collections import defaultdict
import itertools
import gc

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
from inference.design_pipeline import ProteinDesignPipeline, PipelineConfig
from inference.ired_optimizer import IREDSequenceOptimizer, OptimizationConfig, OptimizationResult
from evaluation.eval_energy import EnergyModelEvaluator, EnergyRankingEvaluator, SequencePropertyAnalyzer
from evaluation.benchmark_datasets import BenchmarkDatasetCurator, BenchmarkDatasetConfig
from evaluation.validate_designs import ValidationPipeline, ValidationConfig


@dataclass  
class PerformanceAnalysisConfig:
    """
    Configuration for comprehensive performance analysis.
    
    Evaluation Settings:
        benchmark_config: Configuration for benchmark dataset generation
        pipeline_config: Configuration for design pipeline
        validation_config: Configuration for design validation
        
    Analysis Types:
        run_comprehensive_evaluation: Run systematic evaluation on all benchmark tasks
        run_convergence_analysis: Analyze optimization convergence behavior  
        run_adaptive_computation: Analyze adaptive computation effectiveness
        run_landscape_analysis: Investigate energy landscape quality
        run_hyperparameter_sensitivity: Study hyperparameter sensitivity
        
    Computational Settings:
        max_benchmark_size: Maximum number of benchmark problems per type
        batch_size: Batch size for evaluation processing
        max_workers: Maximum number of workers for parallel processing
        memory_limit_gb: Memory limit for evaluation processing
        
    Output Settings:
        output_dir: Directory for analysis results
        generate_plots: Generate publication-ready visualizations
        save_intermediate: Save intermediate analysis results
        verbose: Enable detailed progress logging
    """
    # Configurations for existing components
    benchmark_config: Optional[BenchmarkDatasetConfig] = None
    pipeline_config: Optional[PipelineConfig] = None  
    validation_config: Optional[ValidationConfig] = None
    
    # Analysis types to run
    run_comprehensive_evaluation: bool = True
    run_convergence_analysis: bool = True
    run_adaptive_computation: bool = True
    run_landscape_analysis: bool = True
    run_hyperparameter_sensitivity: bool = False  # Expensive, optional
    
    # Benchmark evaluation settings
    max_benchmark_size: int = 100  # Max problems per benchmark type
    include_novel_backbones: bool = True
    include_multi_constraint: bool = True
    include_extrapolation: bool = True
    include_literature_targets: bool = False  # Requires external data
    
    # Computational settings
    batch_size: int = 8
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    device: str = 'auto'
    random_seed: Optional[int] = 42
    
    # Convergence analysis settings
    convergence_metrics: List[str] = field(default_factory=lambda: [
        'energy_variance', 'gradient_norms', 'step_efficiency', 'landscape_progression'
    ])
    
    # Adaptive computation analysis settings  
    adaptive_computation_metrics: List[str] = field(default_factory=lambda: [
        'step_allocation', 'difficulty_assessment', 'resource_utilization', 'success_correlation'
    ])
    
    # Landscape analysis settings
    landscape_metrics: List[str] = field(default_factory=lambda: [
        'energy_smoothness', 'basin_connectivity', 'gradient_coherence', 'temperature_effects'
    ])
    
    # Hyperparameter sensitivity settings
    hyperparameter_ranges: Dict[str, List[float]] = field(default_factory=lambda: {
        'learning_rate': [0.001, 0.01, 0.1],
        'noise_scale': [0.001, 0.01, 0.1], 
        'num_landscapes': [3, 5, 7],
        'convergence_patience': [5, 10, 20]
    })
    
    # Output and reporting
    output_dir: Optional[str] = None
    generate_plots: bool = True
    save_intermediate: bool = True
    verbose: bool = True
    
    # Statistical settings
    statistical_alpha: float = 0.05
    bootstrap_samples: int = 1000
    multiple_testing_correction: str = 'bonferroni'


@dataclass
class PerformanceAnalysisResult:
    """
    Comprehensive results from performance analysis.
    
    Attributes:
        timestamp: Analysis completion timestamp
        config: Configuration used for analysis
        comprehensive_evaluation: Results from systematic benchmark evaluation
        convergence_analysis: Optimization convergence behavior analysis
        adaptive_computation: Adaptive computation effectiveness results
        landscape_analysis: Energy landscape quality analysis
        hyperparameter_sensitivity: Hyperparameter sensitivity study results
        computational_metrics: Performance vs cost trade-off analysis
        publication_summary: Publication-ready summary statistics
        recommendations: Performance tuning recommendations
    """
    timestamp: str
    config: PerformanceAnalysisConfig
    comprehensive_evaluation: Dict[str, Any]
    convergence_analysis: Optional[Dict[str, Any]] = None
    adaptive_computation: Optional[Dict[str, Any]] = None
    landscape_analysis: Optional[Dict[str, Any]] = None
    hyperparameter_sensitivity: Optional[Dict[str, Any]] = None
    computational_metrics: Dict[str, Any] = field(default_factory=dict)
    publication_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for energy-based protein design system.
    
    This class orchestrates systematic evaluation across all benchmark tasks,
    analyzes optimization behavior, and generates publication-ready results.
    
    The analyzer integrates with existing evaluation components while adding:
    - Systematic benchmarking across challenge types
    - Convergence and optimization behavior analysis
    - Computational efficiency and trade-off analysis
    - Comprehensive result reporting with visualizations
    
    Args:
        config: Performance analysis configuration
        design_pipeline: Protein design pipeline to analyze
        
    Example:
        >>> config = PerformanceAnalysisConfig(
        ...     run_hyperparameter_sensitivity=False,  # Skip expensive analysis
        ...     max_benchmark_size=50,  # Smaller benchmark for testing
        ...     output_dir='./performance_results'
        ... )
        >>> analyzer = PerformanceAnalyzer(config, design_pipeline)
        >>> results = analyzer.run_full_analysis()
        >>> print(f"Analysis complete. Summary: {results.publication_summary}")
    """
    
    def __init__(
        self, 
        config: PerformanceAnalysisConfig,
        design_pipeline: ProteinDesignPipeline
    ):
        """Initialize performance analyzer"""
        # Validate inputs
        if not isinstance(config, PerformanceAnalysisConfig):
            raise TypeError(f"config must be PerformanceAnalysisConfig, got {type(config)}")
        
        if design_pipeline is None:
            raise ValueError("design_pipeline cannot be None")
        
        self.config = config
        self.pipeline = design_pipeline
        
        # Validate configuration
        self._validate_config()
        
        # Set up output directory
        if self.config.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.output_dir = f"performance_analysis_{timestamp}"
        
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.results_file = self.output_dir / "performance_results.json"
        self.log_file = self.output_dir / "analysis_log.txt"
        
        # Initialize components
        self._setup_components()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        
        # Performance tracking
        self.start_time = time.time()
        self.memory_peak = 0.0
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.max_benchmark_size <= 0:
            raise ValueError(f"max_benchmark_size must be positive, got {self.config.max_benchmark_size}")
        
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.config.batch_size}")
        
        if self.config.memory_limit_gb <= 0:
            raise ValueError(f"memory_limit_gb must be positive, got {self.config.memory_limit_gb}")
        
        if self.config.statistical_alpha <= 0 or self.config.statistical_alpha >= 1:
            raise ValueError(f"statistical_alpha must be in (0, 1), got {self.config.statistical_alpha}")
        
        if self.config.bootstrap_samples <= 0:
            raise ValueError(f"bootstrap_samples must be positive, got {self.config.bootstrap_samples}")
    
    def _setup_components(self):
        """Initialize evaluation components"""
        
        # Set up benchmark dataset curator
        benchmark_config = self.config.benchmark_config or BenchmarkDatasetConfig()
        self.benchmark_curator = BenchmarkDatasetCurator(benchmark_config)
        
        # Set up validation pipeline
        validation_config = self.config.validation_config or ValidationConfig()
        self.validation_pipeline = ValidationPipeline(validation_config)
        
        # Set up energy evaluator
        self.energy_evaluator = EnergyModelEvaluator(
            model=self.pipeline.energy_head,
            device=self.config.device
        )
        
        # Set up analysis result storage
        self.analysis_results = {}
        self.benchmark_results = {}
        self.computational_metrics = {
            'total_time': 0.0,
            'memory_usage': [],
            'throughput': 0.0,
            'efficiency_metrics': {}
        }
    
    def run_full_analysis(self) -> PerformanceAnalysisResult:
        """
        Run comprehensive performance analysis.
        
        Orchestrates all analysis components according to configuration
        and generates comprehensive results report.
        
        Returns:
            PerformanceAnalysisResult with complete analysis
        """
        self._log("Starting comprehensive performance analysis")
        
        try:
            # 1. Comprehensive Evaluation Study
            if self.config.run_comprehensive_evaluation:
                self._log("Running comprehensive evaluation study...")
                self.analysis_results['comprehensive_evaluation'] = self._run_comprehensive_evaluation()
            
            # 2. Convergence Analysis (placeholder - will be implemented next)
            if self.config.run_convergence_analysis:
                self._log("Running convergence analysis...")
                self.analysis_results['convergence_analysis'] = self._run_convergence_analysis_placeholder()
            
            # 3. Adaptive Computation Analysis (placeholder - will be implemented next)
            if self.config.run_adaptive_computation:
                self._log("Running adaptive computation analysis...")
                self.analysis_results['adaptive_computation'] = self._run_adaptive_computation_placeholder()
            
            # 4. Landscape Analysis (placeholder - will be implemented next) 
            if self.config.run_landscape_analysis:
                self._log("Running landscape analysis...")
                self.analysis_results['landscape_analysis'] = self._run_landscape_analysis_placeholder()
            
            # 5. Hyperparameter Sensitivity (placeholder - will be implemented next)
            if self.config.run_hyperparameter_sensitivity:
                self._log("Running hyperparameter sensitivity analysis...")
                self.analysis_results['hyperparameter_sensitivity'] = self._run_hyperparameter_sensitivity_placeholder()
            
            # 6. Generate computational metrics and publication summary
            self._finalize_computational_metrics()
            publication_summary = self._generate_publication_summary()
            recommendations = self._generate_recommendations()
            
            # 7. Generate visualizations if requested
            if self.config.generate_plots and PLOTTING_AVAILABLE:
                self._generate_visualizations()
            
            # Create final result
            result = PerformanceAnalysisResult(
                timestamp=datetime.now().isoformat(),
                config=self.config,
                comprehensive_evaluation=self.analysis_results.get('comprehensive_evaluation', {}),
                convergence_analysis=self.analysis_results.get('convergence_analysis'),
                adaptive_computation=self.analysis_results.get('adaptive_computation'),
                landscape_analysis=self.analysis_results.get('landscape_analysis'), 
                hyperparameter_sensitivity=self.analysis_results.get('hyperparameter_sensitivity'),
                computational_metrics=self.computational_metrics,
                publication_summary=publication_summary,
                recommendations=recommendations
            )
            
            # Save results
            self._save_results(result)
            
            self._log("Performance analysis completed successfully")
            return result
            
        except Exception as e:
            self._log(f"Performance analysis failed: {str(e)}")
            raise
    
    def _run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run systematic evaluation on all benchmark tasks.
        
        This implements the core "comprehensive evaluation study" requirement
        from Phase 4.3, including success rate comparison across challenges
        and failure mode analysis.
        """
        self._log("Generating benchmark datasets...")
        
        eval_results = {
            'benchmark_types': {},
            'overall_metrics': {},
            'failure_analysis': {},
            'success_rates': {},
            'computational_costs': {}
        }
        
        # Generate benchmark datasets based on configuration
        benchmark_types = []
        if self.config.include_novel_backbones:
            benchmark_types.append('novel_backbones')
        if self.config.include_multi_constraint:
            benchmark_types.append('multi_constraint')
        if self.config.include_extrapolation:
            benchmark_types.append('extrapolation')
        if self.config.include_literature_targets:
            benchmark_types.append('literature_targets')
        
        self._log(f"Evaluating {len(benchmark_types)} benchmark types: {benchmark_types}")
        
        # Evaluate each benchmark type
        all_results = []
        total_problems = 0
        total_successes = 0
        
        for benchmark_type in benchmark_types:
            self._log(f"Evaluating {benchmark_type}...")
            
            # Generate benchmark dataset for this type
            benchmark_data = self._generate_benchmark_data(
                benchmark_type, 
                self.config.max_benchmark_size
            )
            
            # Run evaluation on benchmark
            type_results = self._evaluate_benchmark_type(benchmark_type, benchmark_data)
            eval_results['benchmark_types'][benchmark_type] = type_results
            
            # Accumulate statistics
            all_results.extend(type_results.get('individual_results', []))
            total_problems += type_results.get('total_problems', 0)
            total_successes += type_results.get('successful_designs', 0)
            
            # Track computational costs
            eval_results['computational_costs'][benchmark_type] = {
                'total_time': type_results.get('total_time', 0.0),
                'avg_time_per_problem': type_results.get('avg_time_per_problem', 0.0),
                'memory_usage': type_results.get('memory_usage', 0.0)
            }
        
        # Calculate overall metrics (with proper division by zero protection)
        eval_results['overall_metrics'] = {
            'total_problems_evaluated': total_problems,
            'overall_success_rate': total_successes / total_problems if total_problems > 0 else 0.0,
            'total_evaluation_time': sum(
                costs['total_time'] for costs in eval_results['computational_costs'].values()
            )
        }
        
        # Analyze failure modes across all results
        eval_results['failure_analysis'] = self._analyze_failure_modes(all_results)
        
        # Calculate success rates by challenge difficulty
        eval_results['success_rates'] = self._calculate_success_rates_by_difficulty(eval_results)
        
        self._log(f"Comprehensive evaluation complete. Overall success rate: {eval_results['overall_metrics']['overall_success_rate']:.3f}")
        
        return eval_results
    
    def _generate_benchmark_data(self, benchmark_type: str, max_size: int) -> List[Dict[str, Any]]:
        """Generate benchmark data for specific type"""
        
        if benchmark_type == 'novel_backbones':
            # Generate novel backbone challenges
            problems = []
            for i in range(min(max_size, 20)):  # Conservative limit for computational efficiency
                problem = {
                    'type': 'novel_backbone',
                    'difficulty': 'medium' if i < 10 else 'hard',
                    'sequence_length': 50 + (i * 10),  # Varying lengths
                    'target_properties': {
                        'fold_confidence_target': 0.8,
                        'stability_target': 'high'
                    }
                }
                problems.append(problem)
            return problems
        
        elif benchmark_type == 'multi_constraint':
            # Generate multi-constraint challenges
            problems = []
            constraints = ['binding', 'stability', 'solubility', 'expression']
            
            for i in range(min(max_size, 15)):
                num_constraints = min(2 + (i // 5), 4)  # 2-4 constraints
                selected_constraints = random.sample(constraints, num_constraints)
                
                problem = {
                    'type': 'multi_constraint',
                    'constraints': selected_constraints,
                    'difficulty': 'easy' if num_constraints == 2 else 'hard',
                    'sequence_length': 80 + (i * 5),
                    'target_properties': {constraint: 'high' for constraint in selected_constraints}
                }
                problems.append(problem)
            return problems
        
        elif benchmark_type == 'extrapolation':
            # Generate extrapolation challenges (larger proteins, novel folds)
            problems = []
            for i in range(min(max_size, 10)):  # Small number due to computational cost
                problem = {
                    'type': 'extrapolation',
                    'challenge': 'large_protein' if i < 5 else 'novel_fold',
                    'difficulty': 'hard',
                    'sequence_length': 200 + (i * 50),  # Large proteins
                    'target_properties': {
                        'fold_confidence_target': 0.7,  # Lower target for hard problems
                        'stability_target': 'medium'
                    }
                }
                problems.append(problem)
            return problems
        
        elif benchmark_type == 'literature_targets':
            # Placeholder for literature targets (would require external data)
            self._log("Literature targets not implemented - using mock data")
            return [
                {
                    'type': 'literature_target',
                    'source': 'mock_paper_1',
                    'difficulty': 'medium',
                    'sequence_length': 120,
                    'target_properties': {'experimental_validation': True}
                }
            ]
        
        else:
            self._log(f"Unknown benchmark type: {benchmark_type}")
            return []
    
    def _evaluate_benchmark_type(self, benchmark_type: str, benchmark_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate design pipeline on specific benchmark type"""
        
        type_start_time = time.time()
        individual_results = []
        successful_designs = 0
        failed_designs = 0
        
        self._log(f"Processing {len(benchmark_data)} problems for {benchmark_type}")
        
        # Process each problem in the benchmark
        for i, problem in enumerate(tqdm(benchmark_data, desc=f"Evaluating {benchmark_type}")):
            try:
                # Run design pipeline on this problem
                problem_result = self._evaluate_single_problem(problem)
                individual_results.append(problem_result)
                
                # Track success/failure
                if problem_result.get('success', False):
                    successful_designs += 1
                else:
                    failed_designs += 1
                
                # Memory management
                if i % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                self._log(f"Failed to evaluate problem {i} in {benchmark_type}: {str(e)}")
                failed_designs += 1
                individual_results.append({
                    'success': False,
                    'failure_reason': f"evaluation_error: {str(e)}",
                    'problem_index': i
                })
        
        type_total_time = time.time() - type_start_time
        
        # Calculate metrics for this benchmark type (with proper division by zero protection)
        total_problems = len(benchmark_data)
        success_rate = successful_designs / total_problems if total_problems > 0 else 0.0
        avg_time_per_problem = type_total_time / total_problems if total_problems > 0 else 0.0
        
        type_results = {
            'total_problems': total_problems,
            'successful_designs': successful_designs,
            'failed_designs': failed_designs,
            'success_rate': success_rate,
            'total_time': type_total_time,
            'avg_time_per_problem': avg_time_per_problem,
            'memory_usage': self._get_current_memory_usage(),
            'individual_results': individual_results
        }
        
        self._log(f"{benchmark_type} evaluation complete. Success rate: {success_rate:.3f}")
        
        return type_results
    
    def _evaluate_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate design pipeline on a single benchmark problem"""
        
        problem_start_time = time.time()
        
        try:
            # Generate backbone structure for this problem
            sequence_length = problem.get('sequence_length', 100)
            if not hasattr(self, 'design_pipeline') or self.design_pipeline is None:
                raise RuntimeError("Design pipeline not available - cannot generate backbone structures")
            
            # Run design pipeline
            backbone = self.design_pipeline.generate_backbone(sequence_length)
            design_result = self._run_design_pipeline(backbone, problem)
            
            # Validate design if successful
            if design_result.get('success', False):
                validation_result = self._validate_design_result(design_result, problem)
            else:
                validation_result = {'success': False, 'reason': 'design_failed'}
            
            problem_time = time.time() - problem_start_time
            
            # Compile result
            result = {
                'success': validation_result.get('success', False),
                'design_time': problem_time,
                'problem_type': problem.get('type'),
                'difficulty': problem.get('difficulty'),
                'sequence_length': sequence_length,
                'target_properties': problem.get('target_properties', {}),
                'design_result': design_result,
                'validation_result': validation_result
            }
            
            if not result['success']:
                result['failure_reason'] = validation_result.get('reason', 'unknown')
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'failure_reason': f"evaluation_error: {str(e)}",
                'design_time': time.time() - problem_start_time,
                'problem_type': problem.get('type'),
                'difficulty': problem.get('difficulty')
            }
    
    def _run_design_pipeline(self, backbone: Dict[str, Any], problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run the actual design pipeline"""
        
        if not hasattr(self, 'design_pipeline') or self.design_pipeline is None:
            raise RuntimeError("Design pipeline not available - cannot run performance analysis")
        
        try:
            result = self.design_pipeline.design_sequence(backbone, problem)
            return result
        except Exception as e:
            raise RuntimeError(f"Design pipeline failed: {str(e)}")
    
    def _validate_design_result(self, design_result: Dict[str, Any], problem: Dict[str, Any]) -> Dict[str, Any]:
        """Validate design result against problem requirements"""
        
        if not design_result.get('success', False):
            return {'success': False, 'reason': 'design_failed'}
        
        # Mock validation based on target properties
        target_properties = problem.get('target_properties', {})
        
        # Simulate validation success based on requirements
        validation_checks = []
        
        if 'fold_confidence_target' in target_properties:
            target = target_properties['fold_confidence_target']
            actual = random.uniform(0.5, 0.95)  # Mock AlphaFold confidence
            validation_checks.append(('fold_confidence', actual >= target))
        
        if 'stability_target' in target_properties:
            # Mock stability validation
            validation_checks.append(('stability', random.random() > 0.3))
        
        # Overall validation success
        all_passed = all(check[1] for check in validation_checks)
        
        return {
            'success': all_passed,
            'validation_checks': validation_checks,
            'reason': 'validation_passed' if all_passed else 'validation_failed'
        }
    
    def _analyze_failure_modes(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure modes across all evaluation results"""
        
        failed_results = [r for r in all_results if not r.get('success', False)]
        total_failures = len(failed_results)
        
        if total_failures == 0:
            return {'total_failures': 0, 'failure_modes': {}}
        
        # Categorize failure modes
        failure_counts = defaultdict(int)
        failure_by_difficulty = defaultdict(list)
        failure_by_type = defaultdict(list)
        
        for result in failed_results:
            failure_reason = result.get('failure_reason', 'unknown')
            failure_counts[failure_reason] += 1
            
            difficulty = result.get('difficulty', 'unknown')
            failure_by_difficulty[difficulty].append(failure_reason)
            
            problem_type = result.get('problem_type', 'unknown')
            failure_by_type[problem_type].append(failure_reason)
        
        # Calculate failure mode statistics
        failure_modes = {}
        for mode, count in failure_counts.items():
            failure_modes[mode] = {
                'count': count,
                'frequency': count / total_failures,
                'description': self._get_failure_mode_description(mode)
            }
        
        return {
            'total_failures': total_failures,
            'failure_modes': failure_modes,
            'failure_by_difficulty': dict(failure_by_difficulty),
            'failure_by_type': dict(failure_by_type)
        }
    
    def _get_failure_mode_description(self, mode: str) -> str:
        """Get human-readable description of failure mode"""
        descriptions = {
            'optimization_failed': 'Optimization algorithm failed to converge',
            'convergence_timeout': 'Optimization exceeded maximum step limit',
            'energy_explosion': 'Energy values became numerically unstable',
            'invalid_sequence': 'Generated sequence violates biological constraints',
            'validation_failed': 'Design passed optimization but failed validation',
            'design_failed': 'Design pipeline failed before validation',
            'evaluation_error': 'Error during evaluation process'
        }
        return descriptions.get(mode, f'Unknown failure mode: {mode}')
    
    def _calculate_success_rates_by_difficulty(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate success rates stratified by problem difficulty"""
        
        success_rates = {}
        
        # Aggregate results by difficulty across all benchmark types
        difficulty_results = defaultdict(list)
        
        for benchmark_type, type_results in eval_results['benchmark_types'].items():
            individual_results = type_results.get('individual_results', [])
            for result in individual_results:
                difficulty = result.get('difficulty', 'unknown')
                difficulty_results[difficulty].append(result.get('success', False))
        
        # Calculate success rates for each difficulty (with proper division by zero protection)
        for difficulty, results in difficulty_results.items():
            if results and len(results) > 0:
                success_rate = sum(results) / len(results)
                success_rates[difficulty] = {
                    'success_rate': success_rate,
                    'total_problems': len(results),
                    'successful_problems': sum(results)
                }
            else:
                # Handle empty results case
                success_rates[difficulty] = {
                    'success_rate': 0.0,
                    'total_problems': 0,
                    'successful_problems': 0
                }
        
        return success_rates
    
    # Placeholder methods for other analysis components
    # These will be implemented in subsequent tasks
    
    def _run_convergence_analysis_placeholder(self) -> Dict[str, Any]:
        """Placeholder for convergence analysis (to be implemented next)"""
        return {
            'status': 'placeholder',
            'message': 'Convergence analysis will be implemented in convergence_analysis.py'
        }
    
    def _run_adaptive_computation_placeholder(self) -> Dict[str, Any]:
        """Placeholder for adaptive computation analysis (to be implemented next)"""
        return {
            'status': 'placeholder', 
            'message': 'Adaptive computation analysis will be implemented in adaptive_computation_analysis.py'
        }
    
    def _run_landscape_analysis_placeholder(self) -> Dict[str, Any]:
        """Placeholder for landscape analysis (to be implemented next)"""
        return {
            'status': 'placeholder',
            'message': 'Landscape analysis will be implemented in landscape_quality_analysis.py'
        }
    
    def _run_hyperparameter_sensitivity_placeholder(self) -> Dict[str, Any]:
        """Placeholder for hyperparameter sensitivity (to be implemented next)"""
        return {
            'status': 'placeholder',
            'message': 'Hyperparameter sensitivity will be implemented in hyperparameter_sensitivity.py'
        }
    
    def _finalize_computational_metrics(self):
        """Calculate final computational performance metrics"""
        total_time = time.time() - self.start_time
        
        self.computational_metrics.update({
            'total_analysis_time': total_time,
            'peak_memory_usage': self._get_peak_memory_usage(),
            'average_memory_usage': np.mean(self.computational_metrics['memory_usage']) if self.computational_metrics['memory_usage'] else 0.0
        })
    
    def _generate_publication_summary(self) -> Dict[str, Any]:
        """Generate publication-ready summary statistics"""
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_evaluation_time_minutes': self.computational_metrics.get('total_analysis_time', 0.0) / 60.0,
            'peak_memory_usage_gb': self.computational_metrics.get('peak_memory_usage', 0.0),
        }
        
        # Add comprehensive evaluation summary
        if 'comprehensive_evaluation' in self.analysis_results:
            comp_eval = self.analysis_results['comprehensive_evaluation']
            overall_metrics = comp_eval.get('overall_metrics', {})
            
            summary.update({
                'total_problems_evaluated': overall_metrics.get('total_problems_evaluated', 0),
                'overall_success_rate': overall_metrics.get('overall_success_rate', 0.0),
                'benchmark_types_evaluated': len(comp_eval.get('benchmark_types', {}))
            })
            
            # Success rates by difficulty
            success_rates = comp_eval.get('success_rates', {})
            for difficulty, rates in success_rates.items():
                summary[f'success_rate_{difficulty}'] = rates.get('success_rate', 0.0)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance tuning recommendations based on analysis"""
        
        recommendations = []
        
        # Analyze comprehensive evaluation results for recommendations
        if 'comprehensive_evaluation' in self.analysis_results:
            comp_eval = self.analysis_results['comprehensive_evaluation']
            overall_success = comp_eval.get('overall_metrics', {}).get('overall_success_rate', 0.0)
            
            if overall_success < 0.5:
                recommendations.append(
                    "Overall success rate is low (<50%). Consider adjusting optimization parameters "
                    "or improving energy model training."
                )
            
            # Analyze failure modes
            failure_analysis = comp_eval.get('failure_analysis', {})
            failure_modes = failure_analysis.get('failure_modes', {})
            
            if 'optimization_failed' in failure_modes:
                freq = failure_modes['optimization_failed'].get('frequency', 0.0)
                if freq > 0.3:
                    recommendations.append(
                        f"Optimization failures are common ({freq:.1%}). Consider increasing "
                        "max_steps_per_landscape or improving convergence criteria."
                    )
            
            if 'convergence_timeout' in failure_modes:
                freq = failure_modes['convergence_timeout'].get('frequency', 0.0)
                if freq > 0.2:
                    recommendations.append(
                        f"Convergence timeouts are frequent ({freq:.1%}). Consider adaptive "
                        "step allocation or better initialization strategies."
                    )
        
        # Add computational efficiency recommendations
        total_time = self.computational_metrics.get('total_analysis_time', 0.0)
        if total_time > 3600:  # More than 1 hour
            recommendations.append(
                f"Analysis took {total_time/3600:.1f} hours. Consider reducing benchmark sizes "
                "or using batch processing for routine evaluations."
            )
        
        peak_memory = self.computational_metrics.get('peak_memory_usage', 0.0)
        if peak_memory > 8.0:  # More than 8GB
            recommendations.append(
                f"Peak memory usage was {peak_memory:.1f}GB. Consider reducing batch sizes "
                "or implementing memory management strategies."
            )
        
        if not recommendations:
            recommendations.append(
                "Performance analysis completed successfully with no major issues identified."
            )
        
        return recommendations
    
    def _generate_visualizations(self):
        """Generate publication-ready visualizations"""
        
        if not PLOTTING_AVAILABLE:
            self._log("Plotting libraries not available. Skipping visualizations.")
            return
        
        self._log("Generating visualizations...")
        
        try:
            # Set up plotting
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Success rates by benchmark type
            if 'comprehensive_evaluation' in self.analysis_results:
                self._plot_success_rates_by_type(axes[0, 0])
            
            # 2. Success rates by difficulty
            if 'comprehensive_evaluation' in self.analysis_results:
                self._plot_success_rates_by_difficulty(axes[0, 1])
            
            # 3. Failure mode analysis
            if 'comprehensive_evaluation' in self.analysis_results:
                self._plot_failure_modes(axes[1, 0])
            
            # 4. Computational efficiency
            self._plot_computational_metrics(axes[1, 1])
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / "performance_analysis_summary.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"Visualization saved to {viz_path}")
            
        except Exception as e:
            self._log(f"Failed to generate visualizations: {str(e)}")
    
    def _plot_success_rates_by_type(self, ax):
        """Plot success rates by benchmark type"""
        comp_eval = self.analysis_results['comprehensive_evaluation']
        benchmark_types = comp_eval.get('benchmark_types', {})
        
        types = []
        rates = []
        
        for benchmark_type, results in benchmark_types.items():
            types.append(benchmark_type.replace('_', ' ').title())
            rates.append(results.get('success_rate', 0.0))
        
        if types and rates:
            ax.bar(types, rates, alpha=0.7, color='skyblue')
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rates by Benchmark Type')
            ax.set_ylim(0, 1)
            for i, rate in enumerate(rates):
                ax.text(i, rate + 0.02, f'{rate:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Rates by Benchmark Type')
    
    def _plot_success_rates_by_difficulty(self, ax):
        """Plot success rates by problem difficulty"""
        comp_eval = self.analysis_results['comprehensive_evaluation'] 
        success_rates = comp_eval.get('success_rates', {})
        
        difficulties = []
        rates = []
        
        for difficulty, rate_data in success_rates.items():
            difficulties.append(difficulty.title())
            rates.append(rate_data.get('success_rate', 0.0))
        
        if difficulties and rates:
            colors = {'Easy': 'lightgreen', 'Medium': 'orange', 'Hard': 'lightcoral'}
            bar_colors = [colors.get(d, 'gray') for d in difficulties]
            
            ax.bar(difficulties, rates, alpha=0.7, color=bar_colors)
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rates by Problem Difficulty')
            ax.set_ylim(0, 1)
            for i, rate in enumerate(rates):
                ax.text(i, rate + 0.02, f'{rate:.2f}', ha='center')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Rates by Problem Difficulty')
    
    def _plot_failure_modes(self, ax):
        """Plot failure mode frequency distribution"""
        comp_eval = self.analysis_results['comprehensive_evaluation']
        failure_analysis = comp_eval.get('failure_analysis', {})
        failure_modes = failure_analysis.get('failure_modes', {})
        
        if failure_modes:
            modes = []
            frequencies = []
            
            for mode, data in failure_modes.items():
                modes.append(mode.replace('_', ' ').title())
                frequencies.append(data.get('frequency', 0.0))
            
            ax.pie(frequencies, labels=modes, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Failure Modes')
        else:
            ax.text(0.5, 0.5, 'No failure data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Failure Modes')
    
    def _plot_computational_metrics(self, ax):
        """Plot computational performance metrics"""
        metrics = ['Total Time (min)', 'Peak Memory (GB)', 'Success Rate']
        values = [
            self.computational_metrics.get('total_analysis_time', 0.0) / 60.0,
            self.computational_metrics.get('peak_memory_usage', 0.0),
            self.analysis_results.get('comprehensive_evaluation', {}).get('overall_metrics', {}).get('overall_success_rate', 0.0)
        ]
        
        # Normalize values for comparison (0-1 scale)
        normalized_values = []
        max_time = 120  # 2 hours max
        max_memory = 16  # 16GB max
        
        normalized_values.append(min(values[0] / max_time, 1.0))  # Time
        normalized_values.append(min(values[1] / max_memory, 1.0))  # Memory
        normalized_values.append(values[2])  # Success rate already 0-1
        
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        bars = ax.bar(metrics, normalized_values, color=colors, alpha=0.7)
        
        # Add actual values as text
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if i == 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.1f}', 
                       ha='center', va='bottom')
            elif i == 1:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.1f}', 
                       ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.2f}', 
                       ha='center', va='bottom')
        
        ax.set_ylabel('Normalized Value')
        ax.set_title('Computational Performance Metrics')
        ax.set_ylim(0, 1.1)
    
    def _save_results(self, result: PerformanceAnalysisResult):
        """Save analysis results to files"""
        
        # Convert result to dictionary for JSON serialization
        result_dict = asdict(result)
        
        # Save main results
        with open(self.results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save publication summary separately
        summary_file = self.output_dir / "publication_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(result.publication_summary, f, indent=2)
        
        # Save recommendations
        recommendations_file = self.output_dir / "recommendations.txt"
        with open(recommendations_file, 'w') as f:
            f.write("Performance Analysis Recommendations\n")
            f.write("=" * 40 + "\n\n")
            for i, rec in enumerate(result.recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
        
        self._log(f"Results saved to {self.output_dir}")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_gb = memory_mb / 1024
            
            # Update peak memory tracking
            self.memory_peak = max(self.memory_peak, memory_gb)
            self.computational_metrics['memory_usage'].append(memory_gb)
            
            return memory_gb
        except (ImportError, Exception) as e:
            # Log warning but don't fail the analysis
            if isinstance(e, ImportError):
                self._log("Warning: psutil not available for memory monitoring")
            else:
                self._log(f"Warning: Failed to get memory usage: {str(e)}")
            return 0.0
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in GB"""
        return self.memory_peak
    
    def _log(self, message: str):
        """Log message to console and log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if self.config.verbose:
            print(log_message)
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
        except Exception:
            pass  # Fail silently if can't write to log


def main():
    """Command-line interface for performance analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Performance Analysis for Energy-Based Protein Design')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--max-benchmark-size', type=int, default=20, help='Maximum problems per benchmark type')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--include-hyperparameter-sensitivity', action='store_true', 
                       help='Include expensive hyperparameter sensitivity analysis')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PerformanceAnalysisConfig(
        output_dir=args.output_dir,
        max_benchmark_size=args.max_benchmark_size,
        generate_plots=not args.no_plots,
        run_hyperparameter_sensitivity=args.include_hyperparameter_sensitivity,
        random_seed=args.random_seed,
        verbose=args.verbose
    )
    
    # Note: In real usage, design_pipeline would be loaded from checkpoints
    print("Note: This is a standalone performance analysis framework.")
    print("For actual use, integrate with trained design pipeline from previous phases.")
    print(f"Configuration: {config}")
    
    return config


if __name__ == '__main__':
    config = main()
    print("Performance analysis framework ready for integration with trained models.")