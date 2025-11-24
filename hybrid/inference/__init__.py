"""
Inference and Optimization Engine for Hybrid Energy-Based ProteinMPNN

This module implements:
- IRED-style iterative sequence optimization
- Adaptive computation allocation
- End-to-end design pipeline
- Multi-landscape energy minimization
"""

# Inference imports
from .ired_optimizer import IREDSequenceOptimizer, OptimizationConfig, OptimizationResult

# Future imports (Phase 3.3)
# from .design_pipeline import DesignPipeline

__all__ = ['IREDSequenceOptimizer', 'OptimizationConfig', 'OptimizationResult']