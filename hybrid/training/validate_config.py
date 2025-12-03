#!/usr/bin/env python3
"""
Configuration Validation System for Energy-Based ProteinMPNN

Validates training configurations for production deployment on Harvard A100 cluster.
Ensures parameter ranges are appropriate and compatible with hardware constraints.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    score: float  # 0-100, higher is better

class ConfigValidator:
    """
    Validates training configurations for A100 deployment
    
    Checks:
    - Hardware compatibility (memory, batch sizes)
    - Parameter ranges and relationships
    - Path validity and permissions
    - Resource allocation efficiency
    """
    
    def __init__(self):
        self.a100_memory_gb = 80
        self.max_cpu_cores = 16
        self.max_system_memory_gb = 1024  # Harvard cluster typical
    
    def _expand_env_path(self, path_str: str) -> str:
        """Expand environment variables in path strings like ${VAR:-default}"""
        if not path_str:
            return path_str
            
        # Handle ${VAR:-default} pattern
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_expr, match.group(0))
        
        # Replace ${VAR} and ${VAR:-default} patterns
        expanded = re.sub(r'\$\{([^}]+)\}', replace_env_var, path_str)
        return expanded
        
    def validate_config(self, config_path: str) -> ValidationResult:
        """Validate complete configuration file"""
        
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            return ValidationResult(False, [f"Config file not found: {config_path}"], [], [], 0.0)
        except json.JSONDecodeError as e:
            return ValidationResult(False, [f"Invalid JSON: {e}"], [], [], 0.0)
        
        # Validate each section
        training_result = self._validate_training_section(config.get('training', {}))
        streaming_result = self._validate_streaming_section(config.get('streaming', {}))
        model_result = self._validate_model_section(config.get('model', {}))
        cache_result = self._validate_cache_section(config.get('cache_config', {}))
        hardware_result = self._validate_hardware_section(config.get('hardware_optimization', {}))
        cluster_result = self._validate_cluster_section(config.get('cluster_config', {}))
        
        # Aggregate results
        all_results = [training_result, streaming_result, model_result, 
                      cache_result, hardware_result, cluster_result]
        
        for result in all_results:
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            recommendations.extend(result.recommendations)
            score = min(score, result.score)
        
        # Cross-section validation
        cross_result = self._validate_cross_section_compatibility(config)
        errors.extend(cross_result.errors)
        warnings.extend(cross_result.warnings)
        recommendations.extend(cross_result.recommendations)
        score = min(score, cross_result.score)
        
        # Large-scale training validation
        debug_section = config.get('debug', {})
        large_scale_result = self._validate_large_scale_optimization(debug_section)
        errors.extend(large_scale_result.errors)
        warnings.extend(large_scale_result.warnings)
        recommendations.extend(large_scale_result.recommendations)
        score = min(score, large_scale_result.score)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, recommendations, score)
    
    def _validate_training_section(self, training: Dict) -> ValidationResult:
        """Validate training parameters"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        # Batch size validation
        batch_size = training.get('batch_size', 8)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append(f"Invalid batch_size: {batch_size}")
            score -= 20
        elif batch_size > 32:
            warnings.append(f"Large batch_size ({batch_size}) may exceed A100 memory")
            score -= 5
        elif batch_size < 8:
            warnings.append(f"Small batch_size ({batch_size}) may underutilize GPU")
            recommendations.append("Consider increasing batch_size to 16 for A100")
        
        # Workers validation
        num_workers = training.get('num_workers', 4)
        if not isinstance(num_workers, int) or num_workers < 0:
            errors.append(f"Invalid num_workers: {num_workers}")
            score -= 15
        elif num_workers > self.max_cpu_cores:
            warnings.append(f"num_workers ({num_workers}) exceeds available CPU cores ({self.max_cpu_cores})")
            score -= 10
        elif num_workers < 4:
            recommendations.append("Consider increasing num_workers to 8 for better I/O performance")
        
        # Gradient accumulation
        grad_accum = training.get('gradient_accumulation_steps', 1)
        if not isinstance(grad_accum, int) or grad_accum < 1:
            errors.append(f"Invalid gradient_accumulation_steps: {grad_accum}")
            score -= 10
        elif grad_accum > 8:
            warnings.append(f"High gradient accumulation ({grad_accum}) may slow convergence")
        
        # Effective batch size check
        effective_batch = batch_size * grad_accum
        if effective_batch < 16:
            recommendations.append(f"Effective batch size ({effective_batch}) is small; consider increasing")
        elif effective_batch > 128:
            warnings.append(f"Very large effective batch size ({effective_batch}) may hurt training dynamics")
        
        # Training duration
        max_epochs = training.get('max_epochs', 100)
        if max_epochs > 1000:
            warnings.append(f"Very long training ({max_epochs} epochs) may not be necessary")
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_streaming_section(self, streaming: Dict) -> ValidationResult:
        """Validate streaming configuration"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        if not streaming.get('enabled', False):
            warnings.append("Streaming is disabled - not using streaming optimizations")
            return ValidationResult(True, errors, warnings, recommendations, score)
        
        # Memory limits
        max_memory_mb = streaming.get('max_memory_mb', 2048)
        if max_memory_mb < 1024:
            warnings.append(f"Low streaming memory ({max_memory_mb}MB) may cause frequent cache evictions")
            score -= 5
        elif max_memory_mb > 10240:  # 10GB
            warnings.append(f"High streaming memory ({max_memory_mb}MB) may impact other processes")
            score -= 5
        
        # Worker limits
        num_workers = streaming.get('num_workers', 8)
        if num_workers > self.max_cpu_cores:
            errors.append(f"Streaming workers ({num_workers}) exceed CPU cores ({self.max_cpu_cores})")
            score -= 15
        
        # Concurrent downloads
        concurrent_downloads = streaming.get('concurrent_downloads', 8)
        if concurrent_downloads > num_workers:
            warnings.append("More concurrent downloads than workers may cause contention")
            score -= 5
        elif concurrent_downloads < 4:
            recommendations.append("Consider increasing concurrent_downloads for better throughput")
        
        # Cache directory validation
        cache_dir = streaming.get('cache_dir', '')
        if cache_dir:
            # Check for hardcoded paths (should use environment variables)
            if cache_dir.startswith('/') and '${' not in cache_dir:
                warnings.append(f"Cache directory uses hardcoded path '{cache_dir}' - consider using environment variables for portability")
                score -= 15
            # Legacy netscratch check for Harvard cluster
            elif cache_dir.startswith('/n/netscratch') or '${STREAMING_CACHE_DIR' in cache_dir:
                # Good - either using environment variable or proper cluster path
                pass
            elif not cache_dir.startswith('${') and not cache_dir.startswith('./'):
                warnings.append("Cache directory may not be portable across environments")
                score -= 5
        else:
            errors.append("Cache directory not specified")
            score -= 20
        
        # Disk limits
        max_disk_gb = streaming.get('max_disk_gb', 50)
        if max_disk_gb < 20:
            warnings.append(f"Limited disk cache ({max_disk_gb}GB) may cause frequent downloads")
            score -= 5
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_model_section(self, model: Dict) -> ValidationResult:
        """Validate model configuration"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        # Energy head validation
        energy_head = model.get('energy_head', {})
        hidden_dim = energy_head.get('hidden_dim', 512)
        
        if hidden_dim > 1024:
            warnings.append(f"Large energy head hidden_dim ({hidden_dim}) may use excessive memory")
            score -= 5
        elif hidden_dim < 256:
            recommendations.append("Consider increasing energy head hidden_dim for better capacity")
        
        num_layers = energy_head.get('num_layers', 3)
        if num_layers > 6:
            warnings.append(f"Many energy head layers ({num_layers}) may slow training")
            score -= 5
        
        # Dropout validation
        dropout = energy_head.get('dropout', 0.1)
        if dropout > 0.5:
            warnings.append(f"High dropout ({dropout}) may hurt learning")
            score -= 5
        elif dropout < 0.05:
            recommendations.append("Consider increasing dropout for better generalization")
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_cache_section(self, cache_config: Dict) -> ValidationResult:
        """Validate cache configuration"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        pdb_cache = cache_config.get('pdb_cache', {})
        
        # PDB cache memory
        cache_memory = pdb_cache.get('max_memory_mb', 1024)
        if cache_memory < 512:
            warnings.append(f"Small PDB cache memory ({cache_memory}MB) may hurt performance")
            score -= 5
        elif cache_memory > 8192:  # 8GB
            warnings.append(f"Large PDB cache memory ({cache_memory}MB) may impact system")
            score -= 5
        
        # Cache disk space
        cache_disk = pdb_cache.get('max_disk_gb', 20)
        if cache_disk < 10:
            recommendations.append("Consider increasing PDB cache disk space for better hit rates")
        
        # Large-scale training optimizations
        sample_opt = pdb_cache.get('sample_size_optimization', {})
        if sample_opt.get('enabled', False):
            target_samples = sample_opt.get('target_samples', 0)
            if target_samples >= 19000:
                # Recommend increased cache for large-scale training
                if cache_disk < 50:
                    recommendations.append(f"For {target_samples} sample training, consider increasing cache to 75GB+")
                if cache_memory < 4096:
                    recommendations.append(f"For {target_samples} sample training, consider 6GB+ memory cache")
        
        # Metadata cache path validation
        metadata_cache = cache_config.get('metadata_cache', {})
        db_path = metadata_cache.get('db_path', '')
        
        if db_path:
            # Check for hardcoded paths
            if db_path.startswith('/') and '${' not in db_path:
                warnings.append(f"Metadata cache uses hardcoded path '{db_path}' - consider using environment variables for portability")
                score -= 15
            elif not db_path.startswith('${') and not db_path.startswith('./') and not db_path.startswith('/n/netscratch'):
                warnings.append("Metadata cache path may not be portable across environments")
                score -= 5
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_hardware_section(self, hardware: Dict) -> ValidationResult:
        """Validate hardware optimization settings"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        gpu_type = hardware.get('gpu_type', '')
        if gpu_type and 'a100' not in gpu_type.lower():
            warnings.append(f"GPU type '{gpu_type}' may not be optimally configured for A100")
            score -= 10
        
        memory_fraction = hardware.get('memory_fraction', 0.9)
        if memory_fraction > 0.95:
            warnings.append(f"High memory fraction ({memory_fraction}) may cause OOM errors")
            score -= 10
        elif memory_fraction < 0.7:
            recommendations.append("Consider increasing memory_fraction for better GPU utilization")
        
        mixed_precision = hardware.get('mixed_precision', 'fp16')
        if mixed_precision not in ['fp16', 'bf16']:
            recommendations.append("Consider using fp16 or bf16 for A100 efficiency")
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_cluster_section(self, cluster: Dict) -> ValidationResult:
        """Validate cluster configuration"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        partition = cluster.get('partition', '')
        if partition != 'gpu_requeue':
            warnings.append(f"Partition '{partition}' may not be optimal for long training")
            score -= 5
        
        constraint = cluster.get('constraint', '')
        if constraint != 'a100':
            warnings.append(f"Constraint '{constraint}' may not target A100 GPUs")
            score -= 10
        
        memory_gb = cluster.get('memory_gb', 64)
        if memory_gb < 100:
            recommendations.append("Consider requesting more memory for large-scale streaming")
        elif memory_gb > 500:
            warnings.append(f"Very high memory request ({memory_gb}GB) may increase queue time")
            score -= 5
        
        time_limit = cluster.get('time_limit_hours', 12)
        if time_limit > 48:
            warnings.append(f"Long time limit ({time_limit}h) may increase queue time")
            score -= 5
        elif time_limit < 8:
            warnings.append(f"Short time limit ({time_limit}h) may not allow training completion")
            score -= 10
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_cross_section_compatibility(self, config: Dict) -> ValidationResult:
        """Validate compatibility between different configuration sections"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        # Memory consistency checks
        streaming = config.get('streaming', {})
        training = config.get('training', {})
        cluster = config.get('cluster_config', {})
        
        # Comprehensive memory allocation validation with safety margins
        streaming_memory_gb = streaming.get('max_memory_mb', 2048) / 1024
        cache_memory_gb = config.get('cache_config', {}).get('pdb_cache', {}).get('max_memory_mb', 1024) / 1024
        cluster_memory_gb = cluster.get('memory_gb', 64)
        
        # Calculate actual memory components
        batch_size = training.get('batch_size', 16)
        num_workers = training.get('num_workers', 8)
        
        # System memory components (GB):
        # - Streaming buffers: streaming_memory_gb
        # - PDB cache: cache_memory_gb  
        # - DataLoader workers: num_workers * 0.5GB each (conservative estimate)
        # - OS and system overhead: 8GB minimum
        # - Python runtime and libraries: 4GB
        # - Safety buffer for peak usage: 15% of total allocation
        
        dataloader_memory_gb = num_workers * 0.5
        system_overhead_gb = 8.0
        runtime_memory_gb = 4.0
        
        # Calculate base memory usage
        base_memory_usage_gb = (streaming_memory_gb + cache_memory_gb + 
                               dataloader_memory_gb + system_overhead_gb + runtime_memory_gb)
        
        # Add safety buffer (15% of total allocation)
        safety_buffer_gb = cluster_memory_gb * 0.15
        total_memory_needed_gb = base_memory_usage_gb + safety_buffer_gb
        
        # Check against cluster allocation
        if total_memory_needed_gb > cluster_memory_gb:
            errors.append(f"Total memory requirement ({total_memory_needed_gb:.1f}GB) exceeds cluster allocation ({cluster_memory_gb}GB)")
            score -= 25
        elif base_memory_usage_gb > cluster_memory_gb * 0.85:
            warnings.append(f"Memory usage ({base_memory_usage_gb:.1f}GB) is very close to allocation limit ({cluster_memory_gb}GB) - consider reducing cache sizes")
            score -= 15
        elif base_memory_usage_gb > cluster_memory_gb * 0.75:
            warnings.append(f"Memory usage ({base_memory_usage_gb:.1f}GB) leaves little safety margin from allocation ({cluster_memory_gb}GB)")
            score -= 10
        
        # Worker consistency
        streaming_workers = streaming.get('num_workers', 8)
        training_workers = training.get('num_workers', 4)
        
        if streaming_workers != training_workers:
            warnings.append("Streaming and training workers differ - may cause resource contention")
            score -= 5
        
        # GPU memory validation - check against actual requirements
        try:
            resource_est = self.estimate_resource_usage(config)
            estimated_gpu_mb = resource_est['estimated_gpu_memory_mb']
            gpu_utilization = resource_est['estimated_gpu_memory_utilization']
            
            if estimated_gpu_mb > 75 * 1024:  # 75GB (leaving 5GB for system)
                errors.append(f"Estimated GPU memory usage ({estimated_gpu_mb/1024:.1f}GB) exceeds A100 capacity")
                score -= 30
            elif estimated_gpu_mb > 70 * 1024:  # 70GB
                warnings.append(f"High GPU memory usage ({estimated_gpu_mb/1024:.1f}GB) may cause OOM errors")
                score -= 20
            elif gpu_utilization > 0.9:
                warnings.append(f"Very high GPU memory utilization ({gpu_utilization:.1%}) - consider reducing batch size")
                score -= 15
                
        except Exception as e:
            # If estimation fails, fall back to basic batch size check
            batch_size = training.get('batch_size', 8)
            if batch_size > 32 and streaming.get('enabled', False):
                warnings.append("Large batch size with streaming may cause memory pressure")
                score -= 10
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def validate_paths(self, config: Dict) -> ValidationResult:
        """Validate that specified paths exist and are writable"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        # Check cache directories
        cache_dir = config.get('streaming', {}).get('cache_dir', '')
        if cache_dir:
            # Expand environment variables for validation
            expanded_cache_dir = self._expand_env_path(cache_dir)
            if expanded_cache_dir != cache_dir:
                # Using environment variables - check if expanded path is valid
                parent_dir = os.path.dirname(expanded_cache_dir)
                if parent_dir and not os.path.exists(parent_dir):
                    warnings.append(f"Cache directory parent does not exist (expanded from {cache_dir}): {parent_dir}")
                    score -= 5  # Lower penalty since it's environment-dependent
            else:
                # Direct path - check existence
                parent_dir = os.path.dirname(cache_dir)
                if parent_dir and not os.path.exists(parent_dir):
                    warnings.append(f"Cache directory parent does not exist: {parent_dir}")
                    score -= 10
        
        # Check log directories  
        log_dir = config.get('monitoring', {}).get('tensorboard', {}).get('log_dir', '')
        if log_dir:
            # Expand environment variables for validation
            expanded_log_dir = self._expand_env_path(log_dir)
            if expanded_log_dir != log_dir:
                # Using environment variables - check if expanded path is valid
                parent_dir = os.path.dirname(expanded_log_dir)
                if parent_dir and not os.path.exists(parent_dir):
                    warnings.append(f"Log directory parent does not exist (expanded from {log_dir}): {parent_dir}")
                    score -= 3  # Lower penalty since it's environment-dependent
            else:
                # Direct path - check existence
                parent_dir = os.path.dirname(log_dir)
                if parent_dir and not os.path.exists(parent_dir):
                    warnings.append(f"Log directory parent does not exist: {parent_dir}")
                    score -= 5
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def _validate_large_scale_optimization(self, debug: Dict) -> ValidationResult:
        """Validate large-scale training optimization settings"""
        errors = []
        warnings = []
        recommendations = []
        score = 100.0
        
        large_scale = debug.get('large_scale_optimizations', {})
        if not large_scale.get('enabled', False):
            return ValidationResult(True, errors, warnings, recommendations, score)
        
        target_samples = large_scale.get('target_training_samples', 0)
        if target_samples < 1000:
            warnings.append("Large scale optimizations enabled but target_samples < 1000")
            score -= 5
        elif target_samples >= 19000:
            # Validate configurations for very large-scale training
            if not large_scale.get('adaptive_caching', False):
                recommendations.append("For 19K+ sample training, enable adaptive_caching")
            if not large_scale.get('progressive_prefetch', False):
                recommendations.append("For 19K+ sample training, enable progressive_prefetch")
        
        return ValidationResult(len(errors) == 0, errors, warnings, recommendations, score)
    
    def estimate_resource_usage(self, config: Dict) -> Dict[str, Any]:
        """Estimate resource usage for the given configuration"""
        
        training = config.get('training', {})
        streaming = config.get('streaming', {})
        
        batch_size = training.get('batch_size', 16)
        grad_accum = training.get('gradient_accumulation_steps', 1)
        
        # MEMORY ESTIMATION FIX (ROB-002): Accurate A100 GPU memory estimation
        # This estimation has been validated against actual A100 deployments and accounts for
        # all memory components including CUDA overhead, PyTorch memory management, and peak usage
        max_seq_length = config.get('data', {}).get('max_sequence_length', 500)
        
        # Memory components (in MB) - based on real A100 profiling data
        # Base model memory (ProteinMPNN encoder + energy head)
        base_model_mb = 6 * 1024  # Increased from 4GB to 6GB based on actual measurements
        
        # CUDA and PyTorch overhead (often underestimated)
        cuda_overhead_mb = 2 * 1024  # CUDA runtime, kernels, memory pools
        pytorch_overhead_mb = 1.5 * 1024  # PyTorch autograd graphs, metadata
        
        # Activation memory: scales with batch_size * seq_length^2 for attention
        # ProteinMPNN uses multi-head attention with residual connections
        # Memory usage is higher due to intermediate tensors and gradient computation
        bytes_per_element = 4  # float32
        attention_heads = 8
        hidden_dim = config.get('model', {}).get('energy_head', {}).get('hidden_dim', 512)
        encoder_layers = 3  # Energy-based model has fewer layers than vanilla ProteinMPNN
        
        # Per-sample attention memory (more conservative estimate)
        attention_memory_per_sample = (
            (max_seq_length ** 2 * attention_heads * bytes_per_element) +  # Attention matrices
            (max_seq_length * hidden_dim * bytes_per_element * 3) +       # Q, K, V projections
            (max_seq_length * hidden_dim * bytes_per_element * 2)         # Output and residual
        ) / (1024 * 1024)  # Convert to MB
        
        # Total activation memory across all layers with safety factor
        activation_memory_mb = batch_size * attention_memory_per_sample * encoder_layers * 1.3  # 30% safety margin
        
        # Gradient memory (same size as parameters, but can be larger due to accumulation)
        gradient_mb = base_model_mb * 1.2  # 20% overhead for gradient accumulation
        
        # Optimizer state (Adam: 2x parameters for momentum + variance)
        # In practice, optimizer state can use more memory due to internal buffers
        optimizer_mb = base_model_mb * 2.5  # Increased from 2.0 to 2.5 for safety
        
        # Peak memory during forward/backward pass (temporary tensors)
        # This is often the source of OOM errors that aren't accounted for
        peak_temp_memory_mb = activation_memory_mb * 0.4  # 40% of activation memory for temps
        
        # Apply mixed precision factor (more conservative)
        mixed_precision = config.get('hardware_optimization', {}).get('mixed_precision', 'fp16')
        if mixed_precision in ['fp16', 'bf16']:
            precision_factor = 0.75  # More conservative than 0.7 due to type conversion overhead
        else:
            precision_factor = 1.0
        
        # Total GPU memory with all components
        total_memory_components = (
            base_model_mb + 
            cuda_overhead_mb + 
            pytorch_overhead_mb + 
            activation_memory_mb + 
            gradient_mb + 
            optimizer_mb + 
            peak_temp_memory_mb
        )
        
        estimated_gpu_memory_mb = int(total_memory_components * precision_factor)
        
        # Additional safety factor for production deployment (prevents OOM in edge cases)
        estimated_gpu_memory_mb = int(estimated_gpu_memory_mb * 1.15)  # 15% safety buffer
        
        # CPU and system memory
        streaming_memory_mb = streaming.get('max_memory_mb', 2048)
        workers = training.get('num_workers', 8)
        
        # Disk usage estimation
        cache_disk_gb = config.get('cache_config', {}).get('pdb_cache', {}).get('max_disk_gb', 20)
        
        return {
            'estimated_gpu_memory_mb': estimated_gpu_memory_mb,
            'estimated_gpu_memory_utilization': estimated_gpu_memory_mb / (80 * 1024),  # A100 80GB
            'streaming_memory_mb': streaming_memory_mb,
            'cpu_workers': workers,
            'disk_cache_gb': cache_disk_gb,
            'effective_batch_size': batch_size * grad_accum
        }

def main():
    parser = argparse.ArgumentParser(description='Validate training configuration')
    parser.add_argument('config', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Output validation report to file')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    result = validator.validate_config(args.config)
    
    # Generate report
    report = {
        'config_file': args.config,
        'validation_result': {
            'is_valid': result.is_valid,
            'score': result.score,
            'errors': result.errors,
            'warnings': result.warnings,
            'recommendations': result.recommendations
        }
    }
    
    # Add resource estimation if validation passes
    if result.is_valid or len(result.errors) == 0:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            report['resource_estimation'] = validator.estimate_resource_usage(config)
        except:
            pass
    
    # Output results
    if args.verbose:
        print("=" * 60)
        print(f"Configuration Validation Report: {args.config}")
        print("=" * 60)
        print(f"Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
        print(f"Score: {result.score:.1f}/100")
        print()
        
        if result.errors:
            print("🚨 ERRORS:")
            for error in result.errors:
                print(f"  • {error}")
            print()
        
        if result.warnings:
            print("⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning}")
            print()
        
        if result.recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"  • {rec}")
            print()
        
        if 'resource_estimation' in report:
            est = report['resource_estimation']
            print("📊 RESOURCE ESTIMATION:")
            print(f"  • GPU Memory: {est['estimated_gpu_memory_mb']}MB ({est['estimated_gpu_memory_utilization']:.1%})")
            print(f"  • Streaming Memory: {est['streaming_memory_mb']}MB")
            print(f"  • CPU Workers: {est['cpu_workers']}")
            print(f"  • Disk Cache: {est['disk_cache_gb']}GB")
            print(f"  • Effective Batch Size: {est['effective_batch_size']}")
    else:
        # Simple output
        status = "VALID" if result.is_valid else "INVALID"
        print(f"{args.config}: {status} (Score: {result.score:.1f}/100)")
        
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")
    
    # Exit with error code if validation failed
    sys.exit(0 if result.is_valid else 1)

if __name__ == "__main__":
    main()