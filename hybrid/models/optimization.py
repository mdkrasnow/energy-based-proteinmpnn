"""
Model Optimization Utilities for ProteinMPNN-IRED Hybrid System

This module provides model optimization techniques including quantization, pruning,
and other compression methods to improve memory usage and inference speed while
maintaining model accuracy.

Features:
- Post-training quantization for inference acceleration
- Quantization-aware training (QAT) support
- Dynamic and static quantization modes
- Model pruning utilities
- Performance benchmarking and validation
- Integration with existing ProteinMPNN and energy models
"""

import os
import warnings
import copy
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.qconfig import default_qconfig
import numpy as np

# Optional imports for advanced optimization
try:
    from torch.quantization import QConfigMapping
    from torch.ao.quantization import get_default_qconfig
    ADVANCED_QUANTIZATION = True
except ImportError:
    ADVANCED_QUANTIZATION = False
    warnings.warn("Advanced quantization features not available. Using basic quantization.")

try:
    import torch.nn.utils.prune as prune
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    warnings.warn("Pruning utilities not available.")


@dataclass
class OptimizationConfig:
    """
    Configuration for model optimization techniques.
    
    Attributes:
        quantization_enabled: Whether to apply quantization (default: False)
        quantization_mode: Quantization mode ('dynamic', 'static', 'qat') (default: 'dynamic')
        quantization_backend: Backend to use ('fbgemm', 'qnnpack') (default: 'fbgemm')
        calibration_batches: Number of batches for calibration (static quantization) (default: 10)
        qat_epochs: Number of epochs for quantization-aware training (default: 5)
        
        pruning_enabled: Whether to apply pruning (default: False)
        pruning_ratio: Ratio of weights to prune (default: 0.3)
        pruning_method: Pruning method ('magnitude', 'random', 'structured') (default: 'magnitude')
        
        performance_test_enabled: Whether to run performance benchmarks (default: True)
        accuracy_validation_enabled: Whether to validate accuracy (default: True)
        
        device: Device for optimization ('cpu', 'cuda') (default: 'cpu')
        save_optimized_model: Whether to save optimized model (default: True)
    """
    quantization_enabled: bool = False
    quantization_mode: str = 'dynamic'
    quantization_backend: str = 'fbgemm'
    calibration_batches: int = 10
    qat_epochs: int = 5
    
    pruning_enabled: bool = False
    pruning_ratio: float = 0.3
    pruning_method: str = 'magnitude'
    
    performance_test_enabled: bool = True
    accuracy_validation_enabled: bool = True
    
    device: str = 'cpu'
    save_optimized_model: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_quant_modes = ['dynamic', 'static', 'qat']
        if self.quantization_mode not in valid_quant_modes:
            raise ValueError(f"quantization_mode must be one of {valid_quant_modes}, got {self.quantization_mode}")
        
        valid_backends = ['fbgemm', 'qnnpack']
        if self.quantization_backend not in valid_backends:
            raise ValueError(f"quantization_backend must be one of {valid_backends}, got {self.quantization_backend}")
        
        if not 0 <= self.pruning_ratio <= 1:
            raise ValueError(f"pruning_ratio must be in [0, 1], got {self.pruning_ratio}")
        
        if self.calibration_batches <= 0:
            raise ValueError(f"calibration_batches must be positive, got {self.calibration_batches}")


class QuantizationWrapper(nn.Module):
    """
    Wrapper for models to support quantization with improved input handling.
    
    This wrapper adds QuantStub and DeQuantStub to enable quantization
    of existing models without modifying their internal structure.
    Handles various input formats including single tensors, multiple tensors,
    and dictionary inputs.
    """
    
    def __init__(self, model: nn.Module, quantize_inputs: bool = True):
        super().__init__()
        self.model = model
        self.quantize_inputs = quantize_inputs
        
        if quantize_inputs:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        
        # Flag to track if this is a protein design model with complex inputs
        self.complex_input_mode = self._detect_complex_input_model(model)
    
    def _detect_complex_input_model(self, model: nn.Module) -> bool:
        """Detect if this is a protein model with dictionary inputs."""
        model_name = model.__class__.__name__
        return any(name in model_name.lower() for name in ['protein', 'mpnn', 'energy', 'ired'])
    
    def forward(self, *args, **kwargs):
        """Forward pass with intelligent quantization based on input type."""
        
        if not self.quantize_inputs:
            # No input quantization - just pass through
            return self.model(*args, **kwargs)
        
        # Handle different input patterns
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], torch.Tensor):
            # Simple case: single tensor input
            x = self.quant(args[0])
            output = self.model(x)
            if isinstance(output, torch.Tensor):
                return self.dequant(output)
            else:
                # Multiple outputs - dequantize each tensor
                return self._dequantize_outputs(output)
                
        elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            # Dictionary input (common for protein models)
            batch_dict = args[0]
            quantized_dict = {}
            
            # Quantize tensor values in dictionary
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    quantized_dict[key] = self.quant(value)
                else:
                    quantized_dict[key] = value
            
            output = self.model(quantized_dict)
            if isinstance(output, torch.Tensor):
                return self.dequant(output)
            else:
                return self._dequantize_outputs(output)
                
        else:
            # Complex case: multiple arguments or mixed types
            # For protein models, often better to skip input quantization
            # and only quantize internal computations
            if self.complex_input_mode:
                warnings.warn("Complex protein model detected - skipping input quantization for compatibility")
                return self.model(*args, **kwargs)
            else:
                # Try to quantize first argument if it's a tensor
                quantized_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                        quantized_args.append(self.quant(arg))
                    else:
                        quantized_args.append(arg)
                
                output = self.model(*quantized_args, **kwargs)
                if isinstance(output, torch.Tensor):
                    return self.dequant(output)
                else:
                    return self._dequantize_outputs(output)
    
    def _dequantize_outputs(self, outputs):
        """Dequantize multiple outputs."""
        if isinstance(outputs, (tuple, list)):
            return type(outputs)(
                self.dequant(out) if isinstance(out, torch.Tensor) else out
                for out in outputs
            )
        elif isinstance(outputs, dict):
            return {
                key: self.dequant(value) if isinstance(value, torch.Tensor) else value
                for key, value in outputs.items()
            }
        else:
            return outputs


class ModelOptimizer:
    """
    Comprehensive model optimization utility for protein design models.
    
    This class provides quantization, pruning, and other optimization techniques
    to reduce model size and improve inference speed while maintaining accuracy.
    
    Args:
        config: Optimization configuration
        
    Example:
        >>> from models.mpnn_encoder import ProteinMPNNBackboneEncoder
        >>> from models.energy_head import EnergyHead
        >>> 
        >>> # Load models
        >>> encoder = ProteinMPNNBackboneEncoder.from_pretrained()
        >>> energy_head = EnergyHead()
        >>> 
        >>> # Optimize models
        >>> optimizer = ModelOptimizer(OptimizationConfig(quantization_enabled=True))
        >>> optimized_encoder = optimizer.optimize_model(encoder, model_name="encoder")
        >>> optimized_energy = optimizer.optimize_model(energy_head, model_name="energy_head")
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Set quantization backend with platform compatibility
        if self.config.quantization_enabled:
            self._setup_quantization_backend()
        
        # Statistics tracking
        self.optimization_stats = {
            'models_optimized': 0,
            'total_size_reduction': 0.0,
            'total_speedup': 0.0,
            'accuracy_degradation': 0.0
        }
    
    def _setup_quantization_backend(self):
        """Setup quantization backend with platform compatibility checks."""
        import platform
        
        # Platform-specific backend preferences
        if platform.system() == "Darwin":  # macOS
            # FBGEMM not available on macOS, use QNNPACK
            preferred_backend = "qnnpack"
        elif platform.system() == "Linux":
            # FBGEMM preferred on Linux for x86
            preferred_backend = self.config.quantization_backend
        else:
            # Windows and others - try QNNPACK first
            preferred_backend = "qnnpack"
        
        # Test backend availability and set with fallback
        available_backends = []
        
        for backend in [preferred_backend, "qnnpack", "fbgemm"]:
            try:
                torch.backends.quantized.engine = backend
                available_backends.append(backend)
                print(f"✓ Using quantization backend: {backend}")
                self.config.quantization_backend = backend  # Update config to reflect actual backend
                return
            except RuntimeError:
                continue
        
        # If no backends work, disable quantization
        warnings.warn(
            "No quantization backends available on this platform. "
            "Quantization will be disabled. This may happen on some macOS systems."
        )
        self.config.quantization_enabled = False
    
    def optimize_model(
        self, 
        model: nn.Module, 
        model_name: str = "model",
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        validation_fn: Optional[Callable] = None
    ) -> nn.Module:
        """
        Apply optimization techniques to a model.
        
        Args:
            model: Model to optimize
            model_name: Name for saving/logging
            calibration_loader: DataLoader for calibration (static quantization)
            validation_fn: Function to validate model accuracy
            
        Returns:
            Optimized model
        """
        print(f"Optimizing model: {model_name}")
        
        # Create working copy
        optimized_model = copy.deepcopy(model)
        
        # Track original model stats
        original_size = self._get_model_size(model)
        original_speed = None
        
        # Apply optimizations in order
        if self.config.quantization_enabled:
            print(f"Applying {self.config.quantization_mode} quantization...")
            optimized_model = self._apply_quantization(
                optimized_model, 
                calibration_loader=calibration_loader
            )
        
        if self.config.pruning_enabled and PRUNING_AVAILABLE:
            print(f"Applying {self.config.pruning_method} pruning (ratio: {self.config.pruning_ratio})...")
            optimized_model = self._apply_pruning(optimized_model)
        
        # Performance testing
        if self.config.performance_test_enabled:
            print("Running performance benchmarks...")
            original_speed, optimized_speed = self._benchmark_performance(
                model, optimized_model, model_name
            )
        
        # Accuracy validation
        accuracy_change = 0.0
        if self.config.accuracy_validation_enabled and validation_fn:
            print("Validating accuracy...")
            accuracy_change = self._validate_accuracy(
                model, optimized_model, validation_fn
            )
        
        # Update statistics
        optimized_size = self._get_model_size(optimized_model)
        size_reduction = (original_size - optimized_size) / original_size * 100
        
        self.optimization_stats['models_optimized'] += 1
        self.optimization_stats['total_size_reduction'] += size_reduction
        if original_speed and optimized_speed:
            speedup = optimized_speed / original_speed
            self.optimization_stats['total_speedup'] += speedup
        self.optimization_stats['accuracy_degradation'] += abs(accuracy_change)
        
        print(f"Optimization complete for {model_name}:")
        print(f"  Size reduction: {size_reduction:.1f}%")
        if original_speed and optimized_speed:
            speedup = optimized_speed / original_speed
            print(f"  Speed improvement: {speedup:.2f}x")
        if accuracy_change != 0:
            print(f"  Accuracy change: {accuracy_change:.3f}")
        
        # Save optimized model if requested
        if self.config.save_optimized_model:
            save_path = f"optimized_{model_name}.pt"
            self._save_optimized_model(optimized_model, save_path)
            print(f"  Saved to: {save_path}")
        
        return optimized_model
    
    def _apply_quantization(
        self, 
        model: nn.Module, 
        calibration_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """Apply quantization to model based on configuration."""
        
        if self.config.quantization_mode == 'dynamic':
            return self._apply_dynamic_quantization(model)
        elif self.config.quantization_mode == 'static':
            if calibration_loader is None:
                warnings.warn("Static quantization requires calibration_loader, falling back to dynamic")
                return self._apply_dynamic_quantization(model)
            return self._apply_static_quantization(model, calibration_loader)
        elif self.config.quantization_mode == 'qat':
            warnings.warn("QAT not fully implemented, falling back to dynamic quantization")
            return self._apply_dynamic_quantization(model)
        else:
            raise ValueError(f"Unknown quantization mode: {self.config.quantization_mode}")
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        model.eval()
        
        try:
            # Dynamic quantization - quantizes weights ahead of time, activations at runtime
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},  # Layer types to quantize
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            warnings.warn(f"Dynamic quantization failed: {e}. Returning original model.")
            return model
    
    def _apply_static_quantization(
        self, 
        model: nn.Module, 
        calibration_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        model.eval()
        
        try:
            # Wrap model for quantization with improved input handling
            wrapped_model = QuantizationWrapper(model, quantize_inputs=True)
            
            # Set quantization config
            if ADVANCED_QUANTIZATION:
                wrapped_model.qconfig = get_default_qconfig(self.config.quantization_backend)
            else:
                wrapped_model.qconfig = default_qconfig
            
            # Prepare model for quantization
            torch.quantization.prepare(wrapped_model, inplace=True)
            
            # Calibration phase
            print(f"Calibrating with {self.config.calibration_batches} batches...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(calibration_loader):
                    if batch_idx >= self.config.calibration_batches:
                        break
                    
                    try:
                        # Handle different batch formats
                        if isinstance(batch, (list, tuple)):
                            if len(batch) > 0:
                                if isinstance(batch[0], torch.Tensor):
                                    _ = wrapped_model(batch[0])
                                else:
                                    # Skip complex batch formats for now
                                    continue
                        elif isinstance(batch, torch.Tensor):
                            _ = wrapped_model(batch)
                        elif isinstance(batch, dict):
                            # Skip dictionary inputs for now - would need model-specific handling
                            continue
                    except Exception as e:
                        warnings.warn(f"Calibration batch {batch_idx} failed: {e}")
                        continue
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(wrapped_model, inplace=False)
            return quantized_model.model  # Return the wrapped model
            
        except Exception as e:
            warnings.warn(f"Static quantization failed: {e}. Falling back to dynamic quantization.")
            return self._apply_dynamic_quantization(model)
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model."""
        if not PRUNING_AVAILABLE:
            warnings.warn("Pruning not available, skipping.")
            return model
        
        try:
            # Apply magnitude-based unstructured pruning
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    if self.config.pruning_method == 'magnitude':
                        prune.l1_unstructured(module, name='weight', amount=self.config.pruning_ratio)
                    elif self.config.pruning_method == 'random':
                        prune.random_unstructured(module, name='weight', amount=self.config.pruning_ratio)
                    else:
                        warnings.warn(f"Unknown pruning method: {self.config.pruning_method}")
                        break
            
            # Make pruning permanent
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    try:
                        prune.remove(module, 'weight')
                    except ValueError:
                        # No pruning applied to this module
                        pass
            
            return model
            
        except Exception as e:
            warnings.warn(f"Pruning failed: {e}. Returning original model.")
            return model
    
    def _benchmark_performance(
        self, 
        original_model: nn.Module, 
        optimized_model: nn.Module, 
        model_name: str,
        num_runs: int = 10
    ) -> Tuple[float, float]:
        """Benchmark performance of original vs optimized model."""
        
        # Create dummy input for benchmarking
        dummy_input = self._create_dummy_input(original_model)
        
        if dummy_input is None:
            warnings.warn("Could not create dummy input for benchmarking")
            return None, None
        
        # Benchmark original model
        original_model.eval()
        original_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                try:
                    _ = original_model(dummy_input)
                except:
                    continue
            
            # Measure
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    _ = original_model(dummy_input)
                    end_time = time.time()
                    original_times.append(end_time - start_time)
                except Exception:
                    continue
        
        # Benchmark optimized model
        optimized_model.eval()
        optimized_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                try:
                    _ = optimized_model(dummy_input)
                except:
                    continue
            
            # Measure
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    _ = optimized_model(dummy_input)
                    end_time = time.time()
                    optimized_times.append(end_time - start_time)
                except Exception:
                    continue
        
        if len(original_times) == 0 or len(optimized_times) == 0:
            warnings.warn("Benchmarking failed - no successful runs")
            return None, None
        
        original_avg = np.mean(original_times)
        optimized_avg = np.mean(optimized_times)
        
        # Return as throughput (1/time)
        return 1.0 / original_avg, 1.0 / optimized_avg
    
    def _create_dummy_input(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Create dummy input for benchmarking based on model type."""
        
        # Try to determine input size from first layer
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                first_layer = module
                break
        
        if first_layer is None:
            return None
        
        try:
            if isinstance(first_layer, nn.Linear):
                # Linear layer: create [batch_size, input_features]
                batch_size = 2
                input_size = first_layer.in_features
                return torch.randn(batch_size, input_size)
            elif isinstance(first_layer, nn.Conv1d):
                # Conv1D: create [batch_size, channels, length]
                batch_size = 2
                in_channels = first_layer.in_channels
                length = 64  # Arbitrary sequence length
                return torch.randn(batch_size, in_channels, length)
            elif isinstance(first_layer, nn.Conv2d):
                # Conv2D: create [batch_size, channels, height, width]
                batch_size = 2
                in_channels = first_layer.in_channels
                height = width = 32  # Arbitrary image size
                return torch.randn(batch_size, in_channels, height, width)
        except Exception:
            pass
        
        return None
    
    def _validate_accuracy(
        self, 
        original_model: nn.Module, 
        optimized_model: nn.Module, 
        validation_fn: Callable
    ) -> float:
        """Validate accuracy of optimized model vs original."""
        try:
            original_accuracy = validation_fn(original_model)
            optimized_accuracy = validation_fn(optimized_model)
            return optimized_accuracy - original_accuracy
        except Exception as e:
            warnings.warn(f"Accuracy validation failed: {e}")
            return 0.0
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
        return total_size
    
    def _save_optimized_model(self, model: nn.Module, path: str):
        """Save optimized model with metadata."""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimization_config': self.config.__dict__,
            'model_size_mb': self._get_model_size(model),
            'optimization_stats': self.optimization_stats
        }
        
        torch.save(save_dict, path)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        num_models = self.optimization_stats['models_optimized']
        
        report = {
            'total_models_optimized': num_models,
            'average_size_reduction_percent': self.optimization_stats['total_size_reduction'] / max(num_models, 1),
            'average_speedup': self.optimization_stats['total_speedup'] / max(num_models, 1),
            'average_accuracy_degradation': self.optimization_stats['accuracy_degradation'] / max(num_models, 1),
            'configuration': self.config.__dict__
        }
        
        return report
    
    def save_report(self, path: str):
        """Save optimization report to file."""
        report = self.get_optimization_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


# Convenience functions for common optimization tasks

def quantize_model(
    model: nn.Module, 
    mode: str = 'dynamic',
    calibration_loader: Optional[torch.utils.data.DataLoader] = None
) -> nn.Module:
    """
    Quick quantization of a model.
    
    Args:
        model: Model to quantize
        mode: Quantization mode ('dynamic', 'static')
        calibration_loader: DataLoader for calibration (static mode only)
        
    Returns:
        Quantized model
    """
    config = OptimizationConfig(
        quantization_enabled=True,
        quantization_mode=mode
    )
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(model, calibration_loader=calibration_loader)


def prune_model(model: nn.Module, ratio: float = 0.3, method: str = 'magnitude') -> nn.Module:
    """
    Quick pruning of a model.
    
    Args:
        model: Model to prune
        ratio: Ratio of weights to prune
        method: Pruning method
        
    Returns:
        Pruned model
    """
    config = OptimizationConfig(
        pruning_enabled=True,
        pruning_ratio=ratio,
        pruning_method=method
    )
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(model)


def optimize_for_deployment(
    model: nn.Module,
    quantize: bool = True,
    prune: bool = False,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None
) -> nn.Module:
    """
    Comprehensive optimization for deployment.
    
    Args:
        model: Model to optimize
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        calibration_loader: DataLoader for calibration
        
    Returns:
        Optimized model
    """
    config = OptimizationConfig(
        quantization_enabled=quantize,
        quantization_mode='static' if calibration_loader else 'dynamic',
        pruning_enabled=prune,
        performance_test_enabled=True
    )
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(
        model, 
        model_name="deployment_model",
        calibration_loader=calibration_loader
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Optimization utilities...")
    
    try:
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        model = SimpleModel()
        print(f"✓ Test model created")
        
        # Test dynamic quantization
        quantized_model = quantize_model(model, mode='dynamic')
        print(f"✓ Dynamic quantization successful")
        
        # Test pruning (if available)
        if PRUNING_AVAILABLE:
            pruned_model = prune_model(model, ratio=0.2)
            print(f"✓ Pruning successful")
        
        # Test comprehensive optimization
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_mode='dynamic',
            pruning_enabled=PRUNING_AVAILABLE,
            pruning_ratio=0.2
        )
        
        optimizer = ModelOptimizer(config)
        optimized_model = optimizer.optimize_model(model, model_name="test_model")
        
        # Test performance
        test_input = torch.randn(4, 64)
        
        original_output = model(test_input)
        optimized_output = optimized_model(test_input)
        
        print(f"✓ Original output shape: {original_output.shape}")
        print(f"✓ Optimized output shape: {optimized_output.shape}")
        
        # Print optimization report
        report = optimizer.get_optimization_report()
        print(f"✓ Optimization report: {report}")
        
        print("✓ All model optimization tests completed successfully")
        
    except Exception as e:
        print(f"✗ Model optimization test failed: {e}")
        import traceback
        traceback.print_exc()