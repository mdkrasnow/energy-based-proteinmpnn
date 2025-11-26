"""
Memory Profiler for ProteinMPNN-IRED Hybrid System

This module provides comprehensive memory profiling capabilities for identifying
performance bottlenecks in the energy-based protein design pipeline.

Features:
- CPU and GPU memory monitoring
- Component-wise memory tracking
- Memory growth detection
- Performance bottleneck identification
- Context manager for clean profiling sessions
- Visualization and reporting utilities
"""

import os
import gc
import time
import warnings
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import numpy as np

# Optional dependency imports with graceful degradation
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. CPU memory monitoring will be limited.", UserWarning)


@dataclass
class MemorySnapshot:
    """
    Snapshot of memory usage at a specific point in time.
    
    Attributes:
        timestamp: Time of snapshot
        cpu_total: Total CPU memory usage in MB
        cpu_available: Available CPU memory in MB
        cpu_percent: CPU memory usage percentage
        gpu_allocated: GPU memory allocated in MB (per device)
        gpu_reserved: GPU memory reserved in MB (per device)
        gpu_max_allocated: Maximum GPU memory allocated in MB (per device)
        process_rss: Process resident set size in MB
        process_vms: Process virtual memory size in MB
        tag: Optional tag for this snapshot
        component: Component being profiled
        custom_metrics: Additional custom metrics
    """
    timestamp: float
    cpu_total: float
    cpu_available: float
    cpu_percent: float
    gpu_allocated: Dict[int, float] = field(default_factory=dict)
    gpu_reserved: Dict[int, float] = field(default_factory=dict)
    gpu_max_allocated: Dict[int, float] = field(default_factory=dict)
    process_rss: float = 0.0
    process_vms: float = 0.0
    tag: Optional[str] = None
    component: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """
    Comprehensive memory profiler for protein design pipeline.
    
    This profiler tracks memory usage across different components of the system,
    identifies bottlenecks, and provides actionable insights for optimization.
    
    Args:
        enabled: Whether profiling is enabled (default: True)
        track_gpu: Whether to track GPU memory usage (default: True if CUDA available)
        sample_interval: Interval between automatic samples in seconds (default: 1.0)
        max_snapshots: Maximum number of snapshots to retain (default: 1000)
        profile_components: List of components to profile specifically
        
    Example:
        >>> profiler = MemoryProfiler()
        >>> with profiler.profile("model_training"):
        ...     # Training code here
        ...     pass
        >>> report = profiler.get_report()
        >>> profiler.save_report("memory_profile.json")
    """
    
    def __init__(
        self,
        enabled: bool = True,
        track_gpu: bool = None,
        sample_interval: float = 1.0,
        max_snapshots: int = 1000,
        profile_components: Optional[List[str]] = None
    ):
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.max_snapshots = max_snapshots
        self.profile_components = profile_components or []
        
        # GPU tracking setup
        if track_gpu is None:
            self.track_gpu = torch.cuda.is_available()
        else:
            self.track_gpu = track_gpu and torch.cuda.is_available()
        
        # Initialize tracking state
        self.snapshots: List[MemorySnapshot] = []
        self.component_stack: List[str] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self.peak_memory: Dict[str, float] = {}
        self.start_time = time.time()
        
        # Thread safety lock
        self._lock = threading.RLock()
        
        # Process handle for detailed process monitoring (with error handling)
        self.process = None
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                warnings.warn(f"Cannot access process info: {e}. Process monitoring disabled.", UserWarning)
        
        # Get initial baseline
        if self.enabled:
            self.baseline_snapshot = self._take_snapshot("baseline")
    
    def _take_snapshot(
        self, 
        tag: Optional[str] = None, 
        component: Optional[str] = None
    ) -> MemorySnapshot:
        """Take a memory snapshot with thread safety and error handling."""
        if not self.enabled:
            return MemorySnapshot(timestamp=time.time(), cpu_total=0, cpu_available=0, cpu_percent=0)
        
        with self._lock:  # Thread safety
            timestamp = time.time()
            
            # CPU memory info with fallback
            cpu_total = cpu_available = cpu_percent = 0.0
            if PSUTIL_AVAILABLE:
                try:
                    memory = psutil.virtual_memory()
                    cpu_total = memory.total / (1024 ** 2)  # MB
                    cpu_available = memory.available / (1024 ** 2)  # MB
                    cpu_percent = memory.percent
                except Exception as e:
                    warnings.warn(f"Failed to get CPU memory info: {e}", UserWarning)
            
            # Process memory info with fallback
            process_rss = process_vms = 0.0
            if self.process is not None:
                try:
                    process_memory = self.process.memory_info()
                    process_rss = process_memory.rss / (1024 ** 2)  # MB
                    process_vms = process_memory.vms / (1024 ** 2)  # MB
                except Exception as e:
                    # Don't spam warnings for every snapshot
                    if tag == "baseline":  # Only warn once at initialization
                        warnings.warn(f"Failed to get process memory info: {e}", UserWarning)
        
            # GPU memory info with robust error handling
            gpu_allocated = {}
            gpu_reserved = {}
            gpu_max_allocated = {}
            
            if self.track_gpu and torch.cuda.is_available():
                try:
                    device_count = torch.cuda.device_count()
                    for device_id in range(device_count):
                        try:
                            # Check if device is accessible
                            with torch.cuda.device(device_id):
                                allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)  # MB
                                reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)  # MB
                                max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)  # MB
                                
                                gpu_allocated[device_id] = allocated
                                gpu_reserved[device_id] = reserved
                                gpu_max_allocated[device_id] = max_allocated
                        except (RuntimeError, AssertionError) as e:
                            # Device might not be available or access denied
                            if tag == "baseline":  # Only warn once
                                warnings.warn(f"Cannot access GPU device {device_id}: {e}", UserWarning)
                            continue
                except Exception as e:
                    if tag == "baseline":  # Only warn once
                        warnings.warn(f"GPU memory monitoring failed: {e}", UserWarning)
        
        # Use current component from stack
        current_component = component or (self.component_stack[-1] if self.component_stack else None)
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            cpu_total=cpu_total,
            cpu_available=cpu_available,
            cpu_percent=cpu_percent,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_max_allocated=gpu_max_allocated,
            process_rss=process_rss,
            process_vms=process_vms,
            tag=tag,
            component=current_component
        )
        
        return snapshot
    
    def add_snapshot(
        self, 
        tag: Optional[str] = None, 
        component: Optional[str] = None,
        **custom_metrics
    ):
        """Add a memory snapshot with optional custom metrics and thread safety."""
        if not self.enabled:
            return
        
        snapshot = self._take_snapshot(tag, component)
        snapshot.custom_metrics.update(custom_metrics)
        
        with self._lock:  # Thread safety for snapshot list modification
            self.snapshots.append(snapshot)
            
            # Maintain max snapshots limit with circular buffer behavior
            if len(self.snapshots) > self.max_snapshots:
                # Remove oldest snapshots in batches for efficiency
                num_to_remove = max(1, len(self.snapshots) - self.max_snapshots)
                self.snapshots = self.snapshots[num_to_remove:]
            
            # Update peak memory tracking
            if component:
                current_memory = snapshot.process_rss + sum(snapshot.gpu_allocated.values())
                self.peak_memory[component] = max(
                    self.peak_memory.get(component, 0), 
                    current_memory
                )
    
    @contextmanager
    def profile(self, component: str, **custom_metrics):
        """
        Context manager for profiling a specific component.
        
        Args:
            component: Name of the component being profiled
            **custom_metrics: Additional metrics to track
            
        Example:
            >>> with profiler.profile("model_forward"):
            ...     output = model(input_data)
        """
        if not self.enabled:
            yield
            return
        
        # Push component to stack
        self.component_stack.append(component)
        
        # Take start snapshot
        start_snapshot = self._take_snapshot(f"{component}_start", component)
        start_snapshot.custom_metrics.update(custom_metrics)
        self.snapshots.append(start_snapshot)
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Take end snapshot
            end_time = time.time()
            duration = end_time - start_time
            
            end_snapshot = self._take_snapshot(f"{component}_end", component)
            end_snapshot.custom_metrics.update(custom_metrics)
            end_snapshot.custom_metrics['duration_seconds'] = duration
            self.snapshots.append(end_snapshot)
            
            # Pop component from stack
            if self.component_stack and self.component_stack[-1] == component:
                self.component_stack.pop()
    
    def profile_function(self, func: Callable, component: str, *args, **kwargs):
        """
        Profile a function call.
        
        Args:
            func: Function to profile
            component: Component name for profiling
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result and memory statistics
        """
        if not self.enabled:
            return func(*args, **kwargs), {}
        
        with self.profile(component) as _:
            result = func(*args, **kwargs)
        
        # Get memory statistics for this component
        component_snapshots = [s for s in self.snapshots if s.component == component]
        if len(component_snapshots) >= 2:
            start_snap = component_snapshots[-2]
            end_snap = component_snapshots[-1]
            
            memory_stats = {
                'memory_delta_mb': end_snap.process_rss - start_snap.process_rss,
                'gpu_delta_mb': sum(end_snap.gpu_allocated.values()) - sum(start_snap.gpu_allocated.values()),
                'peak_memory_mb': self.peak_memory.get(component, 0),
                'duration_seconds': end_snap.custom_metrics.get('duration_seconds', 0)
            }
        else:
            memory_stats = {}
        
        return result, memory_stats
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if self.track_gpu:
            for device_id in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(device_id)
                except RuntimeError:
                    continue
        
        self.peak_memory.clear()
    
    def force_gc(self):
        """Force garbage collection and GPU cache clearing."""
        gc.collect()
        if self.track_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_growth(self, component: str) -> Dict[str, float]:
        """
        Get memory growth statistics for a specific component.
        
        Args:
            component: Component name to analyze
            
        Returns:
            Dictionary with memory growth statistics
        """
        component_snapshots = [s for s in self.snapshots if s.component == component]
        
        if len(component_snapshots) < 2:
            return {
                'cpu_growth_mb': 0.0,
                'gpu_growth_mb': 0.0,
                'snapshots_count': len(component_snapshots)
            }
        
        # Compare first and last snapshots
        first = component_snapshots[0]
        last = component_snapshots[-1]
        
        cpu_growth = last.process_rss - first.process_rss
        gpu_growth = sum(last.gpu_allocated.values()) - sum(first.gpu_allocated.values())
        
        return {
            'cpu_growth_mb': cpu_growth,
            'gpu_growth_mb': gpu_growth,
            'snapshots_count': len(component_snapshots),
            'peak_cpu_mb': max(s.process_rss for s in component_snapshots),
            'peak_gpu_mb': max(sum(s.gpu_allocated.values()) for s in component_snapshots)
        }
    
    def identify_bottlenecks(self, threshold_mb: float = 100.0) -> Dict[str, Any]:
        """
        Identify memory bottlenecks in the profiling data.
        
        Args:
            threshold_mb: Memory usage threshold for bottleneck identification
            
        Returns:
            Dictionary with bottleneck analysis
        """
        bottlenecks = {
            'high_memory_components': [],
            'growing_memory_components': [],
            'gpu_memory_issues': [],
            'recommendations': []
        }
        
        # Analyze each component
        components = set(s.component for s in self.snapshots if s.component)
        
        for component in components:
            growth_stats = self.get_memory_growth(component)
            
            # High memory usage
            if growth_stats['peak_cpu_mb'] > threshold_mb:
                bottlenecks['high_memory_components'].append({
                    'component': component,
                    'peak_cpu_mb': growth_stats['peak_cpu_mb'],
                    'peak_gpu_mb': growth_stats['peak_gpu_mb']
                })
            
            # Memory growth
            if growth_stats['cpu_growth_mb'] > threshold_mb * 0.1:  # 10% of threshold
                bottlenecks['growing_memory_components'].append({
                    'component': component,
                    'cpu_growth_mb': growth_stats['cpu_growth_mb'],
                    'gpu_growth_mb': growth_stats['gpu_growth_mb']
                })
        
        # GPU-specific issues
        if self.track_gpu:
            for device_id in range(torch.cuda.device_count()):
                device_snapshots = [s for s in self.snapshots if device_id in s.gpu_allocated]
                if device_snapshots:
                    peak_allocated = max(s.gpu_allocated[device_id] for s in device_snapshots)
                    peak_reserved = max(s.gpu_reserved[device_id] for s in device_snapshots)
                    
                    # Check for inefficient memory usage
                    if peak_reserved > peak_allocated * 2:  # More than 2x allocated
                        bottlenecks['gpu_memory_issues'].append({
                            'device_id': device_id,
                            'peak_allocated_mb': peak_allocated,
                            'peak_reserved_mb': peak_reserved,
                            'efficiency_ratio': peak_allocated / peak_reserved if peak_reserved > 0 else 0
                        })
        
        # Generate recommendations
        if bottlenecks['high_memory_components']:
            bottlenecks['recommendations'].append(
                "Consider implementing gradient checkpointing for high-memory components"
            )
        
        if bottlenecks['growing_memory_components']:
            bottlenecks['recommendations'].append(
                "Investigate memory leaks in components with growing memory usage"
            )
        
        if bottlenecks['gpu_memory_issues']:
            bottlenecks['recommendations'].append(
                "GPU memory fragmentation detected - consider torch.cuda.empty_cache() calls"
            )
        
        return bottlenecks
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive memory profiling report.
        
        Returns:
            Dictionary with complete profiling analysis
        """
        if not self.snapshots:
            return {
                'enabled': self.enabled,
                'snapshots_count': 0,
                'message': 'No profiling data available'
            }
        
        # Basic statistics
        total_duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        components = list(set(s.component for s in self.snapshots if s.component))
        
        # Memory statistics
        cpu_usage = [s.process_rss for s in self.snapshots]
        gpu_usage = [sum(s.gpu_allocated.values()) for s in self.snapshots]
        
        peak_cpu = max(cpu_usage) if cpu_usage else 0
        peak_gpu = max(gpu_usage) if gpu_usage else 0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
        avg_gpu = np.mean(gpu_usage) if gpu_usage else 0
        
        # Component analysis
        component_stats = {}
        for component in components:
            component_stats[component] = self.get_memory_growth(component)
        
        # Bottleneck analysis
        bottlenecks = self.identify_bottlenecks()
        
        report = {
            'profiling_info': {
                'enabled': self.enabled,
                'track_gpu': self.track_gpu,
                'total_duration_seconds': total_duration,
                'snapshots_count': len(self.snapshots),
                'components_profiled': components
            },
            'memory_summary': {
                'peak_cpu_mb': peak_cpu,
                'peak_gpu_mb': peak_gpu,
                'average_cpu_mb': avg_cpu,
                'average_gpu_mb': avg_gpu,
                'baseline_cpu_mb': self.baseline_snapshot.process_rss if self.baseline_snapshot else 0,
                'baseline_gpu_mb': sum(self.baseline_snapshot.gpu_allocated.values()) if self.baseline_snapshot else 0
            },
            'component_analysis': component_stats,
            'bottlenecks': bottlenecks,
            'optimization_recommendations': bottlenecks['recommendations']
        }
        
        return report
    
    def save_report(self, filepath: str, include_snapshots: bool = False):
        """
        Save profiling report to file.
        
        Args:
            filepath: Path to save report
            include_snapshots: Whether to include all snapshot data
        """
        report = self.get_report()
        
        if include_snapshots:
            # Convert snapshots to serializable format
            snapshots_data = []
            for snapshot in self.snapshots:
                snapshot_dict = {
                    'timestamp': snapshot.timestamp,
                    'cpu_total': snapshot.cpu_total,
                    'cpu_available': snapshot.cpu_available,
                    'cpu_percent': snapshot.cpu_percent,
                    'gpu_allocated': snapshot.gpu_allocated,
                    'gpu_reserved': snapshot.gpu_reserved,
                    'gpu_max_allocated': snapshot.gpu_max_allocated,
                    'process_rss': snapshot.process_rss,
                    'process_vms': snapshot.process_vms,
                    'tag': snapshot.tag,
                    'component': snapshot.component,
                    'custom_metrics': snapshot.custom_metrics
                }
                snapshots_data.append(snapshot_dict)
            
            report['snapshots'] = snapshots_data
        
        # Add metadata
        report['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
                'gpu_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def print_summary(self):
        """Print a summary of memory usage to console."""
        if not self.enabled or not self.snapshots:
            print("Memory profiler not enabled or no data collected")
            return
        
        report = self.get_report()
        
        print("\n" + "="*60)
        print("MEMORY PROFILING SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Duration: {report['profiling_info']['total_duration_seconds']:.1f}s")
        print(f"Snapshots: {report['profiling_info']['snapshots_count']}")
        print(f"Components: {len(report['profiling_info']['components_profiled'])}")
        
        # Memory summary
        memory = report['memory_summary']
        print(f"\nMemory Usage:")
        print(f"  Peak CPU: {memory['peak_cpu_mb']:.1f} MB")
        print(f"  Peak GPU: {memory['peak_gpu_mb']:.1f} MB")
        print(f"  Avg CPU:  {memory['average_cpu_mb']:.1f} MB")
        print(f"  Avg GPU:  {memory['average_gpu_mb']:.1f} MB")
        
        # Component analysis
        if report['component_analysis']:
            print(f"\nComponent Analysis:")
            for component, stats in report['component_analysis'].items():
                print(f"  {component}:")
                print(f"    CPU Growth: {stats['cpu_growth_mb']:.1f} MB")
                print(f"    GPU Growth: {stats['gpu_growth_mb']:.1f} MB")
                print(f"    Peak CPU: {stats['peak_cpu_mb']:.1f} MB")
        
        # Recommendations
        if report['optimization_recommendations']:
            print(f"\nOptimization Recommendations:")
            for i, rec in enumerate(report['optimization_recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*60 + "\n")


# Convenience functions for easy profiling
_global_profiler = None
_global_profiler_lock = threading.RLock()

def get_profiler() -> MemoryProfiler:
    """Get the global memory profiler instance with thread safety."""
    global _global_profiler
    with _global_profiler_lock:
        if _global_profiler is None:
            _global_profiler = MemoryProfiler(
                # Use conservative settings for global profiler
                sample_interval=5.0,  # Less frequent sampling
                max_snapshots=100     # Lower memory footprint
            )
        return _global_profiler

def enable_profiling():
    """Enable global memory profiling."""
    profiler = get_profiler()
    profiler.enabled = True

def disable_profiling():
    """Disable global memory profiling."""
    profiler = get_profiler()
    profiler.enabled = False

@contextmanager
def profile_memory(component: str, **custom_metrics):
    """
    Context manager for profiling memory usage.
    
    Args:
        component: Name of component being profiled
        **custom_metrics: Additional metrics to track
    """
    profiler = get_profiler()
    with profiler.profile(component, **custom_metrics):
        yield

def add_memory_checkpoint(tag: str, **custom_metrics):
    """Add a memory checkpoint with optional custom metrics."""
    profiler = get_profiler()
    profiler.add_snapshot(tag=tag, **custom_metrics)

def print_memory_summary():
    """Print memory profiling summary."""
    profiler = get_profiler()
    profiler.print_summary()

def save_memory_report(filepath: str, include_snapshots: bool = False):
    """Save memory profiling report to file."""
    profiler = get_profiler()
    profiler.save_report(filepath, include_snapshots)


if __name__ == "__main__":
    # Example usage and testing with proper error handling
    print("Testing MemoryProfiler...")
    
    try:
        # Test basic profiling
        profiler = MemoryProfiler()
        print(f"✓ Profiler initialized (psutil: {PSUTIL_AVAILABLE}, GPU: {profiler.track_gpu})")
        
        with profiler.profile("test_allocation"):
            # Allocate some memory
            test_data = torch.randn(100, 100)  # Smaller for testing
            if torch.cuda.is_available():
                test_data = test_data.cuda()
        print("✓ Basic profiling test passed")
        
        # Test function profiling
        def test_function(size):
            return torch.randn(size, size)
        
        result, stats = profiler.profile_function(test_function, "tensor_creation", 50)
        print(f"✓ Function profiling stats: {stats}")
        
        # Generate and print report
        profiler.print_summary()
        print("✓ Report generation test passed")
        
        # Test global profiler
        with profile_memory("global_test"):
            data = torch.zeros(50, 50)
        print("✓ Global profiler test passed")
        
        print_memory_summary()
        print("✓ All memory profiler tests completed successfully")
        
    except Exception as e:
        print(f"✗ Memory profiler test failed: {e}")
        import traceback
        traceback.print_exc()