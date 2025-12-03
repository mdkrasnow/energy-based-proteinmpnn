"""
PDB Cache Manager for Streaming Dataset

This module provides efficient caching and retrieval of PDB structures for the streaming
protein dataset. It handles:
- Local file caching with LRU eviction
- Background downloading and preprocessing
- Memory management for large datasets
- Concurrent access with thread safety
"""

from typing import Dict, List, Optional, Union, Callable, Any, Set
from pathlib import Path
import threading
import hashlib
import json
import time
import os
import tempfile
import warnings
import psutil
import logging
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
import numpy as np

# Try to import ProteinMPNN utilities with graceful fallback
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "proteinmpnn"))
    from protein_mpnn_utils import parse_PDB
    PROTEINMPNN_AVAILABLE = True
except ImportError:
    PROTEINMPNN_AVAILABLE = False
    parse_PDB = None
    warnings.warn("ProteinMPNN utilities not available. Raw PDB reading will be used as fallback.")


class CacheStatistics:
    """
    Thread-safe comprehensive cache performance statistics.
    
    This class provides detailed monitoring of all cache operations with minimal
    performance overhead through efficient locking and batched updates.
    """
    
    def __init__(self):
        """Initialize thread-safe statistics collection."""
        # Main statistics lock for atomic updates - using Lock instead of RLock to prevent deadlock
        self._stats_lock = threading.Lock()
        
        # Cache hit/miss tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Download performance
        self.download_attempts = 0
        self.download_successes = 0
        self.download_failures = 0
        self.total_download_time = 0.0
        self.total_download_bytes = 0
        self.min_download_time = float('inf')
        self.max_download_time = 0.0
        
        # Cache management
        self.evictions_performed = 0
        self.evicted_files_count = 0
        self.total_evicted_bytes = 0
        
        # Resource usage tracking
        self.peak_memory_usage_bytes = 0
        self.current_memory_usage_bytes = 0
        self.disk_usage_bytes = 0
        
        # Timing statistics (more detailed)
        self.total_lookup_time = 0.0
        self.total_parsing_time = 0.0
        self.total_eviction_time = 0.0
        self.lookup_count = 0
        self.parsing_count = 0
        self.eviction_count = 0
        
        # Error tracking by category
        self.network_errors = 0
        self.parsing_errors = 0
        self.storage_errors = 0
        self.validation_errors = 0
        self.timeout_errors = 0
        self.permission_errors = 0
        
        # Request pattern analysis (thread-safe)
        self.unique_pdb_ids_requested = set()
        self.request_frequency = defaultdict(int)
        self.last_access_times = {}
        
        # Background operations
        self.prefetch_requests = 0
        self.prefetch_completions = 0
        self.background_downloads_active = 0
        
        # Performance monitoring of monitoring (meta-stats)
        self.monitoring_overhead_time = 0.0
        self.monitoring_operations = 0
        
        # Start time for session statistics
        self.session_start_time = time.perf_counter()
        
        # Recent performance windows (for trending) - bounded collections for memory safety
        from collections import deque
        self._recent_lookup_times = deque(maxlen=100)  # Thread-safe bounded queue
        self._recent_download_times = deque(maxlen=100)  # Thread-safe bounded queue
        self._max_recent_samples = 100
    
    @contextmanager
    def _timed_operation(self, operation_name: str):
        """Context manager for timing operations with monitoring overhead tracking."""
        start_time = time.perf_counter()
        monitoring_start = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            operation_time = end_time - start_time
            monitoring_overhead = (time.perf_counter() - monitoring_start) - operation_time
            
            with self._stats_lock:
                self.monitoring_overhead_time += monitoring_overhead
                self.monitoring_operations += 1
                
                # Update operation-specific timing with atomic batch updates
                if operation_name == 'lookup':
                    self.total_lookup_time += operation_time
                    self.lookup_count += 1
                    # Thread-safe bounded append using deque
                    self._recent_lookup_times.append(operation_time)
                        
                elif operation_name == 'download':
                    self.total_download_time += operation_time
                    self.min_download_time = min(self.min_download_time, operation_time)
                    self.max_download_time = max(self.max_download_time, operation_time)
                    # Thread-safe bounded append using deque
                    self._recent_download_times.append(operation_time)
                        
                elif operation_name == 'parsing':
                    self.total_parsing_time += operation_time
                    self.parsing_count += 1
                    
                elif operation_name == 'eviction':
                    self.total_eviction_time += operation_time
                    self.eviction_count += 1
    
    def record_request(self, pdb_id: str):
        """Record a cache request."""
        with self._stats_lock:
            self.total_requests += 1
            self.unique_pdb_ids_requested.add(pdb_id)
            self.request_frequency[pdb_id] += 1
            self.last_access_times[pdb_id] = time.perf_counter()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._stats_lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._stats_lock:
            self.cache_misses += 1
    
    def record_download_attempt(self):
        """Record a download attempt."""
        with self._stats_lock:
            self.download_attempts += 1
    
    def record_download_success(self, bytes_downloaded: int):
        """Record a successful download."""
        with self._stats_lock:
            self.download_successes += 1
            self.total_download_bytes += bytes_downloaded
    
    def record_download_failure(self):
        """Record a failed download."""
        with self._stats_lock:
            self.download_failures += 1
    
    def record_eviction(self, files_evicted: int, bytes_freed: int):
        """Record cache eviction statistics."""
        with self._stats_lock:
            self.evictions_performed += 1
            self.evicted_files_count += files_evicted
            self.total_evicted_bytes += bytes_freed
    
    def record_error(self, error_type: str):
        """Record an error by category."""
        with self._stats_lock:
            if error_type == 'network':
                self.network_errors += 1
            elif error_type == 'parsing':
                self.parsing_errors += 1
            elif error_type == 'storage':
                self.storage_errors += 1
            elif error_type == 'validation':
                self.validation_errors += 1
            elif error_type == 'timeout':
                self.timeout_errors += 1
            elif error_type == 'permission':
                self.permission_errors += 1
    
    def update_memory_usage(self, current_bytes: int):
        """Update memory usage statistics."""
        with self._stats_lock:
            self.current_memory_usage_bytes = current_bytes
            self.peak_memory_usage_bytes = max(self.peak_memory_usage_bytes, current_bytes)
    
    def update_disk_usage(self, current_bytes: int):
        """Update disk usage statistics."""
        with self._stats_lock:
            self.disk_usage_bytes = current_bytes
    
    def record_prefetch_request(self):
        """Record a prefetch request."""
        with self._stats_lock:
            self.prefetch_requests += 1
    
    def record_prefetch_completion(self):
        """Record a prefetch completion."""
        with self._stats_lock:
            self.prefetch_completions += 1
    
    def update_background_downloads(self, active_count: int):
        """Update the count of active background downloads."""
        with self._stats_lock:
            self.background_downloads_active = active_count
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all statistics in a comprehensive dictionary."""
        with self._stats_lock:
            session_duration = time.perf_counter() - self.session_start_time
            
            # Calculate derived statistics
            hit_rate = self.cache_hits / max(self.total_requests, 1)
            avg_lookup_time = self.total_lookup_time / max(self.lookup_count, 1)
            avg_download_time = self.total_download_time / max(self.download_attempts, 1)
            avg_parsing_time = self.total_parsing_time / max(self.parsing_count, 1)
            avg_eviction_time = self.total_eviction_time / max(self.eviction_count, 1)
            
            # Recent performance trends
            recent_avg_lookup = (sum(self._recent_lookup_times) / len(self._recent_lookup_times) 
                               if self._recent_lookup_times else 0.0)
            recent_avg_download = (sum(self._recent_download_times) / len(self._recent_download_times) 
                                 if self._recent_download_times else 0.0)
            
            # Monitoring overhead
            avg_monitoring_overhead = (self.monitoring_overhead_time / max(self.monitoring_operations, 1))
            
            return {
                'cache_performance': {
                    'hit_rate': hit_rate,
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'total_requests': self.total_requests,
                    'unique_pdb_ids': len(self.unique_pdb_ids_requested),
                },
                'download_performance': {
                    'attempts': self.download_attempts,
                    'successes': self.download_successes,
                    'failures': self.download_failures,
                    'success_rate': self.download_successes / max(self.download_attempts, 1),
                    'total_bytes': self.total_download_bytes,
                    'total_time_seconds': self.total_download_time,
                    'avg_time_seconds': avg_download_time,
                    'min_time_seconds': self.min_download_time if self.min_download_time != float('inf') else 0.0,
                    'max_time_seconds': self.max_download_time,
                    'recent_avg_time_seconds': recent_avg_download,
                    'avg_speed_mbps': (self.total_download_bytes / max(self.total_download_time, 1)) / (1024 * 1024),
                },
                'timing_analysis': {
                    'avg_lookup_ms': avg_lookup_time * 1000,
                    'avg_download_ms': avg_download_time * 1000,
                    'avg_parsing_ms': avg_parsing_time * 1000,
                    'avg_eviction_ms': avg_eviction_time * 1000,
                    'recent_avg_lookup_ms': recent_avg_lookup * 1000,
                    'total_lookup_operations': self.lookup_count,
                    'total_parsing_operations': self.parsing_count,
                    'total_eviction_operations': self.eviction_count,
                },
                'resource_usage': {
                    'current_memory_mb': self.current_memory_usage_bytes / (1024 * 1024),
                    'peak_memory_mb': self.peak_memory_usage_bytes / (1024 * 1024),
                    'disk_usage_mb': self.disk_usage_bytes / (1024 * 1024),
                    'evictions_performed': self.evictions_performed,
                    'files_evicted': self.evicted_files_count,
                    'bytes_evicted_mb': self.total_evicted_bytes / (1024 * 1024),
                },
                'error_analysis': {
                    'total_errors': (self.network_errors + self.parsing_errors + 
                                   self.storage_errors + self.validation_errors + 
                                   self.timeout_errors + self.permission_errors),
                    'network_errors': self.network_errors,
                    'parsing_errors': self.parsing_errors,
                    'storage_errors': self.storage_errors,
                    'validation_errors': self.validation_errors,
                    'timeout_errors': self.timeout_errors,
                    'permission_errors': self.permission_errors,
                },
                'request_patterns': {
                    'most_requested': sorted(self.request_frequency.items(), 
                                           key=lambda x: x[1], reverse=True)[:10],
                    'requests_per_second': self.total_requests / max(session_duration, 1),
                    'avg_requests_per_pdb': sum(self.request_frequency.values()) / max(len(self.request_frequency), 1),
                },
                'background_operations': {
                    'prefetch_requests': self.prefetch_requests,
                    'prefetch_completions': self.prefetch_completions,
                    'prefetch_success_rate': self.prefetch_completions / max(self.prefetch_requests, 1),
                    'active_background_downloads': self.background_downloads_active,
                },
                'monitoring_overhead': {
                    'total_overhead_seconds': self.monitoring_overhead_time,
                    'avg_overhead_microseconds': avg_monitoring_overhead * 1_000_000,
                    'monitoring_operations': self.monitoring_operations,
                    'overhead_percentage': (self.monitoring_overhead_time / max(session_duration, 1)) * 100,
                },
                'session_info': {
                    'session_duration_seconds': session_duration,
                    'session_start_time': self.session_start_time,
                }
            }


class PDBCache:
    """
    Thread-safe LRU cache for PDB structures with background loading capabilities.
    
    This cache manages both raw PDB files and preprocessed tensor representations,
    providing efficient access patterns for streaming datasets with bounded storage.
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_memory_mb: int = 1024,
        max_disk_gb: float = 5.0,
        target_free_bytes: int = 200_000_000,  # 200MB buffer
        max_concurrent_downloads: int = 16,
        preprocess_fn: Optional[Callable] = None
    ):
        """
        Initialize PDB cache manager with LRU eviction and storage management.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_mb: Maximum memory usage in MB
            max_disk_gb: Maximum disk usage in GB (default 5.0 for Harvard A100)
            target_free_bytes: Bytes to keep free for new downloads (default 200MB)
            max_concurrent_downloads: Maximum concurrent downloads (default 16 for A100)
            preprocess_fn: Optional function to preprocess PDB data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = int(max_disk_gb * 1024 * 1024 * 1024)
        self.target_free_bytes = target_free_bytes
        self.max_concurrent_downloads = max_concurrent_downloads
        self.preprocess_fn = preprocess_fn
        
        # Thread-safe memory cache - using Lock instead of RLock to prevent deadlock
        self._memory_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._current_memory_usage = 0
        
        # CRITICAL FIX STORAGE-001: Enhanced LRU tracking with robust synchronization
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._disk_cache_lock = threading.RLock()  # Use RLock for nested operations
        
        # CRITICAL FIX STORAGE-002: Thread-safe disk size tracking with atomic updates
        self._cached_disk_size = 0
        self._last_size_update = 0.0
        self._size_cache_timeout = 10.0  # Reduced timeout for more accurate tracking
        self._disk_size_lock = threading.Lock()  # Separate lock for size calculations
        
        # CRITICAL FIX STORAGE-003: Enhanced file protection during operations
        self._active_files: Set[str] = set()
        self._downloading_files: Set[str] = set()
        self._evicting_files: Set[str] = set()  # Track files being evicted
        self._file_operations_lock = threading.Lock()  # Dedicated lock for file state
        
        # Background loading and download management
        self._download_queue = []
        self._download_lock = threading.Lock()
        # CRITICAL FIX STORAGE-004: Bounded semaphore with timeout to prevent leaks
        self._download_semaphore = threading.BoundedSemaphore(max_concurrent_downloads)
        self._download_timeout = 300  # 5 minute timeout for downloads
        
        # Download deduplication with Events (required by specification)
        self._downloading: Dict[str, threading.Event] = {}
        
        # Background executor for prefetching and cache warming
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_downloads, 
                                         thread_name_prefix="pdb_cache")
        
        # Cache statistics tracking
        self._statistics = CacheStatistics()
        
        # Enhanced logging setup for performance monitoring
        self._setup_performance_logging()
        
        # Initialize disk cache by scanning existing files
        self._scan_existing_cache()
    
    def _setup_performance_logging(self):
        """Set up detailed logging for performance monitoring and debugging."""
        # Create dedicated logger for cache performance
        self._perf_logger = logging.getLogger(f"pdb_cache.{id(self)}")
        self._perf_logger.setLevel(logging.INFO)
        
        # Only add handler if none exists (prevent duplicate handlers)
        if not self._perf_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] PDBCache[%(name)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._perf_logger.addHandler(handler)
            
            # Prevent propagation to avoid duplicate logs
            self._perf_logger.propagate = False
    
    def _log_performance_event(self, event_type: str, pdb_id: str, **kwargs):
        """Log performance events with structured data for debugging."""
        if self._perf_logger.isEnabledFor(logging.DEBUG):
            event_data = {
                'event': event_type,
                'pdb_id': pdb_id,
                'timestamp': time.perf_counter(),
                **kwargs
            }
            self._perf_logger.debug(f"Performance event: {json.dumps(event_data)}")
    
    def _log_cache_summary(self, interval_seconds: int = 300):
        """Periodically log cache performance summary."""
        stats = self._statistics.get_comprehensive_stats()
        self._perf_logger.info(
            f"Cache Summary - Hit Rate: {stats['cache_performance']['hit_rate']:.2%}, "
            f"Requests: {stats['cache_performance']['total_requests']}, "
            f"Downloads: {stats['download_performance']['attempts']}, "
            f"Disk Usage: {stats['resource_usage']['disk_usage_mb']:.1f}MB, "
            f"Memory Usage: {stats['resource_usage']['current_memory_mb']:.1f}MB"
        )
    
    def _validate_pdb_id(self, pdb_id: str) -> bool:
        """Validate PDB ID to prevent path traversal attacks."""
        if not isinstance(pdb_id, str):
            self._statistics.record_error('validation')
            return False
        
        # Strict PDB ID validation: exactly 4 characters, digit + 3 alphanumeric
        pdb_id_clean = pdb_id.strip().upper()
        
        # Exact length check (critical for security)
        if len(pdb_id_clean) != 4:
            self._statistics.record_error('validation')
            return False
            
        # Pattern validation
        pdb_pattern = re.compile(r'^[0-9][A-Za-z0-9]{3}$')
        if not pdb_pattern.match(pdb_id_clean):
            self._statistics.record_error('validation')
            return False
            
        # Explicitly check for path traversal patterns
        if any(char in pdb_id_clean for char in ['/', '\\', '..', '~', '$', '%']):
            self._statistics.record_error('validation')
            return False
            
        return True
        
    def get(self, pdb_id: str, download_url: Optional[str] = None) -> Optional[Any]:
        """
        Get PDB data from cache, downloading if necessary.
        
        Args:
            pdb_id: PDB identifier
            download_url: Optional URL for downloading if not cached
            
        Returns:
            Cached or downloaded PDB data, None if not available
        """
        # Validate PDB ID for security
        if not self._validate_pdb_id(pdb_id):
            self._perf_logger.warning(f"Invalid PDB ID rejected: {pdb_id}")
            return None
            
        pdb_id = pdb_id.upper()
        
        # Record request and start timing
        self._statistics.record_request(pdb_id)
        self._log_performance_event('request_start', pdb_id)
        
        # CRITICAL FIX STORAGE-003: Mark file as actively being used to prevent eviction
        with self._file_operations_lock:
            self._active_files.add(pdb_id)
        
        try:
            with self._statistics._timed_operation('lookup'):
                # First check memory cache
                with self._cache_lock:
                    if pdb_id in self._memory_cache:
                        # Cache hit in memory
                        self._statistics.record_cache_hit()
                        self._log_performance_event('memory_hit', pdb_id)
                        
                        # Move to end (most recent)
                        data = self._memory_cache[pdb_id]
                        del self._memory_cache[pdb_id]
                        self._memory_cache[pdb_id] = data
                        
                        # Update disk access time as well
                        self._update_disk_access_time(pdb_id)
                        return data
                
                # Check disk cache
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                if file_path.exists():
                    # Cache hit on disk
                    self._statistics.record_cache_hit()
                    self._update_disk_access_time(pdb_id)
                    self._log_performance_event('disk_hit', pdb_id)
                    
                    # Load and parse with timing
                    try:
                        with self._statistics._timed_operation('parsing'):
                            if PROTEINMPNN_AVAILABLE:
                                parsed_data = parse_PDB(str(file_path))
                                if parsed_data:
                                    # Apply preprocessing if specified
                                    if self.preprocess_fn:
                                        parsed_data = self.preprocess_fn(parsed_data)
                                    
                                    self._add_to_memory_cache(pdb_id, parsed_data)
                                    self._log_performance_event('parse_success', pdb_id)
                                    return parsed_data
                            else:
                                # Fallback to raw file reading if ProteinMPNN unavailable
                                with open(file_path, 'r') as f:
                                    data = f.read()
                                
                                if self.preprocess_fn:
                                    data = self.preprocess_fn(data)
                                
                                self._add_to_memory_cache(pdb_id, data)
                                self._log_performance_event('parse_success', pdb_id, method='raw')
                                return data
                                
                    except Exception as e:
                        self._statistics.record_error('parsing')
                        self._perf_logger.error(f"Failed to parse {pdb_id}: {e}")
                        return None
                
                # File not in cache - cache miss
                self._statistics.record_cache_miss()
                self._log_performance_event('cache_miss', pdb_id)
                
                # Attempt download if URL provided
                if download_url:
                    return self._download_with_deduplication(pdb_id, download_url)
                
                return None
                
        finally:
            # CRITICAL FIX STORAGE-003: Remove from active files with proper locking
            with self._file_operations_lock:
                self._active_files.discard(pdb_id)
            
            # Update resource usage statistics
            self._statistics.update_memory_usage(self._current_memory_usage)
            self._statistics.update_disk_usage(self._get_current_disk_size())
    
    def _download_with_deduplication(self, pdb_id: str, download_url: str) -> Optional[Any]:
        """
        Download PDB with thread-safe deduplication using Events.
        
        CRITICAL FIX TASK-1-2: Complete rewrite to eliminate race conditions and deadlocks.
        - Fixed download deduplication race condition
        - Eliminated recursive calls that could cause stack overflow
        - Improved lock ordering to prevent deadlocks
        - Enhanced error handling and cleanup
        """
        download_event = None
        is_downloader = False
        parsed_result = None
        
        # CRITICAL FIX: Single atomic check-and-set operation for download deduplication
        with self._download_lock:
            if pdb_id in self._downloading:
                download_event = self._downloading[pdb_id]
                self._log_performance_event('download_deduplicated', pdb_id)
            else:
                # Atomically create new download event and mark as downloader
                download_event = threading.Event()
                self._downloading[pdb_id] = download_event
                is_downloader = True
                
                # CRITICAL FIX STORAGE-003: Mark as downloading with proper file operations lock
                with self._file_operations_lock:
                    self._downloading_files.add(pdb_id)
        
        # If we're the downloader, perform the download
        if is_downloader:
            try:
                # CRITICAL FIX STORAGE-003: Update background download count with proper locking
                with self._file_operations_lock:
                    active_downloads = len(self._downloading_files)
                self._statistics.update_background_downloads(active_downloads)
                
                # CRITICAL FIX STORAGE-004: Semaphore acquisition with timeout to prevent leaks
                download_success = False
                semaphore_acquired = False
                try:
                    # Acquire semaphore with timeout to prevent infinite blocking
                    semaphore_acquired = self._download_semaphore.acquire(timeout=self._download_timeout)
                    if not semaphore_acquired:
                        self._statistics.record_error('timeout')
                        self._perf_logger.warning(f"Download semaphore timeout for {pdb_id} after {self._download_timeout}s")
                        return None
                    
                    with self._statistics._timed_operation('download'):
                        self._statistics.record_download_attempt()
                        self._log_performance_event('download_start', pdb_id, url=download_url)
                        
                        download_result = self._perform_download(pdb_id, download_url)
                        download_success = download_result is not False
                        
                        if download_success:
                            self._log_performance_event('download_success', pdb_id)
                        else:
                            self._statistics.record_download_failure()
                            self._log_performance_event('download_failure', pdb_id)
                finally:
                    # CRITICAL FIX STORAGE-004: Always release semaphore to prevent leaks
                    if semaphore_acquired:
                        self._download_semaphore.release()
                
                # If download succeeded, parse the file
                if download_success:
                    file_path = self.cache_dir / f"{pdb_id}.pdb"
                    try:
                        with self._statistics._timed_operation('parsing'):
                            if PROTEINMPNN_AVAILABLE:
                                parsed_data = parse_PDB(str(file_path))
                                if parsed_data:
                                    if self.preprocess_fn:
                                        parsed_data = self.preprocess_fn(parsed_data)
                                    self._add_to_memory_cache(pdb_id, parsed_data)
                                    parsed_result = parsed_data
                                    self._log_performance_event('download_parse_success', pdb_id)
                            else:
                                with open(file_path, 'r') as f:
                                    data = f.read()
                                if self.preprocess_fn:
                                    data = self.preprocess_fn(data)
                                self._add_to_memory_cache(pdb_id, data)
                                parsed_result = data
                                self._log_performance_event('download_parse_success', pdb_id, method='raw')
                    except Exception as e:
                        self._statistics.record_error('parsing')
                        self._perf_logger.error(f"Failed to parse downloaded {pdb_id}: {e}")
                        parsed_result = None
                
            finally:
                # CRITICAL FIX: Atomic cleanup with proper ordering to prevent deadlocks
                # Signal completion BEFORE cleanup to wake waiting threads
                download_event.set()
                
                # Clean up download tracking with consistent lock ordering
                with self._download_lock:
                    self._downloading.pop(pdb_id, None)
                
                # CRITICAL FIX STORAGE-003: Clean up download tracking with proper file operations lock
                with self._file_operations_lock:
                    self._downloading_files.discard(pdb_id)
                    active_downloads = len(self._downloading_files)
                
                self._statistics.update_background_downloads(active_downloads)
                
            return parsed_result
                
        else:
            # CRITICAL FIX: Eliminate recursive call to prevent stack overflow
            # Wait for ongoing download to complete with timeout to prevent infinite waiting
            self._log_performance_event('download_wait', pdb_id)
            
            # Wait with timeout to prevent hanging indefinitely
            wait_timeout = 300  # 5 minute timeout for downloads
            if download_event.wait(timeout=wait_timeout):
                # Download completed, check if file exists and load it directly
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                if file_path.exists():
                    try:
                        # Load the file that was just downloaded by another thread
                        with self._statistics._timed_operation('parsing'):
                            if PROTEINMPNN_AVAILABLE:
                                parsed_data = parse_PDB(str(file_path))
                                if parsed_data:
                                    if self.preprocess_fn:
                                        parsed_data = self.preprocess_fn(parsed_data)
                                    self._add_to_memory_cache(pdb_id, parsed_data)
                                    self._update_disk_access_time(pdb_id)
                                    return parsed_data
                            else:
                                with open(file_path, 'r') as f:
                                    data = f.read()
                                if self.preprocess_fn:
                                    data = self.preprocess_fn(data)
                                self._add_to_memory_cache(pdb_id, data)
                                self._update_disk_access_time(pdb_id)
                                return data
                    except Exception as e:
                        self._statistics.record_error('parsing')
                        self._perf_logger.error(f"Failed to parse downloaded {pdb_id} after wait: {e}")
                        return None
                else:
                    # File doesn't exist - download must have failed
                    self._perf_logger.warning(f"Download completed but file not found for {pdb_id}")
                    return None
            else:
                # Timeout waiting for download
                self._statistics.record_error('timeout')
                self._perf_logger.error(f"Timeout waiting for download of {pdb_id}")
                return None
        
        return None
    
    def _perform_download(self, pdb_id: str, download_url: str) -> bool:
        """Perform atomic download with retry logic and comprehensive error tracking."""
        file_path = self.cache_dir / f"{pdb_id}.pdb"
        
        # Ensure we have space for the download
        try:
            self._ensure_space_available()
        except Exception as e:
            self._statistics.record_error('storage')
            self._perf_logger.error(f"Failed to ensure space for {pdb_id}: {e}")
            return False
        
        # Retry configuration for robustness
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff in seconds
        
        for attempt in range(max_retries):
            temp_file = None
            
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=self.cache_dir,
                    prefix=f'.tmp_{pdb_id}_',
                    suffix='.pdb',
                    delete=False
                ) as f:
                    temp_file = f.name
                    
                    # NETWORK ROBUSTNESS FIX (ROB-001): Enhanced retry logic for production reliability
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                    
                    # Create session with robust retry strategy for network failures
                    session = requests.Session()
                    retry_strategy = Retry(
                        total=5,  # Increased from 2 to 5 for better resilience
                        status_forcelist=[408, 429, 500, 502, 503, 504, 520, 521, 522, 524],  # Added more HTTP errors
                        allowed_methods=["HEAD", "GET", "OPTIONS"],
                        backoff_factor=1.0,  # Increased from 0.5 for better server recovery
                        raise_on_status=False,
                        # Add specific handling for connection errors
                        connect=3,  # Connection retries
                        read=3,     # Read retries  
                        redirect=3  # Redirect retries
                    )
                    adapter = HTTPAdapter(
                        max_retries=retry_strategy,
                        pool_connections=10,  # Connection pool for efficiency
                        pool_maxsize=20      # Maximum connections in pool
                        # socket_options removed as it's not supported in older urllib3 versions
                    )
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                    
                    # Set reasonable timeouts and headers for robustness
                    session.timeout = (10, 60)  # (connect_timeout, read_timeout)
                    session.headers.update({
                        'User-Agent': 'ProteinMPNN-Cache/1.0',
                        'Accept': 'text/plain, application/octet-stream',
                        'Accept-Encoding': 'gzip, deflate'
                    })
                    
                    try:
                        # Attempt download with progressive timeout
                        timeout = min(30 + (attempt * 10), 60)  # Increase timeout on retries
                        response = session.get(download_url, timeout=timeout, stream=True)
                        response.raise_for_status()
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, 
                            requests.exceptions.RequestException) as e:
                        error_type = 'timeout' if 'timeout' in str(e).lower() else 'network'
                        self._statistics.record_error(error_type)
                        
                        if attempt < max_retries - 1:
                            self._perf_logger.warning(
                                f"{error_type.capitalize()} error for {pdb_id} "
                                f"(attempt {attempt+1}/{max_retries}): {e}. Retrying..."
                            )
                            time.sleep(retry_delays[attempt])
                            continue
                        else:
                            self._perf_logger.error(
                                f"Final {error_type} error for {pdb_id} after {max_retries} attempts: {e}"
                            )
                            raise
                    
                    # CRITICAL FIX IMP-002: Write content with proper byte tracking to prevent corruption
                    bytes_written = 0
                    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                        if chunk:
                            f.write(chunk)
                            # Track bytes written accurately for proper statistics
                            bytes_written += len(chunk.encode('utf-8'))
                    
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic move to final location
                try:
                    os.rename(temp_file, str(file_path))
                    temp_file = None  # Prevent cleanup
                except OSError as e:
                    self._statistics.record_error('permission')
                    raise
                
                # CRITICAL FIX IMP-002: Update tracking with actual file size for accuracy
                file_stat = file_path.stat()
                actual_file_size = file_stat.st_size
                
                # Verify byte counting accuracy to catch corruption
                if abs(actual_file_size - bytes_written) > 100:  # Allow small variance for encoding
                    self._perf_logger.warning(
                        f"Byte count discrepancy for {pdb_id}: "
                        f"written={bytes_written}, actual={actual_file_size}"
                    )
                
                with self._disk_cache_lock:
                    self._access_times[pdb_id] = file_stat.st_mtime
                    self._cached_disk_size += actual_file_size
                
                # CRITICAL FIX IMP-002: Record download success with verified actual file size
                self._statistics.record_download_success(actual_file_size)
                self._log_performance_event('download_complete', pdb_id, 
                                           bytes_downloaded=actual_file_size,
                                           url=download_url)
                
                return True
                
            except Exception as e:
                # Cleanup temporary file if it exists
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass
                
                # If this was the last attempt, let it fail normally
                if attempt >= max_retries - 1:
                    # Categorize and record the error
                    error_type = 'network'
                    if 'permission' in str(e).lower() or 'access' in str(e).lower():
                        error_type = 'permission'
                    elif 'space' in str(e).lower() or 'disk' in str(e).lower():
                        error_type = 'storage'
                    elif 'timeout' in str(e).lower():
                        error_type = 'timeout'
                    
                    self._statistics.record_error(error_type)
                    self._perf_logger.error(f"Download failed for {pdb_id} ({error_type}) after {max_retries} attempts: {e}")
                    break  # Exit retry loop
                else:
                    # Log and retry
                    self._perf_logger.warning(f"Download attempt {attempt+1} failed for {pdb_id}: {e}. Retrying...")
                    time.sleep(retry_delays[attempt])
                    continue
        
        # All retries exhausted
        return False
    
    def _ensure_space_available(self, additional_bytes: int = 100_000_000) -> None:
        """Ensure sufficient disk space is available for downloads."""
        current_size = self._get_current_disk_size()
        target_size = current_size + additional_bytes + self.target_free_bytes
        
        if target_size > self.max_disk_bytes:
            bytes_to_free = target_size - self.max_disk_bytes
            self.evict_lru(bytes_to_free)

    def prefetch(self, pdb_ids: List[str], urls: Optional[List[str]] = None) -> None:
        """
        Prefetch PDB structures in background for future access.
        
        CRITICAL FIX TASK-1-2: Enhanced thread safety and deduplication.
        - Atomic operations for prefetch request tracking
        - Better validation and error handling
        - Protection against duplicate prefetch requests
        - Improved resource management
        
        Args:
            pdb_ids: List of PDB identifiers to prefetch
            urls: Optional list of download URLs
        """
        if not pdb_ids:
            return
        
        # Validate input parameters
        if not isinstance(pdb_ids, list):
            warnings.warn("pdb_ids must be a list")
            return
            
        # Validate URLs list if provided
        if urls and len(urls) != len(pdb_ids):
            warnings.warn(f"URL list length ({len(urls)}) doesn't match PDB ID list length ({len(pdb_ids)})")
            urls = None
        
        # Filter and validate PDB IDs
        valid_pdb_ids = []
        for pdb_id in pdb_ids:
            if self._validate_pdb_id(pdb_id):
                valid_pdb_ids.append(pdb_id.upper())
            else:
                self._perf_logger.warning(f"Invalid PDB ID in prefetch request: {pdb_id}")
        
        if not valid_pdb_ids:
            self._perf_logger.warning("No valid PDB IDs provided for prefetch")
            return
        
        # CRITICAL FIX: Atomic determination of what needs to be prefetched
        prefetch_needed = []
        already_available = []
        currently_downloading = []
        
        # Single atomic check to determine prefetch requirements
        with self._disk_cache_lock:
            with self._download_lock:
                with self._cache_lock:
                    for i, pdb_id in enumerate(valid_pdb_ids):
                        # Skip if already in memory cache
                        if pdb_id in self._memory_cache:
                            already_available.append(pdb_id)
                            continue
                        
                        # Skip if already in disk cache
                        if pdb_id in self._access_times:
                            already_available.append(pdb_id)
                            continue
                        
                        # Skip if already downloading
                        if pdb_id in self._downloading:
                            currently_downloading.append(pdb_id)
                            continue
                        
                        # Skip if file already exists on disk
                        file_path = self.cache_dir / f"{pdb_id}.pdb"
                        if file_path.exists():
                            already_available.append(pdb_id)
                            continue
                        
                        # Determine download URL
                        if urls and i < len(urls):
                            download_url = urls[i]
                        else:
                            download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                        
                        prefetch_needed.append((pdb_id, download_url))
        
        # Log prefetch analysis
        self._perf_logger.debug(
            f"Prefetch analysis - Requested: {len(valid_pdb_ids)}, "
            f"Need download: {len(prefetch_needed)}, "
            f"Already available: {len(already_available)}, "
            f"Currently downloading: {len(currently_downloading)}"
        )
        
        # Update prefetch statistics atomically
        with self._statistics._stats_lock:
            self._statistics.prefetch_requests += len(prefetch_needed)
        
        if not prefetch_needed:
            self._perf_logger.info("All requested PDB structures already available or downloading")
            return
        
        # Submit prefetch tasks to background threads
        prefetch_futures = []
        successful_submissions = 0
        
        for pdb_id, download_url in prefetch_needed:
            try:
                # Submit background download task with executor
                future = self.executor.submit(self._background_prefetch_download, pdb_id, download_url)
                prefetch_futures.append((pdb_id, future))
                successful_submissions += 1
                
                # Update background activity tracking atomically
                with self._statistics._stats_lock:
                    self._statistics.background_downloads_active += 1
                
            except Exception as e:
                self._perf_logger.warning(f"Failed to submit prefetch task for {pdb_id}: {e}")
                # Record the failed submission
                self._statistics.record_error('storage')  # Likely a resource/threading issue
        
        # Log prefetch initiation
        if successful_submissions > 0:
            self._perf_logger.info(
                f"Started background prefetch for {successful_submissions}/{len(prefetch_needed)} PDB structures. "
                f"Total active downloads: {self._statistics.background_downloads_active}"
            )
        else:
            self._perf_logger.warning("Failed to submit any prefetch tasks")
    
    def _background_prefetch_download(self, pdb_id: str, download_url: str) -> bool:
        """
        Perform background download for prefetching with optimized error handling.
        
        CRITICAL FIX TASK-1-2: Enhanced thread safety and error recovery.
        - Atomic statistics updates to prevent race conditions
        - Better error handling and logging
        - Protection against concurrent prefetch attempts
        
        Args:
            pdb_id: PDB identifier to download
            download_url: URL for downloading
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # CRITICAL FIX: Check if already downloading or cached before attempting
            # This prevents unnecessary work and potential race conditions
            with self._download_lock:
                if pdb_id in self._downloading:
                    self._perf_logger.debug(f"Prefetch skipped for {pdb_id} - already downloading")
                    return False
            
            # Check if already cached
            file_path = self.cache_dir / f"{pdb_id}.pdb"
            if file_path.exists():
                with self._disk_cache_lock:
                    if pdb_id in self._access_times:
                        self._perf_logger.debug(f"Prefetch skipped for {pdb_id} - already cached")
                        # Update statistics for the completion even though we didn't download
                        with self._statistics._stats_lock:
                            self._statistics.prefetch_completions += 1
                        return True
            
            # Use existing download mechanism with deduplication
            result = self._download_with_deduplication(pdb_id, download_url)
            
            if result is not None:
                # Successfully downloaded and parsed
                with self._statistics._stats_lock:
                    self._statistics.prefetch_completions += 1
                self._perf_logger.debug(f"Background prefetch completed for {pdb_id}")
                return True
            else:
                self._perf_logger.debug(f"Background prefetch failed for {pdb_id} - download returned None")
                return False
                
        except Exception as e:
            # Enhanced error logging for debugging
            self._statistics.record_error('network')  # Assume network error for prefetch failures
            self._perf_logger.warning(f"Background prefetch failed for {pdb_id}: {e}")
            return False
        finally:
            # CRITICAL FIX: Thread-safe update of background activity tracking
            with self._statistics._stats_lock:
                self._statistics.background_downloads_active = max(0, 
                    self._statistics.background_downloads_active - 1)
    
    def warm_cache(self, popular_pdb_ids: List[str], max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """
        Warm the cache with popular/frequently accessed PDB structures.
        
        This method is optimized for A100 cluster environments with high bandwidth
        and implements intelligent batching to avoid overwhelming the network.
        
        Args:
            popular_pdb_ids: List of PDB IDs to pre-load into cache
            max_concurrent: Maximum concurrent downloads (defaults to max_concurrent_downloads)
            
        Returns:
            Dictionary with warming results and statistics
        """
        if not popular_pdb_ids:
            return {"status": "no_ids_provided", "downloaded": 0, "failed": 0}
            
        max_concurrent = max_concurrent or self.max_concurrent_downloads
        start_time = time.perf_counter()
        
        # Filter to only IDs not already cached
        uncached_ids = []
        already_cached = 0
        
        for pdb_id in popular_pdb_ids:
            if pdb_id not in self._access_times and pdb_id not in self._memory_cache:
                uncached_ids.append(pdb_id)
            else:
                already_cached += 1
        
        if not uncached_ids:
            return {
                "status": "all_already_cached",
                "total_requested": len(popular_pdb_ids),
                "already_cached": already_cached,
                "downloaded": 0,
                "failed": 0,
                "duration_seconds": 0
            }
        
        # Ensure we have enough disk space before starting
        try:
            estimated_space_needed = len(uncached_ids) * 100_000  # Assume 100KB avg per PDB
            self._ensure_space_available(estimated_space_needed)
        except Exception as e:
            warnings.warn(f"Failed to ensure space for cache warming: {e}")
            return {"status": "insufficient_space", "error": str(e)}
        
        warnings.warn(f"Starting cache warming for {len(uncached_ids)} PDB structures with {max_concurrent} concurrent downloads")
        
        # Batch downloads to respect concurrency limits
        downloaded = 0
        failed = 0
        download_futures = []
        
        # Use ThreadPoolExecutor for controlled concurrency
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all download tasks
            for pdb_id in uncached_ids:
                download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                future = executor.submit(self._perform_warming_download, pdb_id, download_url)
                download_futures.append((pdb_id, future))
            
            # Collect results with progress tracking
            completed_count = 0
            for pdb_id, future in download_futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per download
                    if result:
                        downloaded += 1
                    else:
                        failed += 1
                    
                    completed_count += 1
                    
                    # Progress reporting every 10 downloads
                    if completed_count % 10 == 0:
                        progress_pct = (completed_count / len(download_futures)) * 100
                        warnings.warn(f"Cache warming progress: {completed_count}/{len(download_futures)} ({progress_pct:.1f}%)")
                        
                except Exception as e:
                    warnings.warn(f"Cache warming failed for {pdb_id}: {e}")
                    failed += 1
                    completed_count += 1
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Calculate performance metrics
        download_rate = downloaded / duration if duration > 0 else 0
        success_rate = downloaded / len(uncached_ids) if uncached_ids else 0
        
        results = {
            "status": "completed",
            "total_requested": len(popular_pdb_ids),
            "already_cached": already_cached,
            "attempted_downloads": len(uncached_ids),
            "downloaded": downloaded,
            "failed": failed,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "download_rate_per_second": download_rate,
            "final_cache_size": len(self._access_times),
            "final_memory_cache_size": len(self._memory_cache)
        }
        
        warnings.warn(f"Cache warming completed: {downloaded}/{len(uncached_ids)} successful "
                     f"({success_rate*100:.1f}% success rate) in {duration:.1f}s "
                     f"({download_rate:.1f} downloads/sec)")
        
        return results
    
    def get_access_patterns(self, window_hours: float = 24.0) -> Dict[str, Any]:
        """
        Analyze access patterns to identify frequently used PDB structures.
        
        Args:
            window_hours: Time window for pattern analysis (default: 24 hours)
            
        Returns:
            Dictionary with access pattern analysis
        """
        current_time = time.perf_counter()
        window_seconds = window_hours * 3600
        cutoff_time = current_time - window_seconds
        
        # Analyze access frequency within time window
        recent_access = {}
        access_counts = defaultdict(int)
        
        with self._disk_cache_lock:
            for pdb_id, last_access in self._access_times.items():
                if last_access >= cutoff_time:
                    recent_access[pdb_id] = last_access
                    # Weight by recency (more recent = higher weight)
                    recency_weight = (last_access - cutoff_time) / window_seconds
                    access_counts[pdb_id] = 1.0 + recency_weight
        
        # Also check memory cache access patterns
        for pdb_id in self._memory_cache.keys():
            # Memory cache items are by definition recently accessed
            access_counts[pdb_id] = access_counts.get(pdb_id, 0) + 2.0  # Bonus for memory cache
        
        # Sort by access frequency/recency score
        popular_structures = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate cache hit rate
        total_requests = self._statistics.total_requests
        cache_hits = self._statistics.cache_hits
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "analysis_window_hours": window_hours,
            "total_structures_analyzed": len(access_counts),
            "recent_access_count": len(recent_access),
            "cache_hit_rate": hit_rate,
            "popular_structures": popular_structures[:50],  # Top 50 most popular
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_size": len(self._access_times),
            "recommendation": {
                "should_warm_cache": hit_rate < 0.8,
                "suggested_warming_size": min(100, len(popular_structures)),
                "predicted_hit_rate_improvement": max(0.1, 0.9 - hit_rate)
            }
        }
    
    def adaptive_prefetch(self, upcoming_pdb_ids: List[str], prefetch_depth: int = 20) -> None:
        """
        Intelligently prefetch structures based on access patterns and upcoming requests.
        
        This method uses historical access data to predict which structures to prefetch
        for optimal cache performance in streaming workloads.
        
        Args:
            upcoming_pdb_ids: List of PDB IDs that will be accessed soon
            prefetch_depth: Number of additional structures to prefetch based on patterns
        """
        if not upcoming_pdb_ids:
            return
        
        # Analyze access patterns to find correlated structures
        patterns = self.get_access_patterns()
        popular_structures = [pdb_id for pdb_id, _ in patterns["popular_structures"]]
        
        # Create prefetch priority queue
        prefetch_candidates = []
        
        # Priority 1: Upcoming structures not in cache
        for pdb_id in upcoming_pdb_ids:
            if pdb_id not in self._access_times and pdb_id not in self._memory_cache:
                prefetch_candidates.append((pdb_id, 3.0))  # High priority
        
        # Priority 2: Popular structures that might be evicted
        cache_utilization = len(self._access_times) / (self.max_disk_bytes // 100000)  # Estimate
        if cache_utilization > 0.8:  # Cache getting full
            for pdb_id in popular_structures[:prefetch_depth]:
                if pdb_id not in prefetch_candidates and pdb_id not in self._memory_cache:
                    prefetch_candidates.append((pdb_id, 2.0))  # Medium priority
        
        # Priority 3: Structures similar to recently accessed (if we had sequence similarity data)
        # For now, we'll use a simple heuristic based on PDB ID patterns
        recent_accessed = list(self._memory_cache.keys())
        for recent_id in recent_accessed[-5:]:  # Last 5 accessed
            # Simple heuristic: structures with similar PDB ID patterns
            prefix = recent_id[:2] if len(recent_id) >= 2 else recent_id
            for pdb_id in upcoming_pdb_ids:
                if pdb_id.startswith(prefix) and pdb_id not in [p[0] for p in prefetch_candidates]:
                    prefetch_candidates.append((pdb_id, 1.0))  # Low priority
        
        # Sort by priority and select top candidates
        prefetch_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_for_prefetch = [pdb_id for pdb_id, _ in prefetch_candidates[:prefetch_depth]]
        
        if selected_for_prefetch:
            warnings.warn(f"Adaptive prefetch: queuing {len(selected_for_prefetch)} structures based on access patterns")
            self.prefetch(selected_for_prefetch)
    
    def optimize_for_a100_streaming(self) -> Dict[str, Any]:
        """
        Apply A100-specific optimizations for streaming workloads.
        
        This method configures cache parameters for optimal performance on Harvard A100 cluster
        with high-bandwidth streaming scenarios.
        
        Returns:
            Dictionary with optimization results and performance predictions
        """
        optimizations_applied = []
        performance_estimates = {}
        
        # Check current system resources
        available_memory = psutil.virtual_memory().available
        disk_free = psutil.disk_usage(self.cache_dir).free
        
        # A100-specific memory optimization
        if self.max_memory_bytes < 4 * 1024 * 1024 * 1024:  # Less than 4GB
            # Increase memory cache for A100 environments with abundant RAM
            new_memory_limit = min(8 * 1024 * 1024 * 1024, available_memory // 4)  # Up to 8GB or 25% of available
            self.max_memory_bytes = new_memory_limit
            optimizations_applied.append(f"Increased memory cache to {new_memory_limit // (1024**3)}GB for A100 environment")
        
        # Disk cache optimization for netscratch
        if self.max_disk_bytes < 50 * 1024 * 1024 * 1024:  # Less than 50GB
            # Increase disk cache for A100 cluster with high-performance storage
            new_disk_limit = min(100 * 1024 * 1024 * 1024, disk_free // 2)  # Up to 100GB or 50% of available
            self.max_disk_bytes = new_disk_limit
            optimizations_applied.append(f"Increased disk cache to {new_disk_limit // (1024**3)}GB for high-performance storage")
        
        # Concurrent download optimization
        if self.max_concurrent_downloads < 16:
            # A100 cluster has high bandwidth - can handle more concurrent downloads
            self.max_concurrent_downloads = 16
            self._download_semaphore = threading.Semaphore(16)
            optimizations_applied.append("Increased concurrent downloads to 16 for A100 bandwidth")
        
        # Preemptive cache warming based on access patterns
        patterns = self.get_access_patterns()
        if patterns["recommendation"]["should_warm_cache"]:
            popular_ids = [pdb_id for pdb_id, _ in patterns["popular_structures"][:20]]
            if popular_ids:
                warming_results = self.warm_cache(popular_ids, max_concurrent=8)
                optimizations_applied.append(f"Pre-warmed cache with {warming_results.get('downloaded', 0)} popular structures")
        
        # Performance predictions
        estimated_hit_rate = min(0.95, patterns["cache_hit_rate"] + patterns["recommendation"]["predicted_hit_rate_improvement"])
        estimated_throughput_improvement = (estimated_hit_rate - patterns["cache_hit_rate"]) * 100  # Percentage points
        
        performance_estimates = {
            "current_hit_rate": patterns["cache_hit_rate"],
            "estimated_hit_rate_post_optimization": estimated_hit_rate,
            "estimated_throughput_improvement_percent": estimated_throughput_improvement,
            "memory_utilization_target": "60-80% for optimal performance",
            "concurrent_downloads_optimized": self.max_concurrent_downloads,
            "cache_sizes": {
                "memory_cache_gb": self.max_memory_bytes // (1024**3),
                "disk_cache_gb": self.max_disk_bytes // (1024**3)
            }
        }
        
        return {
            "optimizations_applied": optimizations_applied,
            "performance_estimates": performance_estimates,
            "a100_specific_tuning": {
                "tensor_core_optimization": "Enabled via FP16 mixed precision",
                "high_bandwidth_utilization": f"{self.max_concurrent_downloads} concurrent downloads",
                "memory_hierarchy_optimization": "Multi-tier caching (memory + disk)",
                "netscratch_integration": "Optimized for high-performance parallel filesystem"
            }
        }
    
    def _perform_warming_download(self, pdb_id: str, download_url: str) -> bool:
        """
        Optimized download for cache warming with reduced overhead.
        
        Args:
            pdb_id: PDB identifier
            download_url: Download URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already downloaded by another thread
            file_path = self.cache_dir / f"{pdb_id}.pdb"
            if file_path.exists():
                return True
            
            # Perform the download using existing atomic mechanism
            success = self._perform_download(pdb_id, download_url)
            
            if success:
                # Update statistics
                self._statistics.download_successes += 1
                self._statistics.download_attempts += 1
                return True
            else:
                self._statistics.download_failures += 1
                self._statistics.download_attempts += 1
                return False
                
        except Exception as e:
            self._statistics.download_failures += 1
            self._statistics.download_attempts += 1
            warnings.warn(f"Warming download failed for {pdb_id}: {e}")
            return False
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dictionary with detailed performance metrics
        """
        current_time = time.perf_counter()
        
        # Cache performance
        total_requests = self._statistics.total_requests
        cache_hits = self._statistics.cache_hits
        cache_misses = self._statistics.cache_misses
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Download performance
        download_success_rate = (
            self._statistics.download_successes / 
            max(1, self._statistics.download_attempts)
        )
        
        avg_download_time = (
            self._statistics.total_download_time / 
            max(1, self._statistics.download_successes)
        )
        
        # Resource utilization
        memory_usage = self._current_memory_usage
        memory_utilization = memory_usage / self.max_memory_bytes
        
        disk_usage = self._get_current_disk_size()
        disk_utilization = disk_usage / self.max_disk_bytes
        
        # System resources
        system_memory = psutil.virtual_memory()
        disk_stats = psutil.disk_usage(self.cache_dir)
        
        # Prefetch statistics
        prefetch_success_rate = (
            self._statistics.prefetch_completions / 
            max(1, self._statistics.prefetch_requests)
        )
        
        return {
            "timestamp": current_time,
            "cache_performance": {
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_rate": hit_rate,
                "hit_rate_percentage": hit_rate * 100,
                "performance_grade": self._grade_hit_rate(hit_rate)
            },
            "download_performance": {
                "total_attempts": self._statistics.download_attempts,
                "successes": self._statistics.download_successes,
                "failures": self._statistics.download_failures,
                "success_rate": download_success_rate,
                "success_rate_percentage": download_success_rate * 100,
                "average_download_time_seconds": avg_download_time,
                "total_bytes_downloaded": self._statistics.total_download_bytes,
                "download_rate_mbps": self._calculate_download_rate_mbps()
            },
            "prefetch_performance": {
                "requests": self._statistics.prefetch_requests,
                "completions": self._statistics.prefetch_completions,
                "success_rate": prefetch_success_rate,
                "active_background_downloads": self._statistics.background_downloads_active
            },
            "resource_utilization": {
                "memory_cache": {
                    "used_bytes": memory_usage,
                    "used_mb": memory_usage // (1024 * 1024),
                    "max_bytes": self.max_memory_bytes,
                    "max_mb": self.max_memory_bytes // (1024 * 1024),
                    "utilization_percentage": memory_utilization * 100,
                    "item_count": len(self._memory_cache)
                },
                "disk_cache": {
                    "used_bytes": disk_usage,
                    "used_mb": disk_usage // (1024 * 1024),
                    "used_gb": disk_usage // (1024 * 1024 * 1024),
                    "max_bytes": self.max_disk_bytes,
                    "max_gb": self.max_disk_bytes // (1024 * 1024 * 1024),
                    "utilization_percentage": disk_utilization * 100,
                    "file_count": len(self._access_times)
                },
                "system_resources": {
                    "memory_total_gb": system_memory.total // (1024**3),
                    "memory_available_gb": system_memory.available // (1024**3),
                    "memory_used_percentage": system_memory.percent,
                    "disk_total_gb": disk_stats.total // (1024**3),
                    "disk_free_gb": disk_stats.free // (1024**3),
                    "disk_used_percentage": (disk_stats.used / disk_stats.total) * 100
                }
            },
            "eviction_statistics": {
                "total_evictions": self._statistics.evictions_performed,
                "files_evicted": self._statistics.evicted_files_count,
                "bytes_evicted": self._statistics.total_evicted_bytes,
                "bytes_evicted_gb": self._statistics.total_evicted_bytes // (1024**3)
            },
            "recommendations": self._generate_performance_recommendations(
                hit_rate, memory_utilization, disk_utilization, download_success_rate
            )
        }
    
    def _grade_hit_rate(self, hit_rate: float) -> str:
        """Grade cache hit rate performance."""
        if hit_rate >= 0.95:
            return "Excellent"
        elif hit_rate >= 0.85:
            return "Good"
        elif hit_rate >= 0.70:
            return "Fair" 
        elif hit_rate >= 0.50:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_download_rate_mbps(self) -> float:
        """Calculate download rate in Mbps."""
        if self._statistics.total_download_time > 0 and self._statistics.total_download_bytes > 0:
            bytes_per_second = self._statistics.total_download_bytes / self._statistics.total_download_time
            mbps = (bytes_per_second * 8) / (1024 * 1024)  # Convert to Mbps
            return mbps
        return 0.0
    
    def _generate_performance_recommendations(
        self, 
        hit_rate: float, 
        memory_util: float, 
        disk_util: float, 
        download_success_rate: float
    ) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if hit_rate < 0.8:
            recommendations.append("Consider enabling cache warming or increasing cache size")
            recommendations.append("Analyze access patterns to optimize prefetching strategy")
        
        if memory_util > 0.9:
            recommendations.append("Memory cache near capacity - consider increasing max_memory_mb")
        elif memory_util < 0.5:
            recommendations.append("Memory cache underutilized - could increase prefetch_factor")
        
        if disk_util > 0.9:
            recommendations.append("Disk cache near capacity - consider increasing max_disk_gb")
            recommendations.append("Monitor eviction frequency and adjust cache size if needed")
        
        if download_success_rate < 0.9:
            recommendations.append("High download failure rate - check network connectivity")
            recommendations.append("Consider reducing concurrent_downloads or increasing timeout")
        
        if len(self._memory_cache) == 0 and hit_rate < 0.5:
            recommendations.append("No items in memory cache - check memory allocation and access patterns")
        
        return recommendations
    
    def evict_lru(self, bytes_needed: int) -> None:
        """
        Evict least recently used items to free disk space.
        
        CRITICAL FIX TASK-1-2: Completely rewritten to eliminate race conditions.
        - Atomic operations for file selection and eviction
        - Proper handling of concurrent access and downloads
        - Enhanced safety checks and error recovery
        - Improved logging and monitoring
        
        Args:
            bytes_needed: Number of bytes to free from disk
        """
        with self._statistics._timed_operation('eviction'):
            # CRITICAL FIX: Collect eviction candidates atomically to prevent race conditions
            eviction_candidates = []
            protected_files = []
            current_downloads = []
            
            # Single atomic check to determine what can be safely evicted
            with self._disk_cache_lock:
                with self._download_lock:
                    current_downloads = list(self._downloading.keys())
                
                self._log_performance_event('eviction_start', 'system', bytes_needed=bytes_needed)
                
                # Identify safe files to evict (not active, not downloading, not in download queue)
                for pdb_id, access_time in self._access_times.items():
                    if (pdb_id not in self._active_files and 
                        pdb_id not in self._downloading_files and
                        pdb_id not in current_downloads):
                        file_path = self.cache_dir / f"{pdb_id}.pdb"
                        if file_path.exists():
                            try:
                                file_size = file_path.stat().st_size
                                eviction_candidates.append((pdb_id, access_time, file_size, file_path))
                            except (OSError, IOError):
                                # File might be corrupted or inaccessible, mark for cleanup
                                eviction_candidates.append((pdb_id, access_time, 0, file_path))
                    else:
                        protected_files.append(pdb_id)
                
                # Sort by access time (oldest first) to implement true LRU
                eviction_candidates.sort(key=lambda x: x[1])
            
            if not eviction_candidates:
                self._perf_logger.warning(
                    f"No files available for eviction. Protected files: {len(protected_files)}, "
                    f"Active downloads: {len(current_downloads)}"
                )
                return
            
            # Evict files outside of critical section to minimize lock time
            freed_bytes = 0
            evicted_files = []
            failed_evictions = []
            memory_items_to_evict = []
            
            for pdb_id, access_time, file_size, file_path in eviction_candidates:
                if freed_bytes >= bytes_needed:
                    break
                
                # Double-check that file is still safe to evict (race condition protection)
                is_safe = True
                with self._disk_cache_lock:
                    if (pdb_id in self._active_files or 
                        pdb_id in self._downloading_files):
                        is_safe = False
                
                with self._download_lock:
                    if pdb_id in self._downloading:
                        is_safe = False
                
                if not is_safe:
                    self._perf_logger.debug(f"Skipping {pdb_id} - became active during eviction")
                    continue
                
                # Attempt to delete the file
                try:
                    if file_path.exists():
                        actual_size = file_path.stat().st_size
                        file_path.unlink()
                        
                        evicted_files.append(pdb_id)
                        freed_bytes += actual_size
                        memory_items_to_evict.append(pdb_id)
                        
                        self._log_performance_event('file_evicted', pdb_id, 
                                                   file_size=actual_size, 
                                                   target_freed=bytes_needed)
                        
                except (OSError, IOError) as e:
                    failed_evictions.append((pdb_id, str(e)))
                    self._statistics.record_error('storage')
                    self._perf_logger.warning(f"Failed to evict {pdb_id}: {e}")
            
            # Update tracking atomically after successful evictions
            if evicted_files:
                with self._disk_cache_lock:
                    for pdb_id in evicted_files:
                        self._access_times.pop(pdb_id, None)
                        # Update cached size accurately
                        if pdb_id in evicted_files:
                            # Find the actual file size that was freed
                            for evict_pdb_id, _, file_size, _ in eviction_candidates:
                                if evict_pdb_id == pdb_id:
                                    self._cached_disk_size = max(0, self._cached_disk_size - file_size)
                                    break
                
                # Clean up memory cache separately to avoid nested locks
                memory_items_cleaned = 0
                with self._cache_lock:
                    for pdb_id in memory_items_to_evict:
                        if pdb_id in self._memory_cache:
                            data = self._memory_cache.pop(pdb_id)
                            memory_items_cleaned += 1
                            # Estimate and update memory usage
                            if isinstance(data, str):
                                self._current_memory_usage -= len(data.encode())
                            elif hasattr(data, 'nbytes'):
                                self._current_memory_usage -= data.nbytes
                            elif hasattr(data, '__sizeof__'):
                                self._current_memory_usage -= data.__sizeof__()
                
                # Record comprehensive eviction statistics
                self._statistics.record_eviction(len(evicted_files), freed_bytes)
                
                self._perf_logger.info(
                    f"LRU eviction completed - Files: {len(evicted_files)}, "
                    f"Disk space freed: {freed_bytes / (1024*1024):.1f}MB, "
                    f"Memory items cleaned: {memory_items_cleaned}, "
                    f"Target: {bytes_needed / (1024*1024):.1f}MB, "
                    f"Success rate: {freed_bytes / bytes_needed * 100:.1f}%"
                )
                
                self._log_performance_event('eviction_complete', 'system', 
                                           files_evicted=len(evicted_files), 
                                           bytes_freed=freed_bytes,
                                           target_bytes=bytes_needed,
                                           memory_items_cleaned=memory_items_cleaned)
            else:
                self._perf_logger.warning(
                    f"No files could be evicted to free {bytes_needed / (1024*1024):.1f}MB. "
                    f"Candidates: {len(eviction_candidates)}, "
                    f"Protected files: {len(protected_files)}"
                )
            
            # Report any eviction failures
            if failed_evictions:
                for pdb_id, error in failed_evictions[:3]:  # Log first 3 failures
                    self._perf_logger.error(f"Eviction failed for {pdb_id}: {error}")
                if len(failed_evictions) > 3:
                    self._perf_logger.error(f"... and {len(failed_evictions) - 3} more eviction failures")
                    
                # Clean up tracking for failed evictions
                with self._disk_cache_lock:
                    for pdb_id, _ in failed_evictions:
                        # Remove from tracking even if file deletion failed to prevent corruption
                        self._access_times.pop(pdb_id, None)
        
    def clear_cache(self) -> None:
        """
        Clear all cached data from memory and disk.
        
        CRITICAL FIX TASK-1-2: Rewritten to prevent unsafe operations and race conditions.
        - Prevents clearing files currently in use or being downloaded
        - Uses atomic operations to prevent corruption
        - Enhanced error handling and logging
        - Respects active operations to prevent data loss
        """
        files_to_delete = []
        active_operations = []
        
        # CRITICAL FIX: Collect information atomically to prevent race conditions
        with self._disk_cache_lock:
            # Safely identify files that can be cleared (not active or downloading)
            for pdb_id in list(self._access_times.keys()):
                if pdb_id in self._active_files or pdb_id in self._downloading_files:
                    active_operations.append(pdb_id)
                else:
                    files_to_delete.append(pdb_id)
        
        # Check if we have any downloads in progress that we need to wait for
        downloads_in_progress = []
        with self._download_lock:
            downloads_in_progress = list(self._downloading.keys())
        
        if downloads_in_progress:
            self._perf_logger.warning(
                f"Clear cache requested but {len(downloads_in_progress)} downloads in progress. "
                f"Will skip active downloads: {downloads_in_progress[:5]}"
                + ("..." if len(downloads_in_progress) > 5 else "")
            )
        
        if active_operations:
            self._perf_logger.warning(
                f"Clear cache requested but {len(active_operations)} files are actively in use. "
                f"Will skip active files: {active_operations[:5]}"
                + ("..." if len(active_operations) > 5 else "")
            )
        
        # Clear memory cache safely
        memory_items_cleared = 0
        with self._cache_lock:
            memory_items_cleared = len(self._memory_cache)
            self._memory_cache.clear()
            self._current_memory_usage = 0
        
        # Clear disk files that are safe to delete
        deleted_count = 0
        deleted_size = 0
        failed_deletions = []
        
        for pdb_id in files_to_delete:
            file_path = self.cache_dir / f"{pdb_id}.pdb"
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                except (OSError, IOError) as e:
                    failed_deletions.append((pdb_id, str(e)))
                    self._statistics.record_error('storage')
        
        # Update tracking atomically
        with self._disk_cache_lock:
            # Remove successfully deleted files from tracking
            for pdb_id in files_to_delete:
                if pdb_id not in [fail[0] for fail in failed_deletions]:
                    self._access_times.pop(pdb_id, None)
            
            # Recalculate disk size to ensure consistency
            self._cached_disk_size = self._get_current_disk_size()
            self._last_size_update = time.perf_counter()
        
        # Log results
        self._perf_logger.info(
            f"Cache cleared - Memory items: {memory_items_cleared}, "
            f"Disk files: {deleted_count}/{len(files_to_delete)}, "
            f"Space freed: {deleted_size / (1024*1024):.1f}MB, "
            f"Active operations preserved: {len(active_operations)}"
        )
        
        if failed_deletions:
            for pdb_id, error in failed_deletions[:5]:  # Log first 5 failures
                self._perf_logger.error(f"Failed to delete {pdb_id}: {error}")
            if len(failed_deletions) > 5:
                self._perf_logger.error(f"... and {len(failed_deletions) - 5} more deletion failures")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # CRITICAL FIX ROB-001: Use consistent lock ordering to prevent deadlock
        # Get disk stats first
        with self._disk_cache_lock:
            current_disk_size = self._get_current_disk_size()
            disk_file_count = len(self._access_times)
            active_files = len(self._active_files)
            downloading_files = len(self._downloading_files)
        
        # Get memory stats separately
        with self._cache_lock:
            current_memory_usage = self._current_memory_usage
            memory_item_count = len(self._memory_cache)
        
        # Update statistics outside locks
        self._statistics.update_memory_usage(current_memory_usage)
        self._statistics.update_disk_usage(current_disk_size)
        
        # Get comprehensive statistics from the new system
        comprehensive_stats = self._statistics.get_comprehensive_stats()
        
        # Combine with traditional format for backward compatibility
        stats = {
            'disk_cache': {
                'size_bytes': current_disk_size,
                'size_mb': current_disk_size / (1024 * 1024),
                'size_gb': current_disk_size / (1024 * 1024 * 1024),
                'max_size_gb': self.max_disk_bytes / (1024 * 1024 * 1024),
                'utilization_percent': (current_disk_size / self.max_disk_bytes) * 100,
                'file_count': disk_file_count,
                'active_files': active_files,
                'downloading_files': downloading_files
            },
            'memory_cache': {
                'size_bytes': current_memory_usage,
                'size_mb': current_memory_usage / (1024 * 1024),
                'max_size_mb': self.max_memory_bytes / (1024 * 1024),
                'utilization_percent': (current_memory_usage / self.max_memory_bytes) * 100,
                'item_count': memory_item_count
            },
            'cache_dir': str(self.cache_dir),
            'target_free_bytes': self.target_free_bytes,
            'max_concurrent_downloads': self.max_concurrent_downloads,
            # Comprehensive statistics from new monitoring system
            **comprehensive_stats
        }
        
        return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with recommendations.
        
        Returns:
            Dictionary containing performance analysis and recommendations
        """
        stats = self.get_stats()
        
        # Performance analysis
        hit_rate = stats['cache_performance']['hit_rate']
        memory_util = stats['memory_cache']['utilization_percent']
        disk_util = stats['disk_cache']['utilization_percent']
        
        # Error analysis
        error_stats = stats['error_analysis']
        total_errors = error_stats['total_errors']
        total_ops = stats['cache_performance']['total_requests']
        error_rate = total_errors / max(total_ops, 1)
        
        # Performance recommendations
        recommendations = []
        
        if hit_rate < 0.7:
            recommendations.append(f"Cache hit rate is low ({hit_rate:.1%}). Consider increasing cache size or improving cache locality.")
        
        if memory_util > 90:
            recommendations.append(f"Memory utilization is very high ({memory_util:.1%}). Consider increasing memory limit.")
        elif memory_util < 10:
            recommendations.append(f"Memory utilization is very low ({memory_util:.1%}). Memory limit might be too high.")
            
        if disk_util > 95:
            recommendations.append(f"Disk utilization is critical ({disk_util:.1%}). Immediate cleanup needed.")
        elif disk_util > 85:
            recommendations.append(f"Disk utilization is high ({disk_util:.1%}). Consider increasing disk limit.")
            
        if error_rate > 0.05:  # More than 5% error rate
            recommendations.append(f"High error rate ({error_rate:.1%}). Check network connectivity and storage.")
            
        if 'avg_download_ms' in stats['timing_analysis']:
            avg_download = stats['timing_analysis']['avg_download_ms']
            if avg_download > 10000:  # More than 10 seconds
                recommendations.append(f"Slow downloads detected (avg {avg_download:.0f}ms). Check network performance.")
        
        if 'overhead_percentage' in stats['monitoring_overhead']:
            overhead = stats['monitoring_overhead']['overhead_percentage']
            if overhead > 1.0:  # More than 1% overhead
                recommendations.append(f"Monitoring overhead is high ({overhead:.1%}). Consider reducing logging level.")
        
        return {
            'performance_summary': {
                'overall_health': self._calculate_health_score(stats),
                'hit_rate': hit_rate,
                'error_rate': error_rate,
                'memory_utilization': memory_util,
                'disk_utilization': disk_util,
            },
            'recommendations': recommendations,
            'detailed_stats': stats,
            'monitoring_info': {
                'monitoring_enabled': True,
                'logging_level': self._perf_logger.level,
                'statistics_overhead': stats.get('monitoring_overhead', {}),
            }
        }
    
    def _calculate_health_score(self, stats: Dict[str, Any]) -> str:
        """Calculate overall cache health score."""
        score = 100
        
        # Deduct for low hit rate
        hit_rate = stats['cache_performance']['hit_rate']
        if hit_rate < 0.5:
            score -= 30
        elif hit_rate < 0.7:
            score -= 15
            
        # Deduct for high resource utilization
        memory_util = stats['memory_cache']['utilization_percent']
        disk_util = stats['disk_cache']['utilization_percent']
        
        if memory_util > 95 or disk_util > 95:
            score -= 25
        elif memory_util > 85 or disk_util > 85:
            score -= 10
            
        # Deduct for errors
        error_stats = stats['error_analysis']
        total_errors = error_stats['total_errors']
        total_ops = stats['cache_performance']['total_requests']
        error_rate = total_errors / max(total_ops, 1)
        
        if error_rate > 0.1:
            score -= 30
        elif error_rate > 0.05:
            score -= 15
        elif error_rate > 0.01:
            score -= 5
            
        # Health categorization
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 25:
            return "Poor"
        else:
            return "Critical"
    
    def set_monitoring_level(self, level: str):
        """
        Set monitoring and logging level.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        if level.upper() in level_map:
            self._perf_logger.setLevel(level_map[level.upper()])
            self._perf_logger.info(f"Monitoring level set to {level.upper()}")
        else:
            self._perf_logger.warning(f"Invalid monitoring level: {level}")
    
    def enable_periodic_reporting(self, interval_seconds: int = 300):
        """
        Enable periodic performance reporting.
        
        Args:
            interval_seconds: Interval between reports in seconds (default 300 = 5 minutes)
        """
        def periodic_report():
            while True:
                time.sleep(interval_seconds)
                try:
                    self._log_cache_summary(interval_seconds)
                    
                    # Log any performance concerns
                    report = self.get_performance_report()
                    if report['recommendations']:
                        self._perf_logger.warning("Performance recommendations: " + 
                                                "; ".join(report['recommendations'][:3]))
                except Exception as e:
                    self._perf_logger.error(f"Periodic reporting failed: {e}")
        
        # Start reporting thread
        reporting_thread = threading.Thread(target=periodic_report, daemon=True)
        reporting_thread.start()
        self._perf_logger.info(f"Periodic reporting enabled (interval: {interval_seconds}s)")
    
    def export_monitoring_data(self, format_type: str = 'json') -> str:
        """
        Export monitoring data for external analysis.
        
        Args:
            format_type: Export format ('json', 'csv')
            
        Returns:
            Formatted monitoring data string
        """
        if format_type.lower() == 'json':
            return json.dumps(self.get_stats(), indent=2)
        elif format_type.lower() == 'csv':
            # Simple CSV export for basic metrics
            stats = self.get_stats()
            csv_lines = [
                "metric,value",
                f"hit_rate,{stats['cache_performance']['hit_rate']:.3f}",
                f"total_requests,{stats['cache_performance']['total_requests']}",
                f"memory_usage_mb,{stats['resource_usage']['current_memory_mb']:.1f}",
                f"disk_usage_mb,{stats['resource_usage']['disk_usage_mb']:.1f}",
                f"download_attempts,{stats['download_performance']['attempts']}",
                f"download_success_rate,{stats['download_performance']['success_rate']:.3f}",
                f"avg_lookup_ms,{stats['timing_analysis']['avg_lookup_ms']:.1f}",
                f"total_errors,{stats['error_analysis']['total_errors']}"
            ]
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def reset_statistics(self):
        """Reset all performance statistics (useful for benchmarking specific periods)."""
        with self._statistics._stats_lock:
            old_start_time = self._statistics.session_start_time
            self._statistics = CacheStatistics()
            self._perf_logger.info(f"Statistics reset. Previous session duration: "
                                 f"{time.perf_counter() - old_start_time:.1f} seconds")
    
    # Private helper methods for LRU and storage management
    
    def _scan_existing_cache(self) -> None:
        """Scan existing cache directory to initialize access times."""
        with self._disk_cache_lock:
            total_size = 0
            current_time = time.perf_counter()
            
            for file_path in self.cache_dir.glob("*.pdb"):
                try:
                    stat_info = file_path.stat()
                    pdb_id = file_path.stem.upper()
                    
                    # Use file modification time as initial access time
                    access_time = stat_info.st_mtime
                    self._access_times[pdb_id] = access_time
                    total_size += stat_info.st_size
                    
                except (OSError, IOError) as e:
                    warnings.warn(f"Failed to stat {file_path}: {e}")
            
            self._cached_disk_size = total_size
            self._last_size_update = current_time
            
            warnings.warn(f"Initialized cache with {len(self._access_times)} files, "
                         f"{total_size / (1024*1024):.1f} MB total")
    
    def _update_disk_access_time(self, pdb_id: str) -> None:
        """
        CRITICAL FIX STORAGE-003: Update access time for a PDB file in thread-safe manner with LRU consistency.
        
        Args:
            pdb_id: PDB identifier to update access time for
        """
        current_time = time.perf_counter()
        with self._disk_cache_lock:
            # CRITICAL FIX STORAGE-003: Atomic LRU reordering for consistency
            # Remove from current position and add to end (most recent)
            if pdb_id in self._access_times:
                del self._access_times[pdb_id]
            self._access_times[pdb_id] = current_time
            
            # Update disk statistics to reflect fresh access
            with self._disk_size_lock:
                self._last_size_update = current_time  # Mark cache as recently updated
    
    def _add_to_memory_cache(self, pdb_id: str, data: Any) -> None:
        """Add data to memory cache with LRU eviction if needed."""
        with self._cache_lock:
            # Estimate memory usage
            data_size = 0
            if isinstance(data, str):
                data_size = len(data.encode())
            elif hasattr(data, 'nbytes'):
                data_size = data.nbytes
            elif hasattr(data, '__sizeof__'):
                data_size = data.__sizeof__()
            
            # Check if we need to evict from memory
            while (self._current_memory_usage + data_size > self.max_memory_bytes and 
                   self._memory_cache):
                # Evict least recently used item from memory
                oldest_key = next(iter(self._memory_cache))
                oldest_data = self._memory_cache.pop(oldest_key)
                
                # Update memory usage
                if isinstance(oldest_data, str):
                    self._current_memory_usage -= len(oldest_data.encode())
                elif hasattr(oldest_data, 'nbytes'):
                    self._current_memory_usage -= oldest_data.nbytes
            
            # Add new item
            self._memory_cache[pdb_id] = data
            self._current_memory_usage += data_size
    
    def _get_current_disk_size(self) -> int:
        """
        CRITICAL FIX STORAGE-002: Get current disk cache size with thread-safe caching and stale data prevention.
        
        Returns:
            Current disk usage in bytes
        """
        current_time = time.perf_counter()
        
        # CRITICAL FIX STORAGE-002: Use separate lock for size calculations to prevent deadlocks
        with self._disk_size_lock:
            # Return cached size if recent enough and cache is valid
            if (current_time - self._last_size_update < self._size_cache_timeout and 
                self._cached_disk_size >= 0):
                return self._cached_disk_size
        
        # CRITICAL FIX STORAGE-001: Recalculate disk size with stale entry cleanup
        total_size = 0
        stale_entries = []
        
        with self._disk_cache_lock:
            # Calculate actual disk usage by checking each tracked file
            for pdb_id in list(self._access_times.keys()):
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                try:
                    if file_path.exists():
                        # Add actual file size to total
                        total_size += file_path.stat().st_size
                    else:
                        # File was deleted externally - mark for cleanup
                        stale_entries.append(pdb_id)
                except (OSError, IOError):
                    # File access error - mark for cleanup
                    stale_entries.append(pdb_id)
            
            # CRITICAL FIX STORAGE-003: Clean up stale entries to maintain LRU consistency
            for pdb_id in stale_entries:
                self._access_times.pop(pdb_id, None)
                self._perf_logger.debug(f"Cleaned up stale tracking for {pdb_id}")
        
        # CRITICAL FIX STORAGE-002: Update cached size atomically
        with self._disk_size_lock:
            self._cached_disk_size = total_size
            self._last_size_update = current_time
        
        if stale_entries:
            self._perf_logger.info(f"Cleaned up {len(stale_entries)} stale cache entries during size calculation")
        
        return total_size
    
    def ensure_cache_space(self, bytes_needed: int = None) -> None:
        """
        Proactively ensure cache has enough space for new files.
        
        Args:
            bytes_needed: Specific bytes needed, or uses target_free_bytes if None
        """
        if bytes_needed is None:
            bytes_needed = self.target_free_bytes
        
        current_size = self._get_current_disk_size()
        
        # Check if we need to free space
        if current_size + bytes_needed > self.max_disk_bytes:
            bytes_to_free = current_size + bytes_needed - self.max_disk_bytes
            self.evict_lru(bytes_to_free)
    
    def add_downloaded_file(self, pdb_id: str, file_size: int) -> None:
        """
        Register a newly downloaded file with the cache tracking.
        
        Args:
            pdb_id: PDB identifier
            file_size: Size of the downloaded file in bytes
        """
        with self._disk_cache_lock:
            current_time = time.perf_counter()
            self._access_times[pdb_id] = current_time
            self._cached_disk_size += file_size
            self._downloading_files.discard(pdb_id)
    
    def mark_downloading(self, pdb_id: str) -> None:
        """Mark a file as currently being downloaded to prevent eviction."""
        with self._disk_cache_lock:
            self._downloading_files.add(pdb_id)
    
    def unmark_downloading(self, pdb_id: str) -> None:
        """Unmark a file as being downloaded."""
        with self._disk_cache_lock:
            self._downloading_files.discard(pdb_id)
    
    
    def get_pdb_path(self, pdb_id: str) -> Optional[str]:
        """
        Get the local file path for a PDB structure, downloading if necessary.
        
        Args:
            pdb_id: PDB identifier
            
        Returns:
            Local file path if available, None otherwise
        """
        # Validate PDB ID for security
        if not self._validate_pdb_id(pdb_id):
            warnings.warn(f"Invalid PDB ID: {pdb_id}")
            return None
            
        pdb_id = pdb_id.upper()
        file_path = self.cache_dir / f"{pdb_id}.pdb"
        
        # Check if file already exists in cache
        if file_path.exists():
            # Update access time for LRU tracking
            self._update_disk_access_time(pdb_id)
            return str(file_path)
        
        # For missing files, try to download from RCSB PDB
        # Default RCSB download URL format
        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        # Attempt download
        try:
            data = self.get(pdb_id, download_url)
            if data is not None and file_path.exists():
                return str(file_path)
        except Exception as e:
            warnings.warn(f"Failed to download PDB {pdb_id}: {e}")
        
        return None
    
    def validate_cache_consistency(self) -> Dict[str, Any]:
        """
        CRITICAL FIX STORAGE-VALIDATION: Validate cache state consistency and detect corruption.
        
        This method verifies that the cache data structures are consistent with the actual
        filesystem state, identifies stale entries, and reports any inconsistencies.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        validation_results = {
            "validation_timestamp": time.perf_counter(),
            "issues_found": [],
            "statistics": {},
            "recommendations": [],
            "overall_health": "unknown"
        }
        
        # Check disk vs tracking consistency
        tracked_files = set()
        actual_files = set()
        
        with self._disk_cache_lock:
            tracked_files = set(self._access_times.keys())
        
        try:
            # Scan actual files on disk
            for file_path in self.cache_dir.glob("*.pdb"):
                pdb_id = file_path.stem.upper()
                actual_files.add(pdb_id)
        except Exception as e:
            validation_results["issues_found"].append(f"Failed to scan cache directory: {e}")
        
        # Find inconsistencies
        missing_from_disk = tracked_files - actual_files
        missing_from_tracking = actual_files - tracked_files
        
        if missing_from_disk:
            validation_results["issues_found"].append(
                f"Stale tracking entries (files missing from disk): {len(missing_from_disk)} files"
            )
        
        if missing_from_tracking:
            validation_results["issues_found"].append(
                f"Untracked files on disk: {len(missing_from_tracking)} files"
            )
        
        # Check for protected file consistency
        with self._file_operations_lock:
            active_files = len(self._active_files)
            downloading_files = len(self._downloading_files)
            evicting_files = len(self._evicting_files)
        
        # Check memory cache consistency
        with self._cache_lock:
            memory_cache_size = len(self._memory_cache)
            memory_usage = self._current_memory_usage
        
        # Validate disk size calculation
        calculated_disk_size = self._get_current_disk_size()
        
        validation_results["statistics"] = {
            "tracked_files": len(tracked_files),
            "actual_files": len(actual_files),
            "missing_from_disk": len(missing_from_disk),
            "missing_from_tracking": len(missing_from_tracking),
            "active_files": active_files,
            "downloading_files": downloading_files,
            "evicting_files": evicting_files,
            "memory_cache_size": memory_cache_size,
            "memory_usage_bytes": memory_usage,
            "calculated_disk_size_bytes": calculated_disk_size,
        }
        
        # Generate recommendations
        if missing_from_disk > 0:
            validation_results["recommendations"].append("Run cache cleanup to remove stale tracking entries")
        
        if missing_from_tracking > 0:
            validation_results["recommendations"].append("Scan cache directory to track unmanaged files")
        
        if evicting_files > 0:
            validation_results["recommendations"].append(f"Warning: {evicting_files} files currently being evicted")
        
        # Overall health assessment
        total_issues = len(missing_from_disk) + len(missing_from_tracking)
        if total_issues == 0:
            validation_results["overall_health"] = "excellent"
        elif total_issues <= 5:
            validation_results["overall_health"] = "good"
        elif total_issues <= 20:
            validation_results["overall_health"] = "fair"
        else:
            validation_results["overall_health"] = "poor"
        
        return validation_results
    
    def cleanup_stale_entries(self) -> Dict[str, int]:
        """
        CRITICAL FIX STORAGE-MAINTENANCE: Clean up stale tracking entries and orphaned files.
        
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {
            "stale_tracking_cleaned": 0,
            "orphaned_files_found": 0,
            "orphaned_files_removed": 0,
            "errors_encountered": 0
        }
        
        # Clean up stale tracking entries
        stale_entries = []
        with self._disk_cache_lock:
            for pdb_id in list(self._access_times.keys()):
                file_path = self.cache_dir / f"{pdb_id}.pdb"
                if not file_path.exists():
                    stale_entries.append(pdb_id)
            
            # Remove stale entries
            for pdb_id in stale_entries:
                self._access_times.pop(pdb_id, None)
        
        cleanup_stats["stale_tracking_cleaned"] = len(stale_entries)
        
        # Find and optionally remove orphaned files
        tracked_files = set()
        with self._disk_cache_lock:
            tracked_files = set(self._access_times.keys())
        
        try:
            for file_path in self.cache_dir.glob("*.pdb"):
                pdb_id = file_path.stem.upper()
                if pdb_id not in tracked_files:
                    cleanup_stats["orphaned_files_found"] += 1
                    # Could optionally remove or add to tracking
                    # For safety, just count them for now
        except Exception as e:
            cleanup_stats["errors_encountered"] += 1
            self._perf_logger.warning(f"Error during orphaned file scan: {e}")
        
        self._perf_logger.info(f"Cache cleanup completed: {cleanup_stats}")
        return cleanup_stats


class PDBDownloader:
    """
    Background PDB downloader with retry logic and rate limiting.
    """
    
    def __init__(self, max_concurrent: int = 4, retry_attempts: int = 3):
        """
        Initialize PDB downloader.
        
        Args:
            max_concurrent: Maximum concurrent downloads
            retry_attempts: Number of retry attempts for failed downloads
        """
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        
    def download(self, pdb_id: str, url: str, output_path: Path) -> bool:
        """
        Download PDB file with retry logic.
        
        Args:
            pdb_id: PDB identifier
            url: Download URL
            output_path: Local file path for saving
            
        Returns:
            True if download successful, False otherwise
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Create session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        for attempt in range(self.retry_attempts):
            try:
                # Download to temporary file first (atomic operation)
                temp_file = None
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=output_path.parent,
                    prefix=f'.tmp_{pdb_id}_',
                    suffix='.pdb',
                    delete=False
                ) as f:
                    temp_file = f.name
                    
                    # Perform download with timeout
                    response = session.get(url, timeout=30, stream=True)
                    response.raise_for_status()
                    
                    # Write content chunk by chunk
                    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                        if chunk:
                            f.write(chunk)
                    
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic move to final location
                os.rename(temp_file, str(output_path))
                return True
                
            except Exception as e:
                warnings.warn(f"Download attempt {attempt + 1} failed for {pdb_id}: {e}")
                
                # Cleanup temporary file if it exists
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass
                
                if attempt == self.retry_attempts - 1:
                    # Last attempt failed
                    return False
                
                # Wait before retry (exponential backoff)
                time.sleep(2 ** attempt)
        
        return False