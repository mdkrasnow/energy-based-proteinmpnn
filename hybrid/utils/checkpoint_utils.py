"""
PyTorch 2.6 Compatible Checkpoint Loading Utilities

This module provides thread-safe checkpoint loading for PyTorch 2.6+ which changed
the default `weights_only` parameter from False to True for security. Our checkpoints
contain numpy random state objects (from np.random.get_state()) which include 
numpy.dtype objects that are now blocked by default.

This module registers safe numpy types automatically on import and provides optional
convenience wrappers for enhanced error messages.

Usage Pattern 1 (automatic - recommended):
    from hybrid.utils import checkpoint_utils  # Safe globals registered automatically
    checkpoint = torch.load(path)  # Now works with weights_only=True

Usage Pattern 2 (explicit wrappers):  
    from hybrid.utils.checkpoint_utils import load_checkpoint
    checkpoint = load_checkpoint(path)  # Better error messages

Thread Safety:
    Safe for multi-GPU and distributed training scenarios. Uses double-check locking
    pattern to prevent race conditions during concurrent imports.

Security Note:
    The registered numpy types are genuinely safe:
    - numpy.dtype: Type descriptors, no executable code
    - numpy.ndarray: Data containers, no executable code  
    - numpy.core.multiarray._reconstruct: Array reconstruction function, internal PyTorch use
"""

import threading
import torch
import numpy as np

# Global state for thread-safe initialization
_safe_globals_initialized = False
_init_lock = threading.Lock()


def _ensure_safe_globals():
    """Register numpy types as safe for PyTorch checkpoint loading.
    
    Thread-safe initialization for multi-GPU/distributed training scenarios.
    Uses double-check locking pattern to prevent race conditions.
    
    Registers these numpy types as safe:
    - numpy.ndarray: NumPy arrays - safe data containers
    - numpy.core.multiarray._reconstruct: Array reconstruction - safe internal function
    - numpy.dtype: Generic dtype type - safe type descriptors
    - Specific dtype classes: Int32DType, Int64DType, Float32DType, etc. - safe type instances
    
    These are needed because our checkpoints contain:
    1. NumPy arrays with various dtypes (from model parameters)
    2. Random states from np.random.get_state() (contains dtype objects)
    3. Array reconstruction functions (internal PyTorch serialization)
    """
    global _safe_globals_initialized
    if not _safe_globals_initialized:
        with _init_lock:
            if not _safe_globals_initialized:  # Double-check inside lock
                # Core numpy types
                safe_types = [
                    np.ndarray,                         # NumPy arrays - safe data containers
                    np.core.multiarray._reconstruct,   # Array reconstruction - safe internal function
                    np.core.multiarray.scalar,         # NumPy scalar values - safe data containers
                    np.dtype,                           # Generic dtype class - safe type descriptors
                ]
                
                # Add specific dtype classes that appear in checkpoints
                # These are needed for numpy arrays with specific dtypes and random states
                dtype_classes = [
                    np.dtypes.Int8DType, np.dtypes.Int16DType, np.dtypes.Int32DType, np.dtypes.Int64DType,
                    np.dtypes.UInt8DType, np.dtypes.UInt16DType, np.dtypes.UInt32DType, np.dtypes.UInt64DType,
                    np.dtypes.Float32DType, np.dtypes.Float64DType,
                    np.dtypes.BoolDType, np.dtypes.Complex64DType, np.dtypes.Complex128DType
                ]
                safe_types.extend(dtype_classes)
                
                torch.serialization.add_safe_globals(safe_types)
                _safe_globals_initialized = True


def load_checkpoint(f, map_location=None, **kwargs):
    """Load PyTorch checkpoint with safe numpy globals pre-registered.
    
    This is a thin wrapper over torch.load() that ensures safe_globals are registered
    and provides better error messages. You can also just import this module and use
    torch.load directly - the safe_globals registration happens automatically.
    
    Args:
        f: File path or file object (same as torch.load)
        map_location: Device mapping (same as torch.load) 
        **kwargs: Additional arguments passed to torch.load
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        RuntimeError: If checkpoint loading fails with enhanced error context
        
    Example:
        >>> from hybrid.utils.checkpoint_utils import load_checkpoint
        >>> checkpoint = load_checkpoint('model.pt', map_location='cpu')
        >>> # OR just use automatic registration:
        >>> from hybrid.utils import checkpoint_utils
        >>> checkpoint = torch.load('model.pt')  # Works with weights_only=True
    """
    try:
        return torch.load(f, map_location=map_location, **kwargs)
    except Exception as e:
        error_msg = (
            f"Failed to load checkpoint from {f}. "
            f"Error: {e}. "
            f"Numpy safe globals are registered. "
            f"Check: (1) File exists and is not corrupted, "
            f"(2) PyTorch version compatibility (current: {torch.__version__}), "
            f"(3) Checkpoint was saved with compatible PyTorch version."
        )
        raise RuntimeError(error_msg) from e


def save_checkpoint(checkpoint, f, **kwargs):
    """Save PyTorch checkpoint with optional metadata logging.
    
    This is a thin wrapper over torch.save() for symmetry with load_checkpoint().
    Provides consistent error handling and optional debugging information.
    
    Args:
        checkpoint: Dictionary to save
        f: File path or file object (same as torch.save)
        **kwargs: Additional arguments passed to torch.save
        
    Example:
        >>> from hybrid.utils.checkpoint_utils import save_checkpoint
        >>> save_checkpoint({'model_state_dict': model.state_dict()}, 'model.pt')
    """
    try:
        torch.save(checkpoint, f, **kwargs)
    except Exception as e:
        error_msg = (
            f"Failed to save checkpoint to {f}. "
            f"Error: {e}. "
            f"Check: (1) Directory exists and is writable, "
            f"(2) Sufficient disk space, "
            f"(3) Valid checkpoint dictionary structure."
        )
        raise RuntimeError(error_msg) from e


# Register safe globals immediately when module is imported
# This ensures automatic registration for all torch.load() calls after import
_ensure_safe_globals()