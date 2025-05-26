# --- utils/jax_utils.py ---
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any

# Configure JAX for float64 (should match other JAX files)
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

def convert_to_hashable(obj: Any) -> Any:
    """
    Recursively convert objects to hashable forms for JAX JIT.
    FIXED: Keep dictionaries as tuples of (key, value) pairs that can be reconstructed.
    """
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        # Convert to sorted tuple of (key, value) pairs - this stays as a tuple, not frozenset
        return tuple(sorted((str(k), convert_to_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, onp.ndarray):
        if obj.ndim == 0:
            return obj.item()
        else:
            return tuple(obj.tolist())
    elif isinstance(obj, jnp.ndarray):
        if obj.ndim == 0:
            return obj.item()
        else:
            return tuple(obj.tolist())
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        return obj.item()
    else:
        return obj