# --- utils/jax_utils.py ---
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Any

# Configure JAX for float64 (should match other JAX files)
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

def convert_to_hashable(obj: Any) -> Any:
    """Recursively convert lists and dictionaries to tuples/frozensets to make objects hashable for JAX JIT."""
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        # Convert dictionary to a frozenset of (key, value) pairs
        # Ensure keys are hashable (often strings, but handle others)
        return frozenset(sorted(((str(k), convert_to_hashable(v)) if not isinstance(k, str) else (k, convert_to_hashable(v))) for k, v in obj.items()))
    elif isinstance(obj, onp.ndarray):
        # Convert numpy arrays to tuples (if not scalars) or scalars
        if obj.ndim == 0:
            return obj.item()
        else:
            return tuple(obj.tolist())
    elif isinstance(obj, jnp.ndarray):
        # Convert JAX arrays to tuples (if not scalars) or scalars
        if obj.ndim == 0:
            return obj.item()
        else:
            return tuple(obj.tolist())
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')): # Handle JAX/numpy scalar
         return obj.item()
    else:
        return obj