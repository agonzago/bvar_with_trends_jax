# parameter_extraction_for_smoother.py
# Extract the RIGHT parameters for simulation smoother

import jax
import jax.numpy as jnp
from typing import Dict, List

_DEFAULT_DTYPE = jnp.float64

def extract_smoother_parameters(posterior_samples: Dict, 
                               config_data: Dict,
                               variable_names: List[str]) -> Dict[str, jax.Array]:
    """
    Extract the CORRECT parameters for simulation smoother.
    
    The smoother needs:
    1. phi_list - VAR coefficients (already computed in MCMC)
    2. Sigma_cycles - Stationary shock covariance (already computed in MCMC) 
    3. Sigma_trends - Trend shock covariance (already computed in MCMC)
    
    NOT the raw A parameters or individual variances!
    """
    smoother_params = {}
    
    print(f"Available MCMC parameters: {list(posterior_samples.keys())}")
    print(f"Extracting parameters for smoother...")
    
    # 1. Extract phi_list (VAR coefficients) - REQUIRED
    if 'phi_list' in posterior_samples:
        phi_list_raw = posterior_samples['phi_list']
        print(f"phi_list type: {type(phi_list_raw)}")
        
        if isinstance(phi_list_raw, list):
            # phi_list is stored as a list of matrices across MCMC samples
            # Convert to JAX array and take mean
            print(f"phi_list is a list with {len(phi_list_raw)} samples")
            print(f"First sample type: {type(phi_list_raw[0])}, shape: {jnp.array(phi_list_raw[0]).shape if hasattr(phi_list_raw[0], 'shape') or isinstance(phi_list_raw[0], (list, tuple)) else 'scalar'}")
            
            # Convert list to JAX array and take mean
            phi_list_array = jnp.array(phi_list_raw)  # Shape: (n_samples, var_order, k_stationary, k_stationary)
            smoother_params['phi_list'] = jnp.mean(phi_list_array, axis=0)
        else:
            # phi_list is already a JAX array
            smoother_params['phi_list'] = jnp.mean(phi_list_raw, axis=0)
        
        print(f"✓ Added phi_list with shape: {smoother_params['phi_list'].shape}")
    else:
        raise ValueError("CRITICAL: phi_list not found in MCMC samples. This should be computed as deterministic.")
    
    # 2. Extract Sigma_cycles (stationary shock covariance) - REQUIRED  
    if 'Sigma_cycles' in posterior_samples:
        Sigma_cycles_raw = posterior_samples['Sigma_cycles']
        print(f"Sigma_cycles type: {type(Sigma_cycles_raw)}")
        
        if isinstance(Sigma_cycles_raw, list):
            # Convert list to JAX array and take mean
            print(f"Sigma_cycles is a list with {len(Sigma_cycles_raw)} samples")
            Sigma_cycles_array = jnp.array(Sigma_cycles_raw)
            smoother_params['Sigma_cycles'] = jnp.mean(Sigma_cycles_array, axis=0)
        else:
            smoother_params['Sigma_cycles'] = jnp.mean(Sigma_cycles_raw, axis=0)
        
        print(f"✓ Added Sigma_cycles with shape: {smoother_params['Sigma_cycles'].shape}")
    else:
        raise ValueError("CRITICAL: Sigma_cycles not found in MCMC samples. This should be computed as deterministic.")
    
    # 3. Extract Sigma_trends (trend shock covariance) - REQUIRED
    if 'Sigma_trends' in posterior_samples:
        Sigma_trends_raw = posterior_samples['Sigma_trends']
        print(f"Sigma_trends type: {type(Sigma_trends_raw)}")
        
        if isinstance(Sigma_trends_raw, list):
            # Convert list to JAX array and take mean
            print(f"Sigma_trends is a list with {len(Sigma_trends_raw)} samples")
            Sigma_trends_array = jnp.array(Sigma_trends_raw)
            smoother_params['Sigma_trends'] = jnp.mean(Sigma_trends_array, axis=0)
        else:
            smoother_params['Sigma_trends'] = jnp.mean(Sigma_trends_raw, axis=0)
        
        print(f"✓ Added Sigma_trends with shape: {smoother_params['Sigma_trends'].shape}")
    else:
        raise ValueError("CRITICAL: Sigma_trends not found in MCMC samples. This should be computed as deterministic.")
    
    # 4. Check for any very small eigenvalues that could cause numerical issues
    print("\nChecking covariance matrices for numerical stability...")
    
    # Check Sigma_cycles
    if smoother_params['Sigma_cycles'].shape[0] > 0:
        eigs_cycles = jnp.linalg.eigvals(smoother_params['Sigma_cycles'])
        min_eig_cycles = jnp.min(jnp.real(eigs_cycles))
        print(f"  Sigma_cycles: min eigenvalue = {min_eig_cycles:.2e}")
        if min_eig_cycles < 1e-8:
            raise ValueError(f"Sigma_cycles has very small eigenvalue ({min_eig_cycles:.2e}). Numerical instability likely.")
    
    # Check Sigma_trends  
    if smoother_params['Sigma_trends'].shape[0] > 0:
        eigs_trends = jnp.linalg.eigvals(smoother_params['Sigma_trends'])
        min_eig_trends = jnp.min(jnp.real(eigs_trends))
        print(f"  Sigma_trends: min eigenvalue = {min_eig_trends:.2e}")
        if min_eig_trends < 1e-8:
            raise ValueError(f"Sigma_trends has very small eigenvalue ({min_eig_trends:.2e}). Numerical instability likely.")
    
    # 5. Extract other essential parameters for state-space construction
    # These are needed to build T_comp, R_comp on-the-fly
    
    # Dimensions
    smoother_params['k_trends'] = config_data['k_trends']
    smoother_params['k_stationary'] = config_data['k_stationary'] 
    smoother_params['var_order'] = config_data['var_order']
    smoother_params['k_states'] = config_data['k_states']
    smoother_params['n_trend_shocks'] = config_data['n_trend_shocks']
    
    # State names for mapping
    smoother_params['trend_names_with_shocks'] = config_data['trend_names_with_shocks']
    smoother_params['trend_var_names'] = config_data['trend_var_names']
    smoother_params['stationary_var_names'] = config_data['stationary_var_names']
    
    # Initial conditions (needed for init_x_comp, init_P_comp)
    smoother_params['init_x_means_flat'] = config_data['init_x_means_flat']
    smoother_params['init_P_diag_flat'] = config_data['init_P_diag_flat']
    
    print(f"\n✓ Successfully extracted smoother parameters: {list(smoother_params.keys())}")
    print(f"✓ Ready for efficient state-space matrix construction")
    
    return smoother_params

def validate_smoother_parameters(smoother_params: Dict) -> None:
    """
    Validate that smoother parameters are consistent and ready for use.
    Fail hard if anything is wrong - no fallbacks.
    """
    required_keys = ['phi_list', 'Sigma_cycles', 'Sigma_trends', 'k_trends', 'k_stationary', 'var_order']
    
    for key in required_keys:
        if key not in smoother_params:
            raise ValueError(f"CRITICAL: Required parameter '{key}' missing from smoother parameters.")
    
    # Check dimensions consistency
    k_stationary = smoother_params['k_stationary']
    k_trends = smoother_params['k_trends']
    var_order = smoother_params['var_order']
    n_trend_shocks = smoother_params['n_trend_shocks']
    
    # phi_list validation: depends on VAR order
    phi_list = smoother_params['phi_list']
    
    if var_order == 1:
        # For VAR(1), phi_list should be (k_stationary, k_stationary) - single matrix
        expected_phi_shape = (k_stationary, k_stationary)
        if phi_list.shape != expected_phi_shape:
            raise ValueError(f"VAR(1): phi_list shape {phi_list.shape} != expected {expected_phi_shape}")
    else:
        # For VAR(p), phi_list should be (var_order, k_stationary, k_stationary)
        expected_phi_shape = (var_order, k_stationary, k_stationary)
        if phi_list.shape != expected_phi_shape:
            raise ValueError(f"VAR({var_order}): phi_list shape {phi_list.shape} != expected {expected_phi_shape}")
    
    print(f"✓ phi_list shape {phi_list.shape} is correct for VAR({var_order})")
    
    # Sigma_cycles should be k_stationary x k_stationary
    sigma_cycles = smoother_params['Sigma_cycles']
    expected_cycles_shape = (k_stationary, k_stationary)
    if sigma_cycles.shape != expected_cycles_shape:
        raise ValueError(f"Sigma_cycles shape {sigma_cycles.shape} != expected {expected_cycles_shape}")
    
    print(f"✓ Sigma_cycles shape {sigma_cycles.shape} is correct")
    
    # Sigma_trends should be n_trend_shocks x n_trend_shocks
    sigma_trends = smoother_params['Sigma_trends']
    expected_trends_shape = (n_trend_shocks, n_trend_shocks)
    if sigma_trends.shape != expected_trends_shape:
        raise ValueError(f"Sigma_trends shape {sigma_trends.shape} != expected {expected_trends_shape}")
    
    print(f"✓ Sigma_trends shape {sigma_trends.shape} is correct")
    
    # Additional consistency checks
    if k_stationary <= 0:
        raise ValueError(f"k_stationary must be > 0, got {k_stationary}")
    
    if k_trends <= 0:
        raise ValueError(f"k_trends must be > 0, got {k_trends}")
    
    if var_order <= 0:
        raise ValueError(f"var_order must be > 0, got {var_order}")
    
    if n_trend_shocks < 0:
        raise ValueError(f"n_trend_shocks must be >= 0, got {n_trend_shocks}")
    
    print("✓ All smoother parameters validated successfully")

# Example usage:
"""
# In your main estimation function:
try:
    smoother_params = extract_smoother_parameters(posterior_samples, config_data, variable_names)
    validate_smoother_parameters(smoother_params)
    
    # Now pass these to the efficient smoother
    smoothed_states, simulation_results = run_efficient_smoother(
        smoother_params, y_data, key, **other_args
    )
except ValueError as e:
    print(f"FAILED: {e}")
    # Don't continue - fix the underlying issue
    raise
"""