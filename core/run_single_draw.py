# --- run_single_draw.py (Simplified Working Version) ---

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from typing import Dict, Any, Tuple, List, Union, Optional
import time

# Configure JAX
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Import the state-space matrix building function and Kalman filter
from core.var_ss_model import build_state_space_matrices_jit
from utils.Kalman_filter_jax import KalmanFilter
from utils.hybrid_dk_smoother import HybridDKSimulationSmoother

def convert_to_hashable(obj):
    """
    Recursively convert lists to tuples to make objects hashable for JAX JIT.
    """
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in obj.items()))
    else:
        return obj

def run_simulation_smoother_single_params_simple(
    posterior_mean_params: Dict[str, jax.Array],
    y_data: jax.Array,
    key: jax.random.PRNGKey,
    config_data: Dict[str, Any],
    static_valid_obs_idx: jax.Array,
    static_n_obs_actual: int,
    num_draws: int
) -> Tuple[jax.Array, Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]]:
    """
    Non-JIT simulation smoother for a single set of parameters.
    This version avoids JIT compilation to sidestep hashability issues.
    """
    
    # Build state-space matrices using the posterior mean parameters
    ss_matrices = build_state_space_matrices_jit(posterior_mean_params, config_data)
    
    # Extract matrices
    T_comp = ss_matrices['T_comp']
    R_comp = ss_matrices['R_comp']
    C_comp = ss_matrices['C_comp']
    H_comp = ss_matrices['H_comp']
    init_x_comp = ss_matrices['init_x_comp']
    init_P_comp = ss_matrices['init_P_comp']
    
    # Prepare static arguments for Kalman filter
    static_C_obs = C_comp[static_valid_obs_idx, :]
    static_H_obs = H_comp[static_valid_obs_idx[:, None], static_valid_obs_idx]
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)
    
    # Create and run Kalman filter for original smoother
    kf = KalmanFilter(T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp)
    filter_results = kf.filter(y_data, static_valid_obs_idx, static_n_obs_actual, 
                              static_C_obs, static_H_obs, static_I_obs)
    
    # Run RTS smoother to get smoothed states
    x_smooth_original, P_smooth_original = kf.smooth(
        y_data, filter_results, static_valid_obs_idx, static_n_obs_actual,
        static_C_obs, static_H_obs, static_I_obs
    )
    
    if num_draws == 1:
        # Return just the smoothed states for single draw
        return x_smooth_original, x_smooth_original
    else:
        # For multiple draws, create hybrid DK smoother
        dk_smoother = HybridDKSimulationSmoother(
            T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp
        )
        
        # Run simulation smoother
        mean_sim, median_sim, all_draws = dk_smoother.run_smoother_draws(
            y_data, key, num_draws, x_smooth_original
        )
        
        return x_smooth_original, (mean_sim, median_sim, all_draws)


def run_simulation_smoother_single_params_jit(
    posterior_mean_params: Dict[str, jax.Array],
    y_data: jax.Array,
    key: jax.random.PRNGKey,
    **static_smoother_args
) -> Tuple[jax.Array, Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]]:
    """
    Wrapper function that calls the simplified (non-JIT) version.
    This avoids all the hashability issues while still providing the same interface.
    """
    
    # Extract necessary arguments from static_smoother_args
    k_endog = static_smoother_args['static_k_endog']
    k_trends = static_smoother_args['static_k_trends']
    k_stationary = static_smoother_args['static_k_stationary']
    p = static_smoother_args['static_p']
    k_states = static_smoother_args['static_k_states']
    n_trend_shocks = static_smoother_args['static_n_trend_shocks']
    num_off_diag = static_smoother_args['static_num_off_diag']
    
    static_off_diag_rows = static_smoother_args['static_off_diag_rows']
    static_off_diag_cols = static_smoother_args['static_off_diag_cols']
    static_valid_obs_idx = static_smoother_args['static_valid_obs_idx']
    static_n_obs_actual = static_smoother_args['static_n_obs_actual']
    
    model_eqs_parsed = static_smoother_args['model_eqs_parsed']
    initial_conds_parsed = static_smoother_args['initial_conds_parsed']
    trend_names_with_shocks = static_smoother_args['trend_names_with_shocks']
    stationary_var_names = static_smoother_args['stationary_var_names']
    trend_var_names = static_smoother_args['trend_var_names']
    measurement_params_config = static_smoother_args['measurement_params_config']
    num_draws = static_smoother_args['num_draws']
    
    # Convert hashable structures back to usable format
    if isinstance(model_eqs_parsed, tuple):
        # Convert tuple back to list format expected by config_data
        model_eqs_list = []
        for obs_idx, terms_tuple in model_eqs_parsed:
            if isinstance(terms_tuple, tuple):
                terms_list = list(terms_tuple)
                model_eqs_list.append((obs_idx, terms_list))
            else:
                model_eqs_list.append((obs_idx, terms_tuple))
        model_eqs_parsed = model_eqs_list
    
    # Convert initial_conds_parsed back to dictionary format
    if isinstance(initial_conds_parsed, (tuple, frozenset)):
        if isinstance(initial_conds_parsed, frozenset):
            initial_conds_dict = {}
            for state_name, state_data in initial_conds_parsed:
                # state_data is a frozenset of (key, value) pairs
                state_dict = dict(state_data)
                initial_conds_dict[state_name] = state_dict
        else:  # It's a tuple
            initial_conds_dict = {}
            for state_name, mean_val, var_val in initial_conds_parsed:
                initial_conds_dict[state_name] = {'mean': mean_val, 'var': var_val}
        initial_conds_parsed = initial_conds_dict
    
    # Convert measurement_params_config back to list of dicts
    measurement_params_list = []
    if isinstance(measurement_params_config, tuple):
        for param_tuple in measurement_params_config:
            if isinstance(param_tuple, tuple) and len(param_tuple) > 0:
                param_dict = dict(param_tuple)
                measurement_params_list.append(param_dict)
    measurement_params_config = measurement_params_list
    
    # Reconstruct config_data structure needed by build_state_space_matrices_jit
    config_data = {
        'k_endog': k_endog,
        'k_trends': k_trends,
        'k_stationary': k_stationary,
        'var_order': p,
        'k_states': k_states,
        'n_trend_shocks': n_trend_shocks,
        'num_off_diag': num_off_diag,
        'static_off_diag_indices': (static_off_diag_rows, static_off_diag_cols),
        'trend_names_with_shocks': trend_names_with_shocks,
        'trend_var_names': trend_var_names,
        'stationary_var_names': stationary_var_names,
        'measurement_param_names_tuple': tuple(p.get('name', '') for p in measurement_params_config if 'name' in p),
        'model_equations_parsed': model_eqs_parsed,
        'initial_conditions_parsed': initial_conds_parsed,
        # Add flat arrays for initial conditions
        'init_x_means_flat': jnp.zeros(k_states, dtype=_DEFAULT_DTYPE),  # Will be reconstructed
        'init_P_diag_flat': jnp.ones(k_states, dtype=_DEFAULT_DTYPE),   # Will be reconstructed
        # Add parsed model equations in detailed format (simplified for now)
        'parsed_model_eqs_jax_detailed': tuple(
            (obs_idx, tuple((0, i, -1, 1.0) for i, _ in enumerate(terms))) 
            for obs_idx, terms in model_eqs_parsed
        )
    }
    
    # Reconstruct flat initial condition arrays from the parsed data
    full_state_names = list(trend_var_names)
    for i in range(p):
        for stat_var in stationary_var_names:
            if i == 0:
                full_state_names.append(stat_var)
            else:
                full_state_names.append(f"{stat_var}_t_minus_{i}")
    
    init_x_means_list = []
    init_P_diag_list = []
    for state_name in full_state_names:
        base_name = state_name.split("_t_minus_")[0] if "_t_minus_" in state_name else state_name
        if state_name in initial_conds_parsed:
            init_x_means_list.append(float(initial_conds_parsed[state_name]['mean']))
            init_P_diag_list.append(float(initial_conds_parsed[state_name]['var']))
        elif base_name in initial_conds_parsed:
            init_x_means_list.append(float(initial_conds_parsed[base_name]['mean']))
            init_P_diag_list.append(float(initial_conds_parsed[base_name]['var']))
        else:
            init_x_means_list.append(0.0)
            init_P_diag_list.append(1.0)
    
    config_data['init_x_means_flat'] = jnp.array(init_x_means_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_list, dtype=_DEFAULT_DTYPE)
    
    # Call the simplified (non-JIT) version
    return run_simulation_smoother_single_params_simple(
        posterior_mean_params,
        y_data,
        key,
        config_data,
        static_valid_obs_idx,
        static_n_obs_actual,
        num_draws
    )