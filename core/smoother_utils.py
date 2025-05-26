# --- smoother_utils.py (Fixed NonConcreteBooleanIndexError) ---
import numpy as np  
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
from typing import Dict, Any, Tuple, List, Union, Optional

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Import necessary functions from your existing modules
from core.var_ss_model import (
    _get_off_diagonal_indices, # Might be needed for building R, C etc.
    _parse_equation_jax,      # Needed for building C matrix
    # parse_initial_state_config, # Not needed here, initial state comes from deterministic sites
    _MODEL_JITTER # Use the same jitter
)
from utils.stationary_prior_jax_simplified import (
    create_companion_matrix_jax # Needed to build T matrix
)
# Need the filter/smoother logic from HybridDKSimulationSmoother
from utils.hybrid_dk_smoother import HybridDKSimulationSmoother 
# Need the raw simulate_state_space from Kalman_filter_jax
from utils.Kalman_filter_jax import simulate_state_space as simulate_state_space_raw


# Helper function to get state indices based on names (for building C, R, init_x, init_P)
# This map needs to be consistent with how the state vector is structured:
# [trends (k_trends), current stationary (k_stationary), lag 1 stationary (k_stationary), ..., lag p-1 stationary (k_stationary)]
def _get_state_indices_map(
    trend_var_names: Tuple[str, ...],
    stationary_var_names: Tuple[str, ...],
    var_order: int
) -> Dict[str, int]:
    """Creates a mapping from state name string to its index in the full state vector."""
    k_trends = len(trend_var_names)
    k_stationary = len(stationary_var_names)
    p = var_order
    
    state_indices = {}
    # Trend states
    for i, name in enumerate(trend_var_names):
        state_indices[name] = i
        
    # Stationary states (current and lagged)
    for lag in range(p):
        for i, name in enumerate(stationary_var_names):
            if lag == 0: # Current cycle
                state_indices[name] = k_trends + i
            else: # Lagged cycles
                lagged_state_name = f"{name}_t_minus_{lag}" 
                lagged_state_idx = k_trends + lag * k_stationary + i
                state_indices[lagged_state_name] = lagged_state_idx

    return state_indices


# Step 1: Extract Smoother Parameters for All Draws
def extract_smoother_parameters_all_draws(
    posterior_samples: Dict[str, jax.Array],
    config_data: Dict[str, Any] # Needed for parameter names etc.
) -> Dict[str, jax.Array]:
    """
    Extracts the necessary parameters for the smoother from all MCMC posterior draws.
    These parameters are the deterministic outputs from the NumPyro model run.
    
    Extracts: phi_list, Sigma_cycles, Sigma_trends_full, init_x_comp, init_P_comp,
              and measurement parameters.
              The first dimension of each extracted array corresponds to the number of draws.
    """
    smoother_params = {}

    # Extract parameters from deterministic sites
    # These names must match the numpyro.deterministic site names in var_ss_model.py
    required_deterministic_keys = [
        "phi_list", # (num_draws, p, k_stat, k_stat)
        "Sigma_cycles", # (num_draws, k_stat, k_stat)
        "Sigma_trends_full", # (num_draws, k_trends, k_trends)
        "init_x_comp", # (num_draws, k_states)
        "init_P_comp", # (num_draws, k_states, k_states)
    ]
    
    for key in required_deterministic_keys:
        if key not in posterior_samples:
            raise ValueError(f"Deterministic key '{key}' not found in posterior_samples. "
                             "Ensure these are exposed in the NumPyro model.")
        # Extract all draws for each parameter
        smoother_params[key] = posterior_samples[key]

    # Extract measurement parameters (if any) - these are sampled parameters, not deterministics
    # Need names from config_data
    measurement_param_names_tuple = config_data.get('measurement_param_names_tuple', ()) # Use .get for safety
    smoother_params['measurement_params'] = {} # Store as a dictionary
    
    for param_name in measurement_param_names_tuple:
        if param_name not in posterior_samples:
            # This should not happen if the MCMC run was successful and the parameter exists in config
            # Consider how to handle this: raise error, or fill with default?
            # For now, let's print a warning and potentially fill with NaNs or a default value
            # if a specific draw count is known, otherwise it's tricky.
            # Given we are taking all draws, if a parameter is missing, it's a more fundamental issue.
            print(f"Warning: Sampled measurement parameter '{param_name}' not found in posterior_samples.")
            # If we need to create a placeholder, we'd need to know the number of draws.
            # This can be inferred from another parameter, e.g., posterior_samples["phi_list"].shape[0]
            # However, it's better to ensure all necessary parameters are present.
            # For now, we'll skip adding it, or one could raise an error.
            # raise ValueError(f"Sampled measurement parameter '{param_name}' not found.")
            # Or, create a dummy array if a default behavior is desired:
            # num_draws = posterior_samples[required_deterministic_keys[0]].shape[0]
            # smoother_params['measurement_params'][param_name] = jnp.full((num_draws,), jnp.nan, dtype=_DEFAULT_DTYPE)
            continue # Skip this parameter if missing

        # Extract all draws for the measurement parameter
        param_value = posterior_samples[param_name]
        # Ensure it has a draw dimension if it's not a scalar shared across draws (unlikely for sampled params)
        if param_value.ndim == 0: # Scalar
            # This case is unlikely for sampled parameters which should vary per draw.
            # If it's a fixed value used across draws, it might need to be broadcasted.
            # For safety, let's assume it should have a draw dimension.
            print(f"Warning: Measurement parameter '{param_name}' is a scalar. Expected at least a draw dimension.")
            # Potentially broadcast or handle as an error. For now, we take it as is.
            # If broadcasting is needed:
            # num_draws = posterior_samples[required_deterministic_keys[0]].shape[0]
            # param_value = jnp.repeat(param_value, num_draws).reshape(num_draws, *param_value.shape)

        smoother_params['measurement_params'][param_name] = param_value
    
    return smoother_params

# Renamed function to serve as a helper for vmap
def _construct_ss_matrices_single_draw_helper(
     smoother_params_single_draw: Dict[str, jax.Array], # Parameters for a single draw
     static_config_data: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """
    Constructs the state-space matrices for a single draw.
    This function is intended to be vmapped.

    Args:
        smoother_params_single_draw: Dictionary of parameters for one draw.
        static_config_data: Static configuration data.

    Returns:
        A dictionary containing the constructed state-space matrices for one draw.
    """
    # --- Extract Static Configuration ---
    k_endog = static_config_data['k_endog']
    k_trends = static_config_data['k_trends']
    k_stationary = static_config_data['k_stationary']
    p = static_config_data['var_order']
    k_states = static_config_data['k_states']
    
    trend_var_names_tuple = static_config_data['trend_var_names']
    stationary_var_names_tuple = static_config_data['stationary_var_names']
    trend_names_with_shocks_tuple = static_config_data['trend_names_with_shocks'] # Subset of trend_var_names with shocks
    
    # Parsed model equations for C matrix construction
    parsed_model_eqs_jax_detailed = static_config_data['parsed_model_eqs_jax_detailed'] # (obs_idx, Tuple[term_type, state_idx_in_C_block, param_idx, sign])
    measurement_param_names_tuple = static_config_data['measurement_param_names_tuple']

    # --- Extract Dynamic Parameters (from smoother_params_single_draw) ---
    # These are now for a single draw
    phi_list = smoother_params_single_draw['phi_list']
    Sigma_cycles = smoother_params_single_draw['Sigma_cycles']
    Sigma_trends_full = smoother_params_single_draw['Sigma_trends_full']
    init_x_mean = smoother_params_single_draw['init_x_comp']
    init_P_cov = smoother_params_single_draw['init_P_comp']
    
    measurement_params_dict = smoother_params_single_draw['measurement_params'] # Dict {name: value} for single draw

    # --- Construct T_comp (from trends + companion matrix from phi_list) ---
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE)) # Trends block (identity)
    
    if k_stationary > 0:
        # phi_list for a single draw is (p, k_stat, k_stat)
        # create_companion_matrix_jax expects a list of arrays if phi_list is (num_draws, p, k_stat, k_stat)
        # However, for a single draw, phi_list is already (p, k_stat, k_stat)
        # We need to ensure it's a list of arrays for create_companion_matrix_jax if it's not already.
        # If phi_list is (p, k_stat, k_stat), then [phi_list[i] for i in range(p)] is appropriate.
        # The existing code: phi_list_jax = [jnp.asarray(phi, dtype=_DEFAULT_DTYPE) for phi in phi_list]
        # This assumes phi_list is an iterable of arrays (e.g., a list of arrays).
        # If phi_list from smoother_params_single_draw is (p, k_stat, k_stat), it should be fine.
        phi_list_for_companion = [jnp.asarray(phi_list[i], dtype=_DEFAULT_DTYPE) for i in range(p)]
        companion_matrix = create_companion_matrix_jax(phi_list_for_companion, p, k_stationary)
        T_comp = T_comp.at[k_trends:, k_trends:].set(companion_matrix) # AR block


    # --- Construct R_aug (Shock Impact Matrix for Simulation) ---
    n_trend_shocks = len(trend_names_with_shocks_tuple)
    n_shocks_sim = n_trend_shocks + k_stationary

    R_aug = jnp.zeros((k_states, n_shocks_sim), dtype=_DEFAULT_DTYPE)

    if n_trend_shocks > 0:
        trend_var_names_list = list(trend_var_names_tuple)
        # Sigma_trends_full for single draw is (k_trends, k_trends)
        trend_shock_variances = jnp.array([
             Sigma_trends_full[trend_var_names_list.index(name), trend_var_names_list.index(name)]
             for name in trend_names_with_shocks_tuple
        ], dtype=_DEFAULT_DTYPE)
        sqrt_trend_variances = jnp.sqrt(jnp.maximum(trend_shock_variances, _MODEL_JITTER))
        for i in range(n_trend_shocks):
            shocked_trend_name = trend_names_with_shocks_tuple[i]
            trend_state_idx = trend_var_names_list.index(shocked_trend_name)
            R_aug = R_aug.at[trend_state_idx, i].set(sqrt_trend_variances[i])

    if k_stationary > 0:
        # Sigma_cycles for single draw is (k_stat, k_stat)
        L_cycles = jsl.cholesky(Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE), lower=True)
        R_aug = R_aug.at[k_trends:k_trends+k_stationary, n_trend_shocks : n_trend_shocks+k_stationary].set(L_cycles)


    # --- Construct C_comp (Measurement Matrix for Simulation) ---
    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    state_indices_full = _get_state_indices_map(trend_var_names_tuple, stationary_var_names_tuple, p)
    measurement_param_names_list = list(measurement_param_names_tuple)

    for obs_idx, terms_for_obs in parsed_model_eqs_jax_detailed:
        for term_type, state_idx_in_C_block, param_idx_if_any, sign in terms_for_obs:
            is_param_term = (term_type == 1)
            param_value = jnp.array(1.0, dtype=_DEFAULT_DTYPE)
            if is_param_term:
                param_name = measurement_param_names_list[param_idx_if_any]
                # measurement_params_dict for single draw has scalar values for each param
                param_value = measurement_params_dict.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE))
            full_state_idx = state_idx_in_C_block
            C_comp = C_comp.at[obs_idx, full_state_idx].add(sign * param_value)

    # 4. H_comp (Observation Noise Covariance for Simulation)
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)

    # --- Initial State Mean and Covariance for Simulation ---
    # init_x_mean for single draw is (k_states,)
    # init_P_cov for single draw is (k_states, k_states)
    init_x_sim = init_x_mean
    init_P_sim = init_P_cov

    return {
        "T_comp": T_comp,
        "R_aug": R_aug,
        "C_comp": C_comp,
        "H_comp": H_comp,
        "init_x_sim": init_x_sim,
        "init_P_sim": init_P_sim
    }

# Step 2: Construct State-Space Matrices from Smoother Parameters (Batched Version)
@jax.jit(static_argnames=['static_config_data']) # JIT this builder
def construct_ss_matrices_all_draws(
     smoother_params_all_draws: Dict[str, jax.Array], # Batched parameters
     static_config_data: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """
    Constructs the state-space matrices (T, R, C, H, init_x, init_P) for all
    smoother draws using the extracted parameters. This function is JIT compiled and vmapped.

    Args:
        smoother_params_all_draws: Dictionary from extract_smoother_parameters_all_draws.
                                   Each JAX array has a leading batch dimension (num_draws).
        static_config_data: Dictionary from load_config_and_prepare_jax_static_args.

    Returns:
        A dictionary where each state-space matrix (e.g., "T_comp") is a JAX array
        with a leading batch dimension representing the number of draws.
        Example: T_comp will have shape (num_draws, k_states, k_states).
    """
    # vmap the helper function over the draw dimension of smoother_params_all_draws
    # static_config_data is passed as None in in_axes, meaning it's treated as static.
    # For smoother_params_all_draws (a dictionary), in_axes=0 means vmap over the
    # leading dimension of each JAX array *value* in the dictionary.
    batched_matrices = jax.vmap(
        _construct_ss_matrices_single_draw_helper,
        in_axes=(0, None) # Map over smoother_params_all_draws, keep static_config_data fixed
    )(smoother_params_all_draws, static_config_data)

    return batched_matrices


# Step 3: Batch Quantile Computation
@jax.jit
def compute_batch_quantiles(
    all_simulated_paths: jax.Array, 
    quantiles_to_compute: Union[List[float], jax.Array]
) -> jax.Array:
    """
    Computes specified quantiles from a batch of simulated paths.

    Args:
        all_simulated_paths: A JAX array of shape (num_draws, T, k_states)
                             containing all simulated state paths.
        quantiles_to_compute: A list or JAX array of quantiles (e.g., [0.025, 0.5, 0.975]).

    Returns:
        A JAX array quantile_estimates of shape (T, k_states, n_quantiles).
    """
    # Ensure quantiles_to_compute is a JAX array
    q_array = jnp.asarray(quantiles_to_compute, dtype=_DEFAULT_DTYPE)
    
    # Scale quantiles from [0, 1] to [0, 100] for jnp.percentile
    q_scaled = q_array * 100.0
    
    # Compute percentiles along the simulation draw axis (axis=0)
    # Input shape: (num_draws, T, k_states)
    # Output shape of percentile: (n_quantiles, T, k_states)
    percentile_results = jnp.percentile(all_simulated_paths, q=q_scaled, axis=0)
    
    # Transpose to desired output shape: (T, k_states, n_quantiles)
    quantile_estimates = jnp.transpose(percentile_results, (1, 2, 0))
    
    return quantile_estimates


# Step 6: Single Simulation Path Function (Potentially deprecated by batch version)
# This function will use the SS matrices from _construct_ss_matrices_single_draw_helper.
# Re-include it here for clarity in the full smoother_utils.py file.

# --- Fixed run_single_simulation_path_for_dk Function ---
# This function might be kept for debugging or single-path specific use cases.
def run_single_simulation_path_for_dk(
    ss_matrices_sim: Dict[str, jax.Array],
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    key: jax.random.PRNGKey,
    config_data: Dict[str, Any], # Note: was static_config_data, now general config_data
    smoother_params_single_draw: Dict[str, jax.Array] 
) -> jax.Array:
    """
    Generates a single simulation path for the Durbin-Koopman smoother.
    """
    T_steps = original_ys_dense.shape[0]
    n_state = ss_matrices_sim['T_comp'].shape[0]
    
    if T_steps == 0:
        return jnp.empty((0, n_state), dtype=_DEFAULT_DTYPE)

    key_sim, key_filter_smooth = random.split(key)

    # 1. Simulate a path from the state space model (x* and y*)
    x_star_path, y_star_dense_sim = simulate_state_space_raw(
        ss_matrices_sim['T_comp'],
        ss_matrices_sim['R_aug'],
        ss_matrices_sim['C_comp'],
        ss_matrices_sim['H_comp'],
        ss_matrices_sim['init_x_sim'], 
        ss_matrices_sim['init_P_sim'],  
        key_sim,
        T_steps
    )

    # 2. Reconstruct Q_comp for the filter from parameters
    Sigma_cycles = smoother_params_single_draw['Sigma_cycles']
    Sigma_trends_full = smoother_params_single_draw['Sigma_trends_full']
    
    # k_trends and k_stationary are needed for Q_comp_filter reconstruction.
    # These should be available in config_data or derived from Sigma shapes.
    # Assuming Sigma_trends_full is (k_trends, k_trends) and Sigma_cycles is (k_stationary, k_stationary)
    k_trends = Sigma_trends_full.shape[0]
    k_stationary = Sigma_cycles.shape[0]
    # n_state must match ss_matrices_sim['T_comp'].shape[0]

    Q_comp_filter = jnp.zeros((n_state, n_state), dtype=_DEFAULT_DTYPE)
    Q_comp_filter = Q_comp_filter.at[:k_trends, :k_trends].set(Sigma_trends_full)
    if k_stationary > 0:
        Q_comp_filter = Q_comp_filter.at[k_trends:k_trends+k_stationary, k_trends:k_trends+k_stationary].set(Sigma_cycles)
    
    Q_comp_filter = (Q_comp_filter + Q_comp_filter.T) / 2.0
    Q_comp_filter = Q_comp_filter + _MODEL_JITTER * jnp.eye(n_state, dtype=_DEFAULT_DTYPE)

    # R_for_filter is the Cholesky of Q_comp_filter
    try:
        R_for_filter = jax.scipy.linalg.cholesky(Q_comp_filter, lower=True)
    except Exception: # More specific: jax.errors.LinAlgError
        # print("Warning: Cholesky of Q_comp_filter failed in single path. Using diagonal sqrt.")
        diag_Q = jnp.diag(Q_comp_filter)
        safe_diag_Q = jnp.maximum(diag_Q, _MODEL_JITTER)
        R_for_filter = jnp.diag(jnp.sqrt(safe_diag_Q))


    # 3. Create temporary smoother and run filter/smoother on simulated data
    temp_dk_smoother = HybridDKSimulationSmoother(
        ss_matrices_sim['T_comp'],
        R_for_filter,
        ss_matrices_sim['C_comp'],
        ss_matrices_sim['H_comp'],
        ss_matrices_sim['init_x_sim'],
        ss_matrices_sim['init_P_sim']
    )

    # Filter and smooth the simulated data
    filter_results_star_dense = temp_dk_smoother._filter_internal(y_star_dense_sim)
    x_smooth_star_dense, _ = temp_dk_smoother._rts_smoother_backend(filter_results_star_dense)

    # 4. Apply the Durbin-Koopman formula
    x_draw = x_smooth_original_dense + (x_star_path - x_smooth_star_dense)

    # Ensure final draw is finite
    x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw))

    return x_draw


# Updated JITted wrapper function (Potentially deprecated)
# @jax.jit(static_argnames=['static_config_data']) # static_config_data may not be static if config_data is used
def jit_run_single_simulation_path_for_dk_wrapper(
    smoother_params_single_draw: Dict[str, jax.Array], 
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    key: jax.random.PRNGKey,
    static_config_data: Dict[str, Any] # Or general config_data if run_single_simulation_path_for_dk changes
):
    """
    JIT wrapper for single simulation path. Consider if this is still needed
    with batch processing.
    """
    ss_matrices_sim_single_draw = _construct_ss_matrices_single_draw_helper(
        smoother_params_single_draw, static_config_data
    )

    simulated_states_path = run_single_simulation_path_for_dk(
        ss_matrices_sim_single_draw, 
        original_ys_dense,
        x_smooth_original_dense,
        key,
        static_config_data, 
        smoother_params_single_draw
    )
    return simulated_states_path


# --- Batch Simulation Smoother Function --- 
def _run_single_simulation_draw_for_batch(
    key_single_draw: jax.random.PRNGKey,
    T_comp_single: jax.Array,
    R_aug_single: jax.Array,
    C_comp_single: jax.Array,
    H_comp_single: jax.Array,
    init_x_sim_single: jax.Array,
    init_P_sim_single: jax.Array,
    Sigma_cycles_single_draw: jax.Array,
    Sigma_trends_full_single_draw: jax.Array,
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    static_config_data: Dict[str, Any]
) -> jax.Array:
    """
    Helper function for run_batch_simulation_smoother.
    Performs Durbin-Koopman simulation for a single draw's parameters.
    """
    T_steps = original_ys_dense.shape[0]
    # n_state can be inferred from T_comp_single.shape[0] or x_smooth_original_dense.shape[1]
    n_state = T_comp_single.shape[0]

    if T_steps == 0:
        return jnp.empty((0, n_state), dtype=_DEFAULT_DTYPE)

    key_sim, key_filter_smooth = random.split(key_single_draw)

    # 1. Simulate a path from the state space model (x* and y*)
    x_star_path, y_star_dense_sim = simulate_state_space_raw(
        T_comp_single,
        R_aug_single,
        C_comp_single,
        H_comp_single,
        init_x_sim_single,
        init_P_sim_single,
        key_sim,
        T_steps
    )

    # 2. Reconstruct Q_comp_filter for this draw
    k_trends = static_config_data['k_trends']
    k_stationary = static_config_data['k_stationary']
    # n_state is already derived: k_states = static_config_data['k_states']

    Q_comp_filter = jnp.zeros((n_state, n_state), dtype=_DEFAULT_DTYPE)
    Q_comp_filter = Q_comp_filter.at[:k_trends, :k_trends].set(Sigma_trends_full_single_draw)
    if k_stationary > 0: # Ensure Sigma_cycles is only set if k_stationary > 0
        Q_comp_filter = Q_comp_filter.at[k_trends:k_trends+k_stationary, k_trends:k_trends+k_stationary].set(Sigma_cycles_single_draw)
    
    Q_comp_filter = (Q_comp_filter + Q_comp_filter.T) / 2.0
    Q_comp_filter = Q_comp_filter + _MODEL_JITTER * jnp.eye(n_state, dtype=_DEFAULT_DTYPE)

    # 3. Create R_for_filter (Cholesky of Q_comp_filter)
    try:
        R_for_filter = jax.scipy.linalg.cholesky(Q_comp_filter, lower=True)
    except Exception: # Catch LinAlgError for non-positive definite
        # Fallback: use diagonal of sqrt if Cholesky fails
        diag_Q = jnp.diag(Q_comp_filter)
        safe_diag_Q = jnp.maximum(diag_Q, _MODEL_JITTER) # Ensure non-negativity before sqrt
        R_for_filter = jnp.diag(jnp.sqrt(safe_diag_Q))
        # Consider logging this event if a logger is available/appropriate
        # print(f"Warning: Cholesky decomposition of Q_comp_filter failed for a draw. Using diagonal sqrt fallback.")


    # 4. Instantiate HybridDKSimulationSmoother
    temp_dk_smoother = HybridDKSimulationSmoother(
        T_comp_single,
        R_for_filter, # Use the Cholesky factor
        C_comp_single,
        H_comp_single,
        init_x_sim_single,
        init_P_sim_single
    )

    # 5. Filter and Smooth Simulated Data
    filter_results_star_dense = temp_dk_smoother._filter_internal(y_star_dense_sim)
    x_smooth_star_dense, _ = temp_dk_smoother._rts_smoother_backend(filter_results_star_dense)

    # 6. Apply Durbin-Koopman Formula
    x_draw = x_smooth_original_dense + (x_star_path - x_smooth_star_dense)

    # Ensure final draw is finite
    x_draw = jnp.where(jnp.isfinite(x_draw), x_draw, jnp.zeros_like(x_draw))

    return x_draw


@jax.jit(static_argnames=('num_draws', 'static_config_data'))
def run_batch_simulation_smoother(
    key: jax.random.PRNGKey,
    num_draws: int,
    ss_matrices_all_draws: Dict[str, jax.Array],
    smoother_params_all_draws: Dict[str, jax.Array], # Contains Sigma_cycles, Sigma_trends_full
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    static_config_data: Dict[str, Any]
) -> jax.Array:
    """
    Generates all Durbin-Koopman simulation paths in parallel using jax.vmap.
    """
    subkeys = random.split(key, num_draws)

    # Extract batched components for vmap
    T_comp_all = ss_matrices_all_draws['T_comp']
    R_aug_all = ss_matrices_all_draws['R_aug']
    C_comp_all = ss_matrices_all_draws['C_comp']
    H_comp_all = ss_matrices_all_draws['H_comp']
    init_x_sim_all = ss_matrices_all_draws['init_x_sim']
    init_P_sim_all = ss_matrices_all_draws['init_P_sim']

    Sigma_cycles_all = smoother_params_all_draws['Sigma_cycles']
    Sigma_trends_full_all = smoother_params_all_draws['Sigma_trends_full']

    # Vmap the helper function over all draws
    all_simulated_paths = jax.vmap(
        _run_single_simulation_draw_for_batch,
        in_axes=(0,  # subkeys
                   0,  # T_comp_single
                   0,  # R_aug_single
                   0,  # C_comp_single
                   0,  # H_comp_single
                   0,  # init_x_sim_single
                   0,  # init_P_sim_single
                   0,  # Sigma_cycles_single_draw
                   0,  # Sigma_trends_full_single_draw
                   None, # original_ys_dense
                   None, # x_smooth_original_dense
                   None  # static_config_data
                  )
    )(subkeys,
      T_comp_all, R_aug_all, C_comp_all, H_comp_all, init_x_sim_all, init_P_sim_all,
      Sigma_cycles_all, Sigma_trends_full_all,
      original_ys_dense, x_smooth_original_dense, static_config_data)

    return all_simulated_paths

# The old run_single_simulation_path_for_dk and its JIT wrapper jit_run_single_simulation_path_for_dk_wrapper
# might be deprecated or removed if this batch function becomes the primary interface.
# For now, they are kept.