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


# # Step 1: Extract Smoother Parameters for Single Draw
# def extract_smoother_parameters_single_draw(
#     posterior_samples: Dict[str, jax.Array],
#     draw_idx: int,
#     config_data: Dict[str, Any] # Needed for parameter names etc.
# ) -> Dict[str, jax.Array]:
#     """
#     Extracts the necessary parameters for the smoother from a single MCMC posterior draw.
#     These parameters are the deterministic outputs from the NumPyro model run.
    
#     Extracts: phi_list, Sigma_cycles, Sigma_trends_full, init_x_comp, init_P_comp,
#               and measurement parameters.
#     """
#     smoother_params = {}

#     # Extract parameters from deterministic sites
#     # These names must match the numpyro.deterministic site names in var_ss_model.py
    
#     required_deterministic_keys = [
#         "phi_list", # list of p, k_stat x k_stat arrays
#         "Sigma_cycles", # k_stat x k_stat matrix
#         "Sigma_trends_full", # k_trends x k_trends matrix
#         "init_x_comp", # k_states vector (mean)
#         "init_P_comp", # k_states x k_states matrix (covariance)
#     ]
    
#     for key in required_deterministic_keys:
#         if key not in posterior_samples:
#             raise ValueError(f"Deterministic key '{key}' not found in posterior_samples. "
#                              "Ensure these are exposed in the NumPyro model.")
#         # Extract the specific draw for each parameter
#         smoother_params[key] = posterior_samples[key][draw_idx]

#     # Extract measurement parameters (if any) - these are sampled parameters, not deterministics
#     # Need names from config_data
#     measurement_param_names_tuple = config_data['measurement_param_names_tuple']
#     smoother_params['measurement_params'] = {} # Store as a dictionary
    
#     for param_name in measurement_param_names_tuple:
#         if param_name not in posterior_samples:
#             # This should not happen if the MCMC run was successful and the parameter exists in config
#             print(f"Warning: Sampled measurement parameter '{param_name}' not found in posterior_samples. Using 0.0.")
#             smoother_params['measurement_params'][param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
#         else:
#             # Extract the specific draw for the measurement parameter
#             smoother_params['measurement_params'][param_name] = posterior_samples[param_name][draw_idx]


#     return smoother_params

# Replace the extract_smoother_parameters_single_draw function with this debug version:

def extract_smoother_parameters_single_draw(
    posterior_samples: Dict[str, jax.Array],
    draw_idx: int,
    config_data: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """
    DEBUG VERSION: Extracts smoother parameters with detailed error reporting.
    """
    print(f"\n=== DEBUG: Extracting parameters for draw {draw_idx} ===")
    print(f"Available keys in posterior_samples: {list(posterior_samples.keys())}")
    
    smoother_params = {}
    
    # Check shapes of deterministic sites
    required_deterministic_keys = [
        "phi_list", "Sigma_cycles", "Sigma_trends_full", 
        "init_x_comp", "init_P_comp"
    ]
    
    for key in required_deterministic_keys:
        if key in posterior_samples:
            param_shape = posterior_samples[key].shape
            print(f"  {key}: shape = {param_shape}")
            
            if len(param_shape) == 0:  # Scalar
                print(f"    ERROR: {key} is scalar, expected array with batch dimension")
                continue
            elif param_shape[0] <= draw_idx:
                print(f"    ERROR: {key} has only {param_shape[0]} samples, need draw_idx={draw_idx}")
                continue
            else:
                smoother_params[key] = posterior_samples[key][draw_idx]
                print(f"    SUCCESS: Extracted {key} for draw {draw_idx}")
        else:
            print(f"    ERROR: {key} not found in posterior_samples")
    
    # Check measurement parameters
    measurement_param_names = config_data.get('measurement_param_names_tuple', ())
    print(f"  Looking for measurement params: {measurement_param_names}")
    
    smoother_params['measurement_params'] = {}
    for param_name in measurement_param_names:
        if param_name in posterior_samples:
            param_shape = posterior_samples[param_name].shape
            print(f"    {param_name}: shape = {param_shape}")
            
            if len(param_shape) == 0:  # Scalar
                smoother_params['measurement_params'][param_name] = posterior_samples[param_name]
            elif param_shape[0] > draw_idx:
                smoother_params['measurement_params'][param_name] = posterior_samples[param_name][draw_idx]
            else:
                print(f"      ERROR: {param_name} has only {param_shape[0]} samples")
                smoother_params['measurement_params'][param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
        else:
            print(f"      WARNING: {param_name} not found, using 0.0")
            smoother_params['measurement_params'][param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
    
    print(f"=== DEBUG: Successfully extracted {len(smoother_params)} parameter groups ===\n")
    return smoother_params

# Step 2: Construct State-Space Matrices from Smoother Parameters
#@jax.jit(static_argnames=['static_config_data']) # JIT this builder
def construct_ss_matrices_from_smoother_params(
     smoother_params: Dict[str, jax.Array],
     static_config_data: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """
    Constructs the state-space matrices (T, R, C, H, init_x, init_P) for a single
    smoother draw using the extracted parameters (phi_list, Sigma_cycles, etc.)
    and static configuration. This function is JIT compiled.

    Args:
        smoother_params: Dictionary from extract_smoother_parameters_single_draw.
                         Should contain: phi_list, Sigma_cycles, Sigma_trends_full,
                         init_x_comp, init_P_comp, measurement_params.
        static_config_data: Dictionary from load_config_and_prepare_jax_static_args.

    Returns:
        A dictionary containing the constructed state-space matrices for the smoother.
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

    # --- Extract Dynamic Parameters (from smoother_params) ---
    # Ensure these keys exist in smoother_params
    phi_list = smoother_params['phi_list']
    Sigma_cycles = smoother_params['Sigma_cycles']
    Sigma_trends_full = smoother_params['Sigma_trends_full'] # k_trends x k_trends
    init_x_mean = smoother_params['init_x_comp'] # k_states vector mean
    init_P_cov = smoother_params['init_P_comp'] # k_states x k_states cov matrix
    
    measurement_params_dict = smoother_params['measurement_params'] # Dict {name: value}

    # --- Construct T_comp (from trends + companion matrix from phi_list) ---
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE)) # Trends block (identity)
    
    if k_stationary > 0:
        # create_companion_matrix_jax takes phi_list (list of p, k_stat x k_stat), p, k_stat
        # Need to ensure phi_list is a list of JAX arrays
        phi_list_jax = [jnp.asarray(phi, dtype=_DEFAULT_DTYPE) for phi in phi_list] # Ensure JAX array and dtype
        companion_matrix = create_companion_matrix_jax(phi_list_jax, p, k_stationary)
        T_comp = T_comp.at[k_trends:, k_trends:].set(companion_matrix) # AR block


    # --- Construct R_aug (Shock Impact Matrix for Simulation) ---
    # R_aug maps (n_trend_shocks + k_stationary) standard normal shocks to state dynamics.
    # R_aug @ R_aug.T = Q_comp_sim = block_diag(Sigma_trends_sim, Sigma_cycles)
    # Sigma_trends_sim is diagonal (n_trend_shocks x n_trend_shocks) with variances for shocked trends.
    
    n_trend_shocks = len(trend_names_with_shocks_tuple)
    n_shocks_sim = n_trend_shocks + k_stationary # Total shocks for simulation R_aug

    R_aug = jnp.zeros((k_states, n_shocks_sim), dtype=_DEFAULT_DTYPE)

    # 1. Trends: Place sqrt(variances) on the diagonal columns corresponding to shocks
    # Get variances for trends *with shocks* from Sigma_trends_full.
    # Requires mapping trend_names_with_shocks_tuple to trend_var_names_tuple indices.
    if n_trend_shocks > 0:
        trend_var_names_list = list(trend_var_names_tuple) # Convert to list for .index()
        trend_shock_variances = jnp.array([
             Sigma_trends_full[trend_var_names_list.index(name), trend_var_names_list.index(name)] # Get diagonal variance
             for name in trend_names_with_shocks_tuple # Iterate through shocked trend names
        ], dtype=_DEFAULT_DTYPE)

        sqrt_trend_variances = jnp.sqrt(jnp.maximum(trend_shock_variances, _MODEL_JITTER))

        # Place these on the diagonal of R_aug's trend block (first n_trend_shocks columns)
        for i in range(n_trend_shocks):
            shocked_trend_name = trend_names_with_shocks_tuple[i]
            trend_state_idx = trend_var_names_list.index(shocked_trend_name) # Index in state vector (0 to k_trends-1)
            R_aug = R_aug.at[trend_state_idx, i].set(sqrt_trend_variances[i])


    # 2. Cycles: Place the Cholesky of Sigma_cycles in R_aug's cycle block.
    # Affects current cycle states (k_trends to k_trends + k_stationary -1).
    # Cycle shocks start at column n_trend_shocks in R_aug.
    if k_stationary > 0:
        L_cycles = jsl.cholesky(Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE), lower=True)
        R_aug = R_aug.at[k_trends:k_trends+k_stationary, n_trend_shocks : n_trend_shocks+k_stationary].set(L_cycles)


    # --- Construct C_comp (Measurement Matrix for Simulation) ---
    # C_comp maps [trends, cycle_t, cycle_t-1, ..., cycle_t-p+1] to observables.
    # Use the detailed parsed structure and dynamic measurement parameter values.
    
    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    
    # Need a mapping from state name string to full state vector index (including lags)
    state_indices_full = _get_state_indices_map(trend_var_names_tuple, stationary_var_names_tuple, p)

    # The parsed_model_eqs_jax_detailed contains (obs_idx, Tuple[term_type, state_idx_in_C_block, param_idx, sign]).
    # state_idx_in_C_block is the index in the [trends | current cycles] block (0 to k_trends+k_stationary-1).
    # We need to map this index to the correct index in the *full* state vector (k_states).
    # Indices 0 to k_trends-1 in the C-block map to indices 0 to k_trends-1 in the full vector (trends).
    # Indices k_trends to k_trends+k_stationary-1 in the C-block map to indices k_trends to k_trends+k_stationary-1 in the full vector (current cycles).
    # So, the state_idx_in_C_block directly corresponds to the index in the full state vector for trends and current cycles.

    measurement_param_names_list = list(measurement_param_names_tuple) # For getting value from dict

    for obs_idx, terms_for_obs in parsed_model_eqs_jax_detailed: 
        for term_type, state_idx_in_C_block, param_idx_if_any, sign in terms_for_obs: 
            is_param_term = (term_type == 1)
            
            param_value = jnp.array(1.0, dtype=_DEFAULT_DTYPE) # Default to 1.0 for direct terms
            if is_param_term:
                # Get parameter name from index
                param_name = measurement_param_names_list[param_idx_if_any]
                # Get parameter value from dynamic measurement_params_dict
                # Use .get with a default in case the parameter is unexpectedly missing
                param_value = measurement_params_dict.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE))

            # The index state_idx_in_C_block corresponds to the index in the full state vector
            # for trends and current cycles.
            full_state_idx = state_idx_in_C_block 
            
            # Add term to C_comp
            C_comp = C_comp.at[obs_idx, full_state_idx].add(sign * param_value)


    # 4. H_comp (Observation Noise Covariance for Simulation)
    # Assuming zero unless specified otherwise in config and sampled.
    # For this model, observation noise is assumed to be part of the state noise R_aug.
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)


    # --- Initial State Mean and Covariance for Simulation ---
    # These come directly from the extracted smoother_params.
    init_x_sim = init_x_mean # k_states vector mean
    init_P_sim = init_P_cov # k_states x k_states cov matrix


    # Return matrices and initial state for simulation
    return {
        "T_comp": T_comp, # k_states x k_states
        "R_aug": R_aug,   # k_states x n_shocks_sim (for simulate_state_space)
        "C_comp": C_comp, # k_endog x k_states
        "H_comp": H_comp, # k_endog x k_endog (zero)
        "init_x_sim": init_x_sim, # k_states
        "init_P_sim": init_P_sim  # k_states x k_states
    }


# Step 3: Implement Online Quantile Estimator (using static methods as before)
# Class definition remains the same, only static methods are used.
class OnlineQuantileEstimator:
    """
    Helper with static methods for updating a fixed-size buffer and computing quantiles.
    Used to manage state (buffer, count) within JAX scans/loops.
    """
    def __init__(self, num_values_to_track: int = 1000, quantiles: List[float] = None):
        self.num_values_to_track = num_values_to_track
        self.quantiles = sorted(quantiles if quantiles is not None else [0.025, 0.5, 0.975])
        # No state is stored in instances; use static methods.

    @staticmethod
    def update_buffer(buffer: jax.Array, count: jax.Array, new_value: jax.Array, num_values_to_track: int):
        """
        Static method to update the fixed-size buffer.
        """
        buffer = buffer.at[count % num_values_to_track].set(new_value)
        count = count + 1
        return buffer, count

    @staticmethod
    def compute_quantiles_from_buffer(buffer: jax.Array, count: jax.Array, quantiles: List[float]):
        """
        Compute quantiles from buffer - COMPLETELY AVOID JAX COMPILATION.
        This runs outside any JAX context to avoid all dynamic indexing issues.
        """
        # Convert to numpy for computation outside JAX
        buffer_np = np.asarray(buffer)
        count_np = int(count)
        
        # Check if we have any data
        if count_np <= 0:
            return jnp.full(len(quantiles), jnp.nan, dtype=_DEFAULT_DTYPE)
        
        # Get finite values
        finite_mask = np.isfinite(buffer_np)
        if not np.any(finite_mask):
            return jnp.full(len(quantiles), jnp.nan, dtype=_DEFAULT_DTYPE)
        
        # Extract finite values
        finite_values = buffer_np[finite_mask]
        
        if len(finite_values) == 0:
            return jnp.full(len(quantiles), jnp.nan, dtype=_DEFAULT_DTYPE)
        
        # Compute quantiles using numpy (outside JAX)
        try:
            quantile_values = np.percentile(finite_values, [q * 100 for q in quantiles])
            return jnp.array(quantile_values, dtype=_DEFAULT_DTYPE)
        except Exception:
            return jnp.full(len(quantiles), jnp.nan, dtype=_DEFAULT_DTYPE)

# Step 6: Single Simulation Path Function
# This function will use the SS matrices from construct_ss_matrices_from_smoother_params.
# This function is already defined in the previous response's smoother_utils.py block.
# Re-include it here for clarity in the full smoother_utils.py file.

# --- Fixed run_single_simulation_path_for_dk Function ---

def run_single_simulation_path_for_dk(
    ss_matrices_sim: Dict[str, jax.Array],
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    key: jax.random.PRNGKey,
    config_data: Dict[str, Any],
    smoother_params_single_draw: Dict[str, jax.Array]  # ADDED: Missing parameter
) -> jax.Array:
    """
    Generates a single simulation path for the Durbin-Koopman smoother.
    FIXED: Added missing smoother_params_single_draw parameter.
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
    
    k_trends = Sigma_trends_full.shape[0]
    k_stationary = Sigma_cycles.shape[0]
    Q_comp_filter = jnp.zeros((n_state, n_state), dtype=_DEFAULT_DTYPE)
    Q_comp_filter = Q_comp_filter.at[:k_trends, :k_trends].set(Sigma_trends_full)
    Q_comp_filter = Q_comp_filter.at[k_trends:k_trends+k_stationary, k_trends:k_trends+k_stationary].set(Sigma_cycles)
    Q_comp_filter = (Q_comp_filter + Q_comp_filter.T) / 2.0
    Q_comp_filter = Q_comp_filter + _MODEL_JITTER * jnp.eye(n_state, dtype=_DEFAULT_DTYPE)

    # R_for_filter is the Cholesky of Q_comp_filter
    try:
        R_for_filter = jax.scipy.linalg.cholesky(Q_comp_filter, lower=True)
    except Exception:
        print("Warning: Cholesky of Q_comp_filter failed. Using diagonal sqrt.")
        R_for_filter = jnp.diag(jnp.sqrt(jnp.diag(Q_comp_filter)))

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


# Updated JITted wrapper function
#@jax.jit(static_argnames=['static_config_data'])
def jit_run_single_simulation_path_for_dk_wrapper(
    smoother_params_single_draw: Dict[str, jax.Array],
    original_ys_dense: jax.Array,
    x_smooth_original_dense: jax.Array,
    key: jax.random.PRNGKey,
    static_config_data: Dict[str, Any]
):
    """
    FIXED: JIT wrapper that properly constructs SS matrices inside JIT scope.
    """
    # Construct state-space matrices for this draw inside the JIT scope
    ss_matrices_sim = construct_ss_matrices_from_smoother_params(
        smoother_params_single_draw, static_config_data
    )

    # Run the simulation path using the constructed matrices
    simulated_states_path = run_single_simulation_path_for_dk(
        ss_matrices_sim,
        original_ys_dense,
        x_smooth_original_dense,
        key,
        static_config_data,
        smoother_params_single_draw  # FIXED: Pass the missing parameter
    )
    return simulated_states_path

# Update the JIT compilation for run_single_simulation_path_for_dk
# It needs static_config_data as a static argument
# It also needs x_smooth_original_dense and smoother_params_single_draw
# to be passed dynamically *into* the JIT function.
# The main function in estimate_bvar_with_dls_priors.py will call this wrapper.