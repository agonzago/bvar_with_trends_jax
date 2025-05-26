# --- simulate_bvar_jax.py (Corrected) ---
# JAX-based function to simulate data for the BVAR with trends model.

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Tuple, List, Dict, Any

# Assume necessary imports from stationary_prior_jax_simplified for matrix building
# Need create_companion_matrix_jax for building the true T matrix
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax

# Assume Kalman_filter_jax is available for the simulation utility
from utils.Kalman_filter_jax import simulate_state_space

# Import parsing function from var_ss_model
from core.var_ss_model import _parse_equation_jax # Needed to build C

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64 # Define it here too, although the calling script defines it

def simulate_bvar_with_trends_jax(
    key: jax.random.PRNGKey,
    T_sim: int, # Number of time steps
    config_data: Dict[str, Any], # Relevant parts of config data (dimensions, names, equations, initial_conditions)
    true_phi_list: List[jax.Array], # True VAR(p) coefficient matrices (list of p, k_stat x k_stat)
    true_sigma_cycles: jax.Array, # True stationary cycle shock covariance (k_stat x k_stat)
    true_sigma_trends_sim: jax.Array, # True trend shock covariance for sim (diagonal, n_trend_shocks x n_trend_shocks)
    true_measurement_params: Dict[str, jax.Array], # True measurement parameter values (dict of scalars)
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Simulates data from a BVAR with trends state space model using true parameters.

    Args:
        key: JAX random key.
        T_sim: Number of time steps to simulate.
        config_data: Dictionary containing parsed YAML configuration details needed for structure.
        true_phi_list: True VAR(p) coefficient matrices.
        true_sigma_cycles: True stationary cycle shock covariance.
        true_sigma_trends_sim: True trend shock covariance for simulation (diagonal).
        true_measurement_params: True measurement parameter values.

    Returns:
        Tuple: (y_simulated, true_states, true_cycles, true_trends)
        y_simulated: Simulated observable data (T_sim, k_endog).
        true_states: True latent state path (T_sim, k_states).
        true_cycles: True stationary cycle path (T_sim, k_stationary).
        true_trends: True trend path (T_sim, k_trends).
    """
    # --- Derive Dimensions and Mappings from Config ---
    p = config_data['var_order']
    observable_names = config_data['variables']['observable_names']
    trend_var_names = config_data['variables']['trend_names']
    stationary_var_names = config_data['variables']['stationary_var_names']
    # Use pre-parsed structures from config_data for building matrices
    model_eqs_parsed = config_data['model_equations_parsed']
    initial_conds_parsed = config_data['initial_conditions_parsed']
    trend_names_with_shocks = config_data['trend_names_with_shocks'] # From config_data parsing
    measurement_params_config = config_data.get('parameters', {}).get('measurement', []) # For names

    k_endog = len(observable_names)
    k_trends = len(trend_var_names)
    k_stationary = len(stationary_var_names)
    k_states = k_trends + k_stationary * p # Total state dimension
    n_trend_shocks = len(trend_names_with_shocks) # Number of trends with shocks in config
    n_shocks_sim = n_trend_shocks + k_stationary # Number of shocks in simulation R_aug


    # Create mappings for state, observables, and shocks (consistent with model)
    trend_state_indices = {name: i for i, name in enumerate(trend_var_names)}
    stationary_state_indices = {name: k_trends + i for i, name in enumerate(stationary_var_names)}
    # Full state indices including lags for p > 1
    state_indices_full = {name: i for i, name in enumerate(trend_var_names)} # Trends 0 to k_trends-1
    for lag in range(p):
         for i, name in enumerate(stationary_var_names):
              if lag == 0: # Current cycle
                   state_indices_full[name] = k_trends + i
              else: # Lagged cycles
                   lagged_state_name = f"{name}_t_minus_{lag}" # Construct name like in state_names property
                   lagged_state_idx = k_trends + lag * k_stationary + i
                   state_indices_full[lagged_state_name] = lagged_state_idx


    # --- Reconstruct True State-Space Matrices from True Parameters ---

    # 1. True T matrix (Companion matrix structure for VAR part + Identity for Trends)
    T_true = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_true = T_true.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE)) # Trends block
    # AR Companion block using true_phi_list
    T_true = T_true.at[k_trends:k_states, k_trends:k_states].set(create_companion_matrix_jax(true_phi_list, p, k_stationary))


    # 2. True R_aug matrix (Shock Impact Matrix for simulate_state_space)
    # R_aug maps (n_trend_shocks + k_stationary) standard normal shocks to state dynamics.
    # R_aug = [L_trends_sim, 0; 0, L_cycles_true], shaped (k_states, n_trend_shocks + k_stationary)
    L_trends_sim_true = jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(true_sigma_trends_sim), 1e-12))) # Cholesky of diagonal Sigma_trends_sim
    L_cycles_true = jnp.linalg.cholesky(true_sigma_cycles + 1e-12 * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)) # Cholesky of Sigma_cycles (add jitter for stability)

    R_aug_true = jnp.zeros((k_states, n_shocks_sim), dtype=_DEFAULT_DTYPE)
    # Trend shock impact: L_trends_sim_true maps first n_trend_shocks std normals to trend states *with shocks*
    trend_state_indices_with_shocks = [trend_state_indices[name] for name in trend_names_with_shocks]
    # Note: L_trends_sim_true is diag(stds). R_aug structure maps shocks to state indices.
    # R_aug[state_idx_with_shock, shock_idx_0_to_n_trend_shocks-1] = std_dev_for_that_shock
    for i, trend_name_with_shock in enumerate(trend_names_with_shocks):
        trend_state_idx = trend_state_indices[trend_name_with_shock]
        R_aug_true = R_aug_true.at[trend_state_idx, i].set(L_trends_sim_true[i, i]) # Use diagonal element

    # Stationary shock impact: L_cycles_true maps next k_stationary std normals to current stationary states
    R_aug_true = R_aug_true.at[k_trends:k_trends+k_stationary, n_trend_shocks:].set(L_cycles_true)


    # 3. True C matrix
    # C has shape (k_endog, k_states). Needs parsed equations and true measurement params.
    C_true = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    # State indices for C matrix (Trends + Current Stationary)
    state_indices_for_C = {name: i for i, name in enumerate(trend_var_names)} # Trends
    state_indices_for_C.update({name: k_trends + i for i, name in enumerate(stationary_var_names)}) # Current Stationary

    # Populate C matrix using the parsed structure and true measurement parameters
    # model_eqs_parsed is list of (obs_idx, parsed_terms)
    measurement_params_config = config_data.get('parameters', {}).get('measurement', []) # For names
    measurement_param_names_list = [p['name'] for p in measurement_params_config]
    for obs_idx, parsed_terms in model_eqs_parsed:
        for param_name, state_name_in_eq, sign in parsed_terms:
            state_idx = state_indices_for_C[state_name_in_eq] # Map the name in the equation to its index
            if param_name is None:
                C_true = C_true.at[obs_idx, state_idx].set(sign * 1.0)
            else:
                param_value = true_measurement_params[param_name]
                C_true = C_true.at[obs_idx, state_idx].set(sign * param_value)


    # 4. True H matrix (zero)
    H_true = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)

    # --- True Initial State Distribution ---
    # Mean (init_x_true): Fixed values from parsed initial_conditions (means)
    init_x_true = jnp.zeros(k_states, dtype=_DEFAULT_DTYPE)
    initial_state_parsed = config_data['initial_conditions_parsed']
    # Full state indices including lags for p > 1 (same as state_indices_full)
    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             init_x_true = init_x_true.at[state_indices_full[state_name_parsed]].set(parsed_config['mean'])

    # Covariance (init_P_true): Diagonal matrix from parsed initial_conditions (variances), repeated for lags
    init_P_true_diag_values = jnp.zeros(k_states, dtype=_DEFAULT_DTYPE)
    # Populate diagonal P0 based on variances from parsed initial_conditions: states
    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             state_idx = state_indices_full[state_name_parsed]
             init_P_true_diag_values = init_P_true_diag_values.at[state_idx].set(parsed_config['var'])

    # For lagged stationary states (p>1): set variances to the same as current state variances.
    if p > 1:
         for lag in range(1, p):
             for i, name in enumerate(stationary_var_names):
                  current_state_name = name # Name in parsed initial conditions
                  lagged_state_name = f"{name}_t_minus_{lag}" # Name in full state vector
                  if current_state_name in initial_state_parsed and lagged_state_name in state_indices_full:
                       current_var = initial_state_parsed[current_state_name]['var']
                       lagged_idx = state_indices_full[lagged_state_name]
                       init_P_true_diag_values = init_P_true_diag_values.at[lagged_idx].set(current_var)


    init_P_true = jnp.diag(init_P_true_diag_values) # Shape (k_states, k_states)


    # --- Simulate State Space Model ---
    key_sim_model, key_sim_init = random.split(key)

    # simulate_state_space takes P_aug (T), R_aug (shock impact), Omega (C), H_obs, init_x, init_P, key, num_steps
    true_states_sim, y_simulated = simulate_state_space(
        T_true, # P_aug
        R_aug_true, # R_aug
        C_true, # Omega
        H_true, # H_obs (zero matrix)
        init_x_true,
        init_P_true,
        key_sim_model, # Use a key for the simulation process
        T_sim
    )

    # Extract true cycles and trends from the simulated true_states
    true_trends = true_states_sim[:, :k_trends]
    # Stationary parts are k_trends to k_states-1.
    # For p=1, true_cycles is k_trends to k_trends+k_stationary-1
    # For p>1, true_cycles are k_trends to k_trends+k_stationary-1.
    # We need the current cycles (t=t) which are the first k_stationary elements
    # in the stationary block.
    true_cycles = true_states_sim[:, k_trends:k_trends+k_stationary]


    return y_simulated, true_states_sim, true_cycles, true_trends


