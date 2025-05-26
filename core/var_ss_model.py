# --- var_ss_model.py (Corrected build_state_space_matrices_jit return) ---

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from typing import Dict, Tuple, List, Any, Sequence, Optional

# Assuming utils directory is in the Python path and contains the necessary files
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax # Removed check_stationarity_jax
from utils.Kalman_filter_jax import KalmanFilter # Use the standard KF for likelihood

import yaml # Added for load_config_and_prepare_jax_static_args

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Add a small jitter for numerical stability
_MODEL_JITTER = 1e-8


# Helper function to get static off-diagonal indices (Moved here)
def _get_off_diagonal_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates row and column indices for off-diagonal elements of an n x n matrix.
    These indices are static once n is known.
    """
    if n <= 1: # No off-diagonal elements for 0x0 or 1x1 matrices
        return jnp.empty((0,), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)

    rows, cols = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
    mask = jnp.eye(n, dtype=bool)
    off_diag_rows = rows[~mask]
    off_diag_cols = cols[~mask]
    return off_diag_rows, off_diag_cols


# Helper function to parse model equations (JAX compatible version of _parse_equation)
# This will be used internally during C matrix construction
def _parse_equation_jax(
    equation: str,
    trend_names: List[str],
    stationary_var_names: List[str],
    measurement_param_names: List[str],
    dtype=_DEFAULT_DTYPE
) -> List[Tuple[Optional[str], str, float]]:
    """
    Parse a measurement equation string into components with signs.
    Returns a list of tuples (param_name, state_name, sign).
    param_name is None for direct state terms.
    This function is executed *outside* the JAX graph (e.g., during model setup)
    as it relies on string parsing. The result is used to build the C matrix inside the model.
    """
    # Pre-process the equation string
    equation = equation.replace(' - ', ' + -')
    if equation.startswith('-'):
        equation = '-' + equation[1:].strip()

    # Split terms by '+'
    terms_str = [t.strip() for t in equation.split('+')]

    parsed_terms = []
    for term_str in terms_str:
        sign = 1.0
        if term_str.startswith('-'):
            sign = -1.0
            term_str = term_str[1:].strip()

        if '*' in term_str:
            parts = [p.strip() for p in term_str.split('*')]
            if len(parts) != 2:
                # In JAX model, we cannot raise Python exceptions easily within the traced code.
                # We'll assume valid equations are passed based on config validation.
                # If this was pure JAX, would need error handling via lax.cond or similar.
                # For setup, we rely on config validation.
                raise ValueError(f"Invalid term '{term_str}': Must have exactly one '*' operator")

            param, state = None, None
            if parts[0] in measurement_param_names:
                param, state = parts[0], parts[1]
            elif parts[1] in measurement_param_names:
                param, state = parts[1], parts[0]
            else:
                 raise ValueError(
                        f"Term '{term_str}' contains no valid parameter. "
                        f"Valid parameters are: {measurement_param_names}"
                    )

            if (state not in trend_names and
                state not in stationary_var_names):
                 raise ValueError(
                        f"Invalid state variable '{state}'. " # Simplified error message within JAX context
                        # f"Invalid state variable '{state}' in term '{term_str}'. "
                        # f"Valid states are: {trend_names + stationary_var_names}"
                    )

            parsed_terms.append((param, state, sign))
        else:
            if (term_str not in trend_names and
                term_str not in stationary_var_names):
                 raise ValueError(
                        f"Invalid state variable '{term_str}'. " # Simplified error message within JAX context
                        # f"Invalid state variable '{term_str}'. "
                        # f"Valid states are: {trend_names + stationary_var_names}"
                    )
            parsed_terms.append((None, term_str, sign))

    return parsed_terms


# Helper function to parse config initial states for NumPyro
# Keep this function here or import it if it lives in config.py
def parse_initial_state_config(initial_conditions_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Parse initial state configuration (means and variances) from config.yaml.
    Replicates BVARConfig._parse_initial_state logic.
    """
    parsed_states = {}
    raw_states_config = initial_conditions_config.get('states', {})

    for state_name, state_config in raw_states_config.items():
        parsed = {}
        if isinstance(state_config, (int, float)):
            parsed = {"mean": float(state_config), "var": 1.0}
        elif isinstance(state_config, dict) and "mean" in state_config:
            parsed = state_config
        elif isinstance(state_config, str):
            parts = state_config.split()
            temp_result = {}
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    key = parts[i].strip().rstrip(':')
                    value = float(parts[i + 1])
                    temp_result[key] = value
            if "mean" in temp_result:
                if "var" not in temp_result:
                    temp_result["var"] = 1.0
                parsed = temp_result

        if "mean" not in parsed:
            # This case should ideally be caught by config validation
            # In JAX, better to return NaN or large value if parsing fails unexpectedly
            # For this helper used in setup, raising an error is acceptable.
            raise ValueError(f"Could not parse mean for initial state '{state_name}'.")
        if "var" not in parsed:
             # This case should ideally be caught by config validation
             raise ValueError(f"Could not parse variance for initial state '{state_name}'.")

        # Ensure var is non-negative, clip at a small value
        var_val = jnp.maximum(jnp.array(parsed["var"], dtype=_DEFAULT_DTYPE), 1e-12) # Ensure positive variance
        parsed_states[state_name] = {"mean": jnp.array(parsed["mean"], dtype=_DEFAULT_DTYPE), "var": var_val}

    return parsed_states


#@jax.jit(static_argnames=["static_config_data"])
def build_state_space_matrices_jit(
    params_dict: Dict[str, jax.Array], 
    static_config_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs state-space matrices and related terms from sampled parameters
    and static configuration data. JIT-compiled for performance.

    Args:
        params_dict: Dictionary of sampled JAX arrays for parameters.
        static_config_data: Dictionary from load_config_and_prepare_jax_static_args.

    Returns:
        A dictionary containing the constructed state-space matrices and terms.
        Includes derived terms like phi_list, Sigma_cycles, Sigma_trends_full for deterministic sites.
    """
    # --- Extract Static Configuration ---
    k_endog = static_config_data['k_endog']
    k_trends = static_config_data['k_trends']
    k_stationary = static_config_data['k_stationary']
    p = static_config_data['var_order']
    k_states = static_config_data['k_states']
    
    static_off_diag_rows, static_off_diag_cols = static_config_data['static_off_diag_indices']
    num_off_diag = static_config_data['num_off_diag'] # Number of off-diagonal elements
    
    n_trend_shocks = static_config_data['n_trend_shocks']
    trend_names_with_shocks_tuple = static_config_data['trend_names_with_shocks'] 
    
    trend_var_names_tuple = static_config_data['trend_var_names']
    stationary_var_names_tuple = static_config_data['stationary_var_names']
    
    parsed_model_eqs_jax_detailed = static_config_data['parsed_model_eqs_jax_detailed']
    measurement_param_names_tuple = static_config_data['measurement_param_names_tuple']

    init_x_means_flat = static_config_data['init_x_means_flat']
    init_P_diag_flat = static_config_data['init_P_diag_flat']

    # --- Extract Dynamic Parameters (sampled values) ---
    # These parameter names match the NumPyro sample sites.
    A_diag = params_dict['A_diag'] 
    A_offdiag_flat = params_dict.get('A_offdiag') # Can be None if num_off_diag is 0
    
    _stationary_variances_values = []
    for name in stationary_var_names_tuple: 
        param_name = f'stationary_var_{name}'
        # Use .get with a default to avoid KeyError if a parameter is unexpectedly missing
        _stationary_variances_values.append(params_dict.get(param_name, jnp.array(1.0, dtype=_DEFAULT_DTYPE)))

    # Ensure the list has the correct length even if some parameters were missing
    if len(_stationary_variances_values) != k_stationary:
         print(f"Warning: Expected {k_stationary} stationary variances, found {len(_stationary_variances_values)}. Filling with defaults.")
         _stationary_variances_values.extend([jnp.array(1.0, dtype=_DEFAULT_DTYPE)] * (k_stationary - len(_stationary_variances_values)))

    if k_stationary > 0:
        stationary_variances_array = jnp.stack(_stationary_variances_values[:k_stationary], axis=-1) # Stack only the first k_stationary
    else:
        stationary_variances_array = jnp.array([], dtype=_DEFAULT_DTYPE)


    # stationary_chol is Cholesky of Correlation matrix (sampled if k_stationary > 1)
    if k_stationary > 1:
        stationary_chol = params_dict.get('stationary_chol', jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE))
        # Validate shape of sampled stationary_chol if found
        if stationary_chol.shape != (k_stationary, k_stationary):
             print(f"Warning: Sampled stationary_chol has incorrect shape {stationary_chol.shape}. Expected ({k_stationary}, {k_stationary}). Using identity.")
             stationary_chol = jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)
    elif k_stationary == 1: 
        stationary_chol = jnp.eye(1, dtype=_DEFAULT_DTYPE) 
    else: # k_stationary == 0
        stationary_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    _trend_variances_values = []
    for name in trend_names_with_shocks_tuple: 
        param_name = f'trend_var_{name}'
        _trend_variances_values.append(params_dict.get(param_name, jnp.array(1.0, dtype=_DEFAULT_DTYPE)))

    # Ensure the list has the correct length
    if len(_trend_variances_values) != n_trend_shocks:
         print(f"Warning: Expected {n_trend_shocks} trend variances, found {len(_trend_variances_values)}. Filling with defaults.")
         _trend_variances_values.extend([jnp.array(1.0, dtype=_DEFAULT_DTYPE)] * (n_trend_shocks - len(_trend_variances_values)))

    if n_trend_shocks > 0:
        trend_variances_array = jnp.stack(_trend_variances_values[:n_trend_shocks], axis=-1)
    else:
        trend_variances_array = jnp.array([], dtype=_DEFAULT_DTYPE)


    _measurement_params_sampled_values = []
    for name in measurement_param_names_tuple: 
        param_name = name
        _measurement_params_sampled_values.append(params_dict.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE))) # Default to 0 for measurement params

    if len(_measurement_params_sampled_values) != len(measurement_param_names_tuple):
         print(f"Warning: Expected {len(measurement_param_names_tuple)} measurement params, found {len(_measurement_params_sampled_values)}. Filling with defaults.")
         _measurement_params_sampled_values.extend([jnp.array(0.0, dtype=_DEFAULT_DTYPE)] * (len(measurement_param_names_tuple) - len(_measurement_params_sampled_values)))

    if len(measurement_param_names_tuple) > 0:
         measurement_params_sampled_array = jnp.array(_measurement_params_sampled_values[:len(measurement_param_names_tuple)], dtype=_DEFAULT_DTYPE)
    else:
         measurement_params_sampled_array = jnp.array([], dtype=_DEFAULT_DTYPE)


    # --- Reconstruct A_draws (for phi_list calculation) ---
    A_draws = jnp.zeros((p, k_stationary, k_stationary), dtype=_DEFAULT_DTYPE)
    if k_stationary > 0:
        # A_diag is expected to be (p, k_stationary) from the sampler
        if 'A_diag' in params_dict and params_dict['A_diag'].shape == (p, k_stationary):
             A_draws = A_draws.at[:, jnp.arange(k_stationary), jnp.arange(k_stationary)].set(params_dict['A_diag'])
        else:
             print(f"Warning: 'A_diag' missing or incorrect shape {params_dict.get('A_diag', 'N/A')} in params_dict. A_draws diagonals will be zero.")

        if num_off_diag > 0 and A_offdiag_flat is not None:
            # A_offdiag is expected to be (p, num_off_diag) from the sampler
             if 'A_offdiag' in params_dict and params_dict['A_offdiag'].shape == (p, num_off_diag):
                 A_draws = A_draws.at[:, static_off_diag_rows, static_off_diag_cols].set(params_dict['A_offdiag'])
             else:
                 print(f"Warning: 'A_offdiag' missing or incorrect shape {params_dict.get('A_offdiag', 'N/A')} in params_dict. A_draws off-diagonals will be zero.")


    # --- Construct Sigma_cycles (from variances and correlation Cholesky) ---
    if k_stationary > 0:
        stationary_variances_safe = jnp.maximum(stationary_variances_array, _MODEL_JITTER)
        
        if k_stationary == 1:
            Sigma_cycles = jnp.diag(stationary_variances_safe)
            # L_cycles = jnp.diag(jnp.sqrt(stationary_variances_safe)) # Not needed for SS construction here
        else:
            std_devs = jnp.sqrt(stationary_variances_safe)
            D_matrix = jnp.diag(std_devs)
            
            # Ensure stationary_chol is the correct shape if fallback was used
            current_stationary_chol = jnp.where(
                 stationary_chol.shape == (k_stationary, k_stationary),
                 stationary_chol,
                 jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)
            )

            Sigma_cycles = D_matrix @ current_stationary_chol @ current_stationary_chol.T @ D_matrix
            
            # Ensure symmetry and positive definiteness
            Sigma_cycles = (Sigma_cycles + Sigma_cycles.T) / 2.0
            Sigma_cycles = Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)

            # L_cycles_comp used later might be needed here if not recomputed
            # L_cycles_comp = jsl.cholesky(Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE), lower=True)


    else:
        Sigma_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        # L_cycles_comp = jnp.empty((0,0), dtype=_DEFAULT_DTYPE) # Not needed for SS construction here
        
    # --- Construct Sigma_trends_full (diagonal from trend variances with shocks) ---
    Sigma_trends_full = jnp.zeros((k_trends, k_trends), dtype=_DEFAULT_DTYPE)
    if n_trend_shocks > 0:
        trend_var_names_list = list(trend_var_names_tuple) # Convert to list for .index()
        for i in range(n_trend_shocks):
            shocked_trend_name = trend_names_with_shocks_tuple[i]
            try:
                trend_state_idx = trend_var_names_list.index(shocked_trend_name)
                Sigma_trends_full = Sigma_trends_full.at[trend_state_idx, trend_state_idx].set(jnp.maximum(trend_variances_array[i], _MODEL_JITTER))
            except ValueError:
                 print(f"Warning: Shocked trend name '{shocked_trend_name}' not found in trend_var_names_tuple during Sigma_trends_full build.")
                 pass
                 
    Sigma_trends_full = (Sigma_trends_full + Sigma_trends_full.T) / 2.0


    # --- Transform A to phi_list ---
    if k_stationary > 0:
        A_list_for_phi = [A_draws[i] for i in range(p)] 
        # Ensure A_list_for_phi elements are JAX arrays
        A_list_for_phi = [jnp.asarray(A, dtype=_DEFAULT_DTYPE) for A in A_list_for_phi]
        phi_list, _ = make_stationary_var_transformation_jax(Sigma_cycles, A_list_for_phi, k_stationary, p)
        # Ensure phi_list elements are JAX arrays
        phi_list = [jnp.asarray(phi, dtype=_DEFAULT_DTYPE) for phi in phi_list]
    else:
        phi_list = []

    # --- Construct T_comp (from trends + companion matrix from phi_list) ---
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE))
    if k_stationary > 0:
        companion_matrix = create_companion_matrix_jax(phi_list, p, k_stationary)
        T_comp = T_comp.at[k_trends:, k_trends:].set(companion_matrix)

    # --- Construct R_comp (Shock Impact Matrix for Likelihood) ---
    # R_comp @ R_comp.T = Q_comp = block_diag(Sigma_trends_full, Sigma_cycles_at_current_state_block)
    # This R_comp is used by the Kalman Filter for likelihood calculation.
    # The dimension of this R_comp is (k_states, k_states).
    # The R_aug used by simulate_state_space is (k_states, num_shocks).
    # Let's calculate Q_comp and then get its Cholesky for R_comp.
    Q_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    Q_comp = Q_comp.at[:k_trends, :k_trends].set(Sigma_trends_full)
    Q_comp = Q_comp.at[k_trends:k_trends+k_stationary, k_trends:k_trends+k_stationary].set(Sigma_cycles)
    Q_comp = (Q_comp + Q_comp.T) / 2.0
    Q_comp = Q_comp + _MODEL_JITTER * jnp.eye(k_states, dtype=_DEFAULT_DTYPE)

    # Calculate R_comp as the Cholesky of Q_comp
    try:
        R_comp = jsl.cholesky(Q_comp, lower=True)
    except Exception:
        print("Warning: Cholesky of Q_comp failed in build_state_space_matrices_jit. Using diagonal sqrt.")
        R_comp = jnp.diag(jnp.sqrt(jnp.diag(Q_comp))) # Fallback to diagonal Cholesky


    # --- Construct C_comp (Measurement Matrix) ---
    # C_comp maps [trends, cycle_t, cycle_t-1, ..., cycle_t-p+1] to observables.
    # Use the detailed parsed structure and dynamic measurement parameter values.
    # The parsed_model_eqs_jax_detailed contains (obs_idx, Tuple[term_type, state_idx_in_C_block, param_idx, sign]).
    # state_idx_in_C_block is the index in the [trends | current cycles] block.
    # We need to map this index to the correct index in the *full* state vector (k_states).
    # Indices 0 to k_trends-1 in the C-block map to indices 0 to k_trends-1 in the full vector (trends).
    # Indices k_trends to k_trends+k_stationary-1 in the C-block map to indices k_trends to k_trends+k_stationary-1 in the full vector (current cycles).
    # So, the state_idx_in_C_block directly corresponds to the index in the full state vector for the relevant components.

    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    measurement_param_names_list = list(measurement_param_names_tuple) # For getting value from dict

    for obs_idx, terms_for_obs in parsed_model_eqs_jax_detailed: 
        for term_type, state_idx_in_C_block, param_idx_if_any, sign in terms_for_obs: 
            is_param_term = (term_type == 1)
            
            param_value = jnp.array(1.0, dtype=_DEFAULT_DTYPE) # Default to 1.0 for direct terms
            if is_param_term:
                # Get parameter name from index
                param_name = measurement_param_names_list[param_idx_if_any]
                # Get parameter value from dynamic measurement_params_dict (which is not in params_dict here!)
                # This is wrong. The measurement parameters are sampled and ARE in params_dict.
                # We need to get the value from params_dict here.
                param_value = params_dict.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE)) # Get from params_dict


            # The index state_idx_in_C_block corresponds to the index in the full state vector
            # for trends and current cycles.
            full_state_idx = state_idx_in_C_block 
            
            # Add term to C_comp
            C_comp = C_comp.at[obs_idx, full_state_idx].add(sign * param_value)


    # 4. H_comp (Observation Noise Covariance)
    # Assuming zero unless specified otherwise in config and sampled.
    # For this model, observation noise is assumed to be part of the state noise R.
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)


    # --- Construct init_x_comp (Initial State Mean Vector) ---
    # Uses the flat array from config parsing.
    init_x_comp = init_x_means_flat

    # --- Construct init_P_comp (Initial State Covariance Matrix) ---
    # Uses the diagonal flat array from config parsing.
    init_P_comp = jnp.diag(init_P_diag_flat) 
    init_P_comp = (init_P_comp + init_P_comp.T) / 2.0 
    init_P_comp = init_P_comp + _MODEL_JITTER * jnp.eye(k_states, dtype=_DEFAULT_DTYPE)
    
    # Return all constructed matrices and derived terms
    return {
        "T_comp": T_comp, # k_states x k_states
        "R_comp": R_comp, # k_states x k_states (Cholesky of Q for KF likelihood)
        "C_comp": C_comp, # k_endog x k_states
        "H_comp": H_comp, # k_endog x k_endog (zero)
        "init_x_comp": init_x_comp, # k_states mean vector
        "init_P_comp": init_P_comp, # k_states x k_states covariance matrix

        # Include parameters/derived terms needed for deterministic sites AND smoother
        # These are the parameters that define the SS model for a draw.
        "phi_list": phi_list,
        "Sigma_cycles": Sigma_cycles,
        "Sigma_trends_full": Sigma_trends_full,
        "init_x_mean_vec": init_x_means_flat, # Pass the flat mean vector separately
        "init_P_cov_matrix": init_P_comp, # Pass the covariance matrix separately
        # A_draws is also useful to return for deterministic sites if needed
        "A_draws": A_draws 
    }


def load_config_and_prepare_jax_static_args(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration, parses it, and prepares a dictionary of static
    arguments suitable for JAX-based models, particularly for the NumPyro BVAR model.
    This function is JIT-compatible in principle IF the config path were static,
    but typically called in Python setup phase.
    """
    print(f"Loading and parsing config from {config_path}")
    # Step 1 & 2: Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
         print(f"Error: Config file not found at {config_path}")
         raise
    except yaml.YAMLError as e:
         print(f"Error parsing YAML config file: {e}")
         raise


    # Step 3: Parse variables
    variables_config = config.get('variables', {})
    observable_names = tuple(variables_config.get('observables', []))
    trend_var_names = tuple(variables_config.get('trends', []))
    stationary_var_names = tuple(variables_config.get('stationary', []))

    # Step 4: Parse var_order
    var_order = int(config.get('var_order', 1))

    # Step 5: Calculate dimensions
    k_endog = len(observable_names)
    k_trends = len(trend_var_names)
    k_stationary = len(stationary_var_names)
    p = var_order
    k_states = k_trends + k_stationary * p

    # Step 6: Parse initial_conditions
    # Helper to create the full list of state names in order (trends + stationary lags)
    full_state_names_list = list(trend_var_names)
    for i in range(p):
        for stat_var in stationary_var_names:
            if i == 0: # Current cycle
                full_state_names_list.append(stat_var)
            else: # Lagged cycles
                lagged_state_name = f"{stat_var}_t_minus_{i}" 
                full_state_names_list.append(lagged_state_name)
    full_state_names_tuple = tuple(full_state_names_list)
    
    # Use existing parse_initial_state_config helper
    raw_initial_conds = config.get('initial_conditions', {})
    parsed_initial_conds_from_yaml = parse_initial_state_config(raw_initial_conds)

    # Create flat initial condition arrays based on the full state names order
    init_x_means_flat_list = []
    init_P_diag_flat_list = []

    for state_name in full_state_names_tuple:
        base_name_for_lag = state_name
        if "_t_minus_" in state_name:
            base_name_for_lag = state_name.split("_t_minus_")[0]

        if state_name in parsed_initial_conds_from_yaml:
            init_x_means_flat_list.append(parsed_initial_conds_from_yaml[state_name]['mean'])
            init_P_diag_flat_list.append(parsed_initial_conds_from_yaml[state_name]['var'])
        elif base_name_for_lag in parsed_initial_conds_from_yaml and base_name_for_lag != state_name : # It's a lagged state, use its current value's spec
            # For lagged states, default mean to 0 or the base state's initial mean? Config is often only for t=0.
            # Let's stick to the config spec if available, otherwise 0.0 mean and base state's var.
            init_mean = parsed_initial_conds_from_yaml[base_name_for_lag].get('mean', 0.0) # Use get with default 0.0 for lagged mean
            init_var = parsed_initial_conds_from_yaml[base_name_for_lag].get('var', 1.0)
            init_x_means_flat_list.append(init_mean)
            init_P_diag_flat_list.append(init_var)
        else: # Default if not specified at all for this state name or its base name
            print(f"Warning: Initial condition not found for state '{state_name}'. Using defaults (mean=0.0, var=1.0).")
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)
            
    init_x_means_flat = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    init_P_diag_flat = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)

    # For the initial_conditions_parsed key for the model, use the dict structure
    initial_conds_parsed_for_model = parsed_initial_conds_from_yaml


    # Step 7: Parse model_equations
    model_equations_config = config.get('model_equations', {})
    
    # Correctly access the measurement parameters list
    parameters_section = config.get('parameters', {})
    measurement_params_config_raw = parameters_section.get('measurement', []) # Corrected access
    
    # Ensure measurement_params_config_raw is actually a list before proceeding
    if not isinstance(measurement_params_config_raw, list):
         print(f"Warning: 'parameters' -> 'measurement' section in config is not a list ({type(measurement_params_config_raw)}). Treating as empty.")
         measurement_params_config_raw = []

    measurement_param_names_list = [p['name'] for p in measurement_params_config_raw if isinstance(p, dict) and 'name' in p] # Ensure item is dict
    measurement_param_names_tuple = tuple(measurement_param_names_list)
    
    # State names relevant for C matrix (trends + current stationary) - these are the columns C maps from
    c_matrix_state_names_list = list(trend_var_names) + list(stationary_var_names)
    state_to_c_col_idx_map = {name: i for i, name in enumerate(c_matrix_state_names_list)}

    # Create the list of (obs_idx, parsed_terms) for the model
    parsed_model_eqs_list_for_model = []
    for obs_idx, obs_name in enumerate(observable_names):
        eq_str = model_equations_config.get(obs_name)
        if eq_str is None:
            # This should be validated upstream, but add robustness
            print(f"Warning: Equation for observable '{obs_name}' not found in model_equations. Skipping parsing for this observable.")
            continue # Skip parsing for this observable
        
        try:
            raw_parsed_terms = _parse_equation_jax(
                eq_str, 
                list(trend_var_names), 
                list(stationary_var_names), 
                list(measurement_param_names_tuple) 
            )
            parsed_model_eqs_list_for_model.append((obs_idx, raw_parsed_terms))
        except ValueError as e:
            print(f"Error parsing equation for observable '{obs_name}': {e}. Skipping parsing for this observable.")
            continue


    # Create the detailed parsed_model_eqs_jax structure (obs_idx, term_type, state_idx_in_C_block, param_idx_if_any, sign)
    # This maps to the [trends | current cycles] block of the state vector.
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names_tuple)}

    parsed_model_eqs_jax_detailed = []
    for obs_idx, obs_name in enumerate(observable_names):
        eq_str = model_equations_config.get(obs_name)
        if eq_str is None: continue # Already handled missing equation above
        
        try:
             raw_parsed_terms = _parse_equation_jax(
                eq_str, list(trend_var_names), list(stationary_var_names), list(measurement_param_names_tuple)
             )
             processed_terms_for_obs = []
             for param_name, state_name_in_eq, sign in raw_parsed_terms:
                 term_type = 0 if param_name is None else 1
                 # Use the state_to_c_col_idx_map to get the index in the [trends | current cycles] block
                 state_index_in_C_block = state_to_c_col_idx_map[state_name_in_eq]
                 param_index_if_any = param_to_idx_map.get(param_name, -1)
                 processed_terms_for_obs.append(
                     (term_type, state_index_in_C_block, param_index_if_any, float(sign))
                 )
             parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))
        except ValueError as e:
             print(f"Error parsing equation for observable '{obs_name}' for detailed structure: {e}. Skipping.")
             continue # Skip if detailed parsing fails


    parsed_model_eqs_jax_detailed_tuple = tuple(parsed_model_eqs_jax_detailed)


    # Step 8: Parse trend_shocks (names)
    trend_shocks_section = config.get('trend_shocks', {}) 
    trend_shocks_config_raw = trend_shocks_section.get('trend_shocks', {}) 
    trend_names_with_shocks_list = []
    # Iterate through all trend names defined in variables, check if they have a shock spec
    for trend_name in trend_var_names:
        shock_spec = trend_shocks_config_raw.get(trend_name)
        if shock_spec and isinstance(shock_spec, dict) and 'distribution' in shock_spec: # Ensure it's a valid shock spec
            trend_names_with_shocks_list.append(trend_name)
    trend_names_with_shocks_tuple = tuple(trend_names_with_shocks_list)
    n_trend_shocks = len(trend_names_with_shocks_tuple)


    # Step 9: Calculate static_off_diag_indices for A matrix
    static_off_diag_rows, static_off_diag_cols = _get_off_diagonal_indices(k_stationary)
    num_off_diag = k_stationary * (k_stationary - 1) if k_stationary > 1 else 0
    static_off_diag_indices = (static_off_diag_rows, static_off_diag_cols)


    # Step 10: Parse stationary_prior (hyperparameters and shock specs)
    stationary_prior_config_raw = config.get('stationary_prior', {})
    hyperparams_raw = stationary_prior_config_raw.get('hyperparameters', {'es': [0.0, 0.0], 'fs': [1.0, 0.5]})
    # Validate es and fs format
    es_list = hyperparams_raw.get('es', [0.0, 0.0])
    fs_list = hyperparams_raw.get('fs', [1.0, 0.5])
    if not (isinstance(es_list, list) and len(es_list) == 2 and all(isinstance(x, (int, float)) for x in es_list)):
         print(f"Warning: 'es' hyperparameter in config is not a list of 2 numbers ({es_list}). Using default [0.0, 0.0].")
         es_list = [0.0, 0.0]
    if not (isinstance(fs_list, list) and len(fs_list) == 2 and all(isinstance(x, (int, float)) for x in fs_list)):
         print(f"Warning: 'fs' hyperparameter in config is not a list of 2 numbers ({fs_list}). Using default [1.0, 0.5].")
         fs_list = [1.0, 0.5]
    
    es_jax = jnp.array(es_list, dtype=_DEFAULT_DTYPE)
    fs_jax = jnp.array(fs_list, dtype=_DEFAULT_DTYPE)

    eta_float = float(stationary_prior_config_raw.get('covariance_prior', {}).get('eta', 1.0))

    stationary_shocks_raw_dict = stationary_prior_config_raw.get('stationary_shocks', {})
    _stationary_shocks_parsed_list = []
    # Iterate through stationary variable names defined in variables section
    for stat_var_name in stationary_var_names: 
        spec = stationary_shocks_raw_dict.get(stat_var_name)
        if not spec or not isinstance(spec, dict) or 'distribution' not in spec: # Ensure valid spec
             # Fallback to default InverseGamma prior if spec is missing/invalid
             print(f"Warning: Missing or invalid shock specification for stationary variable '{stat_var_name}' in config. Using default IG(2, 0.5).")
             dist_name = 'inverse_gamma'
             params = {'alpha': 2.0, 'beta': 0.5}
        else:
            dist_name = spec['distribution'].lower()
            params = spec.get('parameters', {})

        if dist_name == 'inverse_gamma':
            alpha = float(params.get('alpha', 2.0)) 
            beta = float(params.get('beta', 0.5))
            _stationary_shocks_parsed_list.append({'name': stat_var_name, 'dist_idx': 0, 'alpha': alpha, 'beta': beta})
        else:
            print(f"Error: Stationary shock distribution '{dist_name}' for '{stat_var_name}' not supported. Skipping.")
            # Do not add to parsed list if distribution is not supported
            pass
            
    stationary_shocks_parsed_jax = tuple(_stationary_shocks_parsed_list)


    # Step 11: Parse trend_shocks (specs)
    _trend_shocks_parsed_list = []
    # Iterate through trend names identified as having shocks
    for trend_name_with_shock in trend_names_with_shocks_tuple:
        spec = trend_shocks_config_raw.get(trend_name_with_shock) # spec should exist based on how trend_names_with_shocks_tuple was built
        if not spec or not isinstance(spec, dict) or 'distribution' not in spec: # Robustness check
            print(f"Warning: Missing or invalid shock specification for trend variable '{trend_name_with_shock}' in config. This should not happen. Skipping.")
            continue

        dist_name = spec['distribution'].lower()
        params = spec.get('parameters', {})
        if dist_name == 'inverse_gamma': 
            alpha = float(params.get('alpha', 2.0))
            beta = float(params.get('beta', 0.5))
            _trend_shocks_parsed_list.append({'name': trend_name_with_shock, 'dist_idx': 0, 'alpha': alpha, 'beta': beta})
        else:
            print(f"Error: Trend shock distribution '{dist_name}' for '{trend_name_with_shock}' not supported. Skipping.")
            pass

    trend_shocks_parsed_jax = tuple(_trend_shocks_parsed_list)


    # Step 12: Parse parameters (measurement parameters specs)
    _measurement_params_config_parsed_list = []
    # Iterate through the raw list obtained earlier
    for param_spec_raw in measurement_params_config_raw: 
        if not isinstance(param_spec_raw, dict) or 'name' not in param_spec_raw:
             print(f"Warning: Invalid entry in measurement parameters list ({param_spec_raw}). Skipping.")
             continue # Skip invalid entries

        name = param_spec_raw['name']
        prior_info = param_spec_raw.get('prior', {})
        dist_name = prior_info.get('distribution', '').lower()
        params = prior_info.get('parameters', {})
        
        parsed_item = {'name': name}
        if dist_name == 'normal':
            parsed_item['dist_idx'] = 0
            parsed_item['mu'] = float(params.get('mu', 0.0))
            parsed_item['sigma'] = float(params.get('sigma', 1.0))
        elif dist_name == 'half_normal':
            parsed_item['dist_idx'] = 1
            parsed_item['sigma'] = float(params.get('sigma', 1.0))
        else:
            print(f"Error: Measurement parameter prior distribution '{dist_name}' for '{name}' not supported. Skipping.")
            continue # Skip if distribution not supported

        _measurement_params_config_parsed_list.append(parsed_item)

    measurement_params_config_parsed_jax = tuple(_measurement_params_config_parsed_list)

    # Step 13: Return dictionary (static_config_data)
    # This dictionary will be passed as `config_data` to the NumPyro model
    # and also used by the smoother building functions.
    return {
        # Dimensions
        "k_endog": k_endog,
        "k_trends": k_trends,
        "k_stationary": k_stationary,
        "var_order": p,
        "k_states": k_states,
        "num_off_diag": num_off_diag,
        "n_trend_shocks": n_trend_shocks,

        # Names (as tuples for static)
        "observable_names": observable_names,
        "trend_var_names": trend_var_names,
        "stationary_var_names": stationary_var_names,
        "full_state_names_tuple": full_state_names_tuple, # All state names including lags
        "trend_names_with_shocks": trend_names_with_shocks_tuple, # Subset of trend_var_names that have shocks
        "measurement_param_names_tuple": measurement_param_names_tuple, # Names of measurement params

        # Parsed config data structures
        "initial_conditions_parsed": initial_conds_parsed_for_model, # Dict {name: {mean, var}} for t=0 states
        "init_x_means_flat": init_x_means_flat, # Flat array of initial means for all k_states
        "init_P_diag_flat": init_P_diag_flat, # Flat array of initial variances for all k_states (diagonal P0)
        
        "model_equations_parsed": tuple(parsed_model_eqs_list_for_model), # List of (obs_idx, List[Tuple[param_name, state_name, sign]])
        "parsed_model_eqs_jax_detailed": parsed_model_eqs_jax_detailed_tuple, # List of (obs_idx, Tuple[term_type, state_idx_in_C_block, param_idx, sign])
        
        "static_off_diag_indices": static_off_diag_indices, # (rows, cols) for off-diagonals of A
        
        "stationary_hyperparams_es_fs_jax": (es_jax, fs_jax), # JAX arrays for Normal prior on A
        "stationary_cov_prior_eta": eta_float, # LKJCholesky concentration
        
        "stationary_shocks_parsed_spec": stationary_shocks_parsed_jax, # Tuple of dicts for stat shock priors
        "trend_shocks_parsed_spec": trend_shocks_parsed_jax,       # Tuple of dicts for trend shock priors
        "measurement_params_parsed_spec": measurement_params_config_parsed_jax, # Tuple of dicts for measurement param priors

        # Keep raw config sections if needed elsewhere (e.g., printing)
        "raw_config_initial_conds": raw_initial_conds, 
        "raw_config_stationary_prior": stationary_prior_config_raw,
        "raw_config_trend_shocks": trend_shocks_section, 
        "raw_config_measurement_params": measurement_params_config_raw,
        "raw_config_model_eqs_str_dict": model_equations_config,
    }

# Main NumPyro Model
def numpyro_bvar_stationary_model(
    y: jax.Array, 
    config_data: Dict[str, Any], 
    static_valid_obs_idx: jax.Array, 
    static_n_obs_actual: int,
    # The following arguments are technically redundant if config_data is comprehensive,
    # but kept for now to minimize changes to the direct calling signature if used elsewhere.
    # Pass them as tuples for static argument if needed for JIT
    trend_var_names: Tuple[str, ...], 
    stationary_var_names: Tuple[str, ...],  
    observable_names: Tuple[str, ...]       
):
    """
    NumPyro model for BVAR with trends using stationary prior.
    State-space matrix construction is delegated to build_state_space_matrices_jit.
    """
    # --- Dimensions (sourced from config_data for clarity and consistency) ---
    k_endog = config_data['k_endog']
    k_trends = config_data['k_trends']
    k_stationary = config_data['k_stationary']
    p = config_data['var_order']
    k_states = config_data['k_states']
    
    num_off_diag = config_data['num_off_diag']
    # n_trend_shocks = config_data['n_trend_shocks'] # Number of trends with shocks

    # Get parameter names from config_data (used for sampling and collecting into params_for_jit)
    # current_stationary_var_names_cfg = config_data['stationary_var_names'] # Tuple of names
    # trend_names_with_shocks_cfg = config_data['trend_names_with_shocks'] # Tuple of names
    # measurement_param_names_cfg = config_data['measurement_param_names_tuple'] # Tuple of names

    # Raw config sections for prior specifications (used for alpha/beta/mu/sigma)
    stationary_prior_config = config_data['raw_config_stationary_prior']
    trend_shocks_config = config_data['raw_config_trend_shocks'] 
    measurement_params_config = config_data['raw_config_measurement_params']


    # --- Parameter Sampling ---
    # params_for_jit will contain ALL sampled parameters needed by build_state_space_matrices_jit
    params_for_jit = {} 

    # 1. Stationary VAR Coefficients (A_diag, A_offdiag)
    # Hyperparams are sourced from config_data (already JAX arrays)
    es_param_val, fs_param_val = config_data['stationary_hyperparams_es_fs_jax']

    # A_diag has shape (p, k_stationary)
    params_for_jit['A_diag'] = numpyro.sample(
        "A_diag",
        dist.Normal(es_param_val[0], fs_param_val[0]).expand([p, k_stationary])
    )
    if num_off_diag > 0:
        # A_offdiag has shape (p, num_off_diag)
        params_for_jit['A_offdiag'] = numpyro.sample(
            "A_offdiag",
            dist.Normal(es_param_val[1], fs_param_val[1]).expand([p, num_off_diag])
        )

    # 2. Stationary Cycle Shock Variances (sampled by name)
    # Specs are parsed into config_data['stationary_shocks_parsed_spec']
    stationary_shocks_parsed_spec = config_data['stationary_shocks_parsed_spec']
    for shock_spec in stationary_shocks_parsed_spec: 
        # shock_spec = {'name': str, 'dist_idx': int, 'alpha': float, 'beta': float}
        sampled_var = numpyro.sample(
            f"stationary_var_{shock_spec['name']}", # Sample using the full name
            dist.InverseGamma(shock_spec['alpha'], shock_spec['beta'])
        )
        params_for_jit[f"stationary_var_{shock_spec['name']}"] = sampled_var # Add to params_for_jit


    # 3. Stationary Cycle Correlation Cholesky
    if k_stationary > 1:
        eta_conc = config_data['stationary_cov_prior_eta']
        eta_conc = jnp.maximum(eta_conc, _MODEL_JITTER) # Ensure positive concentration
        
        stationary_chol_sampled = numpyro.sample(
            "stationary_chol",
            dist.LKJCholesky(k_stationary, concentration=eta_conc)
        )
        params_for_jit['stationary_chol'] = stationary_chol_sampled # Add to params_for_jit
        
    # 4. Trend Shock Variances (sampled by name)
    # Specs are parsed into config_data['trend_shocks_parsed_spec']
    trend_shocks_parsed_spec = config_data['trend_shocks_parsed_spec']
    for shock_spec in trend_shocks_parsed_spec:
        # shock_spec = {'name': str, 'dist_idx': int, 'alpha': float, 'beta': float}
        sampled_var = numpyro.sample(
            f"trend_var_{shock_spec['name']}", # Sample using the full name
            dist.InverseGamma(shock_spec['alpha'], shock_spec['beta'])
        )
        params_for_jit[f"trend_var_{shock_spec['name']}"] = sampled_var # Add to params_for_jit


    # 5. Measurement Parameters (sampled by name)
    # Specs are parsed into config_data['measurement_params_parsed_spec']
    measurement_params_parsed_spec = config_data['measurement_params_parsed_spec']
    for param_spec in measurement_params_parsed_spec:
        # param_spec = {'name': str, 'dist_idx': int, ...}
        param_name = param_spec['name']
        
        if param_spec['dist_idx'] == 0: # Normal
             mu = param_spec['mu']; sigma = param_spec['sigma']
             sigma = jnp.maximum(sigma, _MODEL_JITTER) # Ensure positive sigma
             params_for_jit[param_name] = numpyro.sample(param_name, dist.Normal(mu, sigma)) # Add to params_for_jit
        elif param_spec['dist_idx'] == 1: # HalfNormal
             sigma = param_spec['sigma']
             sigma = jnp.maximum(sigma, _MODEL_JITTER) # Ensure positive sigma
             params_for_jit[param_name] = numpyro.sample(param_name, dist.HalfNormal(sigma)) # Add to params_for_jit
        # Other distributions would be handled here


    # --- Build State-Space Matrices using JITted function ---
    # Pass the entire config_data as the static argument.
    # params_for_jit contains ALL parameters that build_state_space_matrices_jit expects.
    ss_matrices = build_state_space_matrices_jit(params_for_jit, config_data)

    # Unpack matrices from the returned dictionary for likelihood calculation
    T_comp = ss_matrices['T_comp']
    R_comp = ss_matrices['R_comp'] # This R_comp is k_states x k_states (Cholesky of Q) for KF likelihood
    C_comp = ss_matrices['C_comp']
    H_comp = ss_matrices['H_comp']
    init_x_comp = ss_matrices['init_x_comp'] # k_states mean vector
    init_P_comp = ss_matrices['init_P_comp'] # k_states x k_states covariance matrix

    # --- Initial State Distribution Validity Check ---
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & jnp.all(jnp.diag(init_P_comp) >= _MODEL_JITTER / 2.0)


    # --- Prepare Static Args for Kalman Filter ---
    # static_valid_obs_idx and static_n_obs_actual come directly into the model
    # C_comp and H_comp come from ss_matrices
    static_C_obs = C_comp[static_valid_obs_idx, :] 
    static_H_obs = H_comp[static_valid_obs_idx[:, None], static_valid_obs_idx] 
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)


    # --- Instantiate and Run Kalman Filter ---
    # Use the filter method directly.
    filter_results = KalmanFilter(T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp).filter(
        y, static_valid_obs_idx, static_n_obs_actual, static_C_obs, static_H_obs, static_I_obs
    )

    # --- Compute Total Log-Likelihood ---
    total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])

    # --- Add Likelihood and Penalties ---
    # Penalize invalid initial covariance matrix
    penalty_init_P = jnp.where(is_init_P_valid_computed, 0.0, -1e10)

    # Penalize NaN/Inf in constructed state space matrices
    # Check T, R_comp (k_states x k_states), C, H, init_x, init_P
    matrices_to_check = [T_comp, R_comp, C_comp, H_comp, init_x_comp[None, :], init_P_comp] 
    any_matrix_nan = jnp.array(False)
    for mat in matrices_to_check: 
        any_matrix_nan |= jnp.any(jnp.isnan(mat))

    penalty_matrix_nan = jnp.where(any_matrix_nan, -1e10, 0.0)

    # Add log-likelihood factor
    numpyro.factor("log_likelihood", total_log_likelihood + penalty_init_P + penalty_matrix_nan)

    # --- Expose Parameters and Derived Terms as Deterministic Sites ---
    # Expose only what the smoother needs for each draw, plus other useful diagnostics.
    # The smoother needs: phi_list, Sigma_cycles, Sigma_trends_full, init_x_mean_vec, init_P_cov_matrix
    # It will reconstruct T, R_aug, C, H using these and static config.
    
    numpyro.deterministic("phi_list", ss_matrices['phi_list']) 
    numpyro.deterministic("Sigma_cycles", ss_matrices['Sigma_cycles'])
    numpyro.deterministic("Sigma_trends_full", ss_matrices['Sigma_trends_full']) # Expose full Sigma_trends
    
    # init_x_comp and init_P_comp are the initial state mean vector and covariance matrix
    numpyro.deterministic("init_x_comp", ss_matrices['init_x_mean_vec']) # Expose the mean vector
    numpyro.deterministic("init_P_comp", ss_matrices['init_P_cov_matrix']) # Expose the covariance matrix

    # Expose A_draws for diagnostics/interpretation
    numpyro.deterministic("A_draws", ss_matrices['A_draws']) 

    # Expose dimensional info if needed (already in config_data)
    # numpyro.deterministic("k_states", k_states) 

    # Expose sampled variances and Cholesky factor for diagnostics
    # Stationary variances are already exposed via their names like "stationary_var_..."
    # Stationary Cholesky is exposed as "stationary_chol" if k_stationary > 1
    # Trend variances are already exposed via their names like "trend_var_..."
    # Measurement parameters are already exposed via their names.