# --- var_ss_model.py (Final Corrected) ---
# NumPyro model for the BVAR with stationary prior and trends.

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

    # --- Extract Dynamic Parameters ---
    A_diag = params_dict['A_diag'] 
    # A_offdiag_flat will be None if num_off_diag is 0 and not in params_dict
    A_offdiag_flat = params_dict.get('A_offdiag') 
    
    _stationary_variances_values = []
    for name in stationary_var_names_tuple: # This tuple comes from static_config_data
        _stationary_variances_values.append(params_dict[f'stationary_var_{name}'])
    
    if k_stationary > 0:
        stationary_variances_array = jnp.stack(_stationary_variances_values, axis=-1)
    else:
        stationary_variances_array = jnp.array([], dtype=_DEFAULT_DTYPE)

    # stationary_chol is Cholesky of Correlation matrix
    if k_stationary > 1:
        stationary_chol = params_dict['stationary_chol'] 
    elif k_stationary == 1: 
        stationary_chol = jnp.eye(1, dtype=_DEFAULT_DTYPE) # Correlation matrix is 1, Cholesky is 1
    else: # k_stationary == 0
        stationary_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    _trend_variances_values = []
    for name in trend_names_with_shocks_tuple: # This tuple comes from static_config_data
        _trend_variances_values.append(params_dict[f'trend_var_{name}'])
    
    if n_trend_shocks > 0:
        trend_variances_array = jnp.stack(_trend_variances_values, axis=-1)
    else:
        trend_variances_array = jnp.array([], dtype=_DEFAULT_DTYPE)

    _measurement_params_sampled_values = []
    for name in measurement_param_names_tuple: # This tuple comes from static_config_data
        _measurement_params_sampled_values.append(params_dict[name])
    
    if len(measurement_param_names_tuple) > 0:
        measurement_params_sampled_array = jnp.array(_measurement_params_sampled_values, dtype=_DEFAULT_DTYPE)
    else:
        measurement_params_sampled_array = jnp.array([], dtype=_DEFAULT_DTYPE)


    # --- Reconstruct A_draws ---
    A_draws = jnp.zeros((p, k_stationary, k_stationary), dtype=_DEFAULT_DTYPE)
    if k_stationary > 0:
        A_draws = A_draws.at[:, jnp.arange(k_stationary), jnp.arange(k_stationary)].set(A_diag)
        if num_off_diag > 0 and A_offdiag_flat is not None:
            A_draws = A_draws.at[:, static_off_diag_rows, static_off_diag_cols].set(A_offdiag_flat)

    # --- Construct Sigma_cycles ---
    # if k_stationary > 0:
    #     stationary_D_sds = jnp.diag(jnp.sqrt(jnp.maximum(stationary_variances_array, _MODEL_JITTER)))
        
    #     Sigma_cycles = (stationary_D_sds @ stationary_chol) @ (stationary_D_sds @ stationary_chol).T
        
    #     Sigma_cycles = (Sigma_cycles + Sigma_cycles.T) / 2.0 
    #     Sigma_cycles = Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)
    # else: 
    #     Sigma_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    if k_stationary > 0:
        # Ensure positive variances
        stationary_variances_safe = jnp.maximum(stationary_variances_array, _MODEL_JITTER)
        
        # More stable construction
        if k_stationary == 1:
            Sigma_cycles = jnp.diag(stationary_variances_safe)
            L_cycles = jnp.diag(jnp.sqrt(stationary_variances_safe))
        else:
            # Direct construction without double decomposition
            std_devs = jnp.sqrt(stationary_variances_safe)
            D_matrix = jnp.diag(std_devs)
            Sigma_cycles = D_matrix @ stationary_chol @ stationary_chol.T @ D_matrix
            
            # Ensure symmetry and positive definiteness
            Sigma_cycles = (Sigma_cycles + Sigma_cycles.T) / 2.0
            Sigma_cycles = Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)
            
            # More stable Cholesky
            try:
                L_cycles = jsl.cholesky(Sigma_cycles)
            except:
                # Fallback: use diagonal if Cholesky fails
                L_cycles = jnp.diag(jnp.sqrt(jnp.diag(Sigma_cycles)))
    else:
        Sigma_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        L_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        
    # --- Construct Sigma_trends ---
    if n_trend_shocks > 0:
        Sigma_trends = jnp.diag(jnp.maximum(trend_variances_array, _MODEL_JITTER)) 
        Sigma_trends = (Sigma_trends + Sigma_trends.T) / 2.0 # Ensure symmetry (already diag, but good practice)
    else: 
        Sigma_trends = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    # --- Transform A to phi_list ---
    if k_stationary > 0:
        A_list_for_phi = [A_draws[i] for i in range(p)] 
        phi_list, _ = make_stationary_var_transformation_jax(Sigma_cycles, A_list_for_phi, k_stationary, p)
    else:
        phi_list = [] 

    # --- Construct T_comp ---
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE))
    if k_stationary > 0:
        companion_matrix = create_companion_matrix_jax(phi_list, p, k_stationary)
        T_comp = T_comp.at[k_trends:, k_trends:].set(companion_matrix)

    # --- Construct R_comp ---
    # R_comp maps shocks to state variables. The Kalman filter computes state noise covariance as R_comp @ R_comp.T
    # So R_comp should be such that R_comp @ R_comp.T = block_diag(Sigma_trends_mapped, Sigma_cycles_mapped_to_states)
    # where Sigma_trends_mapped has non-zero elements corresponding to trends with shocks,
    # and Sigma_cycles_mapped_to_states is Sigma_cycles placed at the current stationary state rows.

    R_comp = jnp.zeros((k_states, n_trend_shocks + k_stationary), dtype=_DEFAULT_DTYPE)

    # 1. Handle trend shocks
    if n_trend_shocks > 0:
        # trend_variances_array is already (n_trend_shocks,)
        # L_trends_diag will be sqrt of these variances
        L_trends_diag_values = jnp.sqrt(jnp.maximum(trend_variances_array, _MODEL_JITTER))

        # Place these into R_comp at the correct rows
        # The loop below iterates through the *shocks*. Each shock `j` corresponds to a
        # specific trend variable defined in `trend_names_with_shocks_tuple[j]`.
        # We need to find the row index of this specific trend variable in the overall state vector.
        for j in range(n_trend_shocks):
            shocked_trend_name = trend_names_with_shocks_tuple[j]
            # Find the state index for this shocked trend
            # trend_var_names_tuple contains all k_trends trend names in state order
            target_trend_state_index = -1 
            # It's guaranteed that shocked_trend_name is one of trend_var_names_tuple
            # if the config is valid. We can find the index more directly.
            # However, to be robust or if direct indexing isn't trivial due to JAX tracing,
            # an explicit search or precomputed map (outside JIT if possible) would be needed.
            # For JIT context, if `trend_var_names_tuple` and `trend_names_with_shocks_tuple`
            # are fixed Python tuples of strings, this comparison can work.
            
            # Simpler way to find index if names are unique and present:
            # This loop is fine for JAX as long as tuple elements are comparable.
            for idx in range(k_trends): # k_trends is total number of trend states
                if trend_var_names_tuple[idx] == shocked_trend_name:
                    target_trend_state_index = idx
                    break
            
            # Only proceed if a valid index was found (it should always be found for valid config)
            # The condition `target_trend_state_index != -1` is more for safety in general code;
            # in JAX, if it could fail, it might need `lax.cond` or error if not found.
            # Assuming it's always found based on config structure.
            R_comp = R_comp.at[target_trend_state_index, j].set(L_trends_diag_values[j])

    # 2. Handle cycle shocks
    if k_stationary > 0:
        # Sigma_cycles is (k_stationary, k_stationary)
        # L_cycles is Cholesky of Sigma_cycles
        L_cycles = jsl.cholesky(Sigma_cycles + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE), lower=True)
        
        # Place L_cycles into R_comp
        # It affects the first k_stationary elements of the stationary block of states
        # These are located at indices k_trends to k_trends + k_stationary -1 in the state vector.
        # The shocks for cycles start at column n_trend_shocks in R_comp.
        R_comp = R_comp.at[k_trends : k_trends + k_stationary, 
                           n_trend_shocks : n_trend_shocks + k_stationary].set(L_cycles)
        
    # --- Construct C_comp ---
    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    for obs_idx, terms_for_obs in parsed_model_eqs_jax_detailed: 
        for term_type, state_idx_in_C, param_idx_if_any, sign in terms_for_obs: 
            is_param_term = (term_type == 1)
            param_val = jnp.array(0.0, dtype=_DEFAULT_DTYPE) 
            # Check if measurement_params_sampled_array is non-empty and param_idx_if_any is valid
            if measurement_params_sampled_array.size > 0 and param_idx_if_any >= 0 and param_idx_if_any < measurement_params_sampled_array.shape[0]:
                 param_val = measurement_params_sampled_array[param_idx_if_any]
            
            value_to_add = jnp.where(
                is_param_term,
                sign * param_val, # Use param_val which is correctly indexed or default
                sign * 1.0 
            )
            C_comp = C_comp.at[obs_idx, state_idx_in_C].add(value_to_add)

    # --- Construct H_comp ---
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)
    # observation_noise_variance = 1e-2
    # H_comp = jnp.eye(k_endog, dtype=_DEFAULT_DTYPE) * observation_noise_variance

    # --- Construct init_x_comp ---
    init_x_comp = init_x_means_flat

    # --- Construct init_P_comp ---
    init_P_comp = jnp.diag(init_P_diag_flat) 
    init_P_comp = (init_P_comp + init_P_comp.T) / 2.0 
    init_P_comp = init_P_comp + _MODEL_JITTER * jnp.eye(k_states, dtype=_DEFAULT_DTYPE)
    
    return {
        "T_comp": T_comp, "R_comp": R_comp, "C_comp": C_comp, "H_comp": H_comp,
        "init_x_comp": init_x_comp, "init_P_comp": init_P_comp,
        "Sigma_cycles": Sigma_cycles, "Sigma_trends": Sigma_trends,
        "phi_list": phi_list, "A_draws": A_draws
    }

def load_config_and_prepare_jax_static_args(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration, parses it, and prepares a dictionary of static
    arguments suitable for JAX-based models, particularly for the NumPyro BVAR model.
    """
    # Step 1 & 2: Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    # Helper to create the full list of state names in order
    full_state_names_list = list(trend_var_names)
    for i in range(p):
        for stat_var in stationary_var_names:
            if i == 0:
                full_state_names_list.append(stat_var)
            else:
                full_state_names_list.append(f"{stat_var}_t_minus_{i}")
    full_state_names_tuple = tuple(full_state_names_list)
    
    # Use existing parse_initial_state_config helper
    raw_initial_conds = config.get('initial_conditions', {})
    # The helper parse_initial_state_config is designed for the states listed in the YAML.
    # We need to map this to the full_state_names_tuple structure.
    # The helper returns dict of {state_name: {"mean": val, "var": val}}
    parsed_initial_conds_from_yaml = parse_initial_state_config(raw_initial_conds)

    init_x_means_flat_list = []
    init_P_diag_flat_list = []

    for state_name in full_state_names_tuple:
        # Check if this state_name (which might include _t_minus_lag) is in the parsed YAML keys
        # The YAML usually only specifies current states (e.g., "cycle_gdp")
        # For lagged states, we need to refer to the current state's config or use defaults.
        
        base_name_for_lag = state_name
        if "_t_minus_" in state_name:
            base_name_for_lag = state_name.split("_t_minus_")[0]

        if state_name in parsed_initial_conds_from_yaml:
            init_x_means_flat_list.append(parsed_initial_conds_from_yaml[state_name]['mean'])
            init_P_diag_flat_list.append(parsed_initial_conds_from_yaml[state_name]['var'])
        elif base_name_for_lag in parsed_initial_conds_from_yaml and base_name_for_lag != state_name : # It's a lagged state, use its current value's spec
            init_x_means_flat_list.append(parsed_initial_conds_from_yaml[base_name_for_lag]['mean']) # Lagged means default to current
            init_P_diag_flat_list.append(parsed_initial_conds_from_yaml[base_name_for_lag]['var'])  # Lagged vars default to current
        else: # Default if not specified at all
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)
            
    init_x_means_flat = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    init_P_diag_flat = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)

    # For the initial_conds_parsed_jax (the dictionary structure used by the main model for clarity)
    # This should map to the structure expected by the model's initial state setup logic.
    # The model's logic itself handles mapping these names to the flat arrays.
    # The main model uses 'initial_conditions_parsed' from the config_data dict.
    # So, we will name this key 'initial_conditions_parsed' in the output dict.
    initial_conds_parsed_for_model = parsed_initial_conds_from_yaml


    # Step 7: Parse model_equations
    model_equations_config = config.get('model_equations', {})
    measurement_params_config_raw = config.get('parameters', []) # List of dicts
    measurement_param_names_list = [p['name'] for p in measurement_params_config_raw if 'name' in p]
    measurement_param_names_tuple = tuple(measurement_param_names_list)
    
    # State names relevant for C matrix (trends + current stationary)
    c_matrix_state_names = list(trend_var_names) + list(stationary_var_names)
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names_tuple)}

    parsed_model_eqs_list_for_model = [] # This will be 'model_equations_parsed' for the model
    for obs_idx, obs_name in enumerate(observable_names):
        eq_str = model_equations_config.get(obs_name)
        if eq_str is None:
            raise ValueError(f"Equation for observable '{obs_name}' not found in model_equations.")
        
        raw_parsed_terms = _parse_equation_jax(
            eq_str, 
            list(trend_var_names), 
            list(stationary_var_names), 
            list(measurement_param_names_tuple) 
        )
        # The model expects: list of (obs_idx, parsed_terms)
        # where parsed_terms is List[Tuple[Optional[str], str, float]] -> (param_name, state_name_in_eq, sign)
        # This is directly what _parse_equation_jax returns.
        # The new `parsed_model_eqs_jax` is for a more direct C matrix build, not directly used by current model.
        # So we provide what the current model expects under 'model_equations_parsed'
        parsed_model_eqs_list_for_model.append((obs_idx, raw_parsed_terms))

    # Create the specified parsed_model_eqs_jax structure (for potential future use or direct C matrix construction)
    # This structure is (term_type, state_index_in_C, param_index_if_any, sign)
    parsed_model_eqs_jax_detailed = []
    for obs_idx, obs_name in enumerate(observable_names):
        eq_str = model_equations_config.get(obs_name)
        # Already validated eq_str exists
        raw_parsed_terms = _parse_equation_jax(
            eq_str, list(trend_var_names), list(stationary_var_names), list(measurement_param_names_tuple)
        )
        processed_terms_for_obs = []
        for param_name, state_name_in_eq, sign in raw_parsed_terms:
            term_type = 0 if param_name is None else 1
            state_index_in_C = state_to_c_idx_map[state_name_in_eq]
            param_index_if_any = param_to_idx_map[param_name] if param_name is not None else -1
            processed_terms_for_obs.append(
                (term_type, state_index_in_C, param_index_if_any, float(sign))
            )
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))
    
    parsed_model_eqs_jax_detailed_tuple = tuple(parsed_model_eqs_jax_detailed)


    # Step 8: Parse trend_shocks (names)
    trend_shocks_section = config.get('trend_shocks', {}) # This is the top-level key
    trend_shocks_config_raw = trend_shocks_section.get('trend_shocks', {}) # This is the nested dict
    trend_names_with_shocks_list = []
    for trend_name in trend_var_names:
        if trend_name in trend_shocks_config_raw:
            if 'distribution' in trend_shocks_config_raw[trend_name]:
                 trend_names_with_shocks_list.append(trend_name)
    trend_names_with_shocks_tuple = tuple(trend_names_with_shocks_list)
    n_trend_shocks = len(trend_names_with_shocks_tuple)

    # Step 9: Calculate static_off_diag_indices for A matrix
    static_off_diag_rows, static_off_diag_cols = _get_off_diagonal_indices(k_stationary)
    num_off_diag = k_stationary * (k_stationary - 1) if k_stationary > 1 else 0
    static_off_diag_indices = (static_off_diag_rows, static_off_diag_cols)


    # Step 10: Parse stationary_prior
    stationary_prior_config_raw = config.get('stationary_prior', {})
    hyperparams_raw = stationary_prior_config_raw.get('hyperparameters', {'es': [0.0, 0.0], 'fs': [1.0, 0.5]})
    es_jax = jnp.array(hyperparams_raw.get('es', [0.0, 0.0]), dtype=_DEFAULT_DTYPE)
    fs_jax = jnp.array(hyperparams_raw.get('fs', [1.0, 0.5]), dtype=_DEFAULT_DTYPE)
    eta_float = float(stationary_prior_config_raw.get('covariance_prior', {}).get('eta', 1.0))

    stationary_shocks_raw_dict = stationary_prior_config_raw.get('stationary_shocks', {})
    _stationary_shocks_parsed_list = []
    for stat_var_name in stationary_var_names: 
        spec = stationary_shocks_raw_dict.get(stat_var_name)
        if not spec or 'distribution' not in spec:
            raise ValueError(f"Missing shock specification for stationary variable '{stat_var_name}' in config.")
        dist_name = spec['distribution'].lower()
        params = spec.get('parameters', {})
        if dist_name == 'inverse_gamma':
            alpha = float(params.get('alpha', 2.0)) 
            beta = float(params.get('beta', 0.5))
            _stationary_shocks_parsed_list.append({'name': stat_var_name, 'dist_idx': 0, 'alpha': alpha, 'beta': beta})
        else:
            raise NotImplementedError(f"Stationary shock distribution '{dist_name}' not supported.")
    stationary_shocks_parsed_jax = tuple(_stationary_shocks_parsed_list)


    # Step 11: Parse trend_shocks (specs)
    _trend_shocks_parsed_list = []
    for trend_name_with_shock in trend_names_with_shocks_tuple:
        spec = trend_shocks_config_raw.get(trend_name_with_shock) 
        dist_name = spec['distribution'].lower()
        params = spec.get('parameters', {})
        if dist_name == 'inverse_gamma': 
            alpha = float(params.get('alpha', 2.0))
            beta = float(params.get('beta', 0.5))
            _trend_shocks_parsed_list.append({'name': trend_name_with_shock, 'dist_idx': 0, 'alpha': alpha, 'beta': beta})
        else:
            raise NotImplementedError(f"Trend shock distribution '{dist_name}' not supported.")
    trend_shocks_parsed_jax = tuple(_trend_shocks_parsed_list)


    # Step 12: Parse parameters (measurement parameters specs)
    _measurement_params_config_parsed_list = []
    for param_spec_raw in measurement_params_config_raw: # This is already a list of dicts
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
            raise NotImplementedError(f"Measurement parameter prior distribution '{dist_name}' for '{name}' not supported.")
        _measurement_params_config_parsed_list.append(parsed_item)
    measurement_params_config_parsed_jax = tuple(_measurement_params_config_parsed_list)

    # Step 13: Return dictionary
    # This dictionary will be passed as `config_data` to the NumPyro model
    # It should contain keys that the model expects, e.g., 'initial_conditions_parsed', 'model_equations_parsed'
    return {
        # Parsed items directly used by the model's current signature/logic
        "initial_conditions_parsed": initial_conds_parsed_for_model, # Used by model for init_x, init_P
        "model_equations_parsed": tuple(parsed_model_eqs_list_for_model), # Used by model for C matrix
        "trend_names_with_shocks": trend_names_with_shocks_tuple, # Used by model
        "static_off_diag_indices": static_off_diag_indices, # Used by model
        "num_off_diag": num_off_diag, # Used by model
        "var_order": p, # Used by model

        # Raw config sections, also passed to model for direct access to prior specs etc.
        "raw_config_initial_conds": raw_initial_conds, # For model to use parse_initial_state_config internally if needed (though redundant now)
        "raw_config_stationary_prior": stationary_prior_config_raw,
        "raw_config_trend_shocks": trend_shocks_section, # Pass the whole 'trend_shocks' dict from YAML
        "raw_config_measurement_params": measurement_params_config_raw, # List of dicts

        # --- Items below are more detailed or for alternative uses, not all directly consumed by current model signature ---
        "observable_names": observable_names,
        "trend_var_names": trend_var_names,
        "stationary_var_names": stationary_var_names,
        "k_endog": k_endog,
        "k_trends": k_trends,
        "k_stationary": k_stationary,
        "k_states": k_states,
        "full_state_names_tuple": full_state_names_tuple,
        "init_x_means_flat": init_x_means_flat,
        "init_P_diag_flat": init_P_diag_flat,
        "parsed_model_eqs_jax_detailed": parsed_model_eqs_jax_detailed_tuple, # The (term_type, state_idx, param_idx, sign) structure
        "measurement_param_names_tuple": measurement_param_names_tuple,
        "n_trend_shocks": n_trend_shocks,
        "stationary_hyperparams_es_fs_jax": (es_jax, fs_jax),
        "stationary_cov_prior_eta": eta_float,
        "stationary_shocks_parsed_spec": stationary_shocks_parsed_jax, # Renamed for clarity
        "trend_shocks_parsed_spec": trend_shocks_parsed_jax,       # Renamed for clarity
        "measurement_params_parsed_spec": measurement_params_config_parsed_jax, # Renamed for clarity
        # Keep raw model equations string dict if needed elsewhere
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
    # They can be sourced from config_data if preferred by the caller.
    trend_var_names: List[str], # Expected to match config_data['trend_var_names']
    stationary_var_names: List[str],  # Expected to match config_data['stationary_var_names']
    observable_names: List[str]       # Expected to match config_data['observable_names']
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
    n_trend_shocks = config_data['n_trend_shocks']
    
    # Parameter names from config_data (used for sampling and collecting into params_for_jit)
    current_stationary_var_names = config_data['stationary_var_names'] # Tuple of names
    trend_names_with_shocks_tuple = config_data['trend_names_with_shocks'] # Tuple of names
    measurement_param_names_tuple = config_data['measurement_param_names_tuple'] # Tuple of names

    # Raw config sections for prior specifications
    stationary_prior_config = config_data['raw_config_stationary_prior']
    trend_shocks_config = config_data['raw_config_trend_shocks'] 
    measurement_params_config = config_data['raw_config_measurement_params']


    # --- Parameter Sampling ---
    params_for_jit = {} # Dictionary to hold all sampled parameters for the JIT function

    # 1. Stationary VAR Coefficients (A_diag, A_offdiag)
    hyperparams = stationary_prior_config.get('hyperparameters', {})
    es_param_val = jnp.asarray(hyperparams.get('es', [0.0, 0.0]), dtype=_DEFAULT_DTYPE)
    fs_param_val = jnp.asarray(hyperparams.get('fs', [1.0, 0.5]), dtype=_DEFAULT_DTYPE)

    params_for_jit['A_diag'] = numpyro.sample(
        "A_diag",
        dist.Normal(es_param_val[0], fs_param_val[0]).expand([p, k_stationary])
    )
    if num_off_diag > 0:
        params_for_jit['A_offdiag'] = numpyro.sample(
            "A_offdiag",
            dist.Normal(es_param_val[1], fs_param_val[1]).expand([p, num_off_diag])
        )
    # If num_off_diag == 0, 'A_offdiag' is not added to params_for_jit; 
    # build_state_space_matrices_jit handles this with .get('A_offdiag').

    # 2. Stationary Cycle Shock Variances
    _temp_stationary_variances_list = [] # To store raw sampled variances for deterministic sites
    stationary_shocks_spec_dict = stationary_prior_config.get('stationary_shocks', {})
    for stat_var_name in current_stationary_var_names: 
        shock_spec = stationary_shocks_spec_dict.get(stat_var_name)
        if shock_spec is None or shock_spec.get('distribution','').lower() != 'inverse_gamma':
             # This should ideally be caught by load_config_and_prepare_jax_static_args
             raise ValueError(f"Missing or invalid InverseGamma spec for stationary_var_{stat_var_name}")
        
        spec_params = shock_spec.get('parameters', {}) 
        alpha = spec_params.get('alpha', 2.0) 
        beta = spec_params.get('beta', 0.5)   
        sampled_var = numpyro.sample(
            f"stationary_var_{stat_var_name}",
            dist.InverseGamma(alpha, beta)
        )
        params_for_jit[f'stationary_var_{stat_var_name}'] = sampled_var
        _temp_stationary_variances_list.append(sampled_var)

    # 3. Stationary Cycle Correlation Cholesky
    if k_stationary > 1:
        stationary_chol_sampled = numpyro.sample(
            "stationary_chol",
            dist.LKJCholesky(k_stationary, concentration=stationary_prior_config.get('covariance_prior', {}).get('eta', 1.0))
        )
        params_for_jit['stationary_chol'] = stationary_chol_sampled
    # If k_stationary <= 1, 'stationary_chol' is not sampled and not added to params_for_jit.
    # build_state_space_matrices_jit handles this by creating jnp.eye(1) or jnp.empty((0,0)).
        
    # 4. Trend Shock Variances
    _temp_trend_variances_list = [] # To store raw sampled variances for deterministic sites
    actual_trend_shocks_specs_dict = trend_shocks_config.get('trend_shocks', {})
    for trend_name in trend_names_with_shocks_tuple: 
        shock_spec = actual_trend_shocks_specs_dict.get(trend_name)
        if shock_spec is None or shock_spec.get('distribution','').lower() != 'inverse_gamma':
            # This should ideally be caught by load_config_and_prepare_jax_static_args
            raise ValueError(f"Missing or invalid InverseGamma spec for trend_var_{trend_name}")

        spec_params = shock_spec.get('parameters', {}) 
        alpha = spec_params.get('alpha', 2.0) 
        beta = spec_params.get('beta', 0.5)   
        sampled_var = numpyro.sample(
            f"trend_var_{trend_name}",
            dist.InverseGamma(alpha, beta)
        )
        params_for_jit[f'trend_var_{trend_name}'] = sampled_var
        _temp_trend_variances_list.append(sampled_var)

    # 5. Measurement Parameters
    for param_spec in measurement_params_config: 
        param_name = param_spec['name']
        prior_spec = param_spec.get('prior', {}) 
        dist_name = prior_spec.get('distribution', '').lower()
        spec_params = prior_spec.get('parameters', {}) 

        if dist_name == 'normal':
            mu = spec_params.get('mu', 0.0); sigma = spec_params.get('sigma', 1.0)
            params_for_jit[param_name] = numpyro.sample(param_name, dist.Normal(mu, sigma))
        elif dist_name == 'half_normal':
            sigma = spec_params.get('sigma', 1.0)
            params_for_jit[param_name] = numpyro.sample(param_name, dist.HalfNormal(sigma))
        else:
             # This should ideally be caught by load_config_and_prepare_jax_static_args
             raise NotImplementedError(f"Prior distribution '{dist_name}' not supported for {param_name}")

    # --- Build State-Space Matrices using JITted function ---
    ss_matrices = build_state_space_matrices_jit(params_for_jit, config_data)

    # Unpack matrices from the returned dictionary
    T_comp = ss_matrices['T_comp']
    R_comp = ss_matrices['R_comp']
    C_comp = ss_matrices['C_comp']
    H_comp = ss_matrices['H_comp']
    init_x_comp = ss_matrices['init_x_comp']
    init_P_comp = ss_matrices['init_P_comp']
    Sigma_cycles = ss_matrices['Sigma_cycles'] 
    Sigma_trends = ss_matrices['Sigma_trends'] 
    phi_list = ss_matrices['phi_list']
    # A_draws is also in ss_matrices['A_draws'] if needed for other computations or deterministics
    
    # --- Initial State Distribution Validity Check (using matrices from JIT) ---
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & jnp.all(jnp.diag(init_P_comp) >= _MODEL_JITTER / 2.0)


    # --- Prepare Static Args for Kalman Filter (using matrices from JIT) ---
    static_C_obs = C_comp[static_valid_obs_idx, :] 
    static_H_obs = H_comp[static_valid_obs_idx[:, None], static_valid_obs_idx] 
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)


    # --- Instantiate and Run Kalman Filter ---
    # Dtypes are handled within build_state_space_matrices_jit for its outputs
    kf = KalmanFilter(T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp)
    filter_results = kf.filter(
        y, static_valid_obs_idx, static_n_obs_actual, static_C_obs, static_H_obs, static_I_obs
    )

    # --- Compute Total Log-Likelihood ---
    total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])

    # --- Add Likelihood and Penalties ---
    penalty_init_P = jnp.where(is_init_P_valid_computed, 0.0, -1e10)

    matrices_to_check = [T_comp, R_comp, C_comp, H_comp, init_x_comp[None, :], init_P_comp] 
    any_matrix_nan = jnp.array(False)
    # This loop will be unrolled by JAX JIT if this function itself is JITted,
    # but here it runs in Python mode over JAX arrays.
    for mat in matrices_to_check: 
        any_matrix_nan |= jnp.any(jnp.isnan(mat))
    penalty_matrix_nan = jnp.where(any_matrix_nan, -1e10, 0.0)

    numpyro.factor("log_likelihood", total_log_likelihood + penalty_init_P + penalty_matrix_nan)

    # --- Expose Transformed Parameters ---
    numpyro.deterministic("phi_list", phi_list) 
    numpyro.deterministic("Sigma_cycles", Sigma_cycles) 
    numpyro.deterministic("Sigma_trends", Sigma_trends) 
    numpyro.deterministic("T_comp", T_comp) 
    numpyro.deterministic("R_comp", R_comp) 
    numpyro.deterministic("C_comp", C_comp) 
    numpyro.deterministic("H_comp", H_comp) 
    numpyro.deterministic("init_x_comp", init_x_comp) 
    numpyro.deterministic("init_P_comp", init_P_comp) 
    numpyro.deterministic("A_draws", ss_matrices['A_draws']) # Expose A_draws from JIT

    numpyro.deterministic("k_states", k_states) 

    # Expose originally sampled variances and Cholesky factor for diagnostics
    # Stationary
    # if k_stationary > 0:
    #      for i, name in enumerate(current_stationary_var_names): 
    #           numpyro.deterministic(f"stationary_var_{name}_det", _temp_stationary_variances_list[i])
         
    #      if k_stationary > 1: 
    #         numpyro.deterministic("stationary_chol_det", params_for_jit['stationary_chol'])
    #      elif k_stationary == 1: 
    #         numpyro.deterministic("stationary_chol_det", jnp.eye(1, dtype=_DEFAULT_DTYPE))

    # Trends
    if n_trend_shocks > 0: 
        for i, name in enumerate(trend_names_with_shocks_tuple): 
             numpyro.deterministic(f"trend_var_{name}_det", _temp_trend_variances_list[i])

    # Measurement params
    for param_name in measurement_param_names_tuple: 
         numpyro.deterministic(f"{param_name}_det", params_for_jit[param_name])


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