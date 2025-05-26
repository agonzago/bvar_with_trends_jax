import jax
import jax.numpy as jnp
from typing import Dict, Any, List # Added List for phi_list type hint

# Assuming utils directory is in the Python path and contains the necessary files
# These imports are needed if make_stationary_var_transformation_jax and create_companion_matrix_jax
# are not part of static_config_data but are called directly.
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax

# Define necessary constants if not passed through static_config_data
# For self-containment, let's define them here.
# In a real scenario, these might be imported or part of a shared config.
jax.config.update("jax_enable_x64", True) # Ensure JAX is in 64-bit mode for consistency
_DEFAULT_DTYPE = jnp.float64
_MODEL_JITTER = 1e-8

@jax.jit(static_argnames=["static_config_data"])
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
    num_off_diag = static_config_data['num_off_diag'] 
    
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
    A_offdiag_flat = params_dict.get('A_offdiag') 
    
    _stationary_variances_values = []
    for name in stationary_var_names_tuple:
        _stationary_variances_values.append(params_dict[f'stationary_var_{name}'])
    
    stationary_variances_array = jnp.stack(_stationary_variances_values) if k_stationary > 0 else jnp.array([], dtype=_DEFAULT_DTYPE)

    if k_stationary > 1: stationary_chol = params_dict['stationary_chol']
    elif k_stationary == 1: stationary_chol = jnp.eye(1, dtype=_DEFAULT_DTYPE)
    else: stationary_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    _trend_variances_values = []
    for name in trend_names_with_shocks_tuple:
        _trend_variances_values.append(params_dict[f'trend_var_{name}'])
    trend_variances_array = jnp.stack(_trend_variances_values) if n_trend_shocks > 0 else jnp.array([], dtype=_DEFAULT_DTYPE)

    _measurement_params_values = [params_dict[name] for name in measurement_param_names_tuple]
    measurement_params_array = jnp.array(_measurement_params_values, dtype=_DEFAULT_DTYPE) if _measurement_params_values else jnp.array([],dtype=_DEFAULT_DTYPE)

    # --- Reconstruct A_draws ---
    A_draws = jnp.zeros((p, k_stationary, k_stationary), dtype=_DEFAULT_DTYPE)
    if k_stationary > 0:
        A_draws = A_draws.at[:, jnp.arange(k_stationary), jnp.arange(k_stationary)].set(A_diag)
        if num_off_diag > 0 and A_offdiag_flat is not None:
            A_draws = A_draws.at[:, static_off_diag_rows, static_off_diag_cols].set(A_offdiag_flat)

    # --- Construct Sigma_cycles ---
    if k_stationary > 0:
        stationary_D_sds = jnp.diag(jnp.sqrt(jnp.maximum(stationary_variances_array, _MODEL_JITTER)))
        Sigma_cycles = stationary_chol @ stationary_D_sds @ stationary_chol.T # As per prompt
        Sigma_cycles = (Sigma_cycles + Sigma_cycles.T) / 2.0 + _MODEL_JITTER * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)
    else: Sigma_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)

    # --- Construct Sigma_trends ---
    Sigma_trends = jnp.diag(jnp.maximum(trend_variances_array, _MODEL_JITTER)) if n_trend_shocks > 0 else jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
    if n_trend_shocks > 0: Sigma_trends = (Sigma_trends + Sigma_trends.T) / 2.0

    # --- Transform A to phi_list ---
    phi_list: List[jax.Array] = [] # Ensure phi_list is always a list
    if k_stationary > 0:
         # make_stationary_var_transformation_jax returns (List[jax.Array], jax.Array)
        phi_list_candidate, _ = make_stationary_var_transformation_jax(Sigma_cycles, [A_draws[i] for i in range(p)], k_stationary, p)
        phi_list = phi_list_candidate
    
    # --- Construct T_comp ---
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE))
    if k_stationary > 0: T_comp = T_comp.at[k_trends:, k_trends:].set(create_companion_matrix_jax(phi_list, p, k_stationary))

    # --- Construct R_comp ---
    R_comp = jnp.zeros((k_states, n_trend_shocks + k_stationary), dtype=_DEFAULT_DTYPE)
    for shock_idx, name_w_shock in enumerate(trend_names_with_shocks_tuple):
        for state_idx, trend_name in enumerate(trend_var_names_tuple):
            if name_w_shock == trend_name: R_comp = R_comp.at[state_idx, shock_idx].set(1.0); break
    if k_stationary > 0: R_comp = R_comp.at[k_trends:k_trends+k_stationary, n_trend_shocks:].set(jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE))
    
    # --- Construct C_comp ---
    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    for obs_idx, terms in parsed_model_eqs_jax_detailed:
        for term_type, state_idx_C, param_idx, sign_val in terms:
            # Ensure param_idx is valid before indexing measurement_params_array
            is_param_term_and_valid = (term_type == 1 and param_idx >=0 and param_idx < measurement_params_array.shape[0] and measurement_params_array.size > 0)
            
            param_contribution = jnp.where(is_param_term_and_valid,
                                           measurement_params_array[param_idx],
                                           0.0) # Use 0 if not a valid param term, effectively ignoring it if term_type is 1 but param is invalid
            
            term_value = jnp.where(term_type == 1, # If it's supposed to be a param term
                                   sign_val * param_contribution,
                                   sign_val * 1.0) # If it's a direct state term

            # Add to C_comp only if it's a direct state term, or a valid parameter term
            # This avoids adding sign_val * 0.0 if param_idx was invalid for a param term.
            # Or, simplify: if term_type is 1 but param_idx is invalid, value is sign_val * 0 = 0.
            # If term_type is 0 (direct state), value is sign_val * 1.0.
            # The previous logic was: val = sign_val * (measurement_params_array[param_idx] if term_type == 1 and param_idx >=0 and measurement_params_array.size > 0 else 1.0)
            # This was incorrect as it would use 1.0 for invalid param terms instead of 0.0 * sign_val for the parameter part.
            
            # Corrected logic for term value based on type:
            current_term_value = 0.0
            if term_type == 0: # Direct state term
                current_term_value = sign_val * 1.0
            elif is_param_term_and_valid: # Valid parameter term
                current_term_value = sign_val * measurement_params_array[param_idx]
            # If term_type == 1 but not is_param_term_and_valid, current_term_value remains 0.0, so nothing is added.
            
            C_comp = C_comp.at[obs_idx, state_idx_C].add(current_term_value)
            
    # --- Construct H_comp ---
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)
    # --- Construct init_x_comp ---
    init_x_comp = init_x_means_flat
    # --- Construct init_P_comp ---
    init_P_comp = jnp.diag(init_P_diag_flat)
    init_P_comp = (init_P_comp + init_P_comp.T) / 2.0 + _MODEL_JITTER * jnp.eye(k_states, dtype=_DEFAULT_DTYPE)
    
    return {
        "T_comp": T_comp, "R_comp": R_comp, "C_comp": C_comp, "H_comp": H_comp,
        "init_x_comp": init_x_comp, "init_P_comp": init_P_comp,
        "Sigma_cycles": Sigma_cycles, "Sigma_trends": Sigma_trends,
        "phi_list": phi_list, "A_draws": A_draws
    }
