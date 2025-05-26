# --- estimate_bvar_ss_simulated.py (Fixed Hashability Issues) ---
# This script runs the NumPyro BVAR with trends estimation on SIMULATED data,
# performs simulation smoothing, and compares to the true simulated states.

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
# import arviz as az # Removed as HDI is no longer used
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import time
import os
import yaml
from typing import Dict, Any, List, Tuple

# Configure JAX as requested
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64

# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Add the current directory to sys.path
# sys.path.append(current_dir)
# Set host device count for parallel chains
try:
    import numpyro
    numpyro.set_host_device_count(2)
except:
    pass

# Assuming utils directory is in the Python path
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax
from core.simulate_bvar_jax import simulate_bvar_with_trends_jax
from core.var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices

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

# Import the fixed simulation smoother
from core.run_single_draw import run_simulation_smoother_single_params_jit

def create_sample_config():
    """Create a sample configuration file for testing."""
    sample_config = {
        'var_order': 1,
        'variables': {
            'observables': ['gdp_growth', 'inflation'],
            'trends': ['trend_gdp', 'trend_inf'],
            'stationary': ['cycle_gdp', 'cycle_inf']
        },
        'model_equations': {
            'gdp_growth': 'trend_gdp + cycle_gdp',
            'inflation': 'trend_inf + cycle_inf'
        },
        'initial_conditions': {
            'states': {
                'trend_gdp': {'mean': 0.0, 'var': 1.0},
                'trend_inf': {'mean': 0.0, 'var': 1.0},
                'cycle_gdp': {'mean': 0.0, 'var': 0.5},
                'cycle_inf': {'mean': 0.0, 'var': 0.5}
            }
        },
        'stationary_prior': {
            'hyperparameters': {
                'es': [0.0, 0.0],
                'fs': [1.0, 0.5]
            },
            'covariance_prior': {
                'eta': 1.0
            },
            'stationary_shocks': {
                'cycle_gdp': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 2.0, 'beta': 0.5}
                },
                'cycle_inf': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 2.0, 'beta': 0.5}
                }
            }
        },
        'trend_shocks': {
            'trend_shocks': {
                'trend_gdp': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 2.0, 'beta': 0.05}
                },
                'trend_inf': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 2.0, 'beta': 0.05}
                }
            }
        },
        'parameters': {
            'measurement': []
        }
    }

    with open('bvar_stationary_sim.yml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    print("Created sample configuration file: bvar_stationary_sim.yml")

def load_config_simple(yaml_path: str) -> Dict[str, Any]:
    """Simple config loader that avoids complex parsing issues."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    print(f"Debug: Raw config_dict structure: {config_dict}")

    # Extract variables - handle both simple list format and complex dict format
    variables_config = config_dict.get('variables', {})
    print(f"Debug: variables_config: {variables_config}")

    # Helper function to extract names from variable specifications
    def extract_names(var_list):
        print(f"Debug: extract_names input: {var_list}, type: {type(var_list)}")
        if not var_list:
            return []
        if isinstance(var_list, list):
            if len(var_list) > 0 and isinstance(var_list[0], dict):
                # Format: [{'name': 'gdp_growth'}, {'name': 'inflation'}]
                result = [item['name'] for item in var_list if isinstance(item, dict) and 'name' in item]
                print(f"Debug: extracted from dict format: {result}")
                return result
            else:
                # Format: ['gdp_growth', 'inflation']
                result = [str(item) for item in var_list if not isinstance(item, dict)]
                print(f"Debug: extracted from list format: {result}")
                return result
        elif isinstance(var_list, dict):
            # Maybe it's a single dict?
            if 'name' in var_list:
                return [var_list['name']]
            else:
                print(f"Debug: var_list is dict but no 'name' key: {var_list}")
                return []
        else:
            print(f"Debug: var_list is neither list nor dict: {type(var_list)}")
            return []

    observable_names = extract_names(variables_config.get('observables', variables_config.get('observable', [])))
    trend_names = extract_names(variables_config.get('trends', []))
    stationary_names = extract_names(variables_config.get('stationary', []))

    print(f"Debug: Final extracted observable_names: {observable_names}")
    print(f"Debug: Final extracted trend_names: {trend_names}")
    print(f"Debug: Final extracted stationary_names: {stationary_names}")

    # Validate that all names are strings
    def validate_string_list(name_list, list_name):
        for i, name in enumerate(name_list):
            if not isinstance(name, str):
                print(f"Error: {list_name}[{i}] is not a string: {name} (type: {type(name)})")
                raise TypeError(f"{list_name} contains non-string element: {name}")
        return name_list

    observable_names = validate_string_list(observable_names, "observable_names")
    trend_names = validate_string_list(trend_names, "trend_names")
    stationary_names = validate_string_list(stationary_names, "stationary_names")

    # Calculate dimensions
    k_endog = len(observable_names)
    k_trends = len(trend_names)
    k_stationary = len(stationary_names)
    p = config_dict.get('var_order', 1)
    k_states = k_trends + k_stationary * p

    config_data = {
        'var_order': p,
        'variables': {
            'observable_names': observable_names,
            'trend_names': trend_names,
            'stationary_var_names': stationary_names,
        },
        'model_equations': config_dict.get('model_equations', {}),
        'initial_conditions': config_dict.get('initial_conditions', {}),
        'stationary_prior': config_dict.get('stationary_prior', {}),
        'trend_shocks': config_dict.get('trend_shocks', {}),
        'parameters': config_dict.get('parameters', {'measurement': []}),
        'k_endog': k_endog,
        'k_trends': k_trends,
        'k_stationary': k_stationary,
        'k_states': k_states,
    }

    # Parse initial conditions
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])

    # Fix model_equations format FIRST - convert from list of dicts to dict
    model_equations_raw = config_dict.get('model_equations', {})
    if isinstance(model_equations_raw, list):
        # Convert [{'gdp': 'trend_gdp + cycle_gdp'}, {'inf': 'trend_inf + cycle_inf'}]
        # to {'gdp': 'trend_gdp + cycle_gdp', 'inf': 'trend_inf + cycle_inf'}
        model_equations_dict = {}
        for eq_dict in model_equations_raw:
            if isinstance(eq_dict, dict):
                model_equations_dict.update(eq_dict)
        config_data['model_equations'] = model_equations_dict

    print(f"Debug: Fixed model_equations: {config_data['model_equations']}")

    # Parse model equations - handle the list of dicts format from the YAML
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p]

    parsed_model_eqs_list = []
    raw_model_eqs = config_data['model_equations']  # This is now the converted dict
    observable_indices = {name: i for i, name in enumerate(observable_names)}

    print(f"Debug: About to parse model equations: {raw_model_eqs}")
    print(f"Debug: observable_indices: {observable_indices}")

    if isinstance(raw_model_eqs, dict):
        for obs_name, eq_str in raw_model_eqs.items():
            print(f"Debug: Processing equation {obs_name}: {eq_str}")
            if obs_name in observable_indices:
                try:
                    parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
                    parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))
                    print(f"Debug: Parsed terms for {obs_name}: {parsed_terms}")
                except Exception as parse_error:
                    print(f"Debug: Error parsing equation {obs_name}: {parse_error}")
            else:
                print(f"Debug: Observable {obs_name} not found in indices")

    config_data['model_equations_parsed'] = parsed_model_eqs_list
    print(f"Debug: Final model_equations_parsed: {parsed_model_eqs_list}")

    # Create the parsed_model_eqs_jax_detailed structure that the model expects
    # This is a more detailed format: (obs_idx, [(term_type, state_idx_in_C, param_idx_if_any, sign), ...])
    # where term_type: 0=direct state, 1=parameter term

    # Create state-to-index mapping for C matrix (trends + current stationary)
    c_matrix_state_names = trend_names + stationary_names  # Current states only for C matrix
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names)}

    print(f"Debug: c_matrix_state_names: {c_matrix_state_names}")
    print(f"Debug: state_to_c_idx_map: {state_to_c_idx_map}")

    parsed_model_eqs_jax_detailed = []
    for obs_idx, parsed_terms in parsed_model_eqs_list:
        processed_terms_for_obs = []
        print(f"Debug: Processing obs_idx {obs_idx} with terms: {parsed_terms}")
        for param_name, state_name_in_eq, sign in parsed_terms:
            term_type = 0 if param_name is None else 1  # 0=direct state, 1=parameter term
            if state_name_in_eq in state_to_c_idx_map:
                state_index_in_C = state_to_c_idx_map[state_name_in_eq]
                param_index_if_any = param_to_idx_map[param_name] if param_name is not None else -1
                processed_terms_for_obs.append(
                    (term_type, state_index_in_C, param_index_if_any, float(sign))
                )
                print(f"Debug: Added term: type={term_type}, state_idx={state_index_in_C}, param_idx={param_index_if_any}, sign={sign}")
            else:
                print(f"Debug: State {state_name_in_eq} not found in state_to_c_idx_map")
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))

    config_data['parsed_model_eqs_jax_detailed'] = tuple(parsed_model_eqs_jax_detailed)

    print(f"Debug: parsed_model_eqs_jax_detailed: {config_data['parsed_model_eqs_jax_detailed']}")

    # Identify trend names with shocks - all trend_names should now be validated strings
    trend_shocks_spec = config_data['trend_shocks'].get('trend_shocks', {})

    print(f"Debug: About to check trend shocks with trend_names: {trend_names}")
    print(f"Debug: trend_shocks_spec: {trend_shocks_spec}")

    # Now this should work since all trend_names are verified strings
    config_data['trend_names_with_shocks'] = [
        name for name in trend_names
        if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and 'distribution' in trend_shocks_spec[name]
    ]

    print(f"Debug: trend_names_with_shocks: {config_data['trend_names_with_shocks']}")

    # Add the n_trend_shocks key that the model expects
    config_data['n_trend_shocks'] = len(config_data['trend_names_with_shocks'])

    # Pre-calculate static indices for off-diagonal elements
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows, off_diag_cols)
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)

    # Add additional keys that the model might expect for compatibility
    config_data.update({
        # Tuples of names for model compatibility
        'observable_names': tuple(observable_names),
        'trend_var_names': tuple(trend_names),
        'stationary_var_names': tuple(stationary_names),

        # Raw config sections for model access
        'raw_config_initial_conds': config_data['initial_conditions'],
        'raw_config_stationary_prior': config_data['stationary_prior'],
        'raw_config_trend_shocks': config_data['trend_shocks'],
        'raw_config_measurement_params': measurement_params_config,
        'raw_config_model_eqs_str_dict': config_data['model_equations'],

        # Additional compatibility keys
        'measurement_param_names_tuple': tuple(measurement_param_names),
    })

    # Create initial state flat arrays that the model might expect
    # Helper to create the full list of state names in order
    full_state_names_list = list(trend_names)
    for i in range(p):
        for stat_var in stationary_names:
            if i == 0:
                full_state_names_list.append(stat_var)
            else:
                full_state_names_list.append(f"{stat_var}_t_minus_{i}")

    config_data['full_state_names_tuple'] = tuple(full_state_names_list)

    # Create flat initial condition arrays
    init_x_means_flat_list = []
    init_P_diag_flat_list = []

    initial_conditions_parsed = config_data['initial_conditions_parsed']

    for state_name in full_state_names_list:
        # Check if this state_name is in the parsed initial conditions
        base_name_for_lag = state_name
        if "_t_minus_" in state_name:
            base_name_for_lag = state_name.split("_t_minus_")[0]

        if state_name in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[state_name]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[state_name]['var']))
        elif base_name_for_lag in initial_conditions_parsed and base_name_for_lag != state_name:
            # Lagged state, use current state's values
            init_x_means_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['var']))
        else:
            # Default values
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)

    config_data['init_x_means_flat'] = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)

    print(f"Debug: init_x_means_flat: {config_data['init_x_means_flat']}")
    print(f"Debug: init_P_diag_flat: {config_data['init_P_diag_flat']}")

    return config_data

def define_true_params(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defines a set of true parameters for simulation, matching the model structure."""
    k_trends = len(config_data['variables']['trend_names'])
    k_stationary = len(config_data['variables']['stationary_var_names'])

    p = config_data['var_order']
    trend_names_with_shocks = config_data['trend_names_with_shocks']
    n_trend_shocks = len(trend_names_with_shocks)

    # True VAR(p) coefficient matrices
    if k_stationary == 2 and p >= 1:
        # Initialize Phi_list_true for all p lags
        Phi_list_true = [jnp.zeros((k_stationary, k_stationary), dtype=_DEFAULT_DTYPE) for _ in range(p)]
        # Set the first lag matrix specifically
        Phi_list_true[0] = jnp.array([[0.7, 0.2], [0.1, 0.5]], dtype=_DEFAULT_DTYPE)
        # For any additional lags (if p > 1) with k_stationary == 2, initialize them to a default
        for i in range(1, p):
            Phi_list_true[i] = jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.7 
    else:
        # Original generic logic for other k_stationary values or if p=0 (though p>=1 is typical for VAR)
        Phi_list_true = [jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.7 for _ in range(p)]
        if k_stationary >= 2 and p >= 1:
             # This applies the small off-diagonal elements for k_stationary > 2
             # or if k_stationary == 2 but the specific condition above wasn't met (which is unlikely with this logic but acts as a safe default)
             Phi_list_true[0] = Phi_list_true[0].at[0, 1].set(0.2)
             Phi_list_true[0] = Phi_list_true[0].at[1, 0].set(0.1)
        # Note: For p > 1 in this 'else' branch, Phi_list_true[i] for i > 0 are already 0.7*eye
        # due to the list comprehension, which is a reasonable default.

    # Check stationarity
    try:
        T_check = create_companion_matrix_jax(Phi_list_true, p, k_stationary)
        eigenvalues = jnp.linalg.eigvals(T_check)
        max_abs_eig = jnp.max(jnp.abs(eigenvalues))
        if max_abs_eig >= 1.0 - 1e-6:
            print(f"Warning: True Phi_list results in non-stationary companion matrix (max abs eig: {max_abs_eig}).")
            Phi_list_true = [jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.5 for _ in range(p)]
            if k_stationary >= 2 and p >= 1:
                 Phi_list_true[0] = Phi_list_true[0].at[0, 1].set(0.1)
                 Phi_list_true[0] = Phi_list_true[0].at[1, 0].set(0.05)
            T_check_adj = create_companion_matrix_jax(Phi_list_true, p, k_stationary)
            print(f"Adjusted true Phi_list. Max abs eig: {jnp.max(jnp.abs(jnp.linalg.eigvals(T_check_adj)))}")
    except Exception as e:
         print(f"Warning: Could not check true Phi stationarity: {e}")

    # True stationary cycle shock covariance
    Sigma_cycles_true = jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.5
    if k_stationary >= 2:
        Sigma_cycles_true = Sigma_cycles_true.at[0, 1].set(0.1)
        Sigma_cycles_true = Sigma_cycles_true.at[1, 0].set(0.1)
    
    Sigma_cycles_true = (Sigma_cycles_true + Sigma_cycles_true.T) / 2.0
    Sigma_cycles_true = Sigma_cycles_true  # Scale down to ensure stationarity
    # True trend shock variances
    true_trend_vars_dict_sim = {
        # Increase the variance values significantly
        'trend_gdp': 0.1**2,  # Example: Standard deviation 0.2 (variance 0.04)
        'trend_inf': 0.2**2,  # Example: Standard deviation 0.3 (variance 0.09)
        # You can experiment with these values. Make them large enough
        # that the trend path is clearly not constant over 100 steps.
        # The original values were 0.05**2=0.0025 and 0.0707**2=0.005
    }

    true_trend_vars_with_shocks_sim = jnp.array([
         true_trend_vars_dict_sim.get(name, 0.01) for name in trend_names_with_shocks
    ], dtype=_DEFAULT_DTYPE)
    Sigma_trends_sim_true = jnp.diag(true_trend_vars_with_shocks_sim) if n_trend_shocks > 0 else jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    # True measurement parameters
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p]
    true_measurement_params = {name: 1.0 for name in measurement_param_names}

    return {
        'Phi_list': Phi_list_true,
        'Sigma_cycles': Sigma_cycles_true,
        'Sigma_trends_sim': Sigma_trends_sim_true,
        'measurement_params': true_measurement_params,
    }

# --- Main Script ---
yaml_file_path = 'bvar_stationary_sim.yml'

# Create sample config if it doesn't exist
if not os.path.exists(yaml_file_path):
    print(f"Configuration file {yaml_file_path} not found. Creating sample configuration...")
    create_sample_config()

# --- 1. Load Configuration and Parse ---
print("Loading configuration and parsing...")
try:
    config_data = load_config_simple(yaml_file_path)
    print("Configuration loaded and parsed.")
    print(f"Found {len(config_data['variables']['observable_names'])} observables: {config_data['variables']['observable_names']}")
    print(f"Found {len(config_data['variables']['trend_names'])} trends: {config_data['variables']['trend_names']}")
    print(f"Found {len(config_data['variables']['stationary_var_names'])} stationary variables: {config_data['variables']['stationary_var_names']}")

except Exception as e:
    print(f"Error loading or parsing configuration: {e}")
    raise

# Extract dimensions and names from parsed config
observable_names = config_data['variables']['observable_names']
trend_var_names = config_data['variables']['trend_names']
stationary_var_names = config_data['variables']['stationary_var_names']
k_endog = config_data['k_endog']
k_trends = config_data['k_trends']
k_stationary = config_data['k_stationary']
p = config_data['var_order']
k_states = config_data['k_states']

# Identify trends with shocks defined in YAML config
trend_names_with_shocks_in_config = config_data['trend_names_with_shocks']
n_trend_shocks_config = len(trend_names_with_shocks_in_config)
n_shocks_model = n_trend_shocks_config + k_stationary

print(f"Dimensions: k_endog={k_endog}, k_trends={k_trends}, k_stationary={k_stationary}, p={p}")
print(f"Total state dimension: k_states={k_states}")
print(f"Trends with shocks: {trend_names_with_shocks_in_config}")

# --- 2. Define True Parameters and Simulate Data ---
print("\nDefining true parameters and simulating data...")
key = random.PRNGKey(0)
key_true_params, key_sim, key_mcmc, key_smooth_draws = random.split(key, 4)

T_sim = 1000

# Define true parameters
true_params = define_true_params(config_data)

# Simulate data
print(f"Simulating {T_sim} steps with true parameters...")
y_simulated_jax, true_states_sim, true_cycles_sim, true_trends_sim = simulate_bvar_with_trends_jax(
    key_sim,
    T_sim,
    config_data,
    true_params['Phi_list'],
    true_params['Sigma_cycles'],
    true_params['Sigma_trends_sim'],
    true_params['measurement_params'],
)
print("Data simulation complete.")
print(f"Simulated data shape: {y_simulated_jax.shape}")
print(f"True states shape: {true_states_sim.shape}")

# Create dummy pandas DataFrame for plotting
dummy_dates = pd.period_range(start='1800Q1', periods=T_sim, freq='Q').to_timestamp()
y_simulated_pd = pd.DataFrame(y_simulated_jax, index=dummy_dates, columns=observable_names)

# Identify static NaN handling parameters
valid_obs_mask_cols = jnp.any(jnp.isfinite(y_simulated_jax), axis=0)
static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
static_n_obs_actual = static_valid_obs_idx.shape[0]

if static_n_obs_actual == 0:
    print("Warning: Simulated data resulted in no observed series.")
    raise ValueError("Cannot run MCMC with no observed series.")

# --- 3. Setup and Run MCMC ---
print("\nSetting up MCMC...")

# Create a simple model arguments structure
model_args = {
    'y': y_simulated_jax,
    'config_data': config_data,  # Use our simple config
    'static_valid_obs_idx': static_valid_obs_idx,
    'static_n_obs_actual': static_n_obs_actual,
    'trend_var_names': trend_var_names,
    'stationary_var_names': stationary_var_names,
    'observable_names': observable_names,
}

init_strategy = numpyro.infer.init_to_sample()
kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=init_strategy)

num_warmup = 50
num_samples = 100
num_chains = 2

print(f"Running MCMC with {num_warmup} warmup steps and {num_samples} sampling steps per chain ({num_chains} chains)...")
start_time_mcmc = time.time()

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

try:
    mcmc.run(key_mcmc, **model_args)
    end_time_mcmc = time.time()
    print(f"MCMC completed in {end_time_mcmc - start_time_mcmc:.2f} seconds.")
    mcmc.print_summary()

    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    phi_list_samples = posterior_samples['phi_list']

    print("MCMC successful! Proceeding to simulation smoother...")

except Exception as mcmc_error:
    print(f"MCMC failed with error: {mcmc_error}")
    print("This might be due to:")
    print("1. Configuration structure incompatibility")
    print("2. JAX tracing issues in the model")
    print("3. Missing required keys in config_data")

    # Print available keys for debugging
    print(f"Available config_data keys: {sorted(config_data.keys())}")

    # Try to identify which specific keys the model is looking for
    try:
        print("Attempting to call the model function directly for debugging...")
        # This won't work but might give us more specific error info
        test_result = numpyro_bvar_stationary_model(**model_args)
    except Exception as model_error:
        print(f"Direct model call failed with: {model_error}")

    print("Script finished with MCMC error.")
    raise mcmc_error

# If we get here, MCMC was successful
try:
    print("\nRunning single-draw simulation smoother with posterior mean parameters...")

    # Get posterior mean for sampled parameters
    sampled_param_names = ['A_diag']
    if config_data['num_off_diag'] > 0:
        sampled_param_names.append('A_offdiag')
    if k_stationary > 1:
        sampled_param_names.append('stationary_chol')

    # Add stationary variances
    sampled_param_names.extend([f'stationary_var_{name}' for name in stationary_var_names])

    # Add trend variances
    sampled_param_names.extend([f'trend_var_{name}' for name in trend_names_with_shocks_in_config])

    # Add measurement parameters
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    sampled_param_names.extend([p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p])

    posterior_mean_params = {
        name: jnp.mean(posterior_samples[name], axis=0)
        for name in sampled_param_names
        if name in posterior_samples
    }

    # Run the single-draw smoother routine
    key_smooth_draws = random.PRNGKey(2)
    num_sim_draws = 50
    print(f"Running simulation smoother for {num_sim_draws} draws...")

    start_time_smooth_draws = time.time()

    # Convert model_equations_parsed to hashable format
    model_eqs_hashable = convert_to_hashable(config_data['model_equations_parsed'])
    
    # Convert measurement_params_config to hashable format
    measurement_params_hashable = convert_to_hashable(measurement_params_config)

    # Convert initial conditions to hashable format
    initial_conds_tuple = tuple(
        (state_name, float(state_config['mean']), float(state_config['var']))
        for state_name, state_config in config_data['initial_conditions_parsed'].items()
    )

    static_smoother_args = {
        'static_k_endog': k_endog,
        'static_k_trends': k_trends,
        'static_k_stationary': k_stationary,
        'static_p': p,
        'static_k_states': k_states,
        'static_n_trend_shocks': n_trend_shocks_config,
        'static_n_shocks_state': n_shocks_model,
        'static_num_off_diag': config_data['num_off_diag'],
        'static_off_diag_rows': config_data['static_off_diag_indices'][0],
        'static_off_diag_cols': config_data['static_off_diag_indices'][1],
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'model_eqs_parsed': model_eqs_hashable,
        'initial_conds_parsed': initial_conds_tuple,
        'trend_names_with_shocks': tuple(trend_names_with_shocks_in_config),
        'stationary_var_names': tuple(stationary_var_names),
        'trend_var_names': tuple(trend_var_names),
        'measurement_params_config': measurement_params_hashable,
        'num_draws': num_sim_draws,
    }

    try:
        print(f"Debug: About to call run_simulation_smoother_single_params_jit")
        print(f"Debug: static_smoother_args keys: {list(static_smoother_args.keys())}")

        smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
            posterior_mean_params,
            y_simulated_jax,
            key_smooth_draws,
            **static_smoother_args
        )

        end_time_smooth_draws = time.time()
        print(f"Simulation smoother draws completed in {end_time_smooth_draws - start_time_smooth_draws:.2f} seconds.")

        # --- 4. Process Smoother Results and Plot ---
        # The smoother returns either (mean_smooth, cov_smooth) if num_draws=1 (original smoother)
        # or (mean_smooth, (mean_sim, median_sim, all_draws)) if num_draws > 1 (simulation smoother)
        if num_sim_draws > 1:
            mean_sim_states, median_sim_states, all_sim_draws = simulation_results
            print(f"Smoothed states shape: {smoothed_states_original.shape}")
            print(f"Mean simulated states shape: {mean_sim_states.shape}")

            # Check for NaNs
            print(f"NaNs in smoothed_states_original: {jnp.any(jnp.isnan(smoothed_states_original))}")
            print(f"NaNs in mean_sim_states: {jnp.any(jnp.isnan(mean_sim_states))}")

            # --- 5. Generate Comparison Plots ---
            print("\nGenerating comparison plots...")

            # Get state names for plotting
            state_names = trend_var_names + [
                f"{name}_t_minus_{lag}" if lag > 0 else name
                for lag in range(p) for name in stationary_var_names
            ]
            if p == 1:
                state_names = trend_var_names + stationary_var_names

            try:
                num_states_to_plot = k_trends + k_stationary
                fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(12, 3 * num_states_to_plot), sharex=True)
                if num_states_to_plot == 1:
                    axes = [axes]

                dates = y_simulated_pd.index
                formatter = mdates.DateFormatter('%Y')

                for i in range(num_states_to_plot):
                    ax = axes[i]
                    state_name = state_names[i]

                    # Determine true path
                    if i < k_trends:
                        true_path = true_trends_sim[:, i]
                        component_type = "True Trend"
                    elif i < k_trends + k_stationary:
                        true_path = true_cycles_sim[:, i - k_trends]
                        component_type = "True Cycle"

                    # Plot true state path
                    ax.plot(dates, true_path, label=component_type, color='black', linestyle='-', linewidth=2, alpha=0.7)

                    # Plot smoothed state mean (from original filter/smoother - usually Kalman)
                    ax.plot(dates, smoothed_states_original[:, i], label='Smoothed State Mean (Kalman)', color='blue', linestyle='--')

                    # Plot mean simulated states (from simulation smoother draws)
                    ax.plot(dates, mean_sim_states[:, i], label='Mean Sim State (Est)', color='red', linestyle=':')

                    # Plot simulation smoother band using percentiles
                    try:
                        # all_sim_draws shape is (num_draws, T, k_states)
                        # state_draws_for_percentiles shape: (num_draws, T)
                        state_draws_for_percentiles = all_sim_draws[:, :, i]
                        
                        lower_band = jnp.percentile(state_draws_for_percentiles, 5, axis=0)
                        upper_band = jnp.percentile(state_draws_for_percentiles, 95, axis=0)
                        
                        ax.fill_between(dates, lower_band, upper_band,
                                       color='red', alpha=0.2, label='90% Sim Smoother Band (Percentiles)')

                    except Exception as percentile_e:
                        print(f"Warning: Could not compute percentiles for state {state_name}: {percentile_e}")
                        # Fallback to min/max if percentile calculation fails
                        try:
                            min_draw = jnp.min(all_sim_draws[:, :, i], axis=0)
                            max_draw = jnp.max(all_sim_draws[:, :, i], axis=0)
                            ax.fill_between(dates, min_draw, max_draw, color='red', alpha=0.1, label='Sim Smoother Min/Max')
                        except Exception as min_max_e:
                            print(f"Warning: Min/Max fallback also failed for state {state_name}: {min_max_e}")

                    ax.set_title(f'True vs Estimated State: {state_name}')
                    ax.legend(fontsize=8)
                    ax.grid(True)
                    ax.xaxis.set_major_formatter(formatter)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error during plotting: {e}")

        else: # Case where num_sim_draws == 1 (standard smoother output)
            mean_smooth_states, cov_smooth_states = simulation_results
            print(f"Smoothed states shape: {smoothed_states_original.shape}") # Should be the same as mean_smooth_states
            print(f"Smoothed covariance shape: {cov_smooth_states.shape}")

            print("Single draw simulation smoother completed successfully.")


    except Exception as smoother_error:
        print(f"Error running simulation smoother: {smoother_error}")
        print("This might be due to configuration mismatch or missing data structures.")

except Exception as e:
    print(f"Error in post-MCMC processing: {e}")

print("\nScript finished.")