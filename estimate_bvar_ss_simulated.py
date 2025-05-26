# estimate_bvar_ss_improved.py
# Improved simulation script with all the identified fixes applied

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import time
import os
import yaml
from typing import Dict, Any, List, Tuple

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64

try:
    numpyro.set_host_device_count(2)
except:
    pass

# Import necessary modules
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax
from core.simulate_bvar_jax import simulate_bvar_with_trends_jax
from core.var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices
from core.run_single_draw import run_simulation_smoother_single_params_jit


# from core.complete_dls_prior_elicitation import (
#     data_driven_prior_elicitation, 
#     create_yaml_config_from_priors,
#     print_prior_summary
# )

def convert_to_hashable(obj):
    """Recursively convert lists to tuples to make objects hashable for JAX JIT."""
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in obj.items()))
    else:
        return obj

def create_improved_config():
    """Create an improved configuration that should work better."""
    config = {
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
                'trend_gdp': {'mean': 2.0, 'var': 0.1},
                'trend_inf': {'mean': 1.5, 'var': 0.1},
                'cycle_gdp': {'mean': 0.0, 'var': 0.25},
                'cycle_inf': {'mean': 0.0, 'var': 0.25}
            }
        },
        'stationary_prior': {
            'hyperparameters': {
                'es': [0.7, 0.15],
                'fs': [0.2, 0.15]
            },
            'covariance_prior': {
                'eta': 1.5
            },
            'stationary_shocks': {
                'cycle_gdp': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 3.0, 'beta': 1.0}
                },
                'cycle_inf': {
                    'distribution': 'inverse_gamma', 
                    'parameters': {'alpha': 3.0, 'beta': 1.0}
                }
            }
        },
        'trend_shocks': {
            'trend_shocks': {
                'trend_gdp': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 3.0, 'beta': 0.02}
                },
                'trend_inf': {
                    'distribution': 'inverse_gamma',
                    'parameters': {'alpha': 3.0, 'beta': 0.02}
                }
            }
        },
        'parameters': {
            'measurement': []
        }
    }
    
    with open('bvar_improved.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created improved configuration: bvar_improved.yml")

def load_config_simple(yaml_path: str) -> Dict[str, Any]:
    """Simple config loader."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract variables
    variables_config = config_dict.get('variables', {})
    
    def extract_names(var_list):
        if not var_list:
            return []
        if isinstance(var_list, list):
            return [str(item) for item in var_list if isinstance(item, str)]
        return []

    observable_names = extract_names(variables_config.get('observables', []))
    trend_names = extract_names(variables_config.get('trends', []))
    stationary_names = extract_names(variables_config.get('stationary', []))

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

    #Using prior elicitation 
    config_data = create_config_with_dls_priors(
       data=your_data,
        variable_names=['gdp_growth', 'inflation'],
        training_fraction=0.25
    )
    # Parse initial conditions
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])

    # Parse model equations
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p]

    parsed_model_eqs_list = []
    raw_model_eqs = config_data['model_equations']
    observable_indices = {name: i for i, name in enumerate(observable_names)}

    if isinstance(raw_model_eqs, dict):
        for obs_name, eq_str in raw_model_eqs.items():
            if obs_name in observable_indices:
                parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
                parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))

    config_data['model_equations_parsed'] = parsed_model_eqs_list

    # Create detailed parsing for JAX
    c_matrix_state_names = trend_names + stationary_names
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names)}

    parsed_model_eqs_jax_detailed = []
    for obs_idx, parsed_terms in parsed_model_eqs_list:
        processed_terms_for_obs = []
        for param_name, state_name_in_eq, sign in parsed_terms:
            term_type = 0 if param_name is None else 1
            if state_name_in_eq in state_to_c_idx_map:
                state_index_in_C = state_to_c_idx_map[state_name_in_eq]
                param_index_if_any = param_to_idx_map[param_name] if param_name is not None else -1
                processed_terms_for_obs.append(
                    (term_type, state_index_in_C, param_index_if_any, float(sign))
                )
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))

    config_data['parsed_model_eqs_jax_detailed'] = tuple(parsed_model_eqs_jax_detailed)

    # Identify trend names with shocks
    trend_shocks_spec = config_data['trend_shocks'].get('trend_shocks', {})
    config_data['trend_names_with_shocks'] = [
        name for name in trend_names
        if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and 'distribution' in trend_shocks_spec[name]
    ]

    config_data['n_trend_shocks'] = len(config_data['trend_names_with_shocks'])

    # Pre-calculate static indices
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows, off_diag_cols)
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)

    # Add compatibility keys
    config_data.update({
        'observable_names': tuple(observable_names),
        'trend_var_names': tuple(trend_names),
        'stationary_var_names': tuple(stationary_names),
        'raw_config_initial_conds': config_data['initial_conditions'],
        'raw_config_stationary_prior': config_data['stationary_prior'],
        'raw_config_trend_shocks': config_data['trend_shocks'],
        'raw_config_measurement_params': measurement_params_config,
        'raw_config_model_eqs_str_dict': config_data['model_equations'],
        'measurement_param_names_tuple': tuple(measurement_param_names),
    })

    # Create flat initial condition arrays
    full_state_names_list = list(trend_names)
    for i in range(p):
        for stat_var in stationary_names:
            if i == 0:
                full_state_names_list.append(stat_var)
            else:
                full_state_names_list.append(f"{stat_var}_t_minus_{i}")

    config_data['full_state_names_tuple'] = tuple(full_state_names_list)

    init_x_means_flat_list = []
    init_P_diag_flat_list = []
    initial_conditions_parsed = config_data['initial_conditions_parsed']

    for state_name in full_state_names_list:
        base_name_for_lag = state_name
        if "_t_minus_" in state_name:
            base_name_for_lag = state_name.split("_t_minus_")[0]

        if state_name in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[state_name]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[state_name]['var']))
        elif base_name_for_lag in initial_conditions_parsed and base_name_for_lag != state_name:
            init_x_means_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['var']))
        else:
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)

    config_data['init_x_means_flat'] = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)

    return config_data

def define_improved_true_params(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Define more realistic true parameters that should be more identifiable."""
    k_trends = len(config_data['variables']['trend_names'])
    k_stationary = len(config_data['variables']['stationary_var_names'])
    p = config_data['var_order']
    trend_names_with_shocks = config_data['trend_names_with_shocks']

    # More persistent but stationary VAR coefficients
    if k_stationary == 2 and p >= 1:
        Phi_list_true = [jnp.array([[0.8, 0.1], [0.05, 0.75]], dtype=_DEFAULT_DTYPE)]
        for i in range(1, p):
            Phi_list_true.append(jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.1)
    else:
        Phi_list_true = [jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.8 for _ in range(p)]

    # Check stationarity
    try:
        T_check = create_companion_matrix_jax(Phi_list_true, p, k_stationary)
        eigenvalues = jnp.linalg.eigvals(T_check)
        max_abs_eig = jnp.max(jnp.abs(eigenvalues))
        print(f"True VAR eigenvalues max: {max_abs_eig:.4f}")
        if max_abs_eig >= 0.99:
            print("Warning: VAR coefficients too close to unit root, reducing...")
            Phi_list_true[0] = Phi_list_true[0] * 0.9
    except Exception as e:
        print(f"Warning: Could not check stationarity: {e}")

    # More realistic cycle shock covariance (higher variance, moderate correlation)
    Sigma_cycles_true = jnp.array([[0.8, 0.2], [0.2, 0.6]], dtype=_DEFAULT_DTYPE)
    
    # Ensure positive definiteness
    eigenvals_cycle = jnp.linalg.eigvals(Sigma_cycles_true)
    if jnp.min(eigenvals_cycle) <= 0:
        print("Warning: Adjusting cycle covariance for positive definiteness")
        Sigma_cycles_true = Sigma_cycles_true + 0.1 * jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)

    # More visible trend shock variances (but still small)
    true_trend_vars_dict = {
        'trend_gdp': 0.005,  # Std dev = 0.071
        'trend_inf': 0.003,  # Std dev = 0.055
    }

    true_trend_vars_with_shocks = jnp.array([
        true_trend_vars_dict.get(name, 0.002) for name in trend_names_with_shocks
    ], dtype=_DEFAULT_DTYPE)
    
    Sigma_trends_sim_true = jnp.diag(true_trend_vars_with_shocks) if len(trend_names_with_shocks) > 0 else jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    # No measurement parameters in this setup
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p]
    true_measurement_params = {name: 1.0 for name in measurement_param_names}

    print(f"True Phi matrix: {Phi_list_true[0]}")
    print(f"True Sigma_cycles: {Sigma_cycles_true}")
    print(f"True trend variances: {true_trend_vars_dict}")

    return {
        'Phi_list': Phi_list_true,
        'Sigma_cycles': Sigma_cycles_true,
        'Sigma_trends_sim': Sigma_trends_sim_true,
        'measurement_params': true_measurement_params,
    }

# Main execution
def main():
    yaml_file_path = 'bvar_improved.yml'

    # Create improved config if it doesn't exist
    if not os.path.exists(yaml_file_path):
        print(f"Configuration file {yaml_file_path} not found. Creating improved configuration...")
        create_improved_config()

    # Load configuration
    print("Loading configuration...")
    config_data = load_config_simple(yaml_file_path)
    print("Configuration loaded successfully.")

    # Extract dimensions
    observable_names = config_data['variables']['observable_names']
    trend_var_names = config_data['variables']['trend_names']
    stationary_var_names = config_data['variables']['stationary_var_names']
    k_endog = config_data['k_endog']
    k_trends = config_data['k_trends']
    k_stationary = config_data['k_stationary']
    p = config_data['var_order']
    k_states = config_data['k_states']

    print(f"Dimensions: k_endog={k_endog}, k_trends={k_trends}, k_stationary={k_stationary}, p={p}")
    print(f"Total state dimension: k_states={k_states}")

    # Define true parameters and simulate data
    print("\nDefining true parameters and simulating data...")
    key = random.PRNGKey(42)  # Fixed seed for reproducibility
    key_sim, key_mcmc, key_smooth = random.split(key, 3)

    T_sim = 300  # Longer time series for better identification

    true_params = define_improved_true_params(config_data)

    # Simulate data
    print(f"Simulating {T_sim} time steps...")
    y_simulated_jax, true_states_sim, true_cycles_sim, true_trends_sim = simulate_bvar_with_trends_jax(
        key_sim,
        T_sim,
        config_data,
        true_params['Phi_list'],
        true_params['Sigma_cycles'],
        true_params['Sigma_trends_sim'],
        true_params['measurement_params'],
    )

    print(f"Simulated data shape: {y_simulated_jax.shape}")
    print(f"Data ranges - GDP: [{jnp.min(y_simulated_jax[:, 0]):.3f}, {jnp.max(y_simulated_jax[:, 0]):.3f}]")
    print(f"Data ranges - Inflation: [{jnp.min(y_simulated_jax[:, 1]):.3f}, {jnp.max(y_simulated_jax[:, 1]):.3f}]")

    # Create DataFrame for plotting
    dummy_dates = pd.period_range(start='1950Q1', periods=T_sim, freq='Q').to_timestamp()
    y_simulated_pd = pd.DataFrame(y_simulated_jax, index=dummy_dates, columns=observable_names)

    # Static observation handling
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y_simulated_jax), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]

    if static_n_obs_actual == 0:
        raise ValueError("No observed series found.")

    # Setup and run MCMC
    print("\nSetting up MCMC...")
    model_args = {
        'y': y_simulated_jax,
        'config_data': config_data,
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'trend_var_names': trend_var_names,
        'stationary_var_names': stationary_var_names,
        'observable_names': observable_names,
    }

    kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=numpyro.infer.init_to_sample())
    
    num_warmup = 200  # More warmup for better convergence
    num_samples = 300  # More samples
    num_chains = 2

    print(f"Running MCMC with {num_warmup} warmup and {num_samples} samples per chain...")
    start_time = time.time()

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(key_mcmc, **model_args)
    
    end_time = time.time()
    print(f"MCMC completed in {end_time - start_time:.2f} seconds.")
    mcmc.print_summary()

    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    # Run simulation smoother
    print("\nRunning simulation smoother...")
    sampled_param_names = ['A_diag']
    if config_data['num_off_diag'] > 0:
        sampled_param_names.append('A_offdiag')
    if k_stationary > 1:
        sampled_param_names.append('stationary_chol')

    sampled_param_names.extend([f'stationary_var_{name}' for name in stationary_var_names])
    sampled_param_names.extend([f'trend_var_{name}' for name in config_data['trend_names_with_shocks']])

    posterior_mean_params = {
        name: jnp.mean(posterior_samples[name], axis=0)
        for name in sampled_param_names
        if name in posterior_samples
    }

    # Convert to hashable formats
    model_eqs_hashable = convert_to_hashable(config_data['model_equations_parsed'])
    measurement_params_hashable = convert_to_hashable([])
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
        'static_n_trend_shocks': config_data['n_trend_shocks'],
        'static_n_shocks_state': config_data['n_trend_shocks'] + k_stationary,
        'static_num_off_diag': config_data['num_off_diag'],
        'static_off_diag_rows': config_data['static_off_diag_indices'][0],
        'static_off_diag_cols': config_data['static_off_diag_indices'][1],
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'model_eqs_parsed': model_eqs_hashable,
        'initial_conds_parsed': initial_conds_tuple,
        'trend_names_with_shocks': tuple(config_data['trend_names_with_shocks']),
        'stationary_var_names': tuple(stationary_var_names),
        'trend_var_names': tuple(trend_var_names),
        'measurement_params_config': measurement_params_hashable,
        'num_draws': 100,  # More draws for better statistics
    }

    num_sim_draws = 100
    print(f"Running simulation smoother for {num_sim_draws} draws...")
    
    smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
        posterior_mean_params,
        y_simulated_jax,
        key_smooth,
        **static_smoother_args
    )

    print("Simulation smoother completed successfully.")

    # Process results and create plots
    if num_sim_draws > 1:
        mean_sim_states, median_sim_states, all_sim_draws = simulation_results
        
        print(f"Smoothed states shape: {smoothed_states_original.shape}")
        print(f"Mean simulated states shape: {mean_sim_states.shape}")
        print(f"All simulation draws shape: {all_sim_draws.shape}")

        # Generate plots
        print("\nGenerating comparison plots...")
        state_names = list(trend_var_names) + list(stationary_var_names)
        num_states_to_plot = k_trends + k_stationary
        
        fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(15, 4 * num_states_to_plot), sharex=True)
        if num_states_to_plot == 1:
            axes = [axes]

        dates = y_simulated_pd.index

        for i in range(num_states_to_plot):
            ax = axes[i]
            state_name = state_names[i]

            # True path
            if i < k_trends:
                true_path = true_trends_sim[:, i]
                component_type = "True Trend"
            else:
                true_path = true_cycles_sim[:, i - k_trends]
                component_type = "True Cycle"

            # Plot lines
            ax.plot(dates, true_path, label=component_type, color='black', linewidth=2, alpha=0.8)
            ax.plot(dates, smoothed_states_original[:, i], label='Smoothed (Kalman)', color='blue', linestyle='--', alpha=0.8)
            ax.plot(dates, mean_sim_states[:, i], label='Mean Simulation', color='red', linestyle=':', alpha=0.8)

            # Confidence bands
            try:
                state_draws = all_sim_draws[:, :, i]
                lower_band = jnp.percentile(state_draws, 10, axis=0)
                upper_band = jnp.percentile(state_draws, 90, axis=0)
                
                ax.fill_between(dates, lower_band, upper_band,
                               color='red', alpha=0.2, label='80% Simulation Band')
            except Exception as e:
                print(f"Warning: Could not compute bands for {state_name}: {e}")

            ax.set_title(f'True vs Estimated: {state_name}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print some summary statistics
        print("\n=== ESTIMATION SUMMARY ===")
        print(f"True Phi[0,0]: {true_params['Phi_list'][0][0,0]:.3f}, Estimated: {posterior_mean_params['A_diag'][0,0]:.3f}")
        print(f"True Phi[1,1]: {true_params['Phi_list'][0][1,1]:.3f}, Estimated: {posterior_mean_params['A_diag'][0,1]:.3f}")
        
        if 'A_offdiag' in posterior_mean_params:
            print(f"True Phi[0,1]: {true_params['Phi_list'][0][0,1]:.3f}, Estimated: {posterior_mean_params['A_offdiag'][0,0]:.3f}")
            print(f"True Phi[1,0]: {true_params['Phi_list'][0][1,0]:.3f}, Estimated: {posterior_mean_params['A_offdiag'][0,1]:.3f}")
        
        true_cycle_vars = jnp.diag(true_params['Sigma_cycles'])
        est_cycle_vars = [posterior_mean_params[f'stationary_var_{name}'] for name in stationary_var_names]
        
        print(f"True cycle variances: {true_cycle_vars}")
        print(f"Estimated cycle variances: {est_cycle_vars}")
        
        true_trend_vars = jnp.diag(true_params['Sigma_trends_sim'])
        est_trend_vars = [posterior_mean_params[f'trend_var_{name}'] for name in config_data['trend_names_with_shocks']]
        
        print(f"True trend variances: {true_trend_vars}")
        print(f"Estimated trend variances: {est_trend_vars}")

    else:
        print("Single draw completed.")

    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()