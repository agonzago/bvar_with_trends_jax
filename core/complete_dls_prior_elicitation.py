# debug_dls_estimation.py
# Standalone version with DLS functions and fixed naming

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import yaml
from typing import Dict, Any, List, Tuple, Optional

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") 
_DEFAULT_DTYPE = jnp.float64

try:
    numpyro.set_host_device_count(2)
except:
    pass

# Import your existing modules (adjust paths as needed)
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax
from core.simulate_bvar_jax import simulate_bvar_with_trends_jax
from core.var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices
from core.run_single_draw import run_simulation_smoother_single_params_jit

# Simple DLS implementation (embedded here to avoid import issues)
def simple_dls_trend_cycle(y: np.ndarray, discount_factor: float = 0.98) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simplified DLS for trend-cycle decomposition.
    
    Returns:
        trend, cycle, trend_variance, cycle_variance
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Simple exponential smoothing for trend
    trend = np.zeros(T)
    trend[0] = y[0]
    
    alpha = 1 - discount_factor
    for t in range(1, T):
        trend[t] = discount_factor * trend[t-1] + alpha * y[t]
    
    # Cycle is residual
    cycle = y - trend
    
    # Estimate variances
    trend_diff = np.diff(trend)
    trend_variance = np.var(trend_diff) if len(trend_diff) > 1 else 0.01
    cycle_variance = np.var(cycle) if len(cycle) > 1 else 0.1
    
    # Ensure positive variances
    trend_variance = max(trend_variance, 1e-6)
    cycle_variance = max(cycle_variance, 1e-6)
    
    return trend, cycle, trend_variance, cycle_variance

def suggest_ig_priors(empirical_var: float, alpha: float = 2.5) -> Dict[str, float]:
    """Simple Inverse Gamma prior suggestion."""
    beta = empirical_var * (alpha - 1)
    return {
        'alpha': alpha,
        'beta': beta,
        'implied_mean': beta / (alpha - 1),
        'implied_variance': beta**2 / ((alpha - 1)**2 * (alpha - 2)) if alpha > 2 else np.inf
    }

def create_dls_config(data: pd.DataFrame, 
                     variable_names: List[str],
                     training_fraction: float = 0.3) -> Dict[str, Any]:
    """
    Create BVAR config with DLS-derived priors.
    Fixed naming to match what the model expects.
    """
    
    # Use subset of data for prior elicitation
    n_train = int(training_fraction * len(data))
    train_data = data.iloc[:n_train]
    
    print(f"Using {n_train} observations for prior elicitation")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    
    # Apply DLS to each variable
    dls_results = {}
    for var_name in variable_names:
        y = train_data[var_name].values
        y = y[~np.isnan(y)]  # Remove NaNs
        
        if len(y) < 10:
            print(f"Warning: Too few observations for {var_name}")
            continue
            
        trend, cycle, trend_var, cycle_var = simple_dls_trend_cycle(y, discount_factor=0.98)
        
        # Create prior suggestions
        trend_prior = suggest_ig_priors(trend_var, alpha=2.5)
        cycle_prior = suggest_ig_priors(cycle_var, alpha=2.5)
        
        dls_results[var_name] = {
            'trend_prior': trend_prior,
            'cycle_prior': cycle_prior,
            'initial_level': np.mean(y[:5]) if len(y) >= 5 else y[0],
            'diagnostics': {'trend': trend, 'cycle': cycle}
        }
        
        print(f"{var_name}: trend_var={trend_var:.6f}, cycle_var={cycle_var:.6f}")
    
    # Create configuration with CONSISTENT naming
    config_data = {
        'var_order': 1,
        'variables': {
            'observable_names': variable_names,
            'trend_names': [f'trend_{name}' for name in variable_names],  # Full names
            'stationary_var_names': [f'cycle_{name}' for name in variable_names],  # Full names
        },
        'model_equations': {},
        'initial_conditions': {'states': {}},
        'stationary_prior': {
            'hyperparameters': {'es': [0.7, 0.15], 'fs': [0.2, 0.15]},
            'covariance_prior': {'eta': 1.5},
            'stationary_shocks': {}
        },
        'trend_shocks': {'trend_shocks': {}},
        'parameters': {'measurement': []},
    }
    
    # Calculate dimensions
    k_endog = len(variable_names)
    k_trends = len(variable_names)
    k_stationary = len(variable_names)
    k_states = k_trends + k_stationary
    
    config_data.update({
        'k_endog': k_endog,
        'k_trends': k_trends, 
        'k_stationary': k_stationary,
        'k_states': k_states,
    })
    
    # Fill in model equations
    for var_name in variable_names:
        config_data['model_equations'][var_name] = f'trend_{var_name} + cycle_{var_name}'
    
    # Fill in DLS-derived priors
    for var_name in variable_names:
        if var_name in dls_results:
            result = dls_results[var_name]
            
            # Initial conditions
            config_data['initial_conditions']['states'][f'trend_{var_name}'] = {
                'mean': float(result['initial_level']),
                'var': 0.1
            }
            config_data['initial_conditions']['states'][f'cycle_{var_name}'] = {
                'mean': 0.0,
                'var': float(result['cycle_prior']['implied_mean'])
            }
            
            # Cycle shock priors 
            config_data['stationary_prior']['stationary_shocks'][f'cycle_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': float(result['cycle_prior']['alpha']),
                    'beta': float(result['cycle_prior']['beta'])
                }
            }
            
            # Trend shock priors
            config_data['trend_shocks']['trend_shocks'][f'trend_{var_name}'] = {
                'distribution': 'inverse_gamma', 
                'parameters': {
                    'alpha': float(result['trend_prior']['alpha']),
                    'beta': float(result['trend_prior']['beta'])
                }
            }
    
    # Parse initial conditions
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])
    
    # Parse model equations
    measurement_params_config = []
    measurement_param_names = []
    
    parsed_model_eqs_list = []
    observable_indices = {name: i for i, name in enumerate(variable_names)}
    
    trend_names = [f'trend_{name}' for name in variable_names]
    stationary_names = [f'cycle_{name}' for name in variable_names]
    
    for obs_name, eq_str in config_data['model_equations'].items():
        if obs_name in observable_indices:
            parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
            parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))
    
    config_data['model_equations_parsed'] = parsed_model_eqs_list
    
    # Create detailed parsing for JAX
    c_matrix_state_names = trend_names + stationary_names
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {}
    
    parsed_model_eqs_jax_detailed = []
    for obs_idx, parsed_terms in parsed_model_eqs_list:
        processed_terms_for_obs = []
        for param_name, state_name_in_eq, sign in parsed_terms:
            term_type = 0 if param_name is None else 1
            if state_name_in_eq in state_to_c_idx_map:
                state_index_in_C = state_to_c_idx_map[state_name_in_eq]
                param_index_if_any = -1
                processed_terms_for_obs.append(
                    (term_type, state_index_in_C, param_index_if_any, float(sign))
                )
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))
    
    config_data['parsed_model_eqs_jax_detailed'] = tuple(parsed_model_eqs_jax_detailed)
    
    # Identify trend names with shocks  
    config_data['trend_names_with_shocks'] = trend_names  # All trends have shocks
    config_data['n_trend_shocks'] = len(trend_names)
    
    # Pre-calculate static indices
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows, off_diag_cols)
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)
    
    # Add compatibility keys with CONSISTENT naming
    config_data.update({
        'observable_names': tuple(variable_names),
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
    full_state_names_list = trend_names + stationary_names  # For VAR order 1
    config_data['full_state_names_tuple'] = tuple(full_state_names_list)
    
    init_x_means_flat_list = []
    init_P_diag_flat_list = []
    initial_conditions_parsed = config_data['initial_conditions_parsed']
    
    for state_name in full_state_names_list:
        if state_name in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[state_name]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[state_name]['var']))
        else:
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)
    
    config_data['init_x_means_flat'] = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)
    
    # Store DLS results
    config_data['dls_results'] = dls_results
    
    return config_data

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

def run_bvar_with_dls_debug(data: pd.DataFrame, 
                           variable_names: List[str],
                           training_fraction: float = 0.3) -> Dict[str, Any]:
    """
    Simplified BVAR estimation with DLS priors - debug version.
    """
    
    print("="*80)
    print("BVAR ESTIMATION WITH DLS PRIORS - DEBUG VERSION")
    print("="*80)
    
    # Step 1: Create configuration with DLS priors
    print("Step 1: Creating DLS-based configuration...")
    config_data = create_dls_config(data, variable_names, training_fraction)
    
    print(f"Configuration created successfully:")
    print(f"  - Variables: {variable_names}")
    print(f"  - Trend names: {config_data['variables']['trend_names']}")
    print(f"  - Stationary names: {config_data['variables']['stationary_var_names']}")
    
    # Step 2: Prepare data
    print("\nStep 2: Preparing data...")
    y_data = data[variable_names].values
    print(f"Data shape: {y_data.shape}")
    
    # Handle observations
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y_data), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]
    
    # Step 3: Run MCMC
    print("\nStep 3: Running MCMC...")
    model_args = {
        'y': y_data,
        'config_data': config_data,
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'trend_var_names': config_data['variables']['trend_names'],
        'stationary_var_names': config_data['variables']['stationary_var_names'],
        'observable_names': config_data['variables']['observable_names'],
    }
    
    kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=numpyro.infer.init_to_sample())
    mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=2)  # Reduced for debugging
    
    key = random.PRNGKey(42)
    key_mcmc, key_smooth = random.split(key)
    
    start_time = time.time()
    mcmc.run(key_mcmc, **model_args)
    mcmc_time = time.time() - start_time
    
    print(f"MCMC completed in {mcmc_time:.2f} seconds")
    mcmc.print_summary()
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()
    
    # Step 4: Prepare for simulation smoother with FIXED naming
    print("\nStep 4: Running simulation smoother...")
    
    # CRITICAL FIX: Use base variable names for parameter lookups
    sampled_param_names = ['A_diag']
    if config_data['num_off_diag'] > 0:
        sampled_param_names.append('A_offdiag')
    if config_data['k_stationary'] > 1:
        sampled_param_names.append('stationary_chol')
    
    # Add stationary variances - use BASE variable names 
    for var_name in variable_names:
        sampled_param_names.append(f'stationary_var_{var_name}')
    
    # Add trend variances - use BASE variable names
    for var_name in variable_names:
        sampled_param_names.append(f'trend_var_{var_name}')
    
    print(f"Looking for parameters: {sampled_param_names}")
    print(f"Available parameters: {list(posterior_samples.keys())}")
    
    posterior_mean_params = {
        name: jnp.mean(posterior_samples[name], axis=0)
        for name in sampled_param_names
        if name in posterior_samples
    }
    
    print(f"Found parameters: {list(posterior_mean_params.keys())}")
    
    # Convert to hashable formats
    model_eqs_hashable = convert_to_hashable(config_data['model_equations_parsed'])
    measurement_params_hashable = convert_to_hashable([])
    initial_conds_tuple = tuple(
        (state_name, float(state_config['mean']), float(state_config['var']))
        for state_name, state_config in config_data['initial_conditions_parsed'].items()
    )
    
    static_smoother_args = {
        'static_k_endog': config_data['k_endog'],
        'static_k_trends': config_data['k_trends'],
        'static_k_stationary': config_data['k_stationary'],
        'static_p': config_data['var_order'],
        'static_k_states': config_data['k_states'],
        'static_n_trend_shocks': config_data['n_trend_shocks'],
        'static_n_shocks_state': config_data['n_trend_shocks'] + config_data['k_stationary'],
        'static_num_off_diag': config_data['num_off_diag'],
        'static_off_diag_rows': config_data['static_off_diag_indices'][0],
        'static_off_diag_cols': config_data['static_off_diag_indices'][1],
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'model_eqs_parsed': model_eqs_hashable,
        'initial_conds_parsed': initial_conds_tuple,
        'trend_names_with_shocks': tuple(config_data['trend_names_with_shocks']),
        'stationary_var_names': tuple(config_data['variables']['stationary_var_names']),
        'trend_var_names': tuple(config_data['variables']['trend_names']),
        'measurement_params_config': measurement_params_hashable,
        'num_draws': 50,
    }
    
    try:
        smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
            posterior_mean_params,
            y_data,
            key_smooth,
            **static_smoother_args
        )
        print("Simulation smoother completed successfully!")
        
        # Simple plot
        if simulation_results is not None:
            mean_sim_states, median_sim_states, all_sim_draws = simulation_results
            
            fig, axes = plt.subplots(len(variable_names), 2, figsize=(15, 4*len(variable_names)))
            if len(variable_names) == 1:
                axes = axes.reshape(1, -1)
            elif len(variable_names) == 2 and axes.ndim == 1:
                axes = axes.reshape(1, -1)
            
            dates = data.index
            
            for i, var_name in enumerate(variable_names):
                # Plot trend
                trend_idx = i
                axes[i, 0].plot(dates, smoothed_states_original[:, trend_idx], 'b-', label='Smoothed Trend', alpha=0.8)
                axes[i, 0].plot(dates, mean_sim_states[:, trend_idx], 'r:', label='Mean Sim Trend', alpha=0.8)
                
                # Add uncertainty band for trend
                try:
                    trend_draws = all_sim_draws[:, :, trend_idx]
                    lower = jnp.percentile(trend_draws, 10, axis=0)
                    upper = jnp.percentile(trend_draws, 90, axis=0)
                    axes[i, 0].fill_between(dates, lower, upper, color='red', alpha=0.2, label='80% Band')
                except:
                    pass
                    
                axes[i, 0].set_title(f'{var_name} - Trend Component')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                # Plot cycle  
                cycle_idx = config_data['k_trends'] + i
                axes[i, 1].plot(dates, smoothed_states_original[:, cycle_idx], 'g-', label='Smoothed Cycle', alpha=0.8)
                axes[i, 1].plot(dates, mean_sim_states[:, cycle_idx], 'r:', label='Mean Sim Cycle', alpha=0.8)
                
                # Add uncertainty band for cycle
                try:
                    cycle_draws = all_sim_draws[:, :, cycle_idx]
                    lower = jnp.percentile(cycle_draws, 10, axis=0)
                    upper = jnp.percentile(cycle_draws, 90, axis=0)
                    axes[i, 1].fill_between(dates, lower, upper, color='red', alpha=0.2, label='80% Band')
                except:
                    pass
                    
                axes[i, 1].set_title(f'{var_name} - Cycle Component')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Simulation smoother failed: {e}")
        print("But MCMC was successful, so the main estimation worked!")
        simulation_results = None
        smoothed_states_original = None
    
    # Return results
    results = {
        'config': config_data,
        'posterior_samples': posterior_samples,
        'mcmc_time': mcmc_time,
        'smoothed_states': smoothed_states_original,
        'simulation_results': simulation_results,
        'variable_names': variable_names,
        'data_shape': y_data.shape
    }
    
    print(f"\nEstimation completed successfully!")
    print(f"Variables estimated: {variable_names}")
    print(f"MCMC time: {mcmc_time:.2f} seconds")
    
    return results

def test_with_sample_data():
    """Test the debug version with sample data."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('1960-01-01', periods=150, freq='Q')
    
    # Simulate realistic economic data
    gdp_trend = np.cumsum(np.random.normal(0.008, 0.015, 150)) + 2.0
    gdp_cycle = np.zeros(150)
    for i in range(1, 150):
        gdp_cycle[i] = 0.7 * gdp_cycle[i-1] + np.random.normal(0, 0.2)
    
    inf_trend = np.cumsum(np.random.normal(0.002, 0.008, 150)) + 2.5
    inf_cycle = np.zeros(150)
    for i in range(1, 150):
        inf_cycle[i] = 0.6 * inf_cycle[i-1] + np.random.normal(0, 0.3)
    
    data = pd.DataFrame({
        'gdp_growth': gdp_trend + gdp_cycle,
        'inflation': inf_trend + inf_cycle
    }, index=dates)
    
    print("="*80)
    print("TESTING DLS-BASED BVAR ESTIMATION")
    print("="*80)
    print("Sample data created:")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {list(data.columns)}")
    print(f"Data shape: {data.shape}")
    print("\nFirst few observations:")
    print(data.head())
    
    # Run the estimation
    try:
        results = run_bvar_with_dls_debug(
            data=data,
            variable_names=['gdp_growth', 'inflation'],
            training_fraction=0.25
        )
        
        print("\n" + "="*80)
        print("SUCCESS! DLS-based BVAR estimation completed")
        print("="*80)
        
        # Print some summary info
        posterior = results['posterior_samples']
        print(f"\nPosterior parameter summary:")
        for param_name in ['A_diag', 'stationary_var_gdp_growth', 'stationary_var_inflation']:
            if param_name in posterior:
                param_values = posterior[param_name]
                mean_val = jnp.mean(param_values)
                std_val = jnp.std(param_values)
                print(f"  {param_name}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Show DLS prior info
        if 'dls_results' in results['config']:
            print(f"\nDLS Prior Information:")
            for var_name, dls_info in results['config']['dls_results'].items():
                trend_prior = dls_info['trend_prior']
                cycle_prior = dls_info['cycle_prior']
                print(f"  {var_name}:")
                print(f"    Trend prior: α={trend_prior['alpha']:.2f}, β={trend_prior['beta']:.6f}")
                print(f"    Cycle prior: α={cycle_prior['alpha']:.2f}, β={cycle_prior['beta']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error during estimation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    results = test_with_sample_data()