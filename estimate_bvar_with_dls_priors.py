# estimate_bvar_with_dls_priors_corrected_v9.py
# Integrated estimation script with DLS prior elicitation and improved parameter extraction

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    # Allow multiple devices if available (e.g., multiple CPU cores)
    # This helps parallelize chains
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    numpyro.set_host_device_count(min(num_cpu, 4)) # Use up to 4 devices for MCMC
except Exception as e:
    print(f"Could not set host device count: {e}")
    pass


# Import your existing modules
# Ensure these paths are correct relative to where you run the script
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax # Assuming this is available
from core.simulate_bvar_jax import simulate_bvar_with_trends_jax # Assuming this is available
from core.var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices # Assuming this is available
from core.run_single_draw import run_simulation_smoother_single_params_jit # Assuming this is available

# --- DLS Prior Elicitation Functions (from complete_dls_prior_elicitation.py) ---

def simple_dls_trend_cycle(y: np.ndarray, discount_factor: float = 0.98) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simplified DLS for trend-cycle decomposition.

    Returns:
        trend, cycle, trend_variance, cycle_variance (estimated from components)
    """
    y = np.asarray(y).flatten()
    T = len(y)

    # Simple exponential smoothing for trend
    trend = np.zeros(T, dtype=_DEFAULT_DTYPE)
    if T > 0:
        # Handle potential NaN at the start by finding first valid point
        first_valid_idx = np.argmax(np.isfinite(y)) if np.isfinite(y).any() else -1
        if first_valid_idx != -1:
             trend[first_valid_idx] = y[first_valid_idx]
             alpha = 1 - discount_factor
             for t in range(first_valid_idx + 1, T):
                # Handle NaNs in y: if y[t] is NaN, trend[t] is discount_factor * trend[t-1]
                if np.isfinite(y[t]):
                    trend[t] = discount_factor * trend[t-1] + alpha * y[t]
                else:
                    trend[t] = discount_factor * trend[t-1] # Carry forward last trend estimate
        else: # All NaNs
             trend = np.full(T, np.nan, dtype=_DEFAULT_DTYPE)


    # Cycle is residual (handle NaNs)
    cycle = y - trend
    cycle[~np.isfinite(y)] = np.nan # If y is NaN, cycle is NaN

    # Estimate variances from valid differences/cycles
    trend_diff = np.diff(trend)
    # Use nanvar but require at least two non-NaN finite points
    if np.sum(np.isfinite(trend_diff)) >= 2:
        trend_variance = np.nanvar(trend_diff)
    elif np.sum(np.isfinite(trend)) >= 2:
        # Fallback: estimate variance from the trend itself if diff is all NaN
        trend_variance = np.nanvar(trend) * 0.1 # Scale down as trend level variance is larger
    else:
        trend_variance = 0.01 # Default if almost no valid data


    if np.sum(np.isfinite(cycle)) >= 2:
         cycle_variance = np.nanvar(cycle)
    else:
         cycle_variance = 0.1 # Default if almost no valid data

    # Ensure positive variances
    trend_variance = max(float(trend_variance), 1e-9) # Use smaller minimum variance
    cycle_variance = max(float(cycle_variance), 1e-9)

    return trend, cycle, trend_variance, cycle_variance

def suggest_ig_priors(empirical_var: float, alpha: float = 2.5) -> Dict[str, float]:
    """Simple Inverse Gamma prior suggestion."""
    # Beta = empirical_var * (alpha - 1) to make prior mean match empirical_var
    # Ensure alpha > 1 for finite mean, alpha > 2 for finite variance
    if alpha <= 1:
        # Default alpha to something > 1 if provided one is too small
        print(f"Warning: Alpha <= 1 (is {alpha}) passed to suggest_ig_priors. Setting alpha to 2.5.")
        alpha = 2.5 # Ensure mean is defined
    # if alpha <= 2:
    #      print(f"Warning: Alpha <= 2 (is {alpha}). IG variance is infinite or undefined.")

    beta = empirical_var * (alpha - 1.0)

    # Ensure beta is positive
    beta = max(float(beta), 1e-9) # Use smaller minimum beta

    # Implied mean is beta / (alpha - 1)
    # Implied variance is beta^2 / ((alpha - 1)^2 * (alpha - 2))
    implied_mean = float(beta / (alpha - 1.0)) if alpha > 1 else float('inf')
    implied_variance = float(beta**2 / ((alpha - 1.0)**2 * (alpha - 2.0))) if alpha > 2 else float('inf')


    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'implied_mean': implied_mean,
        'implied_variance': implied_variance
    }


def create_dls_config_results_only(data: pd.DataFrame,
                     variable_names: List[str],
                     training_fraction: float = 0.3,
                     dls_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Apply DLS to training data and return *only* the prior elicitation results.
    This is a helper for the main config creation function.
    """

    # Use subset of data for prior elicitation
    n_total = len(data)
    n_train = int(training_fraction * n_total)
    if n_train < 10: # Require minimum data points for DLS
        print(f"Warning: Training fraction {training_fraction:.1%} results in only {n_train} observations out of {n_total}. Need at least 10 for DLS trend/cycle.")
        n_train = min(max(n_train, 10), n_total) # Ensure at least 10, but not more than total data


    train_data = data.iloc[:n_train]

    print(f"Using {len(train_data)} observations ({train_data.index[0]} to {train_data.index[-1]}) for prior elicitation out of {n_total} total observations.")

    # Default DLS parameters (if not provided)
    default_dls_params = {
        'discount_factor': 0.98,
        'lambda_scaling': 2.0, # Not currently used by simple_dls
        'alpha_shape': 2.5,
        'var_confidence': 0.7 # Not currently used by simple_dls
    }
    actual_dls_params = {**default_dls_params, **(dls_params or {})} # Merge defaults and provided

    # Apply DLS to each variable
    dls_results = {}
    for var_name in variable_names:
        y = train_data[var_name].values.astype(_DEFAULT_DTYPE)

        # Need at least 2 finite points for DLS
        if np.sum(np.isfinite(y)) < 2:
            print(f"Warning: Too few finite observations for {var_name} ({np.sum(np.isfinite(y))}) in training data. Skipping DLS.")
            continue

        try:
            # Apply DLS to the potentially partial training data
            trend, cycle, trend_var_estimated, cycle_var_estimated = simple_dls_trend_cycle(y, discount_factor=actual_dls_params['discount_factor'])

            # Ensure trend and cycle have the same length as y (including NaNs)
            assert len(trend) == len(y)
            assert len(cycle) == len(y)

            # Use initial part of cycle/trend for initial condition prior (only use valid points)
            initial_window = min(10, np.sum(np.isfinite(y))) # Use up to 10 valid points from the start
            first_finite_idx = np.argmax(np.isfinite(y)) if np.isfinite(y).any() else 0

            initial_trend_valid = trend[first_finite_idx : first_finite_idx + initial_window]
            initial_cycle_valid = cycle[first_finite_idx : first_finite_idx + initial_window]


            initial_trend_mean = np.nanmean(initial_trend_valid) if np.sum(np.isfinite(initial_trend_valid)) >= 1 else (y[first_finite_idx] if np.isfinite(y[first_finite_idx]) else 0.0)
            initial_trend_var = np.nanvar(initial_trend_valid) if np.sum(np.isfinite(initial_trend_valid)) >= 2 else trend_var_estimated * 5.0 # Fallback if not enough initial trend points (use estimated shock var * multiplier)

            initial_cycle_mean = np.nanmean(initial_cycle_valid) if np.sum(np.isfinite(initial_cycle_valid)) >= 1 else 0.0
            initial_cycle_var = np.nanvar(initial_cycle_valid) if np.sum(np.isfinite(initial_cycle_valid)) >= 2 else cycle_var_estimated # Fallback if not enough initial cycle points (use estimated shock var)

            # Ensure positive initial variances
            initial_trend_var = max(float(initial_trend_var), 1e-9)
            initial_cycle_var = max(float(initial_cycle_var), 1e-9)


            # Create prior suggestions for SHOCK variances
            # Use the variance estimates from the DLS components
            trend_prior_params = suggest_ig_priors(trend_var_estimated, alpha=actual_dls_params['alpha_shape'])
            cycle_prior_params = suggest_ig_priors(cycle_var_estimated, alpha=actual_dls_params['alpha_shape'])

            dls_results[var_name] = {
                'initial_conditions': {
                    'trend_mean': float(initial_trend_mean),
                    'trend_variance': float(initial_trend_var),
                    'cycle_mean': float(initial_cycle_mean),
                    'cycle_variance': float(initial_cycle_var),
                },
                # Priors for shock variances (Inverse Gamma params)
                'trend_shocks': trend_prior_params,
                'cycle_shocks': cycle_prior_params,
                'diagnostics': {
                    'trend_component': trend, # Store the full DLS components for plotting
                    'cycle_component': cycle
                }
            }

            print(f"DLS for {var_name}:")
            print(f"  Initial State Priors: Trend Mean={dls_results[var_name]['initial_conditions']['trend_mean']:.4f}, Var={dls_results[var_name]['initial_conditions']['trend_variance']:.6f}, Cycle Mean={dls_results[var_name]['initial_conditions']['cycle_mean']:.4f}, Var={dls_results[var_name]['initial_conditions']['cycle_variance']:.6f}")
            print(f"  Trend Shock IG Prior: α={trend_prior_params['alpha']:.2f}, β={trend_prior_params['beta']:.6f} (Implied Mean={trend_prior_params['implied_mean']:.6f})")
            print(f"  Cycle Shock IG Prior: α={cycle_prior_params['alpha']:.2f}, β={cycle_prior_params['beta']:.6f} (Implied Mean={cycle_prior_params['implied_mean']:.6f})")

        except Exception as e:
             print(f"Error during DLS for {var_name}: {e}")
             import traceback
             traceback.print_exc()
             continue


    return dls_results

# --- End of DLS Prior Elicitation Functions ---


def convert_to_hashable(obj):
    """Recursively convert lists to tuples to make objects hashable for JAX JIT."""
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        # Convert keys to string for consistency if they are not already
        return tuple(sorted(((str(k), convert_to_hashable(v)) if not isinstance(k, str) else (k, convert_to_hashable(v))) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
         return tuple(obj.tolist()) # Convert numpy arrays
    elif isinstance(obj, jnp.ndarray):
         return tuple(obj.tolist()) # Convert jax arrays
    else:
        return obj

def load_data_from_csv(file_path: str,
                      date_column: str = 'date',
                      date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV file with proper date parsing.

    Args:
        file_path: Path to CSV file
        date_column: Name of date column
        date_format: Date format string (if None, will try to infer)

    Returns:
        DataFrame with DatetimeIndex
    """
    df = pd.read_csv(file_path)

    if date_format:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    else:
        df[date_column] = pd.to_datetime(df[date_column])

    df.set_index(date_column, inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    print(f"Loaded data: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Variables: {list(df.columns)}")

    return df

def create_config_with_dls_priors(data: pd.DataFrame,
                                 variable_names: Optional[List[str]] = None,
                                 training_fraction: float = 0.3,
                                 var_order: int = 1,
                                 dls_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a complete BVAR configuration using DLS-derived priors.

    Args:
        data: DataFrame with time series data
        variable_names: List of variable names to use (if None, uses all columns)
        training_fraction: Fraction of data for prior elicitation
        var_order: VAR order
        dls_params: Dictionary of DLS parameters (passed to create_dls_config_results_only)

    Returns:
        Configuration dictionary ready for BVAR estimation
    """
    if variable_names is None:
        variable_names = list(data.columns)

    print("="*80)
    print("APPLYING DLS PRIOR ELICITATION")
    print("="*80)

    # Apply DLS prior elicitation to get results
    prior_results = create_dls_config_results_only( # <-- Use the helper function
        data=data[variable_names],
        variable_names=variable_names,
        training_fraction=training_fraction,
        dls_params=dls_params # <-- Pass dls_params here
    )

    # Create base configuration structure
    config_data = {
        'var_order': var_order,
        'variables': {
            'observable_names': variable_names,
            'trend_names': [f'trend_{name}' for name in variable_names],
            'stationary_var_names': [f'cycle_{name}' for name in variable_names],
        },
        'model_equations': {},
        'initial_conditions': {'states': {}},
        'stationary_prior': {
            # Default hyperparameters for VAR coefficients if not overwritten by DLS-like method later
            # es: scale and tightness for own lags
            # fs: scale and tightness for other lags
            'hyperparameters': {
                'es': [0.7, 0.15], # Example values
                'fs': [0.2, 0.15]  # Example values
            },
            'covariance_prior': {'eta': 1.5}, # Prior strength for off-diagonals of VAR shocks (used if sampling Cholesky)
            'stationary_shocks': {} # Will be filled by DLS results (diagonal IG priors)
        },
        'trend_shocks': {'trend_shocks': {}}, # Will be filled by DLS results (diagonal IG priors)
        'parameters': {'measurement': []}, # No measurement parameters in this simple model
    }

    # Calculate dimensions
    k_endog = len(variable_names)
    k_trends = len(variable_names)
    k_stationary = len(variable_names)
    k_states = k_trends + k_stationary * var_order

    config_data.update({
        'k_endog': k_endog,
        'k_trends': k_trends,
        'k_stationary': k_stationary,
        'k_states': k_states,
    })

    # Fill in model equations (simple additive trend + cycle)
    for var_name in variable_names:
        config_data['model_equations'][var_name] = f'trend_{var_name} + cycle_{var_name}'

    # Fill in DLS-derived priors into the config structure
    for var_name in variable_names:
        if var_name in prior_results:
            var_results = prior_results[var_name]

            # Initial conditions (using state names like trend_gdp_growth, cycle_inflation)
            init_conds = var_results['initial_conditions']
            # Only set for the initial (t=0) cycle state, not lagged ones
            config_data['initial_conditions']['states'][f'trend_{var_name}'] = {
                'mean': float(init_conds['trend_mean']),
                'var': float(init_conds['trend_variance'])
            }
            config_data['initial_conditions']['states'][f'cycle_{var_name}'] = {
                'mean': float(init_conds['cycle_mean']),
                'var': float(init_conds['cycle_variance'])
            }

            # Stationary (Cycle) shock priors (using state names like cycle_gdp_growth)
            cycle_prior = var_results['cycle_shocks']
            config_data['stationary_prior']['stationary_shocks'][f'cycle_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': float(cycle_prior['alpha']),
                    'beta': float(cycle_prior['beta'])
                }
            }

            # Trend shock priors (using state names like trend_gdp_growth)
            trend_prior = var_results['trend_shocks']
            config_data['trend_shocks']['trend_shocks'][f'trend_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': float(trend_prior['alpha']),
                    'beta': float(trend_prior['beta'])
                }
            }
        else:
             print(f"Warning: DLS results not available for {var_name}. Using default initial conditions and potentially impacting priors derived from these.")
             # Set some fallback defaults if DLS failed for a variable
             config_data['initial_conditions']['states'][f'trend_{var_name}'] = {'mean': float(data[var_name].iloc[0]) if len(data)>0 and np.isfinite(data[var_name].iloc[0]) else 0.0, 'var': 1.0}
             config_data['initial_conditions']['states'][f'cycle_{var_name}'] = {'mean': 0.0, 'var': 1.0}
             # Priors for shocks would need defaults too, but the model might handle missing priors
             # For now, we assume the model will either use other priors or fail clearly if diagonal priors are missing.


    # --- Start Parsing Section (Copied from original and verified) ---
    # Parse initial conditions
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])

    # Parse model equations
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config if isinstance(p, dict) and 'name' in p]

    parsed_model_eqs_list = []
    observable_indices = {name: i for i, name in enumerate(variable_names)}

    trend_names = config_data['variables']['trend_names']
    stationary_names = config_data['variables']['stationary_var_names']


    for obs_name, eq_str in config_data['model_equations'].items():
        if obs_name in observable_indices:
            parsed_terms = _parse_equation_jax(eq_str,
                                             trend_names,
                                             stationary_names,
                                             measurement_param_names)
            parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))

    config_data['model_equations_parsed'] = parsed_model_eqs_list

    # Create detailed parsing for JAX
    # Build the full state name list including lags for parsing
    full_state_names_list_for_parsing = list(trend_names)
    for i in range(var_order):
        for stat_var in stationary_names:
            if i == 0:
                full_state_names_list_for_parsing.append(stat_var)
            else:
                full_state_names_list_for_parsing.append(f"{stat_var}_t_minus_{i+1}") # Use _t_minus_1, _t_minus_2 etc


    c_matrix_state_names = full_state_names_list_for_parsing # C matrix uses full state vector order
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names)} # Should be empty if no meas params

    parsed_model_eqs_jax_detailed = []
    for obs_idx, parsed_terms in parsed_model_eqs_list:
        processed_terms_for_obs = []
        for param_name, state_name_in_eq, sign in parsed_terms:
            term_type = 0 if param_name is None else 1 # 0 for state, 1 for parameter * state
            if state_name_in_eq in state_to_c_idx_map:
                state_index_in_C = state_to_c_idx_map[state_name_in_eq]
                param_index_if_any = param_to_idx_map.get(param_name, -1) # Use .get for safety
                processed_terms_for_obs.append(
                    (term_type, state_index_in_C, param_index_if_any, float(sign))
                )
            else:
                 print(f"Warning: State name '{state_name_in_state_eq}' from equation for observable '{variable_names[obs_idx]}' not found in state vector names ({c_matrix_state_names}). Term ignored.")
        # Append even if processed_terms_for_obs is empty, as the observable exists
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))

    config_data['parsed_model_eqs_jax_detailed'] = tuple(parsed_model_eqs_jax_detailed)

    # Identify trend names with shocks
    # DLS provides diagonal priors, so all trend vars with DLS priors are assumed to have shocks
    trend_shocks_spec = config_data.get('trend_shocks', {}).get('trend_shocks', {})
    config_data['trend_names_with_shocks'] = [
        name for name in config_data['variables']['trend_names'] # Iterate through all trend names
        if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and trend_shocks_spec[name].get('distribution') == 'inverse_gamma'
    ]
    config_data['n_trend_shocks'] = len(config_data['trend_names_with_shocks'])
    # --- End Parsing Section ---


    # Pre-calculate static indices for stationary shock covariance (used by model for Cholesky if needed)
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows.astype(int), off_diag_cols.astype(int)) # Ensure int
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)


    # Add compatibility keys (verified names)
    config_data.update({
        'observable_names': tuple(variable_names),
        'trend_var_names': tuple(config_data['variables']['trend_names']), # Full trend names
        'stationary_var_names': tuple(config_data['variables']['stationary_var_names']), # Full cycle names
        'raw_config_initial_conds': config_data['initial_conditions'],
        'raw_config_stationary_prior': config_data['stationary_prior'],
        'raw_config_trend_shocks': config_data['trend_shocks'],
        'raw_config_measurement_params': measurement_params_config,
        'raw_config_model_eqs_str_dict': config_data['model_equations'],
        'measurement_param_names_tuple': tuple(measurement_param_names),
    })

    # Create flat initial condition arrays based on the full state order
    config_data['full_state_names_tuple'] = tuple(full_state_names_list_for_parsing) # Use the same list as for parsing

    init_x_means_flat_list = []
    init_P_diag_flat_list = []
    initial_conditions_parsed = config_data['initial_conditions_parsed']

    # Match initial conditions to the order of full_state_names_list_for_parsing
    for state_name in full_state_names_list_for_parsing:
        base_name_for_lag = state_name
        is_lagged = False
        if "_t_minus_" in state_name:
            # Split 'cycle_gdp_growth_t_minus_1' into ['cycle_gdp_growth', '1']
            parts = state_name.split("_t_minus_")
            if len(parts) == 2:
                base_name_for_lag = parts[0]
                is_lagged = True


        # Check for the exact state name first
        if state_name in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[state_name]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[state_name]['var']))
        # If exact name not found (e.g. for lagged states), use the base name's initial conditions
        elif is_lagged and base_name_for_lag in initial_conditions_parsed:
             # Use the same mean and variance as the t=0 component for lagged states
             init_x_means_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['mean']))
             # For lagged states, initial variance is typically small or zero if the initial state (lag 0) is known well.
             # However, if the initial *unobserved* state vector needs to be estimated,
             # the variance for lagged components might reflect uncertainty about those past values.
             # Using the base variance is consistent with the prior attempt.
             init_P_diag_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['var']))
        else:
            # Fallback for states without explicit initial conditions (shouldn't happen with DLS for trends/cycles)
            # or for lagged states whose base names weren't in the initial_conditions_parsed (less likely if handled above)
            print(f"Warning: Initial condition not found for state '{state_name}'. Using 0 mean, 1 variance.")
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)

    config_data['init_x_means_flat'] = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)


    # Store the DLS results for potential analysis
    config_data['dls_prior_results'] = prior_results

    print("="*80)
    print("DLS PRIOR ELICITATION AND CONFIG CREATION COMPLETED")
    print("="*80)

    return config_data

def run_bvar_estimation_with_dls(data: pd.DataFrame,
                                variable_names: Optional[List[str]] = None,
                                training_fraction: float = 0.3,
                                var_order: int = 1,
                                mcmc_params: Optional[Dict] = None,
                                dls_params: Optional[Dict] = None,
                                simulation_draws: int = 100,
                                save_config: bool = True,
                                config_filename: str = "bvar_dls_auto.yml") -> Dict[str, Any]:
    """
    Complete BVAR estimation pipeline with DLS prior elicitation.

    Args:
        data: DataFrame with time series data
        variable_names: Variables to include (if None, uses all)
        training_fraction: Fraction of data for prior elicitation
        var_order: VAR order
        mcmc_params: MCMC configuration
        dls_params: DLS parameters
        simulation_draws: Number of simulation smoother draws
        save_config: Whether to save the generated config to YAML
        config_filename: Filename for saved config

    Returns:
        Dictionary with all estimation results (or error info if smoothing fails)
    """

    if variable_names is None:
        variable_names = list(data.columns)

    # Default MCMC parameters
    if mcmc_params is None:
        mcmc_params = {
            'num_warmup': 300,
            'num_samples': 500,
            'num_chains': 2
        }

    print("="*100)
    print("BVAR ESTIMATION WITH DLS PRIOR ELICITATION")
    print("="*100)
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {variable_names}")
    print(f"VAR order: {var_order}")
    print(f"Training fraction for priors: {training_fraction:.1%}")
    print(f"MCMC Params: {mcmc_params}")
    print(f"DLS Params: {dls_params}")


    # Step 1: Create configuration with DLS priors
    print("\nStep 1: Generating data-driven priors and configuration...")
    config_data = create_config_with_dls_priors(
        data=data,
        variable_names=variable_names,
        training_fraction=training_fraction,
        var_order=var_order,
        dls_params=dls_params # <-- Pass DLS params
    )

    # Step 2: Save configuration if requested
    if save_config:
        # Convert config to YAML-friendly format (exclude JAX arrays/tuples where possible)
        yaml_config = {
            'var_order': config_data['var_order'],
            'variables': {
                'observables': list(config_data['variables']['observable_names']),
                'trends': list(config_data['variables']['trend_names']),
                'stationary': list(config_data['variables']['stationary_var_names'])
            },
            'model_equations': config_data['model_equations'],
            'initial_conditions': config_data['raw_config_initial_conds'], # Use raw config
            'stationary_prior': config_data['raw_config_stationary_prior'], # Use raw config
            'trend_shocks': config_data['raw_config_trend_shocks'], # Use raw config
            'parameters': config_data['raw_config_measurement_params'] # Use raw config
            # Exclude 'dls_prior_results', 'parsed_model_eqs_jax_detailed', etc.
        }

        try:
            with open(config_filename, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {config_filename}")
        except Exception as e:
             print(f"Warning: Could not save configuration to {config_filename}: {e}")


    # Step 3: Prepare data for estimation
    y_data = data[variable_names].values.astype(_DEFAULT_DTYPE) # Ensure correct dtype
    print(f"\nStep 3: Preparing data for estimation...")
    print(f"Data shape: {y_data.shape}")

    # Handle NaN detection (observations are missing for some variables at some time points)
    # The model handles row-wise NaN, but we need valid columns
    # A column is valid if it has *any* finite observation across time
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y_data), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]

    if static_n_obs_actual == 0:
        raise ValueError("No valid observation columns found in data (all columns contain only NaN/Inf).")
    if static_n_obs_actual < config_data['k_endog']:
         print(f"Warning: Only {static_n_obs_actual}/{config_data['k_endog']} observable columns have any finite data.")


    # Prepare data_info dictionary early for use in potential error returns
    data_info = {
        'data_shape': y_data.shape,
        'variable_names': variable_names,
        'date_range': (data.index[0], data.index[-1])
    }


    # Step 4: Run MCMC estimation
    print(f"\nStep 4: Running MCMC estimation...")
    model_args = {
        'y': y_data,
        'config_data': config_data,
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        # Pass full state names, observable names etc. as lists/tuples as required by the model function signature
        'trend_var_names': list(config_data['variables']['trend_names']), # Pass as list
        'stationary_var_names': list(config_data['variables']['stationary_var_names']), # Pass as list
        'observable_names': list(config_data['variables']['observable_names']), # Pass as list
        # Other args needed by numpyro_bvar_stationary_model might be implicitly accessed via config_data
    }

    kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=numpyro.infer.init_to_sample())
    mcmc = MCMC(kernel, **mcmc_params)

    key = random.PRNGKey(42)
    key_mcmc, key_smooth = random.split(key)

    start_time = time.time()
    posterior_samples = None # Initialize in case of MCMC failure
    mcmc_extras = None
    mcmc_time = 0.0
    try:
        mcmc.run(key_mcmc, **model_args)
        mcmc_time = time.time() - start_time
        print(f"\nMCMC completed in {mcmc_time:.2f} seconds")
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        mcmc_extras = mcmc.get_extra_fields() # Store extra fields
    except Exception as e:
        print(f"\nERROR: MCMC estimation failed: {e}")
        import traceback
        traceback.print_exc()
        # Return partial results including the error
        return {
            'config': config_data,
            'mcmc_results': None, # MCMC failed
            'smoothing_results': None,
            'data_info': data_info,
            'error': f"MCMC failed: {e}"
        }

    # Check if MCMC produced any samples
    if posterior_samples is None or not posterior_samples:
         print("\nERROR: MCMC failed to produce any samples.")
         return {
             'config': config_data,
             'mcmc_results': None,
             'smoothing_results': None,
             'data_info': data_info,
             'error': "MCMC produced no samples."
         }


    # Step 5: Run simulation smoother
    print(f"\nStep 5: Running simulation smoother ({simulation_draws} draws)...")

    # Get posterior means for parameters needed by the smoother
    # *** FIX: Collect the SPECIFIC parameters that were sampled, based on the MCMC output ***
    # The smoother function 'run_simulation_smoother_single_params_jit' must be designed
    # to interpret these sampled parameters and build the state-space matrices internally.
    # Based on your MCMC output, the sampled parameters are:
    # 'A_diag', 'A_offdiag', 'stationary_chol', 'stationary_var_cycle_gdp_growth',
    # 'stationary_var_cycle_inflation', 'trend_var_trend_gdp_growth', 'trend_var_trend_inflation'.

    posterior_mean_params = {}
    missing_params_for_smoother = []

    # List all parameters we expect to find from the MCMC output based on previous runs
    expected_sampled_params = ['A_diag'] # Always expected

    if config_data.get('num_off_diag', 0) > 0:
        expected_sampled_params.append('A_offdiag') # Expected if off-diagonals exist

    k_stationary = config_data['k_stationary']
    n_trend_shocks = config_data.get('n_trend_shocks', 0)
    stationary_var_names_config = config_data['variables']['stationary_var_names'] # e.g., ['cycle_gdp_growth', ...]
    trend_names_with_shocks_config = config_data.get('trend_names_with_shocks', []) # e.g., ['trend_gdp_growth', ...]

    # Stationary Shocks: Check for sampled parameters based on k_stationary
    if k_stationary > 1:
        # Model sampled stationary_chol
        expected_sampled_params.append('stationary_chol')
        # Although individual variances were *also* sampled, based on typical state-space models
        # and the presence of stationary_chol, the smoother likely uses stationary_chol
        # to build the stationary shock covariance matrix. We'll include the individual
        # variances as well in case the smoother needs them for some reason, but the primary
        # expectation is that stationary_chol is used when k_stationary > 1.
        for full_state_name in stationary_var_names_config:
             param_name = f'stationary_var_{full_state_name}' # e.g., 'stationary_var_cycle_gdp_growth'
             if param_name in posterior_samples:
                  expected_sampled_params.append(param_name)
             else:
                  print(f"Warning: Individual stationary variance '{param_name}' not found despite being sampled previously.") # Unexpected but handle

    elif k_stationary == 1:
         # Model sampled stationary_var_{full_cycle_state_name}
         if len(stationary_var_names_config) == 1:
              full_state_name = stationary_var_names_config[0]
              param_name = f'stationary_var_{full_state_name}'
              expected_sampled_params.append(param_name)
         else:
              print(f"Configuration Error: k_stationary is 1 but {len(stationary_var_names_config)} stationary variables found.") # Should not happen if config creation is correct

    # Trend Shocks: Check for sampled individual variances for trends WITH shocks
    if n_trend_shocks > 0:
        # Model sampled trend_var_{full_trend_state_name} for EACH trend with a shock
        for full_state_name in trend_names_with_shocks_config:
             param_name = f'trend_var_{full_state_name}'
             expected_sampled_params.append(param_name)
    else:
        print("No trend shocks configured with Inverse Gamma priors. Skipping trend shock parameter extraction.")


    # Now, iterate through the expected parameters and collect their means IF THEY EXIST IN SAMPLES
    found_params = []
    for param_name in expected_sampled_params:
         if param_name in posterior_samples:
              posterior_mean_params[param_name] = jnp.mean(posterior_samples[param_name], axis=0)
              found_params.append(param_name)
         else:
              # Only add to missing if it's a critical parameter (A_diag, primary shock param)
              # For others (like A_offdiag if config expects it but model didn't sample, or redundant stat vars),
              # the smoother should handle it or they aren't strictly necessary.
              is_critical = False
              if param_name == 'A_diag': is_critical = True
              if k_stationary > 1 and param_name == 'stationary_chol': is_critical = True
              if k_stationary == 1 and any(param_name.startswith('stationary_var_') for name in stationary_var_names_config for param_name in [f'stationary_var_{name}']): is_critical = True
              if n_trend_shocks > 0 and any(param_name.startswith('trend_var_') for name in trend_names_with_shocks_config for param_name in [f'trend_var_{name}']): is_critical = True


              if is_critical:
                   missing_params_for_smoother.append(param_name)
              else:
                   print(f"Warning: Expected smoother parameter '{param_name}' not found in posterior samples. Skipping.")


    # --- Final check on collected parameters ---
    if missing_params_for_smoother:
         print(f"ERROR: The following CRITICAL parameters needed for the smoother were missing from MCMC samples: {', '.join(missing_params_for_smoother)}. Cannot run smoother.")
         print("Available parameters in MCMC samples:", list(posterior_samples.keys()))
         return {"config": config_data, "mcmc_results": posterior_samples, "smoothing_results": None, "data_info": data_info, "error": "Missing critical parameters for smoother"}

    # Ensure we have *some* shock parameters if shocks are configured
    has_stationary_shock_param = ('stationary_chol' in posterior_mean_params) or any(k.startswith('stationary_var_') for k in posterior_mean_params.keys())
    has_trend_shock_param = any(k.startswith('trend_var_') for k in posterior_mean_params.keys())

    if (k_stationary > 0 and not has_stationary_shock_param) or (n_trend_shocks > 0 and not has_trend_shock_param):
        print("ERROR: Shock parameters were configured but none were collected for the smoother. Cannot run smoother.")
        print("Collected parameters for smoother:", list(posterior_mean_params.keys()))
        return {"config": config_data, "mcmc_results": posterior_samples, "smoothing_results": None, "data_info": data_info, "error": "No shock parameters collected for smoother despite config."}


    print(f"Extracted posterior means for smoother: {list(posterior_mean_params.keys())}")

    # Convert necessary config components to hashable types for JIT
    model_eqs_jax_detailed_hashable = convert_to_hashable(config_data['parsed_model_eqs_jax_detailed'])
    measurement_params_config_hashable = convert_to_hashable(config_data.get('raw_config_measurement_params', [])) # Use get with default for safety

    # *** FIX: Add initial_conds_parsed back to static_smoother_args ***
    # It seems run_simulation_smoother_single_params_jit expects this dictionary.
    initial_conds_parsed_hashable = convert_to_hashable(config_data['initial_conditions_parsed'])


    static_smoother_args = {
        'static_k_endog': config_data['k_endog'],
        'static_k_trends': config_data['k_trends'],
        'static_k_stationary': config_data['k_stationary'],
        'static_p': config_data['var_order'],
        'static_k_states': config_data['k_states'],
        'static_n_trend_shocks': config_data['n_trend_shocks'],
        'static_n_shocks_state': config_data['k_stationary'] + config_data['n_trend_shocks'], # Total state shocks (stationary + trend)
        'static_num_off_diag': config_data.get('num_off_diag', 0), # Use .get
        'static_off_diag_rows': config_data['static_off_diag_indices'][0],
        'static_off_diag_cols': config_data['static_off_diag_indices'][1],
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'model_eqs_parsed': model_eqs_jax_detailed_hashable, # Use detailed JAX parsing
        'initial_conds_parsed': initial_conds_parsed_hashable, # *** ADDED BACK ***
        'trend_names_with_shocks': tuple(config_data.get('trend_names_with_shocks', [])), # Pass as tuple
        'stationary_var_names': tuple(config_data['variables']['stationary_var_names']), # Pass as tuple
        'trend_var_names': tuple(config_data['variables']['trend_names']), # Pass as tuple
        'measurement_params_config': measurement_params_config_hashable,
        'num_draws': simulation_draws,
        'init_x_means_flat': config_data['init_x_means_flat'], # Pass flat JAX arrays
        'init_P_diag_flat': config_data['init_P_diag_flat'], # Pass flat JAX arrays
    }

    smoothed_states_original = None
    simulation_results = None
    smooth_time = 0.0
    try:
        start_smooth_time = time.time()
        # Pass posterior_mean_params (containing the sampled parameters) and static args
        smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
            posterior_mean_params, # This dictionary contains the sampled parameters' means
            y_data,
            key_smooth,
            **static_smoother_args # This dictionary contains static config info
        )
        smooth_time = time.time() - start_smooth_time
        print(f"Simulation smoother completed in {smooth_time:.2f} seconds")

    except Exception as e:
        print(f"\nERROR: Simulation smoother failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue and return results with smoothing_results=None

    # Step 6: Process results
    results = {
        'config': config_data,
        'mcmc_results': {
            'posterior_samples': posterior_samples,
            'mcmc_time': mcmc_time,
            'mcmc_summary': mcmc_extras # Store extra fields
        },
        'smoothing_results': {
            'smoothed_states': smoothed_states_original,
            'simulation_results': simulation_results,
            'smooth_time': smooth_time
        },
        'data_info': data_info # Use the pre-defined data_info
    }

    print("\n" + "="*100)
    print("ESTIMATION PROCESS COMPLETED")
    if smoothed_states_original is None:
         print("NOTE: Simulation smoother failed.")
    print("="*100)

    return results

def plot_estimation_results(results: Dict[str, Any],
                           data: pd.DataFrame,
                           plot_diagnostics: bool = True,
                           save_plots: bool = False,
                           plot_dir: str = "plots") -> None:
    """
    Create comprehensive plots of estimation results.

    Args:
        results: Results from run_bvar_estimation_with_dls
        data: Original data DataFrame
        plot_diagnostics: Whether to include DLS diagnostic plots
        save_plots: Whether to save plots to files
        plot_dir: Directory for saved plots
    """

    # Check if smoothing results are available
    if results.get('smoothing_results') is None or results['smoothing_results'].get('smoothed_states') is None:
        print("No smoothing results found or smoother failed. Skipping state and fitted values plots.")
        plot_states_and_fitted = False
    else:
        plot_states_and_fitted = True
        smoothed_states = results['smoothing_results']['smoothed_states']
        # simulation_results might be None even if smoothed_states is available (if simulation_draws=0)
        simulation_results = results['smoothing_results'].get('simulation_results')


    config = results.get('config')
    if config is None:
         print("Configuration not found in results. Cannot plot states or fitted values.")
         plot_states_and_fitted = False
         # Can still try DLS diagnostics if raw data is passed

    variable_names = results['data_info']['variable_names']
    dates = data.index # Use full data index for plotting

    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot 1: DLS Diagnostic Plots (if available and requested)
    if plot_diagnostics and config and 'dls_prior_results' in config:
        print("Plotting DLS diagnostics...")
        dls_results = config['dls_prior_results']
        # Only plot for variables that had successful DLS results
        vars_with_dls = [v for v in variable_names if v in dls_results and 'diagnostics' in dls_results[v]]

        if vars_with_dls:
            fig, axes = plt.subplots(len(vars_with_dls), 2, figsize=(15, 4*len(vars_with_dls)))
            if len(vars_with_dls) == 1:
                axes = axes.reshape(1, -1) # Ensure it's a 2D array even for 1 var

            for i, var_name in enumerate(vars_with_dls):
                diagnostics = dls_results[var_name]['diagnostics']
                # Use the dates corresponding to the training data length used for DLS
                dls_dates = data.index[:len(diagnostics['trend_component'])]

                # Plot trend component
                ax_trend = axes[i, 0]
                ax_trend.plot(dls_dates, diagnostics['trend_component'], 'b-', label='DLS Trend', alpha=0.7)
                ax_trend.set_title(f'{var_name} - DLS Trend Component (Training Period)')
                ax_trend.legend()
                ax_trend.grid(True, alpha=0.3)


                # Plot cycle component
                ax_cycle = axes[i, 1]
                ax_cycle.plot(dls_dates, diagnostics['cycle_component'], 'r-', label='DLS Cycle', alpha=0.7)
                ax_cycle.set_title(f'{var_name} - DLS Cycle Component (Training Period)')
                ax_cycle.legend()
                ax_cycle.grid(True, alpha=0.3)

                # Format dates on bottom row
                if i == len(vars_with_dls) - 1:
                    ax_trend.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax_trend.xaxis.set_major_locator(mdates.YearLocator(5))
                    ax_trend.tick_params(axis='x', rotation=45)
                    ax_cycle.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax_cycle.xaxis.set_major_locator(mdates.YearLocator(5))
                    ax_cycle.tick_params(axis='x', rotation=45)
                else:
                     ax_trend.tick_params(axis='x', labelbottom=False)
                     ax_cycle.tick_params(axis='x', labelbottom=False)


            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{plot_dir}/dls_diagnostics.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
             print("No variables had successful DLS results to plot diagnostics.")


    # Plot 2: Estimated States vs Original Data (Trends) and Cycle components (if smoother succeeded)
    if plot_states_and_fitted:
        print("Plotting Estimated States...")
        # Identify state indices in the smoothed_states array
        # This order should match full_state_names_tuple in config_data
        full_state_names = config['full_state_names_tuple']
        state_indices = {name: i for i, name in enumerate(full_state_names)}

        trend_names_config = config['variables']['trend_names'] # e.g. ['trend_gdp_growth', 'trend_inflation']
        stationary_names_config = config['variables']['stationary_var_names'] # e.g. ['cycle_gdp_growth', 'cycle_inflation']

        # We only plot the primary cycle state (lag 0)
        num_trend_states_to_plot = config['k_trends']
        num_cycle_states_to_plot = config['k_stationary'] # Plot the main cycle state (t=0 lag)

        fig, axes = plt.subplots(num_trend_states_to_plot + num_cycle_states_to_plot, 1, figsize=(15, 3*(num_trend_states_to_plot + num_cycle_states_to_plot)))
        # Handle case where axes is not an array (e.g., total states = 1)
        if not isinstance(axes, np.ndarray):
             axes = np.array([axes])
        elif axes.ndim == 1:
            # Ensure it's a 1D iterable array
             pass # It's already a 1D array
        else: # Should be 2D for some reason? Reshape to 1D if needed
            axes = axes.flatten()


        plot_idx = 0

        # Plot Trend States
        for i, trend_name in enumerate(trend_names_config):
            if plot_idx >= len(axes): break # Safety break
            ax = axes[plot_idx]
            state_name_in_state_vector = trend_name # e.g., 'trend_gdp_growth'
            if state_name_in_state_vector not in state_indices:
                 print(f"Warning: Trend state '{state_name_in_state_vector}' not found in state vector names {list(state_indices.keys())}. Skipping plot.")
                 plot_idx += 1
                 continue

            state_vec_idx = state_indices[state_name_in_state_vector]

            # Plot estimated state
            ax.plot(dates, smoothed_states[:, state_vec_idx], 'b-', label='Smoothed State', linewidth=1.5, alpha=0.8)

            if simulation_results:
                mean_sim_states, median_sim_states, all_sim_draws = simulation_results
                # Check if index exists in simulation results (should be same shape as smoothed_states)
                if state_vec_idx < mean_sim_states.shape[1]:
                    ax.plot(dates, mean_sim_states[:, state_vec_idx], 'r:', label='Mean Simulation', linewidth=1.5, alpha=0.8)

                    # Add confidence bands (only if simulation draws available and sufficient)
                    if all_sim_draws is not None and all_sim_draws.shape[0] > 1: # Check if multiple draws exist
                        try:
                            state_draws = all_sim_draws[:, :, state_vec_idx]
                            lower_band = jnp.percentile(state_draws, 10, axis=0)
                            upper_band = jnp.percentile(state_draws, 90, axis=0)

                            ax.fill_between(dates, lower_band, upper_band,
                                           color='red', alpha=0.2, label='80% Simulation Band')
                        except Exception as e:
                            print(f"Warning: Could not compute bands for {state_name_in_state_vector}: {e}")
                    # else:
                         # print(f"Warning: Not enough simulation draws ({simulation_draws}) or draws array shape incorrect {all_sim_draws.shape if all_sim_draws is not None else 'None'} to compute bands for {state_name_in_state_vector}.")
                else:
                     print(f"Warning: State index {state_vec_idx} out of bounds for simulation results (shape {mean_sim_states.shape}). Cannot plot simulation mean/bands.")


            # Also plot the original variable
            original_var_name = trend_name.replace('trend_', '') # e.g., 'gdp_growth'
            if original_var_name in data.columns:
                ax2 = ax.twinx()
                ax2.plot(dates, data[original_var_name].values, 'k-', alpha=0.3, label='Original Data')
                ax2.set_ylabel('Original Data', alpha=0.7)
                ax2.legend(loc='upper right')


            ax.set_title(f'Estimated {state_name_in_state_vector}', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            if plot_idx < len(axes) - 1: # Hide x-axis labels on non-bottom plots
                 ax.tick_params(axis='x', labelbottom=False)


            plot_idx += 1

        # Plot Cycle States (only t=0 lag)
        for i, stationary_name in enumerate(stationary_names_config):
            if plot_idx >= len(axes): break # Safety break
            ax = axes[plot_idx]
            state_name_in_state_vector = stationary_name # e.g., 'cycle_gdp_growth'
            if state_name_in_state_vector not in state_indices:
                 print(f"Warning: Cycle state '{state_name_in_state_vector}' not found in state vector names {list(state_indices.keys())}. Skipping plot.")
                 plot_idx += 1
                 continue
            state_vec_idx = state_indices[state_name_in_state_vector]


            # Plot estimated state
            ax.plot(dates, smoothed_states[:, state_vec_idx], 'g-', label='Smoothed State', linewidth=1.5, alpha=0.8)

            if simulation_results:
                mean_sim_states, median_sim_states, all_sim_draws = simulation_results
                 # Check if index exists in simulation results (should be same shape as smoothed_states)
                if state_vec_idx < mean_sim_states.shape[1]:
                    ax.plot(dates, mean_sim_states[:, state_vec_idx], 'r:', label='Mean Simulation', linewidth=1.5, alpha=0.8)

                    # Add confidence bands (only if simulation draws available and sufficient)
                    if all_sim_draws is not None and all_sim_draws.shape[0] > 1: # Check if multiple draws exist
                        try:
                             # Cycle states in simulation_results are likely after trend states if ordering is consistent
                             # However, using the state_vec_idx from the full state vector is safer if simulation_results match that structure
                            state_draws = all_sim_draws[:, :, state_vec_idx]
                            lower_band = jnp.percentile(state_draws, 10, axis=0)
                            upper_band = jnp.percentile(state_draws, 90, axis=0)

                            ax.fill_between(dates, lower_band, upper_band,
                                           color='red', alpha=0.2, label='80% Simulation Band')
                        except Exception as e:
                            print(f"Warning: Could not compute bands for {state_name_in_state_vector}: {e}")
                    # else:
                         # print(f"Warning: Not enough simulation draws ({simulation_draws}) or draws array shape incorrect to compute bands for {state_name_in_state_vector}.")
                else:
                    print(f"Warning: State index {state_vec_idx} out of bounds for simulation results (shape {mean_sim_states.shape}). Cannot plot simulation mean/bands.")


            ax.set_title(f'Estimated {state_name_in_state_vector}', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            if plot_idx < len(axes) - 1: # Hide x-axis labels on non-bottom plots
                 ax.tick_params(axis='x', labelbottom=False)


            plot_idx += 1


        # Ensure x-axis labels are shown only on the very bottom plot if there are axes
        if len(axes) > 0:
             axes[-1].tick_params(axis='x', rotation=45, labelbottom=True)


        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/estimated_states.png", dpi=300, bbox_inches='tight')
        plt.show()


        # Plot 3: Data vs Fitted Values (Trend + Cycle sum)
        print("Plotting Data vs Fitted Values...")
        fig, axes = plt.subplots(len(variable_names), 1, figsize=(15, 3*len(variable_names)))
        if len(variable_names) == 1:
            axes = [axes]
        elif len(variable_names) > 1 and axes.ndim == 1:
            pass # axes is already a 1D array
        else:
             axes = axes.flatten()


        # Get indices for the t=0 trend and cycle states
        # Use .get for safety in case a state name isn't found for some reason
        trend_state_indices_in_vec = [state_indices.get(f'trend_{name}') for name in variable_names]
        cycle_state_indices_in_vec = [state_indices.get(f'cycle_{name}') for name in variable_names]

        # Check if indices were found and are valid for the smoothed_states shape
        fitted_values_available = True
        for idx in trend_state_indices_in_vec + cycle_state_indices_in_vec:
            if idx is None or idx >= smoothed_states.shape[1]:
                 fitted_values_available = False
                 break
        if not fitted_values_available:
            print("Warning: State indices for fitted values calculation not found or out of bounds for smoothed states. Cannot plot fitted values.")


        for i, var_name in enumerate(variable_names):
            if i >= len(axes): break # Safety break
            ax = axes[i]

            # Original data
            ax.plot(dates, data[var_name], 'k-', label='Observed Data', alpha=0.8, linewidth=1.5)

            # Compute fitted values (trend + cycle) if indices are valid
            if fitted_values_available:
                trend_idx_vec = trend_state_indices_in_vec[i]
                cycle_idx_vec = cycle_state_indices_in_vec[i]
                # Check if indices are within bounds of smoothed_states
                if trend_idx_vec is not None and cycle_idx_vec is not None and trend_idx_vec < smoothed_states.shape[1] and cycle_idx_vec < smoothed_states.shape[1]:
                    fitted_values = smoothed_states[:, trend_idx_vec] + smoothed_states[:, cycle_idx_vec]
                    ax.plot(dates, fitted_values, 'r--', label='Fitted Values', alpha=0.8, linewidth=1.5)
                # else: # Warning already printed if fitted_values_available is False
                    # print(f"Warning: State indices for {var_name} ({trend_idx_vec}, {cycle_idx_vec}) out of bounds for smoothed states (shape {smoothed_states.shape}). Cannot plot fitted values.")


            ax.set_title(f'{var_name} - Observed vs Fitted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(10))


            if i == len(variable_names) - 1: # Only on bottom plot
                ax.tick_params(axis='x', rotation=45, labelbottom=True)
            else:
                ax.tick_params(axis='x', labelbottom=False)


        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/data_vs_fitted.png", dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("Skipping estimated state and fitted value plots because simulation smoother failed or no states were estimated.")


# Example usage function
def example_with_real_data():
    """Example usage with CSV data loading."""

    # Load your data (replace with actual file path)
    # data = load_data_from_csv('your_data.csv', date_column='date')

    # For demonstration, create sample data
    np.random.seed(42)
    dates = pd.date_range('1960-01-01', periods=200, freq='Q')

    # Simulate realistic GDP and inflation data
    # Log-level data with trend and cycle
    log_gdp_trend = np.cumsum(np.random.normal(0.008, 0.002, 200)) + 6.0
    log_gdp_cycle = np.zeros(200)
    for i in range(1, 200):
        log_gdp_cycle[i] = 0.7 * log_gdp_cycle[i-1] + np.random.normal(0, 0.01)

    log_inf_trend = np.cumsum(np.random.normal(0.002, 0.001, 200)) + 1.5
    log_inf_cycle = np.zeros(200)
    for i in range(1, 200):
        log_inf_cycle[i] = 0.6 * log_inf_cycle[i-1] + np.random.normal(0, 0.015)

    log_data = log_gdp_trend + log_gdp_cycle
    log_inf = log_inf_trend + log_inf_cycle

    # Convert to growth rates for VAR estimation
    # Use diff with shift(1) for proper quarterly growth rate relative to previous quarter
    gdp_growth = pd.Series(log_data, index=dates).diff().values # This will have NaN at start
    inflation = pd.Series(log_inf, index=dates).diff().values # This will have NaN at start


    # Add some missing values
    gdp_growth[50:55] = np.nan
    inflation[120:123] = np.nan
    gdp_growth[180] = np.nan # Missing value near the end


    data = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'inflation': inflation
    }, index=dates)

    print("Sample data created:")
    print(data.head())
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {list(data.columns)}")

    # Run estimation with DLS priors
    results = run_bvar_estimation_with_dls(
        data=data,
        variable_names=['gdp_growth', 'inflation'],
        training_fraction=0.25, # Use first 25% for DLS priors
        var_order=1,
        mcmc_params={
            'num_warmup': 200,
            'num_samples': 300,
            'num_chains': 2
        },
        dls_params={ # These will now be passed to the DLS function
            'discount_factor': 0.95, # Lower discount factor -> more responsive trend
            # Try slightly weaker alpha to potentially avoid zero variance
            'alpha_shape': 2.1, # IG variance finite but large if beta is non-zero
        },
        simulation_draws=50,
        save_config=True,
        config_filename='bvar_dls_example.yml'
    )

    # Create plots
    plot_estimation_results(
        results=results,
        data=data,
        plot_diagnostics=True,
        save_plots=True,
        plot_dir='estimation_plots'
    )

    # You can optionally print posterior means here
    if results and results.get('mcmc_results') and results['mcmc_results'].get('posterior_samples'):
        posterior = results['mcmc_results']['posterior_samples']
        print("\n--- Posterior Sample Means ---")
        for param_name in sorted(posterior.keys()):
             try:
                 if posterior[param_name].ndim == 0:
                      print(f"{param_name}: {jnp.mean(posterior[param_name]):.6f}") # Increased precision
                 elif posterior[param_name].ndim == 1:
                      print(f"{param_name}: {jnp.mean(posterior[param_name]):.6f} (vector mean)")
                 else:
                      print(f"{param_name}: shape={posterior[param_name].shape}, mean=\n{jnp.mean(posterior[param_name], axis=0)}")
             except Exception as e:
                  print(f"Could not print mean for {param_name}: {e}")


    return results

if __name__ == "__main__":
    results = example_with_real_data()