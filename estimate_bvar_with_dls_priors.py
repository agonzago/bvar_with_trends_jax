# estimate_bvar_with_dls_priors_fixed_online_smoother.py
# Comprehensive fix for I(2) simulation, parameter mapping, Canova DLS,
# AND Online Simulation Smoother for memory efficiency.

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
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy import linalg
from scipy.optimize import minimize_scalar

# Configure JAX
jax.config.update("jax_enable_x64", True)
# Keep JAX platform flexible, but for pure Python loops calling JIT, CPU can be simpler initially.
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64

# *** Refined device count setting ***
try:
    import multiprocessing
    # Attempt to get physical CPU count or logical if physical is not available
    num_cpu = multiprocessing.cpu_count() 
    # Set host device count, ensuring it's at least 1 and not excessively large
    numpyro.set_host_device_count(min(num_cpu, 8)) # Cap at 8 for safety/common hardware
except Exception as e:
    print(f"Could not set host device count: {e}. Falling back to default (likely 1 or 4).")
    # If setting fails, numpyro will use its default, which is usually okay.
    pass


# Import your existing modules
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax
# from core.simulate_bvar_jax import simulate_bvar_with_trends_jax # This is the function being called now
from core.var_ss_model import (
    numpyro_bvar_stationary_model, 
    parse_initial_state_config, 
    _parse_equation_jax, 
    _get_off_diagonal_indices,
    load_config_and_prepare_jax_static_args, 
    build_state_space_matrices_jit 
)
# from core.run_single_draw import run_simulation_smoother_single_params_jit # Replaced by new online logic
# from core.parameter_extraction_for_smoother import extract_smoother_parameters, validate_smoother_parameters # Not strictly needed now
# Need KalmanFilter for initial smoothing of original data
from utils.Kalman_filter_jax import KalmanFilter
# Need HybridDKSimulationSmoother for its internal filter/smoother used in simulation path
from utils.hybrid_dk_smoother import HybridDKSimulationSmoother

# Import the new smoother utilities
from core.smoother_utils import (
    extract_smoother_parameters_single_draw,
    construct_ss_matrices_from_smoother_params,
    OnlineQuantileEstimator, # Although we use its static methods
    run_single_simulation_path_for_dk,
    jit_run_single_simulation_path_for_dk_wrapper # Need this for JIT compilation
)

# *** Import the convert_to_hashable function from utils.jax_utils ***
from utils.jax_utils import convert_to_hashable


# --- Enhanced DLS Implementation following Canova (2014) ---
# *** RESTORED ***
class CanovaDLS:
    """
    Canova (2014) Dynamic Linear Smoothing implementation.
    Uses structural time series approach with optimal smoothness parameter.
    """
    
    def __init__(self, smoothness_range=(1e-6, 1e2), optimize_smoothness=True):
        self.smoothness_range = smoothness_range
        self.optimize_smoothness = optimize_smoothness
        
    def _build_smoothing_matrices(self, T: int, lambda_smooth: float):
        """Build smoothing matrices for trend extraction."""
        # Second difference matrix for smoothness penalty
        D2 = np.zeros((T-2, T))
        for i in range(T-2):
            D2[i, i:i+3] = [1, -2, 1]
        
        # Identity matrix
        I_T = np.eye(T)
        
        # Smoothing matrix: (I + λD'D)^(-1}
        penalty_matrix = lambda_smooth * (D2.T @ D2)
        smoothing_matrix = I_T + penalty_matrix
        
        return smoothing_matrix, D2
    
    def _log_likelihood_smoothness(self, lambda_smooth: float, y: np.ndarray):
        """Log-likelihood for smoothness parameter optimization."""
        T = len(y)
        if T < 3:
            return -np.inf
            
        smoothing_matrix, D2 = self._build_smoothing_matrices(T, lambda_smooth)
        
        try:
            # Solve (I + λD'D)τ = y for trend τ
            trend = linalg.solve(smoothing_matrix, y)
            
            # Compute residuals
            residuals = y - trend
            
            # Compute second differences of trend (smoothness measure)
            if T > 2:
                trend_second_diff = D2 @ trend
                smoothness_penalty = np.sum(trend_second_diff**2)
            else:
                smoothness_penalty = 0
            
            # Residual variance
            sigma2_residual = np.var(residuals) if len(residuals) > 1 else 1e-6
            
            # Approximate log-likelihood (Canova's criterion)
            log_det_term = np.log(np.linalg.det(smoothing_matrix + 1e-12 * np.eye(T)))
            ll = -0.5 * (T * np.log(sigma2_residual) + 
                        lambda_smooth * smoothness_penalty / sigma2_residual + 
                        log_det_term)
            
            return ll if np.isfinite(ll) else -np.inf
            
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
    
    def extract_trend_cycle(self, y: np.ndarray, lambda_smooth: Optional[float] = None):
        """
        Extract trend and cycle using Canova's DLS method.
        
        Args:
            y: Time series data
            lambda_smooth: Smoothness parameter (if None, will be optimized)
            
        Returns:
            tuple: (trend, cycle, optimal_lambda, trend_variance, cycle_variance)
        """
        y_clean = np.asarray(y).flatten()
        T = len(y_clean)
        
        if T < 3:
            return (np.full(T, np.nan), np.full(T, np.nan), 
                   1.0, 1e-6, 1e-6)
        
        # Handle missing values
        valid_mask = np.isfinite(y_clean)
        if not np.any(valid_mask):
            return (np.full(T, np.nan), np.full(T, np.nan), 
                   1.0, 1e-6, 1e-6)
        
        # Optimize smoothness parameter if not provided
        if lambda_smooth is None and self.optimize_smoothness:
            try:
                result = minimize_scalar(
                    lambda lam: -self._log_likelihood_smoothness(lam, y_clean[valid_mask]),
                    bounds=self.smoothness_range,
                    method='bounded'
                )
                optimal_lambda = result.x if result.success else 1.0
            except:
                optimal_lambda = 1.0
        else:
            optimal_lambda = lambda_smooth if lambda_smooth is not None else 1.0
        
        # Extract trend using optimal smoothness
        try:
            smoothing_matrix, _ = self._build_smoothing_matrices(np.sum(valid_mask), optimal_lambda)
            trend_valid = linalg.solve(smoothing_matrix, y_clean[valid_mask])
            
            # Interpolate trend for missing values
            trend = np.full(T, np.nan)
            trend[valid_mask] = trend_valid
            
            # Linear interpolation for missing trend values
            if not np.all(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                missing_indices = np.where(~valid_mask)[0]
                trend[missing_indices] = np.interp(missing_indices, valid_indices, trend_valid)
            
            # Compute cycle
            cycle = y_clean - trend
            
            # Estimate variances
            trend_diff = np.diff(trend[np.isfinite(trend)])
            trend_variance = np.var(trend_diff) if len(trend_diff) > 1 else 1e-6
            
            cycle_valid = cycle[np.isfinite(cycle)]
            cycle_variance = np.var(cycle_valid) if len(cycle_valid) > 1 else 1e-6
            
            # Ensure positive variances
            trend_variance = max(float(trend_variance), 1e-9)
            cycle_variance = max(float(cycle_variance), 1e-9)
            
            return trend, cycle, optimal_lambda, trend_variance, cycle_variance
            
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"DLS extraction failed: {e}")
            # Fallback to simple HP filter approximation
            from scipy import signal
            try:
                # Simple HP filter with fixed smoothness
                b, a = signal.butter(4, 0.1, 'low')
                trend = signal.filtfilt(b, a, y_clean)
                cycle = y_clean - trend
                
                trend_var = np.var(np.diff(trend)) if len(trend) > 1 else 1e-6
                cycle_var = np.var(cycle) if len(cycle) > 1 else 1e-6
                
                return trend, cycle, optimal_lambda, max(trend_var, 1e-9), max(cycle_var, 1e-9)
            except:
                # Ultimate fallback
                mean_y = np.nanmean(y_clean)
                trend = np.full(T, mean_y)
                cycle = y_clean - trend
                return trend, cycle, optimal_lambda, 1e-6, 1e-6

# *** RESTORED ***
def suggest_ig_priors(empirical_var: float, alpha: float = 2.5) -> Dict[str, float]:
    """Inverse Gamma prior suggestion with safety checks and reasonable minimum variance."""
    if alpha <= 1:
        alpha = 2.5
    
    # Ensure minimum empirical variance to avoid overly tight priors
    min_empirical_var = 1e-6
    empirical_var = max(float(empirical_var), min_empirical_var)
    
    beta = empirical_var * (alpha - 1.0)
    beta = max(float(beta), 1e-9)
    
    implied_mean = float(beta / (alpha - 1.0)) if alpha > 1 else float('inf')
    implied_variance = float(beta**2 / ((alpha - 1.0)**2 * (alpha - 2.0))) if alpha > 2 else float('inf')
    
    # Double-check that implied mean is reasonable
    if implied_mean < min_empirical_var:
        print(f"Warning: IG prior implies very small mean ({implied_mean:.2e}). Adjusting beta.")
        beta = min_empirical_var * (alpha - 1.0)
        implied_mean = float(beta / (alpha - 1.0))
        implied_variance = float(beta**2 / ((alpha - 1.0)**2 * (alpha - 2.0))) if alpha > 2 else float('inf')
    
    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'implied_mean': implied_mean,
        'implied_variance': implied_variance
    }

# --- Fixed create_config_with_canova_dls Function ---

def create_config_with_canova_dls(data: pd.DataFrame,
                                       variable_names: Optional[List[str]] = None,
                                       training_fraction: float = 0.3,
                                       var_order: int = 1,
                                       dls_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create BVAR configuration using Canova DLS priors.
    FIXED: Removes infinite recursion by implementing the actual DLS logic.
    """
    if variable_names is None:
        variable_names = list(data.columns)
    
    if dls_params is None:
        dls_params = {
            'optimize_smoothness': True,
            'smoothness_range': (1e-6, 1e2),
            'alpha_shape': 2.5,
        }
    
    print("="*80)
    print("APPLYING CANOVA (2014) DLS PRIOR ELICITATION")
    print("="*80)
    
    # Initialize DLS processor
    dls_processor = CanovaDLS(
        smoothness_range=dls_params.get('smoothness_range', (1e-6, 1e2)),
        optimize_smoothness=dls_params.get('optimize_smoothness', True)
    )
    
    # Apply DLS to each variable in the training data
    training_size = int(len(data) * training_fraction)
    training_data = data.iloc[:training_size]
    
    prior_results = {}
    alpha_shape = dls_params.get('alpha_shape', 2.5)
    
    for var_name in variable_names:
        print(f"\nProcessing variable: {var_name}")
        
        # Get training data for this variable
        var_data = training_data[var_name].dropna()
        
        if len(var_data) < 10:  # Need minimum data points
            print(f"Warning: Insufficient data for {var_name}. Using default priors.")
            prior_results[var_name] = {
                'trend_shocks': suggest_ig_priors(1.0, alpha_shape),
                'cycle_shocks': suggest_ig_priors(1.0, alpha_shape),
                'initial_conditions': {
                    'trend_mean': float(var_data.iloc[0]) if len(var_data) > 0 else 0.0,
                    'trend_variance': 1.0,
                    'cycle_mean': 0.0,
                    'cycle_variance': 1.0
                },
                'diagnostics': {
                    'optimal_lambda': 1.0,
                    'extracted_trend_var': 1.0,
                    'extracted_cycle_var': 1.0,
                    'trend_component': None,
                    'cycle_component': None
                }
            }
            continue
        
        # Apply DLS trend-cycle decomposition
        try:
            trend, cycle, optimal_lambda, trend_var, cycle_var = dls_processor.extract_trend_cycle(
                var_data.values
            )
            
            # Create prior specifications
            trend_prior = suggest_ig_priors(trend_var, alpha_shape)
            cycle_prior = suggest_ig_priors(cycle_var, alpha_shape)
            
            # Store results
            prior_results[var_name] = {
                'trend_shocks': trend_prior,
                'cycle_shocks': cycle_prior,
                'initial_conditions': {
                    'trend_mean': float(trend[0]) if not np.isnan(trend[0]) else 0.0,
                    'trend_variance': max(float(trend_var), 1e-6),
                    'cycle_mean': float(cycle[0]) if not np.isnan(cycle[0]) else 0.0,
                    'cycle_variance': max(float(cycle_var), 1e-6)
                },
                'diagnostics': {
                    'optimal_lambda': float(optimal_lambda),
                    'extracted_trend_var': float(trend_var),
                    'extracted_cycle_var': float(cycle_var),
                    'trend_component': trend.tolist() if trend is not None else None,
                    'cycle_component': cycle.tolist() if cycle is not None else None
                }
            }
            
            print(f"  Optimal λ: {optimal_lambda:.2e}")
            print(f"  Trend variance: {trend_var:.6f}")
            print(f"  Cycle variance: {cycle_var:.6f}")
            
        except Exception as e:
            print(f"Warning: DLS failed for {var_name}: {e}. Using defaults.")
            prior_results[var_name] = {
                'trend_shocks': suggest_ig_priors(1.0, alpha_shape),
                'cycle_shocks': suggest_ig_priors(1.0, alpha_shape),
                'initial_conditions': {
                    'trend_mean': float(var_data.iloc[0]),
                    'trend_variance': 1.0,
                    'cycle_mean': 0.0,
                    'cycle_variance': 1.0
                },
                'diagnostics': {
                    'optimal_lambda': 1.0,
                    'extracted_trend_var': 1.0,
                    'extracted_cycle_var': 1.0,
                    'trend_component': None,
                    'cycle_component': None
                }
            }
    
    # Create configuration structure
    temp_config_structure = {
        'var_order': var_order,
        'variables': {
            'observables': variable_names,
            'trends': [f'trend_{name}' for name in variable_names],
            'stationary': [f'cycle_{name}' for name in variable_names],
        },
        'model_equations': {name: f'trend_{name} + cycle_{name}' for name in variable_names},
        'initial_conditions': {'states': {}},
        'stationary_prior': {
            'hyperparameters': {'es': [0.7, 0.15], 'fs': [0.2, 0.15]},
            'covariance_prior': {'eta': 1.5},
            'stationary_shocks': {}
        },
        'trend_shocks': {'trend_shocks': {}},
        'parameters': {'measurement': []},
    }
    
    # Integrate DLS results into the configuration
    for var_name in variable_names:
        if var_name in prior_results:
            result = prior_results[var_name]
            init_conds = result['initial_conditions']
            
            # Set initial conditions
            temp_config_structure['initial_conditions']['states'][f'trend_{var_name}'] = {
                'mean': init_conds['trend_mean'],
                'var': init_conds['trend_variance']
            }
            temp_config_structure['initial_conditions']['states'][f'cycle_{var_name}'] = {
                'mean': init_conds['cycle_mean'],
                'var': init_conds['cycle_variance'],
            }
            
            # Set shock priors
            cycle_prior = result['cycle_shocks']
            temp_config_structure['stationary_prior']['stationary_shocks'][f'cycle_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': cycle_prior['alpha'],
                    'beta': cycle_prior['beta'],
                }
            }
            
            trend_prior = result['trend_shocks']
            temp_config_structure['trend_shocks']['trend_shocks'][f'trend_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': trend_prior['alpha'],
                    'beta': trend_prior['beta'],
                }
            }
    
    # Save temporary config and parse it
    temp_config_path = "temp_dls_config.yml"
    try:
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config_structure, f, default_flow_style=False, sort_keys=False)
        
        # Load and parse the temporary config
        config_data = load_config_and_prepare_jax_static_args(temp_config_path)
        
        # Add the raw DLS results for diagnostics
        config_data['canova_dls_results'] = prior_results

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print("="*80)
    print("CANOVA DLS PRIOR ELICITATION AND CONFIG PARSING COMPLETED")
    print("="*80)
    
    return config_data

# --- I(2) Data Simulation for BVAR Estimation ---
# *** RESTORED function definition ***
def simulate_i2_economic_data(dates: pd.DatetimeIndex, 
                             variables: List[str],
                             seed: int = 42) -> pd.DataFrame:
    """
    Simulate I(2) economic data that needs to be first-differenced.
    
    Args:
        dates: DatetimeIndex for the time series
        variables: List of variable names
        seed: Random seed
        
        Returns:
        DataFrame with I(2) level data and corresponding growth rates
    """
    np.random.seed(seed)
    T = len(dates)
    
    # Simulate I(2) processes (levels with stochastic trends)
    data_levels = {}
    data_growth = {}
    
    for var in variables:
        # I(2) simulation: trend has its own trend (stochastic)
        
        # Level 1: Stochastic trend of the trend
        trend_drift = np.random.normal(0, 0.002, T)  # Very small innovations to trend drift
        trend_level = np.cumsum(trend_drift) + 0.008  # Base trend around 0.8% quarterly
        
        # Level 2: Stochastic trend (integrated once)
        trend_innovations = np.random.normal(0, 0.005, T)
        stochastic_trend = np.cumsum(trend_level + trend_innovations)
        
        # Level 3: Final I(2) level (integrated twice)
        cycle_component = np.zeros(T)
        for t in range(1, T):
            cycle_component[t] = 0.7 * cycle_component[t-1] + np.random.normal(0, 0.01)
        
        # I(2) level series
        if var == 'gdp_growth':
            base_level = 10.0  # Log GDP level
            shock_scale = 0.008
        elif var == 'inflation':
            base_level = 2.0   # Inflation level
            shock_scale = 0.015
        else:
            base_level = 5.0
            shock_scale = 0.01
            
        level_series = base_level + stochastic_trend + cycle_component
        
        # Add variable-specific shocks
        level_series += np.random.normal(0, shock_scale, T)
        
        # Store levels and compute growth rates (first differences)
        data_levels[f'{var}_level'] = level_series
        
        # First difference to get growth rates (what we actually estimate)
        growth_rate = np.diff(level_series)
        growth_rate = np.concatenate([[np.nan], growth_rate])  # Pad with NaN for first observation
        
        data_growth[var] = growth_rate
    
    # Combine into DataFrame
    all_data = {**data_levels, **data_growth}
    df = pd.DataFrame(all_data, index=dates)
    
    # Add some realistic missing values
    if 'gdp_growth' in df.columns:
        df.loc[df.index[50:55], 'gdp_growth'] = np.nan
    if 'inflation' in df.columns:
        df.loc[df.index[120:123], 'inflation'] = np.nan
        
    return df


# --- Parameter Mapping for Smoother (Keep this section as is) ---
# NOTE: This function is used to map MCMC output *names* to smoother *names*,
# not to structure the parameters for the new online smoother logic.
# The online smoother extracts matrices directly from deterministic sites.
# This function might still be useful for debugging or if you need
# posterior means of parameters *before* they are used to build matrices.
# Let's keep it but be mindful of its role.

def extract_posterior_params_for_smoother(posterior_samples: Dict, 
                                        config_data: Dict,
                                        base_variable_names: List[str]) -> Dict[str, jax.Array]:
    """
    Extract posterior mean parameters with correct naming for the smoother.
    
    Key fix: Maps between full state names (used in MCMC) and base variable names (expected by smoother).
    """
    posterior_mean_params = {}
    
    print(f"Available MCMC parameters: {list(posterior_samples.keys())}")
    print(f"Base variable names: {base_variable_names}")
    print(f"Stationary var names from config: {config_data['variables']['stationary_var_names']}")
    print(f"Trend names with shocks: {config_data.get('trend_names_with_shocks', [])}")
    
    # Always include A_diag
    if 'A_diag' in posterior_samples:
        posterior_mean_params['A_diag'] = jnp.mean(posterior_samples['A_diag'], axis=0)
        print(f"Added A_diag with shape: {posterior_mean_params['A_diag'].shape}")
    
    # Include A_offdiag if present
    if config_data.get('num_off_diag', 0) > 0 and 'A_offdiag' in posterior_samples:
        posterior_mean_params['A_offdiag'] = jnp.mean(posterior_samples['A_offdiag'], axis=0)
        print(f"Added A_offdiag with shape: {posterior_mean_params['A_offdiag'].shape}")
    
    # Include stationary_chol if present (for multivariate stationary shocks)
    if config_data['k_stationary'] > 1 and 'stationary_chol' in posterior_samples:
        posterior_mean_params['stationary_chol'] = jnp.mean(posterior_samples['stationary_chol'], axis=0)
        print(f"Added stationary_chol with shape: {posterior_mean_params['stationary_chol'].shape}")
    
    # Map stationary shock variances: from 'stationary_var_cycle_VAR' to 'stationary_var_VAR'
    stationary_var_names_config = config_data['variables']['stationary_var_names']
    for full_state_name in stationary_var_names_config:  # e.g., 'cycle_gdp_growth'
        mcmc_param_name = f'stationary_var_{full_state_name}'  # e.g., 'stationary_var_cycle_gdp_growth'
        
        if mcmc_param_name in posterior_samples:
            # Extract base variable name: 'cycle_gdp_growth' -> 'gdp_growth'
            if full_state_name.startswith('cycle_'):
                base_var_name = full_state_name[6:]  # Remove 'cycle_' prefix
                smoother_param_name = f'stationary_var_{base_var_name}'  # e.g., 'stationary_var_gdp_growth'
                posterior_mean_params[smoother_param_name] = jnp.mean(posterior_samples[mcmc_param_name], axis=0)
                print(f"Mapped {mcmc_param_name} -> {smoother_param_name}")
            else:
                # If it doesn't start with cycle_, just use it directly but map to base name
                print(f"Warning: Stationary state name '{full_state_name}' doesn't start with 'cycle_'")
                # Try to match with base variable names
                for base_var in base_variable_names:
                    if base_var in full_state_name:
                        smoother_param_name = f'stationary_var_{base_var}'
                        posterior_mean_params[smoother_param_name] = jnp.mean(posterior_samples[mcmc_param_name], axis=0)
                        print(f"Mapped {mcmc_param_name} -> {smoother_param_name} (by matching)")
                        break
    
    # Map trend shock variances: from 'trend_var_trend_VAR' to 'trend_var_VAR'  
    trend_names_with_shocks = config_data.get('trend_names_with_shocks', [])
    for full_state_name in trend_names_with_shocks:  # e.g., 'trend_gdp_growth'
        mcmc_param_name = f'trend_var_{full_state_name}'  # e.g., 'trend_var_trend_gdp_growth'
        
        if mcmc_param_name in posterior_samples:
            # Extract base variable name: 'trend_gdp_growth' -> 'gdp_growth'
            if full_state_name.startswith('trend_'):
                base_var_name = full_state_name[6:]  # Remove 'trend_' prefix
                smoother_param_name = f'trend_var_{base_var_name}'  # e.g., 'trend_var_gdp_growth'
                posterior_mean_params[smoother_param_name] = jnp.mean(posterior_samples[mcmc_param_name], axis=0)
                print(f"Mapped {mcmc_param_name} -> {smoother_param_name}")
            else:
                # If it doesn't start with trend_, try to match with base variable names
                print(f"Warning: Trend state name '{full_state_name}' doesn't start with 'trend_'")
                for base_var in base_variable_names:
                    if base_var in full_state_name:
                        smoother_param_name = f'trend_var_{base_var}'
                        posterior_mean_params[smoother_param_name] = jnp.mean(posterior_samples[mcmc_param_name], axis=0)
                        print(f"Mapped {mcmc_param_name} -> {smoother_param_name} (by matching)")
                        break
    
    print(f"Final smoother parameters: {list(posterior_mean_params.keys())}")
    
    # Check for very small variance parameters that might cause numerical issues
    small_variance_threshold = 1e-6
    for param_name, param_value in posterior_mean_params.items():
        if 'var_' in param_name:
            if isinstance(param_value, jnp.ndarray) and param_value.ndim == 0:
                val = float(param_value)
                if val < small_variance_threshold:
                    print(f"WARNING: Very small variance parameter {param_name}: {val:.2e}")
                    # Optionally add a floor to prevent numerical issues
                    posterior_mean_params[param_name] = jnp.array(max(val, small_variance_threshold), dtype=_DEFAULT_DTYPE)
                    print(f"  Adjusted to: {float(posterior_mean_params[param_name]):.2e}")
    
    return posterior_mean_params

# Replace the extract_smoother_parameters_single_draw function with this fixed version:

def extract_smoother_parameters_single_draw(
    posterior_samples: Dict[str, jax.Array],
    draw_idx: int,
    config_data: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """
    FIXED: Extracts smoother parameters handling phi_list correctly.
    """
    smoother_params = {}
    
    # Handle regular deterministic arrays
    regular_deterministic_keys = [
        "Sigma_cycles", "Sigma_trends_full", 
        "init_x_comp", "init_P_comp"
    ]
    
    for key in regular_deterministic_keys:
        if key in posterior_samples:
            param_data = posterior_samples[key]
            if hasattr(param_data, 'shape') and len(param_data.shape) > 0:
                if param_data.shape[0] > draw_idx:
                    smoother_params[key] = param_data[draw_idx]
                else:
                    raise ValueError(f"{key} has only {param_data.shape[0]} samples, need draw {draw_idx}")
            else:
                # Scalar case
                smoother_params[key] = param_data
        else:
            raise ValueError(f"Required key '{key}' not found in posterior_samples")
    
    # Handle phi_list specially (it's a list of arrays, not a single array)
    if 'phi_list' in posterior_samples:
        phi_list_data = posterior_samples['phi_list']
        
        if isinstance(phi_list_data, list):
            # phi_list is a list of (num_samples, p, k_stat, k_stat) arrays
            # We need to extract the draw_idx-th sample from each element
            phi_list_for_draw = []
            for phi_i in phi_list_data:
                if hasattr(phi_i, 'shape') and len(phi_i.shape) > 0:
                    if phi_i.shape[0] > draw_idx:
                        phi_list_for_draw.append(phi_i[draw_idx])
                    else:
                        raise ValueError(f"phi_list element has only {phi_i.shape[0]} samples, need draw {draw_idx}")
                else:
                    phi_list_for_draw.append(phi_i)
            smoother_params['phi_list'] = phi_list_for_draw
        else:
            # If phi_list is stored as a single array somehow
            if hasattr(phi_list_data, 'shape') and len(phi_list_data.shape) > 0:
                smoother_params['phi_list'] = phi_list_data[draw_idx]
            else:
                smoother_params['phi_list'] = phi_list_data
    else:
        raise ValueError("Required key 'phi_list' not found in posterior_samples")
    
    # Handle measurement parameters
    measurement_param_names = config_data.get('measurement_param_names_tuple', ())
    smoother_params['measurement_params'] = {}
    
    for param_name in measurement_param_names:
        if param_name in posterior_samples:
            param_data = posterior_samples[param_name]
            if hasattr(param_data, 'shape') and len(param_data.shape) > 0:
                if param_data.shape[0] > draw_idx:
                    smoother_params['measurement_params'][param_name] = param_data[draw_idx]
                else:
                    smoother_params['measurement_params'][param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
            else:
                smoother_params['measurement_params'][param_name] = param_data
        else:
            smoother_params['measurement_params'][param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)
    
    return smoother_params

def run_bvar_estimation_with_fixes(data: pd.DataFrame,
                                  variable_names: Optional[List[str]] = None,
                                  training_fraction: float = 0.3,
                                  var_order: int = 1,
                                  mcmc_params: Optional[Dict] = None,
                                  dls_params: Optional[Dict] = None,
                                  simulation_draws_per_mcmc: int = 100,
                                  online_smoother_buffer_size: int = 1000,
                                  save_config: bool = True,
                                  config_filename: str = "bvar_canova_dls.yml") -> Dict[str, Any]:
    """
    Complete BVAR estimation with all fixes:
    1. I(2) data simulation with proper differencing
    2. Canova (2014) DLS prior elicitation  
    3. Fixed parameter mapping for smoother
    4. Online Simulation Smoother for memory efficiency.
    FIXED: Properly handles all variable scoping and NonConcreteBooleanIndexError.
    """
    
    if variable_names is None:
        variable_names = list(data.columns)
    
    if mcmc_params is None:
        mcmc_params = {
            'num_warmup': 300,
            'num_samples': 500,
            'num_chains': 2
        }

    # Initialize results dictionary early to ensure data_info is always included
    results = {
        'config': None,
        'mcmc_results': None,
        'smoothing_results': None,
        'data_info': {
            'data_shape': data[variable_names].values.shape,
            'variable_names': variable_names,
            'date_range': (data.index[0], data.index[-1])
        },
        'error': None
    }
    
    print("="*100)
    print("BVAR ESTIMATION WITH CANOVA DLS AND I(2) DATA HANDLING")
    print("="*100)
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {variable_names}")
    print(f"VAR order: {var_order}")
    print(f"Training fraction: {training_fraction:.1%}")
    
    # Step 1: Create configuration with Canova DLS priors and parse it
    print("\nStep 1: Generating Canova DLS priors and parsing config...")
    try:
        config_data = create_config_with_canova_dls( 
            data=data,
            variable_names=variable_names,
            training_fraction=training_fraction,
            var_order=var_order,
            dls_params=dls_params
        )
        results['config'] = config_data
    except Exception as e:
         print(f"\nERROR: Configuration generation failed: {e}")
         results['error'] = f"Config generation failed: {e}"
         import traceback
         traceback.print_exc()
         return results

    # Step 2: Save configuration if requested
    if save_config:
        yaml_config_to_save = {
            'var_order': config_data['var_order'],
            'variables': {
                'observables': list(config_data['observable_names']),
                'trends': list(config_data['trend_var_names']),
                'stationary': list(config_data['stationary_var_names'])
            },
            'model_equations': config_data['raw_config_model_eqs_str_dict'],
            'initial_conditions': config_data.get('raw_config_initial_conds', {}),
            'stationary_prior': config_data.get('raw_config_stationary_prior', {}),
            'trend_shocks': config_data.get('raw_config_trend_shocks', {}),
            'parameters': config_data.get('raw_config_measurement_params', []),
        }
        
        try:
            with open(config_filename, 'w') as f:
                yaml.dump(yaml_config_to_save, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {config_filename}")
        except Exception as e:
            print(f"Warning: Could not save configuration: {e}")
    
    # Step 3: Prepare data for estimation
    y_data = data[variable_names].values.astype(_DEFAULT_DTYPE)
    T_obs = y_data.shape[0]
    k_endog = config_data['k_endog']
    k_states = config_data['k_states']
    
    print(f"\nStep 3: Preparing data...")
    print(f"Data shape: {y_data.shape}")
    print(f"Time steps: {T_obs}, Observables: {k_endog}, States: {k_states}")
    
    # Handle observations for MCMC
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y_data), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]
    
    if static_n_obs_actual == 0:
        results['error'] = "No valid observation columns found."
        raise ValueError("No valid observation columns found.")

    # Step 4: Run MCMC estimation
    print(f"\nStep 4: Running MCMC estimation...")
    
    model_args = {
        'y': y_data,
        'config_data': config_data,
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'trend_var_names': config_data['trend_var_names'],
        'stationary_var_names': config_data['stationary_var_names'],
        'observable_names': config_data['observable_names'],
    }
    
    kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=numpyro.infer.init_to_sample())
    mcmc_params_actual = {**{'num_warmup': 300, 'num_samples': 500, 'num_chains': 2}, **(mcmc_params or {})}
    mcmc = MCMC(kernel, **mcmc_params_actual)
    
    key = random.PRNGKey(42)
    key_mcmc, key_smoother_global = random.split(key)

    start_time = time.time()
    try:
        mcmc.run(key_mcmc, **model_args)
        mcmc_time = time.time() - start_time
        print(f"\nMCMC completed in {mcmc_time:.2f} seconds")
        posterior_samples = mcmc.get_samples()
        mcmc_extras = mcmc.get_extra_fields()
        num_mcmc_samples = mcmc_params_actual['num_samples'] * mcmc_params_actual['num_chains']
        print(f"Extracted {num_mcmc_samples} MCMC samples.")
        results['mcmc_results'] = {'posterior_samples': posterior_samples, 'mcmc_time': mcmc_time, 'mcmc_summary': mcmc_extras}
    except Exception as e:
        print(f"\nERROR: MCMC estimation failed: {e}")
        results['error'] = f"MCMC failed: {e}"
        import traceback
        traceback.print_exc()
        return results
    
    # Step 5: Run standard RTS smoother on original data (using posterior mean parameters)
    print(f"\nStep 5: Running standard RTS smoother on original data (posterior mean)...")
    
    try:
        # Extract posterior mean parameters for building SS matrices
        posterior_mean_params_for_rts_builder = {}
        
        params_expected_by_builder = ['A_diag']
        if config_data['num_off_diag'] > 0: 
            params_expected_by_builder.append('A_offdiag')
        if config_data['k_stationary'] > 1: 
            params_expected_by_builder.append('stationary_chol')
        params_expected_by_builder.extend([f'stationary_var_{name}' for name in config_data['stationary_var_names']])
        params_expected_by_builder.extend([f'trend_var_{name}' for name in config_data['trend_names_with_shocks']])
        params_expected_by_builder.extend(list(config_data['measurement_param_names_tuple']))

        for param_name in params_expected_by_builder:
            if param_name in posterior_samples:
                 mean_val = jnp.mean(jnp.asarray(posterior_samples[param_name], dtype=_DEFAULT_DTYPE), axis=0)
                 posterior_mean_params_for_rts_builder[param_name] = mean_val

        # Build state-space matrices using posterior mean parameters
        ss_matrices_posterior_mean = build_state_space_matrices_jit(
            posterior_mean_params_for_rts_builder, config_data
        )
        
        # Run filter and smoother using KalmanFilter
        kf_mean = KalmanFilter(
            ss_matrices_posterior_mean['T_comp'],
            ss_matrices_posterior_mean['R_comp'], 
            ss_matrices_posterior_mean['C_comp'],
            ss_matrices_posterior_mean['H_comp'],
            ss_matrices_posterior_mean['init_x_comp'], 
            ss_matrices_posterior_mean['init_P_comp'] 
        )
        
        static_C_obs_mean = ss_matrices_posterior_mean['C_comp'][static_valid_obs_idx, :]
        static_H_obs_mean = ss_matrices_posterior_mean['H_comp'][static_valid_obs_idx[:, None], static_valid_obs_idx]
        static_I_obs_mean = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)

        filter_results_original = kf_mean.filter(
            y_data, static_valid_obs_idx, static_n_obs_actual, 
            static_C_obs_mean, static_H_obs_mean, static_I_obs_mean
        )
        
        x_smooth_original_dense, _ = kf_mean.smooth(
            y_data, filter_results_original, static_valid_obs_idx, static_n_obs_actual,
             static_C_obs_mean, static_H_obs_mean, static_I_obs_mean
        )
        print("Standard RTS smoothing on original data (posterior mean) completed.")
        
        results['smoothing_results'] = {'smoothed_states_original_mean_rts': x_smooth_original_dense}

    except Exception as e:
        print(f"\nERROR: Standard RTS smoothing on posterior mean failed: {e}")
        results['error'] = f"Standard RTS smoothing failed: {e}"
        import traceback
        traceback.print_exc()
        return results 

    # Step 6: Initialize Online Quantile Estimators Grid
    print(f"\nStep 6: Initializing online quantile estimators ({T_obs} time steps x {k_states} states)...")
    
    buffer_grid = jnp.full(
        (T_obs, k_states, online_smoother_buffer_size), 
        jnp.nan, 
        dtype=_DEFAULT_DTYPE
    )
    count_grid = jnp.zeros((T_obs, k_states), dtype=jnp.int32)
    
    quantiles_to_track = [0.025, 0.5, 0.975]
    
    # Step 7: Main Processing Loop for Online Simulation Smoothing
    print(f"\nStep 7: Running online simulation smoother ({num_mcmc_samples} MCMC samples x {simulation_draws_per_mcmc} smoother draws)...")
    
    total_simulation_draws = num_mcmc_samples * simulation_draws_per_mcmc
    processed_draws = 0
    start_smooth_sim_time = time.time()
    
    # Generate keys for each simulation draw
    all_smoother_keys = random.split(key_smoother_global, total_simulation_draws)

    key_idx = 0
    
    # Convert config_data to hashable format for JIT
    # hashable_config_data = convert_to_hashable(config_data)
    # jit_run_single_simulation_path = jax.jit(jit_run_single_simulation_path_for_dk_wrapper, static_argnames=['static_config_data'])

    for mcmc_draw_idx in range(num_mcmc_samples):
        try:
            smoother_params_single_draw = extract_smoother_parameters_single_draw(
                posterior_samples, mcmc_draw_idx, config_data 
            )
        except Exception as e:
            print(f"\nWarning: Failed to extract smoother parameters for MCMC draw {mcmc_draw_idx}: {e}. Skipping this draw.")
            key_idx += simulation_draws_per_mcmc
            continue

        # Inner loop: Run simulation smoother draws for this MCMC sample
        for smoother_draw_idx in range(simulation_draws_per_mcmc):
            current_smoother_key = all_smoother_keys[key_idx]
            
            try:
                # Build SS matrices for this draw (don't JIT this part due to config_data)
                ss_matrices_sim = construct_ss_matrices_from_smoother_params(
                    smoother_params_single_draw, config_data
                )
                
                # Run simulation path (can JIT the core simulation)
                simulated_states_path = run_single_simulation_path_for_dk(
                    ss_matrices_sim,
                    y_data,
                    x_smooth_original_dense,
                    current_smoother_key,
                    config_data,
                    smoother_params_single_draw
                )

                # Update buffers using simple Python loops (no JAX)
                for t in range(T_obs):
                    for s in range(k_states):
                        new_value = float(simulated_states_path[t, s])  # Convert to Python float
                        
                        current_count = int(count_grid[t, s])
                        buffer_idx = current_count % online_smoother_buffer_size
                        
                        # Update buffer and count
                        buffer_grid = buffer_grid.at[t, s, buffer_idx].set(new_value)
                        count_grid = count_grid.at[t, s].set(current_count + 1)

                processed_draws += 1
                key_idx += 1
                
            except Exception as e:
                print(f"Warning: Simulation draw failed: {e}")
                key_idx += 1
                processed_draws += 1
                
    # Loop through each MCMC sample
    for mcmc_draw_idx in range(num_mcmc_samples):
        try:
             smoother_params_single_draw = extract_smoother_parameters_single_draw(
                posterior_samples, mcmc_draw_idx, config_data 
             )

        except Exception as e:
            print(f"\nWarning: Failed to extract smoother parameters for MCMC draw {mcmc_draw_idx}: {e}. Skipping this draw.")
            key_idx += simulation_draws_per_mcmc
            continue

        # Inner loop: Run simulation smoother draws for this MCMC sample
        for smoother_draw_idx in range(simulation_draws_per_mcmc):
            current_smoother_key = all_smoother_keys[key_idx]
            
            try:
                # Run a single simulation path
                simulated_states_path = jit_run_single_simulation_path(
                    smoother_params_single_draw,
                    y_data,
                    x_smooth_original_dense,
                    current_smoother_key,
                    static_config_data=hashable_config_data
                )

                # Update the online quantile estimators using Python loops
                # This avoids the JAX compilation issues
                for t in range(T_obs):
                    for s in range(k_states):
                        new_value = simulated_states_path[t, s]
                        
                        # Update buffer and count
                        current_count = count_grid[t, s]
                        buffer_idx = current_count % online_smoother_buffer_size
                        buffer_grid = buffer_grid.at[t, s, buffer_idx].set(new_value)
                        count_grid = count_grid.at[t, s].set(current_count + 1)

                processed_draws += 1
                key_idx += 1
                
            except Exception as e:
                 print(f"\nWarning: Failed to run smoother draw {smoother_draw_idx} for MCMC sample {mcmc_draw_idx}: {e}. Skipping.")
                 key_idx += 1
                 processed_draws += 1

        # Report progress periodically
        if (mcmc_draw_idx + 1) % 10 == 0 or (mcmc_draw_idx + 1) == num_mcmc_samples:
             elapsed_time = time.time() - start_smooth_sim_time
             avg_time_per_mcmc = elapsed_time / (mcmc_draw_idx + 1) if (mcmc_draw_idx + 1) > 0 else 0
             remaining_mcmc = num_mcmc_samples - (mcmc_draw_idx + 1)
             estimated_remaining_time = avg_time_per_mcmc * remaining_mcmc if avg_time_per_mcmc > 0 else float('inf')
             print(f"Processed MCMC sample {mcmc_draw_idx + 1}/{num_mcmc_samples}. "
                   f"Total draws processed: {processed_draws}/{total_simulation_draws}. "
                   f"Elapsed: {elapsed_time:.2f}s. Est. remaining: {estimated_remaining_time:.2f}s.")

    smooth_sim_time = time.time() - start_smooth_sim_time
    print(f"\nOnline simulation smoothing completed in {smooth_sim_time:.2f} seconds.")
    print(f"Successfully processed {processed_draws}/{total_simulation_draws} total simulation draws.")

    # Step 8: Compute Final HDI/Quantiles from collected buffers
    print(f"\nStep 8: Computing final quantiles and HDI...")
    
    final_quantiles_grid = jnp.full(
        (T_obs, k_states, len(quantiles_to_track)),
        jnp.nan,
        dtype=_DEFAULT_DTYPE
    )

    # Use Python loops to avoid JAX compilation issues
    print("Computing quantiles using Python loops to avoid JIT compilation issues...")
    
    for t in range(T_obs):
        for s in range(k_states):
            buffer_ts = buffer_grid[t, s]
            count_ts = count_grid[t, s]
            
            # Compute quantiles for this (time, state) pair
            computed_qs = OnlineQuantileEstimator.compute_quantiles_from_buffer(
                buffer_ts, count_ts, quantiles_to_track
            )
            
            final_quantiles_grid = final_quantiles_grid.at[t, s].set(computed_qs)
        
        # Print progress every 10% of time steps
        if (t + 1) % max(1, T_obs // 10) == 0:
            print(f"  Processed {t + 1}/{T_obs} time steps ({100*(t+1)/T_obs:.1f}%)")

    # Extract median, lower and upper HDI bounds
    try:
        median_idx = quantiles_to_track.index(0.5)
        lower_hdi_idx = quantiles_to_track.index(0.025)
        upper_hdi_idx = quantiles_to_track.index(0.975)
        
        median_draws = final_quantiles_grid[:, :, median_idx]
        hdi_lower_draws = final_quantiles_grid[:, :, lower_hdi_idx]
        hdi_upper_draws = final_quantiles_grid[:, :, upper_hdi_idx]
    except ValueError:
         print("Warning: Standard quantiles (0.025, 0.5, 0.975) not found. HDI/Median fields will be NaN.")
         median_draws = jnp.full((T_obs, k_states), jnp.nan, dtype=_DEFAULT_DTYPE)
         hdi_lower_draws = jnp.full((T_obs, k_states), jnp.nan, dtype=_DEFAULT_DTYPE)
         hdi_upper_draws = jnp.full((T_obs, k_states), jnp.nan, dtype=_DEFAULT_DTYPE)

    # Compute mean from accumulated buffer values
    print("Computing means from buffers...")
    mean_draws = jnp.zeros((T_obs, k_states), dtype=_DEFAULT_DTYPE)
    
    for t in range(T_obs):
        for s in range(k_states):
            buffer_ts = buffer_grid[t, s]
            finite_mask = jnp.isfinite(buffer_ts)
            finite_count = jnp.sum(finite_mask)
            
            if finite_count > 0:
                finite_sum = jnp.sum(jnp.where(finite_mask, buffer_ts, 0.0))
                mean_val = finite_sum / finite_count
            else:
                mean_val = jnp.nan
                
            mean_draws = mean_draws.at[t, s].set(mean_val)

    print("Quantiles, HDI, and Mean computed successfully.")

    # Step 9: Package results
    results['smoothing_results']['online_smoother_quantiles'] = {
        'quantiles_to_track': quantiles_to_track,
        'estimated_quantiles': final_quantiles_grid,
        'median': median_draws,
        'hdi_lower': hdi_lower_draws,
        'hdi_upper': hdi_upper_draws,
        'mean': mean_draws,
        'buffer_size': online_smoother_buffer_size,
        'total_sim_draws_processed': processed_draws
    }
    results['smoothing_results']['smooth_sim_time'] = smooth_sim_time
    
    print("\n" + "="*100)
    if results.get('smoothing_results', {}).get('online_smoother_quantiles', {}).get('estimated_quantiles') is not None:
        print("ESTIMATION AND ONLINE SMOOTHING COMPLETED SUCCESSFULLY")
    else:
        print("ESTIMATION COMPLETED WITH SMOOTHING ISSUES")
    print("="*100)
    
    return results
# --- Enhanced Plotting with I(2) Data Visualization (Adapt plotting) ---

def plot_results_with_canova_dls(results: Dict[str, Any],
                                data: pd.DataFrame,
                                save_plots: bool = False,
                                plot_dir: str = "plots") -> None:
    """
    Enhanced plotting that shows:
    1. Original I(2) levels vs trends
    2. Canova DLS diagnostics
    3. Estimated states and fitted values (using online smoother results)
    """
    
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Ensure config and data_info are available
    config = results.get('config')
    data_info = results.get('data_info')

    if not config or not data_info:
        print("Cannot plot results: config or data_info is missing.")
        return

    variable_names = data_info['variable_names']
    dates = data.index
    
    # Plot 1: I(2) Levels and Growth Rates (if available)
    level_vars = [col for col in data.columns if col.endswith('_level')]
    if level_vars:
        print("Plotting I(2) levels and growth rates...")
        fig, axes = plt.subplots(len(variable_names), 2, figsize=(15, 4*len(variable_names)))
        if len(variable_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(variable_names):
            level_col = f'{var_name}_level'
            
            # Plot levels
            if level_col in data.columns:
                axes[i, 0].plot(dates, data[level_col], 'b-', label='I(2) Level', alpha=0.7)
                axes[i, 0].set_title(f'{var_name} - I(2) Level Series')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
            
            # Plot growth rates (first differences)
            axes[i, 1].plot(dates, data[var_name], 'r-', label='Growth Rate (Estimation Data)', alpha=0.8, linewidth=1.5)
            axes[i, 1].set_title(f'{var_name} - Growth Rate (Estimation Data)')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Format dates
            for ax in axes[i, :]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(5))
                if i == len(variable_names) - 1:
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.tick_params(axis='x', labelbottom=False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/i2_data_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot 2: Canova DLS Diagnostics
    if config and 'canova_dls_results' in config:
        print("Plotting Canova DLS diagnostics...")
        dls_results = config['canova_dls_results']
        vars_with_dls = [v for v in variable_names if v in dls_results and 'diagnostics' in dls_results[v]]
        
        if vars_with_dls:
            fig, axes = plt.subplots(len(vars_with_dls), 2, figsize=(15, 4*len(vars_with_dls)))
            if len(vars_with_dls) == 1:
                axes = axes.reshape(1, -1)
            
            for i, var_name in enumerate(vars_with_dls):
                diagnostics = dls_results[var_name].get('diagnostics') # Use .get for safety
                if diagnostics: # Plot only if diagnostics exist
                     dls_dates = data.index[:len(diagnostics.get('trend_component', []))] # Use .get with default empty list
                     
                     # Plot Canova trend
                     ax_trend = axes[i, 0]
                     trend_comp = diagnostics.get('trend_component')
                     optimal_lambda = diagnostics.get('optimal_lambda', np.nan)
                     if trend_comp is not None:
                         ax_trend.plot(dls_dates, trend_comp, 'b-', 
                                      label=f'Canova Trend (λ={optimal_lambda:.2e})', alpha=0.8)
                     ax_trend.set_title(f'{var_name} - Canova DLS Trend')
                     ax_trend.legend()
                     ax_trend.grid(True, alpha=0.3)
                     
                     # Plot Canova cycle
                     ax_cycle = axes[i, 1]
                     cycle_comp = diagnostics.get('cycle_component')
                     if cycle_comp is not None:
                          ax_cycle.plot(dls_dates, cycle_comp, 'r-', 
                                       label='Canova Cycle', alpha=0.8)
                     ax_cycle.set_title(f'{var_name} - Canova DLS Cycle')
                     ax_cycle.legend()
                     ax_cycle.grid(True, alpha=0.3)
                     
                     # Format dates
                     for ax in [ax_trend, ax_cycle]:
                         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                         ax.xaxis.set_major_locator(mdates.YearLocator(5))
                         if i == len(vars_with_dls) - 1:
                             ax.tick_params(axis='x', rotation=45)
                         else:
                             ax.tick_params(axis='x', labelbottom=False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{plot_dir}/canova_dls_diagnostics.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Plot 3: Estimated States and HDI (using online smoother results)
    smoothing_results = results.get('smoothing_results')
    if smoothing_results and 'online_smoother_quantiles' in smoothing_results:
        print("Plotting estimated states (median and HDI)...")
        
        online_results = smoothing_results['online_smoother_quantiles']
        # Ensure these keys exist before accessing
        median_states = online_results.get('median')
        hdi_lower = online_results.get('hdi_lower')
        hdi_upper = online_results.get('hdi_upper')
        mean_states = online_results.get('mean')
        
        # Baseline RTS smooth on posterior mean (for comparison)
        smoothed_states_rts = smoothing_results.get('smoothed_states_original_mean_rts')
        
        # Check if median_states exists and has expected dimensions
        if median_states is not None and median_states.shape == (data.shape[0], config['k_states']):
             
            full_state_names = config['full_state_names_tuple']
            # state_indices = {name: i for i, name in enumerate(full_state_names)} # Not needed here

            # Plot trends and cycles (only current state components)
            # Determine which states are trends and which are current cycles
            trend_names = config['trend_var_names']
            current_cycle_names = config['stationary_var_names']
            
            fig, axes = plt.subplots(len(variable_names), 2, figsize=(15, 4*len(variable_names)))
            if len(variable_names) == 1:
                axes = axes.reshape(1, -1)
            
            # Get indices of trends and current cycles in the full state vector
            trend_indices_in_full_state = [full_state_names.index(name) for name in trend_names]
            current_cycle_indices_in_full_state = [full_state_names.index(name) for name in current_cycle_names]


            for i, var_name in enumerate(variable_names):
                # Plot trend
                ax_trend = axes[i, 0]
                
                trend_idx = trend_indices_in_full_state[i] # Assuming order matches variable_names

                ax_trend.plot(dates, median_states[:, trend_idx], 'g-', 
                               label='Smoother Median', linewidth=1.5, alpha=0.8)
                if mean_states is not None:
                    ax_trend.plot(dates, mean_states[:, trend_idx], 'r:',
                                  label='Smoother Mean', linewidth=1.5, alpha=0.8)
                
                if hdi_lower is not None and hdi_upper is not None:
                     ax_trend.fill_between(dates, hdi_lower[:, trend_idx], hdi_upper[:, trend_idx], 
                                           color='green', alpha=0.2, label='Estimated HDI')
                
                if smoothed_states_rts is not None:
                     ax_trend.plot(dates, smoothed_states_rts[:, trend_idx], 'k--',
                                   label='RTS (Posterior Mean)', linewidth=1.5, alpha=0.6)
                                   
                ax_trend.set_title(f'{var_name} - Estimated Trend')
                ax_trend.legend()
                ax_trend.grid(True, alpha=0.3)
            
                # Plot cycle (current cycle)
                ax_cycle = axes[i, 1]
                
                cycle_idx = current_cycle_indices_in_full_state[i] # Assuming order matches variable_names

                ax_cycle.plot(dates, median_states[:, cycle_idx], 'g-',
                               label='Smoother Median', linewidth=1.5, alpha=0.8)
                if mean_states is not None:
                     ax_cycle.plot(dates, mean_states[:, cycle_idx], 'r:',
                                   label='Smoother Mean', linewidth=1.5, alpha=0.8)
                                   
                if hdi_lower is not None and hdi_upper is not None:
                     ax_cycle.fill_between(dates, hdi_lower[:, cycle_idx], hdi_upper[:, cycle_idx],
                                           color='green', alpha=0.2, label='Estimated HDI')

                if smoothed_states_rts is not None:
                     ax_cycle.plot(dates, smoothed_states_rts[:, cycle_idx], 'k--',
                                   label='RTS (Posterior Mean)', linewidth=1.5, alpha=0.6)

                ax_cycle.set_title(f'{var_name} - Estimated Cycle')
                ax_cycle.legend()
                ax_cycle.grid(True, alpha=0.3)
            
                # Format dates
                for ax in [axes[i, 0], axes[i, 1]]: # Format both axes for this variable
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax.xaxis.set_major_locator(mdates.YearLocator(10))
                    if i == len(variable_names) - 1:
                        ax.tick_params(axis='x', rotation=45)
                    else:
                        ax.tick_params(axis='x', labelbottom=False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{plot_dir}/estimated_states_hdi.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
             print("Smoother median/mean states not available for plotting fitted values.")

        # Plot 4: Data vs Fitted Values (using estimated mean/median states)
        if median_states is not None and median_states.shape[0] == data.shape[0]:
            print("Plotting data vs fitted values (median)...")
            fig, axes = plt.subplots(len(variable_names), 1, figsize=(15, 3*len(variable_names)))
            if len(variable_names) == 1:
                axes = [axes]
            
            # Need a mapping from state name string to full state vector index (including lags)
            full_state_names = config['full_state_names_tuple']
            state_indices_full = {name: i for i, name in enumerate(full_state_names)}

            # Need mean of measurement parameters if they exist (for fitted values)
            measurement_param_names_tuple = config['measurement_param_names_tuple']
            measurement_param_means = {}
            mcmc_results = results.get('mcmc_results')
            if mcmc_results and mcmc_results.get('posterior_samples'):
                 posterior_samples = mcmc_results['posterior_samples']
                 for param_name in measurement_param_names_tuple:
                      if param_name in posterior_samples:
                            # Use mean of sampled measurement params
                            mean_val = jnp.mean(posterior_samples[param_name], axis=0)
                            # Ensure mean_val is a JAX array, handle potential scalar case
                            measurement_param_means[param_name] = jnp.asarray(mean_val, dtype=_DEFAULT_DTYPE)
                      else:
                            # Fallback if parameter not found (should be handled by config parsing warnings)
                            print(f"Warning: Measurement parameter '{param_name}' not found for plotting fitted values. Using 0.0.")
                            measurement_param_means[param_name] = jnp.array(0.0, dtype=_DEFAULT_DTYPE)

            # Get the parsed model equations details from config_data
            parsed_model_eqs_for_model = config['model_equations_parsed'] # List of (obs_idx, List[Tuple[param_name, state_name, sign]])


            for i, var_name in enumerate(variable_names):
                ax = axes[i]
                # Original data
                ax.plot(dates, data[var_name], 'k-', label='Observed Data', alpha=0.8, linewidth=1.5)
                
                # Compute fitted values using median states and mean measurement parameters
                fitted_values_median = jnp.zeros(data.shape[0], dtype=_DEFAULT_DTYPE)
                
                # Iterate through the parsed equation terms for this observable (obs_idx == i)
                terms_for_this_obs = None
                for obs_idx_check, terms_list in parsed_model_eqs_for_model:
                     if obs_idx_check == i:
                          terms_for_this_obs = terms_list
                          break

                if terms_for_this_obs is not None:
                    for param_name, state_name_in_eq, sign in terms_for_this_obs:
                        # state_name_in_eq is the string name (e.g., 'trend_gdp_growth', 'cycle_gdp_growth')
                        full_state_idx = state_indices_full[state_name_in_eq] # Map name to full state index
                        
                        if param_name is None: # Direct state term
                            fitted_values_median += sign * median_states[:, full_state_idx]
                        else: # Parameter * state term
                            param_value = measurement_param_means.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE)) # Get mean parameter value
                            fitted_values_median += sign * param_value * median_states[:, full_state_idx]

                ax.plot(dates, fitted_values_median, 'r--', label='Fitted (Median States, Mean Params)', alpha=0.8, linewidth=1.5)

                # Optional: Plot fitted values using mean states and mean measurement parameters
                if mean_states is not None and mean_states.shape == (data.shape[0], config['k_states']):
                    fitted_values_mean_states = jnp.zeros(data.shape[0], dtype=_DEFAULT_DTYPE)
                    if terms_for_this_obs is not None:
                         for param_name, state_name_in_eq, sign in terms_for_this_obs:
                              full_state_idx = state_indices_full[state_name_in_eq]
                              if param_name is None:
                                   fitted_values_mean_states += sign * mean_states[:, full_state_idx]
                              else:
                                   param_value = measurement_param_means.get(param_name, jnp.array(0.0, dtype=_DEFAULT_DTYPE))
                                   fitted_values_mean_states += sign * param_value * mean_states[:, full_state_idx]
                                   
                    ax.plot(dates, fitted_values_mean_states, 'b:', label='Fitted (Mean States, Mean Params)', alpha=0.6, linewidth=1.5)


                ax.set_title(f'{var_name} - Observed vs Fitted')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(10))
                
                if i == len(variable_names) - 1:
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.tick_params(axis='x', labelbottom=False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{plot_dir}/data_vs_fitted_online_smoother.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
             print("Smoother median/mean states not available for plotting fitted values.")


    # Print summary (Keep this section as is, but ensure keys exist)
    if results and results.get('mcmc_results'):
        print("\n" + "="*80)
        print("ESTIMATION SUMMARY")
        print("="*80)
        
        config_data = results.get('config')
        if config_data and 'canova_dls_results' in config_data:
            print("\nCanova DLS Prior Information:")
            for var_name, dls_info in config_data['canova_dls_results'].items():
                diag = dls_info.get('diagnostics')
                if diag:
                     print(f"  {var_name}:")
                     print(f"    Optimal λ: {diag.get('optimal_lambda', np.nan):.2e}")
                     print(f"    Trend variance: {diag.get('extracted_trend_var', np.nan):.6f}")
                     print(f"    Cycle variance: {diag.get('extracted_cycle_var', np.nan):.6f}")
        
        smoothing_results = results.get('smoothing_results', {})
        online_smooth_info = smoothing_results.get('online_smoother_quantiles')
        
        print(f"\nOnline Simulation Smoothing Info:")
        if online_smooth_info:
            print(f"  Quantiles tracked: {online_smooth_info.get('quantiles_to_track', 'N/A')}")
            print(f"  Buffer size per (t, s): {online_smooth_info.get('buffer_size', 'N/A')}")
            print(f"  Total simulation draws processed: {online_smooth_info.get('total_sim_draws_processed', 0)}")
            # Print a sample of median/HDI for the first few time steps/states
            print("\nSample Smoother Estimates (Time 0, State 0 & 1):")
            median_draws = online_smooth_info.get('median')
            hdi_lower_draws = online_smooth_info.get('hdi_lower')
            hdi_upper_draws = online_smooth_info.get('hdi_upper')
            mean_draws = online_smooth_info.get('mean') # Also get mean
            full_state_names = config_data.get('full_state_names_tuple') # Get from config_data
            
            if median_draws is not None and full_state_names is not None:
                if median_draws.shape[0] > 0 and median_draws.shape[1] > 0:
                    state_name_0 = full_state_names[0] if len(full_state_names) > 0 else "State 0"
                    median_0 = median_draws[0, 0] if median_draws.shape[1] > 0 else np.nan
                    lower_0 = hdi_lower_draws[0, 0] if hdi_lower_draws is not None and hdi_lower_draws.shape[1] > 0 else np.nan
                    upper_0 = hdi_upper_draws[0, 0] if hdi_upper_draws is not None and hdi_upper_draws.shape[1] > 0 else np.nan
                    mean_0 = mean_draws[0, 0] if mean_draws is not None and mean_draws.shape[1] > 0 else np.nan
                    print(f"  {state_name_0}: Median={float(median_0):.4f}, Mean={float(mean_0):.4f}, 95% HDI=[{float(lower_0):.4f}, {float(upper_0):.4f}]")

                if median_draws.shape[0] > 0 and median_draws.shape[1] > 1:
                    state_name_1 = full_state_names[1] if len(full_state_names) > 1 else "State 1"
                    median_1 = median_draws[0, 1] if median_draws.shape[1] > 1 else np.nan
                    lower_1 = hdi_lower_draws[0, 1] if hdi_lower_draws is not None and hdi_lower_draws.shape[1] > 1 else np.nan
                    upper_1 = hdi_upper_draws[0, 1] if hdi_upper_draws is not None and hdi_upper_draws.shape[1] > 1 else np.nan
                    mean_1 = mean_draws[0, 1] if mean_draws is not None and mean_draws.shape[1] > 1 else np.nan
                    print(f"  {state_name_1}: Median={float(median_1):.4f}, Mean={float(mean_1):.4f}, 95% HDI=[{float(lower_1):.4f}, {float(upper_1):.4f}]")

        else:
            print("  Online simulation smoothing results not available.")

        if results.get('error'):
             print(f"\nErrors encountered during estimation: {results['error']}")

        return results
    else:
        print("Estimation failed!")
        return None

# *** RESTORED function definition ***
def example_with_i2_data():
    """Example using I(2) simulated data with all fixes."""
    
    # Create sample I(2) data
    dates = pd.date_range('1960-01-01', periods=200, freq='QE')
    
    print("Creating I(2) sample data...")
    data = simulate_i2_economic_data(dates, ['gdp_growth', 'inflation'], seed=42)
    
    print("Sample I(2) data created:")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {list(data.columns)}")
    print("\nFirst few observations:")
    print(data[['gdp_growth', 'inflation']].head(10))
    
    # Run estimation with all fixes
    results = run_bvar_estimation_with_fixes(
        data=data,
        variable_names=['gdp_growth', 'inflation'],
        training_fraction=0.25,
        var_order=1,
        mcmc_params={
            'num_warmup': 200,
            'num_samples': 300, # Reduced for quicker example
            'num_chains': 2
        },
        dls_params={
            'optimize_smoothness': True,
            'smoothness_range': (1e-6, 1e2),
            'alpha_shape': 2.5,
        },
        simulation_draws_per_mcmc=50, # Number of smoother draws per MCMC sample
        online_smoother_buffer_size=500, # Buffer size (should be > simulation_draws_per_mcmc)
        save_config=True,
        config_filename='bvar_canova_dls_fixed_online.yml'
    )
    
    # Create comprehensive plots
    plot_results_with_canova_dls(
        results=results,
        data=data,
        save_plots=True,
        plot_dir='estimation_plots_online_smoother'
    )
    
    # Print summary
    if results and results.get('mcmc_results'):
        # Posterior samples are still in the results dictionary for inspection if needed
        # posterior = results['mcmc_results']['posterior_samples']
        print("\n" + "="*80)
        print("ESTIMATION SUMMARY")
        print("="*80)
        
        config_data = results.get('config')
        if config_data and 'canova_dls_results' in config_data:
            print("\nCanova DLS Prior Information:")
            for var_name, dls_info in config_data['canova_dls_results'].items():
                diag = dls_info.get('diagnostics')
                if diag:
                     print(f"  {var_name}:")
                     print(f"    Optimal λ: {diag.get('optimal_lambda', np.nan):.2e}")
                     print(f"    Trend variance: {diag.get('extracted_trend_var', np.nan):.6f}")
                     print(f"    Cycle variance: {diag.get('extracted_cycle_var', np.nan):.6f}")
        
        smoothing_results = results.get('smoothing_results', {})
        online_smooth_info = smoothing_results.get('online_smoother_quantiles')
        
        print(f"\nOnline Simulation Smoothing Info:")
        if online_smooth_info:
            print(f"  Quantiles tracked: {online_smooth_info.get('quantiles_to_track', 'N/A')}")
            print(f"  Buffer size per (t, s): {online_smooth_info.get('buffer_size', 'N/A')}")
            print(f"  Total simulation draws processed: {online_smooth_info.get('total_sim_draws_processed', 0)}")
            # Print a sample of median/HDI for the first few time steps/states
            print("\nSample Smoother Estimates (Time 0, State 0 & 1):")
            median_draws = online_smooth_info.get('median')
            hdi_lower_draws = online_smooth_info.get('hdi_lower')
            hdi_upper_draws = online_smooth_info.get('hdi_upper')
            mean_draws = online_smooth_info.get('mean') # Also get mean
            full_state_names = config_data.get('full_state_names_tuple') # Get from config_data
            
            if median_draws is not None and full_state_names is not None:
                if median_draws.shape[0] > 0 and median_draws.shape[1] > 0:
                    state_name_0 = full_state_names[0] if len(full_state_names) > 0 else "State 0"
                    median_0 = median_draws[0, 0] if median_draws.shape[1] > 0 else np.nan
                    lower_0 = hdi_lower_draws[0, 0] if hdi_lower_draws is not None and hdi_lower_draws.shape[1] > 0 else np.nan
                    upper_0 = hdi_upper_draws[0, 0] if hdi_upper_draws is not None and hdi_upper_draws.shape[1] > 0 else np.nan
                    mean_0 = mean_draws[0, 0] if mean_draws is not None and mean_draws.shape[1] > 0 else np.nan
                    print(f"  {state_name_0}: Median={float(median_0):.4f}, Mean={float(mean_0):.4f}, 95% HDI=[{float(lower_0):.4f}, {float(upper_0):.4f}]")

                if median_draws.shape[0] > 0 and median_draws.shape[1] > 1:
                    state_name_1 = full_state_names[1] if len(full_state_names) > 1 else "State 1"
                    median_1 = median_draws[0, 1] if median_draws.shape[1] > 1 else np.nan
                    lower_1 = hdi_lower_draws[0, 1] if hdi_lower_draws is not None and hdi_lower_draws.shape[1] > 1 else np.nan
                    upper_1 = hdi_upper_draws[0, 1] if hdi_upper_draws is not None and hdi_upper_draws.shape[1] > 1 else np.nan
                    mean_1 = mean_draws[0, 1] if mean_draws is not None and mean_draws.shape[1] > 1 else np.nan
                    print(f"  {state_name_1}: Median={float(median_1):.4f}, Mean={float(mean_1):.4f}, 95% HDI=[{float(lower_1):.4f}, {float(upper_1):.4f}]")

        else:
            print("  Online simulation smoothing results not available.")

        if results.get('error'):
             print(f"\nErrors encountered during estimation: {results['error']}")

        return results
    else:
        print("Estimation failed!")
        return None


if __name__ == "__main__":
    # *** ENSURE THE FUNCTION DEFINITION IS ABOVE THIS BLOCK ***
    results = example_with_i2_data()