# estimate_bvar_with_dls_priors_fixed.py
# Comprehensive fix for I(2) simulation, parameter mapping, and Canova DLS

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
from scipy import linalg
from scipy.optimize import minimize_scalar

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
_DEFAULT_DTYPE = jnp.float64

try:
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    numpyro.set_host_device_count(min(num_cpu, 4))
except Exception as e:
    print(f"Could not set host device count: {e}")
    pass

# Import your existing modules
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax
from core.simulate_bvar_jax import simulate_bvar_with_trends_jax
from core.var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices
from core.run_single_draw import run_simulation_smoother_single_params_jit
from core.parameter_extraction_for_smoother import extract_smoother_parameters, validate_smoother_parameters
# --- Enhanced DLS Implementation following Canova (2014) ---

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

# --- I(2) Data Simulation for BVAR Estimation ---

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

# --- Fixed Parameter Mapping for Smoother ---

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

# --- Enhanced DLS Prior Elicitation ---

def create_dls_config_with_canova(data: pd.DataFrame,
                                 variable_names: List[str],
                                 training_fraction: float = 0.3,
                                 dls_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Apply Canova (2014) DLS to training data for prior elicitation.
    """
    
    # Default DLS parameters
    default_dls_params = {
        'optimize_smoothness': True,
        'smoothness_range': (1e-6, 1e2),
        'alpha_shape': 2.5,
    }
    actual_dls_params = {**default_dls_params, **(dls_params or {})}
    
    # Use subset of data for prior elicitation
    n_total = len(data)
    n_train = max(int(training_fraction * n_total), 20)  # Minimum 20 observations
    train_data = data.iloc[:n_train]
    
    print(f"Using {len(train_data)} observations for Canova DLS prior elicitation")
    
    # Initialize Canova DLS
    dls_extractor = CanovaDLS(
        smoothness_range=actual_dls_params['smoothness_range'],
        optimize_smoothness=actual_dls_params['optimize_smoothness']
    )
    
    dls_results = {}
    
    for var_name in variable_names:
        y = train_data[var_name].values.astype(_DEFAULT_DTYPE)
        
        # Need at least some finite observations
        if np.sum(np.isfinite(y)) < 10:
            print(f"Warning: Too few finite observations for {var_name}. Using defaults.")
            continue
        
        try:
            # Apply Canova DLS
            trend, cycle, optimal_lambda, trend_var, cycle_var = dls_extractor.extract_trend_cycle(y)
            
            # Compute initial conditions from early observations
            initial_window = min(10, np.sum(np.isfinite(y)))
            early_trend = trend[:initial_window]
            early_cycle = cycle[:initial_window]
            
            early_trend_valid = early_trend[np.isfinite(early_trend)]
            early_cycle_valid = early_cycle[np.isfinite(early_cycle)]
            
            initial_trend_mean = np.mean(early_trend_valid) if len(early_trend_valid) > 0 else 0.0
            initial_cycle_mean = np.mean(early_cycle_valid) if len(early_cycle_valid) > 0 else 0.0
            
            initial_trend_var = np.var(early_trend_valid) if len(early_trend_valid) > 1 else trend_var * 5.0
            initial_cycle_var = np.var(early_cycle_valid) if len(early_cycle_valid) > 1 else cycle_var
            
            # Ensure positive variances
            initial_trend_var = max(float(initial_trend_var), 1e-9)
            initial_cycle_var = max(float(initial_cycle_var), 1e-9)
            
            # Create prior suggestions
            trend_prior = suggest_ig_priors(trend_var, alpha=actual_dls_params['alpha_shape'])
            cycle_prior = suggest_ig_priors(cycle_var, alpha=actual_dls_params['alpha_shape'])
            
            dls_results[var_name] = {
                'initial_conditions': {
                    'trend_mean': float(initial_trend_mean),
                    'trend_variance': float(initial_trend_var),
                    'cycle_mean': float(initial_cycle_mean), 
                    'cycle_variance': float(initial_cycle_var),
                },
                'trend_shocks': trend_prior,
                'cycle_shocks': cycle_prior,
                'diagnostics': {
                    'trend_component': trend,
                    'cycle_component': cycle,
                    'optimal_lambda': optimal_lambda,
                    'extracted_trend_var': trend_var,
                    'extracted_cycle_var': cycle_var
                }
            }
            
            print(f"Canova DLS for {var_name}:")
            print(f"  Optimal λ: {optimal_lambda:.6f}")
            print(f"  Trend shock var: {trend_var:.6f} -> IG(α={trend_prior['alpha']:.2f}, β={trend_prior['beta']:.6f})")
            print(f"  Cycle shock var: {cycle_var:.6f} -> IG(α={cycle_prior['alpha']:.2f}, β={cycle_prior['beta']:.6f})")
            
        except Exception as e:
            print(f"Error during Canova DLS for {var_name}: {e}")
            continue
    
    return dls_results

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

# --- Rest of the estimation pipeline (convert_to_hashable, etc.) ---

def convert_to_hashable(obj):
    """Recursively convert lists to tuples to make objects hashable for JAX JIT."""
    if isinstance(obj, list):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted(((str(k), convert_to_hashable(v)) if not isinstance(k, str) else (k, convert_to_hashable(v))) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:  # Scalar numpy array
            return float(obj.item())
        else:
            return tuple(obj.tolist())
    elif isinstance(obj, jnp.ndarray):
        if obj.ndim == 0:  # Scalar JAX array  
            return float(obj.item())
        else:
            return tuple(obj.tolist())
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):  # JAX/numpy scalar
        return float(obj.item())
    else:
        return obj

def create_config_with_canova_dls(data: pd.DataFrame,
                                 variable_names: Optional[List[str]] = None,
                                 training_fraction: float = 0.3,
                                 var_order: int = 1,
                                 dls_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create BVAR configuration using Canova DLS priors.
    """
    if variable_names is None:
        variable_names = list(data.columns)
    
    print("="*80)
    print("APPLYING CANOVA (2014) DLS PRIOR ELICITATION")
    print("="*80)
    
    # Apply Canova DLS
    prior_results = create_dls_config_with_canova(
        data=data[variable_names],
        variable_names=variable_names,
        training_fraction=training_fraction,
        dls_params=dls_params
    )
    
    # Create configuration structure (same as before, just using Canova results)
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
    k_states = k_trends + k_stationary * var_order
    
    config_data.update({
        'k_endog': k_endog,
        'k_trends': k_trends,
        'k_stationary': k_stationary,
        'k_states': k_states,
    })
    
    # Fill in model equations
    for var_name in variable_names:
        config_data['model_equations'][var_name] = f'trend_{var_name} + cycle_{var_name}'
    
    # Fill in Canova DLS priors
    for var_name in variable_names:
        if var_name in prior_results:
            result = prior_results[var_name]
            init_conds = result['initial_conditions']
            
            # Initial conditions
            config_data['initial_conditions']['states'][f'trend_{var_name}'] = {
                'mean': float(init_conds['trend_mean']),
                'var': float(init_conds['trend_variance'])
            }
            config_data['initial_conditions']['states'][f'cycle_{var_name}'] = {
                'mean': float(init_conds['cycle_mean']),
                'var': float(init_conds['cycle_variance'])
            }
            
            # Cycle shock priors
            cycle_prior = result['cycle_shocks']
            config_data['stationary_prior']['stationary_shocks'][f'cycle_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': float(cycle_prior['alpha']),
                    'beta': float(cycle_prior['beta'])
                }
            }
            
            # Trend shock priors
            trend_prior = result['trend_shocks']
            config_data['trend_shocks']['trend_shocks'][f'trend_{var_name}'] = {
                'distribution': 'inverse_gamma',
                'parameters': {
                    'alpha': float(trend_prior['alpha']),
                    'beta': float(trend_prior['beta'])
                }
            }
        else:
            print(f"Warning: No Canova DLS results for {var_name}. Using defaults.")
            # Set defaults
            config_data['initial_conditions']['states'][f'trend_{var_name}'] = {
                'mean': float(data[var_name].iloc[0]) if len(data) > 0 and np.isfinite(data[var_name].iloc[0]) else 0.0, 
                'var': 1.0
            }
            config_data['initial_conditions']['states'][f'cycle_{var_name}'] = {
                'mean': 0.0, 
                'var': 1.0
            }
    
    # Parse configuration (rest of the parsing logic remains the same)
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
            parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
            parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))
    
    config_data['model_equations_parsed'] = parsed_model_eqs_list
    
    # Create detailed parsing for JAX
    full_state_names_list_for_parsing = list(trend_names)
    for i in range(var_order):
        for stat_var in stationary_names:
            if i == 0:
                full_state_names_list_for_parsing.append(stat_var)
            else:
                full_state_names_list_for_parsing.append(f"{stat_var}_t_minus_{i+1}")
    
    c_matrix_state_names = full_state_names_list_for_parsing
    state_to_c_idx_map = {name: i for i, name in enumerate(c_matrix_state_names)}
    param_to_idx_map = {name: i for i, name in enumerate(measurement_param_names)}
    
    parsed_model_eqs_jax_detailed = []
    for obs_idx, parsed_terms in parsed_model_eqs_list:
        processed_terms_for_obs = []
        for param_name, state_name_in_eq, sign in parsed_terms:
            term_type = 0 if param_name is None else 1
            if state_name_in_eq in state_to_c_idx_map:
                state_index_in_C = state_to_c_idx_map[state_name_in_eq]
                param_index_if_any = param_to_idx_map.get(param_name, -1)
                processed_terms_for_obs.append(
                    (term_type, state_index_in_C, param_index_if_any, float(sign))
                )
        parsed_model_eqs_jax_detailed.append((obs_idx, tuple(processed_terms_for_obs)))
    
    config_data['parsed_model_eqs_jax_detailed'] = tuple(parsed_model_eqs_jax_detailed)
    
    # Identify trend names with shocks
    trend_shocks_spec = config_data.get('trend_shocks', {}).get('trend_shocks', {})
    config_data['trend_names_with_shocks'] = [
        name for name in config_data['variables']['trend_names']
        if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and trend_shocks_spec[name].get('distribution') == 'inverse_gamma'
    ]
    config_data['n_trend_shocks'] = len(config_data['trend_names_with_shocks'])
    
    # Pre-calculate static indices
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows.astype(int), off_diag_cols.astype(int))
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)
    
    # Add compatibility keys
    config_data.update({
        'observable_names': tuple(variable_names),
        'trend_var_names': tuple(config_data['variables']['trend_names']),
        'stationary_var_names': tuple(config_data['variables']['stationary_var_names']),
        'raw_config_initial_conds': config_data['initial_conditions'],
        'raw_config_stationary_prior': config_data['stationary_prior'],
        'raw_config_trend_shocks': config_data['trend_shocks'],
        'raw_config_measurement_params': measurement_params_config,
        'raw_config_model_eqs_str_dict': config_data['model_equations'],
        'measurement_param_names_tuple': tuple(measurement_param_names),
    })
    
    # Create flat initial condition arrays
    config_data['full_state_names_tuple'] = tuple(full_state_names_list_for_parsing)
    
    init_x_means_flat_list = []
    init_P_diag_flat_list = []
    initial_conditions_parsed = config_data['initial_conditions_parsed']
    
    for state_name in full_state_names_list_for_parsing:
        base_name_for_lag = state_name
        is_lagged = False
        if "_t_minus_" in state_name:
            parts = state_name.split("_t_minus_")
            if len(parts) == 2:
                base_name_for_lag = parts[0]
                is_lagged = True
        
        if state_name in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[state_name]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[state_name]['var']))
        elif is_lagged and base_name_for_lag in initial_conditions_parsed:
            init_x_means_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['mean']))
            init_P_diag_flat_list.append(float(initial_conditions_parsed[base_name_for_lag]['var']))
        else:
            print(f"Warning: Initial condition not found for state '{state_name}'. Using defaults.")
            init_x_means_flat_list.append(0.0)
            init_P_diag_flat_list.append(1.0)
    
    config_data['init_x_means_flat'] = jnp.array(init_x_means_flat_list, dtype=_DEFAULT_DTYPE)
    config_data['init_P_diag_flat'] = jnp.array(init_P_diag_flat_list, dtype=_DEFAULT_DTYPE)
    
    # Store Canova DLS results
    config_data['canova_dls_results'] = prior_results
    
    print("="*80)
    print("CANOVA DLS PRIOR ELICITATION COMPLETED")
    print("="*80)
    
    return config_data

# --- Main Estimation Function with All Fixes ---

def run_bvar_estimation_with_fixes(data: pd.DataFrame,
                                  variable_names: Optional[List[str]] = None,
                                  training_fraction: float = 0.3,
                                  var_order: int = 1,
                                  mcmc_params: Optional[Dict] = None,
                                  dls_params: Optional[Dict] = None,
                                  simulation_draws: int = 100,
                                  save_config: bool = True,
                                  config_filename: str = "bvar_canova_dls.yml") -> Dict[str, Any]:
    """
    Complete BVAR estimation with all fixes:
    1. I(2) data simulation with proper differencing
    2. Canova (2014) DLS prior elicitation  
    3. Fixed parameter mapping for smoother
    """
    
    if variable_names is None:
        variable_names = list(data.columns)
    
    if mcmc_params is None:
        mcmc_params = {
            'num_warmup': 300,
            'num_samples': 500,
            'num_chains': 2
        }
    
    print("="*100)
    print("BVAR ESTIMATION WITH CANOVA DLS AND I(2) DATA HANDLING")
    print("="*100)
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Variables: {variable_names}")
    print(f"VAR order: {var_order}")
    print(f"Training fraction: {training_fraction:.1%}")
    
    # Step 1: Create configuration with Canova DLS priors
    print("\nStep 1: Generating Canova DLS priors...")
    config_data = create_config_with_canova_dls(
        data=data,
        variable_names=variable_names,
        training_fraction=training_fraction,
        var_order=var_order,
        dls_params=dls_params
    )
    
    # Step 2: Save configuration if requested
    if save_config:
        yaml_config = {
            'var_order': config_data['var_order'],
            'variables': {
                'observables': list(config_data['variables']['observable_names']),
                'trends': list(config_data['variables']['trend_names']),
                'stationary': list(config_data['variables']['stationary_var_names'])
            },
            'model_equations': config_data['model_equations'],
            'initial_conditions': config_data['raw_config_initial_conds'],
            'stationary_prior': config_data['raw_config_stationary_prior'],
            'trend_shocks': config_data['raw_config_trend_shocks'],
            'parameters': config_data['raw_config_measurement_params']
        }
        
        try:
            with open(config_filename, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {config_filename}")
        except Exception as e:
            print(f"Warning: Could not save configuration: {e}")
    
    # Step 3: Prepare data for estimation
    y_data = data[variable_names].values.astype(_DEFAULT_DTYPE)
    print(f"\nStep 3: Preparing data...")
    print(f"Data shape: {y_data.shape}")
    
    # Handle observations
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y_data), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]
    
    if static_n_obs_actual == 0:
        raise ValueError("No valid observation columns found.")
    
    # Step 4: Run MCMC estimation
    print(f"\nStep 4: Running MCMC estimation...")
    model_args = {
        'y': y_data,
        'config_data': config_data,
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'trend_var_names': list(config_data['variables']['trend_names']),
        'stationary_var_names': list(config_data['variables']['stationary_var_names']),
        'observable_names': list(config_data['variables']['observable_names']),
    }
    
    kernel = NUTS(model=numpyro_bvar_stationary_model, init_strategy=numpyro.infer.init_to_sample())
    mcmc = MCMC(kernel, **mcmc_params)
    
    key = random.PRNGKey(42)
    key_mcmc, key_smooth = random.split(key)
    
    start_time = time.time()
    try:
        mcmc.run(key_mcmc, **model_args)
        mcmc_time = time.time() - start_time
        print(f"\nMCMC completed in {mcmc_time:.2f} seconds")
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        mcmc_extras = mcmc.get_extra_fields()
    except Exception as e:
        print(f"\nERROR: MCMC estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_data,
            'mcmc_results': None,
            'smoothing_results': None,
            'error': f"MCMC failed: {e}"
        }
    
    smoother_params = extract_smoother_parameters(posterior_samples, config_data, variable_names)
    validate_smoother_parameters(smoother_params)
    
    # Step 5: Extract parameters with fixed mapping for smoother
    print(f"\nStep 5: Extracting parameters for smoother...")
    
    # CRITICAL FIX: Use the corrected parameter extraction function
    posterior_mean_params = extract_posterior_params_for_smoother(
        posterior_samples, config_data, variable_names
    )
    
    print(f"Extracted parameters: {list(posterior_mean_params.keys())}")
    
    # Verify we have the required parameters
    required_base_params = ['A_diag']
    for var_name in variable_names:
        required_base_params.extend([
            f'stationary_var_{var_name}',
            f'trend_var_{var_name}'
        ])
    
    missing_critical = [p for p in required_base_params if p not in posterior_mean_params]
    if missing_critical:
        print(f"ERROR: Missing critical parameters for smoother: {missing_critical}")
        return {
            'config': config_data,
            'mcmc_results': {'posterior_samples': posterior_samples, 'mcmc_time': mcmc_time},
            'smoothing_results': None,
            'error': f"Missing parameters: {missing_critical}"
        }
    
    # Step 6: Run simulation smoother with fixed parameters
    print(f"\nStep 6: Running simulation smoother...")
    
    # Convert to hashable formats with error handling
    try:
        model_eqs_hashable = convert_to_hashable(config_data['parsed_model_eqs_jax_detailed'])
        measurement_params_hashable = convert_to_hashable([])
        
        # CRITICAL FIX: Convert initial conditions to the tuple format expected by smoother
        # The smoother expects: (state_name, mean_val, var_val)
        # But we have: {state_name: {'mean': val, 'var': val}}
        initial_conds_tuple = tuple(
            (state_name, float(state_config['mean']), float(state_config['var']))
            for state_name, state_config in config_data['initial_conditions_parsed'].items()
        )
        initial_conds_hashable = initial_conds_tuple
        print(f"Converted initial conditions to tuple format: {len(initial_conds_tuple)} states")
        
    except Exception as e:
        print(f"Error converting to hashable format: {e}")
        # Fallback: create simplified tuple version
        initial_conds_tuple = tuple(
            (state_name, 0.0, 1.0)  # Default values
            for state_name in config_data['full_state_names_tuple']
        )
        initial_conds_hashable = initial_conds_tuple
        model_eqs_hashable = convert_to_hashable(config_data['model_equations_parsed'])  # Use simpler version
        measurement_params_hashable = tuple()
    
    static_smoother_args = {
        'static_k_endog': config_data['k_endog'],
        'static_k_trends': config_data['k_trends'],
        'static_k_stationary': config_data['k_stationary'],
        'static_p': config_data['var_order'],
        'static_k_states': config_data['k_states'],
        'static_n_trend_shocks': config_data['n_trend_shocks'],
        'static_n_shocks_state': config_data['k_stationary'] + config_data['n_trend_shocks'],
        'static_num_off_diag': config_data.get('num_off_diag', 0),
        'static_off_diag_rows': config_data['static_off_diag_indices'][0],
        'static_off_diag_cols': config_data['static_off_diag_indices'][1],
        'static_valid_obs_idx': static_valid_obs_idx,
        'static_n_obs_actual': static_n_obs_actual,
        'model_eqs_parsed': model_eqs_hashable,
        'initial_conds_parsed': initial_conds_hashable,
        'trend_names_with_shocks': tuple(config_data.get('trend_names_with_shocks', [])),
        'stationary_var_names': tuple(config_data['variables']['stationary_var_names']),
        'trend_var_names': tuple(config_data['variables']['trend_names']),
        'measurement_params_config': measurement_params_hashable,
        'num_draws': simulation_draws,
    }
    
    smoothed_states_original = None
    simulation_results = None
    smooth_time = 0.0
    
    try:
        start_smooth_time = time.time()
        smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
            posterior_mean_params,
            y_data,
            key_smooth,
            **static_smoother_args
        )
        smooth_time = time.time() - start_smooth_time
        print(f"Simulation smoother completed in {smooth_time:.2f} seconds")
        
    except Exception as e:
        print(f"\nERROR: Simulation smoother failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue with partial results
    
    # Step 7: Package results
    results = {
        'config': config_data,
        'mcmc_results': {
            'posterior_samples': posterior_samples,
            'mcmc_time': mcmc_time,
            'mcmc_summary': mcmc_extras
        },
        'smoothing_results': {
            'smoothed_states': smoothed_states_original,
            'simulation_results': simulation_results,
            'smooth_time': smooth_time
        },
        'data_info': {
            'data_shape': y_data.shape,
            'variable_names': variable_names,
            'date_range': (data.index[0], data.index[-1])
        },
        'parameter_mapping': posterior_mean_params  # Include the fixed mapping
    }
    
    print("\n" + "="*100)
    if smoothed_states_original is not None:
        print("ESTIMATION COMPLETED SUCCESSFULLY")
    else:
        print("ESTIMATION COMPLETED WITH SMOOTHER ISSUES")
    print("="*100)
    
    return results

# --- Enhanced Plotting with I(2) Data Visualization ---

def plot_results_with_canova_dls(results: Dict[str, Any],
                                data: pd.DataFrame,
                                save_plots: bool = False,
                                plot_dir: str = "plots") -> None:
    """
    Enhanced plotting that shows:
    1. Original I(2) levels vs trends
    2. Canova DLS diagnostics
    3. Estimated states and fitted values
    """
    
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    config = results.get('config')
    variable_names = results['data_info']['variable_names']
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
            axes[i, 1].plot(dates, data[var_name], 'r-', label='Growth Rate (First Diff)', alpha=0.7)
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
                diagnostics = dls_results[var_name]['diagnostics']
                dls_dates = data.index[:len(diagnostics['trend_component'])]
                
                # Plot Canova trend
                ax_trend = axes[i, 0]
                ax_trend.plot(dls_dates, diagnostics['trend_component'], 'b-', 
                             label=f'Canova Trend (λ={diagnostics["optimal_lambda"]:.2e})', alpha=0.8)
                ax_trend.set_title(f'{var_name} - Canova DLS Trend')
                ax_trend.legend()
                ax_trend.grid(True, alpha=0.3)
                
                # Plot Canova cycle
                ax_cycle = axes[i, 1]
                ax_cycle.plot(dls_dates, diagnostics['cycle_component'], 'r-', 
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
    
    # Plot 3: Estimated States (if smoother succeeded)
    smoothing_results = results.get('smoothing_results')
    if smoothing_results and smoothing_results.get('smoothed_states') is not None:
        print("Plotting estimated states...")
        smoothed_states = smoothing_results['smoothed_states']
        simulation_results = smoothing_results.get('simulation_results')
        
        full_state_names = config['full_state_names_tuple']
        state_indices = {name: i for i, name in enumerate(full_state_names)}
        
        # Plot trends and cycles
        fig, axes = plt.subplots(len(variable_names), 2, figsize=(15, 4*len(variable_names)))
        if len(variable_names) == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(variable_names):
            # Plot trend
            trend_name = f'trend_{var_name}'
            if trend_name in state_indices:
                trend_idx = state_indices[trend_name]
                axes[i, 0].plot(dates, smoothed_states[:, trend_idx], 'b-', 
                               label='Estimated Trend', linewidth=1.5, alpha=0.8)
                
                # Add simulation bands if available
                if simulation_results and len(simulation_results) == 3:
                    mean_sim, median_sim, all_draws = simulation_results
                    if trend_idx < mean_sim.shape[1]:
                        axes[i, 0].plot(dates, mean_sim[:, trend_idx], 'r:', 
                                       label='Simulation Mean', linewidth=1.5)
                        
                        if all_draws is not None and all_draws.shape[0] > 1:
                            try:
                                lower = jnp.percentile(all_draws[:, :, trend_idx], 10, axis=0)
                                upper = jnp.percentile(all_draws[:, :, trend_idx], 90, axis=0)
                                axes[i, 0].fill_between(dates, lower, upper, 
                                                       color='red', alpha=0.2, label='80% Band')
                            except:
                                pass
                
                axes[i, 0].set_title(f'{var_name} - Estimated Trend')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
            
            # Plot cycle
            cycle_name = f'cycle_{var_name}'
            if cycle_name in state_indices:
                cycle_idx = state_indices[cycle_name]
                axes[i, 1].plot(dates, smoothed_states[:, cycle_idx], 'g-',
                               label='Estimated Cycle', linewidth=1.5, alpha=0.8)
                
                # Add simulation bands if available
                if simulation_results and len(simulation_results) == 3:
                    mean_sim, median_sim, all_draws = simulation_results
                    if cycle_idx < mean_sim.shape[1]:
                        axes[i, 1].plot(dates, mean_sim[:, cycle_idx], 'r:',
                                       label='Simulation Mean', linewidth=1.5)
                        
                        if all_draws is not None and all_draws.shape[0] > 1:
                            try:
                                lower = jnp.percentile(all_draws[:, :, cycle_idx], 10, axis=0)
                                upper = jnp.percentile(all_draws[:, :, cycle_idx], 90, axis=0)
                                axes[i, 1].fill_between(dates, lower, upper,
                                                       color='red', alpha=0.2, label='80% Band')
                            except:
                                pass
                
                axes[i, 1].set_title(f'{var_name} - Estimated Cycle')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
            
            # Format dates
            for ax in axes[i, :]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator(10))
                if i == len(variable_names) - 1:
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.tick_params(axis='x', labelbottom=False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/estimated_states.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: Data vs Fitted Values
        print("Plotting data vs fitted values...")
        fig, axes = plt.subplots(len(variable_names), 1, figsize=(15, 3*len(variable_names)))
        if len(variable_names) == 1:
            axes = [axes]
        
        for i, var_name in enumerate(variable_names):
            # Original data
            axes[i].plot(dates, data[var_name], 'k-', label='Observed Data', alpha=0.8, linewidth=1.5)
            
            # Compute fitted values (trend + cycle)
            trend_name = f'trend_{var_name}'
            cycle_name = f'cycle_{var_name}'
            
            if trend_name in state_indices and cycle_name in state_indices:
                trend_idx = state_indices[trend_name]
                cycle_idx = state_indices[cycle_name]
                fitted_values = smoothed_states[:, trend_idx] + smoothed_states[:, cycle_idx]
                axes[i].plot(dates, fitted_values, 'r--', label='Fitted Values', alpha=0.8, linewidth=1.5)
            
            axes[i].set_title(f'{var_name} - Observed vs Fitted')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axes[i].xaxis.set_major_locator(mdates.YearLocator(10))
            
            if i == len(variable_names) - 1:
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].tick_params(axis='x', labelbottom=False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{plot_dir}/data_vs_fitted.png", dpi=300, bbox_inches='tight')
        plt.show()

# --- Example Usage with I(2) Data ---

def example_with_i2_data():
    """Example using I(2) simulated data with all fixes."""
    
    # Create sample I(2) data
    dates = pd.date_range('1960-01-01', periods=200, freq='Q')
    
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
            'num_samples': 300,
            'num_chains': 2
        },
        dls_params={
            'optimize_smoothness': True,
            'smoothness_range': (1e-6, 1e2),
            'alpha_shape': 2.5,
        },
        simulation_draws=50,
        save_config=True,
        config_filename='bvar_canova_dls_fixed.yml'
    )
    
    # Create comprehensive plots
    plot_results_with_canova_dls(
        results=results,
        data=data,
        save_plots=True,
        plot_dir='estimation_plots_fixed'
    )
    
    # Print summary
    if results and results.get('mcmc_results'):
        posterior = results['mcmc_results']['posterior_samples']
        print("\n" + "="*80)
        print("ESTIMATION SUMMARY")
        print("="*80)
        
        if 'parameter_mapping' in results:
            print("\nParameter Mapping (MCMC -> Smoother):")
            for param_name, param_value in results['parameter_mapping'].items():
                if isinstance(param_value, jnp.ndarray):
                    if param_value.ndim == 0:
                        print(f"  {param_name}: {float(param_value):.6f}")
                    else:
                        print(f"  {param_name}: shape={param_value.shape}, mean={jnp.mean(param_value):.6f}")
        
        if 'canova_dls_results' in results['config']:
            print("\nCanova DLS Prior Information:")
            for var_name, dls_info in results['config']['canova_dls_results'].items():
                diag = dls_info['diagnostics']
                print(f"  {var_name}:")
                print(f"    Optimal λ: {diag['optimal_lambda']:.2e}")
                print(f"    Trend variance: {diag['extracted_trend_var']:.6f}")
                print(f"    Cycle variance: {diag['extracted_cycle_var']:.6f}")
        
        smoothing_success = results.get('smoothing_results', {}).get('smoothed_states') is not None
        print(f"\nSmoothing Success: {smoothing_success}")
        
        return results
    else:
        print("Estimation failed!")
        return None

if __name__ == "__main__":
    results = example_with_i2_data()