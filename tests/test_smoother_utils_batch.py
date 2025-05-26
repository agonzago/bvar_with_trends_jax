import unittest
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Dict, Any, List, Tuple

# Configure JAX for float64, and ensure it's running on CPU for consistency in tests
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Important for consistent test behavior
_DEFAULT_DTYPE = jnp.float64

# Functions to test from core.smoother_utils
from core.smoother_utils import (
    extract_smoother_parameters_all_draws,
    construct_ss_matrices_all_draws,
    run_batch_simulation_smoother,
    compute_batch_quantiles,
    _get_state_indices_map # Helper, might be useful or needed indirectly
)
from core.var_ss_model import _MODEL_JITTER # Used in some functions

class TestSmootherUtilsBatch(unittest.TestCase):

    def setUp(self):
        self.key = random.PRNGKey(0)
        self.num_draws = 2
        self.T = 50  # Number of time steps
        self.k_endog = 2 # Number of endogenous variables / observables
        self.k_trends = 2
        self.k_stationary = 2
        self.var_order = 2 # p
        self.k_states = self.k_trends + self.k_stationary * self.var_order # Total states

        # Mock config_data for extract_smoother_parameters_all_draws
        self.mock_config_data_extract = {
            'measurement_param_names_tuple': ('meas_param1', 'meas_param2')
        }

        # Mock posterior_samples for extract_smoother_parameters_all_draws
        self.mock_posterior_samples = {
            'phi_list': jnp.ones((self.num_draws, self.var_order, self.k_stationary, self.k_stationary), dtype=_DEFAULT_DTYPE),
            'Sigma_cycles': jnp.array([jnp.eye(self.k_stationary, dtype=_DEFAULT_DTYPE)] * self.num_draws),
            'Sigma_trends_full': jnp.array([jnp.eye(self.k_trends, dtype=_DEFAULT_DTYPE)] * self.num_draws),
            'init_x_comp': jnp.ones((self.num_draws, self.k_states), dtype=_DEFAULT_DTYPE),
            'init_P_comp': jnp.array([jnp.eye(self.k_states, dtype=_DEFAULT_DTYPE)] * self.num_draws),
            'meas_param1': jnp.ones((self.num_draws,), dtype=_DEFAULT_DTYPE),
            'meas_param2': jnp.ones((self.num_draws,), dtype=_DEFAULT_DTYPE) * 2.0,
        }

        # Mock static_config_data for construct_ss_matrices_all_draws and integration test
        # This needs to be fairly complete for the functions to run
        self.trend_var_names = tuple(f'trend_var{i+1}' for i in range(self.k_trends))
        self.stationary_var_names = tuple(f'stat_var{i+1}' for i in range(self.k_stationary))
        
        # Mock parsed equations: obs_idx, (term_type, state_idx_in_C_block, param_idx, sign)
        # term_type: 0 for direct state, 1 for parameter * state
        # state_idx_in_C_block: index in [trends | current cycles] block
        # param_idx: index in measurement_param_names_tuple if term_type=1
        mock_parsed_eqs = []
        # Eq1: obs1 = trend_var1 + meas_param1 * stat_var1
        mock_parsed_eqs.append(
            (0, [ (0, 0, -1, 1.0), # trend_var1 (state_idx 0 in C-block)
                   (1, self.k_trends + 0, 0, 1.0) # meas_param1 * stat_var1 (state_idx k_trends+0 in C-block, param_idx 0)
                 ]) 
        )
        # Eq2: obs2 = trend_var2 + meas_param2 * stat_var2
        mock_parsed_eqs.append(
            (1, [ (0, 1, -1, 1.0), # trend_var2 (state_idx 1 in C-block)
                   (1, self.k_trends + 1, 1, 1.0) # meas_param2 * stat_var2 (state_idx k_trends+1 in C-block, param_idx 1)
                 ])
        )


        self.mock_static_config_data = {
            'k_endog': self.k_endog,
            'k_trends': self.k_trends,
            'k_stationary': self.k_stationary,
            'var_order': self.var_order,
            'k_states': self.k_states,
            'trend_var_names': self.trend_var_names,
            'stationary_var_names': self.stationary_var_names,
            'trend_names_with_shocks': self.trend_var_names, # Assume all trends have shocks
            'parsed_model_eqs_jax_detailed': mock_parsed_eqs,
            'measurement_param_names_tuple': ('meas_param1', 'meas_param2'),
            '_MODEL_JITTER': _MODEL_JITTER # if used directly
        }
        self.n_trend_shocks = len(self.mock_static_config_data['trend_names_with_shocks'])
        self.n_shocks_sim = self.n_trend_shocks + self.k_stationary


    def test_extract_smoother_parameters_all_draws(self):
        smoother_params = extract_smoother_parameters_all_draws(
            self.mock_posterior_samples, self.mock_config_data_extract
        )
        self.assertIn('phi_list', smoother_params)
        self.assertEqual(smoother_params['phi_list'].shape, (self.num_draws, self.var_order, self.k_stationary, self.k_stationary))
        self.assertIn('Sigma_cycles', smoother_params)
        self.assertEqual(smoother_params['Sigma_cycles'].shape, (self.num_draws, self.k_stationary, self.k_stationary))
        self.assertIn('Sigma_trends_full', smoother_params)
        self.assertEqual(smoother_params['Sigma_trends_full'].shape, (self.num_draws, self.k_trends, self.k_trends))
        self.assertIn('init_x_comp', smoother_params)
        self.assertEqual(smoother_params['init_x_comp'].shape, (self.num_draws, self.k_states))
        self.assertIn('init_P_comp', smoother_params)
        self.assertEqual(smoother_params['init_P_comp'].shape, (self.num_draws, self.k_states, self.k_states))
        
        self.assertIn('measurement_params', smoother_params)
        self.assertIn('meas_param1', smoother_params['measurement_params'])
        self.assertEqual(smoother_params['measurement_params']['meas_param1'].shape, (self.num_draws,))
        self.assertIn('meas_param2', smoother_params['measurement_params'])
        self.assertEqual(smoother_params['measurement_params']['meas_param2'].shape, (self.num_draws,))

    def test_construct_ss_matrices_all_draws(self):
        # Create mock smoother_params_all_draws based on the output of the previous function
        mock_smoother_params_all_draws = {
            'phi_list': self.mock_posterior_samples['phi_list'],
            'Sigma_cycles': self.mock_posterior_samples['Sigma_cycles'],
            'Sigma_trends_full': self.mock_posterior_samples['Sigma_trends_full'],
            'init_x_comp': self.mock_posterior_samples['init_x_comp'],
            'init_P_comp': self.mock_posterior_samples['init_P_comp'],
            'measurement_params': {
                'meas_param1': self.mock_posterior_samples['meas_param1'],
                'meas_param2': self.mock_posterior_samples['meas_param2']
            }
        }
        
        ss_matrices = construct_ss_matrices_all_draws(
            mock_smoother_params_all_draws, self.mock_static_config_data
        )
        
        self.assertIn('T_comp', ss_matrices)
        self.assertEqual(ss_matrices['T_comp'].shape, (self.num_draws, self.k_states, self.k_states))
        self.assertIn('R_aug', ss_matrices)
        self.assertEqual(ss_matrices['R_aug'].shape, (self.num_draws, self.k_states, self.n_shocks_sim))
        self.assertIn('C_comp', ss_matrices)
        self.assertEqual(ss_matrices['C_comp'].shape, (self.num_draws, self.k_endog, self.k_states))
        self.assertIn('H_comp', ss_matrices)
        self.assertEqual(ss_matrices['H_comp'].shape, (self.num_draws, self.k_endog, self.k_endog))
        self.assertIn('init_x_sim', ss_matrices)
        self.assertEqual(ss_matrices['init_x_sim'].shape, (self.num_draws, self.k_states))
        self.assertIn('init_P_sim', ss_matrices)
        self.assertEqual(ss_matrices['init_P_sim'].shape, (self.num_draws, self.k_states, self.k_states))

    def test_run_batch_simulation_smoother(self):
        # Mock ss_matrices_all_draws
        mock_ss_matrices_all_draws = {
            'T_comp': jnp.array([jnp.eye(self.k_states, dtype=_DEFAULT_DTYPE) * 0.95] * self.num_draws),
            'R_aug': jnp.array([jnp.eye(self.k_states, M=self.n_shocks_sim, dtype=_DEFAULT_DTYPE) * 0.1] * self.num_draws),
            'C_comp': jnp.array([jnp.ones((self.k_endog, self.k_states), dtype=_DEFAULT_DTYPE)] * self.num_draws),
            'H_comp': jnp.array([jnp.eye(self.k_endog, dtype=_DEFAULT_DTYPE) * 0.01] * self.num_draws),
            'init_x_sim': jnp.zeros((self.num_draws, self.k_states), dtype=_DEFAULT_DTYPE),
            'init_P_sim': jnp.array([jnp.eye(self.k_states, dtype=_DEFAULT_DTYPE)] * self.num_draws)
        }
        
        # Mock smoother_params_all_draws (only need Sigmas for Q reconstruction inside)
        mock_smoother_params_for_batch_run = {
            'Sigma_cycles': self.mock_posterior_samples['Sigma_cycles'],
            'Sigma_trends_full': self.mock_posterior_samples['Sigma_trends_full']
        }

        mock_original_ys_dense = jnp.ones((self.T, self.k_endog), dtype=_DEFAULT_DTYPE)
        mock_x_smooth_original_dense = jnp.zeros((self.T, self.k_states), dtype=_DEFAULT_DTYPE)
        
        all_sim_paths = run_batch_simulation_smoother(
            self.key, self.num_draws, mock_ss_matrices_all_draws,
            mock_smoother_params_for_batch_run, mock_original_ys_dense,
            mock_x_smooth_original_dense, self.mock_static_config_data
        )
        
        self.assertEqual(all_sim_paths.shape, (self.num_draws, self.T, self.k_states))
        self.assertTrue(jnp.all(jnp.isfinite(all_sim_paths)))

    def test_compute_batch_quantiles(self):
        mock_all_simulated_paths = random.normal(self.key, (self.num_draws, self.T, self.k_states), dtype=_DEFAULT_DTYPE)
        quantiles_to_compute = [0.05, 0.5, 0.95]
        
        quantile_estimates = compute_batch_quantiles(mock_all_simulated_paths, jnp.array(quantiles_to_compute))
        
        self.assertEqual(quantile_estimates.shape, (self.T, self.k_states, len(quantiles_to_compute)))
        self.assertTrue(jnp.all(jnp.isfinite(quantile_estimates)))

    def test_integration_smoother_pipeline(self):
        # Simplified End-to-End Test
        
        # 1. Mock MCMC Posterior Samples (as if from NumPyro)
        # These would be the direct output from mcmc.get_samples()
        # and include deterministic sites.
        mcmc_posterior_samples = self.mock_posterior_samples # Use the one from setUp

        # 2. Mock Static Config Data (as used in estimate_bvar_with_dls_priors.py)
        # This is 'config_data' in that script, which is then passed around.
        static_config_data_smoother = self.mock_static_config_data

        # 3. Mock other necessary inputs
        original_ys_for_smoother_dense = jnp.ones((self.T, self.k_endog), dtype=_DEFAULT_DTYPE)
        # x_smooth_rts_dense_original_data: (T, k_states) - result of RTS on original data
        x_smooth_rts_dense_original_data = jnp.zeros((self.T, self.k_states), dtype=_DEFAULT_DTYPE)
        
        smoother_quantiles = [0.1, 0.5, 0.9]
        num_mcmc_draws = self.num_draws # Matches the leading dimension of mock_posterior_samples

        key_main, key_sim_smoother = random.split(self.key)

        # --- Start of refactored logic from estimate_bvar_with_dls_priors.py ---
        
        # Step A: Extract smoother parameters for all draws
        smoother_params_all_draws = extract_smoother_parameters_all_draws(
            mcmc_posterior_samples, static_config_data_smoother
        )
        self.assertIsInstance(smoother_params_all_draws, dict)
        self.assertEqual(smoother_params_all_draws['phi_list'].shape[0], num_mcmc_draws)

        # Step B: Construct state-space matrices for all draws
        ss_matrices_all_draws = construct_ss_matrices_all_draws(
            smoother_params_all_draws, static_config_data_smoother
        )
        self.assertIsInstance(ss_matrices_all_draws, dict)
        self.assertEqual(ss_matrices_all_draws['T_comp'].shape[0], num_mcmc_draws)
        self.assertEqual(ss_matrices_all_draws['T_comp'].shape[1:], (self.k_states, self.k_states))


        # Step C: Run Batch Simulation Smoother
        all_simulated_paths = run_batch_simulation_smoother(
            key_sim_smoother,
            num_mcmc_draws,
            ss_matrices_all_draws,
            smoother_params_all_draws, # Pass the full dict for Sigmas
            original_ys_for_smoother_dense,
            x_smooth_rts_dense_original_data,
            static_config_data_smoother
        )
        self.assertEqual(all_simulated_paths.shape, (num_mcmc_draws, self.T, self.k_states))
        self.assertTrue(jnp.all(jnp.isfinite(all_simulated_paths)))

        # Step D: Compute Batch Quantiles
        simulated_state_quantiles_smoother = compute_batch_quantiles(
            all_simulated_paths,
            jnp.array(smoother_quantiles, dtype=_DEFAULT_DTYPE)
        )
        self.assertEqual(simulated_state_quantiles_smoother.shape, 
                         (self.T, self.k_states, len(smoother_quantiles)))
        self.assertTrue(jnp.all(jnp.isfinite(simulated_state_quantiles_smoother)))
        
        # --- End of refactored logic ---

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
